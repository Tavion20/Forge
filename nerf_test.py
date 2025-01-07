import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies=10):
        super().__init__()
        self.num_frequencies = num_frequencies
    
    def forward(self, x):
        outputs = []
        for i in range(self.num_frequencies):
            outputs.append(torch.sin(2**i * x))
            outputs.append(torch.cos(2**i * x))
        return torch.cat(outputs, dim=-1)

class NeRFModel(nn.Module):
    def __init__(self, 
                 pos_enc_frequencies=10,
                 depth_enc_frequencies=4,
                 hidden_dim=256):
        super().__init__()
        
        self.xyz_encoding = PositionalEncoding(pos_enc_frequencies)
        self.depth_encoding = PositionalEncoding(depth_enc_frequencies)
        
        xyz_enc_dim = pos_enc_frequencies * 6  # 2 for sin/cos
        depth_enc_dim = depth_enc_frequencies * 2
        
        # Keep the same structure as original but with improvements
        self.mlp = nn.Sequential(
            nn.Linear(xyz_enc_dim + depth_enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.density_head = nn.Linear(hidden_dim, 1)
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )
        
    def forward(self, x, d):
        x_encoded = self.xyz_encoding(x)
        d_encoded = self.depth_encoding(d)
        
        features = torch.cat([x_encoded, d_encoded], dim=-1)
        features = self.mlp(features)
        
        density = F.relu(self.density_head(features))
        rgb = self.rgb_head(features)
        
        return density, rgb

def train_step(model, optimizer, rgbd_images, poses, rays, num_samples=64):
    optimizer.zero_grad()
    
    # Improved sampling
    near, far = 2.0, 6.0
    t_vals = torch.linspace(0., 1., num_samples)
    z_vals = near * (1. - t_vals) + far * t_vals
    z_vals = z_vals + torch.rand_like(z_vals) * (far - near) / num_samples  # Stratified sampling
    
    points = rays[..., None, :] + z_vals[..., None] * poses[..., None, :]
    
    # Get model predictions
    points_flat = points.reshape(-1, 3)
    depths_flat = torch.ones_like(points_flat[:, :1])
    
    densities, colors = model(points_flat, depths_flat)
    
    # Improved volume rendering
    densities = densities.reshape(-1, num_samples)
    colors = colors.reshape(-1, num_samples, 3)
    
    # Calculate transmittance
    delta = z_vals[..., 1:] - z_vals[..., :-1]
    delta = torch.cat([delta, torch.tensor([1e10]).expand(delta[..., :1].shape)], -1)
    alpha = 1.0 - torch.exp(-F.relu(densities) * delta)
    
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    
    rgb_rendered = torch.sum(weights[..., None] * colors, dim=1)
    
    # Loss computation with regularization
    mse_loss = F.mse_loss(rgb_rendered, rgbd_images[..., :3])
    regularization_loss = 0.1 * torch.mean(densities)  # Density regularization
    
    loss = mse_loss + regularization_loss
    loss.backward()
    
    optimizer.step()
    return loss.item()