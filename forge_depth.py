import torch
import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return x * attention

class UncertaintyModule(nn.Module):
    def __init__(self, in_channels):
        super(UncertaintyModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        return torch.exp(self.conv(x))

class UniqueDepthEstimationModel(nn.Module):
    def __init__(self):
        super(UniqueDepthEstimationModel, self).__init__()
        
        # Use a pretrained ResNet50 backbone
        resnet = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last fully connected layer
        
        # Decoder layers
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        ])
        
        # Attention modules
        self.attention_modules = nn.ModuleList([
            AttentionModule(1024),
            AttentionModule(512),
            AttentionModule(256),
            AttentionModule(128),
            AttentionModule(64)
        ])
        
        # Final depth prediction layer
        self.depth_pred = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
        # Uncertainty prediction module
        self.uncertainty_module = UncertaintyModule(64)
        
        # Surface normal prediction layer
        self.normal_pred = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Encode using ResNet50 backbone
        features = self.backbone(x)
        
        # Decode
        for i, decode_layer in enumerate(self.decoder):
            features = decode_layer(features)
            features = self.attention_modules[i](features)
            
        # Depth prediction
        depth = self.depth_pred(features)
        
        # Uncertainty prediction
        uncertainty = self.uncertainty_module(features)
        
        # Surface normal prediction
        normals = self.normal_pred(features)
        normals = F.normalize(normals, dim=1)
        
        return depth, uncertainty, normals

# Load the trained model
model = UniqueDepthEstimationModel()
model.load_state_dict(torch.load("./unique_depth_model_epoch_13.pth", map_location=torch.device('cpu')), strict=False)
model.eval()  # Set model to evaluation mode

# Preprocess the input image
def preprocess_image(image_path):
    # Load the image and convert it to RGB
    image = Image.open(image_path).convert('RGB')
    
    # Define the image transformations (resize, convert to tensor, normalize)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image to 256x256
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    
    image = transform(image)
    
    image = image.unsqueeze(0)  # Add batch dimension
    return image

image_path = "./test/test_image9.jpg"
input_image = preprocess_image(image_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_image = input_image.to(device)

with torch.no_grad(): 
    depth_pred, uncertainty_pred, normals_pred = model(input_image)

depth_pred = depth_pred.squeeze(0).squeeze(0).cpu().numpy()  # Convert to numpy array (H, W)

# Load the original image for side-by-side comparison
original_image = Image.open(image_path).convert('RGB')
original_image = original_image.resize((256, 256))  # Resize to match depth map dimensions
original_image_np = np.array(original_image)

# Create a subplot to display the original image and depth map
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Show original image
axes[0].imshow(original_image_np)
axes[0].set_title('Original Image')
axes[0].axis('off')  # Turn off axis

# Show depth map
axes[1].imshow(depth_pred, cmap='plasma')  # Use a colormap for better visualization
axes[1].set_title('Depth Map')
axes[1].axis('off')  # Turn off axis

plt.tight_layout()
plt.show()
