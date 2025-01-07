import torch
import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

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
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        ])
        
        self.attention_modules = nn.ModuleList([
            AttentionModule(1024),
            AttentionModule(512),
            AttentionModule(256),
            AttentionModule(128),
            AttentionModule(64)
        ])
        
        self.depth_pred = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
        self.uncertainty_module = UncertaintyModule(64)
        
        self.normal_pred = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        features = self.backbone(x)
        
        for i, decode_layer in enumerate(self.decoder):
            features = decode_layer(features)
            features = self.attention_modules[i](features)
            
        depth = self.depth_pred(features)
        
        uncertainty = self.uncertainty_module(features)
        
        normals = self.normal_pred(features)
        normals = F.normalize(normals, dim=1)
        
        return depth, uncertainty, normals

def depth_uncertainty_normal_loss(depth_pred, depth_gt, uncertainty, normal_pred, normal_gt, alpha=0.5, beta=0.1):
    depth_loss = torch.abs(depth_pred - depth_gt) * torch.exp(-uncertainty) + uncertainty
    depth_loss = depth_loss.mean()
    
    normal_loss = 1 - F.cosine_similarity(normal_pred, normal_gt, dim=1).mean()
    
    total_loss = depth_loss + alpha * normal_loss + beta * uncertainty.mean()
    
    return total_loss


import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import models 
import torch.nn as nn

model = UniqueDepthEstimationModel()
model.load_state_dict(torch.load("./unique_depth_model_epoch_13.pth", map_location=torch.device('cpu')), strict=False)
model.eval() 

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    image = transform(image)
    
    image = image.unsqueeze(0)
    
    return image



image_path = "./test/test_image7.jpg"
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_image = preprocess_image(image_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_image = input_image.to(device)

with torch.no_grad(): 
    depth_pred, uncertainty_pred, normals_pred = model(input_image)


depth_pred = depth_pred.squeeze(0).squeeze(0).cpu().numpy()

import numpy as np
import open3d as o3d
import cv2

# Intrinsic camera parameters (you may need to replace these with the real values for your camera)
f_x = 525.0  # Focal length along the x-axis (in pixels)
f_y = 525.0  # Focal length along the y-axis (in pixels)
c_x = 319.5  # Principal point (x-coordinate of image center)
c_y = 239.5  # Principal point (y-coordinate of image center)

def resize_rgb_to_match_depth(rgb_image, depth_image):
    h, w = depth_image.shape
    return cv2.resize(rgb_image, (w, h))

def depth_to_point_cloud(depth, rgb=None):
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    
    Z = depth
    X = (i - c_x) * Z / f_x
    Y = (j - c_y) * Z / f_y
    
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    if rgb is not None:
        rgb_flat = rgb.reshape(-1, 3)  
        point_cloud.colors = o3d.utility.Vector3dVector(rgb_flat / 255.0) 
    
    return point_cloud

depth_pred_np = depth_pred.squeeze() 

rgb_image_resized = resize_rgb_to_match_depth(img, depth_pred_np)

point_cloud = depth_to_point_cloud(depth_pred_np, rgb=rgb_image_resized)

o3d.visualization.draw_geometries([point_cloud])

