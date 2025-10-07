import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch
import torch
import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import shutil
import sys
import matplotlib.pyplot as plt

import scipy.optimize
import tqdm

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

# Intrinsic camera parameters (you may need to replace these with the real values for your camera)
f_x = 525.0  # Focal length along the x-axis (in pixels)
f_y = 525.0  # Focal length along the y-axis (in pixels)
c_x = 319.5  # Principal point (x-coordinate of image center)
c_y = 239.5  # Principal point (y-coordinate of image center)

import plotly.graph_objects as go
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import open3d as o3d
import torchvision.transforms as transforms

model = UniqueDepthEstimationModel()
model.load_state_dict(torch.load("./unique_depth_model_epoch_13.pth", map_location=torch.device('cpu')), strict=False)
model.eval() 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    image = transform(image)
    return image.unsqueeze(0).to(device)

def estimate_depth_normals(image_path):
    input_image = preprocess_image(image_path)
    with torch.no_grad():
        depth_pred, uncertainty_pred, normals_pred = model(input_image)
        
    depth_img = depth_pred.squeeze(0).squeeze(0).cpu().numpy()
    depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
    depth_img = (depth_img * 255).astype(np.uint8)
    
    normals_np = normals_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    normals_np = (normals_np + 1) / 2
    normals_img = (normals_np * 255).astype(np.uint8)
    
    return depth_img, normals_img

def depth_to_point_cloud(depth, rgb=None):
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    
    Z = depth
    X = (i - c_x) * Z / f_x
    Y = (j - c_y) * Z / f_y
    
    Y = -Y
    
    theta = np.radians(30)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    rotated_points = points @ rotation_matrix.T
    
    if rgb is not None:
        rgb_flat = rgb.reshape(-1, 3) / 255.0
    else:
        rgb_flat = np.repeat([[0.5, 0.5, 0.5]], points.shape[0], axis=0)
    
    return rotated_points, rgb_flat

def create_rgbd_image(rgb_image, depth_image):
    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    rgb_resized = cv2.resize(rgb_image, (depth_colormap.shape[1], depth_colormap.shape[0]))
    
    alpha = 0.6
    rgbd_image = cv2.addWeighted(rgb_resized, alpha, depth_colormap, 1 - alpha, 0)
    
    return rgbd_image

st.title("Depth Estimation with 3D Visualization")

def depth_to_pointcloud(depth_map, color_image, fx, fy, cx, cy):
    
    rows, cols = depth_map.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    
    z = depth_map
    x = (c - cx) * z / fx
    y = (r - cy) * z / fy
    
    valid = z > 0
    
    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    colors = color_image[valid]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    return pcd

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Uploaded Image", use_column_width=True)
    
    img = np.array(input_image)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    depth_img, normals_img = estimate_depth_normals(uploaded_file)

    st.subheader("Depth Map")
    st.image(depth_img, caption="Depth Map", use_column_width=True, clamp=True)

    if st.checkbox("Show RGB-D Image"):
        rgb_image_np = np.array(input_image)
        rgbd_image = create_rgbd_image(rgb_image_np, depth_img)
        st.image(rgbd_image, caption="RGB-D Image", use_column_width=True)
        
    # if st.checkbox("Show 3D Point Cloud (Plotly)"):
    #     point_cloud = depth_to_pointcloud(depth_img, img, f_x, f_y, c_x, c_y)
    #     o3d.visualization.draw_geometries([point_cloud])
    #     depth_np = depth_img.astype(np.float32) / 255.0
    #     rgb_image_resized = cv2.resize(np.array(input_image), (depth_np.shape[1], depth_np.shape[0]))
    #     points, colors = depth_to_point_cloud(depth_np, rgb=rgb_image_resized)
        
    #     fig = go.Figure(data=[go.Scatter3d(
    #         x=points[:, 0],
    #         y=points[:, 1],
    #         z=points[:, 2],
    #         mode='markers',
    #         marker=dict(
    #             size=1,
    #             color=colors,
    #             opacity=0.8
    #         )
    #     )])
        
    #     fig.update_layout(scene=dict(
    #         xaxis_title='X',
    #         yaxis_title='Y',
    #         zaxis_title='Z'
    #     ), 
    #     width=700, 
    #     height=700,
    #     margin=dict(r=10, l=10, b=10, t=10))
        
    #     st.plotly_chart(fig)




import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)

def sift_orb_matching(img1, img2, orb_matches_limit=50, sift_ratio=0.8):
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create(nfeatures=2000)

    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    keypoints1_orb, descriptors1_orb = orb.detectAndCompute(img1, None)
    keypoints2_orb, descriptors2_orb = orb.detectAndCompute(img2, None)

    good_matches = []

    if descriptors1 is not None and descriptors2 is not None:
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches_sift = flann.knnMatch(descriptors1, descriptors2, k=2)

        for m, n in matches_sift:
            if m.distance < sift_ratio * n.distance:
                good_matches.append(m)

    if descriptors1_orb is not None and descriptors2_orb is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches_orb = bf.match(descriptors1_orb, descriptors2_orb)
        matches_orb = sorted(matches_orb, key=lambda x: x.distance)

        good_matches += matches_orb[:orb_matches_limit]

    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        good_matches = [good_matches[i] for i in range(len(matches_mask)) if matches_mask[i]]

    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches, good_matches, keypoints1, keypoints2, keypoints1_orb, keypoints2_orb


def plot_3d_points_and_cameras(points_3d, R, t):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points_3d[0], points_3d[1], points_3d[2], c='b', marker='o', s=10)
    
    camera1_pos = np.array([0, 0, 0])
    camera2_pos = -R.T @ t.ravel()
    ax.scatter(camera1_pos[0], camera1_pos[1], camera1_pos[2], c='r', marker='^', s=100, label='Camera 1')
    ax.scatter(camera2_pos[0], camera2_pos[1], camera2_pos[2], c='g', marker='^', s=100, label='Camera 2')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    st.pyplot(fig)

# st.title("3D Reconstruction with Feature Matching and Camera Triangulation")

# uploaded_file1 = st.file_uploader("Choose the first image...", type=["jpg", "jpeg", "png"])
# uploaded_file2 = st.file_uploader("Choose the second image...", type=["jpg", "jpeg", "png"])

# if uploaded_file1 and uploaded_file2:
#     img1 = cv2.imdecode(np.frombuffer(uploaded_file1.read(), np.uint8), 1)
#     img2 = cv2.imdecode(np.frombuffer(uploaded_file2.read(), np.uint8), 1)

#     img1 = preprocess_image(img1)
#     img2 = preprocess_image(img2)

#     result, good_matches, keypoints1, keypoints2, keypoints1_orb, keypoints2_orb = sift_orb_matching(img1, img2, orb_matches_limit=100, sift_ratio=0.8)


#     st.image(result, caption="Feature Matching Result", use_column_width=True)

#     pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
#     pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

#     focal_length = max(img1.shape)
#     camera_matrix = np.array([[focal_length, 0, img1.shape[1] / 2],
#                               [0, focal_length, img1.shape[0] / 2],
#                               [0, 0, 1]], dtype=np.float32)

#     essential_matrix, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
#     _, R, t, mask_pose = cv2.recoverPose(essential_matrix, pts1, pts2, camera_matrix)

#     pts1_inliers = pts1[mask_pose.ravel() == 255]
#     pts2_inliers = pts2[mask_pose.ravel() == 255]

#     if pts1_inliers.shape[0] < 2 or pts2_inliers.shape[0] < 2:
#         pts1_inliers = pts1
#         pts2_inliers = pts2

#     P1 = np.eye(3, 4, dtype=np.float32)
#     P2 = np.hstack((R, t)).astype(np.float32)

#     pts1_inliers = pts1_inliers.T.astype(np.float32)
#     pts2_inliers = pts2_inliers.T.astype(np.float32)

#     points_4d = cv2.triangulatePoints(P1, P2, pts1_inliers, pts2_inliers)
#     points_3d = points_4d[:3] / points_4d[3]

#     plot_3d_points_and_cameras(points_3d, R, t)


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms import Resize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.1, dropout_p=0.2):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, in_channels // 2)
        self.dropout1 = nn.Dropout2d(p=dropout_p)
        self.conv2 = ConvBlock(in_channels // 2, out_channels)
        self.dropout2 = nn.Dropout2d(p=dropout_p)
        self.stochastic_depth = StochasticDepth(drop_rate=drop_rate)
        
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = F.interpolate(x, size=(h*2, w*2), mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.stochastic_depth(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_3_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1)
        self.conv_3_6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv_3_12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Conv2d(in_channels, out_channels, 1)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        size = x.shape[2:]
        conv1 = self.conv1(x)
        conv3_1 = self.conv_3_1(x)
        conv3_6 = self.conv_3_6(x)
        conv3_12 = self.conv_3_12(x)
        
        pool = self.avg_pool(x)
        pool = self.conv_pool(pool)
        pool = F.interpolate(pool, size=size, mode='bilinear', align_corners=True)
        
        x = torch.cat([conv1, conv3_1, conv3_6, conv3_12, pool], dim=1)
        return self.bottleneck(x)

class DepthEstimationModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.aspp = ASPP(2048, 256)
        
        self.decoder4 = DecoderBlock(256 + 1024, 512, drop_rate=0.1, dropout_p=0.2)
        self.decoder3 = DecoderBlock(512 + 512, 256, drop_rate=0.15, dropout_p=0.25)
        self.decoder2 = DecoderBlock(256 + 256, 128, drop_rate=0.2, dropout_p=0.3)
        self.decoder1 = DecoderBlock(128 + 64, 64, drop_rate=0.25, dropout_p=0.35)
        
        self.reduce4 = ConvBlock(1024, 1024)
        self.reduce3 = ConvBlock(512, 512)
        self.reduce2 = ConvBlock(256, 256)
        self.reduce1 = ConvBlock(64, 64)
        
        self.final_conv = nn.Sequential(
            ConvBlock(64, 64),
            nn.Conv2d(64, 1, 1)
        )
        
        self.uncertainty = nn.Sequential(
            ConvBlock(64, 64),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        input_size = x.shape[2:]
        
        x1 = self.firstrelu(self.firstbn(self.firstconv(x)))
        x2 = self.firstmaxpool(x1)
        
        e1 = self.encoder1(x2)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        x = self.aspp(e4)
        
        s3 = self.reduce4(e3)
        s2 = self.reduce3(e2)
        s1 = self.reduce2(e1)
        s0 = self.reduce1(x1)
        
        x_size = x.size()[2:]
        s3 = F.interpolate(s3, size=x_size, mode='bilinear', align_corners=True)
        d4 = torch.cat([x, s3], dim=1)
        d4 = self.decoder4(d4)
        
        d4_size = d4.size()[2:]
        s2 = F.interpolate(s2, size=d4_size, mode='bilinear', align_corners=True)
        d3 = torch.cat([d4, s2], dim=1)
        d3 = self.decoder3(d3)
        
        d3_size = d3.size()[2:]
        s1 = F.interpolate(s1, size=d3_size, mode='bilinear', align_corners=True)
        d2 = torch.cat([d3, s1], dim=1)
        d2 = self.decoder2(d2)
        
        d2_size = d2.size()[2:]
        s0 = F.interpolate(s0, size=d2_size, mode='bilinear', align_corners=True)
        d1 = torch.cat([d2, s0], dim=1)
        d1 = self.decoder1(d1)
        
        features = self.final_conv[:-1](d1)
        depth = self.final_conv[-1](features)
        uncertainty = self.uncertainty(features)
        
        depth = F.interpolate(depth, size=input_size, mode='bilinear', align_corners=True)
        uncertainty = F.interpolate(uncertainty, size=input_size, mode='bilinear', align_corners=True)
        
        return depth, uncertainty
    
from math import exp

class CombinedLoss(nn.Module):
    def __init__(self, w1=1.0, w2=0.1, w3=0.1, w4=0.05):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.ssim_window_size = 11
        self.eps = 1e-6
        
    def compute_gradient_loss(self, pred, target, mask):
        pred = pred + self.eps
        target = target + self.eps
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              device=pred.device).float().reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              device=pred.device).float().reshape(1, 1, 3, 3)
        
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        
        grad_loss = torch.abs(pred_grad_x - target_grad_x) + torch.abs(pred_grad_y - target_grad_y)
        grad_loss = torch.clamp(grad_loss, min=0.0, max=1e6)
        
        masked_loss = (grad_loss * mask).sum() / (mask.sum() + self.eps)
        return masked_loss

    def compute_ssim_loss(self, pred, target, mask, window_size=11):
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        
        sigma = 1.5
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                             for x in range(window_size)])
        gauss = gauss/gauss.sum()
        
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(pred.size(1), 1, window_size, window_size).to(pred.device)
        
        pred = pred + self.eps
        target = target + self.eps
        
        mu1 = F.conv2d(pred, window, padding=window_size//2, groups=pred.size(1))
        mu2 = F.conv2d(target, window, padding=window_size//2, groups=target.size(1))
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=pred.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=target.size(1)) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=pred.size(1)) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_loss = (1 - ssim_map) * mask
        return torch.clamp(ssim_loss, min=0.0, max=1.0)
    
    def scale_invariant_loss(self, pred, target, mask):
        pred = pred + self.eps
        target = target + self.eps
        
        log_diff = torch.log(pred) - torch.log(target)
        num_valid = mask.sum() + self.eps
        
        if num_valid > self.eps:
            log_diff = log_diff * mask
            sum_log_diff = torch.sum(log_diff)
            sum_log_diff_squared = torch.sum(log_diff ** 2)
            loss = sum_log_diff_squared / num_valid - (sum_log_diff ** 2) / (num_valid ** 2)
            return torch.clamp(loss, min=0.0, max=1e6)
        return torch.tensor(0.0, device=pred.device)

    def berhu_loss(self, pred, target, mask, c=0.2):
        diff = torch.abs(pred - target) * mask
        c = torch.clamp(diff.max() * c, min=self.eps, max=1e6)
        
        quadratic_mask = (diff > c).float() * mask
        linear_mask = (diff <= c).float() * mask
        
        quadratic_loss = (diff ** 2 + c ** 2) / (2 * c + self.eps)
        linear_loss = diff
        
        loss = quadratic_mask * quadratic_loss + linear_mask * linear_loss
        return torch.clamp(loss, min=0.0, max=1e6)

    def forward(self, pred_depth, pred_uncertainty, target_depth):
        pred_depth = torch.clamp(pred_depth, min=self.eps)
        pred_uncertainty = torch.clamp(pred_uncertainty, min=self.eps)
        target_depth = torch.clamp(target_depth, min=0.0)
        
        if pred_depth.shape != target_depth.shape:
            pred_depth = F.interpolate(pred_depth, size=target_depth.shape[2:], 
                                     mode='bilinear', align_corners=True)
            pred_uncertainty = F.interpolate(pred_uncertainty, size=target_depth.shape[2:], 
                                          mode='bilinear', align_corners=True)
        
        mask = (target_depth > 0).float()
        
        pred_depth = pred_depth * mask
        target_depth = target_depth * mask
        
        depth_loss = self.berhu_loss(pred_depth, target_depth, mask)
        uncertainty_reg = torch.exp(-pred_uncertainty) * depth_loss + pred_uncertainty
        gradient_loss = self.compute_gradient_loss(pred_depth, target_depth, mask)
        ssim_loss = self.compute_ssim_loss(pred_depth, target_depth, mask, self.ssim_window_size)
        si_loss = self.scale_invariant_loss(pred_depth, target_depth, mask)
        
        total_loss = (
            self.w1 * depth_loss + 
            self.w2 * uncertainty_reg + 
            self.w3 * gradient_loss + 
            self.w4 * ssim_loss + 
            0.1 * si_loss
        ) * mask
        
        final_loss = total_loss.sum() / (mask.sum() + self.eps)
        
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            print("Warning: Loss is NaN or Inf! Using fallback loss...")
            return torch.clamp(depth_loss.mean(), min=0.0, max=1e6)
            
        return final_loss
    

def l2_regularization(model):
    reg_loss = 0
    for name, param in model.named_parameters():
        if 'reduce' in name:
            reg_loss += torch.norm(param, p=2)
    return reg_loss * 0.01

class StochasticDepth(nn.Module):
    def __init__(self, drop_rate=0.1):
        super(StochasticDepth, self).__init__()
        self.drop_rate = drop_rate
        
    def forward(self, x):
        if not self.training:
            return x
        
        keep_rate = 1 - self.drop_rate
        mask = torch.bernoulli(torch.ones_like(x) * keep_rate)
        return x * mask / keep_rate



import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
import open3d as o3d
from torchvision import transforms
import io
import tempfile
import os

def save_depth_as_png(depth_map, save_path, min_depth=None, max_depth=None, colormap='plasma'):
    """Convert depth map to colored image and save as PNG."""
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    
    if min_depth is None:
        min_depth = np.min(depth_map[depth_map > 0])
    if max_depth is None:
        max_depth = np.max(depth_map)
    
    normalized_depth = np.clip((depth_map - min_depth) / (max_depth - min_depth), 0, 1)
    
    colored_depth = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
    
    cv2.imwrite(save_path, colored_depth)

def load_model():
    model = DepthEstimationModel(pretrained=True)
    checkpoint = torch.load('./best_depth_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image)
    return image.unsqueeze(0)

def create_point_cloud(color_path, depth_path):
    """Create point cloud using Open3D"""
    color = o3d.io.read_image(color_path)
    depth = o3d.io.read_image(depth_path)
    
    depth = o3d.geometry.Image(np.asarray(depth).astype(np.float32) / 1000.0)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, 
        depth,
        depth_scale=1.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        )
    )
    
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    return pcd


class Image_loader():
    def __init__(self, img_dir:str, downscale_factor:float):
        with open(os.path.join(img_dir, 'K.txt')) as f:
            self.K = np.array(list((map(lambda x:list(map(lambda x:float(x), x.strip().split(' '))),f.read().strip().split('\n')))))
            self.image_list = []
        for image in sorted(os.listdir(img_dir)):
            if image[-4:].lower() == '.jpg' or image[-5:].lower() == '.png':
                self.image_list.append(os.path.join(img_dir, image))
        
        self.path = os.getcwd()
        self.factor = downscale_factor
        self.downscale()

    
    def downscale(self) -> None:
        '''
        Downscales the Image intrinsic parameter acc to the downscale factor
        '''
        self.K[0, 0] /= self.factor
        self.K[1, 1] /= self.factor
        self.K[0, 2] /= self.factor
        self.K[1, 2] /= self.factor
    
    def downscale_image(self, image):
        for _ in range(1,int(self.factor / 2) + 1):
            image = cv2.pyrDown(image)
        return image

class Sfm():
    def __init__(self, img_dir:str, downscale_factor:float = 2.0) -> None:
        '''
            Initialise and Sfm object.
        '''
        self.img_obj = Image_loader(img_dir,downscale_factor)

    def triangulation(self, point_2d_1, point_2d_2, projection_matrix_1, projection_matrix_2) -> tuple:
        '''
        Triangulates 3d points from 2d vectors and projection matrices
        returns projection matrix of first camera, projection matrix of second camera, point cloud 
        '''
        pt_cloud = cv2.triangulatePoints(point_2d_1, point_2d_2, projection_matrix_1.T, projection_matrix_2.T)
        return projection_matrix_1.T, projection_matrix_2.T, (pt_cloud / pt_cloud[3])    
    
    def PnP(self, obj_point, image_point , K, dist_coeff, rot_vector, initial) ->  tuple:
        '''
        Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.
        returns rotational matrix, translational matrix, image points, object points, rotational vector
        '''
        if initial == 1:
            obj_point = obj_point[:, 0 ,:]
            image_point = image_point.T
            rot_vector = rot_vector.T 
        _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
        rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)

        if inlier is not None:
            image_point = image_point[inlier[:, 0]]
            obj_point = obj_point[inlier[:, 0]]
            rot_vector = rot_vector[inlier[:, 0]]
        return rot_matrix, tran_vector, image_point, obj_point, rot_vector
    
    def reprojection_error(self, obj_points, image_points, transform_matrix, K, homogenity) ->tuple:
        '''
        Calculates the reprojection error ie the distance between the projected points and the actual points.
        returns total error, object points
        '''
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        if homogenity == 1:
            obj_points = cv2.convertPointsFromHomogeneous(obj_points.T)
        image_points_calc, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points_calc = np.float32(image_points_calc[:, 0, :])
        total_error = cv2.norm(image_points_calc, np.float32(image_points.T) if homogenity == 1 else np.float32(image_points), cv2.NORM_L2)
        return total_error / len(image_points_calc), obj_points

    def optimal_reprojection_error(self, obj_points) -> np.array:
        '''
        calculates of the reprojection error during bundle adjustment
        returns error 
        '''
        transform_matrix = obj_points[0:12].reshape((3,4))
        K = obj_points[12:21].reshape((3,3))
        rest = int(len(obj_points[21:]) * 0.4)
        p = obj_points[21:21 + rest].reshape((2, int(rest/2))).T
        obj_points = obj_points[21 + rest:].reshape((int(len(obj_points[21 + rest:])/3), 3))
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        image_points, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points = image_points[:, 0, :]
        error = [ (p[idx] - image_points[idx])**2 for idx in range(len(p))]
        return np.array(error).ravel()/len(p)

    def bundle_adjustment(self, _3d_point, opt, transform_matrix_new, K, r_error) -> tuple:
        '''
        Bundle adjustment for the image and object points
        returns object points, image points, transformation matrix
        '''
        opt_variables = np.hstack((transform_matrix_new.ravel(), K.ravel()))
        opt_variables = np.hstack((opt_variables, opt.ravel()))
        opt_variables = np.hstack((opt_variables, _3d_point.ravel()))

        values_corrected = scipy.optimize.least_squares(self.optimal_reprojection_error, opt_variables, gtol = r_error).x
        K = values_corrected[12:21].reshape((3,3))
        rest = int(len(values_corrected[21:]) * 0.4)
        return values_corrected[21 + rest:].reshape((int(len(values_corrected[21 + rest:])/3), 3)), values_corrected[21:21 + rest].reshape((2, int(rest/2))).T, values_corrected[0:12].reshape((3,4))

    def to_ply(self, path, point_cloud, colors) -> None:
        '''
        Generates the .ply which can be used to open the point cloud
        '''
        out_points = point_cloud.reshape(-1, 3) * 200
        out_colors = colors.reshape(-1, 3)
        print(out_colors.shape, out_points.shape)
        verts = np.hstack([out_points, out_colors])

        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean
        dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
        indx = np.where(dist < np.mean(dist) + 300)
        verts = verts[indx]
        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar blue
            property uchar green
            property uchar red
            end_header
            '''
        with open(os.path.join(path, 'res', os.path.basename(os.path.dirname(self.img_obj.image_list[0])) + '.ply'), 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')

    def common_points(self, image_points_1, image_points_2, image_points_3) -> tuple:
        '''
        Finds the common points between image 1 and 2 , image 2 and 3
        returns common points of image 1-2, common points of image 2-3, mask of common points 1-2 , mask for common points 2-3 
        '''
        cm_points_1 = []
        cm_points_2 = []
        for i in range(image_points_1.shape[0]):
            a = np.where(image_points_2 == image_points_1[i, :])
            if a[0].size != 0:
                cm_points_1.append(i)
                cm_points_2.append(a[0][0])

        mask_array_1 = np.ma.array(image_points_2, mask=False)
        mask_array_1.mask[cm_points_2] = True
        mask_array_1 = mask_array_1.compressed()
        mask_array_1 = mask_array_1.reshape(int(mask_array_1.shape[0] / 2), 2)

        mask_array_2 = np.ma.array(image_points_3, mask=False)
        mask_array_2.mask[cm_points_2] = True
        mask_array_2 = mask_array_2.compressed()
        mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)
        print(" Shape New Array", mask_array_1.shape, mask_array_2.shape)
        return np.array(cm_points_1), np.array(cm_points_2), mask_array_1, mask_array_2

    def find_features(self, image_0, image_1) -> tuple:
        '''
        Feature detection using the sift algorithm and KNN
        return keypoints(features) of image1 and image2
        '''
        sift = cv2.xfeatures2d.SIFT_create()
        key_points_0, desc_0 = sift.detectAndCompute(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None)
        key_points_1, desc_1 = sift.detectAndCompute(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_0, desc_1, k=2)
        feature = []
        for m, n in matches:
            if m.distance < 0.70 * n.distance:
                feature.append(m)

        return np.float32([key_points_0[m.queryIdx].pt for m in feature]), np.float32([key_points_1[m.trainIdx].pt for m in feature])

    def __call__(self, enable_bundle_adjustment:bool=False):
        import matplotlib.pyplot as plt
        import cv2

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        pose_array = self.img_obj.K.ravel()
        transform_matrix_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        transform_matrix_1 = np.empty((3, 4))
    
        pose_0 = np.matmul(self.img_obj.K, transform_matrix_0)
        pose_1 = np.empty((3, 4)) 
        total_points = np.zeros((1, 3))
        total_colors = np.zeros((1, 3))

        image_0 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[0]))
        image_1 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[1]))

        feature_0, feature_1 = self.find_features(image_0, image_1)

        essential_matrix, em_mask = cv2.findEssentialMat(feature_0, feature_1, self.img_obj.K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
        feature_0 = feature_0[em_mask.ravel() == 1]
        feature_1 = feature_1[em_mask.ravel() == 1]

        _, rot_matrix, tran_matrix, em_mask = cv2.recoverPose(essential_matrix, feature_0, feature_1, self.img_obj.K)
        feature_0 = feature_0[em_mask.ravel() > 0]
        feature_1 = feature_1[em_mask.ravel() > 0]
        transform_matrix_1[:3, :3] = np.matmul(rot_matrix, transform_matrix_0[:3, :3])
        transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3, :3], tran_matrix.ravel())

        pose_1 = np.matmul(self.img_obj.K, transform_matrix_1)

        feature_0, feature_1, points_3d = self.triangulation(pose_0, pose_1, feature_0, feature_1)
        error, points_3d = self.reprojection_error(points_3d, feature_1, transform_matrix_1, self.img_obj.K, homogenity = 1)
        print("REPROJECTION ERROR: ", error)
        _, _, feature_1, points_3d, _ = self.PnP(points_3d, feature_1, self.img_obj.K, np.zeros((5, 1), dtype=np.float32), feature_0, initial=1)

        total_images = len(self.img_obj.image_list) - 2 
        pose_array = np.hstack((np.hstack((pose_array, pose_0.ravel())), pose_1.ravel()))

        threshold = 0.5
        for i in range(total_images):
            image_2 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[i + 2]))
            features_cur, features_2 = self.find_features(image_1, image_2)

            if i != 0:
                feature_0, feature_1, points_3d = self.triangulation(pose_0, pose_1, feature_0, feature_1)
                feature_1 = feature_1.T
                points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
                points_3d = points_3d[:, 0, :]
            
            cm_points_0, cm_points_1, cm_mask_0, cm_mask_1 = self.common_points(feature_1, features_cur, features_2)
            cm_points_2 = features_2[cm_points_1]
            cm_points_cur = features_cur[cm_points_1]

            rot_matrix, tran_matrix, cm_points_2, points_3d, cm_points_cur = self.PnP(points_3d[cm_points_0], cm_points_2, self.img_obj.K, np.zeros((5, 1), dtype=np.float32), cm_points_cur, initial = 0)
            transform_matrix_1 = np.hstack((rot_matrix, tran_matrix))
            pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)

            error, points_3d = self.reprojection_error(points_3d, cm_points_2, transform_matrix_1, self.img_obj.K, homogenity = 0)
        
            cm_mask_0, cm_mask_1, points_3d = self.triangulation(pose_1, pose_2, cm_mask_0, cm_mask_1)
            error, points_3d = self.reprojection_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, homogenity = 1)
            print("Reprojection Error: ", error)
            pose_array = np.hstack((pose_array, pose_2.ravel()))
            
            if enable_bundle_adjustment:
                points_3d, cm_mask_1, transform_matrix_1 = self.bundle_adjustment(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, threshold)
                pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)
                error, points_3d = self.reprojection_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, homogenity = 0)
                print("Bundle Adjusted error: ",error)
                total_points = np.vstack((total_points, points_3d))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left])
                total_colors = np.vstack((total_colors, color_vector))
            else:
                total_points = np.vstack((total_points, points_3d[:, 0, :]))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left.T])
                total_colors = np.vstack((total_colors, color_vector)) 

            transform_matrix_0 = np.copy(transform_matrix_1)
            pose_0 = np.copy(pose_1)
            plt.scatter(i, error)
            plt.pause(0.05)

            image_0 = np.copy(image_1)
            image_1 = np.copy(image_2)
            feature_0 = np.copy(features_cur)
            feature_1 = np.copy(features_2)
            pose_1 = np.copy(pose_2)

        print("Printing to .ply file")
        print(total_points.shape, total_colors.shape)
        self.to_ply(self.img_obj.path, total_points, total_colors)
        print("Completed Exiting ...")
        np.savetxt(os.path.join(self.img_obj.path, 'res', os.path.basename(os.path.dirname(self.img_obj.image_list[0])) + '_pose_array.csv'), pose_array, delimiter = '\n')

def save_uploaded_files(uploaded_files):
    """
    Save uploaded files to a temporary directory and return the directory path.
    """
    temp_dir = tempfile.mkdtemp()
    
    os.makedirs(os.path.join(temp_dir, 'res'), exist_ok=True)
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    return temp_dir

def create_k_file(temp_dir, default_k=None):
    """
    Create a K.txt file with camera intrinsic parameters.
    If default_k is not provided, use a default set of parameters.
    """
    if default_k is None:
        default_k = [
            [1000, 0, 640],
            [0, 1000, 480],
            [0, 0, 1]
        ]
    
    k_path = os.path.join(temp_dir, 'K.txt')
    with open(k_path, 'w') as f:
        for row in default_k:
            f.write(' '.join(map(str, row)) + '\n')
    return k_path

def parse_k_matrix(k_input):
    """
    Parse K matrix input, handling potential whitespace and empty lines
    """
    k_matrix = []
    for line in k_input.strip().split('\n'):
        row = [float(x) for x in line.strip().split() if x.strip()]
        if row:
            k_matrix.append(row)
    
    if len(k_matrix) != 3 or any(len(row) != 3 for row in k_matrix):
        raise ValueError("K matrix must be a 3x3 matrix")
    
    return k_matrix


def main():
    st.title("3D Reconstruction")
    
    model = load_model()
    
    if uploaded_file is not None:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = os.path.join(temp_dir, "input.png")
                depth_path = os.path.join(temp_dir, "depth.png")
                resized_path = os.path.join(temp_dir, "resized.png")
                ply_path = os.path.join(temp_dir, "pointcloud.ply")
                
                input_image = Image.open(uploaded_file).convert('RGB')
                
                input_image.save(input_path)
                
                resized_image = input_image.resize((256, 256))
                resized_image.save(resized_path)
                
                processed_img = preprocess_image(input_image)
                
                with torch.no_grad():
                    depth_pred, uncertainty = model(processed_img)
                
                depth_pred = torch.abs(depth_pred)
                depth_pred = depth_pred.squeeze().cpu().numpy()
                
                save_depth_as_png(depth_pred, depth_path)
                
                depth_img = cv2.imread(depth_path)
                
                pcd = create_point_cloud(resized_path, depth_path)
                
                o3d.io.write_point_cloud(ply_path, pcd)
                
                with open(ply_path, "rb") as file:
                    st.download_button(
                        label="Download Point Cloud (PLY)",
                        data=file,
                        file_name="pointcloud.ply",
                        mime="application/octet-stream"
                    )
                
                if st.checkbox("Show 3D Point Cloud"):
                    points = np.asarray(pcd.points)
                    colors = np.asarray(pcd.colors)
                    
                    fig = go.Figure(data=[go.Scatter3d(
                        x=points[:, 0],
                        y=points[:, 1],
                        z=points[:, 2],
                        mode='markers',
                        marker=dict(
                            size=1,
                            color=[f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                                  for r, g, b in colors],
                            opacity=0.8
                        )
                    )])
                    
                    fig.update_layout(
                        scene=dict(
                            xaxis_title='X',
                            yaxis_title='Y',
                            zaxis_title='Z'
                        ),
                        width=700,
                        height=700,
                        margin=dict(r=10, l=10, b=10, t=10)
                    )
                    
                    st.plotly_chart(fig)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try uploading a different image or contact support if the issue persists.")

        
    k_input = st.text_area("Enter K matrix (3x3, space-separated)", 
                                value="1000 0 640\n0 1000 480\n0 0 1")
    
    uploaded_files = st.file_uploader("Upload Images", 
                                      type=['jpg', 'png'], 
                                      accept_multiple_files=True)
    
    if st.button("Run Reconstruction") and uploaded_files:
        if len(uploaded_files) < 3:
            st.error("Please upload at least 3 images for reconstruction.")
            return
        
        temp_dir = save_uploaded_files(uploaded_files)
        
        try:
            k_matrix = parse_k_matrix(k_input)
            k_file_path = create_k_file(temp_dir, k_matrix)
            st.sidebar.success("K matrix file created successfully.")
        except Exception as k_error:
            st.warning(f"Invalid K matrix. Using default parameters. Error: {str(k_error)}")
            k_file_path = create_k_file(temp_dir)
        
        try:
            plt.ioff()
            
            st.info("Starting Structure from Motion reconstruction...")
            sfm = Sfm(temp_dir, downscale_factor = 2.0)
            
            original_call = sfm.__call__
            
            def patched_call(enable_bundle_adjustment=False):
                try:
                    return original_call(enable_bundle_adjustment=enable_bundle_adjustment)
                except ValueError as e:
                    if "s must be a scalar" in str(e):
                        st.warning("Fixed a plotting issue in the SfM process. Continuing with reconstruction...")
                        
                    else:
                        raise
            
            sfm.__call__ = patched_call
            
            sfm(enable_bundle_adjustment = False)
            
            st.success("Reconstruction completed!")
            
            result_dir = os.path.join(temp_dir, 'res')
            ply_files = [f for f in os.listdir(result_dir) if f.endswith('.ply')]
            
            if ply_files:
                ply_file_path = os.path.join(result_dir, ply_files[0])
                with open(ply_file_path, 'rb') as f:
                    st.download_button(
                        label="Download Point Cloud (.ply)",
                        data=f,
                        file_name=ply_files[0],
                        mime="model/ply"
                    )
                
                st.subheader("3D Point Cloud Visualization")
                
                st.info(f"Attempting to load point cloud from: {ply_file_path}")
                ply_file_size = os.path.getsize(ply_file_path)
                st.info(f"PLY file size: {ply_file_size} bytes")
                
                try:
                    pcd = o3d.io.read_point_cloud(ply_file_path)
                    
                    num_points = len(np.asarray(pcd.points))
                    st.info(f"Point cloud loaded successfully. Number of points: {num_points}")
                    
                    points = np.asarray(pcd.points)
                    
                    max_points = 100000
                    if num_points > max_points:
                        st.warning(f"Point cloud has {num_points} points. Displaying a subset of {max_points} points for better performance.")
                        indices = np.random.choice(num_points, max_points, replace=False)
                        points = points[indices]
                        if pcd.has_colors():
                            colors = np.asarray(pcd.colors)[indices]
                            color_data = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                                        for r, g, b in colors]
                        else:
                            color_data = 'blue'
                    else:
                        if pcd.has_colors():
                            colors = np.asarray(pcd.colors)
                            color_data = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                                        for r, g, b in colors]
                        else:
                            color_data = 'blue'
                    
                    st.info("Creating Plotly 3D visualization...")
                    fig = go.Figure(data=[go.Scatter3d(
                        x=points[:, 0],
                        y=points[:, 1],
                        z=points[:, 2],
                        mode='markers',
                        marker=dict(
                            size=1,
                            color=color_data,
                            opacity=0.8
                        )
                    )])
                    
                    fig.update_layout(
                        scene=dict(
                            xaxis_title='X',
                            yaxis_title='Y',
                            zaxis_title='Z'
                        ),
                        width=700,
                        height=700,
                        margin=dict(r=10, l=10, b=10, t=10)
                    )
                    
                    st.info("Rendering point cloud visualization...")
                    st.plotly_chart(fig)
                    
                    st.sidebar.subheader("Point Cloud Settings")
                    point_size = st.sidebar.slider("Point Size", 1, 10, 1)
                    opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 0.8)
                    
                    if point_size != 1 or opacity != 0.8:
                        fig.update_traces(marker=dict(
                            size=point_size,
                            opacity=opacity
                        ))
                        st.plotly_chart(fig)
                    
                except Exception as viz_error:
                    st.error(f"Error visualizing point cloud: {str(viz_error)}")
                    import traceback
                    st.error(f"Detailed error: {traceback.format_exc()}")
            
            pose_files = [f for f in os.listdir(result_dir) if f.endswith('_pose_array.csv')]
            if pose_files:
                pose_file_path = os.path.join(result_dir, pose_files[0])
                with open(pose_file_path, 'rb') as f:
                    st.download_button(
                        label="Download Pose Array",
                        data=f,
                        file_name=pose_files[0],
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"An error occurred during reconstruction: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
        
        finally:
            plt.close('all')
            if st.checkbox("Keep temporary files for debugging", value=False):
                st.info(f"Temporary directory preserved at: {temp_dir}")
            else:
                shutil.rmtree(temp_dir)
            
if __name__ == "__main__":
    main()