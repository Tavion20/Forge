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

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Uploaded Image", use_column_width=True)

    depth_img, normals_img = estimate_depth_normals(uploaded_file)  

    st.subheader("Depth Map")
    st.image(depth_img, caption="Depth Map", use_column_width=True, clamp=True)

    if st.checkbox("Show RGB-D Image"):
        rgb_image_np = np.array(input_image)
        rgbd_image = create_rgbd_image(rgb_image_np, depth_img)
        st.image(rgbd_image, caption="RGB-D Image", use_column_width=True)
        
    if st.checkbox("Show 3D Point Cloud (Plotly)"):
        depth_np = depth_img.astype(np.float32) / 255.0
        rgb_image_resized = cv2.resize(np.array(input_image), (depth_np.shape[1], depth_np.shape[0]))
        points, colors = depth_to_point_cloud(depth_np, rgb=rgb_image_resized)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=colors,
                opacity=0.8
            )
        )])
        
        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ), 
        width=700, 
        height=700,
        margin=dict(r=10, l=10, b=10, t=10))
        
        st.plotly_chart(fig)




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

st.title("3D Reconstruction with Feature Matching and Camera Triangulation")

uploaded_file1 = st.file_uploader("Choose the first image...", type=["jpg", "jpeg", "png"])
uploaded_file2 = st.file_uploader("Choose the second image...", type=["jpg", "jpeg", "png"])

if uploaded_file1 and uploaded_file2:
    img1 = cv2.imdecode(np.frombuffer(uploaded_file1.read(), np.uint8), 1)
    img2 = cv2.imdecode(np.frombuffer(uploaded_file2.read(), np.uint8), 1)

    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)

    result, good_matches, keypoints1, keypoints2, keypoints1_orb, keypoints2_orb = sift_orb_matching(img1, img2, orb_matches_limit=100, sift_ratio=0.8)


    st.image(result, caption="Feature Matching Result", use_column_width=True)

    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    focal_length = max(img1.shape)
    camera_matrix = np.array([[focal_length, 0, img1.shape[1] / 2],
                              [0, focal_length, img1.shape[0] / 2],
                              [0, 0, 1]], dtype=np.float32)

    essential_matrix, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(essential_matrix, pts1, pts2, camera_matrix)

    pts1_inliers = pts1[mask_pose.ravel() == 255]
    pts2_inliers = pts2[mask_pose.ravel() == 255]

    if pts1_inliers.shape[0] < 2 or pts2_inliers.shape[0] < 2:
        pts1_inliers = pts1
        pts2_inliers = pts2

    P1 = np.eye(3, 4, dtype=np.float32)
    P2 = np.hstack((R, t)).astype(np.float32)

    pts1_inliers = pts1_inliers.T.astype(np.float32)
    pts2_inliers = pts2_inliers.T.astype(np.float32)

    points_4d = cv2.triangulatePoints(P1, P2, pts1_inliers, pts2_inliers)
    points_3d = points_4d[:3] / points_4d[3]

    plot_3d_points_and_cameras(points_3d, R, t)
