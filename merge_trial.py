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
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

class MultiImageProcessor:
    def __init__(self):
        self.model = UniqueDepthEstimationModel()
        self.model.load_state_dict(torch.load("./unique_depth_model_epoch_13.pth", 
                                            map_location=torch.device('cpu')), strict=False)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Camera parameters
        self.focal_length = None
        self.focal_length = 525.0
        self.camera_matrix = np.array([
            [self.focal_length, 0, 319.5],
            [0, self.focal_length, 239.5],
            [0, 0, 1]
        ], dtype=np.float32)
        
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0).to(self.device)
    
    def estimate_depth(self, image_path):
        input_image = self.preprocess_image(image_path)
        with torch.no_grad():
            depth_pred, uncertainty_pred, normals_pred = self.model(input_image)
        
        depth_img = depth_pred.squeeze(0).squeeze(0).cpu().numpy()
        depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
        depth_img = (depth_img * 255).astype(np.uint8)
        
        return depth_img
    
    def create_rgbd_image(self, rgb_image, depth_image):
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        rgb_resized = cv2.resize(rgb_image, (depth_colormap.shape[1], depth_colormap.shape[0]))
        rgbd_image = cv2.addWeighted(rgb_resized, 0.6, depth_colormap, 0.4, 0)
        
        return rgbd_image
    
    def process_image_pair(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2
            
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)
        
        # Match features
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 10:
            return None, None, None
            
        # Get matched key points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Calculate essential matrix
        if self.camera_matrix is None:
            self.focal_length = max(img1_gray.shape)
            self.camera_matrix = np.array([
                [self.focal_length, 0, img1_gray.shape[1] / 2],
                [0, self.focal_length, img1_gray.shape[0] / 2],
                [0, 0, 1]
            ], dtype=np.float32)
            
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, self.camera_matrix)
        
        return R, t, E
    
    def generate_point_cloud(self, rgb_image, depth_image):
        """
        Generate a 3D point cloud from RGB and depth images.
        """
        height, width = depth_image.shape
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Mesh grid
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        z = depth_image / 1000.0  # Convert depth to meters (if needed)

        x3d = (x - cx) * z / fx
        y3d = (y - cy) * z / fy
        z3d = z

        # Combine into point cloud format
        points = np.stack((x3d, y3d, z3d), axis=-1).reshape(-1, 3)
        colors = rgb_image.reshape(-1, 3) / 255.0  # Normalize colors

        # Filter valid points
        valid_mask = z3d.flatten() > 0
        points = points[valid_mask]
        colors = colors[valid_mask]

        # Create Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        return point_cloud


# def main():
#     st.title("Multi-Image 3D Reconstruction with Depth Estimation")
    
#     processor = MultiImageProcessor()
    
#     # File uploader for multiple images
#     uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
#     if uploaded_files:
#         num_images = len(uploaded_files)
#         st.write(f"Processing {num_images} images...")
        
#         # Create columns for displaying images
#         cols = st.columns(min(3, num_images))
        
#         # Process each image for depth estimation
#         processed_images = []
#         for idx, file in enumerate(uploaded_files):
#             with cols[idx % 3]:
#                 # Display original image
#                 input_image = Image.open(file)
#                 st.image(input_image, caption=f"Image {idx+1}", use_column_width=True)
                
#                 # Estimate depth
#                 depth_img = processor.estimate_depth(file)
#                 st.image(depth_img, caption=f"Depth Map {idx+1}", use_column_width=True)
                
#                 # Create RGBD image
#                 rgb_np = np.array(input_image)
#                 rgbd_image = processor.create_rgbd_image(rgb_np, depth_img)
#                 st.image(rgbd_image, caption=f"RGB-D {idx+1}", use_column_width=True)
                
#                 processed_images.append((np.array(input_image), depth_img))
        
#         # Camera pose estimation and 3D reconstruction
#         if len(processed_images) >= 2:
#             st.subheader("Camera Poses and 3D Reconstruction")
            
#             # Process consecutive pairs of images
#             for i in range(len(processed_images) - 1):
#                 img1, _ = processed_images[i]
#                 img2, _ = processed_images[i + 1]
                
#                 R, t, E = processor.process_image_pair(img1, img2)
#                 if R is not None and t is not None:
#                     camera_poses.append((R, t))
            
#             # Visualize camera poses
#             if camera_poses:
#                 fig = plt.figure(figsize=(10, 10))
#                 ax = fig.add_subplot(111, projection='3d')
                
#                 # Plot first camera at origin
#                 ax.scatter(0, 0, 0, c='r', marker='^', s=100, label='Camera 1')
                
#                 # Plot subsequent cameras
#                 current_position = np.zeros(3)
#                 current_rotation = np.eye(3)
#                 cmap = plt.get_cmap("viridis")
                
#                 for idx, (R, t) in enumerate(camera_poses):
#                     current_rotation = current_rotation @ R
#                     current_position = current_position + current_rotation @ t.ravel()
                    
#                     color = cmap(idx / len(camera_poses))  # Normalize index to colormap range

#                     ax.scatter(current_position[0], 
#                             current_position[1], 
#                             current_position[2], 
#                             c=[color], marker='^', s=100, 
#                             label=f'Camera {idx+2}')
                
#                 ax.set_xlabel('X')
#                 ax.set_ylabel('Y')
#                 ax.set_zlabel('Z')
#                 ax.legend()
                
#                 st.pyplot(fig)

# def create_point_cloud(depth_map, rgb_image, camera_intrinsics):
#     # Get depth map dimensions
#     h, w = depth_map.shape
    
#     # Create mesh grid of pixel coordinates
#     i, j = np.meshgrid(np.arange(w), np.arange(h))
    
#     # Convert pixel coordinates to 3D space
#     z = depth_map
#     x = (j - camera_intrinsics[0, 2]) * z / camera_intrinsics[0, 0]
#     y = (i - camera_intrinsics[1, 2]) * z / camera_intrinsics[1, 1]
    
#     # Stack into 3D points
#     points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    
#     # Get RGB values for the point cloud
#     colors = rgb_image.reshape(-1, 3) / 255.0  # Normalize to [0, 1]
    
#     # Create Open3D point cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
    
#     return pcd
    def generate_point_cloud(self, rgb_image, depth_image):
        """
        Generate a 3D point cloud from RGB and depth images.
        """
        height, width = depth_image.shape
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Ensure rgb_image is resized to match depth_image
        rgb_image = cv2.resize(rgb_image, (width, height))

        # Create mesh grid
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert depth to meters (assuming depth is normalized between 0-255)
        z = depth_image.astype(float) / 255.0  # Normalize to 0-1 range
        z = z * 10.0  # Scale to reasonable depth range (e.g., 0-10 meters)

        # Calculate 3D coordinates
        x3d = (x - cx) * z / fx
        y3d = (y - cy) * z / fy
        z3d = z

        # Combine into point cloud format
        points = np.stack((x3d, y3d, z3d), axis=-1).reshape(-1, 3)
        colors = rgb_image.reshape(-1, 3) / 255.0  # Normalize colors

        # Filter out invalid points (e.g., zero depth)
        valid_mask = (z3d.flatten() > 0) & (z3d.flatten() < 10)  # Remove points too far or too close
        points = points[valid_mask]
        colors = colors[valid_mask]

        # Create Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # # Optional: Remove statistical outliers
        # if len(points) > 100:  # Only if we have enough points
        #     point_cloud, _ = point_cloud.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)

        return point_cloud

def main():
    st.title("Multi-Image 3D Reconstruction with Depth Estimation and Point Cloud")
    
    processor = MultiImageProcessor()
    
    # File uploader for multiple images
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        num_images = len(uploaded_files)
        st.write(f"Processing {num_images} images...")
        
        # Create columns for displaying images
        cols = st.columns(min(3, num_images))
        
        # Process each image for depth estimation
        processed_images = []
        point_clouds = []
        
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 3]:
                try:
                    # Display original image
                    input_image = Image.open(file)
                    st.image(input_image, caption=f"Image {idx+1}", use_column_width=True)
                    
                    # Convert PIL Image to numpy array
                    rgb_np = np.array(input_image)
                    
                    # Estimate depth
                    depth_img = processor.estimate_depth(file)
                    st.image(depth_img, caption=f"Depth Map {idx+1}", use_column_width=True)
                    
                    # Create RGBD image
                    rgbd_image = processor.create_rgbd_image(rgb_np, depth_img)
                    st.image(rgbd_image, caption=f"RGB-D {idx+1}", use_column_width=True)
                    
                    # Generate and store point cloud
                    st.write("Generating point cloud...")
                    point_cloud = processor.generate_point_cloud(rgb_np, depth_img)
                    point_clouds.append(point_cloud)
                    st.success(f"Generated point cloud for image {idx+1}")
                    
                    processed_images.append((rgb_np, depth_img))
                    
                except Exception as e:
                    st.error(f"Error processing image {idx+1}: {str(e)}")
                    st.write("Error details:", e)
                    continue
        
        # Visualize individual point clouds
        if point_clouds:
            st.subheader("Point Cloud Visualization")
            
            for idx, pcd in enumerate(point_clouds):
                try:
                    o3d.visualization.draw_geometries([pcd])
                    # # Save point cloud to PLY file
                    # output_filename = f"point_cloud_{idx+1}.ply"
                    # o3d.io.write_point_cloud(output_filename, pcd)
                    # st.write(f"Point cloud {idx+1} saved as {output_filename}")
                    
                    # Create visualization
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(visible=False)  # Create hidden window
                    vis.add_geometry(pcd)
                    
                    # Capture view and save as image
                    vis.poll_events()
                    vis.update_renderer()
                    image = vis.capture_screen_float_buffer(False)
                    vis.destroy_window()
                    
                    # Convert to numpy array and display
                    image_np = np.asarray(image)
                    st.image(image_np, caption=f"Point Cloud {idx+1}", use_column_width=True)
                    
                except Exception as e:
                    st.error(f"Error visualizing point cloud {idx+1}: {str(e)}")
                    continue

if __name__ == "__main__":
    main()