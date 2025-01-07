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

import streamlit as st
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple

# Assuming the previously defined model and modules (UniqueDepthEstimationModel, MultiImageProcessor) exist.
class MultiVideoProcessor:
    def __init__(self):
        self.model = UniqueDepthEstimationModel()
        self.model.load_state_dict(torch.load("./unique_depth_model_epoch_13.pth", 
                                              map_location=torch.device('cpu')), strict=False)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.focal_length = None
        self.camera_matrix = None

    def preprocess_frame(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def estimate_depth(self, frame):
        input_image = self.preprocess_frame(frame)
        with torch.no_grad():
            depth_pred, _, _ = self.model(input_image)

        depth_img = depth_pred.squeeze(0).squeeze(0).cpu().numpy()
        depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
        depth_img = (depth_img * 255).astype(np.uint8)

        return depth_img

    def process_frame_pair(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)

        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good_matches) < 10:
            return None, None, None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        if self.camera_matrix is None:
            self.focal_length = max(img1_gray.shape)
            self.camera_matrix = np.array([
                [self.focal_length, 0, img1_gray.shape[1] / 2],
                [0, self.focal_length, img1_gray.shape[0] / 2],
                [0, 0, 1]
            ], dtype=np.float32)

        E, mask = cv2.findEssentialMat(src_pts, dst_pts, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, self.camera_matrix)

        return R, t, E

def main():
    st.title("Video 3D Reconstruction with Depth Estimation")

    processor = MultiVideoProcessor()

    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if video_file:
        temp_file = f"temp_video.{video_file.name.split('.')[-1]}"
        with open(temp_file, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(temp_file)
        frame_interval = st.slider("Frame interval for processing", 1, 30, 5)

        frames = []
        depth_maps = []
        camera_poses = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if len(frames) % frame_interval == 0:
                frames.append(frame)
                depth_map = processor.estimate_depth(frame)
                depth_maps.append(depth_map)

                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {len(frames)}")
                st.image(depth_map, caption=f"Depth Map {len(frames)}")

        cap.release()

        if len(frames) >= 2:
            st.subheader("Camera Pose Estimation")

            for i in range(len(frames) - 1):
                R, t, _ = processor.process_frame_pair(frames[i], frames[i + 1])
                if R is not None and t is not None:
                    camera_poses.append((R, t))

            if camera_poses:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')

                ax.scatter(0, 0, 0, c='r', marker='^', s=100, label='Camera 1')

                current_position = np.zeros(3)
                current_rotation = np.eye(3)
                cmap = plt.get_cmap("viridis")

                for idx, (R, t) in enumerate(camera_poses):
                    current_rotation = current_rotation @ R
                    current_position = current_position + current_rotation @ t.ravel()

                    color = cmap(idx / len(camera_poses))
                    ax.scatter(current_position[0], 
                               current_position[1], 
                               current_position[2], 
                               c=[color], marker='^', s=100, 
                               label=f'Camera {idx+2}')

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.legend()

                st.pyplot(fig)

if __name__ == "__main__":
    main()
