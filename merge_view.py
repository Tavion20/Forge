import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
import torch
from scipy.spatial.transform import Rotation
import streamlit as st

# [Previous helper functions remain the same...]
def estimate_focal_length(image_size):
    """Estimate focal length using image size heuristic"""
    return max(image_size) * 1.2

def estimate_camera_intrinsics(image):
    """Estimate camera intrinsics without calibration"""
    height, width = image.shape[:2]
    focal_length = estimate_focal_length((width, height))
    cx = width / 2
    cy = height / 2
    camera_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((5,1), dtype=np.float32)
    return camera_matrix, dist_coeffs

class MultiViewReconstructor:
    def __init__(self):
        self.images = []
        self.camera_matrices = []
        self.feature_detector = cv2.SIFT_create(nfeatures=3000)
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )

    def add_image(self, image):
        """Add a new image to the reconstruction"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        self.images.append({
            'image': image,
            'gray': gray,
            'keypoints': None,
            'descriptors': None,
            'R': None,
            't': None
        })
        
        kp, des = self.feature_detector.detectAndCompute(gray, None)
        self.images[-1]['keypoints'] = kp
        self.images[-1]['descriptors'] = des
        
        K, dist = estimate_camera_intrinsics(image)
        self.camera_matrices.append(K)
        
        return len(self.images) - 1

    def match_images(self, idx1, idx2, ratio=0.7):
        """Match features between two images"""
        matches = self.matcher.knnMatch(
            self.images[idx1]['descriptors'],
            self.images[idx2]['descriptors'],
            k=2
        )
        
        good_matches = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)
                
        return good_matches

    def estimate_pose(self, idx1, idx2, matches):
        """Estimate relative pose between two images"""
        pts1 = np.float32([self.images[idx1]['keypoints'][m.queryIdx].pt 
                          for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([self.images[idx2]['keypoints'][m.trainIdx].pt 
                          for m in matches]).reshape(-1, 1, 2)
        
        E, mask = cv2.findEssentialMat(
            pts1, pts2, 
            self.camera_matrices[idx1],
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        _, R, t, mask = cv2.recoverPose(
            E, pts1, pts2, 
            self.camera_matrices[idx1]
        )
        
        return R, t, pts1[mask.ravel() == 255], pts2[mask.ravel() == 255]

    def triangulate_points(self, idx1, idx2, pts1, pts2):
        """Triangulate 3D points from two views"""
        P1 = np.dot(self.camera_matrices[idx1], np.hstack((np.eye(3), np.zeros((3,1)))))
        P2 = np.dot(self.camera_matrices[idx2], np.hstack((self.images[idx2]['R'], self.images[idx2]['t'])))
        
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_4d = points_4d.squeeze(axis=1)
        print("Shape of points_4d after squeeze:", points_4d.shape)
        
        mask = np.abs(points_4d[3]) > 1e-8
        points_3d = np.zeros((3, points_4d.shape[1]))



        # Perform the operation
        epsilon = 1e-8
        points_3d[:, mask] = points_4d[:3, mask] / (points_4d[3, mask] + epsilon)

        
        valid_mask = ~np.any(np.isnan(points_3d) | np.isinf(points_3d), axis=0)
        points_3d = points_3d[:, valid_mask]
        
        return points_3d.T, valid_mask

    def reconstruct(self):
        """Perform multi-view reconstruction"""
        if len(self.images) < 2:
            raise ValueError("Need at least 2 images for reconstruction")
            
        self.images[0]['R'] = np.eye(3)
        self.images[0]['t'] = np.zeros((3,1))
        
        all_3d_points = []
        all_colors = []
        
        for i in range(1, len(self.images)):
            matches = self.match_images(i-1, i)
            
            if len(matches) < 8:
                continue
                
            R, t, pts1, pts2 = self.estimate_pose(i-1, i, matches)
            
            self.images[i]['R'] = R
            self.images[i]['t'] = t
            
            # Triangulate points and get valid mask
            points_3d, valid_mask = self.triangulate_points(i-1, i, pts1, pts2)
            
            if len(points_3d) == 0:
                continue
                
            # Extract colors for valid points only
            img = self.images[i-1]['image']
            valid_pts1 = pts1[valid_mask.ravel()]
            valid_pts1 = valid_pts1[:, 0, :]  # Extract the (x, y) coordinates

            st.write("pts1 shape:", pts1.shape)
            st.write("valid_mask shape:", valid_mask.shape)
            st.write("valid_pts1 shape:", valid_pts1.shape)

            colors = []
            for pt in valid_pts1:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    colors.append(img[y, x])
                else:
                    colors.append(np.array([0, 0, 0]))

            colors = np.array(colors)

            
            all_3d_points.append(points_3d)
            all_colors.append(colors)
            
        if not all_3d_points:
            raise ValueError("No valid 3D points could be reconstructed")
            
        all_3d_points = np.vstack(all_3d_points).astype(np.float64)
        all_colors = np.vstack(all_colors).astype(np.float64)
        
        return all_3d_points, all_colors

    def create_open3d_point_cloud(self, points, colors):
        """Create an Open3D point cloud"""
        valid_mask = ~np.any(np.isnan(points) | np.isinf(points), axis=1)
        points = points[valid_mask]
        colors = colors[valid_mask]
        
        if len(points) == 0:
            raise ValueError("No valid points for point cloud creation")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        # print(type(pcd))

        # if len(points) > 100:
        #     pcd, _ = pcd.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)
        
        return pcd

    def save_point_cloud(self, filename, points, colors):
        """Save point cloud to file"""
        pcd = self.create_open3d_point_cloud(points, colors)
        o3d.io.write_point_cloud(filename, pcd)

# Streamlit UI
st.title("Multi-view 3D Reconstruction")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if len(uploaded_files) >= 2:
    reconstructor = MultiViewReconstructor()
    
    # Add all images
    for file in uploaded_files:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        reconstructor.add_image(img)
        st.image(img, caption=f"Added image: {file.name}", use_column_width=True)
    
    if st.button("Perform 3D Reconstruction"):
        with st.spinner("Reconstructing 3D model..."):
            points_3d, colors = reconstructor.reconstruct()
            
            # Create and display point cloud
            pcd = reconstructor.create_open3d_point_cloud(points_3d, colors)
            o3d.visualization.draw_geometries([pcd])
            # Save point cloud
            output_file = "reconstruction.ply"
            o3d.io.write_point_cloud(output_file, pcd)
            st.success(f"3D reconstruction completed! Saved to {output_file}")
            
            # Display statistics
            st.write(f"Number of 3D points: {len(points_3d)}")
            st.write(f"Point cloud bounds:")
            st.write(f"X: [{points_3d[:,0].min():.2f}, {points_3d[:,0].max():.2f}]")
            st.write(f"Y: [{points_3d[:,1].min():.2f}, {points_3d[:,1].max():.2f}]")
            st.write(f"Z: [{points_3d[:,2].min():.2f}, {points_3d[:,2].max():.2f}]")
                