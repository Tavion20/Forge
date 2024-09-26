import torch
import cv2
import numpy as np
import open3d as o3d

# Load MiDaS model and transformation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to('cpu')
midas.eval()

# Load the transformation for the model
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

def get_depth_map_and_rgb(img_path):
    """
    Reads an image, applies MiDaS model to generate a depth map, 
    and returns the depth map and RGB image.
    """
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Open3D

    input_batch = transform(img_rgb).to('cpu')

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],  # Match original image size
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    return depth_map, img_rgb  # Return the depth map and the RGB image

def create_colored_point_cloud(depth_map, img_rgb):
    """
    Creates a colored point cloud from a depth map and RGB image.
    """
    # Normalize depth map for visualization (optional)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Create a point cloud from the depth image
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth_map.astype(np.float32)),
        o3d.camera.PinholeCameraIntrinsic(
            width=img_rgb.shape[1],
            height=img_rgb.shape[0],
            fx=500, fy=500,  # Adjust focal length if necessary
            cx=img_rgb.shape[1] / 2,
            cy=img_rgb.shape[0] / 2,
        )
    )

    # Assign colors to the point cloud from the RGB image
    colors = img_rgb.astype(np.float32) / 255.0  # Normalize RGB values between 0 and 1
    pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))

    return pcd

def combine_point_clouds(pcd_list):
    """
    Combines multiple point clouds into one.
    """
    combined_pcd = pcd_list[0]
    for pcd in pcd_list[1:]:
        combined_pcd += pcd
    return combined_pcd

# Load and process three images
image_paths = ["./test_img2.jpg"]  # Replace with your image paths
point_clouds = []

for img_path in image_paths:
    depth_map, img_rgb = get_depth_map_and_rgb(img_path)
    pcd = create_colored_point_cloud(depth_map, img_rgb)
    point_clouds.append(pcd)

# Combine the three point clouds into one
combined_pcd = combine_point_clouds(point_clouds)

# Visualize the combined point cloud
o3d.visualization.draw_geometries([combined_pcd])
