import cv2
import numpy as np
import open3d as o3d
import os

def extract_frames(video_path, output_folder, frame_interval=10):
    """Extract frames from video at given interval."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video = cv2.VideoCapture(video_path)
    count = 0
    frame_count = 0
    
    while True:
        success, frame = video.read()
        if not success:
            break
        if count % frame_interval == 0:
            cv2.imwrite(os.path.join(output_folder, f'frame_{frame_count:04d}.jpg'), frame)
            frame_count += 1
        count += 1
    
    video.release()
    return frame_count

def create_point_cloud_from_images(image_folder):
    """Create point cloud from images using Open3D."""
    rgb_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')])
    
    # Read the first image to get dimensions
    first_image = o3d.io.read_image(rgb_files[0])
    width, height = np.asarray(first_image).shape[1], np.asarray(first_image).shape[0]
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, width, width, width/2, height/2)
    
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for i in range(len(rgb_files)):
        print(f"Integrating frame {i+1}/{len(rgb_files)}")
        color = o3d.io.read_image(rgb_files[i])
        # Create a dummy depth image with the same dimensions as the color image
        depth = o3d.geometry.Image(np.ones((height, width), dtype=np.uint16) * 1000)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=3.0, convert_rgb_to_intensity=False)
        
        if i == 0:
            pose = np.identity(4)
        else:
            success, pose = estimate_motion(rgbd, rgbd_prev, intrinsic)
            if not success:
                continue
            pose = np.dot(pose_prev, pose)
        
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
        rgbd_prev = rgbd
        pose_prev = pose

    pcd = volume.extract_point_cloud()
    return pcd

def estimate_motion(source, target, intrinsic, max_depth_diff=0.07, max_depth=3.0):
    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_diff_max = max_depth_diff
    option.depth_max = max_depth
    
    odo_init = np.identity(4)
    success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
        source, target, intrinsic, odo_init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
    
    return success, trans

# Main process
video_path = './test_vid2.mp4'
output_folder = './ext'

# Extract frames
num_frames = extract_frames(video_path, output_folder)
print(f"Extracted {num_frames} frames.")

# Create point cloud
# point_cloud = create_point_cloud_from_images(output_folder)

# Visualize the point cloud
# o3d.visualization.draw_geometries([point_cloud])

# # Save the point cloud
# o3d.io.write_point_cloud("output_pointcloud.ply", point_cloud)