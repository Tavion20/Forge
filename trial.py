import cv2
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

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

def create_mesh_from_pointcloud(pcd, depth_map, voxel_size=0.05, depth=8, scale=1.1, linear_fit=False):
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, scale=scale, linear_fit=linear_fit)
    
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    mesh.scale(np.max(depth_map), center=mesh.get_center())
    
    return mesh

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to('cpu')
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

img = cv2.imread("./test_image2.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to('cpu')

with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()

height, width = img.shape[:2]
fx = fy = max(height, width) 
cx, cy = width / 2, height / 2  

point_cloud = depth_to_pointcloud(depth_map, img, fx, fy, cx, cy)

mesh = create_mesh_from_pointcloud(point_cloud, depth_map)

o3d.visualization.draw_geometries([point_cloud, mesh])

o3d.io.write_point_cloud("output_pointcloud.ply", point_cloud)
o3d.io.write_triangle_mesh("output_mesh.ply", mesh)