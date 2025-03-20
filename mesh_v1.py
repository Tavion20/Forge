import open3d as o3d
import numpy as np

# Load the point cloud
pcd = o3d.io.read_point_cloud("./res/GustavIIAdolf.ply")

# Check if the file is loaded correctly
if not pcd.has_points():
    print("Error: Point cloud file not loaded or empty.")
    exit()

# Remove outliers
print("Removing outliers...")
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)

# Downsample for uniform point distribution
print("Downsampling...")
pcd = pcd.voxel_down_sample(voxel_size=0.002)

# Estimate normals
print("Estimating normals...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))
pcd.orient_normals_consistent_tangent_plane(k=100)

# Poisson surface reconstruction
print("Running Poisson reconstruction...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)

# Convert densities to numpy array
densities = np.asarray(densities)

# Compute a threshold to remove low-density areas (background)
density_threshold = np.percentile(densities, 5)  # Removes bottom 5% of low-density points
vertices_to_remove = densities < density_threshold

# Create a clean mesh by keeping only high-density vertices
mesh.remove_vertices_by_mask(vertices_to_remove)

# Apply Laplacian smoothing to refine the mesh
print("Applying Laplacian smoothing...")
mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)

# Save the improved mesh
o3d.io.write_triangle_mesh("./res/output_mesh_clean.ply", mesh)
print("Mesh saved as output_mesh_clean.ply")

# Visualize
o3d.visualization.draw_geometries([mesh])
