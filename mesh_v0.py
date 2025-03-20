import open3d as o3d

# Load the point cloud
pcd = o3d.io.read_point_cloud("./res/GustavIIAdolf.ply")

# Check if the file is loaded correctly
if not pcd.has_points():
    print("Error: Point cloud file not loaded or empty.")
    exit()

# Compute normals (Fix for missing normals)
print("Estimating normals...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Run Poisson surface reconstruction
print("Running Poisson reconstruction...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

# Save the generated mesh
o3d.io.write_triangle_mesh("./res/output_mesh.ply", mesh)
print("Mesh saved as output_mesh.ply")

# Visualize
o3d.visualization.draw_geometries([mesh])
