import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

def detect_and_match_features(img1, img2, min_matches=10):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), None)
    
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        raise ValueError("Insufficient features detected in one or both images")
    
    flann_params = dict(algorithm=1, trees=5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    matches = matcher.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    print(f"Number of good matches: {len(good_matches)}")
    
    if len(good_matches) < min_matches:
        raise ValueError(f"Not enough good matches found. Found {len(good_matches)}, minimum required is {min_matches}")
    
    return kp1, kp2, good_matches

def visualize_matches(img1, kp1, img2, kp2, matches, pair_index):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(20,10))
    plt.imshow(img_matches)
    plt.title(f"Matches for pair {pair_index}: {len(matches)}")
    plt.savefig(f"matches_{pair_index}.png")
    plt.close()

def estimate_pose(kp1, kp2, matches, K):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    if len(src_pts) < 5 or len(dst_pts) < 5:
        raise ValueError(f"Not enough points for pose estimation. Found {len(src_pts)} points.")
    
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    if E is None or E.shape != (3, 3):
        raise ValueError(f"Failed to estimate Essential matrix. Shape: {E.shape if E is not None else None}")
    
    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, K, mask=mask)
    
    if R is None or t is None:
        raise ValueError("Failed to recover pose from Essential matrix")
    
    print(f"Pose estimation results:")
    print(f"R:\n{R}")
    print(f"t:\n{t}")
    print(f"Number of inliers: {np.sum(mask)}")
    
    return R, t, mask

def triangulate_points(kp1, kp2, matches, K, R, t, mask, img1, img2):
    P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(K, np.hstack((R, t)))
    
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    
    points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    
    points_3d = points_4d[:3, :] / points_4d[3, :]
    points_3d = points_3d.T
    
    mask = mask.ravel() != 0
    z_threshold = np.median(points_3d[:, 2]) * 10  # Adjust this threshold as needed
    mask = np.logical_and(mask, np.abs(points_3d[:, 2]) < z_threshold)
    points_3d = points_3d[mask]
    
    # Get colors from the first image
    colors = np.array([img1[int(kp1[m.queryIdx].pt[1]), int(kp1[m.queryIdx].pt[0])] for m in matches])[mask]
    
    if len(points_3d) == 0:
        raise ValueError("No valid points after triangulation and filtering")
    
    print(f"Triangulated points statistics:")
    print(f"  Number of points: {points_3d.shape[0]}")
    print(f"  Mean: {np.mean(points_3d, axis=0)}")
    print(f"  Std dev: {np.std(points_3d, axis=0)}")
    print(f"  Min: {np.min(points_3d, axis=0)}")
    print(f"  Max: {np.max(points_3d, axis=0)}")
    
    return points_3d, colors

def process_image_pair(img1, img2, K, pair_index):
    kp1, kp2, matches = detect_and_match_features(img1, img2)
    visualize_matches(img1, kp1, img2, kp2, matches, pair_index)
    
    R, t, mask = estimate_pose(kp1, kp2, matches, K)
    points_3d, colors = triangulate_points(kp1, kp2, matches, K, R, t, mask, img1, img2)
    
    return points_3d, colors

def estimate_camera_matrix(img):
    height, width = img.shape[:2]
    focal_length = max(height, width)
    cx, cy = width / 2, height / 2
    K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
    return K

# Main script
input_folder = './extracted'
image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

if len(image_files) < 2:
    raise ValueError("At least two images are required for 3D reconstruction.")

all_points_3d = []
all_colors = []

for i in range(len(image_files) - 1):
    img1 = load_image(os.path.join(input_folder, image_files[i]))
    img2 = load_image(os.path.join(input_folder, image_files[i + 1]))
    
    K = estimate_camera_matrix(img1)
    print(f"Estimated camera matrix for pair {i}:\n{K}")
    
    print(f"Processing image pair {i+1} and {i+2}")
    try:
        points_3d, colors = process_image_pair(img1, img2, K, i)
        all_points_3d.append(points_3d)
        all_colors.append(colors)
        print(f"Successfully processed pair {i+1} and {i+2}")
    except Exception as e:
        print(f"Error processing pair {i+1} and {i+2}: {str(e)}")
        print(f"Skipping to next pair")
        continue

if not all_points_3d:
    raise ValueError("No valid 3D points were generated from any image pair.")

# Combine all point clouds
combined_points = np.vstack(all_points_3d)
combined_colors = np.vstack(all_colors)

# Create combined point cloud
pcd_combined = o3d.geometry.PointCloud()
pcd_combined.points = o3d.utility.Vector3dVector(combined_points)
pcd_combined.colors = o3d.utility.Vector3dVector(combined_colors / 255.0)  # Normalize colors to [0, 1]

# Remove outliers
cl, ind = pcd_combined.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd_combined = pcd_combined.select_by_index(ind)

# Visualize combined point cloud
o3d.visualization.draw_geometries([pcd_combined], window_name="Combined Point Cloud")

# Save results
o3d.io.write_point_cloud("output_pointcloud_combined.ply", pcd_combined)

# Additional diagnostic information
print("\nFinal point cloud information:")
print(f"Total number of points: {len(pcd_combined.points)}")
print(f"Point cloud center: {pcd_combined.get_center()}")
print(f"Point cloud dimensions: {pcd_combined.get_max_bound() - pcd_combined.get_min_bound()}")

# Create a histogram of point distances from the center
distances = np.asarray([np.linalg.norm(p - pcd_combined.get_center()) for p in pcd_combined.points])
plt.figure(figsize=(10, 5))
plt.hist(distances, bins=50)
plt.title("Histogram of Point Distances from Center")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.savefig("point_distance_histogram.png")
plt.close()

print("Point distance histogram saved as 'point_distance_histogram.png'")