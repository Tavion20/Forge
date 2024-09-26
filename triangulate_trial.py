import cv2
import numpy as np

# Step 1: Load images and detect keypoints and descriptors using SIFT
img1 = cv2.imread('./test/test_image7.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./test/test_image8.jpg', cv2.IMREAD_GRAYSCALE)


sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Step 2: Match the descriptors using FLANN-based matcher
flann_index_kdtree = 1
index_params = dict(algorithm=flann_index_kdtree, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Get the matched keypoints from the good matches
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

# Step 3: Compute the Essential matrix
focal_length = 1.0  # Assuming a normalized focal length or use known camera parameters
camera_matrix = np.array([[focal_length, 0, img1.shape[1] / 2],
                          [0, focal_length, img1.shape[0] / 2],
                          [0, 0, 1]])  # Intrinsic camera matrix

essential_matrix, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Step 4: Recover relative camera pose (Rotation and Translation)
_, R, t, mask_pose = cv2.recoverPose(essential_matrix, pts1, pts2, camera_matrix)

# Apply the mask to get the inlier points
pts1_inliers = pts1[mask_pose.ravel() == 1]
pts2_inliers = pts2[mask_pose.ravel() == 1]

# Step 5: Triangulate points
P1 = np.eye(3, 4, dtype=np.float32)  # Camera 1 at the origin
P2 = np.hstack((R, t)).astype(np.float32)  # Camera 2 with the recovered pose

# Ensure the points are reshaped properly (2xN) and converted to float32
pts1_inliers = pts1_inliers.T.astype(np.float32)
pts2_inliers = pts2_inliers.T.astype(np.float32)

# Triangulate the 3D points
points_4d = cv2.triangulatePoints(P1, P2, pts1_inliers, pts2_inliers)

# Convert from homogeneous coordinates to 3D
points_3d = points_4d[:3] / points_4d[3]  # Normalize the homogeneous coordinates

print("Triangulated 3D points:\n", points_3d.T)  # 3D coordinates of the points

# The translation vector `t` is the relative position of the second camera
print("Camera 2 Position (relative to Camera 1):", t)
