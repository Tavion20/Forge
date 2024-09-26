import cv2
import numpy as np
import open3d as o3d

# Step 1: Frame extraction from video
def extract_frames(video_path, frame_rate=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while cap.isOpened():
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % (fps // frame_rate) == 0:
            frames.append(frame)

    cap.release()
    return frames

# Step 2: Feature detection (using SIFT)
def detect_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def extract_matched_keypoints(kpts1, kpts2, matches):
    pts1 = np.float32([kpts1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kpts2[m.trainIdx].pt for m in matches])
    return pts1, pts2

def estimate_pose(kpts1, kpts2, K):
    # Find essential matrix
    E, mask = cv2.findEssentialMat(kpts1, kpts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Recover relative camera pose
    _, R, t, mask_pose = cv2.recoverPose(E, kpts1, kpts2, K)
    return R, t



# Step 5: Triangulation (to generate 3D points)
def triangulate_points(pts1, pts2, K, R, t):
    # Convert points to homogeneous coordinates by reshaping them to (2, N)
    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 2).T
    pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 2).T

    # Projection matrix for the first camera (assumed to be at the origin)
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # [I|0]

    # Projection matrix for the second camera
    P2 = np.hstack((R, t))  # [R|t]

    # Triangulate the points
    points_4D = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)

    # Convert from homogeneous coordinates (4D) to 3D
    points_3D = points_4D[:3] / points_4D[3]

    return points_3D.T  # Return in shape (N, 3)


# Step 6: Visualizing the 3D points using Open3D
def visualize_point_cloud(points_3D):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3D)
    o3d.visualization.draw_geometries([pcd])

def main():
    # Intrinsic camera matrix (K) -- Adjust for your camera
    K = np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1]])

    # Step 1: Extract frames from video
    video_path = './test_vid2.mp4'  # Replace with your video file path
    frames = extract_frames(video_path, frame_rate=2)

    # Step 2: Detect features in the first two frames
    keypoints1, descriptors1 = detect_features(frames[0])
    keypoints2, descriptors2 = detect_features(frames[1])

    # Step 3: Match features between the two frames
    matches = match_features(descriptors1, descriptors2)

    # Step 4: Extract matched keypoints based on the matches
    pts1, pts2 = extract_matched_keypoints(keypoints1, keypoints2, matches)

    # Step 5: Estimate the camera pose (R, t)
    R, t = estimate_pose(pts1, pts2, K)

    # Step 6: Triangulate points to get the 3D point cloud
    points_3D = triangulate_points(pts1, pts2, K, R, t)

    # Step 7: Visualize the 3D point cloud
    visualize_point_cloud(points_3D)

if __name__ == "__main__":
    main()

