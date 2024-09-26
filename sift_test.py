import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

def extract_frames(video_path, num_frames=100):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = total_frames // num_frames
    
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Change to RGB
    
    cap.release()
    return frames

def match_features(frames):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    all_kps = []
    all_descs = []
    
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        all_kps.append(kp)
        all_descs.append(desc)
    
    matches = []
    for i in range(len(frames) - 1):
        matches.append(bf.knnMatch(all_descs[i], all_descs[i+1], k=2))
    
    return all_kps, matches

def filter_matches(matches, ratio=0.75):
    good_matches = []
    for match in matches:
        good = []
        for m, n in match:
            if m.distance < ratio * n.distance:
                good.append(m)
        good_matches.append(good)
    return good_matches

def estimate_motion(kps1, kps2, matches, K):
    src_pts = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, K)
    
    return R, t

def create_point_cloud(frames, all_kps, good_matches, K):
    points_3d = []
    colors = []
    
    R_total = np.eye(3)
    t_total = np.zeros((3, 1))
    
    for i in range(len(frames) - 1):
        R, t = estimate_motion(all_kps[i], all_kps[i+1], good_matches[i], K)
        
        R_total = R @ R_total
        t_total = R @ t_total + t
        
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((R_total, t_total))
        
        points_4d = cv2.triangulatePoints(K @ P1, K @ P2,
                                          np.float32([all_kps[i][m.queryIdx].pt for m in good_matches[i]]).T,
                                          np.float32([all_kps[i+1][m.trainIdx].pt for m in good_matches[i]]).T)
        
        points_3d_homogeneous = cv2.convertPointsFromHomogeneous(points_4d.T)
        points_3d.extend(points_3d_homogeneous)
        
        for m in good_matches[i]:
            kp = all_kps[i][m.queryIdx]
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= y < frames[i].shape[0] and 0 <= x < frames[i].shape[1]:
                colors.append(frames[i][y, x])
    
    return np.array(points_3d).astype(np.float64), np.array(colors).astype(np.float64)

def visualize_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    o3d.visualization.draw_geometries([pcd])

def main(video_path):
    frames = extract_frames(video_path)
    all_kps, matches = match_features(frames)
    good_matches = filter_matches(matches)
    
    # Assuming a simple camera matrix (you may need to calibrate your camera for better results)
    K = np.array([[10, 0, frames[0].shape[1]/2],
                  [0, 10, frames[0].shape[0]/2],
                  [0, 0, 1]])
    
    points_3d, colors = create_point_cloud(frames, all_kps, good_matches, K)
    visualize_point_cloud(points_3d, colors)

if __name__ == "__main__":
    video_path = "./test_vid2.mp4"
    main(video_path)