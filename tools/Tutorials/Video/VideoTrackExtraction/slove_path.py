#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
from apriltag import apriltag
from ultralytics import YOLO

# =========================
# Configuration Area
# =========================
USER_ID = 5  # Current User ID to process
IMAGES_ROOT = f"images/user{USER_ID}"        # Input directory
OUTPUT_ROOT = f"output_test/user{USER_ID}"        # Output directory

# Please confirm these two files exist in current directory
K_PATH = "dot_calibration_results/camera_K.npy"
DIST_PATH = "dot_calibration_results/camera_dist.npy"

TAG_FAMILY = "tagCustom48h12"
TAG_SIZE_M = 0.17
TAG_WORLD_ID = 6

YOLO_MODEL_NAME = "yolov8n-pose.pt"
PERSON_CLASS_ID = 0

# Final Coordinate System Transformation Constants
X_OFFSET_FINAL = 5.65 # lab:5.65 home:2.45
Y_FLIP_CENTER  = 3.5 # lab:3.5 home :1.0


Z_PLANE = 0.05
ALPHA   = 0.7       # Exponential Time Smoothing
SMOOTH_WINDOW = 21  # Savgol window
SMOOTH_POLY   = 2

# Path Mapping Relationship
PATH_ALIAS = {6: 1, 7: 2, 8: 3, 9: 4}

# ===== Real Start/End for 1~4 (Final Coordinate System) =====
# PATHS_1_4_FINAL = {
#     1: (np.array([1.0,  1.0,  0.0], np.float32), np.array([5.5,  3.5,  0.0], np.float32)),
#     2: (np.array([1.0,  3.5,  0.0], np.float32), np.array([5.5,  1.0,  0.0], np.float32)),
#     3: (np.array([1.0,  2.25, 0.0], np.float32), np.array([5.5,  2.25, 0.0], np.float32)),
#     4: (np.array([3.25, 1.0,  0.0], np.float32), np.array([3.25, 3.5,  0.0], np.float32)),
# }

PATHS_1_4_FINAL = {
    # Path 1: L-shape
    1: (np.array([-1.2, 0.6, 0.0], np.float32), np.array([1.4, 2.85, 0.0], np.float32)),
    # Path 2: Straight line
    2: (np.array([1.4, 0.6, 0.0], np.float32), np.array([1.4, 2.9, 0.0], np.float32)),
}
# ===== Path 5: One loop (Final Coordinate System Waypoints) =====
PATH5_WAYPOINTS_FINAL = [
    (1.0,  2.25),
    (3.25, 1.0),
    (5.5,  2.25),
    (3.25, 3.5),
    (1.0,  2.25),
]

def list_videos(folder):
    exts = (".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV", ".mkv", ".MKV")
    vids = [os.path.join(folder, n) for n in sorted(os.listdir(folder)) if n.endswith(exts)]
    return vids

def invert_final_coord(x_final, y_final):
    x_scaled = float(x_final) - float(X_OFFSET_FINAL)
    y_scaled = float(Y_FLIP_CENTER) - float(y_final)
    return x_scaled, y_scaled

def detect_apriltag(gray, detector, target_id=TAG_WORLD_ID):
    detections = detector.detect(gray)
    if not detections:
        return None, None, None
    for det in detections:
        if det["id"] == target_id:
            corners = det["lb-rb-rt-lt"].astype(np.float32)
            return det["id"], det["center"], corners
    return None, None, None

def estimate_tag_pose(K, dist, corners):
    half = TAG_SIZE_M / 2.0
    obj_points = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(obj_points, corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    R_ct, _ = cv2.Rodrigues(rvec)
    return rvec, tvec, R_ct

def pixel_to_plane(u, v, K, dist, R_ct, tvec, z_plane=Z_PLANE):
    pts = np.array([[[u, v]]], dtype=np.float32)
    undist = cv2.undistortPoints(pts, K, dist)
    x_norm, y_norm = undist[0, 0]
    d_cam = np.array([[x_norm, y_norm, 1.0]], dtype=np.float32).T

    R_tc = R_ct.T
    C_world = -R_tc @ tvec
    d_world = R_tc @ d_cam

    Cw = C_world.reshape(3)
    dw = d_world.reshape(3)
    if abs(dw[2]) < 1e-6:
        raise RuntimeError("ray parallel to plane")

    lam = (z_plane - Cw[2]) / dw[2]
    P = Cw + lam * dw
    return float(P[0]), float(P[1]), float(P[2])

def detect_person_foot(img_bgr, model, device):
    results = model(img_bgr, device=device, verbose=False)[0]
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("no detection")

    best_idx, best_conf = -1, -1.0
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == PERSON_CLASS_ID and conf > best_conf:
            best_conf = conf
            best_idx = i
    if best_idx < 0:
        raise RuntimeError("no person")

    x1, y1, x2, y2 = boxes[best_idx].xyxy[0].cpu().numpy().tolist()
    u_foot = float((x1 + x2) / 2.0)
    v_foot = float(y2)
    return u_foot, v_foot

def savgol_1d(y, window=11, poly=2):
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    if n < 3: return y.copy()

    window = int(window)
    if window % 2 == 0: window += 1
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < poly + 2:
        window = poly + 2
        if window % 2 == 0: window += 1
        window = min(window, n if n % 2 == 1 else n - 1)

    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float32)
    A = np.vander(x, N=poly + 1, increasing=True)
    ATA_inv = np.linalg.pinv(A.T @ A)
    coeffs = (ATA_inv @ A.T)[0]

    ypad = np.pad(y, (half, half), mode="reflect")
    out = np.convolve(ypad, coeffs[::-1], mode="valid")
    return out.astype(np.float32)

def smooth_traj_xy(traj_xy, window=11, poly=2):
    traj_xy = np.asarray(traj_xy, dtype=np.float32)
    xs = savgol_1d(traj_xy[:, 0], window=window, poly=poly)
    ys = savgol_1d(traj_xy[:, 1], window=window, poly=poly)
    return np.stack([xs, ys], axis=1).astype(np.float32)

def read_video_meas_xy(video_path, K, dist, model, detector, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open: {video_path}")

    traj = []
    prev = None

    while True:
        ret, frame = cap.read()
        if not ret: break
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = detect_apriltag(gray, detector, TAG_WORLD_ID)
            if res[0] is None:
                continue # Skip frames without tag
            _, _, corners = res
            _, tvec, R_ct = estimate_tag_pose(K, dist, corners)

            u, v = detect_person_foot(frame, model, device)
            x, y, _ = pixel_to_plane(u, v, K, dist, R_ct, tvec, z_plane=Z_PLANE)

            if prev is not None:
                x = ALPHA * x + (1.0 - ALPHA) * prev[0]
                y = ALPHA * y + (1.0 - ALPHA) * prev[1]
            prev = (x, y)

            traj.append([x, y])
        except Exception:
            pass

    cap.release()
    traj = np.asarray(traj, dtype=np.float32)
    if len(traj) < 20:
        raise RuntimeError(f"too few valid points: {len(traj)}")
    return traj

# ==========================================
# Core Fix: Vector Alignment Mode (No Scaling)
# ==========================================
def solve_calib_for_path_1_4_like(traj_meas_xy, path_id_final):
    p0f, p1f = PATHS_1_4_FINAL[path_id_final]
    target_start = invert_final_coord(p0f[0], p0f[1])
    target_end   = invert_final_coord(p1f[0], p1f[1])

    meas_start = traj_meas_xy[0]
    meas_end   = traj_meas_xy[-1]

    vec_meas = meas_end - meas_start
    vec_target = np.array(target_end) - np.array(target_start)

    # Calculate rotation only, no scaling
    angle_meas = np.arctan2(vec_meas[1], vec_meas[0])
    angle_target = np.arctan2(vec_target[1], vec_target[0])
    delta_angle = angle_target - angle_meas
    
    c, s = np.cos(delta_angle), np.sin(delta_angle)
    R = np.array([[c, -s], [s, c]])
    
    # Rotate measurement start (no scaling)
    meas_start_rotated = R @ meas_start
    # Calculate translation
    translation = np.array(target_start) - meas_start_rotated

    return {
        "ROTATION": delta_angle,
        "TRANS_X": translation[0],
        "TRANS_Y": translation[1],
        "MODE": "vector_align"
    }

def solve_calib_for_path5(traj_meas_xy):
    targets_scaled = [invert_final_coord(x, y) for (x, y) in PATH5_WAYPOINTS_FINAL]
    targets_scaled = np.asarray(targets_scaled, dtype=np.float32)
    traj = np.asarray(traj_meas_xy, dtype=np.float32)

    meas_pts = []
    for tx, ty in targets_scaled:
        # Find nearest point
        d2 = (traj[:, 0] - tx)**2 + (traj[:, 1] - ty)**2
        idx = int(np.argmin(d2))
        meas_pts.append(traj[idx])
    meas_pts = np.asarray(meas_pts, dtype=np.float32)

    # Calculate shift only, no scaling
    # Simple alignment: Target mean - Measurement mean
    tx = np.mean(targets_scaled[:, 0] - meas_pts[:, 0])
    ty = np.mean(targets_scaled[:, 1] - meas_pts[:, 1])

    return {"SHIFT_X": tx, "SHIFT_Y": ty,
            "PATH_ID": 5, "MODE": "path5_loop_shift_only"}

def apply_calib_and_save(traj_meas_xy, calib, out_npy_path):
    traj = np.asarray(traj_meas_xy, dtype=np.float32).copy()

    if calib.get("MODE") == "vector_align":
        theta = calib["ROTATION"]
        tx, ty = calib["TRANS_X"], calib["TRANS_Y"]
        
        c, s_theta = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s_theta], [s_theta, c]])
        
        # Rotation and translation only
        traj_rotated = (R @ traj.T).T 
        traj[:, 0] = traj_rotated[:, 0] + tx
        traj[:, 1] = traj_rotated[:, 1] + ty
    else:
        # Path 5 mode, shift only
        tx, ty = calib["SHIFT_X"], calib["SHIFT_Y"]
        traj[:, 0] = traj[:, 0] + tx
        traj[:, 1] = traj[:, 1] + ty

    traj[:, 0] = traj[:, 0] + X_OFFSET_FINAL
    traj[:, 1] = -(traj[:, 1] - Y_FLIP_CENTER)

    traj_s = smooth_traj_xy(traj, window=SMOOTH_WINDOW, poly=SMOOTH_POLY)

    os.makedirs(os.path.dirname(out_npy_path), exist_ok=True)
    np.save(out_npy_path, traj_s)
    return traj_s

def load_camera_params_from_npy():
    if not os.path.exists(K_PATH):
        print(f"ERROR: K file not found: {K_PATH}")
        return None, None
    K = np.load(K_PATH)
    dist = np.load(DIST_PATH)
    print("Loaded camera K:\n", K)
    print("Loaded dist:", dist)
    return K, dist

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    K, dist = load_camera_params_from_npy()
    if K is None: return

    model = YOLO(YOLO_MODEL_NAME)
    detector = apriltag(TAG_FAMILY)

    # 1. Force clean old cache (Optional but recommended)
    # import shutil
    # if os.path.exists(OUTPUT_ROOT):
    #     print("Cleaning old output...")
    #     shutil.rmtree(OUTPUT_ROOT)

    for path_id in range(5,6):
        in_dir = os.path.join(IMAGES_ROOT, str(path_id))
        if not os.path.isdir(in_dir):
            print(f"[SKIP] missing folder: {in_dir}")
            continue

        videos = list_videos(in_dir)
        if not videos:
            print(f"[SKIP] no video in: {in_dir}")
            continue

        out_dir = os.path.join(OUTPUT_ROOT, str(path_id))
        os.makedirs(out_dir, exist_ok=True)

        # 2. Calibration
        calib_path = os.path.join(out_dir, "calib_xy.npy")
        calib = None

        if os.path.exists(calib_path):
            print(f"[INFO] Reuse existing calib: {calib_path}")
            calib = np.load(calib_path, allow_pickle=True).item()
        else:
            calib_video = videos[0]
            print(f"Calculating Calib for Path {path_id} using {calib_video}")
            try:
                traj_meas_calib = read_video_meas_xy(calib_video, K, dist, model, detector, device)
                if path_id == 5:
                    calib = solve_calib_for_path5(traj_meas_calib)
                else:
                    base_id = PATH_ALIAS.get(path_id, path_id)
                    calib = solve_calib_for_path_1_4_like(traj_meas_calib, base_id)
                np.save(calib_path, calib)
            except Exception as e:
                print(f"[ERROR] Calib failed for path {path_id}: {e}")
                continue

        # 3. Batch processing
        for video in videos:
            try:
                base = os.path.splitext(os.path.basename(video))[0]
                out_npy = os.path.join(out_dir, base + ".npy")
                
                print(f"Processing: {base} ...")
                traj_meas = read_video_meas_xy(video, K, dist, model, detector, device)
                traj_final = apply_calib_and_save(traj_meas, calib, out_npy)
            except Exception as e:
                print(f"[WARN] Failed to process {video}: {e}")

    print("\nALL DONE.")

if __name__ == "__main__":
    main()