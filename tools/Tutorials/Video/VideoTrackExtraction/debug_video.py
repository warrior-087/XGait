import os
import cv2
import numpy as np
import torch
from apriltag import apriltag
from ultralytics import YOLO

# ================= Configuration Area =================
# [Crucial] Enter the path of the failed video detection
TARGET_VIDEO_PATH = "masked_images/user6/4/user6-4-1.mp4" 

# Keep other configurations unchanged
K_PATH = "dot_calibration_results/camera_K.npy"
DIST_PATH = "dot_calibration_results/camera_dist.npy"
TAG_FAMILY = "tagCustom48h12"
TAG_SIZE_M = 0.17
TAG_WORLD_ID = 6
YOLO_MODEL_NAME = "yolov8n-pose.pt"
Z_PLANE = 0.05
MIRROR_X_AXIS = True 
# ===========================================

# --- Copy necessary helper functions (maintain logic consistency) ---
def detect_apriltag(gray, detector, target_id=TAG_WORLD_ID):
    detections = detector.detect(gray)
    if not detections: return None, None, None
    for det in detections:
        if det["id"] == target_id:
            return det["id"], det["center"], det["lb-rb-rt-lt"].astype(np.float32)
    return None, None, None

def estimate_tag_pose(K, dist, corners):
    half = TAG_SIZE_M / 2.0
    obj_points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj_points, corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok: raise RuntimeError("solvePnP failed")
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
    Cw, dw = C_world.reshape(3), d_world.reshape(3)
    if abs(dw[2]) < 1e-6: raise RuntimeError("ray parallel")
    lam = (z_plane - Cw[2]) / dw[2]
    P = Cw + lam * dw
    return float(P[0]), float(P[1]), float(P[2])

def detect_person_foot_debug(img_bgr, model, device):
    """Pose detection with debug info"""
    results = model(img_bgr, device=device, verbose=False)[0]
    if results.keypoints is None or len(results.keypoints) == 0:
        return None, None, "No Person"
    
    best_idx = np.argmax(results.boxes.conf.cpu().numpy())
    kpts = results.keypoints.xy[best_idx].cpu().numpy()
    confs = results.keypoints.conf[best_idx].cpu().numpy()
    
    # Check foot confidence
    if confs[15] > 0.3 and confs[16] > 0.3:
        u, v = (kpts[15] + kpts[16]) / 2.0
        status = "Both Feet"
    elif confs[15] > 0.3:
        u, v = kpts[15]
        status = "Left Foot Only"
    elif confs[16] > 0.3:
        u, v = kpts[16]
        status = "Right Foot Only"
    else:
        # If foot confidence is low, fallback to BBox
        box = results.boxes.xyxy[best_idx].cpu().numpy()
        u, v = (box[0] + box[2])/2.0, box[3]
        status = "BBox fallback (Feet invisible)"
        
    return u, v, status

# ================= Core Diagnosis Function =================
def run_diagnosis():
    if not os.path.exists(TARGET_VIDEO_PATH):
        print(f"File not found: {TARGET_VIDEO_PATH}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Start diagnosis: {TARGET_VIDEO_PATH}")
    print(f"   Device: {device}")

    # Load model
    K = np.load(K_PATH)
    dist = np.load(DIST_PATH)
    model = YOLO(YOLO_MODEL_NAME)
    detector = apriltag(TAG_FAMILY)

    cap = cv2.VideoCapture(TARGET_VIDEO_PATH)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create output video
    out_path = "debug_output.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    prev_pos = None
    frame_idx = 0
    valid_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        debug_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        error_msg = []
        is_valid_frame = True

        # ---------------- STEP 1: Tag Detection ----------------
        tid, center, corners = detect_apriltag(gray, detector, TAG_WORLD_ID)
        
        if tid is None:
            is_valid_frame = False
            error_msg.append("NO TAG")
            cv2.putText(debug_frame, "NO TAG DETECTED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        else:
            # Draw Tag
            corners_int = corners.astype(int)
            for i in range(4):
                cv2.line(debug_frame, tuple(corners_int[i]), tuple(corners_int[(i+1)%4]), (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Tag {tid}", (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ---------------- STEP 2: Person Detection ----------------
        u, v = None, None
        if is_valid_frame: # Detect person only if Tag exists
            try:
                # Get calibration matrix
                _, tvec, R_ct = estimate_tag_pose(K, dist, corners)
                
                # Detect person
                u, v, status = detect_person_foot_debug(frame, model, device)
                
                if u is None:
                    is_valid_frame = False
                    error_msg.append("NO PERSON")
                    cv2.putText(debug_frame, "NO PERSON", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                else:
                    # Draw foot points
                    cv2.circle(debug_frame, (int(u), int(v)), 10, (255, 0, 0), -1)
                    cv2.putText(debug_frame, status, (int(u)+15, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
                    
                    # Transform coordinates
                    x, y, _ = pixel_to_plane(u, v, K, dist, R_ct, tvec, z_plane=Z_PLANE)
                    if MIRROR_X_AXIS: x = -x
                    
                    # ---------------- STEP 3: Jump Filtering ----------------
                    if prev_pos is not None:
                        dist_jump = np.linalg.norm(np.array([x,y]) - np.array(prev_pos))
                        if dist_jump > 0.5:
                            is_valid_frame = False
                            error_msg.append(f"JUMP {dist_jump:.2f}m")
                            cv2.putText(debug_frame, f"JUMP ERROR: {dist_jump:.2f}m", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                            # Draw previous frame position contrast
                            # (Cannot draw pixel point from previous frame here because view changed, but text on UI is sufficient)
                    
                    if is_valid_frame:
                        prev_pos = (x, y)
                        valid_count += 1
                        info_text = f"X: {x:.2f}, Y: {y:.2f}"
                        cv2.putText(debug_frame, info_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            except Exception as e:
                print(f"Frame {frame_idx} Calc Error: {e}")
                is_valid_frame = False

        # Print console log
        status_sym = "" if is_valid_frame else ""
        msg_str = " | ".join(error_msg) if error_msg else "OK"
        print(f"Frame {frame_idx}: {status_sym} -> {msg_str}")

        writer.write(debug_frame)

    cap.release()
    writer.release()
    print("-" * 30)
    print(f"Diagnosis complete!")
    print(f"Total frames: {frame_idx}, Valid frames: {valid_count}")
    print(f"Please open the generated video file for details: {out_path}")

if __name__ == "__main__":
    run_diagnosis()