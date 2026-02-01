import cv2
import os
import glob
from tqdm import tqdm

# ================= Configuration Area =================
# Current User ID to process
USER_ID = 6

# [Critical] Specify Path IDs to process
# Format: [PathID1, PathID2, ...]
# e.g.: Process only path 9 -> [9]
# e.g.: Process paths 6 and 9 -> [6, 9]
# To process all paths, fill in all IDs or leave empty [] (code will process all if empty)
TARGET_PATH_IDS = [5] 

# Raw video root directory
INPUT_ROOT = f"images/user{USER_ID}"

# Processed video output root directory
OUTPUT_ROOT = f"masked_images/user{USER_ID}"

# Right side mask ratio (0.25 = mask right 1/4)
MASK_RATIO_RIGHT = 0.21
MASK_COLOR = (0, 0, 0) # Black
# ===========================================

def process_single_video(input_path, output_path, mask_ratio):
    """Read video, mask right area, and save"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {input_path}")
        return False

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mask_x_start = int(width * (1.0 - mask_ratio))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    desc_text = f"{os.path.basename(os.path.dirname(input_path))}/{os.path.basename(input_path)}"
    
    with tqdm(total=total_frames, desc=desc_text, unit="fr") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Draw black box
            cv2.rectangle(frame, 
                          (mask_x_start, 0), 
                          (width, height), 
                          MASK_COLOR, 
                          thickness=-1)

            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()
    return True

def batch_process_masking():
    if not os.path.exists(INPUT_ROOT):
        print(f"❌ Input directory not found: {INPUT_ROOT}")
        return

    print(f"🚀 Start video masking process (User {USER_ID})...")
    
    # 1. Determine paths to process
    if TARGET_PATH_IDS:
        print(f"🎯 Target processing paths: {TARGET_PATH_IDS}")
        target_dirs = [str(pid) for pid in TARGET_PATH_IDS]
    else:
        print(f"🎯 No path specified, processing all subfolders in {INPUT_ROOT}.")
        target_dirs = [d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))]

    print(f"🛡️  Mask Ratio: {MASK_RATIO_RIGHT*100:.1f}% (Right)")
    print("-" * 30)

    count = 0
    
    # 2. Iterate through specified path folders
    for path_id_str in target_dirs:
        current_in_dir = os.path.join(INPUT_ROOT, path_id_str)
        
        # Check if fold exists
        if not os.path.exists(current_in_dir):
            print(f"⚠️ Path folder does not exist, skipping: {current_in_dir}")
            continue

        # Find videos (mp4, mov, avi)
        video_files = []
        for ext in ["*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI"]:
            video_files.extend(glob.glob(os.path.join(current_in_dir, ext)))
            
        if not video_files:
            print(f"   [Path {path_id_str}] No video found, skipping.")
            continue

        print(f"📂 [Path {path_id_str}] Found {len(video_files)} videos")

        for in_path in video_files:
            # Build output path (Maintain structure: masked_images/user6/9/xxx.mp4)
            relative_path = os.path.relpath(in_path, INPUT_ROOT)
            out_path = os.path.join(OUTPUT_ROOT, relative_path)
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            if process_single_video(in_path, out_path, MASK_RATIO_RIGHT):
                count += 1

    print("\n" + "="*30)
    print(f"✅ All done! Generated {count} processed videos.")
    print(f"📁 Output location: {OUTPUT_ROOT}")
    print("💡 Tip: Remember to change the input path in YOLO tracking code to this new directory.")


if __name__ == "__main__":
    batch_process_masking()