import cv2
import os
import glob
from tqdm import tqdm

# ================= Configuration Area =================
# Current User ID to process
USER_ID = 17

# Specify Path IDs to process
TARGET_PATH_IDS = [9] 

# Raw video root directory
INPUT_ROOT = f"images/user{USER_ID}"

# Processed video output root directory
OUTPUT_ROOT = f"masked_images/user{USER_ID}"

# [Critical] Left side mask ratio (0.0 ~ 1.0)
# 0.1 means masking the leftmost 10% of the screen
# 0.25 means masking the leftmost 25% of the screen
MASK_RATIO_LEFT = 0.15

# Mask color (BGR format)
MASK_COLOR = (0, 0, 0) # Black
# ===========================================

def process_single_video(input_path, output_path, mask_ratio):
    """Read video, mask [LEFT] area, and save"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {input_path}")
        return False

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # [Critical] Calculate end X coordinate for mask (Left masking)
    # e.g. Width 1920, Ratio 0.1 -> Mask range is x=0 to x=192
    mask_x_end = int(width * mask_ratio)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    desc_text = f"{os.path.basename(os.path.dirname(input_path))}/{os.path.basename(input_path)}"
    
    with tqdm(total=total_frames, desc=desc_text, unit="fr") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # [Critical] Draw black box - Mask left side
            cv2.rectangle(frame, 
                          (0, 0),           # Top-left (x=0, y=0)
                          (mask_x_end, height), # Bottom-right (x=end point, y=bottom)
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
    
    if TARGET_PATH_IDS:
        print(f"🎯 Target processing paths: {TARGET_PATH_IDS}")
        target_dirs = [str(pid) for pid in TARGET_PATH_IDS]
    else:
        print(f"🎯 No path specified, processing all subfolders in {INPUT_ROOT}.")
        target_dirs = [d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))]

    # Print prompt
    print(f"🛡️  Mask Direction: [LEFT]")
    print(f"🛡️  Mask Ratio: {MASK_RATIO_LEFT*100:.1f}%")
    print("-" * 30)

    count = 0
    for path_id_str in target_dirs:
        current_in_dir = os.path.join(INPUT_ROOT, path_id_str)
        if not os.path.exists(current_in_dir):
            print(f"⚠️ Path folder does not exist, skipping: {current_in_dir}")
            continue

        video_files = []
        for ext in ["*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI"]:
            video_files.extend(glob.glob(os.path.join(current_in_dir, ext)))
            
        if not video_files:
            print(f"   [Path {path_id_str}] No video found, skipping.")
            continue

        print(f"📂 [Path {path_id_str}] Found {len(video_files)} videos")

        for in_path in video_files:
            relative_path = os.path.relpath(in_path, INPUT_ROOT)
            out_path = os.path.join(OUTPUT_ROOT, relative_path)
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            if process_single_video(in_path, out_path, MASK_RATIO_LEFT):
                count += 1

    print("\n" + "="*30)
    print(f"✅ All done! Generated {count} processed videos.")
    print(f"📁 Output location: {OUTPUT_ROOT}")


if __name__ == "__main__":
    batch_process_masking()