import os
import glob
from collections import Counter

# ================= Configuration Area =================
# Change this to your root directory path, e.g. "images/user5"
# TARGET_FOLDER = "images/user101/1/"
TARGET_FOLDER = "output_converted/user112/"

# ===========================================

def count_actual_mp4_files():
    if not os.path.exists(TARGET_FOLDER):
        print(f"Error: Path not found '{TARGET_FOLDER}'")
        return

    print(f"Counting in folder: {TARGET_FOLDER} ...\n")

    # 1. Recursively find all .MP4 and .mp4 files
    # recursive=True will automatically traverse all subfolders (1, 2, ... 9 etc.)
    pattern_upper = os.path.join(TARGET_FOLDER, "**", "*.npy")
    pattern_lower = os.path.join(TARGET_FOLDER, "**", "*.MP4")

    files_upper = glob.glob(pattern_upper, recursive=True)
    files_lower = glob.glob(pattern_lower, recursive=True)

    # Merge and deduplicate (prevent duplicates due to case-insensitivity)
    all_files = list(set(files_upper + files_lower))
    total_count = len(all_files)

    # 2. Group count by folder
    # Get parent directory path for each file
    parent_dirs = [os.path.dirname(f) for f in all_files]
    # Use Counter to automatically count occurrences of each folder
    folder_counts = Counter(parent_dirs)

    # 3. Print detailed report
    print(f"{'Subfolder Name':<30} | {'Actual File Count'}")
    print("-" * 50)

    # Sort folder paths for easier viewing (user5/1, user5/2...)
    for folder_path in sorted(folder_counts.keys()):
        count = folder_counts[folder_path]
        # Show only relative paths for cleaner look
        rel_path = os.path.relpath(folder_path, os.path.dirname(TARGET_FOLDER))
        print(f"{rel_path:<30} | {count}")

    print("-" * 50)
    print(f" Total File Count: {total_count}")


if __name__ == "__main__":
    count_actual_mp4_files()