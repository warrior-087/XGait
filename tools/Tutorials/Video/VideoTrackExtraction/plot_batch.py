import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ================= Configuration Area =================

# 1. [Critical] Specify list of User IDs to plot
# Format: ["ID1", "ID2", ...]
# TARGET_USERS = ["1", "2", "3", "4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"]
TARGET_USERS = ["5"]
# 2. Specify list of Path IDs to plot
# If you only want to plot paths 6 and 9, write [6, 9]
# If you want to plot all, write [1, 2, 3, 4, 5, 6, 7, 8, 9]
# TARGET_PATHS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
TARGET_PATHS = [5]
DATA_ROOT = "output_test"  # Data root directory

# ===========================================

# ===== Ground Truth Path Definition (for reference lines) =====
PATHS_GT = {
    1: [(1.0, 1.0), (5.5, 3.5)],
    2: [(1.0, 3.5), (5.5, 1.0)],
    3: [(1.0, 2.25), (5.5, 2.25)],
    4: [(3.25, 1.0), (3.25, 3.5)],
    # Paths 6-9 mapped to 1-4
    6: [(1.0, 1.0), (5.5, 3.5)],
    7: [(1.0, 3.5), (5.5, 1.0)],
    8: [(1.0, 2.25), (5.5, 2.25)],
    9: [(3.25, 1.0), (3.25, 3.5)],
}

# Reference points for Path 5 (Rectangular Loop)
PATH5_GT = [
    (1.0, 2.25),
    (3.25, 1.0),
    (5.5, 2.25),
    (3.25, 3.5),
    (1.0, 2.25),
]


def plot_single_path(user_id, path_id):
    """
    Plot single path and save
    """
    # 1. Construct folder path
    target_dir = os.path.join(DATA_ROOT, f"user{user_id}", str(path_id))

    # Check if folder exists
    if not os.path.isdir(target_dir):
        # Only print when not all paths exist to avoid spamming
        # print(f"[Skip] User {user_id} Path {path_id}: Folder not found")
        return

    # 2. Get all .npy files
    search_pattern = os.path.join(target_dir, "*.npy")
    npy_files = glob.glob(search_pattern)

    # Filter out calib_xy.npy
    npy_files = [f for f in npy_files if "calib" not in os.path.basename(f)]

    if not npy_files:
        print(f"[Skip] User {user_id} Path {path_id}: No .npy files")
        return

    print(f"User {user_id} | Path {path_id}: Found {len(npy_files)} files. Plotting...")

    # 3. Start plotting
    plt.figure(figsize=(10, 6), dpi=150)

    # --- Plot all samples ---
    count = 0
    for f_path in npy_files:
        try:
            data = np.load(f_path)

            # Simple check for data shape
            if data.ndim != 2 or data.shape[1] < 2:
                continue

            X = data[:, 0]
            Y = data[:, 1]

            # Plot trajectory (Blue thin line)
            plt.plot(X, Y, color='blue', alpha=0.15, linewidth=1)
            count += 1
        except Exception as e:
            print(f"  [Error] {os.path.basename(f_path)}: {e}")

    # --- Plot Ground Truth reference lines ---
    if path_id == 5:
        gt_arr = np.array(PATH5_GT)
        plt.plot(gt_arr[:, 0], gt_arr[:, 1], 'r--o', linewidth=2, label="Ground Truth", zorder=10)
    elif path_id in PATHS_GT:
        p0, p1 = PATHS_GT[path_id]
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r--o', linewidth=2, label="Ground Truth", zorder=10)
        # Mark Start and End
        plt.text(p0[0], p0[1], " Start", color='red', fontsize=12, fontweight='bold', verticalalignment='bottom')
        plt.text(p1[0], p1[1], " End", color='red', fontsize=12, fontweight='bold', verticalalignment='bottom')

    # --- Set chart style ---
    plt.title(f"User {user_id} - Path {path_id} - All Samples (N={count})")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")

    # Set display range (adjust according to your venue size)
    plt.xlim(0, 7.0)
    plt.ylim(0, 5.0)

    plt.axis("equal")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    # --- Save and Close ---
    # Save image to output/userX/, or save directly to current directory
    # Saved to the current script directory for easy viewing, with filename prefix
    save_name = f"plot_user{user_id}_path{path_id}.png"
    plt.savefig(save_name)
    # print(f"  -> Saved: {save_name}")

    plt.close()  # Release memory


def main():
    print(f"Start batch plotting...")
    print(f"Target Users: {TARGET_USERS}")
    print(f"Target Paths: {TARGET_PATHS}")
    print("-" * 30)

    # Nested loop: Iterate Users -> Iterate Paths
    for uid in TARGET_USERS:
        for pid in TARGET_PATHS:
            plot_single_path(uid, pid)

    print("-" * 30)
    print("All plotting complete.")


if __name__ == "__main__":
    main()