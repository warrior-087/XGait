import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ================= Configuration Area =================
USER_ID = 11
PATH_ID = 5
DATA_DIR = f"output/user{USER_ID}/{PATH_ID}"

# Define standard start and end points (based on your ground truth)
POINT_A = np.array([1.0, 2.25])  # Bottom-left
POINT_B = np.array([1.0, 2.25])  # Top-right

# Tolerance threshold (meters)
# As long as the start/end points of the trajectory are within this range of the standard point, it is considered valid
# Recommended to set larger (e.g., 2.0 meters) to prevent accidental deletion of incomplete but generally normal trajectories
PROXIMITY_THRESH = 0.5 
# ===========================================

def check_start_and_end():
    if not os.path.exists(DATA_DIR):
        print(f"❌ Directory not found: {DATA_DIR}")
        return

    files = glob.glob(os.path.join(DATA_DIR, "*.npy"))
    files = [f for f in files if "calib" not in f]
    
    valid_forward = []   # Valid forward trip (A->B)
    valid_backward = []  # Valid backward trip (B->A)
    outliers = []        # Outliers (incomplete, wrong start, wrong end)

    print(f"Strictly checking start/end points of {len(files)} trajectories...\n")

    for f in files:
        try:
            traj = np.load(f)
            if len(traj) < 2: continue
            
            start_pt = traj[0]
            end_pt   = traj[-1]
            
            # --- Core Logic: Two-ended check ---
            
            # 1. Check if A -> B (Forward)
            # Start near A, End near B
            s_to_A = np.linalg.norm(start_pt - POINT_A)
            e_to_B = np.linalg.norm(end_pt - POINT_B)
            is_forward = (s_to_A < PROXIMITY_THRESH) and (e_to_B < PROXIMITY_THRESH)
            
            # 2. Check if B -> A (Backward)
            # Start near B, End near A
            s_to_B = np.linalg.norm(start_pt - POINT_B)
            e_to_A = np.linalg.norm(end_pt - POINT_A)
            is_backward = (s_to_B < PROXIMITY_THRESH) and (e_to_A < PROXIMITY_THRESH)
            
            if is_forward:
                valid_forward.append(traj)
            elif is_backward:
                valid_backward.append(traj)
            else:
                # Record specific error reason for your review
                reason = []
                if not (s_to_A < PROXIMITY_THRESH or s_to_B < PROXIMITY_THRESH):
                    reason.append(f"Start deviation (Dist to A:{s_to_A:.1f}m, Dist to B:{s_to_B:.1f}m)")
                if not (e_to_A < PROXIMITY_THRESH or e_to_B < PROXIMITY_THRESH):
                    reason.append(f"End deviation (Dist to A:{e_to_A:.1f}m, Dist to B:{e_to_B:.1f}m)")
                if not reason: # Indicates start/end are close to some point respectively, but not paired (e.g. A->A or B->B)
                     reason.append("Spinning in place (A->A or B->B)")
                
                print(f"🚩 Outlier file: {os.path.basename(f)}")
                print(f"   Reason: {', '.join(reason)}")
                outliers.append((f, traj))

        except Exception as e:
            print(f"Error reading {f}: {e}")

    # --- Statistics ---
    print("-" * 30)
    print(f"✅ Valid Forward (A->B): {len(valid_forward)}")
    print(f"✅ Valid Backward (B->A): {len(valid_backward)}")
    print(f"❌ Outliers: {len(outliers)}")

    if len(outliers) == 0:
        print("🎉 Perfect! All trajectories meet the start-to-end requirements.")
        return

    # --- Visualization ---
    plt.figure(figsize=(10, 6), dpi=120)
    plt.title(f"Strict Start-End Validation (Thresh={PROXIMITY_THRESH}m)")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Draw standard points
    plt.plot(POINT_A[0], POINT_A[1], 'g*', markersize=18, label='Point A', zorder=10)
    plt.plot(POINT_B[0], POINT_B[1], 'm*', markersize=18, label='Point B', zorder=10)

    # Draw valid forward (Blue)
    for t in valid_forward:
        plt.plot(t[:, 0], t[:, 1], 'b-', alpha=0.2, linewidth=1)
        
    # Draw valid backward (Orange)
    for t in valid_backward:
        plt.plot(t[:, 0], t[:, 1], 'orange', alpha=0.2, linewidth=1)

    # Draw outliers (Red, Bold)
    for name, t in outliers:
        plt.plot(t[:, 0], t[:, 1], 'r-', alpha=0.9, linewidth=2)
        plt.plot(t[0, 0], t[0, 1], 'rx') # Mark Start
        plt.plot(t[-1, 0], t[-1, 1], 'ro') # Mark End

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='b', label='Valid A->B'),
        Line2D([0], [0], color='orange', label='Valid B->A'),
        Line2D([0], [0], color='r', label='Outlier (Incomplete/Wrong)'),
    ]
    plt.legend(handles=legend_elements)
    plt.axis('equal')
    plt.show()

    # --- Deletion Prompt ---
    print("\n💡 Suggested Action:")
    print("Red trajectories do not meet start/end requirements.")
    user_input = input("Delete these outlier files? (y/n): ")
    if user_input.lower() == 'y':
        for f, _ in outliers:
            os.remove(f)
            print(f"Deleted: {os.path.basename(f)}")
        print("Cleanup complete!")

if __name__ == "__main__":
    check_start_and_end()