# AprilTag + YOLOv8 Pedestrian Trajectory Extraction System

This is an automated trajectory extraction system based on computer vision. It combines **AprilTag** (for camera pose estimation and coordinate system calibration) and **YOLOv8-Pose** (for human keypoint detection) to extract the real-world 2D coordinates $(x, y)$ of pedestrians from video footage.

## Dependencies

Ensure the following Python libraries are installed:
```bash
pip install opencv-python numpy matplotlib scipy torch ultralytics apriltag
```

## Directory Structure

```text
Project_Root/
│
├── images/                     # Raw video input directory
│   └── user{ID}/               # Classified by User ID
│       ├── 1/                  # Classified by Path ID (e.g., Path 1)
│       │   ├── video1.mp4
│       │   └── ...
│       ├── 5/                  # Path 5
│       └── calibration/        # Directory for calibration board images
│
├── masked_images/              # (Optional) Directory for masked/processed videos
│
├── output/                     # Result output directory
│   └── user{ID}/
│       ├── 1/
│       │   ├── video1.npy      # Extracted trajectory coordinate files
│       │   └── ...
│
├── dot_calibration_results/    # Camera calibration parameters (K, dist)
│   ├── camera_K.npy
│   └── camera_dist.npy
│
├── *.py                        # All Python scripts
└── README.md
```

## Core Scripts

### 1. Camera Calibration
*   **File**: `calibrate_camera_with_chessboard.py`
*   **Function**: Calculates the camera's intrinsic matrix (K) and distortion coefficients (dist) using images of a dot calibration board.
*   **Input**: `.bmp` or `.jpg` images in `images/user{ID}/calibration/`.
*   **Output**: Generates `camera_K.npy` and `camera_dist.npy` in `dot_calibration_results/`.
*   **Note**: **Must run this first** to generate calibration files before running the core extraction code.

### 2. Video Preprocessing (Masking)
*   **File**: `mask_video_left.py` / `mask_video_right.py`
*   **Function**: Ensures only the target subject is visible in the video. If there are other subjects or distractions, use this to mask them out.
*   **Config**: Modify `USER_ID` and `MASK_RATIO_LEFT` (masking ratio).
*   **Output**: Processed videos are saved in the `masked_images/` directory.

### 3. Core Trajectory Extraction
*   **File**: `solve_path.py`
*   **Function**: The core script of the system. It reads the video, performs PnP solving via AprilTags to get raw physical coordinates, and then uses YOLOv8 to detect the person. It automatically calculates calibration parameters based on path types (linear or loop) to align irregular raw trajectories to the standard coordinate system.
*   **Config**: 
    *   `USER_ID`: Current User ID being processed.
    *   `X_OFFSET_FINAL` / `Y_FLIP_CENTER`: **Critical**. Physical offsets for the final output coordinate system (varies between Lab and Home environments; verify with ground measurements).
    *   `PATHS_1_4_FINAL`: Defines the standard start and end coordinates for paths.

### 4. Batch Plotting
*   **File**: `plot_batch.py`
*   **Function**: Reads all `.npy` trajectory files in `output/` and plots them on a single graph, along with Ground Truth reference lines.
*   **Use Case**: Quickly check the overall quality and consistency of a batch of data.

### 5. Outlier Detection
*   **File**: `detect_outliers.py`
*   **Function**: Automatically filters out "bad data" (e.g., incomplete walks or severe start/end deviations) based on a proximity threshold (`PROXIMITY_THRESH`).
*   **Feature**: Provides a visual interface and supports one-click deletion of outlier files.

### 6. Debugging Tool
*   **File**: `debug_video.py`
*   **Function**: Diagnose individual videos frame-by-frame.
*   **Output**: Generates a video marking detected AprilTags, human skeletons, foot coordinates, and any jump errors.
*   **Use Case**: Use this to find the cause when a specific video fails to extract (e.g., Tag occlusion, YOLO misdetection).

### 7. File Management
*   **Files**: `count.py` / `number_vedio.py`
*   **`count.py`**: Counts videos and generated npy files in each folder to check for missing data.
*   **`number_vedio.py`**: Renames generated files sequentially (e.g., `user5-5-1.npy`) for easier management.

## Workflow

1.  **Prepare Data**: Place recorded videos in `images/userX/path_id/` and calibration images in `images/userX/calibration/`.
2.  **Calibration**: Run `calibrate_camera_with_chessboard.py` to ensure `.npy` parameter files are generated.
3.  **Masking (Optional)**: If the environment is cluttered, run `mask_video_left.py` to clean up video edges.
4.  **Extract Trajectories**: Modify `USER_ID` in `solve_path.py`. Run the script. The program automatically finds the first usable video in the path as a "calibration baseline," calculates translation/rotation parameters, and processes the remaining videos in batch.
5.  **Quality Check**: Run `plot_batch.py` to verify trajectory shapes. Run `detect_outliers.py` to remove invalid data.
6.  **Rename**: Run `number_vedio.py` to standardize filenames.

> **Note**: If large deviations occur in the extracted path, consider cropping or masking the video to ensure there is **one and only one** person in the frame at all times.


