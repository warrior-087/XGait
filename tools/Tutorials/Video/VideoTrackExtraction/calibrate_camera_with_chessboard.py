import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from datetime import datetime


def calibrate_camera_with_dots(images, pattern_size, dot_spacing, output_dir="dot_calibration_results"):
    """
    Calibrate camera using a dot calibration pattern, including subpixel optimization.
    :param images: List of calibration images
    :param pattern_size: Dot grid size (cols, rows)
    :param dot_spacing: Actual spacing between dots (in meters)
    :param output_dir: Directory to output results
    :return: Camera matrix, distortion coefficients, rotation vectors, translation vectors, reprojection error
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    calibration_report = os.path.join(output_dir, "calibration_report.txt")

    # Prepare calibration board 3D points (Z=0 plane)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * dot_spacing

    # Store 3D points and 2D points
    objpoints = []  # 3D points in world coordinate system
    imgpoints = []  # 2D points in image coordinate system
    all_centers = []  # Store all detected centers (for visualization)
    image_sizes = []  # Store image dimensions

    # Subpixel refinement parameters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Process each image
    detected_images = 0
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: Unable to read image {fname}")
            continue

        # Record image dimensions (should be consistent across all images)
        if not image_sizes:
            image_size = img.shape[:2]
            image_sizes.append(image_size)
        elif img.shape[:2] != image_size:
            print(f"Warning: Image {fname} size inconsistent ({img.shape[:2]} vs {image_size})")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find circle grid
        ret, centers = cv2.findCirclesGrid(
            gray, pattern_size,
            flags=cv2.CALIB_CB_SYMMETRIC_GRID
        )

        # Find chessboard corners
        # ret, centers = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            detected_images += 1

            # Convert center coordinates to float (required for subpixel refinement)
            centers = np.array(centers, dtype=np.float32)

            # Refine center positions - significantly improves accuracy
            centers_refined = cv2.cornerSubPix(gray, centers, (5, 5), (-1, -1), criteria)

            # Store points
            objpoints.append(objp)
            imgpoints.append(centers_refined)
            all_centers.append((centers, centers_refined))

            # Visualize detection results
            img_draw = img.copy()
            cv2.drawChessboardCorners(img_draw, pattern_size, centers_refined, ret)

            # cv2.imshow("img_draw", img_draw)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            # Add comparison before and after subpixel refinement
            for j in range(min(5, len(centers))):
                orig_point = tuple(centers[j].ravel().astype(int))
                refined_point = tuple(centers_refined[j].ravel().astype(int))
                cv2.circle(img_draw, orig_point, 8, (0, 0, 255), -1)  # Original point (Red)
                cv2.circle(img_draw, refined_point, 4, (0, 255, 0), -1)  # Refined point (Green)
                cv2.line(img_draw, orig_point, refined_point, (255, 0, 0), 1)  # Connecting line (Blue)
                cv2.putText(img_draw, f"{j}", tuple(np.array(refined_point) + [10, 10]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Save detection results
            img_name = os.path.basename(fname)
            cv2.imwrite(os.path.join(output_dir, f"detected_{img_name}"), img_draw)
            print(f"Detection successful {img_name} ({detected_images}/{len(images)})")
        else:
            print(f"No circle grid detected in image {fname}")

    # Check if there are enough images
    if detected_images < 5:
        raise ValueError(f"Insufficient circle grid images detected ({detected_images}), at least 5 required")

    print(f"\nStarting camera calibration using {detected_images} images...")

    # Execute camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # Calculate reprojection error
    mean_error = 0
    per_image_errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        per_image_errors.append(error)
        mean_error += error

    mean_error /= len(objpoints)

    # Calculate improvement from subpixel refinement
    avg_refinement = calculate_refinement_improvement(all_centers)

    # Save calibration results
    # Save as .npy files respectively for convenient subsequent loading
    np.save(os.path.join(output_dir, "camera_K.npy"), mtx)  # Save intrinsic matrix (corresponds to K)
    np.save(os.path.join(output_dir, "camera_dist.npy"), dist)  # Save distortion coefficients (corresponds to dist)

    # Keep npz just in case rotation/translation vectors are needed (optional)
    np.savez(os.path.join(output_dir, "camera_params.npz"),
             mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    print(f"Intrinsic parameters saved to: {os.path.join(output_dir, 'camera_K.npy')}")
    print(f"Distortion coefficients saved to: {os.path.join(output_dir, 'camera_dist.npy')}")

    # Generate calibration report
    generate_calibration_report(calibration_report, mtx, dist, mean_error,
                                per_image_errors, detected_images, avg_refinement)

    # Visualize calibration results
    visualize_calibration_results(mtx, dist, image_size, output_dir, per_image_errors, mean_error,
                                  objpoints, imgpoints, rvecs, tvecs, dot_spacing)

    return mtx, dist, rvecs, tvecs, mean_error


def calculate_refinement_improvement(all_centers):
    """Calculate improvement brought by subpixel refinement"""
    total_displacement = 0.0
    total_points = 0

    for centers, refined_centers in all_centers:
        displacement = np.linalg.norm(centers - refined_centers, axis=1)
        total_displacement += np.sum(displacement)
        total_points += displacement.size

    avg_displacement = total_displacement / total_points
    return avg_displacement


def generate_calibration_report(filename, mtx, dist, mean_error, per_image_errors, num_images, avg_refinement):
    """Generate detailed calibration report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, "w") as f:
        f.write("Dot Board Calibration Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated at: {timestamp}\n")
        f.write(f"Calibration images used: {num_images}\n")
        f.write(f"Average reprojection error: {mean_error:.4f} pixels\n")
        f.write(f"Average subpixel refinement displacement: {avg_refinement:.4f} pixels\n\n")

        f.write("Intrinsic Matrix (K):\n")
        f.write(np.array2string(mtx, precision=5, suppress_small=True))
        f.write("\n\nDistortion Coefficients (k1, k2, p1, p2, k3):\n")
        f.write(np.array2string(dist, precision=5, suppress_small=True))
        f.write("\n\n")

        f.write("Reprojection Errors per Image:\n")
        for i, error in enumerate(per_image_errors):
            f.write(f"  Image {i + 1}: {error:.4f} pixels\n")

        f.write("\nCalibration Analysis:\n")
        f.write(f"- Reprojection error < 0.3 pixels: {'Excellent' if mean_error < 0.3 else 'Good' if mean_error < 0.5 else 'Fair'}\n")
        f.write(f"- Subpixel refinement significantly improved accuracy (avg displacement: {avg_refinement:.2f} pixels)\n")
        f.write("- Dot board typically provides higher accuracy than chessboard\n")
        f.write("- Recommend capturing more images at different positions and angles to further improve accuracy\n")

    print(f"Calibration report saved to: {filename}")


def visualize_calibration_results(mtx, dist, image_size, output_dir, per_image_errors, mean_error,
                                  objpoints, imgpoints, rvecs, tvecs, dot_spacing):
    """Visualize calibration results"""
    # Create undistorted image example
    h, w = image_size
    sample_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw grid to visualize distortion correction effect
    grid_size = 10
    for i in range(0, w, w // grid_size):
        cv2.line(sample_image, (i, 0), (i, h), (0, 255, 0), 2)
    for i in range(0, h, h // grid_size):
        cv2.line(sample_image, (0, i), (w, i), (0, 255, 0), 2)

    # Add center marker
    cv2.circle(sample_image, (w // 2, h // 2), 15, (0, 0, 255), -1)

    # Distortion correction
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(sample_image, mtx, dist, None, newcameramtx)

    # Save visualization results
    cv2.imwrite(os.path.join(output_dir, "distortion_visualization.png"),
                np.hstack((sample_image, dst)))

    # Plot reprojection error graph
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(per_image_errors) + 1), per_image_errors, color='skyblue')
    plt.axhline(y=mean_error, color='r', linestyle='-', linewidth=2,
                label=f'Average error: {mean_error:.4f} pixel')

    # Mark best and worst images
    min_idx = np.argmin(per_image_errors)
    max_idx = np.argmax(per_image_errors)
    plt.plot(min_idx + 1, per_image_errors[min_idx], 'go', markersize=10, label='The best Image')
    plt.plot(max_idx + 1, per_image_errors[max_idx], 'ro', markersize=10, label='The worst image')

    plt.title('Distribution of reprojection error of dot calibration plate')
    plt.xlabel('Image Number')
    plt.ylabel('Reprojection error (pixels)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reprojection_errors.png"), dpi=300)
    plt.close()

    # Create 3D point projection visualization
    create_3d_projection_visualization(objpoints, imgpoints, mtx, dist, rvecs, tvecs,
                                       image_size, dot_spacing, output_dir)

    print("Visualization results saved")


def create_3d_projection_visualization(objpoints, imgpoints, mtx, dist, rvecs, tvecs,
                                       image_size, dot_spacing, output_dir):
    """Create 3D point projection visualization"""
    # Select an image for visualization
    if len(imgpoints) == 0:
        return

    idx = 0  # Use the first image
    img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    # Project 3D points
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3) * dot_spacing
    imgpts, _ = cv2.projectPoints(axis, rvecs[idx], tvecs[idx], mtx, dist)

    # Draw coordinate system
    origin = tuple(imgpoints[idx][0].ravel().astype(int))
    cv2.line(img, origin, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 3)  # X-axis (Red)
    cv2.line(img, origin, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3)  # Y-axis (Green)
    cv2.line(img, origin, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 3)  # Z-axis (Blue)

    # Draw all points
    for i, point in enumerate(imgpoints[idx]):
        color = (0, 255, 255)  # Yellow
        pos = tuple(point.ravel().astype(int))
        cv2.circle(img, pos, 4, color, -1)
        cv2.putText(img, str(i), (pos[0] + 10, pos[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

    # Save results
    cv2.imwrite(os.path.join(output_dir, "3d_projection.png"), img)


def undistort_image(image_path, mtx, dist, output_dir):
    """Undistort image using calibration parameters, maintaining original size"""
    img = cv2.imread(image_path)
    if img is None:
        return

    h, w = img.shape[:2]

    # Method 1: Direct correction involving no cropping (keeps original size but may have black borders)
    # dst = cv2.undistort(img, mtx, dist, None, None)

    # Method 2: Use optimal matrix and manually adjust size
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Save results
    img_name = os.path.basename(image_path)
    cv2.imwrite(os.path.join(output_dir, f"undistorted_{img_name}"), dst)
    return dst


if __name__ == "__main__":
    # ==================== Parameter Configuration ====================
    # Calibration Board Settings
    DOT_PATTERN_SIZE = (8, 11)  # (Cols, Rows) - Corner (Dot) Grid Size
    DOT_SPACING = 0.0215  # Actual Corner (Dot) Spacing (Unit: meters)

    # Image Path
    IMAGE_DIR = "images/user5/标定"
    OUTPUT_DIR = "dot_calibration_results/"

    # Get calibration images    images = glob.glob(os.path.join(IMAGE_DIR, "*.bmp"))
    images += glob.glob(os.path.join(IMAGE_DIR, "*.JPG"))

    # image_list = os.listdir(IMAGE_DIR)
    # images = [IMAGE_DIR + str(i) + ".png" for i in range(1, len(image_list)+1)]
    # images = [IMAGE_DIR + i for i in image_list]

    if not images:
        raise FileNotFoundError(f"在 {IMAGE_DIR} 目录中未找到标定图像")

    print(f"找到 {len(images)} 张标定图像")

    # ==================== 执行标定 ====================
    # 标准标定流程
    print("\n=== 标准标定流程 ===")
    mtx, dist, rvecs, tvecs, mean_error = calibrate_camera_with_dots(
        images, DOT_PATTERN_SIZE, DOT_SPACING, OUTPUT_DIR
    )

    # ==================== 显示结果 ====================
    print("\n相机标定完成!")
    print(f"平均重投影误差: {mean_error:.4f} 像素")
    print("\n相机内参矩阵:")
    print(mtx)
    print("\n畸变系数:")
    print(dist)

    # ==================== 测试畸变矫正 ====================
    if images:
        # 使用第一张图像测试畸变矫正
        test_image = images[0]
        undistorted_img = undistort_image(test_image, mtx, dist, OUTPUT_DIR)

        if undistorted_img is not None:
            # 确保图像尺寸一致
            original_img = cv2.imread(test_image)
            h1, w1 = original_img.shape[:2]
            h2, w2 = undistorted_img.shape[:2]

            # 如果高度不一致，调整较小图像的高度
            if h1 != h2:
                # 创建新背景（黑色）
                new_undistorted = np.zeros((h1, w2, 3), dtype=np.uint8)

                # 计算垂直偏移（将矫正图像居中放置）
                y_offset = (h1 - h2) // 2

                # 确保偏移有效
                if y_offset >= 0 and h2 <= h1:
                    new_undistorted[y_offset:y_offset + h2, 0:w2] = undistorted_img
                    undistorted_img = new_undistorted
                else:
                    print("警告：矫正图像高度大于原始图像，无法调整")
                    # 简单调整大小作为备选方案
                    undistorted_img = cv2.resize(undistorted_img, (w1, h1))

            # 显示原始和矫正后的图像
            combined = np.hstack((original_img, undistorted_img))

            # 调整显示大小
            scale = 0.5
            h, w = combined.shape[:2]
            combined_display = cv2.resize(combined, (int(w * scale), int(h * scale)))

            cv2.imshow("原始图像 vs 矫正后图像", combined_display)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            cv2.imwrite("result.png", combined_display)