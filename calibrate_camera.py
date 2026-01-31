#!/usr/bin/env python3
"""
Camera Calibration Script

Calibrates camera intrinsics from checkerboard images. The calibration is done
at the native capture resolution, then the intrinsics are transformed to match
the 640x480 center-cropped-and-scaled feed used by live_demo.py.

Foscam C2 feed pipeline:
  1. Camera captures at 1920x1080
  2. RTSP stream delivers 640x480 (center crop 1440x1080 then scale to 640x480)

Usage:
    python calibrate_camera.py [--pattern 9x6] [--images foscam-c2/]
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np


def find_checkerboard_corners(images, pattern_size, show=False):
    """Find checkerboard corners in a set of images.

    Args:
        images: List of image file paths.
        pattern_size: (cols, rows) inner corners of checkerboard.
        show: If True, display detected corners.

    Returns:
        obj_points: List of 3D points in real world space.
        img_points: List of 2D points in image plane.
        img_size: (width, height) of images.
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    obj_points = []
    img_points = []
    img_size = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"  Skipping (can't read): {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            # Refine corners to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners)
            print(f"  Found corners: {os.path.basename(fname)}")
        else:
            print(f"  No corners: {os.path.basename(fname)}")

    return obj_points, img_points, img_size


def calibrate(obj_points, img_points, img_size):
    """Run camera calibration.

    Returns:
        camera_matrix, dist_coeffs, rms_error
    """
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None)
    return camera_matrix, dist_coeffs, ret


def transform_intrinsics(camera_matrix, dist_coeffs, native_size, target_size):
    """Transform intrinsics from native resolution to target (center-crop + scale).

    Pipeline: native (1920x1080) -> center crop to 4:3 (1440x1080) -> scale to 640x480.

    Args:
        camera_matrix: 3x3 intrinsic matrix at native resolution.
        dist_coeffs: Distortion coefficients.
        native_size: (w, h) of calibration images.
        target_size: (w, h) of live feed.

    Returns:
        transformed_matrix, dist_coeffs (distortion coeffs unchanged by crop+scale).
    """
    native_w, native_h = native_size
    target_w, target_h = target_size

    # Step 1: Center crop to 4:3 aspect ratio
    crop_h = native_h  # Keep full height
    crop_w = int(crop_h * target_w / target_h)  # 1080 * 640/480 = 1440
    crop_x = (native_w - crop_w) // 2  # (1920 - 1440) / 2 = 240

    # Adjust principal point for crop
    K_cropped = camera_matrix.copy()
    K_cropped[0, 2] -= crop_x  # Shift cx

    # Step 2: Scale from crop size to target size
    scale_x = target_w / crop_w
    scale_y = target_h / crop_h

    K_target = K_cropped.copy()
    K_target[0, 0] *= scale_x  # fx
    K_target[1, 1] *= scale_y  # fy
    K_target[0, 2] *= scale_x  # cx
    K_target[1, 2] *= scale_y  # cy

    return K_target, dist_coeffs


def save_calibration(camera_matrix, dist_coeffs, rms, output_path):
    """Save calibration to JSON."""
    data = {
        '_comment': 'Camera intrinsics for 640x480 feed (center-crop + scale from 1920x1080)',
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.flatten().tolist(),
        'rms_error': float(rms),
        'frame_size': [640, 480]
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved calibration to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Camera calibration from checkerboard images')
    parser.add_argument('--pattern', default='9x6',
                        help='Checkerboard pattern as COLSxROWS inner corners (default: 9x6)')
    parser.add_argument('--images', default='foscam-c2/',
                        help='Directory containing checkerboard images')
    parser.add_argument('--output', default='calibration/camera.json',
                        help='Output calibration file')
    args = parser.parse_args()

    # Parse pattern size
    cols, rows = map(int, args.pattern.split('x'))
    pattern_size = (cols, rows)

    # Find images
    images = sorted(glob.glob(os.path.join(args.images, '*.png')) +
                    glob.glob(os.path.join(args.images, '*.jpg')) +
                    glob.glob(os.path.join(args.images, '*.PNG')))
    if not images:
        print(f"No images found in {args.images}")
        sys.exit(1)

    print(f"Calibrating from {len(images)} images (pattern: {cols}x{rows})")
    print("-" * 40)

    # Find corners
    obj_points, img_points, img_size = find_checkerboard_corners(images, pattern_size)
    if len(obj_points) < 3:
        print(f"Need at least 3 images with detected corners, got {len(obj_points)}")
        sys.exit(1)

    print(f"\nUsing {len(obj_points)}/{len(images)} images for calibration")
    print(f"Native image size: {img_size[0]}x{img_size[1]}")

    # Calibrate at native resolution
    camera_matrix, dist_coeffs, rms = calibrate(obj_points, img_points, img_size)
    print(f"RMS reprojection error: {rms:.3f}")
    print(f"Distortion coefficients: k1={dist_coeffs[0,0]:.3f}, k2={dist_coeffs[0,1]:.3f}, "
          f"p1={dist_coeffs[0,2]:.4f}, p2={dist_coeffs[0,3]:.4f}, k3={dist_coeffs[0,4]:.3f}")

    # Transform to 640x480 feed
    target_size = (640, 480)
    K_target, d_target = transform_intrinsics(
        camera_matrix, dist_coeffs, img_size, target_size)

    print(f"\nTransformed intrinsics for {target_size[0]}x{target_size[1]}:")
    print(f"  fx={K_target[0,0]:.1f}, fy={K_target[1,1]:.1f}")
    print(f"  cx={K_target[0,2]:.1f}, cy={K_target[1,2]:.1f}")

    # Save
    save_calibration(K_target, d_target, rms, args.output)


if __name__ == '__main__':
    main()
