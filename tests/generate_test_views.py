#!/usr/bin/env python3
"""
Generate Warped Test Views

Takes real example images and applies known perspective transforms to simulate
camera movement. The warped images are used by test_geometry.py to verify that
the homography system handles camera displacement correctly.

Usage:
    python tests/generate_test_views.py
"""

import cv2
import numpy as np
import os
import sys

# Reference images (strong landmark detection)
REFERENCE_IMAGES = [
    'example/27-B2-UNMUTE.PNG',
    'example/09-B2-UNMUTE.PNG',
    'example/PP-S1-UNMUTE.PNG',
]

# Warp scenarios: name -> (description, transform function)
SCENARIOS = {
    'shift_right_30px': 'Camera bumped 30px right',
    'shift_down_20px': 'Camera sagged 20px down',
    'rotate_3deg': 'Camera twisted 3 degrees CW',
    'zoom_in_10pct': 'Camera moved 10% closer',
    'perspective_tilt': 'Camera angled slightly',
    'combined_shift_rot': 'Camera bumped + twisted',
}

OUTPUT_DIR = 'tests/warped_views'


def make_translation_matrix(dx, dy, w, h):
    """Create an affine translation matrix."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return M


def make_rotation_matrix(angle_deg, w, h):
    """Create a rotation matrix around frame center."""
    cx, cy = w / 2, h / 2
    return cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)


def make_zoom_matrix(scale, w, h):
    """Create a zoom (scale) matrix centered on frame."""
    cx, cy = w / 2, h / 2
    return cv2.getRotationMatrix2D((cx, cy), 0, scale)


def make_perspective_matrix(w, h):
    """Create a perspective tilt transform."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # Slight convergence toward top-right (simulates camera angled up-right)
    dst = np.float32([[10, 5], [w - 5, 10], [w, h], [0, h - 5]])
    return cv2.getPerspectiveTransform(src, dst)


def make_combined_matrix(w, h):
    """Create combined translation + rotation."""
    # Shift 15px right, 10px down, rotate 2 degrees
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), 2, 1.0)
    M[0, 2] += 15
    M[1, 2] += 10
    return M


def apply_warp(frame, scenario_name):
    """Apply a warp transform to a frame.

    Args:
        frame: BGR image.
        scenario_name: Name of the warp scenario.

    Returns:
        warped: Warped BGR image.
        transform: The transform matrix used (2x3 affine or 3x3 perspective).
    """
    h, w = frame.shape[:2]

    if scenario_name == 'shift_right_30px':
        M = make_translation_matrix(30, 0, w, h)
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE), M

    elif scenario_name == 'shift_down_20px':
        M = make_translation_matrix(0, 20, w, h)
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE), M

    elif scenario_name == 'rotate_3deg':
        M = make_rotation_matrix(-3, w, h)  # Negative = CW
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE), M

    elif scenario_name == 'zoom_in_10pct':
        M = make_zoom_matrix(1.1, w, h)
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE), M

    elif scenario_name == 'perspective_tilt':
        M = make_perspective_matrix(w, h)
        return cv2.warpPerspective(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE), M

    elif scenario_name == 'combined_shift_rot':
        M = make_combined_matrix(w, h)
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE), M

    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for img_path in REFERENCE_IMAGES:
        if not os.path.exists(img_path):
            print(f"Skipping (not found): {img_path}")
            continue

        frame = cv2.imread(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"Processing: {base_name}")

        for scenario_name in SCENARIOS:
            warped, transform = apply_warp(frame, scenario_name)
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}_{scenario_name}.png")
            cv2.imwrite(output_path, warped)
            print(f"  {scenario_name}: {output_path}")

    print(f"\nGenerated {len(REFERENCE_IMAGES) * len(SCENARIOS)} warped images in {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
