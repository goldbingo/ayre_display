#!/usr/bin/env python3
"""Generate distorted variants for all example/ images.

Simulates camera angle/position changes via perspective (homography) and
affine (rotation, zoom, shift, combined) transforms.

Output: distorted/<basename>_<variant>.png
"""

import cv2
import numpy as np
import os
import glob

EXAMPLE_DIR = 'example'
OUTPUT_DIR = 'distorted'


# ---------------------------------------------------------------------------
# Perspective variants (use warpPerspective)
# ---------------------------------------------------------------------------

def perspective_left(w, h):
    """Camera panned left — left side closer, right side farther."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, 8], [w - 12, 0], [w - 12, h], [0, h - 8]])
    return cv2.getPerspectiveTransform(src, dst)


def perspective_right(w, h):
    """Camera panned right — right side closer, left side farther."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[12, 0], [w, 8], [w, h - 8], [12, h]])
    return cv2.getPerspectiveTransform(src, dst)


def perspective_mild(w, h):
    """Subtle upward tilt — top edge slightly narrower."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[4, 3], [w - 4, 3], [w, h], [0, h]])
    return cv2.getPerspectiveTransform(src, dst)


def perspective_shift_left(w, h):
    """Camera bumped left + tilted — pan left with 10px shift."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[-10, 6], [w - 20, 0], [w - 20, h], [-10, h - 6]])
    return cv2.getPerspectiveTransform(src, dst)


def perspective_shift_right(w, h):
    """Camera bumped right + tilted — pan right with 10px shift."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[20, 0], [w + 10, 6], [w + 10, h - 6], [20, h]])
    return cv2.getPerspectiveTransform(src, dst)


def perspective_steep(w, h):
    """Steeper downward angle — bottom compressed more."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, 0], [w, 0], [w - 10, h - 6], [10, h - 6]])
    return cv2.getPerspectiveTransform(src, dst)


def perspective_tilt(w, h):
    """Combined tilt — top-left pulled in, bottom-right pulled in."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[5, 5], [w, 0], [w - 5, h - 5], [0, h]])
    return cv2.getPerspectiveTransform(src, dst)


def perspective_rot3(w, h):
    """Perspective with slight 3-degree rotation component."""
    cx, cy = w / 2, h / 2
    a = np.radians(3)
    cos_a, sin_a = np.cos(a), np.sin(a)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts = []
    for x, y in src:
        rx = cos_a * (x - cx) - sin_a * (y - cy) + cx
        ry = sin_a * (x - cx) + cos_a * (y - cy) + cy
        pts.append([rx, ry])
    dst = np.float32(pts)
    # Add slight perspective on top
    dst[0] += [3, 2]
    dst[3] += [-3, -2]
    return cv2.getPerspectiveTransform(src, dst)


def perspective_down_tilt(w, h):
    """Camera tilted downward — bottom edge narrower, top wider."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, 0], [w, 0], [w - 6, h - 4], [6, h - 4]])
    return cv2.getPerspectiveTransform(src, dst)


def perspective_left_strong(w, h):
    """Strong left pan — more extreme than perspective_left."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, 15], [w - 20, 0], [w - 20, h], [0, h - 15]])
    return cv2.getPerspectiveTransform(src, dst)


def perspective_right_strong(w, h):
    """Strong right pan — more extreme than perspective_right."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[20, 0], [w, 15], [w, h - 15], [20, h]])
    return cv2.getPerspectiveTransform(src, dst)


def perspective_diag_tl(w, h):
    """Diagonal bump — top-left corner pushed in."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[10, 8], [w, 0], [w, h], [0, h]])
    return cv2.getPerspectiveTransform(src, dst)


def perspective_diag_br(w, h):
    """Diagonal bump — bottom-right corner pushed in."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, 0], [w, 0], [w - 10, h - 8], [0, h]])
    return cv2.getPerspectiveTransform(src, dst)


def perspective_twist_cw(w, h):
    """Slight clockwise twist — top-right in, bottom-left in."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, 0], [w - 6, 8], [w, h], [6, h - 8]])
    return cv2.getPerspectiveTransform(src, dst)


def perspective_twist_ccw(w, h):
    """Slight counter-clockwise twist — top-left in, bottom-right in."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[6, 8], [w, 0], [w - 6, h - 8], [0, h]])
    return cv2.getPerspectiveTransform(src, dst)


def perspective_up_shift_left(w, h):
    """Upward tilt + shift left — combined vertical and horizontal bump."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[-8, 5], [w - 12, 5], [w - 4, h], [-4, h]])
    return cv2.getPerspectiveTransform(src, dst)


def perspective_down_shift_right(w, h):
    """Downward tilt + shift right — combined vertical and horizontal bump."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[12, 0], [w + 4, 0], [w - 4, h - 5], [8, h - 5]])
    return cv2.getPerspectiveTransform(src, dst)


# ---------------------------------------------------------------------------
# Affine variants (use warpAffine) — return 2x3 matrices
# ---------------------------------------------------------------------------

def _rotate_matrix(w, h, angle):
    """Rotation around image center."""
    return cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)


def _zoom_matrix(w, h, scale):
    """Zoom (scale) around image center."""
    return cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)


def _shift_matrix(dx, dy):
    """Pure translation."""
    return np.float32([[1, 0, dx], [0, 1, dy]])


def _rot_zoom_matrix(w, h, angle, scale):
    """Rotation + zoom around image center."""
    return cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)


def rotate_3deg(w, h):
    return _rotate_matrix(w, h, 3)

def rotate_5deg(w, h):
    return _rotate_matrix(w, h, 5)

def rotate_7deg(w, h):
    return _rotate_matrix(w, h, 7)

def rotate_neg4deg(w, h):
    return _rotate_matrix(w, h, -4)

def zoom_in_10pct(w, h):
    return _zoom_matrix(w, h, 1.10)

def zoom_in_15pct(w, h):
    return _zoom_matrix(w, h, 1.15)

def zoom_out_10pct(w, h):
    return _zoom_matrix(w, h, 0.90)

def zoom_out_15pct(w, h):
    return _zoom_matrix(w, h, 0.85)

def shift_right_30px(w, h):
    return _shift_matrix(30, 0)

def shift_down_20px(w, h):
    return _shift_matrix(0, 20)

def shift_up_30px(w, h):
    return _shift_matrix(0, -30)

def shift_diag_30px(w, h):
    return _shift_matrix(20, 20)

def rot3_zoom_in_10(w, h):
    return _rot_zoom_matrix(w, h, 3, 1.10)

def rot5_zoom_out_10(w, h):
    return _rot_zoom_matrix(w, h, 5, 0.90)

def shift20_zoom_in_10(w, h):
    """Shift 20px right + zoom in 10%."""
    M = _zoom_matrix(w, h, 1.10)
    M[0, 2] += 20
    return M

def shift25_rot4_zoom8(w, h):
    """Shift 25px + rotate 4° + zoom 8%."""
    M = _rot_zoom_matrix(w, h, 4, 1.08)
    M[0, 2] += 25
    return M

def shiftn15_rotn3_zoomout8(w, h):
    """Shift -15px + rotate -3° + zoom out 8%."""
    M = _rot_zoom_matrix(w, h, -3, 0.92)
    M[0, 2] -= 15
    return M

def combined_shift_rot(w, h):
    """Combined shift + rotation: 15px right, 10px down, 2° rotation."""
    M = _rotate_matrix(w, h, 2)
    M[0, 2] += 15
    M[1, 2] += 10
    return M


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

# Perspective variants (3x3 matrix, use warpPerspective)
PERSPECTIVE_VARIANTS = {
    'perspective_left': perspective_left,
    'perspective_right': perspective_right,
    'perspective_mild': perspective_mild,
    'perspective_shift_left': perspective_shift_left,
    'perspective_shift_right': perspective_shift_right,
    'perspective_steep': perspective_steep,
    'perspective_tilt': perspective_tilt,
    'perspective_rot3': perspective_rot3,
    'perspective_down_tilt': perspective_down_tilt,
    'perspective_left_strong': perspective_left_strong,
    'perspective_right_strong': perspective_right_strong,
    'perspective_diag_tl': perspective_diag_tl,
    'perspective_diag_br': perspective_diag_br,
    'perspective_twist_cw': perspective_twist_cw,
    'perspective_twist_ccw': perspective_twist_ccw,
    'perspective_up_shift_left': perspective_up_shift_left,
    'perspective_down_shift_right': perspective_down_shift_right,
}

# Affine variants (2x3 matrix, use warpAffine)
AFFINE_VARIANTS = {
    'rotate_3deg': rotate_3deg,
    'rotate_5deg': rotate_5deg,
    'rotate_7deg': rotate_7deg,
    'rotate_neg4deg': rotate_neg4deg,
    'zoom_in_10pct': zoom_in_10pct,
    'zoom_in_15pct': zoom_in_15pct,
    'zoom_out_10pct': zoom_out_10pct,
    'zoom_out_15pct': zoom_out_15pct,
    'shift_right_30px': shift_right_30px,
    'shift_down_20px': shift_down_20px,
    'shift_up_30px': shift_up_30px,
    'shift_diag_30px': shift_diag_30px,
    'rot3_zoom_in_10': rot3_zoom_in_10,
    'rot5_zoom_out_10': rot5_zoom_out_10,
    'shift20_zoom_in_10': shift20_zoom_in_10,
    'shift25_rot4_zoom8': shift25_rot4_zoom8,
    'shiftn15_rotn3_zoomout8': shiftn15_rotn3_zoomout8,
    'combined_shift_rot': combined_shift_rot,
}


# Images too dark for distortion to produce viable test cases
SKIP_BASES = {
    '14-B2-UNMUTE-gap-bright',
    '27-B2-UNMUTE-misread',
    'PP-S1-UNMUTE-dark',
}

# Per-variant skips: panel moves out of frame under extreme transforms
SKIP_PAIRS = {
    ('08-B2-UNMUTE-0vs1-close', 'shift25_rot4_zoom8'),
    ('10-B2-UNMUTE-ambiguous', 'shift25_rot4_zoom8'),
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = sorted(glob.glob(os.path.join(EXAMPLE_DIR, '*.png')) +
                    glob.glob(os.path.join(EXAMPLE_DIR, '*.PNG')))

    generated = 0
    skipped = 0
    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  SKIP (unreadable): {img_path}")
            continue
        base = os.path.splitext(os.path.basename(img_path))[0]
        if base in SKIP_BASES:
            continue
        h, w = frame.shape[:2]

        # Perspective variants
        for name, make_matrix in PERSPECTIVE_VARIANTS.items():
            if (base, name) in SKIP_PAIRS:
                continue
            out_path = os.path.join(OUTPUT_DIR, f"{base}_{name}.png")
            if os.path.exists(out_path):
                skipped += 1
                continue
            M = make_matrix(w, h)
            warped = cv2.warpPerspective(frame, M, (w, h),
                                         borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(out_path, warped)
            generated += 1

        # Affine variants
        for name, make_matrix in AFFINE_VARIANTS.items():
            if (base, name) in SKIP_PAIRS:
                continue
            out_path = os.path.join(OUTPUT_DIR, f"{base}_{name}.png")
            if os.path.exists(out_path):
                skipped += 1
                continue
            M = make_matrix(w, h)
            warped = cv2.warpAffine(frame, M, (w, h),
                                    borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(out_path, warped)
            generated += 1

        print(f"  {base}")

    print(f"\nDone: {generated} new, {skipped} already existed")
    print(f"Total distorted images: {len(glob.glob(os.path.join(OUTPUT_DIR, '*.png')))}")


if __name__ == '__main__':
    main()
