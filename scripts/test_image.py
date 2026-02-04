#!/usr/bin/env python3
"""Test digit recognition on one or more images using current code.

Runs the full pipeline (panel detection, LED, slant correction, digit
recognition) via test_on_image() and optionally displays the overlay.

Supports multi-frame composite images:
    640x480   - single frame, test as-is
    1280x480  - raw|display pair, use left half; output raw|overlay (1280x480)
    1920x480  - 3 frames, test each; output 3 overlays side-by-side (1920x480)
    2560x480  - 4 frames; output 2560x480
    3200x480  - 5 frames; output 3200x480

Usage:
    python scripts/test_image.py path/to/image.png
    python scripts/test_image.py image1.png image2.png
    python scripts/test_image.py logs/  # process all PNGs in directory

Options:
    --save       Save overlay image next to source (adds _overlay suffix)
    --no-display Don't show the image in a window
"""
import cv2
import glob
import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
from segment_reader import test_on_image, set_undistort

TMP_DIR = os.path.join(PROJECT_ROOT, '.claude', 'tmp')


def extract_frames(img_path):
    """Extract 640x480 frames from a possibly composite image.

    Returns list of (frame_path, label) tuples. For single-frame images
    the original path is returned. For composites, slices are saved to
    tmp files.
    """
    img = cv2.imread(img_path)
    if img is None:
        return []

    h, w = img.shape[:2]
    base = os.path.splitext(os.path.basename(img_path))[0]

    if h != 480 or w % 640 != 0:
        # Non-standard size, try as-is
        return [(img_path, None)]

    n_frames = w // 640

    if n_frames == 1:
        return [(img_path, None)]

    os.makedirs(TMP_DIR, exist_ok=True)

    if n_frames == 2:
        # 1280x480 = raw|display, use left half
        tmp_path = os.path.join(TMP_DIR, f'{base}_raw.png')
        cv2.imwrite(tmp_path, img[:, :640])
        return [(tmp_path, 'left half of 1280x480')]

    # 3+ frames: test each slice
    frames = []
    for i in range(n_frames):
        tmp_path = os.path.join(TMP_DIR, f'{base}_frame{i}.png')
        cv2.imwrite(tmp_path, img[:, i * 640:(i + 1) * 640])
        frames.append((tmp_path, f'frame {i+1}/{n_frames}'))
    return frames


def main():
    save = '--save' in sys.argv
    no_display = '--no-display' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    if not args:
        print(__doc__.strip())
        sys.exit(1)

    # Collect all image paths
    images = []
    for arg in args:
        if os.path.isdir(arg):
            images.extend(sorted(
                glob.glob(os.path.join(arg, '*.png')) +
                glob.glob(os.path.join(arg, '*.PNG')) +
                glob.glob(os.path.join(arg, '*.jpg')) +
                glob.glob(os.path.join(arg, '*.JPG'))
            ))
        elif os.path.isfile(arg):
            images.append(arg)
        else:
            print(f"Warning: {arg} not found, skipping")

    if not images:
        print("No images found")
        sys.exit(1)

    set_undistort(True)
    print(f"Testing {len(images)} image(s)\n")

    for img_path in images:
        frames = extract_frames(img_path)
        if not frames:
            print(f"ERROR: Could not load {img_path}\n")
            continue

        name = os.path.basename(img_path)
        if len(frames) > 1 or frames[0][1]:
            print(f"--- {name}: {frames[0][1] or f'{len(frames)} frames'} ---")

        overlays = []
        raw_frames = []
        for frame_path, label in frames:
            test_on_image(frame_path)

            base_name = os.path.splitext(os.path.basename(frame_path))[0]
            overlay_path = os.path.join(PROJECT_ROOT, 'debug', f'{base_name}_overlay.png')
            if os.path.exists(overlay_path):
                overlays.append(cv2.imread(overlay_path))
                raw_frames.append(cv2.imread(frame_path))

            print()

        # Build output matching input layout
        if overlays and (save or not no_display):
            img = cv2.imread(img_path)
            n_frames = img.shape[1] // 640 if img is not None else 1

            if n_frames == 1:
                # 640x480: raw | overlay
                combined = np.hstack([raw_frames[0], overlays[0]])
            elif n_frames == 2:
                # 1280x480: raw | overlay
                combined = np.hstack([raw_frames[0], overlays[0]])
            else:
                # 3+ frames: overlay per frame, same width as input
                combined = np.hstack(overlays[:n_frames])

            if save:
                out_path = img_path.rsplit('.', 1)[0] + '_overlay.png'
                cv2.imwrite(out_path, combined)
                print(f"  Saved: {out_path}")

            if not no_display:
                cv2.imshow(f'Test: {name}', combined)
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
                if key == 27:  # ESC
                    break


if __name__ == '__main__':
    main()
