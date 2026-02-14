#!/usr/bin/env python3
"""Test distorted images and report failures.

Checks:
  - UNMUTE images: at least one source LED (B1/B2/S1/S2) detected
  - MUTE images: detect_red_button returns muted=True
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import segment_reader as sr

# Auto-generate distorted images if directory is missing or empty
if not os.path.isdir('distorted') or not glob.glob('distorted/*.png'):
    import subprocess
    print("Generating distorted test images...")
    subprocess.check_call([sys.executable,
                           os.path.join(os.path.dirname(__file__), 'gen_perspective_variants.py')])

failures = []
for path in sorted(glob.glob('distorted/*.png')):
    base = os.path.basename(path)
    if 'MUTE_NA' in base:
        expected_mute = None  # washout — mute status unknown, just test panel
    elif 'MUTE' in base and 'UNMUTE' not in base:
        expected_mute = True
    elif 'UNMUTE' in base:
        expected_mute = False
    else:
        expected_mute = None  # no mute/unmute label — just test panel detection

    # Reset geometry state between images
    sr._geometry._corner_xy = None
    sr._geometry._homography = None
    sr._geometry._scale = 1.0
    sr._geometry._smoothed_homography = None
    sr._geometry._smoothed_scale = 1.0
    sr._geometry._golden_homography = None
    sr._corner_template_idx = 0

    frame = cv2.imread(path)
    if frame is None:
        failures.append(base)
        continue

    panel_rect, _ = sr.detect_panel(frame)
    if panel_rect is None:
        failures.append(base)
        continue

    if expected_mute is None:
        pass  # panel detected — that's enough
    elif expected_mute:
        # Use detect_red_button for mute detection
        corner_result, _ = sr._find_corner(frame, return_debug=True)
        valid_corner = corner_result if (corner_result and corner_result[0] is not None) else None
        if valid_corner is None:
            continue  # corner out of frame — can't locate mute button
        # Skip if mute region is mostly out of frame
        h_frame, w_frame = frame.shape[:2]
        mute = sr._geometry.get_mute_region(valid_corner[0], valid_corner[1])
        btn_x, btn_y, half = mute
        clipped_w = min(w_frame, btn_x + half) - max(0, btn_x - half)
        clipped_h = min(h_frame, btn_y + half) - max(0, btn_y - half)
        if clipped_w < half or clipped_h < half:
            continue  # mute region mostly out of frame
        is_muted, _ = sr.detect_red_button(frame, corner_result=valid_corner)
        if not is_muted:
            failures.append(base)
    else:
        # Run corner detection first to calibrate geometry (matches real pipeline)
        sr._find_corner(frame)
        # Check source LED detection
        leds, _, dbg = sr.detect_button_leds(frame, panel_rect, return_debug=True)
        lit = [k for k, v in leds.items() if v]
        if not lit:
            # Extract expected source from filename (e.g. 09-B1-UNMUTE -> B1)
            parts = base.split('-')
            src = parts[1] if len(parts) >= 2 else None
            if src in ('B1',) and dbg and dbg.get('predicted_b1_box'):
                bx, _, bw, _ = dbg['predicted_b1_box']
                if bx + bw / 2 < 0:
                    continue  # B1 zone mostly out of frame
            failures.append(base)

print(f'Failures: {len(failures)}')
for f in failures:
    print(f)
