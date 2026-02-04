#!/usr/bin/env python3
"""Reconstruct debug overlay from logged frame txt + image.

Parses the key/value txt file saved by log_issue_frame(), re-runs
slant correction and digit extraction, then draws the full overlay
using draw_display_overlay() — the same code used by test_on_image().

Usage:
    python scripts/overlay_from_log.py logs/20260203_201314_led_fail.png
    python scripts/overlay_from_log.py logs/20260203_201314_led_fail.txt
    python scripts/overlay_from_log.py logs/  # process all txt+png pairs

Options:
    --save       Save overlay image next to source (adds _overlay suffix)
    --no-display Don't show the image in a window
"""
import ast
import cv2
import glob
import numpy as np
import os
import re
import sys

# Import from segment_reader
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
from segment_reader import (draw_display_overlay, draw_led_debug, draw_mute_debug,
                            correct_slant, _extract_digit_with_padding,
                            get_geometry, set_undistort)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_txt(path):
    """Parse a log txt file into a dict.

    Returns (data, sections) where:
      - data: flat dict of unprefixed keys
      - sections: ordered list of (prefix, section_data) for prefixed keys
    """
    data = {}
    sections = []
    section_map = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            m = re.match(r'^([^:]+):\s*(.*)', line)
            if not m:
                continue
            full_key, val = m.group(1).strip(), m.group(2).strip()
            try:
                parsed_val = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                parsed_val = val

            # Check for prefixed key (e.g. "NA_glitch1/panel")
            if '/' in full_key:
                prefix, key = full_key.split('/', 1)
                if prefix not in section_map:
                    section_map[prefix] = {}
                    sections.append((prefix, section_map[prefix]))
                section_map[prefix][key] = parsed_val
            else:
                data[full_key] = parsed_val

    return data, sections


def setup_geometry(data):
    """Set up geometry from txt data: corner position + homography.

    Must be called before undistort_roi (for derotation) and
    build_led_debug_info (for predicted B1 box).
    """
    geometry = get_geometry()
    set_undistort(True)

    corner_pos = data.get('corner_position')
    if corner_pos:
        geometry.set_corner(*corner_pos)

    # Reconstruct homography from corner + button positions
    buttons = data.get('button_positions', [])
    led_region = data.get('led_region')
    if corner_pos and len(buttons) >= 3 and led_region:
        btn_left, btn_top = led_region[0], led_region[1]
        button_centers = {}
        for name, btn in zip(['B2', 'S1', 'S2'], buttons):
            cx = btn_left + btn[0] + btn[2] / 2
            cy = btn_top + btn[1] + btn[3] / 2
            button_centers[name] = (cx, cy)
        geometry.compute_homography(corner_pos, button_centers)

    return geometry


def build_led_debug_info(data):
    """Reconstruct led_debug_info dict from parsed txt data."""
    led_region = data.get('led_region')
    led_zones = data.get('led_zones')
    if not led_region or not led_zones:
        return None

    led_lit = data.get('led_lit')
    if led_lit == 'None' or led_lit is None:
        led_lit = None

    # txt stores (name, lx1, lx2, ly1, ly2); draw_led_debug expects (lx1, lx2, ly1, ly2, name)
    zones = [(lx1, lx2, ly1, ly2, name) for name, lx1, lx2, ly1, ly2 in led_zones]
    zone_names = [name for name, *_ in led_zones]
    leds = {name: (name == led_lit) for name in zone_names}

    led_position = data.get('led_position')
    if led_position == 'None' or led_position is None:
        led_position = None

    buttons = data.get('button_positions', [])

    # Compute predicted B1 box from geometry (same as live pipeline)
    predicted_b1_box = None
    geometry = get_geometry()
    if len(buttons) >= 3:
        widths = [b[2] for b in buttons]
        heights = [b[3] for b in buttons]
        avg_width = sum(widths) / len(widths)
        avg_height = sum(heights) / len(heights)

        b1_proj = geometry.project_landmark('B1')
        if b1_proj is not None:
            btn_left, btn_top = led_region[0], led_region[1]
            b1_abs_x, b1_abs_y = b1_proj
            b1_center = b1_abs_x - btn_left
            b1_y_rel = b1_abs_y - btn_top
            b1_x = int(b1_center - avg_width / 2)
            b1_y = int(b1_y_rel - avg_height / 2)
            predicted_b1_box = (b1_x, b1_y, int(avg_width), int(avg_height))

    return {
        'region': led_region,
        'zones': zones,
        'buttons': buttons,
        'predicted_b1_box': predicted_b1_box,
        'led_position': led_position,
        'lit_led': led_lit,
        'leds': leds,
    }


def build_corner_debug(data):
    """Reconstruct corner_debug tuple from parsed txt data.

    Uses corner_position from the log to build the (search_rect, match_rect,
    crop_size) tuple that draw_corner_debug expects, without re-running
    template matching.
    """
    corner_pos = data.get('corner_position')
    if not corner_pos:
        return None

    cx, cy = corner_pos
    # All corner templates are 150x150, cropped to bottom-right quadrant (75x75)
    crop_size = (75, 75)

    # Search region uses geometry (already set up by setup_geometry)
    geometry = get_geometry()
    search_x, search_y, search_size = geometry.get_corner_search_region(640, 480)

    search_rect = (search_x, search_y, search_size, search_size)
    match_rect = (cx, cy, crop_size[0], crop_size[1])

    return (search_rect, match_rect, crop_size)


def build_mute_debug_info(data):
    """Reconstruct mute_debug_info dict from parsed txt data."""
    mute_region = data.get('mute_region')
    if not mute_region:
        return None

    mute_status = data.get('mute_status', '')
    mute_center = data.get('mute_led_center')
    if mute_center == 'None' or mute_center is None:
        mute_center = None
    else:
        mute_center = tuple(mute_center)

    return {
        'region': mute_region,
        'is_lit': mute_status == 'MUTE',
        'led_center': mute_center,
        'red_pixels': data.get('mute_pixels', 0),
    }



def draw_overlay(frame, data):
    """Draw full debug overlay using draw_display_overlay from segment_reader.

    Re-runs slant correction and digit extraction from the raw frame.
    All other data (scores, matches, corner, LEDs) comes from the txt log.
    """
    panel = data.get('panel')
    if not panel:
        # No panel data — draw minimal overlay
        return _draw_minimal_overlay(frame, data)

    px, py, pw, ph = panel
    gap_x = data.get('gap_x')
    left_box = data.get('left_box')
    right_box = data.get('right_box')
    reading = str(data.get('reading', '??'))
    left_score = data.get('left_score', 0)
    right_score = data.get('right_score', 0)
    left_match = data.get('left_match')
    right_match = data.get('right_match')

    # Set up geometry (corner + homography) before undistort_roi
    geometry = setup_geometry(data)
    panel_img = geometry.undistort_roi(frame, px, py, pw, ph)
    corrected_img, _, _ = correct_slant(panel_img)

    # Re-extract digit images using boxes from txt
    left_digit_img = None
    right_digit_img = None
    if corrected_img is not None and gap_x is not None:
        if left_box:
            left_digit_img = _extract_digit_with_padding(
                corrected_img, left_box, right_bound=gap_x)
        if right_box:
            right_digit_img = _extract_digit_with_padding(
                corrected_img, right_box, left_bound=gap_x)

    # Parse digit and second-best from txt data
    left_digit = reading[0] if isinstance(reading, str) and len(reading) == 2 else '?'
    right_digit = reading[1] if isinstance(reading, str) and len(reading) == 2 else '?'

    # Parse "X:0.788" format for 2nd candidates
    left_2nd_str = data.get('left_2nd', '')
    right_2nd_str = data.get('right_2nd', '')
    left_second, left_second_score = _parse_2nd(left_2nd_str)
    right_second, right_second_score = _parse_2nd(right_2nd_str)

    led_status = data.get('led_status', '??')
    mute_status = data.get('mute_status', '')

    # Reconstruct corner debug from txt data (don't re-run detection)
    corner_debug = build_corner_debug(data)

    overlay = draw_display_overlay(
        frame, panel, corrected_img, gap_x,
        left_digit_img, right_digit_img,
        left_digit, right_digit, left_score, right_score,
        left_match, right_match,
        left_second, left_second_score,
        right_second, right_second_score,
        reading, led_status, mute_status,
        corner_debug=corner_debug,
        led_debug_info=build_led_debug_info(data),
        mute_debug_info=build_mute_debug_info(data),
        frame_skipped=(data.get('frame_skipped') == 'yes'),
    )

    return overlay


def _parse_2nd(s):
    """Parse '8:0.825' format into (digit, score)."""
    if not s or not isinstance(s, str):
        return 'X', 0.0
    parts = s.split(':')
    if len(parts) == 2:
        try:
            return parts[0], float(parts[1])
        except ValueError:
            pass
    return s, 0.0


def _draw_minimal_overlay(frame, data):
    """Draw minimal overlay when no panel data is available."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Corner marker
    corner_pos = data.get('corner_position')
    corner_score = data.get('corner_score', 0)
    if corner_pos:
        cx, cy = corner_pos
        cv2.drawMarker(overlay, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(overlay, f"corner:{corner_score:.2f}",
                    (cx - 50, cy - 15), FONT, 0.35, (0, 255, 255), 1)

    # LED zones
    draw_led_debug(overlay, build_led_debug_info(data))
    draw_mute_debug(overlay, build_mute_debug_info(data))

    # Status text
    led_status = data.get('led_status', '??')
    mute_status = data.get('mute_status', '')
    reading = data.get('reading', '??')
    status_text = f"LED:{led_status}  {mute_status}  [{reading}]"
    text_size = cv2.getTextSize(status_text, FONT, 1.0, 2)[0]
    bg_x2 = 15 + text_size[0] + 10
    roi = overlay[5:40, 5:bg_x2]
    overlay[5:40, 5:bg_x2] = (roi * 0.5).astype(roi.dtype)
    cv2.putText(overlay, status_text, (10, 30), FONT, 1.0, (255, 255, 255), 2)

    return overlay


def find_pairs(path):
    """Find matching txt+png pairs from a path (file or directory)."""
    pairs = []
    if os.path.isdir(path):
        txts = sorted(glob.glob(os.path.join(path, '*.txt')))
        for txt in txts:
            png = txt[:-4] + '.png'
            if os.path.exists(png):
                pairs.append((png, txt))
    else:
        base = path.rsplit('.', 1)[0]
        txt = base + '.txt'
        png = base + '.png'
        if os.path.exists(txt) and os.path.exists(png):
            pairs.append((png, txt))
        else:
            print(f"Error: Need both {os.path.basename(png)} and {os.path.basename(txt)}")
            sys.exit(1)
    return pairs


def main():
    save = '--save' in sys.argv
    no_display = '--no-display' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    if not args:
        print(__doc__.strip())
        sys.exit(1)

    pairs = find_pairs(args[0])
    if not pairs:
        print(f"No txt+png pairs found in {args[0]}")
        sys.exit(1)

    print(f"Found {len(pairs)} pair(s)")

    for png_path, txt_path in pairs:
        name = os.path.basename(png_path)
        data, sections = parse_txt(txt_path)
        img = cv2.imread(png_path)
        if img is None:
            print(f"  Skip: cannot read {name}")
            continue

        h, w = img.shape[:2]

        # 1280x480 = raw|display: extract raw half, redraw full overlay
        if w == 1280 and h == 480:
            raw = img[:, :640].copy()
            overlay = draw_overlay(raw, data)
            overlay = np.hstack([raw, overlay])
            print(f"  {name}: reading={data.get('reading')} led={data.get('led_status')} mute={data.get('mute_status')} (raw|overlay)")

        # Composite: multiple 640-wide frames with prefixed txt sections
        elif sections and h == 480 and w > 640 and w % 640 == 0:
            n_frames = w // 640
            if len(sections) != n_frames:
                print(f"  Skip: {name} has {n_frames} frames but {len(sections)} sections")
                continue
            overlay_frames = []
            for i, (prefix, section_data) in enumerate(sections):
                frame_slice = img[:, i * 640:(i + 1) * 640].copy()
                overlay_slice = draw_overlay(frame_slice, section_data)
                cv2.putText(overlay_slice, prefix, (10, h - 10), FONT, 0.5, (0, 255, 255), 1)
                overlay_frames.append(overlay_slice)
            overlay = np.hstack(overlay_frames)
            print(f"  {name}: {n_frames} frames [{', '.join(p for p, _ in sections)}]")

        # Single 640x480 frame with flat keys
        elif w == 640 and h == 480:
            raw = img.copy()
            overlay = draw_overlay(raw, data)
            overlay = np.hstack([raw, overlay])
            print(f"  {name}: reading={data.get('reading')} led={data.get('led_status')} mute={data.get('mute_status')}")

        else:
            print(f"  Skip: {name} unsupported size ({w}x{h})")
            continue

        if save:
            out_path = png_path.replace('.png', '_overlay.png')
            cv2.imwrite(out_path, overlay)
            print(f"    Saved: {out_path}")

        if not no_display:
            cv2.imshow(f'Overlay: {name}', overlay)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == 27:  # ESC
                break


if __name__ == '__main__':
    main()
