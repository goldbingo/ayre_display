#!/usr/bin/env python3
"""Live camera demo for 7-segment display reader."""

import cv2
import sys
import os
import argparse
import time
import segment_reader
from segment_reader import (SegmentReader, detect_panel, detect_button_leds, detect_red_button,
                            correct_slant, find_digit_gap, define_digit_boxes, _TEMPLATE_SIZE,
                            _find_corner, draw_corner_debug, draw_led_debug, draw_mute_debug, draw_digit_debug,
                            _extract_digit_with_padding, log_detection, log_issue_frame, close_log,
                            reload_templates)
import numpy as np
import subprocess
import shutil
import os
import json

# Notification settings (loaded from .claude/notify_config.json)
ICLOUD_ALERTS_DIR = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/SegmentReaderAlerts")
_notify_config_path = os.path.join(os.path.dirname(__file__), ".claude", "notify_config.json")
if os.path.exists(_notify_config_path):
    with open(_notify_config_path) as f:
        _notify_config = json.load(f)
    IMESSAGE_RECIPIENT = _notify_config.get("imessage_recipient")
    ICLOUD_LINK = _notify_config.get("icloud_link")
else:
    IMESSAGE_RECIPIENT = None
    ICLOUD_LINK = None

def send_notification(message, image_path=None):
    """Send iMessage notification with iCloud link to image."""
    if not IMESSAGE_RECIPIENT:
        return  # Notifications disabled (no config)
    try:
        # Copy image to iCloud folder
        if image_path and os.path.exists(image_path):
            os.makedirs(ICLOUD_ALERTS_DIR, exist_ok=True)
            image_name = os.path.basename(image_path)
            dest = os.path.join(ICLOUD_ALERTS_DIR, image_name)
            shutil.copy2(image_path, dest)
            full_message = f"{message}\\n📷 {image_name}\\n{ICLOUD_LINK}"
        else:
            full_message = message

        script = f'''
        tell application "Messages"
            set targetService to 1st account whose service type = iMessage
            set targetBuddy to participant "{IMESSAGE_RECIPIENT}" of targetService
            send "{full_message}" to targetBuddy
        end tell
        '''
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)
    except Exception as e:
        print(f"Notification failed: {e}", flush=True)

# Use TCP transport for RTSP streams
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class DemoState:
    """Holds frame-to-frame state for the live demo."""
    def __init__(self):
        # MUTE/LED state tracking
        self.last_led = "NA"
        self.last_mute = "UNMUTE"
        self.last_led_debug = None
        self.last_mute_debug = None
        # LED history for glitch detection (A-A-?-?-?-A-A pattern, up to 3 glitch frames)
        self.led_history = []
        self.stable_led = None
        self.frame_history = []  # Store recent frames for glitch logging [(raw, display), ...]
        # Pending issues to log after display frame is ready
        self.pending_led_fail = False
        self.pending_mute_na = False
        self.pending_led_transition = None  # (from_led, to_led) for B1/B2 transitions
        self.prev_led_for_transition = None  # Track previous LED for transition detection
        # Headless mode print state
        self.last_time = 0
        self.last_print = None
        self.last_mute_print = ""


def build_debug_info(reader, reading, led_status, mute_status, corner_score,
                     led_debug_info, mute_debug_info, corner_result=None):
    """Build debug info dict for logging alongside captured frames."""
    info = {}

    # Panel info
    if reader.panel_rect:
        px, py, pw, ph = reader.panel_rect
        info['panel'] = f'({px}, {py}, {pw}, {ph})'
    info['detection_method'] = reader.detection_method or 'unknown'
    if reader.gap_x:
        info['gap_x'] = reader.gap_x

    # Reading info
    info['reading'] = reading
    if reader.last_scores:
        left_score, right_score = reader.last_scores
        info['left_score'] = f'{left_score:.3f}'
        info['right_score'] = f'{right_score:.3f}'
    if reader.last_second:
        (left_2nd, left_2nd_score), (right_2nd, right_2nd_score) = reader.last_second
        info['left_2nd'] = f'{left_2nd}:{left_2nd_score:.3f}'
        info['right_2nd'] = f'{right_2nd}:{right_2nd_score:.3f}'

    # Digit extraction boxes
    if reader.digit_debug:
        if reader.digit_debug.get('left_box'):
            info['left_box'] = str(reader.digit_debug['left_box'])
        if reader.digit_debug.get('right_box'):
            info['right_box'] = str(reader.digit_debug['right_box'])
        if reader.digit_debug.get('left_match'):
            info['left_match'] = str(reader.digit_debug['left_match'])
        if reader.digit_debug.get('right_match'):
            info['right_match'] = str(reader.digit_debug['right_match'])

    # Corner info
    info['corner_score'] = f'{corner_score:.3f}' if corner_score else 'N/A'
    if corner_result:
        info['corner_position'] = f'({corner_result[0]}, {corner_result[1]})'

    # Brightness confidence
    if reader.brightness_conf:
        info['brightness_conf'] = f'{reader.brightness_conf:.3f}'

    # LED info
    info['led_status'] = led_status
    if led_debug_info:
        info['led_region'] = str(led_debug_info.get('region'))
        info['led_lit'] = led_debug_info.get('lit_led')
        info['led_position'] = str(led_debug_info.get('led_position'))
        buttons = led_debug_info.get('buttons')
        if buttons:
            info['buttons_detected'] = len(buttons)
            info['button_positions'] = str(buttons)
        zones = led_debug_info.get('zones')
        if zones:
            info['led_zones'] = str([(z[4], int(z[0]), int(z[1]), int(z[2]), int(z[3])) for z in zones])

    # MUTE info
    info['mute_status'] = mute_status
    if mute_debug_info:
        info['mute_region'] = str(mute_debug_info.get('region'))
        info['mute_pixels'] = mute_debug_info.get('red_pixels', 0)
        info['mute_method'] = mute_debug_info.get('method')
        if mute_debug_info.get('led_center'):
            info['mute_led_center'] = str(mute_debug_info.get('led_center'))

    return info


def learn_digit(digit_debug, position, correct_digit):
    """Save a digit from reader.digit_debug as a new template.

    Args:
        digit_debug: reader.digit_debug dict containing 'left_img' and 'right_img'
        position: 'left' or 'right'
        correct_digit: The correct digit character (0-9, P)

    Returns:
        filename of saved template, or None if failed
    """
    if digit_debug is None:
        return None

    # Get the exact image shown on display (already used for matching)
    img_key = f'{position}_img'
    digit_img = digit_debug.get(img_key)
    if digit_img is None:
        return None

    # Auto-trim with adaptive threshold based on brightness
    if len(digit_img.shape) == 3:
        gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = digit_img

    # Calculate brightness (top 10% pixel average)
    flat = gray.flatten()
    top10_threshold = np.percentile(flat, 90)
    brightness = flat[flat >= top10_threshold].mean()

    # Select threshold: Otsu for bright/normal, fixed for dim
    orig_area = gray.shape[0] * gray.shape[1]
    if brightness >= 100:
        # Bright/Normal: use Otsu's auto threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            cx, cy, cw, ch = cv2.boundingRect(coords)
    else:
        # Dim: use fixed threshold with escalation
        trim_thresh = 30

        for thresh_try in [trim_thresh, 80, 100, 120]:
            _, thresh = cv2.threshold(gray, thresh_try, 255, cv2.THRESH_BINARY)
            coords = cv2.findNonZero(thresh)
            if coords is not None:
                cx, cy, cw, ch = cv2.boundingRect(coords)
                trim_area = cw * ch
                if trim_area < orig_area * 0.9:
                    break

    if coords is not None:
        # Special handling for digit 1: width = height / 2
        if correct_digit == '1':
            img_h, img_w = gray.shape[:2]

            # Vertical: extend 6px, pad if exceeds boundary
            top = cy - 6
            bottom = cy + ch + 6
            pad_top = max(0, -top)
            pad_bottom = max(0, bottom - img_h)
            top = max(0, top)
            bottom = min(img_h, bottom)

            # Horizontal: keep left at 0, right at content edge
            right = cx + cw

            # Extract region
            cropped = gray[top:bottom, 0:right]

            # Pad top/bottom if needed (replicate edge rows)
            if pad_top > 0:
                top_row = cropped[0:1, :]
                top_padding = np.tile(top_row, (pad_top, 1))
                cropped = np.vstack([top_padding, cropped])
            if pad_bottom > 0:
                bottom_row = cropped[-1:, :]
                bottom_padding = np.tile(bottom_row, (pad_bottom, 1))
                cropped = np.vstack([cropped, bottom_padding])

            # Adjust width to height / 2
            new_h = cropped.shape[0]
            target_w = int(new_h / 2)
            current_w = cropped.shape[1]

            if current_w > target_w:
                # Trim from left
                digit_img = cropped[:, current_w - target_w:]
            elif current_w < target_w:
                # Pad left by replicating leftmost column
                pad_w = target_w - current_w
                padding = np.tile(cropped[:, 0:1], (1, pad_w))
                digit_img = np.hstack([padding, cropped])
            else:
                digit_img = cropped
        else:
            # Standard trim for other digits
            digit_img = gray[cy:cy+ch, cx:cx+cw]
    else:
        digit_img = gray

    # Find next available letter suffix
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)

    existing = [f for f in os.listdir(templates_dir)
                if f.startswith(f'digit_{correct_digit}') and f.endswith('.png')]
    used_letters = set()
    for f in existing:
        name = f.replace('digit_', '').replace('.png', '')
        if len(name) >= 2:
            used_letters.add(name[1])

    next_letter = None
    for c in 'abcdefghijklmnopqrstuvwxyz':
        if c not in used_letters:
            next_letter = c
            break

    if next_letter is None:
        return None

    filename = f'digit_{correct_digit}{next_letter}.png'
    filepath = os.path.join(templates_dir, filename)
    if not cv2.imwrite(filepath, digit_img):
        print(f"Warning: Failed to write template {filepath}", flush=True)
        return None

    # Reload templates from disk to pick up the new one
    segment_reader.reload_templates()

    return filename


def open_stream(source, width=640, height=480):
    """Open camera or RTSP stream."""
    if source.startswith('rtsp://') or source.startswith('http://'):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        is_stream = True
    else:
        cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        is_stream = False
    return cap, is_stream


def main():
    parser = argparse.ArgumentParser(description='Live 7-segment display reader')
    parser.add_argument('--width', '-W', type=int, default=640,
                        help='Frame width (default: 640)')
    parser.add_argument('--height', '-H', type=int, default=480,
                        help='Frame height (default: 480)')
    parser.add_argument('--skip', '-s', type=int, default=3,
                        help='Process every Nth frame (default: 3, use 1 to disable)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without display (print readings to console)')
    args = parser.parse_args()

    # Read camera address from webcam.link file
    webcam_link_path = os.path.join(os.path.dirname(__file__), 'webcam.link')
    if not os.path.exists(webcam_link_path):
        print(f"Error: {webcam_link_path} not found", flush=True)
        sys.exit(1)
    with open(webcam_link_path, 'r') as f:
        camera = f.read().strip()

    # Open camera or stream
    cap, is_stream = open_stream(camera, args.width, args.height)
    if is_stream:
        print(f"Opening stream: {camera.split('@')[-1]}", flush=True)  # Hide credentials
    else:
        print(f"Opening camera: {camera}", flush=True)

    if not cap.isOpened():
        print("Error: Could not open video source", flush=True)
        sys.exit(1)

    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}", flush=True)
    if args.headless:
        print("Headless mode: Ctrl+C to quit", flush=True)
    else:
        print("Press 'q' quit, 'c' reset, 's' save, 'l#/r#' learn (e.g. l6, r8)", flush=True)
    print("-" * 40, flush=True)

    # No cache, no auto-learn - detect fresh every frame
    reader = SegmentReader(auto_learn=False)

    # Frame-to-frame state
    state = DemoState()

    # Frame skipping for CPU efficiency (process every Nth frame)
    frame_skip = max(1, args.skip)
    frame_count = 0
    if frame_skip > 1:
        print(f"Frame skip: {frame_skip} (processing every {frame_skip} frames)", flush=True)

    # Skip initial frames for RTSP streams
    if is_stream:
        for _ in range(30):
            cap.read()

    fail_count = 0
    max_fails = 50  # Reconnect after this many consecutive failures
    reconnect_delay = 2  # Seconds to wait before reconnecting
    pending_learn = None  # 'left' or 'right' when L or R pressed

    while True:
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if is_stream and fail_count >= max_fails:
                print(f"Connection lost. Reconnecting in {reconnect_delay}s...", flush=True)
                cap.release()
                time.sleep(reconnect_delay)
                cap, _ = open_stream(camera, args.width, args.height)
                if cap.isOpened():
                    print("Reconnected successfully", flush=True)
                    # Skip initial frames
                    for _ in range(30):
                        cap.read()
                    fail_count = 0
                else:
                    print("Reconnect failed, will retry...", flush=True)
                    fail_count = 0  # Reset to trigger another reconnect attempt after max_fails
                    time.sleep(reconnect_delay)  # Extra delay before next attempt
            continue

        fail_count = 0  # Reset on successful read
        frame_count += 1

        # Always run digit recognition (no caching of recognized digits)
        reading, cache_hit = reader.read(frame)
        # Debug: save frame when detecting wrong readings
        if reading in ["08", "P6", "6P", "01", "00", "09", "03", "18"]:
            debug_path = f'/tmp/debug_{reading}.png'
            if not cv2.imwrite(debug_path, frame):
                print(f"Warning: Failed to write {debug_path}", flush=True)

        # Frame skipping for LED/MUTE detection only
        if frame_count % frame_skip == 0:
            # Detect LED (enlarge zones when in fallback mode)
            leds, _, led_debug_info = detect_button_leds(frame, reader.panel_rect, return_debug=True,
                                                          detection_method=reader.detection_method)
            lit_leds = [k for k, v in leds.items() if v]
            led_status = lit_leds[0] if lit_leds else "NA"
            # Detect MUTE (red button)
            is_muted, _, mute_debug_info = detect_red_button(frame, return_debug=True)
            mute_pixels = mute_debug_info.get('red_pixels', 0) if mute_debug_info else 0
            # If pixel count is abnormally high (>100), mark as unreliable
            if mute_pixels > 100:
                mute_status = "MUTE_NA"
            else:
                mute_status = "MUTE" if is_muted else "UNMUTE"

        else:
            led_status = state.last_led
            mute_status = state.last_mute
            led_debug_info = state.last_led_debug
            mute_debug_info = state.last_mute_debug

        # Store last LED and MUTE status
        state.last_led = led_status
        state.last_mute = mute_status
        state.last_led_debug = led_debug_info
        state.last_mute_debug = mute_debug_info

        # Log detection data (every processed frame)
        if frame_count % frame_skip == 0:
            corner_result, _ = _find_corner(frame, return_debug=True)
            corner_score = corner_result[2] if corner_result else 0
            left_score, right_score = reader.last_scores
            mute_pixels = mute_debug_info.get('red_pixels', 0) if mute_debug_info else 0
            log_detection(
                panel_rect=reader.panel_rect,
                gap_x=reader.gap_x,
                left_score=left_score,
                right_score=right_score,
                reading=reading,
                led_status=led_status,
                corner_score=corner_score,
                detection_method=reader.detection_method,
                brightness_conf=reader.brightness_conf,
                mute_status=mute_status,
                mute_pixels=mute_pixels,
                issue='led_fail' if led_status == 'NA' else ('mute_na' if mute_status == 'MUTE_NA' else None)
            )
            # Mark issues for logging after display frame is ready
            state.pending_led_fail = (led_status == 'NA')
            state.pending_mute_na = (mute_status == 'MUTE_NA')

            # Detect LED transition to B1 (unusual state)
            if led_status == 'B1' and state.prev_led_for_transition != 'B1':
                state.pending_led_transition = (state.prev_led_for_transition, led_status)
            else:
                state.pending_led_transition = None
            state.prev_led_for_transition = led_status

            # Track LED history for glitch detection (A-A-?-?-?-A-A pattern)
            state.led_history.append(led_status)
            if len(state.led_history) > 8:
                state.led_history.pop(0)

            # Detect glitch: 1-3 different frames surrounded by stable frames
            # Patterns: A-A-B-A-A (1), A-A-B-B-A-A (2), A-A-B-B-B-A-A (3)
            def detect_glitch(h):
                """Detect glitch pattern in LED history, returns (glitch_count, stable_led, glitch_frames) or None."""
                if len(h) < 5:
                    return None
                # Check 1-frame glitch: A-A-B-A-A
                if len(h) >= 5 and h[-5] == h[-4] == h[-2] == h[-1] and h[-3] != h[-1]:
                    return (1, h[-1], [h[-3]])
                # Check 2-frame glitch: A-A-B-B-A-A
                if len(h) >= 6 and h[-6] == h[-5] == h[-2] == h[-1] and h[-4] != h[-1] and h[-3] != h[-1]:
                    return (2, h[-1], [h[-4], h[-3]])
                # Check 3-frame glitch: A-A-B-B-B-A-A
                if len(h) >= 7 and h[-7] == h[-6] == h[-2] == h[-1] and h[-5] != h[-1] and h[-4] != h[-1] and h[-3] != h[-1]:
                    return (3, h[-1], [h[-5], h[-4], h[-3]])
                return None

            glitch = detect_glitch(state.led_history)
            if glitch and len(state.frame_history) >= glitch[0]:
                glitch_count, stable_led, glitch_leds = glitch
                glitch_str = '->'.join(glitch_leds)
                # Log the glitch frame(s)
                saved_path = None
                for i in range(glitch_count):
                    idx = -(glitch_count - i)  # Get frames from history
                    if abs(idx) <= len(state.frame_history):
                        raw_frame, display_frame = state.frame_history[idx]
                        path = log_issue_frame(raw_frame, 'led_glitch',
                                       extra_info=f'{glitch_count}f_{glitch_str}_in_{stable_led}',
                                       display_frame=display_frame)
                        if path:
                            saved_path = path
                print(f"LED GLITCH ({glitch_count}f): {stable_led} -> {glitch_str} -> {stable_led}", flush=True)
                send_notification(f"LED GLITCH ({glitch_count}f): {stable_led} -> {glitch_str} -> {stable_led}", saved_path)

        if args.headless:
            # Headless mode: print when reading changes or every minute
            now = time.time()
            if reading != state.last_print or mute_status != state.last_mute_print or (now - state.last_time) >= 60:
                print(f"Reading: {reading}  LED: {led_status}  {mute_status}", flush=True)
                state.last_print = reading
                state.last_mute_print = mute_status
                state.last_time = now

            # Build debug info for logging (headless mode)
            debug_info = build_debug_info(reader, reading, led_status, mute_status,
                                          corner_score, led_debug_info, mute_debug_info,
                                          corner_result=corner_result)

            # Log LED fail (no display frame in headless mode)
            if state.pending_led_fail:
                path = log_issue_frame(frame, 'led_fail', debug_info=debug_info)
                send_notification(f"LED FAIL: detection failed", path)
                state.pending_led_fail = False

            # Log MUTE_NA (no display frame in headless mode)
            if state.pending_mute_na:
                path = log_issue_frame(frame, 'mute_na', extra_info=f'{mute_pixels}px', debug_info=debug_info)
                send_notification(f"MUTE_NA: {mute_pixels}px (abnormal)", path)
                state.pending_mute_na = False

            # Log LED transition to B1/B2 (no display frame in headless mode)
            if state.pending_led_transition:
                from_led, to_led = state.pending_led_transition
                log_issue_frame(frame, 'led_transition', extra_info=f'{from_led}_to_{to_led}', debug_info=debug_info)
                state.pending_led_transition = None

            # Store current frame for glitch detection
            state.frame_history.append((frame.copy(), None))
            if len(state.frame_history) > 5:
                state.frame_history.pop(0)
        else:
            # Save original frame for learning (before overlays)
            original_frame = frame.copy()

            # Draw panel rectangle if detected
            if reader.panel_rect:
                x, y, w, h = reader.panel_rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw corner search area and match location
            corner_result, corner_debug = _find_corner(frame, return_debug=True)
            draw_corner_debug(frame, corner_debug)

            # Draw LED zones and detection
            draw_led_debug(frame, led_debug_info)

            # Draw MUTE LED detection area
            draw_mute_debug(frame, mute_debug_info)

            # Draw reading, LED and MUTE status at top left with semi-transparent background
            status_text = f"{reading}  LED:{led_status}  {mute_status}"
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = 10
            # Draw semi-transparent black background (region-based for efficiency)
            bg_x1, bg_y1 = 5, 5
            bg_x2, bg_y2 = text_x + text_size[0] + 10, 40
            roi = frame[bg_y1:bg_y2, bg_x1:bg_x2]
            dark_roi = (roi * 0.5).astype(roi.dtype)
            frame[bg_y1:bg_y2, bg_x1:bg_x2] = dark_roi
            cv2.putText(frame, status_text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Display extracted digit images at top-right of frame with labels
            if reader.digit_debug:
                left_img = reader.digit_debug.get('left_img')
                right_img = reader.digit_debug.get('right_img')
                left_score, right_score = reader.last_scores if reader.last_scores else (0.0, 0.0)
                if reader.last_second:
                    (left_second, left_second_score), (right_second, right_second_score) = reader.last_second
                else:
                    left_second, left_second_score = 'X', 0.0
                    right_second, right_second_score = 'X', 0.0
                left_digit = reading[0] if len(reading) > 0 else 'X'
                right_digit = reading[1] if len(reading) > 1 else 'X'

                frame_w = frame.shape[1]
                x_offset = frame_w - 10  # Start from right edge
                img_y = 5
                label_font = cv2.FONT_HERSHEY_SIMPLEX
                label_scale = 0.7
                label_thick = 2

                if right_img is not None:
                    # Make a copy to avoid modifying the original
                    if len(right_img.shape) == 2:
                        right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
                    else:
                        right_img = right_img.copy()
                    # Draw matched template box on the image
                    right_match = reader.digit_debug.get('right_match')
                    if right_match and right_match.get('match_pos') and right_match.get('template_size'):
                        mx, my = right_match['match_pos']
                        tw, th = right_match['template_size']
                        cv2.rectangle(right_img, (mx, my), (mx + tw, my + th), (0, 255, 0), 1)
                    h, w = right_img.shape[:2]
                    x_offset -= w
                    if x_offset >= 0:
                        frame[img_y:img_y+h, x_offset:x_offset+w] = right_img
                        cv2.rectangle(frame, (x_offset, img_y), (x_offset+w, img_y+h), (0, 255, 255), 1)
                        # Labels under right image with semi-transparent background
                        label1 = f"{right_digit}:{int(right_score*100)}%"
                        label2 = f"{right_second}:{int(right_second_score*100)}%"
                        text_size1 = cv2.getTextSize(label1, label_font, label_scale, label_thick)[0]
                        text_size2 = cv2.getTextSize(label2, label_font, label_scale, label_thick)[0]
                        max_text_w = max(text_size1[0], text_size2[0])
                        bg_x1, bg_y1 = max(0, x_offset - 3), img_y + h + 3
                        bg_x2, bg_y2 = x_offset + max_text_w + 3, min(frame.shape[0], img_y + h + 48)
                        if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                            roi = frame[bg_y1:bg_y2, bg_x1:bg_x2]
                            frame[bg_y1:bg_y2, bg_x1:bg_x2] = (roi * 0.5).astype(roi.dtype)
                        cv2.putText(frame, label1, (x_offset, img_y+h+20), label_font, label_scale, (0, 255, 255), label_thick)
                        cv2.putText(frame, label2, (x_offset, img_y+h+42), label_font, label_scale, (128, 255, 255), label_thick)
                        right_x = x_offset
                    x_offset -= 5

                if left_img is not None:
                    # Make a copy to avoid modifying the original
                    if len(left_img.shape) == 2:
                        left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
                    else:
                        left_img = left_img.copy()
                    # Draw matched template box on the image
                    left_match = reader.digit_debug.get('left_match')
                    if left_match and left_match.get('match_pos') and left_match.get('template_size'):
                        mx, my = left_match['match_pos']
                        tw, th = left_match['template_size']
                        cv2.rectangle(left_img, (mx, my), (mx + tw, my + th), (0, 255, 0), 1)
                    h, w = left_img.shape[:2]
                    x_offset -= w
                    if x_offset >= 0:
                        frame[img_y:img_y+h, x_offset:x_offset+w] = left_img
                        cv2.rectangle(frame, (x_offset, img_y), (x_offset+w, img_y+h), (255, 0, 255), 1)
                        # Labels under left image with semi-transparent background
                        label1 = f"{left_digit}:{int(left_score*100)}%"
                        label2 = f"{left_second}:{int(left_second_score*100)}%"
                        text_size1 = cv2.getTextSize(label1, label_font, label_scale, label_thick)[0]
                        text_size2 = cv2.getTextSize(label2, label_font, label_scale, label_thick)[0]
                        max_text_w = max(text_size1[0], text_size2[0])
                        bg_x1, bg_y1 = max(0, x_offset - 3), img_y + h + 3
                        bg_x2, bg_y2 = x_offset + max_text_w + 3, min(frame.shape[0], img_y + h + 48)
                        if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                            roi = frame[bg_y1:bg_y2, bg_x1:bg_x2]
                            frame[bg_y1:bg_y2, bg_x1:bg_x2] = (roi * 0.5).astype(roi.dtype)
                        cv2.putText(frame, label1, (x_offset, img_y+h+20), label_font, label_scale, (255, 0, 255), label_thick)
                        cv2.putText(frame, label2, (x_offset, img_y+h+42), label_font, label_scale, (255, 128, 255), label_thick)

            # Show pending learn indicator
            if pending_learn is not None:
                pos = 'LEFT' if pending_learn == 'left' else 'RIGHT'
                prompt = f"LEARN {pos}: Type digit (0-9, P) or ESC"
                cv2.rectangle(frame, (10, 10), (420, 45), (0, 0, 200), -1)
                cv2.putText(frame, prompt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Build debug info for logging
            corner_score_display = corner_result[2] if corner_result else 0
            debug_info = build_debug_info(reader, reading, led_status, mute_status,
                                          corner_score_display, led_debug_info, mute_debug_info,
                                          corner_result=corner_result)

            # Log LED fail with both raw and display frames (now that overlays are drawn)
            if state.pending_led_fail:
                path = log_issue_frame(original_frame, 'led_fail', display_frame=frame, debug_info=debug_info)
                send_notification(f"LED FAIL: detection failed", path)
                state.pending_led_fail = False

            # Log MUTE_NA with both raw and display frames
            if state.pending_mute_na:
                path = log_issue_frame(original_frame, 'mute_na', extra_info=f'{mute_pixels}px', display_frame=frame, debug_info=debug_info)
                send_notification(f"MUTE_NA: {mute_pixels}px (abnormal)", path)
                state.pending_mute_na = False

            # Log LED transition to B1/B2 with both raw and display frames
            if state.pending_led_transition:
                from_led, to_led = state.pending_led_transition
                log_issue_frame(original_frame, 'led_transition', extra_info=f'{from_led}_to_{to_led}', display_frame=frame, debug_info=debug_info)
                state.pending_led_transition = None

            # Store current frames for glitch detection
            state.frame_history.append((original_frame.copy(), frame.copy()))
            if len(state.frame_history) > 5:
                state.frame_history.pop(0)

            # Log pending issues with both raw and display frames
            if reader.pending_issue:
                issue_type, confidence, extra_info = reader.pending_issue
                log_issue_frame(original_frame, issue_type, confidence, extra_info, display_frame=frame, debug_info=debug_info)
                reader.clear_pending_issue()

            # Show frame
            cv2.imshow('7-Segment Reader', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                reader.reset_cache()
                print("Cache reset")
            elif key == ord('s'):
                # Save combined frame (raw + display side by side) with timestamp
                timestamp_str = time.strftime('%Y%m%d_%H%M%S')
                filename = f'logs/manual_{timestamp_str}.png'
                combined = np.hstack([original_frame, frame])
                cv2.imwrite(filename, combined)
                # Save debug text file
                txt_filename = f'logs/manual_{timestamp_str}.txt'
                with open(txt_filename, 'w') as f:
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Manual save (s key)\n\n")
                    for key_name, value in debug_info.items():
                        f.write(f"{key_name}: {value}\n")
                print(f"Saved {filename} + {txt_filename}")
                # Show on display
                save_frame = frame.copy()
                cv2.rectangle(save_frame, (10, 10), (350, 45), (0, 200, 0), -1)
                cv2.putText(save_frame, f"Saved: {filename}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('7-Segment Reader', save_frame)
                cv2.waitKey(1000)
            elif key in (ord('l'), ord('L')):
                # Start learning left digit
                pending_learn = 'left'
                current_left = reading[0] if len(reading) > 0 else 'X'
                print(f"LEARN LEFT - Current: {current_left} - Type correct digit (0-9, P)", flush=True)
            elif key in (ord('r'), ord('R')):
                # Start learning right digit
                pending_learn = 'right'
                current_right = reading[1] if len(reading) > 1 else 'X'
                print(f"LEARN RIGHT - Current: {current_right} - Type correct digit (0-9, P)", flush=True)
            elif pending_learn is not None:
                # Digit key after L or R
                c = chr(key).upper() if key < 256 else ''
                if c in '0123456789P':
                    position = pending_learn
                    fname = learn_digit(reader.digit_debug, position, c)
                    if fname:
                        reload_templates()  # Reload so new template works immediately
                        reader.reset_cache()  # Force full search to use new template
                        msg = f"Learned {position[0].upper()}{c} -> {fname}"
                        print(msg, flush=True)
                        # Show on screen
                        learn_frame = frame.copy()
                        cv2.rectangle(learn_frame, (10, 50), (500, 90), (0, 200, 0), -1)
                        cv2.putText(learn_frame, msg, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.imshow('7-Segment Reader', learn_frame)
                        cv2.waitKey(1500)
                    else:
                        msg = f"Failed to learn {position} digit"
                        print(msg, flush=True)
                        learn_frame = frame.copy()
                        cv2.rectangle(learn_frame, (10, 50), (400, 90), (0, 0, 200), -1)
                        cv2.putText(learn_frame, msg, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.imshow('7-Segment Reader', learn_frame)
                        cv2.waitKey(1500)
                    pending_learn = None
                elif key == 27:  # ESC to cancel
                    print("Cancelled", flush=True)
                    pending_learn = None
                else:
                    print(f"Invalid digit. Type 0-9 or P, ESC to cancel", flush=True)

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
