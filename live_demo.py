#!/usr/bin/env python3
"""Live camera demo for 7-segment display reader."""

import cv2
import sys
import os
import argparse
import time

# Redirect stdout/stderr to log file for crash debugging
_LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
if '--log' in sys.argv:
    os.makedirs(_LOG_DIR, exist_ok=True)
    _log_path = os.path.join(_LOG_DIR, 'live_demo.log')
    _log_file = open(_log_path, 'a')
    _log_file.write(f"\n{'='*60}\n")
    _log_file.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    _log_file.write(f"{'='*60}\n")
    _log_file.flush()
    sys.stdout = _log_file
    sys.stderr = _log_file
import segment_reader
from segment_reader import (SegmentReader, detect_panel, detect_button_leds, detect_red_button,
                            correct_slant, find_digit_gap, define_digit_boxes, recognize_digit,
                            _TEMPLATE_SIZE, _find_corner, draw_corner_debug, draw_led_debug,
                            draw_mute_debug, draw_digit_debug, _extract_digit_with_padding,
                            log_detection, log_issue_frame, close_log, reload_templates,
                            get_digit_1_issue, disable_logging)
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

_notifications_enabled = True

def send_notification(message, image_path=None):
    """Send iMessage notification with iCloud link to image."""
    if not _notifications_enabled or not IMESSAGE_RECIPIENT:
        return  # Notifications disabled
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
        self.last_corner_score = 0
        self.last_corner_result = None
        # LED history for glitch detection (A-A-?-?-?-A-A pattern, up to 3 glitch frames)
        self.led_history = []
        self.stable_led = None
        self.frame_history = []  # Store recent frames for glitch logging [(raw, display), ...]
        # Pending issues to log after display frame is ready
        self.pending_led_fail = False
        self.pending_mute_na = False
        self.pending_digit_1_issue = None  # Dict with score_1, score_7, gap
        self.pending_led_transition = None  # (from_led, to_led) for B1/B2 transitions
        self.prev_led_for_transition = None  # Track previous LED for transition detection
        # Context capture for ambiguous/low-conf readings
        # Stores: (issue_type, confidence, extra_info, debug_info, before_frames, issue_frame, after_frames)
        self.pending_context_capture = None
        self.context_after_frames = []  # Frames captured after issue
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
    if corner_result and corner_result[0] is not None:
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

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        return 0


def open_stream(source, width=640, height=480):
    """Open camera or RTSP stream."""
    if source.startswith('rtsp://') or source.startswith('http://'):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Set timeouts to prevent hanging (OpenCV 4.5+)
        if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10s open timeout
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)   # 5s read timeout
        else:
            # Older OpenCV - cap.read() may block indefinitely on network issues
            print("Warning: OpenCV < 4.5 - no read timeout protection", flush=True)
        is_stream = True
    else:
        cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        is_stream = False
    return cap, is_stream


def run_benchmark(cap, n_frames=1000):
    """Run pipeline benchmark for n_frames using real code behavior."""
    import time as time_module

    # Skip initial frames (with timeout protection)
    warmup_start = time_module.time()
    for _ in range(30):
        if time_module.time() - warmup_start > 15:
            print("Warning: Benchmark warmup timeout", flush=True)
            break
        cap.read()

    times = {
        'read_frame': [],
        'reader_read': [],
        'led_detect': [],
        'mute_detect': [],
        'corner_detect': [],
        'total': [],
    }

    # Use SegmentReader like real code
    reader = SegmentReader()
    frame_skip = 1  # Same as default in main()
    skipped_frames = 0
    processed_frames = 0
    last_corner_result = None  # Cache for skipped frames

    print(f'Benchmarking {n_frames} frames (real code behavior)...', flush=True)

    for i in range(n_frames):
        t_total_start = time_module.perf_counter()

        # Read frame
        t0 = time_module.perf_counter()
        ret, frame = cap.read()
        times['read_frame'].append(time_module.perf_counter() - t0)

        if not ret:
            break

        # SegmentReader.read() - includes panel detection, slant, gap, digit recognition
        # Also includes frame diff skip logic
        t0 = time_module.perf_counter()
        reading, cache_hit = reader.read(frame)
        times['reader_read'].append(time_module.perf_counter() - t0)

        if reader.frame_skipped:
            skipped_frames += 1
        else:
            processed_frames += 1

        # LED detection (on every frame like real code with frame_skip=1)
        if (i + 1) % frame_skip == 0 or i == 0:
            # Corner detection only when frame actually processed (not skipped by diff)
            if not reader.frame_skipped:
                t0 = time_module.perf_counter()
                corner_result, _ = _find_corner(frame, return_debug=True)
                times['corner_detect'].append(time_module.perf_counter() - t0)
                last_corner_result = corner_result
            else:
                corner_result = last_corner_result  # Use cached

            t0 = time_module.perf_counter()
            leds, _ = detect_button_leds(frame, reader.panel_rect, detection_method=reader.detection_method)
            times['led_detect'].append(time_module.perf_counter() - t0)

            # MUTE detection (reuse corner_result)
            # Pass None if corner_result has invalid coordinates (None, None, score)
            valid_corner = corner_result if (corner_result and corner_result[0] is not None) else None
            t0 = time_module.perf_counter()
            is_muted, _ = detect_red_button(frame, corner_result=valid_corner)
            times['mute_detect'].append(time_module.perf_counter() - t0)

        times['total'].append(time_module.perf_counter() - t_total_start)

        if (i+1) % 200 == 0:
            print(f'  {i+1}/{n_frames}...', flush=True)

    # Print results
    print(f'\n=== Timing Results ({len(times["total"])} frames) ===', flush=True)
    print(f'Skipped by diff: {skipped_frames}, Processed: {processed_frames}', flush=True)
    print(f'\n{"Stage":<20} {"Mean (ms)":>10} {"Std (ms)":>10} {"Min (ms)":>10} {"Max (ms)":>10}', flush=True)
    print('-' * 62, flush=True)

    for stage, t_list in times.items():
        if t_list:
            arr = np.array(t_list) * 1000  # to ms
            print(f'{stage:<20} {arr.mean():>10.2f} {arr.std():>10.2f} {arr.min():>10.2f} {arr.max():>10.2f}', flush=True)

    total_mean = np.mean(times['total']) * 1000
    print(f'\nOverall: {total_mean:.2f} ms/frame = {1000/total_mean:.1f} FPS', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Live 7-segment display reader')
    parser.add_argument('--width', '-W', type=int, default=640,
                        help='Frame width (default: 640)')
    parser.add_argument('--height', '-H', type=int, default=480,
                        help='Frame height (default: 480)')
    parser.add_argument('--skip', '-s', type=int, default=1,
                        help='Process every Nth frame (default: 1)')
    parser.add_argument('--display', action='store_true',
                        help='Show display window (default: headless)')
    parser.add_argument('--log', action='store_true',
                        help='Enable logging to files (default: no logging)')
    parser.add_argument('--benchmark', '-b', type=int, nargs='?', const=1000, metavar='N',
                        help='Run benchmark for N frames (default: 1000) and exit')
    parser.add_argument('--drain', type=int, default=2, metavar='N',
                        help='Drain N frames before each read for lower latency (default: 2)')
    args = parser.parse_args()

    # Set headless based on --display flag
    args.headless = not args.display

    if not args.log:
        disable_logging()
        global _notifications_enabled
        _notifications_enabled = False

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

    # Run benchmark if requested
    if args.benchmark:
        run_benchmark(cap, args.benchmark)
        cap.release()
        return

    if args.headless:
        print("Headless mode: Ctrl+C to quit", flush=True)
    else:
        print("Press 'q' quit, 'c' reset, 's' save, 'l#/r#' learn (e.g. l6, r8)", flush=True)
    print("-" * 40, flush=True)

    # Detect fresh every frame
    reader = SegmentReader()

    # Frame-to-frame state
    state = DemoState()

    # Frame count for first-frame detection
    frame_count = 0

    # Skip initial frames for RTSP streams (with timeout protection)
    if is_stream:
        warmup_start = time.time()
        warmup_timeout = 15  # Max 15s for warmup
        for _ in range(30):
            if time.time() - warmup_start > warmup_timeout:
                print("Warning: Warmup timeout, continuing...", flush=True)
                break
            cap.read()

    fail_count = 0
    max_fails = 50  # Reconnect after this many consecutive failures
    reconnect_delay = 2  # Seconds to wait before reconnecting
    pending_learn = None  # 'left' or 'right' when L or R pressed
    target_fps = 15  # Target frame rate to limit CPU usage
    frame_interval = 1.0 / target_fps
    last_frame_time = time.time()
    last_successful_frame = time.time()  # Watchdog timer
    watchdog_timeout = 30  # Force reconnect if no frames for 30 seconds

    while True:
        # Frame rate limiting to reduce CPU usage
        elapsed = time.time() - last_frame_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        last_frame_time = time.time()

        # Drain buffer for lower latency if requested (with timeout protection)
        if args.drain > 0:
            drain_start = time.time()
            for _ in range(args.drain):
                if time.time() - drain_start > 2:  # Max 2s for drain
                    break
                cap.grab()
        ret, frame = cap.read()
        if not ret or (is_stream and time.time() - last_successful_frame > watchdog_timeout):
            if not ret:
                fail_count += 1
            else:
                # Watchdog triggered - frame received but too slow
                print(f"Watchdog: No frames for {watchdog_timeout}s, reconnecting...", flush=True)
                fail_count = max_fails  # Force reconnect
            if is_stream and fail_count >= max_fails:
                print(f"Connection lost. Reconnecting in {reconnect_delay}s...", flush=True)
                cap.release()
                time.sleep(reconnect_delay)
                cap, _ = open_stream(camera, args.width, args.height)
                if cap.isOpened():
                    print("Reconnected successfully", flush=True)
                    # Skip initial frames (with timeout protection)
                    warmup_start = time.time()
                    for _ in range(30):
                        if time.time() - warmup_start > 15:
                            print("Warning: Reconnect warmup timeout", flush=True)
                            break
                        cap.read()
                    fail_count = 0
                else:
                    print("Reconnect failed, will retry...", flush=True)
                    fail_count = 0  # Reset to trigger another reconnect attempt after max_fails
                    time.sleep(reconnect_delay)  # Extra delay before next attempt
            continue

        fail_count = 0  # Reset on successful read
        last_successful_frame = time.time()  # Update watchdog
        frame_count += 1

        # Always run digit recognition (no caching of recognized digits)
        try:
            reading, cache_hit = reader.read(frame)
        except Exception as e:
            print(f"Error in reader.read: {e}", flush=True)
            reading, cache_hit = "XX", False

        # Debug: save frame when detecting wrong readings
        if reading in ["08", "P6", "6P", "01", "00", "09", "03", "18"]:
            debug_path = f'/tmp/debug_{reading}.png'
            if not cv2.imwrite(debug_path, frame):
                print(f"Warning: Failed to write {debug_path}", flush=True)

        # Corner detection (use cache when digit frame skipped, but always run if no cache)
        if not reader.frame_skipped or state.last_corner_result is None:
            try:
                corner_result, _ = _find_corner(frame, return_debug=True)
                corner_score = corner_result[2] if corner_result else 0
                state.last_corner_score = corner_score
                state.last_corner_result = corner_result
            except Exception as e:
                print(f"Error in corner detection: {e}", flush=True)
                corner_result = None
                corner_score = 0
        else:
            corner_result = state.last_corner_result
            corner_score = state.last_corner_score

        # LED detection (every frame)
        try:
            leds, _, led_debug_info = detect_button_leds(frame, reader.panel_rect, return_debug=True,
                                                          detection_method=reader.detection_method)
            lit_leds = [k for k, v in leds.items() if v]
            led_status = lit_leds[0] if lit_leds else "NA"
        except Exception as e:
            print(f"Error in LED detection: {e}", flush=True)
            led_status = "NA"
            led_debug_info = None

        # MUTE detection (every frame - only 0.3ms)
        # Pass None if corner_result has invalid coordinates (None, None, score)
        valid_corner = corner_result if (corner_result and corner_result[0] is not None) else None
        try:
            is_muted, _, mute_debug_info = detect_red_button(frame, return_debug=True, corner_result=valid_corner)
            mute_pixels = mute_debug_info.get('red_pixels', 0) if mute_debug_info else 0
            if mute_pixels > 100:
                mute_status = "MUTE_NA"
            else:
                mute_status = "MUTE" if is_muted else "UNMUTE"
        except Exception as e:
            print(f"Error in MUTE detection: {e}", flush=True)
            is_muted = False
            mute_debug_info = None
            mute_status = "UNMUTE"
            mute_pixels = 0

        # Store last LED and MUTE status
        state.last_led = led_status
        state.last_mute = mute_status
        state.last_led_debug = led_debug_info
        state.last_mute_debug = mute_debug_info

        # Log detection data
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
            dim_enhanced=reader.dim_enhanced,
            frame_skip=reader.frame_skipped,
            diff_edge=reader.frame_diff_edge,
            issue='led_fail' if led_status == 'NA' else ('mute_na' if mute_status == 'MUTE_NA' else None)
        )
        # Mark issues for logging after display frame is ready
        state.pending_led_fail = (led_status == 'NA')
        state.pending_mute_na = (mute_status == 'MUTE_NA')
        state.pending_digit_1_issue = get_digit_1_issue()

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
        if glitch and len(state.frame_history) >= glitch[0] + 3:
            glitch_count, stable_led, glitch_leds = glitch
            glitch_str = '->'.join(glitch_leds)
            # Create composite image: before -> glitch(es) -> after
            # Pattern A-A-B-A-A: before=-4, glitch=-3, after=-2
            # Pattern A-A-B-B-A-A: before=-5, glitches=-4,-3, after=-2
            before_idx = -(glitch_count + 3)
            after_idx = -2
            glitch_indices = [-(glitch_count + 2) + i for i in range(glitch_count)]

            frames_to_show = []
            labels = []
            # Before frame (stable)
            if abs(before_idx) <= len(state.frame_history):
                frames_to_show.append(state.frame_history[before_idx][0])
                labels.append(f'{stable_led} (before)')
            # Glitch frame(s)
            for i, idx in enumerate(glitch_indices):
                if abs(idx) <= len(state.frame_history):
                    frames_to_show.append(state.frame_history[idx][0])
                    labels.append(f'{glitch_leds[i]} (glitch)')
            # After frame (stable)
            if abs(after_idx) <= len(state.frame_history):
                frames_to_show.append(state.frame_history[after_idx][0])
                labels.append(f'{stable_led} (after)')

            # Create composite with labels
            if len(frames_to_show) >= 3:
                labeled_frames = []
                for frm, lbl in zip(frames_to_show, labels):
                    frm_copy = frm.copy()
                    cv2.putText(frm_copy, lbl, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    labeled_frames.append(frm_copy)
                composite = np.hstack(labeled_frames)
                saved_path = log_issue_frame(composite, 'led_glitch',
                               extra_info=f'{glitch_count}f_{glitch_str}_in_{stable_led}')
            else:
                saved_path = None
            # LED glitch logged to file, no stdout
            send_notification(f"LED GLITCH ({glitch_count}f): {stable_led} -> {glitch_str} -> {stable_led}", saved_path)

        if args.headless:
            # Headless mode: print when reading changes or every 1 minute
            now = time.time()
            time_since_print = now - state.last_time if state.last_time else 0
            if reading != state.last_print or mute_status != state.last_mute_print or time_since_print >= 60:
                print(f"Reading: {reading}  {led_status}  {mute_status}", flush=True)
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

            # Log digit "1" low confidence with "7" close (penalty issue)
            if state.pending_digit_1_issue:
                d1 = state.pending_digit_1_issue
                extra = f"1:{d1['score_1']:.2f}_7:{d1['score_7']:.2f}"
                path = log_issue_frame(frame, 'digit_1_penalty', extra_info=extra, debug_info=debug_info)
                send_notification(f"DIGIT 1 LOW: {d1['score_1']:.0%} (7 at {d1['score_7']:.0%})", path)
                state.pending_digit_1_issue = None

            # Log LED transition to B1/B2 (no display frame in headless mode)
            if state.pending_led_transition:
                from_led, to_led = state.pending_led_transition
                log_issue_frame(frame, 'led_transition', extra_info=f'{from_led}_to_{to_led}', debug_info=debug_info)
                state.pending_led_transition = None

            # Store current frame for glitch detection
            state.frame_history.append((frame.copy(), None))
            if len(state.frame_history) > 12:
                state.frame_history.pop(0)

            # Context capture: collect after-frames for pending context
            if state.pending_context_capture is not None:
                state.context_after_frames.append(frame.copy())
                if len(state.context_after_frames) >= 5:
                    # Have all frames - create composite
                    issue_type, confidence, extra_info, issue_debug, before_frames, issue_frame = state.pending_context_capture
                    composite_frames = []

                    # Add before frames with labels
                    for i, frm in enumerate(before_frames):
                        f = frm.copy()
                        cv2.putText(f, f'n-{len(before_frames)-i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        composite_frames.append(f)

                    # Add issue frame
                    f = issue_frame.copy()
                    cv2.putText(f, 'ISSUE', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    composite_frames.append(f)

                    # Add after frames
                    for i, frm in enumerate(state.context_after_frames):
                        f = frm.copy()
                        cv2.putText(f, f'n+{i+1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        composite_frames.append(f)

                    if len(composite_frames) >= 3:
                        scale = 0.33
                        resized = [cv2.resize(f, None, fx=scale, fy=scale) for f in composite_frames]
                        composite = np.hstack(resized)
                        log_issue_frame(composite, f'{issue_type}_ctx', confidence, extra_info, debug_info=issue_debug)
                        # Also save full-size raw issue frame
                        log_issue_frame(issue_frame, f'{issue_type}_raw', confidence, extra_info, debug_info=issue_debug)

                    state.pending_context_capture = None
                    state.context_after_frames = []

            # Start new context capture if issue detected
            elif reader.pending_issue:
                issue_type, confidence, extra_info = reader.pending_issue
                # Snapshot 5 frames before (from history, excluding current frame which is issue)
                before_frames = []
                history_len = len(state.frame_history)
                for i in range(max(0, history_len - 6), history_len - 1):  # -6 to -2 (5 frames before current)
                    before_frames.append(state.frame_history[i][0].copy())
                # Issue frame is the last one added
                issue_frame = state.frame_history[-1][0].copy() if history_len > 0 else frame.copy()
                state.pending_context_capture = (issue_type, confidence, extra_info, debug_info.copy(), before_frames, issue_frame)
                state.context_after_frames = []
                reader.clear_pending_issue()
        else:
            # Save original frame for learning (before overlays)
            original_frame = frame.copy()

            # Draw panel rectangle if detected (dashed when skipped, solid when active)
            if reader.panel_rect:
                x, y, w, h = reader.panel_rect
                if reader.frame_skipped:
                    # Draw dashed rectangle
                    dash_len = 10
                    for i in range(0, w, dash_len * 2):
                        cv2.line(frame, (x + i, y), (x + min(i + dash_len, w), y), (0, 255, 0), 2)
                        cv2.line(frame, (x + i, y + h), (x + min(i + dash_len, w), y + h), (0, 255, 0), 2)
                    for i in range(0, h, dash_len * 2):
                        cv2.line(frame, (x, y + i), (x, y + min(i + dash_len, h)), (0, 255, 0), 2)
                        cv2.line(frame, (x + w, y + i), (x + w, y + min(i + dash_len, h)), (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw corner search area and match location
            corner_result, corner_debug = _find_corner(frame, return_debug=True)
            draw_corner_debug(frame, corner_debug)

            # Draw LED zones and detection
            draw_led_debug(frame, led_debug_info)

            # Draw MUTE LED detection area
            draw_mute_debug(frame, mute_debug_info)

            # Draw LED and MUTE status at top left with semi-transparent background
            status_text = f"LED:{led_status}  {mute_status}"
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = 10
            # Draw semi-transparent black background (region-based for efficiency)
            bg_x1, bg_y1 = 5, 5
            bg_x2, bg_y2 = text_x + text_size[0] + 10, 40
            roi = frame[bg_y1:bg_y2, bg_x1:bg_x2]
            dark_roi = (roi * 0.5).astype(roi.dtype)
            frame[bg_y1:bg_y2, bg_x1:bg_x2] = dark_roi
            cv2.putText(frame, status_text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Display extracted digit images at top-right, gap debug to the left
            gap_debug_width = 0
            if reader.digit_debug:
                left_img = reader.digit_debug.get('left_img')
                right_img = reader.digit_debug.get('right_img')
                left_score, right_score = reader.last_scores if reader.last_scores else (0.0, 0.0)
                if reader.last_second:
                    (left_second, left_second_score), (right_second, right_second_score) = reader.last_second
                else:
                    left_second, left_second_score = 'X', 0.0
                    right_second, right_second_score = 'X', 0.0
                # Use raw digits for display (before PP/XX conversion)
                left_digit, right_digit = reader.raw_digits

                frame_w = frame.shape[1]
                x_offset = frame_w - 10  # Start from right edge
                img_y = 5  # Top of frame
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
                        # White/gray when skipped, cyan when active
                        if reader.frame_skipped:
                            cv2.putText(frame, label1, (x_offset, img_y+h+20), label_font, label_scale, (255, 255, 255), label_thick)
                            cv2.putText(frame, label2, (x_offset, img_y+h+42), label_font, label_scale, (100, 100, 100), label_thick)
                        else:
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
                        bg_height = 48
                        bg_x2, bg_y2 = x_offset + max_text_w + 3, min(frame.shape[0], img_y + h + bg_height)
                        if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                            roi = frame[bg_y1:bg_y2, bg_x1:bg_x2]
                            frame[bg_y1:bg_y2, bg_x1:bg_x2] = (roi * 0.5).astype(roi.dtype)
                        # White/gray when skipped, magenta when active
                        if reader.frame_skipped:
                            cv2.putText(frame, label1, (x_offset, img_y+h+20), label_font, label_scale, (255, 255, 255), label_thick)
                            cv2.putText(frame, label2, (x_offset, img_y+h+42), label_font, label_scale, (100, 100, 100), label_thick)
                        else:
                            cv2.putText(frame, label1, (x_offset, img_y+h+20), label_font, label_scale, (255, 0, 255), label_thick)
                            cv2.putText(frame, label2, (x_offset, img_y+h+42), label_font, label_scale, (255, 128, 255), label_thick)

                # Draw final reading below 2nd candidate, right-aligned with black background
                reading_y = img_y + h + 95
                reading_font_scale = 1.5
                reading_thick = 3
                reading_size = cv2.getTextSize(reading, cv2.FONT_HERSHEY_SIMPLEX, reading_font_scale, reading_thick)[0]
                reading_x = frame.shape[1] - reading_size[0] - 10  # Right-aligned
                # Semi-transparent black background
                bg_x1 = reading_x - 5
                bg_y1 = reading_y - reading_size[1] - 5
                bg_x2 = frame.shape[1] - 5
                bg_y2 = reading_y + 8
                if bg_x1 >= 0 and bg_y1 >= 0:
                    roi = frame[bg_y1:bg_y2, bg_x1:bg_x2]
                    frame[bg_y1:bg_y2, bg_x1:bg_x2] = (roi * 0.5).astype(roi.dtype)
                cv2.putText(frame, reading, (reading_x, reading_y), cv2.FONT_HERSHEY_SIMPLEX, reading_font_scale, (0, 255, 0), reading_thick)

                # Draw gap debug to the left of digit images
                corrected_img = reader.digit_debug.get('corrected_img')
                gap_x = reader.digit_debug.get('gap_x')
                if corrected_img is not None and gap_x is not None:
                    # Compute column brightness histogram
                    gray = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
                    col_sums = np.sum(gray, axis=0).astype(np.float64)
                    kernel = np.ones(5) / 5
                    smoothed = np.convolve(col_sums, kernel, mode='same')

                    # Create histogram (same width as corrected image)
                    corr_h, corr_w = corrected_img.shape[:2]
                    hist_h = 30
                    hist_img = np.zeros((hist_h, corr_w, 3), dtype=np.uint8)
                    max_val = max(smoothed) if max(smoothed) > 0 else 1
                    for gx in range(corr_w):
                        bar_h = int(smoothed[gx] / max_val * (hist_h - 2))
                        cv2.line(hist_img, (gx, hist_h), (gx, hist_h - bar_h), (80, 80, 80), 1)

                    # Draw gap line (yellow)
                    cv2.line(hist_img, (gap_x, 0), (gap_x, hist_h), (0, 255, 255), 2)

                    # Mark local minima
                    center = corr_w // 2
                    search_limit = int(corr_w * 0.15)
                    for i in range(max(1, center - search_limit), min(len(smoothed) - 1, center + search_limit)):
                        if smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]:
                            bar_h = int(smoothed[i] / max_val * (hist_h - 2))
                            cv2.circle(hist_img, (i, hist_h - bar_h), 2, (0, 255, 255), -1)

                    # Stack vertically (corrected image without gap line, histogram with gap line)
                    gap_debug_img = np.vstack([corrected_img, hist_img])
                    debug_h, debug_w = gap_debug_img.shape[:2]

                    # Place to the left of digit images
                    debug_x = x_offset - debug_w - 10
                    debug_y = img_y
                    if debug_x >= 0 and debug_y + debug_h <= frame.shape[0]:
                        frame[debug_y:debug_y+debug_h, debug_x:debug_x+debug_w] = gap_debug_img
                        cv2.rectangle(frame, (debug_x, debug_y), (debug_x+debug_w, debug_y+debug_h), (100, 100, 100), 1)

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

            # Log digit "1" low confidence with "7" close (penalty issue)
            if state.pending_digit_1_issue:
                d1 = state.pending_digit_1_issue
                extra = f"1:{d1['score_1']:.2f}_7:{d1['score_7']:.2f}"
                path = log_issue_frame(original_frame, 'digit_1_penalty', extra_info=extra, display_frame=frame, debug_info=debug_info)
                send_notification(f"DIGIT 1 LOW: {d1['score_1']:.0%} (7 at {d1['score_7']:.0%})", path)
                state.pending_digit_1_issue = None

            # Log LED transition to B1/B2 with both raw and display frames
            if state.pending_led_transition:
                from_led, to_led = state.pending_led_transition
                log_issue_frame(original_frame, 'led_transition', extra_info=f'{from_led}_to_{to_led}', display_frame=frame, debug_info=debug_info)
                state.pending_led_transition = None

            # Store current frames for glitch detection
            state.frame_history.append((original_frame.copy(), frame.copy()))
            if len(state.frame_history) > 12:
                state.frame_history.pop(0)

            # Context capture: collect after-frames for pending context
            if state.pending_context_capture is not None:
                state.context_after_frames.append(original_frame.copy())
                if len(state.context_after_frames) >= 5:
                    # Have all frames - create composite
                    issue_type, confidence, extra_info, issue_debug, before_frames, issue_frame, issue_display = state.pending_context_capture
                    composite_frames = []

                    # Add before frames with labels
                    for i, frm in enumerate(before_frames):
                        f = frm.copy()
                        cv2.putText(f, f'n-{len(before_frames)-i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        composite_frames.append(f)

                    # Add issue frame
                    f = issue_frame.copy()
                    cv2.putText(f, 'ISSUE', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    composite_frames.append(f)

                    # Add after frames
                    for i, frm in enumerate(state.context_after_frames):
                        f = frm.copy()
                        cv2.putText(f, f'n+{i+1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        composite_frames.append(f)

                    if len(composite_frames) >= 3:
                        scale = 0.33
                        resized = [cv2.resize(f, None, fx=scale, fy=scale) for f in composite_frames]
                        composite = np.hstack(resized)
                        log_issue_frame(composite, f'{issue_type}_ctx', confidence, extra_info, debug_info=issue_debug)
                        # Also save full-size issue frame (raw left, display right)
                        if issue_display is not None:
                            log_issue_frame(issue_frame, f'{issue_type}_display', confidence, extra_info,
                                            display_frame=issue_display, debug_info=issue_debug)

                    state.pending_context_capture = None
                    state.context_after_frames = []

            # Start new context capture if issue detected
            elif reader.pending_issue:
                issue_type, confidence, extra_info = reader.pending_issue
                # Snapshot 5 frames before (from history, excluding current frame which is issue)
                before_frames = []
                history_len = len(state.frame_history)
                for i in range(max(0, history_len - 6), history_len - 1):  # -6 to -2 (5 frames before current)
                    before_frames.append(state.frame_history[i][0].copy())
                # Issue frame is the last one added (both raw and display)
                issue_frame = state.frame_history[-1][0].copy() if history_len > 0 else original_frame.copy()
                issue_display = state.frame_history[-1][1].copy() if history_len > 0 and state.frame_history[-1][1] is not None else frame.copy()
                state.pending_context_capture = (issue_type, confidence, extra_info, debug_info.copy(), before_frames, issue_frame, issue_display)
                state.context_after_frames = []
                reader.clear_pending_issue()

            cv2.imshow('7-Segment Reader', frame)

            # Handle key presses (30ms wait reduces CPU usage)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                reader.reset_cache()
                print("Cache reset")
            elif key == ord('s'):
                # Save combined frame (raw + display side by side) with timestamp
                os.makedirs('logs', exist_ok=True)
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
