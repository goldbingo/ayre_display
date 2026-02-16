#!/usr/bin/env python3
"""Live camera demo for 7-segment display reader."""

import cv2
import sys
import os
import argparse
import time
import signal
import atexit

# Determine mode from argv (before argparse, needed for log dir and PID)
_display_mode = '--display' in sys.argv
_LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs' if _display_mode else os.path.join('logs', 'headless'))

# PID file management — kill stale instance of same mode, write new PID
_PID_FILE = f'/tmp/live_demo_{"display" if _display_mode else "headless"}.pid'

def _kill_stale_pid():
    """Kill previous instance of the same mode if still running."""
    if os.path.exists(_PID_FILE):
        try:
            old_pid = int(open(_PID_FILE).read().strip())
            # Verify it's actually a live_demo.py process
            cmd_out = os.popen(f'ps -p {old_pid} -o command=').read().strip()
            if 'live_demo.py' in cmd_out:
                os.kill(old_pid, signal.SIGKILL)
                time.sleep(0.5)
        except (ValueError, ProcessLookupError, OSError):
            pass
        try:
            os.remove(_PID_FILE)
        except OSError:
            pass

def _write_pid():
    """Write current PID to mode-specific PID file."""
    with open(_PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

def _cleanup_pid():
    """Remove PID file on exit."""
    try:
        os.remove(_PID_FILE)
    except OSError:
        pass

_kill_stale_pid()
_write_pid()
atexit.register(_cleanup_pid)

# Tee stdout/stderr to both terminal and log file
class _TeeWriter:
    """Write to both terminal and log file."""
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

if '--log' in sys.argv:
    os.makedirs(_LOG_DIR, exist_ok=True)
    _log_path = os.path.join(_LOG_DIR, 'live_demo.log')
    _log_file = open(_log_path, 'a')
    _log_file.write(f"\n{'='*60}\n")
    _log_file.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    _log_file.write(f"Mode: {'display' if _display_mode else 'headless'}\n")
    _log_file.write(f"{'='*60}\n")
    _log_file.flush()
    sys.stdout = _TeeWriter(sys.stdout, _log_file)
    sys.stderr = _TeeWriter(sys.stderr, _log_file)
import segment_reader
from segment_reader import (SegmentReader, FrameResult, detect_panel, detect_button_leds, detect_red_button,
                            correct_slant, find_digit_gap, define_digit_boxes, recognize_digit,
                            _TEMPLATE_SIZE, _find_corner, draw_corner_debug, draw_led_debug,
                            draw_mute_debug, draw_digit_debug, draw_display_overlay,
                            _extract_digit_with_padding,
                            log_detection, log_issue_frame, close_log, reload_templates,
                            get_digit_1_issue, disable_logging, set_undistort,
                            set_tracking, get_geometry, set_log_dir, get_noise_mean)
segment_reader.set_log_dir(_LOG_DIR)
import numpy as np

# MQTT support (optional - requires paho-mqtt)
_mqtt_client = None
_mqtt_base_topic = None
try:
    import paho.mqtt.client as mqtt
    _MQTT_AVAILABLE = True
except ImportError:
    _MQTT_AVAILABLE = False
import subprocess
import shutil
import json

# Code version (git short hash, computed once at startup)
try:
    _code_version = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'],
        cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    _code_version = 'unknown'

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
_notification_cooldown = {}  # Track last notification time per issue type
_notification_count = {}  # Track issue count during cooldown
_NOTIFICATION_COOLDOWN_SECONDS = 600  # 10 minutes cooldown

def send_notification(message, image_path=None, issue_type=None):
    """Send iMessage notification with iCloud link to image.

    Args:
        message: Notification message text
        image_path: Optional path to image to attach
        issue_type: Optional issue type for cooldown (e.g., 'led_fail', 'mute_na')
                   If provided, notifications of same type are rate-limited.
    """
    if not _notifications_enabled or not IMESSAGE_RECIPIENT:
        return  # Notifications disabled

    # Check cooldown for this issue type
    if issue_type:
        import time
        now = time.time()
        last_time = _notification_cooldown.get(issue_type, 0)
        if now - last_time < _NOTIFICATION_COOLDOWN_SECONDS:
            # Still in cooldown - increment counter
            _notification_count[issue_type] = _notification_count.get(issue_type, 0) + 1
            return
        # Cooldown expired - include count in message if there were suppressed notifications
        count = _notification_count.get(issue_type, 0)
        if count > 0:
            message = f"{message} (+{count} suppressed)"
            _notification_count[issue_type] = 0
        _notification_cooldown[issue_type] = now
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


def init_mqtt(config_path):
    """Initialize MQTT client from config file.

    Config JSON format:
    {
        "broker": "hostname:port",
        "base_topic": "home/ayre",
        "user": "username",       # optional
        "password": "password",   # optional
        "ca_cert": "/path/to.crt" # optional, for TLS
    }

    Returns True if connected successfully.
    """
    global _mqtt_client, _mqtt_base_topic

    if not _MQTT_AVAILABLE:
        print("Error: paho-mqtt not installed. Run: pip install paho-mqtt", flush=True)
        return False

    if not os.path.exists(config_path):
        print(f"Error: MQTT config not found: {config_path}", flush=True)
        return False

    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading MQTT config: {e}", flush=True)
        return False

    broker = config.get('broker', '')
    _mqtt_base_topic = config.get('base_topic', 'home/ayre')

    # Parse broker host:port
    if ':' in broker:
        host, port_str = broker.rsplit(':', 1)
        port = int(port_str)
    else:
        host = broker
        port = 1883  # Default MQTT port

    # Create client
    _mqtt_client = mqtt.Client()

    # Set Last Will - published by broker if we disconnect unexpectedly
    _mqtt_client.will_set(f"{_mqtt_base_topic}/status", "offline", retain=True)

    # Set credentials if provided
    if config.get('user'):
        _mqtt_client.username_pw_set(config['user'], config.get('password', ''))

    # Configure TLS if ca_cert provided
    if config.get('ca_cert'):
        _mqtt_client.tls_set(ca_certs=config['ca_cert'])

    # Connect
    try:
        _mqtt_client.connect(host, port, keepalive=60)
        _mqtt_client.loop_start()  # Start background thread for network loop
        # Publish online status (overwrites any stale offline from previous crash)
        _mqtt_client.publish(f"{_mqtt_base_topic}/status", "online", retain=True)
        print(f"MQTT connected to {host}:{port}, base topic: {_mqtt_base_topic}", flush=True)
        return True
    except Exception as e:
        print(f"MQTT connection failed: {e}", flush=True)
        _mqtt_client = None
        return False


_mqtt_last_reading = None
_mqtt_last_led = None
_mqtt_last_mute = None


def publish_mqtt(reading, led_status, mute_status, raw_reading=None, publish_all=False):
    """Publish current state to MQTT topics.

    Topics published:
    - {base}/7seg/num -> raw_reading (e.g., "07", "PP", before XX conversion)
    - {base}/vol -> reading (only 00-66)
    - {base}/source -> LED status (e.g., "S2")
    - {base}/mute -> "off" or "on"

    Args:
        raw_reading: Raw recognized digits before PP/XX conversion (for 7seg/num)
        publish_all: If True, publish all topics (heartbeat).
                     If False, only publish changed values.

    Skips publishing if reading contains "X" or LED is "NA" (invalid state).
    """
    global _mqtt_last_reading, _mqtt_last_led, _mqtt_last_mute

    if _mqtt_client is None:
        return

    # Don't publish invalid readings or LED failures
    if 'X' in reading or led_status == 'NA':
        return

    # Use raw_reading for num topic, fall back to reading
    num_value = raw_reading if raw_reading else reading

    # Convert mute status
    if mute_status == "UNMUTE":
        mute_val = "off"
    elif mute_status == "MUTE":
        mute_val = "on"
    else:
        mute_val = "unknown"

    try:
        base = _mqtt_base_topic

        # Publish reading if changed or publish_all
        if publish_all or num_value != _mqtt_last_reading:
            _mqtt_client.publish(f"{base}/7seg/num", num_value, retain=True)
            # Only publish valid volume readings (00-66) to vol topic
            if reading.isdigit() and int(reading) <= 66:
                _mqtt_client.publish(f"{base}/vol", reading, retain=True)
            _mqtt_last_reading = num_value

        # Publish LED status if changed or publish_all
        if publish_all or led_status != _mqtt_last_led:
            _mqtt_client.publish(f"{base}/source", led_status, retain=True)
            _mqtt_last_led = led_status

        # Publish mute status if changed or publish_all
        if publish_all or mute_val != _mqtt_last_mute:
            _mqtt_client.publish(f"{base}/mute", mute_val, retain=True)
            _mqtt_last_mute = mute_val
    except Exception as e:
        print(f"MQTT publish error: {e}", flush=True)


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
        # Reading history for glitch detection (A-B-A pattern)
        self.reading_history = []
        # Mute history for glitch detection (A-B-A pattern)
        self.mute_history = []
        self.stable_led = None
        self.frame_history = []  # Store recent frames for glitch logging [(raw, display, debug_info), ...]
        # Pending issues to log after display frame is ready
        self.pending_led_fail = False
        self.pending_mute_na = False
        self.prev_washout = False
        self.pending_washout_transition = None  # ('enter', noise_mean) or ('exit', noise_mean)
        self.pending_digit_1_issue = None  # Dict with score_1, score_7, gap
        self.pending_gap_ambiguous = None  # (confidence, extra_info, debug_info)
        self.pending_gap_wide_valley = None  # (confidence, extra_info, debug_info)
        self.pending_corner_low_score = None  # extra_info string or None
        # last_led_debug_info/last_mute_debug_info now cached in SegmentReader._last_led_debug/_last_mute_debug
        self.pending_mute_homography_outlier = None  # (dx, dy, dist) raw vs smoothed
        self.pending_led_transition = None  # (from_led, to_led) for B1/B2 transitions
        self.prev_led_for_transition = None  # Track previous LED for transition detection
        # Context capture for ambiguous/low-conf readings
        # Stores: (issue_type, confidence, extra_info, debug_info, before_frames, issue_frame, after_frames)
        self.pending_context_capture = None
        self.context_after_frames = []  # Frames captured after issue
        # Headless mode print state
        self.last_time = 0
        self.last_print = None
        self.last_led_print = ""
        self.last_mute_print = ""
        # FPS tracking
        self.fps_frame_count = 0
        self.fps_start_time = None


def build_debug_info(reader, reading, led_status, mute_status, corner_score,
                     led_debug_info, mute_debug_info, corner_result=None,
                     washout=False, cached_led_debug_info=None,
                     cached_mute_debug_info=None, noise_mean=None):
    """Build debug info dict for logging alongside captured frames."""
    info = {}
    info['code_version'] = _code_version
    info['frame_skipped'] = 'yes' if reader.frame_skipped else 'no'
    info['washout'] = 'yes' if washout else 'no'

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
    if corner_result and len(corner_result) > 3:
        info['corner_template'] = str(corner_result[3])


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
        if led_debug_info.get('predicted_b1_box'):
            info['predicted_b1_box'] = str(led_debug_info['predicted_b1_box'])
        # Per-button LED dot positions (landmarks for homography)
        led_dots = led_debug_info.get('led_dots')
        if led_dots:
            for name, val in led_dots.items():
                if name.startswith('_'):
                    continue
                pos, found = val
                info[f'led_dot_{name}'] = f'({int(pos[0])}, {int(pos[1])}) {found}'
    elif washout and cached_led_debug_info:
        # During washout, use cached LED region from last good frame
        info['led_region'] = str(cached_led_debug_info.get('region'))

    # MUTE info
    info['mute_status'] = mute_status
    if mute_debug_info:
        info['mute_region'] = str(mute_debug_info.get('region'))
        if mute_debug_info.get('led_center'):
            info['mute_led_center'] = str(mute_debug_info.get('led_center'))
        # Local contrast fields
        if mute_debug_info.get('mute_rr') is not None:
            info['mute_rr'] = f"{mute_debug_info['mute_rr']:.2f}"
            info['mute_re'] = f"{mute_debug_info.get('mute_re', 0):.1f}"
        if mute_debug_info.get('mute_h_age') is not None:
            info['mute_h_age'] = mute_debug_info['mute_h_age']
    elif washout and cached_mute_debug_info:
        # During washout, use cached mute region from last good frame
        info['mute_region'] = str(cached_mute_debug_info.get('region'))

    # Noise mean (washout detection value)
    if noise_mean is not None:
        info['noise_mean'] = f'{noise_mean:.1f}'

    # LED detection method
    if led_debug_info and led_debug_info.get('led_method'):
        info['led_method'] = led_debug_info['led_method']

    # Geometry method
    info['geo_method'] = reader.geo_method

    return info


def _build_overlay(original_frame, reader, corner_debug, led_debug_info,
                   mute_debug_info, led_status, mute_status, reading, washout,
                   cached_led_debug_info=None, cached_mute_debug_info=None,
                   corner_score=None):
    """Generate display overlay frame from reader state.

    Returns overlay BGR image (same size as original_frame).
    """
    _led_info = led_debug_info or (cached_led_debug_info if washout else None)
    _mute_info = mute_debug_info or (cached_mute_debug_info if washout else None)

    if reader.panel_rect and reader.digit_debug:
        left_img = reader.digit_debug.get('left_img')
        right_img = reader.digit_debug.get('right_img')
        corrected_img = reader.digit_debug.get('corrected_img')
        gap_x_vis = reader.digit_debug.get('gap_x')
        left_match = reader.digit_debug.get('left_match')
        right_match = reader.digit_debug.get('right_match')
        left_score, right_score = reader.last_scores if reader.last_scores else (0.0, 0.0)
        if reader.last_second:
            (left_second, left_second_score), (right_second, right_second_score) = reader.last_second
        else:
            left_second, left_second_score = 'X', 0.0
            right_second, right_second_score = 'X', 0.0
        left_digit, right_digit = reader.raw_digits

        overlay = draw_display_overlay(
            original_frame, reader.panel_rect, corrected_img, gap_x_vis,
            left_img, right_img,
            left_digit, right_digit, left_score, right_score,
            left_match, right_match,
            left_second, left_second_score,
            right_second, right_second_score,
            reading, led_status, mute_status,
            corner_debug=corner_debug,
            corner_score=corner_score,
            led_debug_info=_led_info,
            mute_debug_info=_mute_info,
            frame_skipped=reader.frame_skipped,
            washout=washout)
    else:
        # Minimal overlay (no panel or digit data)
        overlay = original_frame.copy()
        if reader.panel_rect:
            x, y, w, h = reader.panel_rect
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        draw_corner_debug(overlay, corner_debug, corner_score=corner_score)
        draw_led_debug(overlay, _led_info, dashed=washout)
        draw_mute_debug(overlay, _mute_info, dashed=washout)
        status_text = f"LED:{led_status}  {mute_status}"
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        bg_x2 = 10 + text_size[0] + 10
        roi = overlay[25:60, 5:bg_x2]
        overlay[25:60, 5:bg_x2] = (roi * 0.5).astype(roi.dtype)
        cv2.putText(overlay, status_text, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return overlay


def _build_composite(frame_history, indices, labels, debug_prefix_labels):
    """Build raw + overlay composites from frame_history.

    Args:
        frame_history: List of (raw, overlay, debug_info) tuples.
        indices: List of negative indices into frame_history.
        labels: List of label strings to draw on each frame.
        debug_prefix_labels: List of prefix strings for debug dict keys.

    Returns:
        (raw_composite, overlay_composite, debug_dict) or (None, None, {})
        if not enough frames.
    """
    raw_frames, overlay_frames = [], []
    debug = {}
    for idx, label, prefix in zip(indices, labels, debug_prefix_labels):
        if abs(idx) <= len(frame_history):
            raw, overlay, info = frame_history[idx]
            r = raw.copy()
            cv2.putText(r, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            raw_frames.append(r)
            o = overlay.copy()
            cv2.putText(o, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            overlay_frames.append(o)
            if info:
                for key, val in info.items():
                    debug[f'{prefix}/{key}'] = val

    if len(raw_frames) < 3:
        return None, None, {}
    raw_comp = np.hstack(raw_frames)
    ovl_comp = np.hstack(overlay_frames)
    return raw_comp, ovl_comp, debug


def _capture_issue(raw_frame, overlay_frame, issue_type, debug_info,
                   confidence=0, extra_info=None):
    """Save issue capture with raw and overlay as separate files.

    Returns path of saved raw file, or None.
    """
    raw_path = log_issue_frame(raw_frame, issue_type, confidence=confidence,
                               extra_info=extra_info, debug_info=debug_info)
    if raw_path and overlay_frame is not None:
        # Save overlay alongside raw (same basename with _overlay suffix)
        ovl_path = raw_path.replace('.png', '_overlay.png')
        cv2.imwrite(ovl_path, overlay_frame)
    return raw_path


def _capture_composite(raw_composite, overlay_composite, issue_type,
                       debug_info, confidence=0, extra_info=None):
    """Save composite capture: raw file + overlay file.

    Returns path of saved raw file, or None.
    """
    raw_path = log_issue_frame(raw_composite, issue_type, confidence=confidence,
                               extra_info=extra_info, debug_info=debug_info)
    if raw_path and overlay_composite is not None:
        ovl_path = raw_path.replace('.png', '_overlay.png')
        cv2.imwrite(ovl_path, overlay_composite)
    return raw_path


def _start_context_capture(state, frame, debug_info, issue_type, confidence, extra_info):
    """Snapshot before-frames from history and start collecting after-frames."""
    history_len = len(state.frame_history)
    before_raw = []
    before_ovl = []
    for i in range(max(0, history_len - 6), history_len - 1):
        before_raw.append(state.frame_history[i][0].copy())
        ovl = state.frame_history[i][1]
        before_ovl.append(ovl.copy() if ovl is not None else state.frame_history[i][0].copy())
    # Issue frame is the last one added
    if history_len > 0:
        issue_raw = state.frame_history[-1][0].copy()
        issue_ovl = state.frame_history[-1][1]
        issue_ovl = issue_ovl.copy() if issue_ovl is not None else issue_raw.copy()
    else:
        issue_raw = frame.copy()
        issue_ovl = frame.copy()
    state.pending_context_capture = (issue_type, confidence, extra_info, debug_info.copy(),
                                     before_raw, before_ovl, issue_raw, issue_ovl)
    state.context_after_frames = []


def _finish_context_capture(state):
    """Build and save context composite from collected frames."""
    issue_type, confidence, extra_info, issue_debug, before_raw, before_ovl, issue_raw, issue_ovl = state.pending_context_capture

    def _label_frames(frames, labels):
        result = []
        for frm, lbl in zip(frames, labels):
            f = frm.copy()
            color = (0, 0, 255) if lbl == 'ISSUE' else (0, 255, 255)
            cv2.putText(f, lbl, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            result.append(f)
        return result

    n_before = len(before_raw)
    labels_before = [f'n-{n_before - i}' for i in range(n_before)]
    labels_after = [f'n+{i+1}' for i in range(len(state.context_after_frames))]
    all_labels = labels_before + ['ISSUE'] + labels_after

    all_raw = before_raw + [issue_raw] + [af[0] for af in state.context_after_frames]
    all_ovl = before_ovl + [issue_ovl] + [af[1] for af in state.context_after_frames]

    if len(all_raw) >= 3:
        raw_labeled = _label_frames(all_raw, all_labels)
        ovl_labeled = _label_frames(all_ovl, all_labels)
        raw_comp = np.hstack(raw_labeled)
        ovl_comp = np.hstack(ovl_labeled)
        _capture_composite(raw_comp, ovl_comp, issue_type, issue_debug,
                           confidence=confidence, extra_info=extra_info)

    state.pending_context_capture = None
    state.context_after_frames = []


def _draw_dashed_rect_magenta(frame, x, y, w, h, thickness=2, dash_len=10):
    """Draw a dashed magenta rectangle."""
    MAGENTA = (255, 0, 255)
    for i in range(0, w, dash_len * 2):
        cv2.line(frame, (x + i, y), (x + min(i + dash_len, w), y), MAGENTA, thickness)
        cv2.line(frame, (x + i, y + h), (x + min(i + dash_len, w), y + h), MAGENTA, thickness)
    for i in range(0, h, dash_len * 2):
        cv2.line(frame, (x, y + i), (x, y + min(i + dash_len, h)), MAGENTA, thickness)
        cv2.line(frame, (x + w, y + i), (x + w, y + min(i + dash_len, h)), MAGENTA, thickness)


def draw_alignment_overlay(frame, ref):
    """Draw magenta alignment overlay from calibration reference.

    Args:
        frame: BGR frame to draw on (modified in place).
        ref: Dict with corner_xy, button_centers, panel_rect, button_rects.
    """
    MAGENTA = (255, 0, 255)
    THICK = 2

    # Dashed panel rectangle
    if 'panel_rect' in ref:
        px, py, pw, ph = ref['panel_rect']
        _draw_dashed_rect_magenta(frame, px, py, pw, ph, THICK)
        cv2.putText(frame, "REF", (px, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAGENTA, THICK)

    # Dashed mute region
    if 'mute_region' in ref:
        ml, mt, mr, mb = ref['mute_region']
        _draw_dashed_rect_magenta(frame, ml, mt, mr - ml, mb - mt, THICK)
        cv2.putText(frame, "MUTE", (ml, mt - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, MAGENTA, THICK)

    # Corner crosshair
    if 'corner_xy' in ref:
        cx, cy = ref['corner_xy']
        arm = 14
        cv2.line(frame, (cx - arm, cy), (cx + arm, cy), MAGENTA, THICK)
        cv2.line(frame, (cx, cy - arm), (cx, cy + arm), MAGENTA, THICK)

    # Button rectangles (dashed, same size as detected button boxes)
    if 'button_rects' in ref:
        for name, (bx, by, bw, bh) in ref['button_rects'].items():
            _draw_dashed_rect_magenta(frame, bx, by, bw, bh, THICK)
            cv2.putText(frame, name, (bx + 5, by + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, MAGENTA, THICK)
    elif 'button_centers' in ref:
        for name, (bx, by) in ref['button_centers'].items():
            half = 8
            _draw_dashed_rect_magenta(frame, bx - half, by - half, half * 2, half * 2, THICK)
            cv2.putText(frame, name, (bx + half + 3, by - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, MAGENTA, THICK)


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
    parser.add_argument('--target-fps', '-t', type=float, default=None, metavar='FPS',
                        help='Target processing fps (auto-skip to achieve target)')
    parser.add_argument('--display', action='store_true',
                        help='Show display window (default: headless)')
    parser.add_argument('--log', action='store_true',
                        help='Enable logging to files (default: no logging)')
    parser.add_argument('--benchmark', '-b', type=int, nargs='?', const=1000, metavar='N',
                        help='Run benchmark for N frames (default: 1000) and exit')
    parser.add_argument('--drain', type=int, default=0, metavar='N',
                        help='Drain N frames before each read for lower latency (default: 0)')
    parser.add_argument('--mqtt-config', type=str, metavar='PATH',
                        help='Path to MQTT config JSON (enables MQTT publishing)')
    parser.add_argument('--undistort', action='store_true',
                        help='Enable de-rotation, scale normalization, and bidirectional gap detection')
    parser.add_argument('--track', action='store_true',
                        help='Enable landmark tracking (reuse golden positions when landmarks disappear)')
    parser.add_argument('--camera', type=str, metavar='URL',
                        help='Camera URL or device index (e.g., rtsp://... or 0)')
    parser.add_argument('--camera-file', type=str, metavar='PATH',
                        help='File containing camera URL (default: webcam.link)')
    args = parser.parse_args()

    # Check for conflicting options
    if args.target_fps and args.skip > 1:
        print("Error: --target-fps and --skip are mutually exclusive. Use one or the other.", flush=True)
        sys.exit(1)
    if args.camera and args.camera_file:
        print("Error: --camera and --camera-file are mutually exclusive. Use one or the other.", flush=True)
        sys.exit(1)

    # Set headless based on --display flag
    args.headless = not args.display

    if args.undistort:
        set_undistort(True)

    if args.track:
        set_tracking(True)

    if not args.log:
        disable_logging()
        global _notifications_enabled
        _notifications_enabled = False

    # Initialize MQTT if config provided
    if args.mqtt_config:
        if not init_mqtt(args.mqtt_config):
            print("Warning: MQTT initialization failed, continuing without MQTT", flush=True)

    # Resolve camera source: --camera URL, --camera-file PATH, or default webcam.link
    if args.camera:
        camera = args.camera
    else:
        camera_file = args.camera_file or os.path.join(os.path.dirname(__file__), 'webcam.link')
        if not os.path.exists(camera_file):
            print(f"Error: {camera_file} not found. Use --camera URL or --camera-file PATH.", flush=True)
            sys.exit(1)
        with open(camera_file, 'r') as f:
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
        keys_hint = "Press 'q' quit, 'c' reset, 's' save, 'l#/r#' learn (e.g. l6, r8)"
        if args.track:
            keys_hint += ", 'a' align"
        print(keys_hint, flush=True)
    print("-" * 40, flush=True)

    # Alignment overlay state (toggled by 'a' in --display --track mode)
    _align_mode = False

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
    last_processed_time = time.time()  # For --target-fps time-based skipping
    last_successful_frame = time.time()  # Watchdog timer
    watchdog_timeout = 30  # Force reconnect if no frames for 30 seconds

    # Adaptive --target-fps control
    if args.target_fps:
        from collections import deque
        process_timestamps = deque(maxlen=10)  # Track last 10 processed frames
        adaptive_interval = 1.0 / args.target_fps  # Start with ideal interval
        min_interval = 0.01  # Don't go faster than 100 fps
        max_interval = 10.0  # Don't go slower than 0.1 fps

    while True:
        frame_count += 1

        # Skip logic: either count-based (--skip) or adaptive time-based (--target-fps)
        should_skip = False
        if args.target_fps:
            # Adaptive time-based skip
            should_skip = (time.time() - last_processed_time) < adaptive_interval
        elif args.skip > 1:
            # Count-based skip: process every Nth frame
            should_skip = (frame_count % args.skip != 0)

        if should_skip:
            if not cap.grab():
                fail_count += 1
                if is_stream and fail_count >= max_fails:
                    print(f"Connection lost (skip). Reconnecting in {reconnect_delay}s...", flush=True)
                    cap.release()
                    time.sleep(reconnect_delay)
                    cap, _ = open_stream(camera, args.width, args.height)
                    # Wait for first valid frame
                    warmup_start = time.time()
                    for _ in range(150):
                        if time.time() - warmup_start > 15:
                            break
                        ret_w, _ = cap.read()
                        if ret_w:
                            break
                    fail_count = 0
                    last_successful_frame = time.time()
            else:
                fail_count = 0
                last_successful_frame = time.time()
            continue

        last_processed_time = time.time()  # Update for --target-fps

        # Adaptive fps control: measure actual fps and adjust interval
        if args.target_fps:
            process_timestamps.append(last_processed_time)
            if len(process_timestamps) >= 5:
                # Calculate actual fps from recent timestamps
                time_span = process_timestamps[-1] - process_timestamps[0]
                if time_span > 0:
                    actual_fps = (len(process_timestamps) - 1) / time_span
                    # Adjust interval proportionally to error
                    error_ratio = actual_fps / args.target_fps
                    # Smooth adjustment: move 20% toward ideal
                    adaptive_interval = adaptive_interval * (0.8 + 0.2 * error_ratio)
                    adaptive_interval = max(min_interval, min(max_interval, adaptive_interval))

        # Process frame: drain accumulated frames, then read
        # --target-fps: grab() already clears buffer during skips, use args.drain only
        # --skip N: drain max(skip-1, args.drain) to clear accumulated frames
        if args.target_fps:
            drain_count = args.drain
        elif args.skip > 1:
            drain_count = max(args.skip - 1, args.drain)
        else:
            drain_count = args.drain
        if drain_count > 0:
            drain_start = time.time()
            for _ in range(drain_count):
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
                    # Wait for first valid frame (h264 needs IDR/keyframe after reconnect)
                    warmup_start = time.time()
                    got_valid = False
                    for _ in range(150):  # Up to 150 attempts (~10s at stream rate)
                        if time.time() - warmup_start > 15:
                            break
                        ret_w, _ = cap.read()
                        if ret_w:
                            got_valid = True
                            break
                    if got_valid:
                        # Drain a few more to clear buffered frames
                        for _ in range(10):
                            cap.read()
                        fail_count = 0
                        last_successful_frame = time.time()
                        print(f"Stream recovered after {time.time() - warmup_start:.1f}s", flush=True)
                    else:
                        print(f"Reconnect warmup timeout ({time.time() - warmup_start:.1f}s, no valid frames)", flush=True)
                        cap.release()
                        fail_count = 0
                        time.sleep(reconnect_delay * 2)  # Extra delay before retry
                        cap, _ = open_stream(camera, args.width, args.height)
                else:
                    print("Reconnect failed, will retry...", flush=True)
                    fail_count = 0  # Reset to trigger another reconnect attempt after max_fails
                    time.sleep(reconnect_delay * 2)  # Extra delay before next attempt
            continue

        fail_count = 0  # Reset on successful read
        last_successful_frame = time.time()  # Update watchdog
        state.fps_frame_count += 1  # Count frames for fps calculation

        # Increment homography age counter (#72 A/B logging)
        get_geometry().increment_homography_age()

        # Unified detection: digits + corner + LED + mute (#79)
        proc_start = time.perf_counter()  # Start timing for proc_ms
        try:
            result = reader.detect(frame)
        except Exception as e:
            print(f"Error in reader.detect: {e}", flush=True)
            result = FrameResult(reading="XX", cache_hit=False,
                                 led_status="NA", mute_status="UNMUTE")

        # Extract local variables from FrameResult
        reading = result.reading
        cache_hit = result.cache_hit
        corner_result = result.corner_result
        corner_debug = result.corner_debug
        corner_score = corner_result[2] if corner_result else 0
        corner_tmpl_idx = corner_result[3] if corner_result and len(corner_result) > 3 else None
        led_status = result.led_status
        led_debug_info = result.led_debug_info
        mute_status = result.mute_status
        mute_debug_info = result.mute_debug_info
        _noise_mean = result.noise_mean
        washout = result.washout

        # Debug: save frame when detecting wrong readings
        if reading in ["08", "P6", "6P", "01", "00", "09", "03", "18"]:
            debug_path = f'/tmp/debug_{reading}.png'
            if not cv2.imwrite(debug_path, frame):
                print(f"Warning: Failed to write {debug_path}", flush=True)

        # Mark low-score corner for deferred capture (after overlay is built)
        if corner_score and 0.85 <= corner_score < 0.93:
            state.pending_corner_low_score = f's{corner_score:.3f}_t{corner_tmpl_idx}'
        else:
            state.pending_corner_low_score = None

        # Detect washout transitions (logged in CSV only, no image capture)
        if washout and not state.prev_washout:
            state.pending_washout_transition = ('enter', _noise_mean)
        elif not washout and state.prev_washout:
            state.pending_washout_transition = ('exit', _noise_mean)
        state.prev_washout = washout

        # Store last LED and MUTE status
        state.last_led = led_status
        state.last_mute = mute_status
        state.last_led_debug = led_debug_info
        state.last_mute_debug = mute_debug_info

        # Gap diagnostics (after LED/mute so debug_info has current-frame values)
        if not reader.frame_skipped and reader.panel_rect and reader.gap_x:
            px, py, pw, ph = reader.panel_rect
            panel_img = frame[py:py+ph, px:px+pw]
            corrected, _, _ = correct_slant(panel_img, 8.0)
            gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
            col_sums = np.sum(gray, axis=0).astype(np.float64)
            kernel = np.ones(5) / 5
            smoothed = np.convolve(col_sums, kernel, mode='same')
            search_lo = int(len(smoothed) * 0.35)
            search_hi = int(len(smoothed) * 0.65)
            mins = []
            for i in range(search_lo + 1, search_hi - 1):
                if smoothed[i] <= smoothed[i-1] and smoothed[i] <= smoothed[i+1]:
                    mins.append((i, smoothed[i]))
            if len(mins) >= 2:
                mins.sort(key=lambda m: m[1])
                best_x, best_val = mins[0]
                second_x, second_val = None, None
                for mx, mv in mins[1:]:
                    if abs(mx - best_x) >= 10:
                        second_x, second_val = mx, mv
                        break
                if second_val is not None:
                    valley_ratio = second_val / best_val if best_val > 0 else 999
                    if valley_ratio < 1.2:
                        if not washout:
                            gap_debug = build_debug_info(reader, reading,
                                led_status, mute_status, corner_score,
                                led_debug_info, mute_debug_info,
                                corner_result=corner_result)
                            state.pending_gap_ambiguous = (valley_ratio,
                                f'v1_x{best_x}_v2_x{second_x}_ratio{valley_ratio:.2f}', gap_debug)
            # U-shaped valley check
            gx = reader.gap_x
            if 0 < gx < len(smoothed) - 1:
                min_val = smoothed[gx]
                threshold = min_val * 1.10
                left_edge = gx
                while left_edge > 0 and smoothed[left_edge - 1] <= threshold:
                    left_edge -= 1
                right_edge = gx
                while right_edge < len(smoothed) - 1 and smoothed[right_edge + 1] <= threshold:
                    right_edge += 1
                valley_width = right_edge - left_edge
                if valley_width >= 9:
                    left_peak_x = int(np.argmax(smoothed[:gx]))
                    right_peak_x = gx + int(np.argmax(smoothed[gx:]))
                    left_digit = reading[0] if len(reading) >= 2 else ''
                    right_digit = reading[1] if len(reading) >= 2 else ''
                    valley_center = (left_edge + right_edge) / 2
                    dist_to_left = abs(valley_center - left_peak_x)
                    dist_to_right = abs(valley_center - right_peak_x)
                    is_expected = False
                    if right_digit in ('1', '3', '7'):
                        is_expected = True
                    elif left_digit == 'P' and dist_to_right < dist_to_left:
                        is_expected = True
                    # Skip during rapid display transitions (3+ different values in recent history)
                    rh = state.reading_history
                    recent = set(rh[-3:]) | {reading} if len(rh) >= 3 else set()
                    is_transition = len(recent) >= 3
                    if not is_expected and not is_transition and not washout:
                        gap_debug = build_debug_info(reader, reading,
                            led_status, mute_status, corner_score,
                            led_debug_info, mute_debug_info,
                            corner_result=corner_result)
                        state.pending_gap_wide_valley = (valley_width / 20.0,
                            f'gap{gx}_w{valley_width}_L{left_edge}_R{right_edge}', gap_debug)

        # Calculate processing time
        proc_ms = (time.perf_counter() - proc_start) * 1000

        # Log detection data
        left_score, right_score = reader.last_scores
        led_method = led_debug_info.get('led_method') if led_debug_info else None
        # Local contrast fields
        mc_rr = mute_debug_info.get('mute_rr') if mute_debug_info else None
        mc_re = mute_debug_info.get('mute_re') if mute_debug_info else None
        mc_gr = mute_debug_info.get('mute_gr') if mute_debug_info else None
        mc_led_r = mute_debug_info.get('mute_led_r') if mute_debug_info else None
        mc_ref_r = mute_debug_info.get('mute_ref_r') if mute_debug_info else None
        mc_led_sx = mute_debug_info.get('mute_led_sx') if mute_debug_info else None
        mc_led_sy = mute_debug_info.get('mute_led_sy') if mute_debug_info else None
        mc_led_rx = mute_debug_info.get('mute_led_rx') if mute_debug_info else None
        mc_led_ry = mute_debug_info.get('mute_led_ry') if mute_debug_info else None
        mc_h_age = mute_debug_info.get('mute_h_age') if mute_debug_info else None
        log_detection(
            panel_rect=reader.panel_rect,
            gap_x=reader.gap_x,
            left_score=left_score,
            right_score=right_score,
            reading=reading,
            led_status=led_status,
            corner_score=corner_score,
            corner_tmpl=corner_tmpl_idx,
            detection_method=reader.detection_method,
            mute_status=mute_status,
            dim_enhanced=reader.dim_enhanced,
            frame_skip=reader.frame_skipped,
            diff_edge=reader.frame_diff_edge,
            diff_mode='3ch',
            led_method=led_method,
            proc_ms=proc_ms,
            issue='led_fail' if led_status == 'NA' and not washout else ('mute_na' if mute_status == 'MUTE_NA' and not washout else None),
            geo_method=reader.geo_method,
            geo_scale=reader.geo_scale,
            geo_rotation=reader.geo_rotation,
            undistorted=reader.undistorted,
            noise_mean=_noise_mean,
            mute_rr=mc_rr,
            mute_re=mc_re,
            mute_gr=mc_gr,
            mute_led_r=mc_led_r,
            mute_ref_r=mc_ref_r,
            mute_h_age=mc_h_age,
        )
        # Detect mute homography outlier: raw projection jumps >5px from smoothed
        state.pending_mute_homography_outlier = None
        if mc_led_sx is not None and mc_led_rx is not None and mc_h_age == 0:
            h_dx = mc_led_rx - mc_led_sx
            h_dy = mc_led_ry - mc_led_sy
            h_dist = (h_dx**2 + h_dy**2) ** 0.5
            if h_dist > 5:
                state.pending_mute_homography_outlier = (h_dx, h_dy, h_dist)

        # Mark issues for logging after display frame is ready
        state.pending_led_fail = (led_status == 'NA' and not washout)
        state.pending_mute_na = (mute_status == 'MUTE_NA' and not washout)
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

        # Store current frame with overlay so frame_history aligns with led_history
        frame_info = build_debug_info(reader, reading, led_status, mute_status,
                                      corner_score, led_debug_info, mute_debug_info,
                                      corner_result=corner_result, washout=washout,
                                      cached_led_debug_info=result.last_led_debug,
                                      cached_mute_debug_info=result.last_mute_debug,
                                      noise_mean=_noise_mean)
        frame_overlay = _build_overlay(frame, reader, corner_debug,
                                       led_debug_info, mute_debug_info,
                                       led_status, mute_status, reading, washout,
                                       cached_led_debug_info=result.last_led_debug,
                                       cached_mute_debug_info=result.last_mute_debug,
                                       corner_score=corner_score)
        state.frame_history.append((frame.copy(), frame_overlay, frame_info))
        if len(state.frame_history) > 12:
            state.frame_history.pop(0)

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
            before_idx = -(glitch_count + 3)
            after_idx = -2
            glitch_indices = [-(glitch_count + 2) + i for i in range(glitch_count)]

            indices = [before_idx] + glitch_indices + [after_idx]
            labels = ([f'{stable_led} (before)'] +
                      [f'{glitch_leds[i]} (glitch)' for i in range(glitch_count)] +
                      [f'{stable_led} (after)'])
            prefixes = ([f'{stable_led}_before'] +
                        [f'{glitch_leds[i]}_glitch{i+1}' for i in range(glitch_count)] +
                        [f'{stable_led}_after'])

            raw_comp, ovl_comp, comp_debug = _build_composite(
                state.frame_history, indices, labels, prefixes)
            if raw_comp is not None:
                comp_debug['glitch_type'] = 'led'
                comp_debug['glitch_count'] = glitch_count
                comp_debug['stable_led'] = stable_led
                comp_debug['glitch_leds'] = glitch_str
                saved_path = _capture_composite(raw_comp, ovl_comp, 'led_glitch',
                               comp_debug, extra_info=f'{glitch_count}f_{glitch_str}_in_{stable_led}')
            else:
                saved_path = None
            send_notification(f"LED GLITCH ({glitch_count}f): {stable_led} -> {glitch_str} -> {stable_led}", saved_path, issue_type='led_glitch')

        # Track reading history for glitch detection (A-B-A pattern)
        state.reading_history.append(reading)
        if len(state.reading_history) > 8:
            state.reading_history.pop(0)

        # Detect reading glitch: A-B-A where B != 'XX' (single-frame wrong reading)
        # Skip if reading was transitioning (before frames differ from stable)
        rh = state.reading_history
        if (len(rh) >= 3 and rh[-3] == rh[-1] and rh[-3] != rh[-2]
                and rh[-2] != 'XX'
                and not (len(rh) >= 4 and rh[-4] != rh[-3])):
            glitch_reading = rh[-2]
            stable_reading = rh[-1]

            indices = [-5, -4, -3, -2, -1]
            labels = [stable_reading, stable_reading, stable_reading,
                      f'>>>{glitch_reading}<<<', stable_reading]
            prefixes = ['before3', 'before2', 'before1', 'glitch', 'after']

            raw_comp, ovl_comp, rg_debug = _build_composite(
                state.frame_history, indices, labels, prefixes)
            if raw_comp is not None:
                rg_debug['glitch_type'] = 'reading'
                rg_debug['stable_reading'] = stable_reading
                rg_debug['glitch_reading'] = glitch_reading
                saved_path = _capture_composite(raw_comp, ovl_comp, 'reading_glitch',
                               rg_debug, extra_info=f'{stable_reading}_to_{glitch_reading}')
            else:
                saved_path = None
            send_notification(f"READING GLITCH: {stable_reading} -> {glitch_reading} -> {stable_reading}",
                              saved_path, issue_type='reading_glitch')

        # Track mute history for glitch detection (A-B-A pattern)
        state.mute_history.append(mute_status)
        if len(state.mute_history) > 8:
            state.mute_history.pop(0)

        # Detect mute glitch: A-B-A (single-frame mute status flip)
        mh = state.mute_history
        if (len(mh) >= 3 and mh[-3] == mh[-1] and mh[-3] != mh[-2]
                and mh[-2] != 'MUTE_NA'):
            glitch_mute = mh[-2]
            stable_mute = mh[-1]

            indices = [-5, -4, -3, -2, -1]
            labels = [stable_mute, stable_mute, stable_mute,
                      f'>>>{glitch_mute}<<<', stable_mute]
            prefixes = ['before3', 'before2', 'before1', 'glitch', 'after']

            raw_comp, ovl_comp, mg_debug = _build_composite(
                state.frame_history, indices, labels, prefixes)
            if raw_comp is not None:
                mg_debug['glitch_type'] = 'mute'
                mg_debug['stable_mute'] = stable_mute
                mg_debug['glitch_mute'] = glitch_mute
                saved_path = _capture_composite(raw_comp, ovl_comp, 'mute_glitch',
                               mg_debug, extra_info=f'{stable_mute}_to_{glitch_mute}')
            else:
                saved_path = None
            send_notification(f"MUTE GLITCH: {stable_mute} -> {glitch_mute} -> {stable_mute}",
                              saved_path, issue_type='mute_glitch')

        # Print status when reading changes or every 1 minute
        now = time.time()
        time_since_print = now - state.last_time if state.last_time else 0
        state_changed = reading != state.last_print or led_status != state.last_led_print or mute_status != state.last_mute_print
        minute_elapsed = time_since_print >= 60
        if state_changed or minute_elapsed:
            # Calculate fps only on minute interval (not on reading changes)
            fps_str = ""
            if minute_elapsed and state.fps_start_time is not None:
                elapsed = now - state.fps_start_time
                if elapsed > 0:
                    fps = state.fps_frame_count / elapsed
                    fps_str = f"  [{fps:.2f} fps]"
                # Reset fps counter after minute print
                state.fps_frame_count = 0
                state.fps_start_time = now
            elif state.fps_start_time is None:
                state.fps_start_time = now
            print(f"Reading: {reading}  {led_status}  {mute_status}{fps_str}", flush=True)
            # Touch heartbeat file for watchdog
            try:
                open('/tmp/live_demo_heartbeat', 'a').close()
                os.utime('/tmp/live_demo_heartbeat', None)
            except:
                pass
            # Publish to MQTT: all values on minute heartbeat, only changed values otherwise
            raw_reading = ''.join(reader.raw_digits) if reader.raw_digits else reading
            publish_mqtt(reading, led_status, mute_status, raw_reading=raw_reading, publish_all=minute_elapsed)
            state.last_print = reading
            state.last_led_print = led_status
            state.last_mute_print = mute_status
            state.last_time = now

        if args.headless:
            # Build debug info for logging (headless mode)
            debug_info = build_debug_info(reader, reading, led_status, mute_status,
                                          corner_score, led_debug_info, mute_debug_info,
                                          corner_result=corner_result, washout=washout,
                                          cached_led_debug_info=result.last_led_debug,
                                          cached_mute_debug_info=result.last_mute_debug,
                                          noise_mean=_noise_mean)

            # Clear washout transition (no image capture needed)
            if state.pending_washout_transition:
                state.pending_washout_transition = None

            # Get overlay from frame_history (generated earlier)
            overlay_frame = state.frame_history[-1][1] if state.frame_history else None

            # Log LED fail
            if state.pending_led_fail:
                path = _capture_issue(frame, overlay_frame, 'led_fail', debug_info)
                send_notification(f"LED FAIL: detection failed", path, issue_type='led_fail')
                state.pending_led_fail = False

            # Log MUTE_NA
            if state.pending_mute_na:
                path = _capture_issue(frame, overlay_frame, 'mute_na', debug_info, extra_info='washout')
                send_notification(f"MUTE_NA: abnormal", path, issue_type='mute_na')
                state.pending_mute_na = False

            # Log digit "1" low confidence with "7" close (penalty issue)
            if state.pending_digit_1_issue:
                d1 = state.pending_digit_1_issue
                extra = f"1:{d1['score_1']:.2f}_7:{d1['score_7']:.2f}"
                path = _capture_issue(frame, overlay_frame, 'digit_1_penalty', debug_info, extra_info=extra)
                send_notification(f"DIGIT 1 LOW: {d1['score_1']:.0%} (7 at {d1['score_7']:.0%})", path, issue_type='digit_1_low')
                state.pending_digit_1_issue = None

            # Log LED transition to B1/B2
            if state.pending_led_transition:
                from_led, to_led = state.pending_led_transition
                _capture_issue(frame, overlay_frame, 'led_transition', debug_info, extra_info=f'{from_led}_to_{to_led}')
                state.pending_led_transition = None

            # Log mute homography outlier (raw vs smoothed >5px)
            if state.pending_mute_homography_outlier:
                h_dx, h_dy, h_dist = state.pending_mute_homography_outlier
                _capture_issue(frame, overlay_frame, 'mute_homography_outlier', debug_info, extra_info=f'd{h_dist:.1f}_dx{h_dx:.1f}_dy{h_dy:.1f}')
                state.pending_mute_homography_outlier = None

            # Log corner low score
            if state.pending_corner_low_score:
                _capture_issue(frame, overlay_frame, 'corner_low_score', debug_info, extra_info=state.pending_corner_low_score)
                state.pending_corner_low_score = None

            # Log gap issues
            if state.pending_gap_ambiguous:
                conf, extra, gd = state.pending_gap_ambiguous
                _capture_issue(frame, overlay_frame, 'gap_ambiguous', gd, confidence=conf, extra_info=extra)
                state.pending_gap_ambiguous = None
            if state.pending_gap_wide_valley:
                conf, extra, gd = state.pending_gap_wide_valley
                _capture_issue(frame, overlay_frame, 'gap_wide_valley', gd, confidence=conf, extra_info=extra)
                state.pending_gap_wide_valley = None

            # Context capture: collect after-frames for pending context
            if state.pending_context_capture is not None:
                state.context_after_frames.append((frame.copy(), overlay_frame.copy() if overlay_frame is not None else frame.copy()))
                if len(state.context_after_frames) >= 5:
                    _finish_context_capture(state)

            # Start new context capture if issue detected
            elif reader.pending_issue:
                issue_type, confidence, extra_info = reader.pending_issue
                # Skip ambiguous/low_conf during digit↔PP transitions
                rh = state.reading_history
                has_pp = 'PP' in rh[-4:]
                has_digit = any(r not in ('PP', 'XX') for r in rh[-4:])
                skip = (issue_type in ('ambiguous', 'low_conf') and has_pp and has_digit)
                if not skip:
                    _start_context_capture(state, frame, debug_info, issue_type, confidence, extra_info)
                    state.context_after_frames = []
                reader.clear_pending_issue()
        else:
            # Save original frame for learning (before overlays)
            original_frame = frame.copy()

            # Use overlay already generated and stored in frame_history
            frame[:] = state.frame_history[-1][1]

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
                                          corner_result=corner_result, washout=washout,
                                          cached_led_debug_info=result.last_led_debug,
                                          cached_mute_debug_info=result.last_mute_debug,
                                          noise_mean=_noise_mean)

            # Clear washout transition (no image capture needed)
            if state.pending_washout_transition:
                state.pending_washout_transition = None

            # Get overlay from frame_history (generated earlier)
            overlay_frame = state.frame_history[-1][1] if state.frame_history else frame

            # Log LED fail
            if state.pending_led_fail:
                path = _capture_issue(original_frame, overlay_frame, 'led_fail', debug_info)
                send_notification(f"LED FAIL: detection failed", path, issue_type='led_fail')
                state.pending_led_fail = False

            # Log MUTE_NA
            if state.pending_mute_na:
                path = _capture_issue(original_frame, overlay_frame, 'mute_na', debug_info, extra_info='washout')
                send_notification(f"MUTE_NA: abnormal", path, issue_type='mute_na')
                state.pending_mute_na = False

            # Log digit "1" low confidence with "7" close (penalty issue)
            if state.pending_digit_1_issue:
                d1 = state.pending_digit_1_issue
                extra = f"1:{d1['score_1']:.2f}_7:{d1['score_7']:.2f}"
                path = _capture_issue(original_frame, overlay_frame, 'digit_1_penalty', debug_info, extra_info=extra)
                send_notification(f"DIGIT 1 LOW: {d1['score_1']:.0%} (7 at {d1['score_7']:.0%})", path, issue_type='digit_1_low')
                state.pending_digit_1_issue = None

            # Log LED transition to B1/B2
            if state.pending_led_transition:
                from_led, to_led = state.pending_led_transition
                _capture_issue(original_frame, overlay_frame, 'led_transition', debug_info, extra_info=f'{from_led}_to_{to_led}')
                state.pending_led_transition = None

            # Log mute homography outlier (raw vs smoothed >5px)
            if state.pending_mute_homography_outlier:
                h_dx, h_dy, h_dist = state.pending_mute_homography_outlier
                _capture_issue(original_frame, overlay_frame, 'mute_homography_outlier', debug_info, extra_info=f'd{h_dist:.1f}_dx{h_dx:.1f}_dy{h_dy:.1f}')
                state.pending_mute_homography_outlier = None

            # Log gap issues
            if state.pending_gap_ambiguous:
                conf, extra, gd = state.pending_gap_ambiguous
                _capture_issue(original_frame, overlay_frame, 'gap_ambiguous', gd, confidence=conf, extra_info=extra)
                state.pending_gap_ambiguous = None
            if state.pending_gap_wide_valley:
                conf, extra, gd = state.pending_gap_wide_valley
                _capture_issue(original_frame, overlay_frame, 'gap_wide_valley', gd, confidence=conf, extra_info=extra)
                state.pending_gap_wide_valley = None

            # Log corner low score
            if state.pending_corner_low_score:
                _capture_issue(original_frame, overlay_frame, 'corner_low_score', debug_info, extra_info=state.pending_corner_low_score)
                state.pending_corner_low_score = None

            # Context capture: collect after-frames for pending context
            if state.pending_context_capture is not None:
                state.context_after_frames.append((original_frame.copy(), overlay_frame.copy() if overlay_frame is not None else original_frame.copy()))
                if len(state.context_after_frames) >= 5:
                    _finish_context_capture(state)

            # Start new context capture if issue detected
            elif reader.pending_issue:
                issue_type, confidence, extra_info = reader.pending_issue
                # Skip ambiguous/low_conf during digit↔PP transitions
                rh = state.reading_history
                has_pp = 'PP' in rh[-4:]
                has_digit = any(r not in ('PP', 'XX') for r in rh[-4:])
                skip = (issue_type in ('ambiguous', 'low_conf') and has_pp and has_digit)
                if not skip:
                    _start_context_capture(state, frame, debug_info, issue_type, confidence, extra_info)
                    state.context_after_frames = []
                reader.clear_pending_issue()

            # Alignment overlay (magenta reference positions)
            if _align_mode and args.track:
                ref = get_geometry().get_calibration_ref()
                if ref:
                    draw_alignment_overlay(frame, ref)


            cv2.imshow('7-Segment Reader', frame)

            # Handle key presses (30ms wait reduces CPU usage)
            key = cv2.waitKey(30) & 0xFF

            # Check if window was closed - recreate it and continue
            if cv2.getWindowProperty('7-Segment Reader', cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed, recreating...", flush=True)
                cv2.destroyAllWindows()
                cv2.namedWindow('7-Segment Reader', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('7-Segment Reader', 640, 480)

            if key == ord('q'):
                break
            elif key == ord('a') and args.track:
                _align_mode = not _align_mode
                print(f"Align mode {'ON' if _align_mode else 'OFF'}", flush=True)
            elif key == ord('c'):
                reader.reset_cache()
                print("Cache reset")
            elif key == ord('s'):
                # Save raw + overlay as separate files with timestamp
                os.makedirs(_LOG_DIR, exist_ok=True)
                timestamp_str = time.strftime('%Y%m%d_%H%M%S')
                raw_filename = os.path.join(_LOG_DIR, f'manual_{timestamp_str}.png')
                ovl_filename = os.path.join(_LOG_DIR, f'manual_{timestamp_str}_overlay.png')
                cv2.imwrite(raw_filename, original_frame)
                cv2.imwrite(ovl_filename, frame)
                # Save debug text file
                txt_filename = os.path.join(_LOG_DIR, f'manual_{timestamp_str}.txt')
                with open(txt_filename, 'w') as f:
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Manual save (s key)\n\n")
                    for key_name, value in debug_info.items():
                        f.write(f"{key_name}: {value}\n")
                print(f"Saved {raw_filename} + {ovl_filename}")
                # Show on display
                save_frame = frame.copy()
                cv2.rectangle(save_frame, (10, 10), (350, 45), (0, 200, 0), -1)
                cv2.putText(save_frame, f"Saved: {raw_filename}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
    # Clean up MQTT
    if _mqtt_client is not None:
        _mqtt_client.publish(f"{_mqtt_base_topic}/status", "offline", retain=True)
        _mqtt_client.loop_stop()
        _mqtt_client.disconnect()
    print("Done")


if __name__ == "__main__":
    main()
