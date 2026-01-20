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
                            _extract_digit_with_padding)
import numpy as np

# Use TCP transport for RTSP streams
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def learn_digit(frame, panel_rect, position, correct_digit):
    """Save a digit from the current frame as a new template.

    Args:
        frame: Current video frame
        panel_rect: Panel rectangle (x, y, w, h)
        position: 'left' or 'right'
        correct_digit: The correct digit character (0-9, P)

    Returns:
        filename of saved template, or None if failed
    """
    if panel_rect is None:
        return None

    x, y, w, h = panel_rect
    panel_img = frame[y:y+h, x:x+w]
    corrected, _, _ = correct_slant(panel_img, 8.0)
    gap_x, _ = find_digit_gap(corrected)
    left_box, right_box, _ = define_digit_boxes(corrected, gap_x)

    if position == 'left':
        bx, by, bw, bh = left_box
    else:
        bx, by, bw, bh = right_box

    digit_img = corrected[by:by+bh, bx:bx+bw]
    gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, _TEMPLATE_SIZE)

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
    cv2.imwrite(filepath, resized)

    # Add to in-memory cache immediately
    if segment_reader._digit_templates is not None:
        if correct_digit not in segment_reader._digit_templates:
            segment_reader._digit_templates[correct_digit] = []
        segment_reader._digit_templates[correct_digit].append(resized)

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
        print("Press 'q' quit, 'r' reset, 's' save, 'l' learn", flush=True)
    print("-" * 40, flush=True)

    # No cache, no auto-learn - detect fresh every frame
    reader = SegmentReader(auto_learn=False)

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
        if reading in ["08", "P6", "6P", "01", "00", "09", "03"]:
            cv2.imwrite(f'/tmp/debug_{reading}.png', frame)

        # Frame skipping for LED/MUTE detection only
        if frame_count % frame_skip == 0:
            # Detect LED
            leds, _, led_debug_info = detect_button_leds(frame, reader.panel_rect, return_debug=True)
            lit_leds = [k for k, v in leds.items() if v]
            led_status = lit_leds[0] if lit_leds else "NA"
            # Detect MUTE (red button)
            is_muted, _, mute_debug_info = detect_red_button(frame, return_debug=True)
            mute_status = "MUTE" if is_muted else "UNMUTE"

            # Auto-save frame when MUTE detected (with 5 second cooldown)
            if is_muted:
                last_save_time = getattr(main, '_last_mute_save_time', 0)
                now = time.time()
                if now - last_save_time >= 5:  # 5 second cooldown
                    mute_filename = f'/tmp/mute_{int(now)}_{frame_count}.png'
                    cv2.imwrite(mute_filename, frame)
                    print(f"MUTE detected! Saved: {mute_filename}", flush=True)
                    main._last_mute_save_time = now

            # Auto-save frame when B1 detected (with 5 second cooldown)
            if led_status == "B1":
                last_b1_save_time = getattr(main, '_last_b1_save_time', 0)
                now = time.time()
                if now - last_b1_save_time >= 5:  # 5 second cooldown
                    b1_filename = f'/tmp/b1_{int(now)}_{frame_count}.png'
                    cv2.imwrite(b1_filename, frame)
                    print(f"B1 detected! Saved: {b1_filename}", flush=True)
                    main._last_b1_save_time = now
        else:
            led_status = getattr(main, '_last_led', "NA")
            mute_status = getattr(main, '_last_mute', "UNMUTE")
            led_debug_info = getattr(main, '_last_led_debug', None)
            mute_debug_info = getattr(main, '_last_mute_debug', None)

        # Store last LED and MUTE status
        main._last_led = led_status
        main._last_mute = mute_status
        main._last_led_debug = led_debug_info
        main._last_mute_debug = mute_debug_info

        if args.headless:
            # Headless mode: print when reading changes or every minute
            now = time.time()
            last_time = getattr(main, '_last_time', 0)
            last_print = getattr(main, '_last_print', None)
            last_mute_print = getattr(main, '_last_mute_print', "")

            if reading != last_print or mute_status != last_mute_print or (now - last_time) >= 60:
                print(f"Reading: {reading}  LED: {led_status}  {mute_status}", flush=True)
                main._last_print = reading
                main._last_mute_print = mute_status
                main._last_time = now
        else:
            # Save original frame for learning (before overlays)
            original_frame = frame.copy()

            # Draw panel rectangle if detected
            if reader.panel_rect:
                x, y, w, h = reader.panel_rect
                # Replace panel area with slant-corrected image
                panel_img = frame[y:y+h, x:x+w]
                corrected, _, _ = correct_slant(panel_img, 8.0)
                # Resize corrected to match original panel size and overlay
                corrected_resized = cv2.resize(corrected, (w, h))
                frame[y:y+h, x:x+w] = corrected_resized
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw corner search area and match location
            corner_result, corner_debug = _find_corner(frame, return_debug=True)
            draw_corner_debug(frame, corner_debug)

            # Draw LED zones and detection
            draw_led_debug(frame, led_debug_info)

            # Draw MUTE LED detection area
            draw_mute_debug(frame, mute_debug_info)

            # Draw digit search areas and match positions
            draw_digit_debug(frame, reader.panel_rect, reader.digit_debug)

            # Draw gap line and save debug image once
            if reader.panel_rect and reader.gap_x is not None:
                px, py, pw, ph = reader.panel_rect
                # Scale gap_x using corrected_size from digit_debug (same coords as gap_x)
                corrected_size = reader.digit_debug.get('corrected_size') if reader.digit_debug else None
                if corrected_size:
                    cw, ch = corrected_size
                    gap_frame_x = px + int(reader.gap_x * pw / cw)
                else:
                    gap_frame_x = px + reader.gap_x  # Fallback: no scaling
                cv2.line(frame, (gap_frame_x, py), (gap_frame_x, py + ph), (0, 165, 255), 1)
                cv2.putText(frame, "gap", (gap_frame_x + 2, py + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)

                # Save gap debug image once
                if not getattr(main, '_gap_debug_saved', False):
                    panel_img = original_frame[py:py+ph, px:px+pw]
                    corrected, _, _ = correct_slant(panel_img, 8.0)
                    gap_x, gap_debug = find_digit_gap(corrected, debug=True)
                    if gap_debug is not None:
                        cv2.imwrite('/tmp/gap_debug.png', gap_debug)
                        print(f"Saved gap debug to /tmp/gap_debug.png (gap_x={gap_x})", flush=True)
                    main._gap_debug_saved = True

            # Draw reading on frame with confidence scores
            if not reader.auto_learn:
                status = "DIS"  # Cache disabled
            else:
                status = "HIT" if cache_hit else "MISS"
            left_score, right_score = reader.last_scores
            (left_second, left_second_score), (right_second, right_second_score) = reader.last_second
            text1 = f"[{status}] {reading} ({int(left_score*100):2d}%,{int(right_score*100):2d}%)  LED:{led_status}  {mute_status}"
            text2 = f"[2nd] {left_second}{right_second} ({int(left_second_score*100):2d}%,{int(right_second_score*100):2d}%)"

            # Draw text with fixed-width characters (monospace simulation)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            char_width = 14  # Fixed width per character
            char_height = 20
            line_spacing = 5

            max_len = max(len(text1), len(text2))
            box_w = max_len * char_width + 10
            box_h = 2 * char_height + line_spacing + 10

            cv2.rectangle(frame, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)

            # Draw text1 character by character
            for i, c in enumerate(text1):
                cv2.putText(frame, c, (15 + i * char_width, 10 + char_height), font, font_scale, (0, 255, 255), thickness)
            # Draw text2 character by character
            for i, c in enumerate(text2):
                cv2.putText(frame, c, (15 + i * char_width, 10 + char_height + line_spacing + char_height), font, font_scale, (128, 255, 255), thickness)

            # Show frame
            cv2.imshow('7-Segment Reader', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                reader.reset_cache()
                print("Cache reset")
            elif key == ord('s'):
                # Save current frame for debugging
                cv2.imwrite('debug_live_frame.png', frame)
                print("Saved debug_live_frame.png")
            elif key == ord('l'):
                # Learn mode: type digits, press Enter to confirm
                print(f"LEARN MODE - Current: {reading} - Type correct digits, Enter to confirm, ESC to cancel", flush=True)
                correct = ""
                while True:
                    # Show learning prompt on frame
                    learn_frame = frame.copy()
                    display = correct + "_" * (2 - len(correct))
                    prompt = f"Correct: {display}  (Enter=OK, ESC=Cancel)"
                    cv2.rectangle(learn_frame, (10, 50), (450, 90), (0, 0, 200), -1)
                    cv2.putText(learn_frame, prompt, (15, 80), font, 0.7, (255, 255, 255), 2)
                    cv2.imshow('7-Segment Reader', learn_frame)

                    k = cv2.waitKey(0) & 0xFF
                    if k == 27:  # ESC to cancel
                        print("Cancelled", flush=True)
                        correct = ""
                        break
                    elif k in [13, 10]:  # Enter/Return to confirm
                        if len(correct) == 2:
                            break
                        else:
                            print("Need 2 digits", flush=True)
                    elif k == 8 or k == 127:  # Backspace/Delete
                        correct = correct[:-1]
                    else:
                        c = chr(k).upper()
                        if c in '0123456789P' and len(correct) < 2:
                            correct += c

                if len(correct) == 2:
                    learned = []
                    if correct[0] != reading[0]:
                        fname = learn_digit(original_frame, reader.panel_rect, 'left', correct[0])
                        if fname:
                            learned.append(f"L:{correct[0]}={fname}")
                    if correct[1] != reading[1]:
                        fname = learn_digit(original_frame, reader.panel_rect, 'right', correct[1])
                        if fname:
                            learned.append(f"R:{correct[1]}={fname}")

                    # Show result on screen for 2 seconds
                    learn_frame = frame.copy()
                    if learned:
                        msg = "Learned: " + ", ".join(learned)
                        color = (0, 200, 0)  # Green
                        print(f"Learned: {', '.join(learned)}", flush=True)
                    else:
                        msg = "No differences to learn"
                        color = (0, 165, 255)  # Orange
                        print(msg, flush=True)
                    cv2.rectangle(learn_frame, (10, 50), (600, 90), color, -1)
                    cv2.putText(learn_frame, msg, (15, 80), font, 0.6, (255, 255, 255), 2)
                    cv2.imshow('7-Segment Reader', learn_frame)
                    cv2.waitKey(2000)  # Show for 2 seconds

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
