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
                            _find_dial, draw_dial_debug, draw_led_debug, draw_mute_debug, draw_digit_debug)
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
    parser.add_argument('--camera', '-c', type=str, default='0',
                        help='Camera index or RTSP URL (default: 0)')
    parser.add_argument('--width', '-W', type=int, default=640,
                        help='Frame width (default: 640)')
    parser.add_argument('--height', '-H', type=int, default=480,
                        help='Frame height (default: 480)')
    parser.add_argument('--skip', '-s', type=int, default=3,
                        help='Process every Nth frame (default: 3, use 1 to disable)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without display (print readings to console)')
    args = parser.parse_args()

    # Open camera or stream
    cap, is_stream = open_stream(args.camera, args.width, args.height)
    if is_stream:
        print(f"Opening stream: {args.camera.split('@')[-1]}", flush=True)  # Hide credentials
    else:
        print(f"Opening camera: {args.camera}", flush=True)

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
                cap, _ = open_stream(args.camera, args.width, args.height)
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
            led_status = lit_leds[0] if lit_leds else "None"
            # Detect MUTE (red button)
            is_muted, _, mute_debug_info = detect_red_button(frame, return_debug=True)
            mute_status = "MUTE" if is_muted else "UNMUTE"
        else:
            led_status = getattr(main, '_last_led', "None")
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

            # Draw dial search area and match location
            dial_result, dial_debug = _find_dial(frame, return_debug=True)
            draw_dial_debug(frame, dial_debug)

            # Draw LED zones and detection
            draw_led_debug(frame, led_debug_info)

            # Draw MUTE LED detection area
            draw_mute_debug(frame, mute_debug_info)

            # Draw digit search areas and match positions
            draw_digit_debug(frame, reader.panel_rect, reader.digit_debug)

            # Draw reading on frame with confidence scores
            status = "HIT" if cache_hit else "MISS"
            left_score, right_score = reader.last_scores
            text = f"[{status}] {reading} ({int(left_score*100)}%,{int(right_score*100)}%)  LED:{led_status}  {mute_status}"

            # Draw text with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            cv2.rectangle(frame, (10, 10), (20 + text_w, 20 + text_h + baseline), (0, 0, 0), -1)
            cv2.putText(frame, text, (15, 15 + text_h), font, font_scale, (0, 255, 255), thickness)

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
                        fname = learn_digit(frame, reader.panel_rect, 'left', correct[0])
                        if fname:
                            learned.append(f"L:{correct[0]}={fname}")
                    if correct[1] != reading[1]:
                        fname = learn_digit(frame, reader.panel_rect, 'right', correct[1])
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
