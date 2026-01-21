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
                            _extract_digit_with_padding, log_detection, log_issue_frame, close_log)
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
        if reading in ["08", "P6", "6P", "01", "00", "09", "03", "18"]:
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

            # Auto-save frame when transitioning to MUTE (only on state change)
            prev_mute = getattr(main, '_prev_mute_state', False)
            if is_muted and not prev_mute:
                now = time.time()
                mute_filename = f'/tmp/mute_{int(now)}_{frame_count}.png'
                cv2.imwrite(mute_filename, frame)
                print(f"MUTE detected! Saved: {mute_filename}", flush=True)
            main._prev_mute_state = is_muted

            # Auto-save frame when transitioning to B1 (only on state change)
            prev_led = getattr(main, '_prev_led_state', None)
            if led_status == "B1" and prev_led != "B1":
                now = time.time()
                b1_filename = f'/tmp/b1_{int(now)}_{frame_count}.png'
                cv2.imwrite(b1_filename, frame)
                print(f"B1 detected! Saved: {b1_filename}", flush=True)
            main._prev_led_state = led_status
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

        # Log detection data (every processed frame)
        if frame_count % frame_skip == 0:
            corner_result, _ = _find_corner(frame, return_debug=True)
            corner_score = corner_result[2] if corner_result else 0
            left_score, right_score = reader.last_scores
            log_detection(
                panel_rect=reader.panel_rect,
                gap_x=reader.gap_x,
                left_score=left_score,
                right_score=right_score,
                reading=reading,
                led_status=led_status,
                corner_score=corner_score,
                detection_method=reader.detection_method,
                issue='led_fail' if led_status == 'NA' else None
            )
            # Save frame if LED detection failed
            if led_status == 'NA':
                log_issue_frame(frame, 'led_fail')

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
            # Draw semi-transparent black background
            bg_x1, bg_y1 = 5, 5
            bg_x2, bg_y2 = text_x + text_size[0] + 10, 40
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, status_text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Display extracted digit images at top-right of frame with labels
            if reader.digit_debug:
                left_img = reader.digit_debug.get('left_img')
                right_img = reader.digit_debug.get('right_img')
                left_score, right_score = reader.last_scores
                (left_second, left_second_score), (right_second, right_second_score) = reader.last_second
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
                    # Limit image width to fit on screen (max 200px each)
                    max_img_w = 200
                    if right_img.shape[1] > max_img_w:
                        scale = max_img_w / right_img.shape[1]
                        right_img = cv2.resize(right_img, None, fx=scale, fy=scale)
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
                    bg_x1, bg_y1 = x_offset - 3, img_y + h + 3
                    bg_x2, bg_y2 = x_offset + max_text_w + 3, img_y + h + 48
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
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
                    # Limit image width to fit on screen (max 200px each)
                    max_img_w = 200
                    if left_img.shape[1] > max_img_w:
                        scale = max_img_w / left_img.shape[1]
                        left_img = cv2.resize(left_img, None, fx=scale, fy=scale)
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
                    bg_x1, bg_y1 = x_offset - 3, img_y + h + 3
                    bg_x2, bg_y2 = x_offset + max_text_w + 3, img_y + h + 48
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                    cv2.putText(frame, label1, (x_offset, img_y+h+20), label_font, label_scale, (255, 0, 255), label_thick)
                    cv2.putText(frame, label2, (x_offset, img_y+h+42), label_font, label_scale, (255, 128, 255), label_thick)

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
                    cv2.putText(learn_frame, prompt, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
                    cv2.putText(learn_frame, msg, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow('7-Segment Reader', learn_frame)
                    cv2.waitKey(2000)  # Show for 2 seconds

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
