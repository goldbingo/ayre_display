#!/usr/bin/env python3
"""Timing analysis for the segment reader pipeline."""

import cv2
import time
import numpy as np
from segment_reader import (
    SegmentReader, detect_panel, correct_slant, find_digit_gap,
    define_digit_boxes, recognize_digit_template, _extract_digit_with_padding,
    match_single_template
)


def analyze_pipeline_timing(frame, iterations=100):
    """Analyze timing of each pipeline step.

    Args:
        frame: BGR image from camera/file
        iterations: Number of iterations for averaging

    Returns:
        dict of step timings in milliseconds
    """
    timings = {
        'detect_panel': [],
        'correct_slant': [],
        'find_digit_gap': [],
        'define_digit_boxes': [],
        'extract_digits': [],
        'recognize_left': [],
        'recognize_right': [],
        'single_template_2x': [],  # Time to check 2 single templates (1st + 2nd)
        'total': [],
        'total_single': [],  # Total with single-template quick-check
    }

    # Track best templates for single-template quick-check mode
    left_best_templates = None  # ((digit, idx), (digit, idx))
    right_best_templates = None

    for i in range(iterations):
        total_start = time.perf_counter()

        # Step 1: Panel detection
        t0 = time.perf_counter()
        panel_rect, _ = detect_panel(frame)
        t1 = time.perf_counter()
        timings['detect_panel'].append((t1 - t0) * 1000)

        if panel_rect is None:
            print(f"Iteration {i}: Panel not detected")
            continue

        x, y, w, h = panel_rect
        panel_img = frame[y:y+h, x:x+w]

        # Step 2: Slant correction
        t0 = time.perf_counter()
        corrected_img, _, _ = correct_slant(panel_img, 8.0)
        t1 = time.perf_counter()
        timings['correct_slant'].append((t1 - t0) * 1000)

        # Step 3: Gap detection
        t0 = time.perf_counter()
        gap_x, _ = find_digit_gap(corrected_img)
        t1 = time.perf_counter()
        timings['find_digit_gap'].append((t1 - t0) * 1000)

        # Step 4: Define digit boxes
        t0 = time.perf_counter()
        left_box, right_box, _ = define_digit_boxes(corrected_img, gap_x)
        t1 = time.perf_counter()
        timings['define_digit_boxes'].append((t1 - t0) * 1000)

        # Step 5: Extract digits
        t0 = time.perf_counter()
        left_digit_img = _extract_digit_with_padding(corrected_img, left_box, right_bound=gap_x)
        right_digit_img = _extract_digit_with_padding(corrected_img, right_box, left_bound=gap_x)
        t1 = time.perf_counter()
        timings['extract_digits'].append((t1 - t0) * 1000)

        # Step 6a: Recognize left digit (full search)
        t0 = time.perf_counter()
        left_digit, left_score, left_debug = recognize_digit_template(left_digit_img, auto_learn=False, return_debug=True)
        t1 = time.perf_counter()
        timings['recognize_left'].append((t1 - t0) * 1000)

        # Step 6b: Recognize right digit (full search)
        t0 = time.perf_counter()
        right_digit, right_score, right_debug = recognize_digit_template(right_digit_img, auto_learn=False, return_debug=True)
        t1 = time.perf_counter()
        timings['recognize_right'].append((t1 - t0) * 1000)

        total_end = time.perf_counter()
        timings['total'].append((total_end - total_start) * 1000)

        # Single-template quick-check mode timing (after first iteration)
        if left_best_templates is not None and right_best_templates is not None:
            # Convert to grayscale once
            left_gray = cv2.cvtColor(left_digit_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_digit_img, cv2.COLOR_BGR2GRAY)

            # Time 4 single-template matches (2 for each digit: 1st and 2nd candidate)
            t0 = time.perf_counter()
            (ld1, li1), (ld2, li2) = left_best_templates
            (rd1, ri1), (rd2, ri2) = right_best_templates
            match_single_template(left_gray, ld1, li1)
            match_single_template(left_gray, ld2, li2)
            match_single_template(right_gray, rd1, ri1)
            match_single_template(right_gray, rd2, ri2)
            t1 = time.perf_counter()
            timings['single_template_2x'].append((t1 - t0) * 1000)

            # Total with single-template quick-check
            single_total = (timings['detect_panel'][-1] + timings['correct_slant'][-1] +
                          timings['find_digit_gap'][-1] + timings['define_digit_boxes'][-1] +
                          timings['extract_digits'][-1] + timings['single_template_2x'][-1])
            timings['total_single'].append(single_total)

        # Update best templates for next iteration
        left_best_templates = (
            (left_digit, left_debug.get('best_template_idx', 0)),
            (left_debug.get('second_digit', 'X'), left_debug.get('second_template_idx', 0)),
        )
        right_best_templates = (
            (right_digit, right_debug.get('best_template_idx', 0)),
            (right_debug.get('second_digit', 'X'), right_debug.get('second_template_idx', 0)),
        )

    # Calculate statistics
    stats = {}
    for step, times in timings.items():
        if times:
            stats[step] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
            }

    return stats


def print_timing_report(stats):
    """Print a formatted timing report."""
    print("\n" + "=" * 70)
    print("PIPELINE TIMING ANALYSIS - FULL SEARCH")
    print("=" * 70)
    print(f"{'Step':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-" * 70)

    total_mean = 0
    for step in ['detect_panel', 'correct_slant', 'find_digit_gap',
                 'define_digit_boxes', 'extract_digits', 'recognize_left', 'recognize_right']:
        if step in stats:
            s = stats[step]
            print(f"{step:<20} {s['mean']:>10.3f}   {s['std']:>10.3f}   {s['min']:>10.3f}   {s['max']:>10.3f}")
            total_mean += s['mean']

    print("-" * 70)
    if 'total' in stats:
        s = stats['total']
        print(f"{'TOTAL (full)':<20} {s['mean']:>10.3f}   {s['std']:>10.3f}   {s['min']:>10.3f}   {s['max']:>10.3f}")

    # Single-template quick-check comparison
    if 'single_template_2x' in stats:
        print("\n" + "=" * 70)
        print("SINGLE-TEMPLATE MODE (checking 1 template per candidate, 4 total)")
        print("=" * 70)
        print(f"{'Step':<25} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-" * 70)
        s = stats['single_template_2x']
        print(f"{'4 single-template matches':<25} {s['mean']:>10.3f}   {s['std']:>10.3f}   {s['min']:>10.3f}   {s['max']:>10.3f}")
        print("-" * 70)
        if 'total_single' in stats:
            s = stats['total_single']
            print(f"{'TOTAL (single-template)':<25} {s['mean']:>10.3f}   {s['std']:>10.3f}   {s['min']:>10.3f}   {s['max']:>10.3f}")

        # Comparison
        if 'total' in stats and 'total_single' in stats:
            full_time = stats['total']['mean']
            single_time = stats['total_single']['mean']
            speedup = full_time / single_time if single_time > 0 else 0
            savings = ((full_time - single_time) / full_time) * 100 if full_time > 0 else 0
            print(f"\nSingle-template speedup: {speedup:.2f}x ({savings:.1f}% faster)")

    print("\n" + "=" * 70)
    print("BREAKDOWN (% of total - full search)")
    print("=" * 70)
    if 'total' in stats and stats['total']['mean'] > 0:
        total = stats['total']['mean']
        for step in ['detect_panel', 'correct_slant', 'find_digit_gap',
                     'define_digit_boxes', 'extract_digits', 'recognize_left', 'recognize_right']:
            if step in stats:
                pct = (stats[step]['mean'] / total) * 100
                bar = "#" * int(pct / 2)
                print(f"{step:<20} {pct:>5.1f}% {bar}")

    print("\n" + "=" * 70)
    fps_full = 1000 / stats['total']['mean'] if stats.get('total', {}).get('mean', 0) > 0 else 0
    fps_single = 1000 / stats['total_single']['mean'] if stats.get('total_single', {}).get('mean', 0) > 0 else 0
    print(f"Theoretical max FPS (full search):      {fps_full:.1f}")
    print(f"Theoretical max FPS (single-template):  {fps_single:.1f}")
    print("=" * 70)


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Timing analysis for segment reader pipeline')
    parser.add_argument('--image', '-i', type=str, help='Test image path')
    parser.add_argument('--live', '-l', action='store_true', help='Use live camera')
    parser.add_argument('--iterations', '-n', type=int, default=100, help='Number of iterations (default: 100)')
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: Could not load image: {args.image}")
            return
        print(f"Loaded image: {args.image}")
        print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
    elif args.live:
        # Read camera address from webcam.link file
        webcam_link_path = os.path.join(os.path.dirname(__file__), 'webcam.link')
        if not os.path.exists(webcam_link_path):
            print(f"Error: {webcam_link_path} not found")
            return
        with open(webcam_link_path, 'r') as f:
            camera = f.read().strip()

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(camera, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        # Skip initial frames
        for _ in range(30):
            cap.read()

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Error: Could not capture frame")
            return
        print(f"Captured frame from camera")
        print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
    else:
        # Try to find a debug image
        test_images = ['debug_live_frame.png', '/tmp/debug_live_frame.png']
        frame = None
        for img_path in test_images:
            if os.path.exists(img_path):
                frame = cv2.imread(img_path)
                if frame is not None:
                    print(f"Using: {img_path}")
                    print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
                    break

        if frame is None:
            print("No image provided. Use --image or --live")
            return

    print(f"\nRunning {args.iterations} iterations...")
    stats = analyze_pipeline_timing(frame, iterations=args.iterations)
    print_timing_report(stats)


if __name__ == "__main__":
    main()
