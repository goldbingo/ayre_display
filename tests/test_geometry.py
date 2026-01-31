#!/usr/bin/env python3
"""
Geometry System Tests

Tests the DeviceGeometry system with both real and warped images to verify:
1. Geometry model loads correctly and produces expected values
2. Similarity transform correctly adapts to camera movement
3. Detection pipeline handles warped images
4. No performance regression from geometry overhead

Usage:
    python tests/test_geometry.py
"""

import cv2
import glob
import os
import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from device_geometry import DeviceGeometry, get_geometry
from segment_reader import (detect_panel, detect_button_leds, _find_corner,
                            _detect_buttons, SegmentReader)
import device_geometry


# Test tracking
_passed = 0
_failed = 0


def assert_true(condition, msg):
    global _passed, _failed
    if condition:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL: {msg}")


def assert_close(actual, expected, tol, msg):
    """Assert that actual is within tol of expected."""
    if isinstance(expected, (tuple, list)):
        for a, e, label in zip(actual, expected, range(len(expected))):
            if abs(a - e) > tol:
                global _failed
                _failed += 1
                print(f"  FAIL: {msg} (component {label}: {a} vs {e}, tol={tol})")
                return
        global _passed
        _passed += 1
    else:
        if abs(actual - expected) > tol:
            _failed += 1
            print(f"  FAIL: {msg} ({actual} vs {expected}, tol={tol})")
        else:
            _passed += 1


# ===================================================================
# Test 1: Geometry model loads correctly
# ===================================================================
def test_model_loading():
    print("Test 1: Model loading")
    geo = DeviceGeometry()

    assert_true(geo.panel_offset == (-262, -90), f"panel_offset={geo.panel_offset}")
    assert_true(geo.panel_size == (145, 105), f"panel_size={geo.panel_size}")
    assert_true(geo.mute_offset == (200, 43), f"mute_offset={geo.mute_offset}")
    assert_true(geo.mute_search_radius == 40, f"mute_search_radius={geo.mute_search_radius}")
    assert_true(geo.button_panel_gap == 65, f"button_panel_gap={geo.button_panel_gap}")
    assert_true(geo.corner_search_size == 150, f"corner_search_size={geo.corner_search_size}")
    assert_true(geo.led_min_area == 60, f"led_min_area={geo.led_min_area}")
    assert_true(geo.led_max_area == 1200, f"led_max_area={geo.led_max_area}")
    assert_true('B2' in geo.landmark_positions, "B2 landmark missing")
    assert_true('S1' in geo.landmark_positions, "S1 landmark missing")
    assert_true('S2' in geo.landmark_positions, "S2 landmark missing")


# ===================================================================
# Test 2: Camera intrinsics auto-loading
# ===================================================================
def test_intrinsics():
    print("Test 2: Intrinsics loading")
    geo = DeviceGeometry()

    assert_true(geo.has_intrinsics(), "Intrinsics not loaded")
    assert_true(geo._map_x is not None, "Remap table X not precomputed")
    assert_true(geo._map_y is not None, "Remap table Y not precomputed")
    assert_true(geo._map_x.shape == (480, 640), f"Map shape={geo._map_x.shape}")

    # Test point undistortion
    pts = np.array([[320.0, 240.0]])  # Frame center should be minimally affected
    corrected = geo.undistort_points(pts)
    assert_close(corrected[0][0], 320.0, 5.0, "Center X undistort")
    assert_close(corrected[0][1], 240.0, 5.0, "Center Y undistort")


# ===================================================================
# Test 3: Translation-only projection (no homography)
# ===================================================================
def test_translation_projection():
    print("Test 3: Translation projection")
    geo = DeviceGeometry()

    # Without corner set, should return None
    assert_true(geo.get_panel_rect() is None, "Panel rect without corner should be None")
    assert_true(geo.get_mute_region() is None, "Mute region without corner should be None")

    # Set corner
    geo.set_corner(413, 318)
    panel = geo.get_panel_rect()
    mute = geo.get_mute_region()

    assert_true(panel is not None, "Panel rect should not be None")
    assert_close(panel[0], 413 - 262, 1, "Panel X")
    assert_close(panel[1], 318 - 90, 1, "Panel Y")
    assert_true(panel[2] == 145, f"Panel W={panel[2]}")
    assert_true(panel[3] == 105, f"Panel H={panel[3]}")

    assert_true(mute is not None, "Mute region should not be None")
    assert_close(mute[0], 413 + 200, 1, "Mute X")
    assert_close(mute[1], 318 + 43, 1, "Mute Y")
    assert_true(mute[2] == 40, f"Mute radius={mute[2]}")


# ===================================================================
# Test 4: Similarity transform computation
# ===================================================================
def test_similarity_transform():
    print("Test 4: Similarity transform")
    geo = DeviceGeometry()

    corner = (413, 318)
    buttons = {
        'B2': (115, 426),
        'S1': (228, 429),
        'S2': (335, 421),
    }

    ok = geo.compute_homography(corner, buttons)
    assert_true(ok, "Homography computation failed")
    assert_close(geo._scale, 1.0, 0.01, "Scale should be ~1.0")

    # Panel should match translation closely
    panel = geo.get_panel_rect()
    assert_true(panel is not None, "Panel from homography should not be None")
    assert_close(panel[0], 413 - 262, 3, "Panel X from homography")
    assert_close(panel[1], 318 - 90, 3, "Panel Y from homography")


# ===================================================================
# Test 5: Original images still pass
# ===================================================================
def test_original_images():
    print("Test 5: Original image recognition")
    device_geometry._default_geometry = None  # Reset singleton

    images = sorted(glob.glob('example/*.png') + glob.glob('example/*.PNG'))
    reader = SegmentReader()

    fail_count = 0
    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Reset geometry state between images
        geo = device_geometry._default_geometry
        if geo:
            geo._homography = None

        reading, _ = reader.read(frame)
        expected = os.path.basename(img_path).split('-')[0]

        if reading != expected:
            # Allow known edge cases (XX/1X)
            if expected in ('1X', '10') and reading in ('1X', 'XX'):
                pass
            else:
                print(f"  MISMATCH: {os.path.basename(img_path)}: got {reading}, expected {expected}")
                fail_count += 1

        # Reset reader cache between independent images
        reader.reset_cache()

    assert_true(fail_count == 0, f"{fail_count} images failed recognition")


# ===================================================================
# Test 6: Warped images - landmark detection
# ===================================================================
def test_warped_landmarks():
    print("Test 6: Warped image landmarks")
    warped_dir = 'tests/warped_views'
    if not os.path.exists(warped_dir):
        print("  SKIP: Run tests/generate_test_views.py first")
        return

    # Test on shift and rotation warps (these should still find corner)
    testable_warps = ['shift_right_30px', 'shift_down_20px', 'rotate_3deg',
                      'combined_shift_rot']
    found = 0
    total = 0

    for warp in testable_warps:
        pattern = os.path.join(warped_dir, f'*_{warp}.png')
        for img_path in glob.glob(pattern):
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            total += 1

            # Reset geometry
            device_geometry._default_geometry = None
            _ = get_geometry()

            cr = _find_corner(frame, min_match=0.80)  # Lower threshold for warped
            if cr is not None:
                found += 1

    if total > 0:
        rate = found / total
        assert_true(rate >= 0.5, f"Corner detection rate on warped images: {found}/{total} ({rate:.0%})")
    else:
        print("  SKIP: No warped images found")


# ===================================================================
# Test 7: Warped images - digit recognition
# ===================================================================
def test_warped_recognition():
    print("Test 7: Warped image recognition")
    warped_dir = 'tests/warped_views'
    if not os.path.exists(warped_dir):
        print("  SKIP: Run tests/generate_test_views.py first")
        return

    # Small shifts should produce correct readings
    mild_warps = ['shift_right_30px', 'shift_down_20px']
    correct = 0
    total = 0

    for warp in mild_warps:
        pattern = os.path.join(warped_dir, f'*_{warp}.png')
        for img_path in glob.glob(pattern):
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            expected = os.path.basename(img_path).split('-')[0]
            if expected in ('1X', 'XX'):
                continue  # Skip ambiguous images
            total += 1

            # Reset state
            device_geometry._default_geometry = None
            _ = get_geometry()
            reader = SegmentReader()

            reading, _ = reader.read(frame)
            if reading == expected:
                correct += 1

    if total > 0:
        rate = correct / total
        assert_true(rate >= 0.3, f"Warped recognition: {correct}/{total} ({rate:.0%})")
    else:
        print("  SKIP: No warped images found")


# ===================================================================
# Test 8: Performance benchmark
# ===================================================================
def test_performance():
    print("Test 8: Performance")
    device_geometry._default_geometry = None
    geo = get_geometry()

    frame = cv2.imread('example/27-B2-UNMUTE.PNG')
    if frame is None:
        print("  SKIP: Reference image not found")
        return

    # Benchmark corner search
    N = 50
    start = time.time()
    for _ in range(N):
        geo._corner_xy = None
        geo._homography = None
        _find_corner(frame, min_match=0.85)
    elapsed = (time.time() - start) / N * 1000
    assert_true(elapsed < 20, f"Corner search: {elapsed:.1f}ms per call (>20ms)")
    print(f"  Corner search: {elapsed:.1f}ms")

    # Benchmark full pipeline
    reader = SegmentReader()
    start = time.time()
    for _ in range(N):
        geo._homography = None
        reader._prev_frame_roi = None  # Disable frame diff skip
        reader.read(frame)
    elapsed = (time.time() - start) / N * 1000
    assert_true(elapsed < 100, f"Full pipeline: {elapsed:.1f}ms per call (>100ms)")
    print(f"  Full pipeline: {elapsed:.1f}ms")

    # Benchmark similarity transform computation
    corner = (413, 318)
    buttons = {'B2': (115, 426), 'S1': (228, 429), 'S2': (335, 421)}
    start = time.time()
    for _ in range(10000):
        geo.compute_homography(corner, buttons)
    elapsed = (time.time() - start) / 10000 * 1000
    print(f"  Similarity transform: {elapsed:.3f}ms")


# ===================================================================
# Test 9: Default button zones match original
# ===================================================================
def test_button_zones():
    print("Test 9: Button zones")
    geo = DeviceGeometry()

    bw, bh = 416, 140  # Typical button region size
    zones = geo.get_default_button_zones(bw, bh)

    assert_true(len(zones) == 4, f"Expected 4 zones, got {len(zones)}")
    names = [z[4] for z in zones]
    assert_true('B1' in names, "B1 zone missing")
    assert_true('B2' in names, "B2 zone missing")
    assert_true('S1' in names, "S1 zone missing")
    assert_true('S2' in names, "S2 zone missing")

    # Verify zone positions are reasonable
    for left_x, right_x, top_y, bottom_y, name in zones:
        assert_true(left_x < right_x, f"{name}: left > right")
        assert_true(top_y < bottom_y, f"{name}: top > bottom")
        assert_true(right_x <= bw, f"{name}: right > bw")
        assert_true(bottom_y <= bh, f"{name}: bottom > bh")


# ===================================================================
# Test 10: ROI undistortion
# ===================================================================
def test_undistort_roi():
    print("Test 10: ROI undistortion")
    geo = DeviceGeometry()

    frame = cv2.imread('example/27-B2-UNMUTE.PNG')
    if frame is None:
        print("  SKIP: Reference image not found")
        return

    # Undistort a corner ROI
    roi = geo.undistort_roi(frame, 370, 270, 150, 150)
    assert_true(roi.shape == (150, 150, 3), f"ROI shape={roi.shape}")

    # Undistort panel ROI
    roi2 = geo.undistort_roi(frame, 151, 228, 145, 105)
    assert_true(roi2.shape == (105, 145, 3), f"Panel ROI shape={roi2.shape}")

    # Without intrinsics, should return raw crop
    geo2 = DeviceGeometry()
    geo2._camera_matrix = None
    geo2._map_x = None
    geo2._map_y = None
    roi3 = geo2.undistort_roi(frame, 370, 270, 150, 150)
    assert_true(roi3.shape == (150, 150, 3), f"Raw ROI shape={roi3.shape}")


def main():
    tests = [
        test_model_loading,
        test_intrinsics,
        test_translation_projection,
        test_similarity_transform,
        test_original_images,
        test_warped_landmarks,
        test_warped_recognition,
        test_performance,
        test_button_zones,
        test_undistort_roi,
    ]

    for test in tests:
        # Reset singleton between tests
        device_geometry._default_geometry = None
        try:
            test()
        except Exception as e:
            global _failed
            _failed += 1
            print(f"  ERROR: {e}")

    print(f"\n{'=' * 40}")
    print(f"Results: {_passed} passed, {_failed} failed")
    if _failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
