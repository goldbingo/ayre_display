#!/usr/bin/env python3
"""Stream simulation tests for mute detection zone (#69).

Verifies that detect_red_button uses homography-based mute region
(from camera_mount.json or landmark detection) instead of fixed offsets.

For visual inspection, use: python scripts/test_image.py <image> --save
"""

import cv2
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import device_geometry
import segment_reader
from segment_reader import detect_panel, detect_red_button, set_tracking, disable_logging

disable_logging()

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'example')
DISTORTED_DIR = os.path.join(os.path.dirname(__file__), '..', 'distorted')


def _reset_geometry():
    """Reset geometry singleton state for clean test isolation."""
    geo = segment_reader._geometry
    geo._homography = None
    geo._corner_xy = None
    geo._scale = 1.0
    geo._geo_method = 'none'
    geo._tracking_enabled = False
    geo._golden_landmarks = None
    geo._golden_homography = None
    geo._golden_corner_xy = None
    geo._golden_scale = None
    geo._calibration_ref = None
    # Remove persisted golden state to prevent cross-test contamination
    if os.path.exists(geo._golden_path):
        os.remove(geo._golden_path)
    # Block camera_mount.json fallback during most tests
    geo._saved_calibration_path = geo._calibration_path
    geo._calibration_path = '/dev/null/nonexistent'


def _reset_geometry_with_calibration():
    """Reset geometry but keep camera_mount.json accessible."""
    geo = segment_reader._geometry
    geo._homography = None
    geo._corner_xy = None
    geo._scale = 1.0
    geo._geo_method = 'none'
    geo._tracking_enabled = False
    geo._golden_landmarks = None
    geo._golden_homography = None
    geo._golden_corner_xy = None
    geo._golden_scale = None
    geo._calibration_ref = None
    if os.path.exists(geo._golden_path):
        os.remove(geo._golden_path)
    # Keep real calibration path
    if hasattr(geo, '_saved_calibration_path'):
        geo._calibration_path = geo._saved_calibration_path


def _restore_calibration_path():
    """Restore calibration path after test."""
    geo = segment_reader._geometry
    if hasattr(geo, '_saved_calibration_path'):
        geo._calibration_path = geo._saved_calibration_path


def _black_frame(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _white_frame(w=640, h=480):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _load(path):
    img = cv2.imread(path)
    assert img is not None, f"Failed to load: {path}"
    return img


def _feed_mute(frames):
    """Feed frames through detect_panel + detect_red_button, return debug info list."""
    results = []
    for frame in frames:
        detect_panel(frame)
        is_lit, _, debug_info = detect_red_button(frame, return_debug=True)
        results.append(debug_info)
    return results


def _valid_region(debug_info, frame_w=640, frame_h=480):
    """Check that region bounds are within frame and non-degenerate."""
    r = debug_info['region']
    left, top, right, bottom = r
    return (left >= 0 and top >= 0 and right <= frame_w and bottom <= frame_h
            and right > left and bottom > top)


# ---------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------

def test_case_1_normal_homography():
    """Case 1: Landmark frame uses homography for mute region (not fixed offset)."""
    _reset_geometry()

    frame = _load(os.path.join(EXAMPLE_DIR, '27-B2-UNMUTE.PNG'))
    detect_panel(frame)
    _, _, debug_info = detect_red_button(frame, return_debug=True)

    assert debug_info['method'] == 'corner', \
        f"Expected method 'corner', got {debug_info['method']!r}"
    assert _valid_region(debug_info), \
        f"Region bounds invalid: {debug_info['region']}"

    # Verify region is near expected mute position (from camera_mount.json: mute_center=[613,361])
    left, top, right, bottom = debug_info['region']
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    assert 550 < cx < 640, f"Mute center X={cx} not near expected ~613"
    assert 300 < cy < 420, f"Mute center Y={cy} not near expected ~361"

    print("  PASS: Case 1 - Normal homography mute region")


def test_case_2_blackout_recovery():
    """Case 2: After blackout, mute region still valid via persistent homography."""
    _reset_geometry()
    set_tracking(True)

    frames = [
        _load(os.path.join(EXAMPLE_DIR, '27-B2-UNMUTE.PNG')),
        _black_frame(),
    ]
    results = _feed_mute(frames)

    # Frame 0: landmark detection works, method should be 'corner'
    assert results[0]['method'] == 'corner', \
        f"Frame 0: expected 'corner', got {results[0]['method']!r}"

    # Frame 1: blackout — no corner found, but homography persists
    assert results[1]['method'] in ('homography', 'corner'), \
        f"Frame 1: expected 'homography' or 'corner', got {results[1]['method']!r}"
    assert results[1]['method'] != 'fallback', \
        "Frame 1: should NOT use fallback with persistent homography"
    assert _valid_region(results[1]), \
        f"Frame 1: region bounds invalid: {results[1]['region']}"

    print("  PASS: Case 2 - Blackout recovery")


def test_case_3_overexposure_recovery():
    """Case 3: After overexposure, mute region still valid."""
    _reset_geometry()
    set_tracking(True)

    frames = [
        _load(os.path.join(EXAMPLE_DIR, '27-B2-UNMUTE.PNG')),
        _white_frame(),
    ]
    results = _feed_mute(frames)

    assert results[0]['method'] == 'corner'
    assert results[1]['method'] != 'fallback', \
        "Overexposed frame should NOT use fallback with persistent homography"
    assert _valid_region(results[1])

    print("  PASS: Case 3 - Overexposure recovery")


def test_case_4_camera_angle_change():
    """Case 4: Distorted variants update homography, mute region adapts."""
    _reset_geometry()

    frames = [
        _load(os.path.join(DISTORTED_DIR, '27-B2-UNMUTE_rotate_3deg.png')),
        _load(os.path.join(DISTORTED_DIR, '27-B2-UNMUTE_perspective_left.png')),
        _load(os.path.join(DISTORTED_DIR, '27-B2-UNMUTE_zoom_in_10pct.png')),
    ]
    results = _feed_mute(frames)

    for i, r in enumerate(results):
        assert r['method'] == 'corner', \
            f"Frame {i}: expected 'corner', got {r['method']!r}"
        assert _valid_region(r), \
            f"Frame {i}: region bounds invalid: {r['region']}"

    # Regions should differ between distorted variants (adapted to each)
    regions = [r['region'] for r in results]
    assert regions[0] != regions[1] or regions[0] != regions[2], \
        "Regions should differ for distorted variants"

    print("  PASS: Case 4 - Camera angle change")


def test_case_5_mute_through_blackout():
    """Case 5: MUTE detection works on valid frames around blackout."""
    _reset_geometry()
    set_tracking(True)

    mute_path = os.path.join(EXAMPLE_DIR, '08-S2-MUTE.PNG')
    frames = [
        _load(mute_path),
        _black_frame(),
        _load(mute_path),
    ]
    results = _feed_mute(frames)

    # MUTE frame should be detected as lit
    assert results[0]['is_lit'], "Frame 0: MUTE frame should be detected as lit"
    assert results[0]['method'] == 'corner'
    assert _valid_region(results[0])

    # Blackout frame: method should not be fallback
    assert results[1]['method'] != 'fallback', \
        "Frame 1: should NOT use fallback with persistent homography"

    # MUTE frame again: still detected
    assert results[2]['is_lit'], "Frame 2: MUTE frame should still be detected as lit"
    assert _valid_region(results[2])

    print("  PASS: Case 5 - MUTE through blackout")


def test_case_6_startup_from_calibration():
    """Case 6: On startup, mute region available from camera_mount.json homography."""
    _reset_geometry_with_calibration()

    geo = segment_reader._geometry
    # Load initial homography from camera_mount.json
    geo._load_initial_homography()

    # Homography should be set from calibration data
    assert geo._homography is not None, \
        "Homography should be loaded from camera_mount.json"

    # get_mute_region() should work without any frames processed
    mute = geo.get_mute_region()
    assert mute is not None, "Mute region should be available from calibration homography"

    btn_x, btn_y, region_half = mute
    # Verify it's near expected position (camera_mount.json: mute_center=[613,361])
    assert 550 < btn_x < 650, f"Mute center X={btn_x} not near expected ~613"
    assert 300 < btn_y < 420, f"Mute center Y={btn_y} not near expected ~361"

    # Now feed a frame — detect_red_button should use homography, not fallback
    frame = _load(os.path.join(EXAMPLE_DIR, '27-B2-UNMUTE.PNG'))
    detect_panel(frame)
    _, _, debug_info = detect_red_button(frame, return_debug=True)
    assert debug_info['method'] != 'fallback', \
        f"Should not use fallback, got {debug_info['method']!r}"
    assert _valid_region(debug_info)

    print("  PASS: Case 6 - Startup from calibration")


def test_case_7_no_calibration_fallback():
    """Case 7: Without camera_mount.json, falls back to fixed region."""
    _reset_geometry()
    geo = segment_reader._geometry
    # Ensure no homography or corner
    assert geo._homography is None
    assert geo._corner_xy is None

    frame = _black_frame()
    detect_panel(frame)
    _, _, debug_info = detect_red_button(frame, return_debug=True)

    assert debug_info['method'] == 'fallback', \
        f"Expected 'fallback' with no calibration, got {debug_info['method']!r}"

    print("  PASS: Case 7 - No calibration fallback")


# ---------------------------------------------------------------

if __name__ == '__main__':
    print("Mute detection zone stream simulation tests (#69)")
    print("=" * 55)

    tests = [
        test_case_1_normal_homography,
        test_case_2_blackout_recovery,
        test_case_3_overexposure_recovery,
        test_case_4_camera_angle_change,
        test_case_5_mute_through_blackout,
        test_case_6_startup_from_calibration,
        test_case_7_no_calibration_fallback,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        finally:
            _restore_calibration_path()

    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
