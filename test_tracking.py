#!/usr/bin/env python3
"""Stream simulation tests for landmark tracking (--track mode).

Feeds frames sequentially through detect_panel() with tracking enabled.
The geometry singleton carries state between calls, simulating a real camera feed.
"""

import cv2
import numpy as np
import os
import sys

# Reset geometry singleton between test groups
import device_geometry
import segment_reader
from segment_reader import detect_panel, set_tracking, disable_logging

disable_logging()

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'example')
DISTORTED_DIR = os.path.join(os.path.dirname(__file__), 'distorted')


def _reset_geometry():
    """Reset geometry singleton state for clean test isolation.

    Clears all transform, golden, and calibration state. Removes persisted
    golden_state.json so set_tracking(True) won't load stale data from
    a previous test or from camera_mount.json fallback.
    """
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
    # Temporarily block camera_mount.json fallback during tests
    geo._saved_calibration_path = geo._calibration_path
    geo._calibration_path = '/dev/null/nonexistent'


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


def _feed(frames):
    """Feed a sequence of frames through detect_panel, return list of (panel_rect, method)."""
    results = []
    for frame in frames:
        panel_rect, method = detect_panel(frame)
        results.append((panel_rect, method))
    return results


# ---------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------

def test_case_1_blackout_recovery():
    """Case 1: Blackout recovery — landmarks → black → landmarks."""
    _reset_geometry()
    set_tracking(True)

    frames = [
        _load(os.path.join(DISTORTED_DIR, '27-B2-UNMUTE_rotate_3deg.png')),
        _load(os.path.join(DISTORTED_DIR, '27-B2-UNMUTE_perspective_left.png')),
        _black_frame(),
        _load(os.path.join(DISTORTED_DIR, '27-B2-UNMUTE_zoom_in_10pct.png')),
    ]
    results = _feed(frames)

    assert results[0][1] == 'landmark', f"Frame 0: expected 'landmark', got {results[0][1]!r}"
    assert results[1][1] == 'landmark', f"Frame 1: expected 'landmark', got {results[1][1]!r}"
    assert results[2][1] == 'tracked',  f"Frame 2: expected 'tracked', got {results[2][1]!r}"
    assert results[2][0] is not None,   "Frame 2: tracked panel_rect should not be None"
    assert results[3][1] == 'landmark', f"Frame 3: expected 'landmark', got {results[3][1]!r}"

    print("  PASS: Case 1 - Blackout recovery")


def test_case_2_overexposure_recovery():
    """Case 2: Overexposure recovery — landmarks → white → white → landmarks."""
    _reset_geometry()
    set_tracking(True)

    frames = [
        _load(os.path.join(EXAMPLE_DIR, '27-B2-UNMUTE.PNG')),
        _white_frame(),
        _white_frame(),
        _load(os.path.join(EXAMPLE_DIR, '27-B2-UNMUTE.PNG')),
    ]
    results = _feed(frames)

    assert results[0][1] == 'landmark', f"Frame 0: expected 'landmark', got {results[0][1]!r}"
    assert results[1][1] == 'tracked',  f"Frame 1: expected 'tracked', got {results[1][1]!r}"
    assert results[1][0] is not None,   "Frame 1: tracked panel_rect should not be None"
    assert results[2][1] == 'tracked',  f"Frame 2: expected 'tracked', got {results[2][1]!r}"
    assert results[2][0] is not None,   "Frame 2: tracked should persist across multiple bad frames"
    assert results[3][1] == 'landmark', f"Frame 3: expected 'landmark', got {results[3][1]!r}"

    # Verify tracked panel_rect matches the landmark panel_rect
    assert results[1][0] == results[0][0], \
        f"Tracked rect {results[1][0]} should match landmark rect {results[0][0]}"

    print("  PASS: Case 2 - Overexposure recovery")


def test_case_3_camera_bump():
    """Case 3: Camera bump — golden updates when landmarks shift >5px."""
    _reset_geometry()
    set_tracking(True)

    frames = [
        _load(os.path.join(EXAMPLE_DIR, '27-B2-UNMUTE.PNG')),
        _load(os.path.join(DISTORTED_DIR, '27-B2-UNMUTE_shift_right_30px.png')),
        _black_frame(),
        _load(os.path.join(DISTORTED_DIR, '27-B2-UNMUTE_shift_right_30px.png')),
    ]
    results = _feed(frames)

    assert results[0][1] == 'landmark', f"Frame 0: expected 'landmark', got {results[0][1]!r}"
    assert results[1][1] == 'landmark', f"Frame 1: expected 'landmark', got {results[1][1]!r}"
    assert results[2][1] == 'tracked',  f"Frame 2: expected 'tracked', got {results[2][1]!r}"
    assert results[2][0] is not None,   "Frame 2: tracked panel_rect should not be None"
    assert results[3][1] == 'landmark', f"Frame 3: expected 'landmark', got {results[3][1]!r}"

    # The tracked rect (from golden after camera bump) should match the shifted
    # landmark rect, not the original. Since the shift is 30px (>5px threshold),
    # golden should have been updated to the shifted position.
    assert results[2][0] == results[1][0], \
        f"Tracked rect {results[2][0]} should match shifted landmark rect {results[1][0]}"

    print("  PASS: Case 3 - Camera bump (golden updates)")


def test_case_4_value_change_through_blackout():
    """Case 4: Display value changes through blackout — panel position preserved."""
    _reset_geometry()
    set_tracking(True)

    frames = [
        _load(os.path.join(EXAMPLE_DIR, '27-B2-UNMUTE.PNG')),
        _black_frame(),
        _load(os.path.join(EXAMPLE_DIR, '09-B2-UNMUTE.PNG')),
    ]
    results = _feed(frames)

    assert results[0][1] == 'landmark', f"Frame 0: expected 'landmark', got {results[0][1]!r}"
    assert results[1][1] == 'tracked',  f"Frame 1: expected 'tracked', got {results[1][1]!r}"
    assert results[2][1] == 'landmark', f"Frame 2: expected 'landmark', got {results[2][1]!r}"

    print("  PASS: Case 4 - Value change through blackout")


def test_case_5_multiple_blackouts():
    """Case 5: Multiple blackouts — tracking persists across repeated outages."""
    _reset_geometry()
    set_tracking(True)

    frames = [
        _load(os.path.join(EXAMPLE_DIR, 'PP-S1-UNMUTE.PNG')),
        _black_frame(),
        _load(os.path.join(EXAMPLE_DIR, 'PP-S1-UNMUTE.PNG')),
        _black_frame(),
        _black_frame(),
        _load(os.path.join(EXAMPLE_DIR, 'PP-S1-UNMUTE.PNG')),
    ]
    results = _feed(frames)

    assert results[0][1] == 'landmark', f"Frame 0: expected 'landmark', got {results[0][1]!r}"
    assert results[1][1] == 'tracked',  f"Frame 1: expected 'tracked', got {results[1][1]!r}"
    assert results[2][1] == 'landmark', f"Frame 2: expected 'landmark', got {results[2][1]!r}"
    assert results[3][1] == 'tracked',  f"Frame 3: expected 'tracked', got {results[3][1]!r}"
    assert results[4][1] == 'tracked',  f"Frame 4: expected 'tracked', got {results[4][1]!r}"
    assert results[5][1] == 'landmark', f"Frame 5: expected 'landmark', got {results[5][1]!r}"

    # All tracked frames should have valid panel_rect
    for i in [1, 3, 4]:
        assert results[i][0] is not None, f"Frame {i}: tracked panel_rect should not be None"

    print("  PASS: Case 5 - Multiple blackouts")


def test_case_6_different_sources():
    """Case 6: Different source images — verify tracking works across panel positions."""
    sources = [
        ('27-B2-UNMUTE.PNG', '27'),
        ('09-B2-UNMUTE.PNG', '09'),
        ('PP-S1-UNMUTE.PNG', 'PP'),
        ('34-S2-UNMUTE.PNG', '34'),
        ('06-B2-UNMUTE-6vsP-close.png', '06'),
    ]

    for filename, label in sources:
        _reset_geometry()
        set_tracking(True)

        frames = [
            _load(os.path.join(EXAMPLE_DIR, filename)),
            _black_frame(),
            _load(os.path.join(EXAMPLE_DIR, filename)),
        ]
        results = _feed(frames)

        assert results[0][1] == 'landmark', \
            f"  [{label}] Frame 0: expected 'landmark', got {results[0][1]!r}"
        assert results[1][1] == 'tracked', \
            f"  [{label}] Frame 1: expected 'tracked', got {results[1][1]!r}"
        assert results[1][0] is not None, \
            f"  [{label}] Frame 1: tracked panel_rect should not be None"
        assert results[2][1] == 'landmark', \
            f"  [{label}] Frame 2: expected 'landmark', got {results[2][1]!r}"

        # Tracked rect should match initial landmark rect
        assert results[1][0] == results[0][0], \
            f"  [{label}] Tracked rect {results[1][0]} != landmark rect {results[0][0]}"

    # Also test with overexposure instead of blackout
    for filename, label in sources[:2]:
        _reset_geometry()
        set_tracking(True)

        frames = [
            _load(os.path.join(EXAMPLE_DIR, filename)),
            _white_frame(),
            _load(os.path.join(EXAMPLE_DIR, filename)),
        ]
        results = _feed(frames)

        assert results[0][1] == 'landmark', \
            f"  [{label}/white] Frame 0: expected 'landmark', got {results[0][1]!r}"
        assert results[1][1] == 'tracked', \
            f"  [{label}/white] Frame 1: expected 'tracked', got {results[1][1]!r}"

    print("  PASS: Case 6 - Different source images")


def test_control_tracking_disabled():
    """Control test: no 'tracked' when tracking is disabled."""
    _reset_geometry()
    set_tracking(False)

    frames = [
        _load(os.path.join(EXAMPLE_DIR, '27-B2-UNMUTE.PNG')),
        _black_frame(),
    ]
    results = _feed(frames)

    assert results[0][1] == 'landmark', f"Frame 0: expected 'landmark', got {results[0][1]!r}"
    assert results[1][1] != 'tracked', \
        f"Frame 1: should NOT be 'tracked' when disabled, got {results[1][1]!r}"

    print("  PASS: Control - No tracking when disabled")


# ---------------------------------------------------------------

if __name__ == '__main__':
    print("Landmark tracking stream simulation tests")
    print("=" * 50)

    tests = [
        test_case_1_blackout_recovery,
        test_case_2_overexposure_recovery,
        test_case_3_camera_bump,
        test_case_4_value_change_through_blackout,
        test_case_5_multiple_blackouts,
        test_case_6_different_sources,
        test_control_tracking_disabled,
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
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
