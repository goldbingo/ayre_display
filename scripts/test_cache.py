#!/usr/bin/env python3
"""Streaming-style tests for cache behaviour.

Tests cache paths that single-image tests cannot reach:
  1. cache_ttl removed from SegmentReader
  2. Normal cache round-trip (save → load restores state)
  3. Zombie cache cleared after failure threshold
  4. Cache rebuilt after clear
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import segment_reader as sr

# Pick a real example image for tests that need a frame
_example_dir = os.path.join(os.path.dirname(__file__), '..', 'example')
_example_images = sorted(glob.glob(os.path.join(_example_dir, '*.png')))
if not _example_images:
    print('ERROR: no example images found')
    sys.exit(1)

# Choose an image with a known good LED (B2-UNMUTE) for reliable detection
_frame_path = None
for p in _example_images:
    if 'B2-UNMUTE' in os.path.basename(p):
        _frame_path = p
        break
if _frame_path is None:
    _frame_path = _example_images[0]

_frame = cv2.imread(_frame_path)
if _frame is None:
    print(f'ERROR: could not load {_frame_path}')
    sys.exit(1)

_cache_file = sr._CACHE_FILE
passed = 0
failed = 0


def _reset_state():
    """Reset all detection state between tests."""
    sr._geometry._corner_xy = None
    sr._geometry._homography = None
    sr._geometry._scale = 1.0
    sr._geometry._geo_method = 'none'
    sr._button_zone_cache = None
    sr._panel_cache = None
    sr._cache_led_fail_count = 0
    if os.path.exists(_cache_file):
        os.remove(_cache_file)


def _report(name, ok):
    global passed, failed
    if ok:
        passed += 1
        print(f'  PASS  {name}')
    else:
        failed += 1
        print(f'  FAIL  {name}')


# ── Test 1: cache_ttl removed ──────────────────────────────────────────────
print('Test 1: cache_ttl removed')
_reset_state()
reader = sr.SegmentReader()
ok = not hasattr(reader, 'cache_ttl')
_report('SegmentReader has no cache_ttl attribute', ok)

# ── Test 2: Normal cache round-trip ────────────────────────────────────────
print('Test 2: Normal cache round-trip')
_reset_state()
reader = sr.SegmentReader()
reading, _ = reader.read(_frame)
reader.save_cache()

ok_file = os.path.exists(_cache_file)
_report('cache file created after save_cache()', ok_file)

reader2 = sr.SegmentReader()
loaded = reader2.load_cache()
_report('load_cache() returns True', loaded)

ok_panel = reader2._panel_rect is not None
_report('panel_rect restored', ok_panel)

ok_reading = reader2._last_reading is not None
_report('last_reading restored', ok_reading)

# ── Test 3: Zombie cache cleared on failure threshold ──────────────────────
print('Test 3: Zombie cache cleared on failure threshold')
_reset_state()

# First, run real detection to calibrate geometry
sr._find_corner(_frame)
panel_rect, _ = sr.detect_panel(_frame)

# Seed cache with intentionally wrong button zones (shifted far off-screen)
sr._button_zone_cache = [
    (-500, -400, 10, 30, 'B1'),
    (-300, -200, 10, 30, 'B2'),
    (-100, 0, 10, 30, 'S1'),
]
sr._save_cache()
sr._cache_led_fail_count = 0

# Monkey-patch _detect_buttons to return empty so cache fallback is used.
# Without this, real button detection succeeds and overwrites the bad cache.
_orig_detect_buttons = sr._detect_buttons
sr._detect_buttons = lambda region: []

# Run detect_button_leds in a loop — LED detection will fail using wrong zones
for i in range(sr._CACHE_FAIL_THRESHOLD + 1):
    sr.detect_button_leds(_frame, panel_rect, detection_method='tracked')

ok_cleared = sr._button_zone_cache is None
_report('_button_zone_cache is None after threshold failures', ok_cleared)

ok_file_gone = not os.path.exists(_cache_file)
_report('cache file deleted', ok_file_gone)

# ── Test 4: Cache rebuilt after clear ──────────────────────────────────────
print('Test 4: Cache rebuilt after clear')
# Restore real button detection so fresh zones can be found
sr._detect_buttons = _orig_detect_buttons
# State carries over from test 3 (cache is cleared, geometry still calibrated)
# Run one more detection — should detect zones fresh and save them
sr.detect_button_leds(_frame, panel_rect, detection_method='tracked')

ok_rebuilt = sr._button_zone_cache is not None
_report('_button_zone_cache rebuilt with fresh zones', ok_rebuilt)

ok_file_back = os.path.exists(_cache_file)
_report('cache file recreated', ok_file_back)

# ── Summary ────────────────────────────────────────────────────────────────
print(f'\n{passed} passed, {failed} failed')
sys.exit(0 if failed == 0 else 1)
