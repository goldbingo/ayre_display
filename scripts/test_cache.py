#!/usr/bin/env python3
"""Streaming-style tests for cache behaviour.

Tests cache paths that single-image tests cannot reach:
  1. cache_ttl removed from SegmentReader
  2. Normal cache round-trip (save → load restores state)
  3. Zombie cache cleared after failure threshold
  4. Cache rebuilt after clear
"""
import sys, os, glob, json
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

# ── Test 5: _panel_cache global removed ────────────────────────────────────
print('Test 5: _panel_cache global removed')
ok = not hasattr(sr, '_panel_cache')
_report('module has no _panel_cache attribute', ok)

# ── Test 6: Numpy values survive save→load round-trip ─────────────────────
print('Test 6: Numpy values survive save→load round-trip')
_reset_state()
import numpy as np
reader = sr.SegmentReader()
reader._panel_rect = (np.int64(100), np.int64(200), np.int64(50), np.int64(60))
reader._gap_x = np.int64(125)
reader._left_box = (np.int64(100), np.int64(200), np.int64(50), np.int64(60))
reader._right_box = (np.int64(130), np.int64(200), np.int64(50), np.int64(60))
reader._last_reading = '42'
reader.save_cache()

reader2 = sr.SegmentReader()
loaded = reader2.load_cache()
_report('load_cache returns True', loaded)

ok_types = all(isinstance(v, int) for v in reader2._panel_rect)
_report('panel_rect values are plain Python int', ok_types)

ok_gap = isinstance(reader2._gap_x, int) and reader2._gap_x == 125
_report('gap_x is plain int with correct value', ok_gap)

# ── Test 7: Button zone save preserves panel data on disk ─────────────────
print('Test 7: Button zone save preserves panel data on disk')
_reset_state()
reader = sr.SegmentReader()
reader._panel_rect = (10, 20, 30, 40)
reader._gap_x = 15
reader._left_box = (10, 14, 30, 40)
reader._right_box = (16, 20, 30, 40)
reader._last_reading = '77'
reader.save_cache()

# Now simulate detect_button_leds saving zones (no panel_data)
sr._button_zone_cache = [(100, 200, 10, 30, 'B1'), (300, 400, 10, 30, 'B2')]
sr._save_cache()  # no panel_data — should preserve panel on disk

with open(_cache_file, 'r') as f:
    disk = json.load(f)
ok_panel = 'panel' in disk and disk['panel']['gap_x'] == 15
_report('panel data preserved after button zone save', ok_panel)

ok_zones = 'button_zones' in disk and len(disk['button_zones']) == 2
_report('button zones written', ok_zones)

# ── Test 8: Panel save preserves button zones on disk ─────────────────────
print('Test 8: Panel save preserves button zones on disk')
_reset_state()
# Seed button zones and save to disk
sr._button_zone_cache = [(50, 60, 5, 15, 'S1')]
sr._save_cache()  # writes zones only

# Now save panel data — should preserve zones
sr._save_cache(panel_data={'panel_rect': [1, 2, 3, 4], 'gap_x': 99,
                           'left_box': [1, 2, 3, 4], 'right_box': [1, 2, 3, 4],
                           'last_reading': '11'})

with open(_cache_file, 'r') as f:
    disk = json.load(f)
ok_zones = 'button_zones' in disk and disk['button_zones'][0]['name'] == 'S1'
_report('button zones preserved after panel save', ok_zones)

ok_panel = 'panel' in disk and disk['panel']['gap_x'] == 99
_report('panel data written', ok_panel)

# ── Test 9: clear_cache() wipes both sections ─────────────────────────────
print('Test 9: clear_cache() wipes both sections')
_reset_state()
# Write both sections to disk
sr._button_zone_cache = [(1, 2, 3, 4, 'B1')]
sr._save_cache(panel_data={'panel_rect': [1, 2, 3, 4], 'gap_x': 5,
                           'left_box': None, 'right_box': None, 'last_reading': None})

sr.clear_cache()
ok_file = not os.path.exists(_cache_file)
_report('cache file deleted', ok_file)

ok_zones = sr._button_zone_cache is None
_report('_button_zone_cache is None', ok_zones)

# ── Test 10: _load_panel_from_cache() handles missing/corrupt file ────────
print('Test 10: _load_panel_from_cache() handles missing/corrupt file')
_reset_state()

# Missing file
result = sr._load_panel_from_cache()
ok_missing = result is None
_report('returns None when file missing', ok_missing)

# Invalid JSON
with open(_cache_file, 'w') as f:
    f.write('not json!!!')
result = sr._load_panel_from_cache()
ok_corrupt = result is None
_report('returns None for corrupt JSON', ok_corrupt)

# Valid JSON, no panel key
with open(_cache_file, 'w') as f:
    json.dump({'button_zones': []}, f)
result = sr._load_panel_from_cache()
ok_no_key = result is None
_report('returns None when panel key absent', ok_no_key)

# ── Test 11: _save_cache() no panel_data, no existing file ───────────────
print('Test 11: _save_cache() no panel_data, no existing file (first-run)')
_reset_state()
sr._button_zone_cache = [(10, 20, 5, 15, 'B1')]
sr._save_cache()  # no panel_data, no file on disk

with open(_cache_file, 'r') as f:
    disk = json.load(f)
ok_zones = 'button_zones' in disk
_report('button_zones written', ok_zones)

ok_no_panel = 'panel' not in disk
_report('no panel key (none to preserve)', ok_no_panel)

# ── Test 12: _save_cache() no panel_data, corrupt existing file ──────────
print('Test 12: _save_cache() no panel_data, corrupt existing file')
_reset_state()
with open(_cache_file, 'w') as f:
    f.write('GARBAGE')
sr._button_zone_cache = [(10, 20, 5, 15, 'B1')]
sr._save_cache()  # no panel_data, corrupt file on disk

with open(_cache_file, 'r') as f:
    disk = json.load(f)
ok_zones = 'button_zones' in disk
_report('button_zones written despite corrupt existing file', ok_zones)

ok_no_panel = 'panel' not in disk
_report('no panel key (corrupt file ignored)', ok_no_panel)

# ── Test 13: Real detect_button_leds doesn't wipe panel data ─────────────
print('Test 13: Real detect_button_leds doesn\'t wipe panel data')
_reset_state()

# Process a frame to populate geometry
sr._find_corner(_frame)
panel_rect2, _ = sr.detect_panel(_frame)
if panel_rect2 is not None:
    # Save panel data via SegmentReader
    reader = sr.SegmentReader()
    reader.read(_frame)
    reader.save_cache()

    # Verify panel is on disk
    with open(_cache_file, 'r') as f:
        before = json.load(f)
    had_panel = 'panel' in before

    # Run detect_button_leds — internally calls _save_cache() with no panel_data
    sr.detect_button_leds(_frame, panel_rect2, detection_method='tracked')

    with open(_cache_file, 'r') as f:
        after = json.load(f)
    ok = had_panel and 'panel' in after
    _report('panel data survives detect_button_leds', ok)
else:
    _report('panel data survives detect_button_leds (SKIPPED: no panel detected)', False)

# ── Summary ────────────────────────────────────────────────────────────────
print(f'\n{passed} passed, {failed} failed')
sys.exit(0 if failed == 0 else 1)
