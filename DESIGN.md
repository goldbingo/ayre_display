# 7-Segment Display Reader - Design Document

## Overview

This system reads 2-digit numbers from a 7-segment LED display via camera feed. It's designed for real-time monitoring of equipment displays (e.g., audio mixers, industrial panels).

**Key Features:**
- Real-time digit recognition from video stream
- Button LED state detection (B1, B2, S1, S2)
- Mute button (red LED) detection
- Manual template learning via keyboard shortcuts
- Adaptive caching for performance
- Frame skip optimization for CPU efficiency
- Low-latency RTSP streaming with buffer drain

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         live_demo.py                            │
│                    (Camera capture + UI)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SegmentReader Class                        │
│  - Manages detection state and caching                          │
│  - Provides read(frame) -> "XX" API                             │
│  - Handles cache TTL and re-detection                           │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │   Panel     │    │    LED      │    │   Digit     │
   │  Detection  │    │  Detection  │    │ Recognition │
   └─────────────┘    └─────────────┘    └─────────────┘
```

## Processing Pipeline

### 1. Panel Detection (`detect_panel()`)

Finds the dark panel containing the LED display using a cascade of methods:

```
Primary: predict_panel_from_landmarks()
    └── Corner template matching + button detection
    └── Button search region: x=0 to corner_x (left of corner only)
    └── Uses rightmost 3 buttons (B2, S1, S2) - skips B1 if 4 detected
    └── Triangulation from known geometry

Fallback 1: Corner-only detection
    └── Uses _CORNER_TO_PANEL_X/Y offsets

Fallback 2: Brightness-based detection
    └── Thresholds top 3% brightness
    └── Finds contours in valid region
```

**Key Constants:**
- `_PANEL_WIDTH = 145`, `_PANEL_HEIGHT = 105` (reduced width to avoid slant correction artifacts)
- `_CORNER_TO_PANEL_X = 266`, `_CORNER_TO_PANEL_Y = 86`

### 2. Slant Correction (`correct_slant()`)

The LED digits are italicized. A fixed 8.0° shear transform corrects this:

```python
# Shear matrix for slant correction
M = [[1, -tan(angle), offset], [0, 1, 0]]
```

### 3. Digit Gap Detection (`find_digit_gap()`)

Finds the vertical gap between two digits using column brightness projection:

```
1. Convert panel to grayscale
2. Sum pixel values per column → brightness profile
3. Smooth with 5-pixel moving average kernel
4. Search from center (50%) outward for first local minimum
   - Check left and right candidates alternately
   - Local minimum: value lower than both neighbors
   - Search limited to 35%-65% range (±15% from center)
5. Return gap x-position
```

**Center-Outward Search:** The algorithm starts at the center and searches outward to find the valley between digits. This avoids finding false valleys inside hollow digits like "0" or "8" which have internal gaps.

**Key Constant:** `_SEGMENT_LIT_THRESHOLD = 0.5`

### 4. Digit Box Definition (`define_digit_boxes()`)

From gap position, defines bounding boxes for left and right digits:

```
1. Find content bounds in each half
2. Expand to include all lit segments
3. Add padding for template matching tolerance
```

### 5. Digit Recognition (`recognize_digit_template()`)

Template matching against pre-captured digit images:

```
1. Load templates from templates/digit_*.png
2. For each digit (0-9, P):
   - Match all variant templates
   - Track best score per digit
   - Apply position penalty for "1" (see below)
3. Return digit with highest score

Thresholds:
- _TEMPLATE_CONFIDENCE_THRESHOLD = 0.80
- _TEMPLATE_AMBIGUITY_GAP = 0.05
```

**Position Penalty for "1":** The left vertical bars of digits 0, 6, 8, P can match "1" templates. To prevent false positives, "1" matches on the left 30% of the digit box are penalized by 30% **during template comparison**. This ensures templates that match on the right side (like `digit_1g.png`) are preferred over those that match on the left and get penalized.

**Manual Learning:** New templates can be saved via keyboard shortcuts in live_demo.py (`l#` for left digit, `r#` for right digit, e.g., `l6` saves left digit as "6").

## LED Detection

### Button LEDs (`detect_button_leds()`)

Detects which of 4 buttons (B1, B2, S1, S2) has its LED lit:

```
1. Extract button region below panel
2. Detect button rectangles via edge detection
3. Use rightmost 3 buttons (B2, S1, S2) to define zones
   - Skips B1 if 4 buttons detected
   - Falls back to cached zones or fixed proportions if <3 buttons
4. Create green LED mask (HSV filtering)
5. Find brightest zone among button areas
```

**Key Constants:**
- `_BUTTON_REGION_RIGHT_RATIO = 0.65`
- `_BUTTON_REGION_TOP_RATIO = 0.70`
- `_LED_MIN_AREA = 100`, `_LED_MAX_AREA = 1200`

### Mute LED (`detect_red_button()`)

Detects red mute button state using `_detect_red_pixels()`:

```
1. Find corner template position
2. Offset to known red button location
3. Detect LED pixels:
   - Red pixels: HSV H=0-20 or 150-180, S≥50, V≥80
   - White pixels: HSV any H, S≤50, V≥200 (overexposed LED)
4. Filter for bulb-like shapes (area 5-500px, aspect <3, compactness >30%)
5. Threshold: ≥15 LED pixels = lit
```

**Note:** Webcams can overexpose the red LED, causing it to appear white. The detection handles both cases.

## Frame Skip Optimization

Skips full processing when frame content unchanged from reference:

```
1. Extract ROI from frame (200:350, 100:350)
2. Compare to reference frame: diff = sum(abs(current - reference))
3. If diff < 190,000: reuse previous reading (skip processing)
4. If diff >= 190,000: update reference to current frame, then full processing
```

**Thresholds:**
- Exposure cycle variation: ~30K swings every 30 frames
- Skip threshold: 190,000
- Digit change: 160K+ permanent increase

**Performance:**
- Skip rate: ~92% when stable
- Skipped frame: 0.33ms
- Processed frame: 3.29ms
- Speedup: 10x (83% CPU reduction)

**Reference Update:** When diff exceeds threshold, reference updates to current frame before processing. This keeps diff stable relative to recent frames rather than drifting from first frame.

## Dim Digit Enhancement

Enhances dim digits before template matching:

```python
def _enhance_dim_digit(digit_img):
    gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    if gray.max() < 150:  # Dim threshold
        blue = digit_img[:, :, 0]  # Extract blue channel
        enhanced = cv2.normalize(blue, None, 0, 255, cv2.NORM_MINMAX)
        return enhanced, True
    return gray, False
```

**Note:** Mean brightness unreliable (low due to black background), only max used.

## Caching Strategy

The `SegmentReader` class maintains frame-to-frame caches:

| Cache | TTL | Purpose |
|-------|-----|---------|
| `_panel_rect` | 100 frames | Panel bounding box |
| `_gap_x` | 100 frames | Digit separator position |
| `_left_box`, `_right_box` | 100 frames | Digit bounding boxes |
| `_left_best_templates` | 100 frames | Quick-check template indices |
| `_prev_frame_roi` | Until change | Reference for frame skip |

**Cache File:** `last_ref.txt` persists panel/zone data across sessions.

## File Structure

```
segment_reader.py    # Core recognition library (3240 lines)
live_demo.py         # Real-time camera demo
templates/
  ├── corner_template.png    # Reference corner for localization
  ├── digit_0a.png          # Digit templates (multiple variants)
  ├── digit_0b.png
  ├── digit_1a.png
  └── ...
logs/                # Issue frames for debugging
debug/               # Per-image debug output
```

## Key Classes

### `SegmentReader`

Main API for digit reading:

```python
reader = SegmentReader(cache_ttl=100)
reading, changed = reader.read(frame)  # Returns "17", True/False
reader.reset_cache()  # Force re-detection
```

**Properties:**
- `last_reading` - Most recent successful reading
- `confidence` - (left_score, right_score) tuple
- `digit_debug` - Debug info for last recognition

## Configuration Constants

```python
# Detection Thresholds
_TEMPLATE_CONFIDENCE_THRESHOLD = 0.80
_TEMPLATE_AMBIGUITY_GAP = 0.05
_MIN_DIGIT_HEIGHT = 10
_MIN_DIGIT_WIDTH = 5

# Panel Detection
_CORNER_TO_PANEL_X = 266
_CORNER_TO_PANEL_Y = 86
_BRIGHTNESS_PERCENTILE = 97
_MIN_BRIGHTNESS_THRESHOLD = 100
_PANEL_MARGIN_TOP_RATIO = 0.15
_PANEL_MARGIN_BOTTOM_RATIO = 0.85

# Button/LED Detection
_BUTTON_REGION_RIGHT_RATIO = 0.65
_BUTTON_REGION_TOP_RATIO = 0.70
_LED_MIN_AREA = 100
_LED_MAX_AREA = 1200
_LED_MAX_ASPECT_RATIO = 3
```

## Dependencies

- **OpenCV** (`cv2`) - Image processing, template matching
- **NumPy** - Array operations
- **Python 3.10+** - Type hints, walrus operator

## Logging System

### Detection CSV (`logs/detection.csv`)

Logs every frame with columns:
```
timestamp, panel_x, panel_y, panel_w, panel_h, gap_x,
left_score, right_score, reading, led_status,
corner_score, detection_method, brightness_conf,
mute_status, mute_pixels, dim_enhanced, frame_skip, diff_edge, issue
```

### Issue Frame Capture

```python
log_issue_frame(frame, 'low_conf', confidence=0.75, extra_info='17')
```

**Issue Types:**
- `low_conf` - Recognition below threshold
- `ambiguous` - Close scores between digits
- `led_fail` - LED detection failed
- `led_glitch` - B1/B2 flicker pattern detected
- `led_transition` - LED state changed to B1/B2
- `mute_na` - Abnormal MUTE pixel count (>100)
- `digit_1_penalty` - Digit "1" low confidence with "7" close

**Cooldown:** 30 seconds between saves of same issue type.

### iMessage Alerts

Instant notifications via AppleScript:
- LED FAIL
- MUTE_NA
- LED GLITCH
- DIGIT 1 LOW

Config: `.claude/notify_config.json`

### Hourly Summary

Cron job sends iMessage summary at :00 each hour:
```
[2026-01-25 16]
Frames: 19,653
Readings: 09, 10, 11...
LED: B2:19653
MUTE: UNMUTE:19653
Conf: L=92%(min 66%) R=91%(min 65%)
Issues: none
Skip: 99% (1074 near threshold)
```

## Testing

```bash
# Unit tests
python -m pytest test_segment_reader.py -v

# Test on example images
python segment_reader.py

# Live demo (defaults: headless, no logging, drain 2)
python live_demo.py

# With display window and logging
python live_demo.py --display --log

# Skip frames for lower CPU
python live_demo.py --skip 15

# Adjust buffer drain for latency tuning
python live_demo.py --drain 3
```

## Known Limitations

1. **Fixed geometry** - Panel offsets calibrated for specific camera position
2. **Slant angle** - Fixed at 8.0°, not auto-detected
3. **Two digits only** - Hardcoded for 2-digit display
4. **Lighting sensitive** - Blue LED detection requires consistent lighting

## Future Improvements

1. Auto-calibration for new camera positions
2. Dynamic slant angle detection
3. Support for variable digit counts
4. Confidence-based frame interpolation

## CPU Optimization

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--skip N` | 1 | Process every Nth frame (skip before pipeline) |
| `--drain N` | 2 | Grab N frames before read for lower latency |
| `--display` | off | Show GUI window (adds ~6% CPU) |
| `--log` | off | Enable logging to files |

### Diff-Based Skip

| Detection | Method | Cost | Latency |
|-----------|--------|------|---------|
| Digits | Diff ROI (100K threshold) | ~0.2ms skip, ~1ms detect | Instant |
| LED | Diff small region (15K threshold, 40×320px) | ~0.1ms skip, ~1.6ms detect | Instant |
| MUTE | Every frame (no skip) | ~0.3ms | Instant |

**LED region**: Small strip below panel covering LED indicators only (not entire button area).

**Fallback mode**: When panel detection uses fallback (not landmark), LED detection runs every frame (diff region may not cover LEDs).

### Performance (RTSP Stream)

| Mode | CPU | Notes |
|------|-----|-------|
| Default (headless, drain 2) | ~3% | Production recommended |
| `--display` | ~5% | With GUI window |
| `--display --log` | ~5% | Development mode |
| `--skip 7` | ~0.7% | Low CPU mode (process every 7th frame) |

### Buffer Drain for Low Latency

RTSP streams have inherent buffering (~2 seconds). The `--drain N` option grabs N frames before each read to get fresher frames:

```python
# Drain buffer before reading
for _ in range(args.drain):
    cap.grab()  # Fast, discards frame
ret, frame = cap.read()  # Gets next frame
```

- `--drain 0`: No draining (default buffer behavior)
- `--drain 2`: Good balance (default)
- `--drain 3+`: Lower latency, slightly higher CPU

### Recommendations

- **Production (headless)**: Default settings (~3% CPU)
- **Development**: Use `--display --log` (~5% CPU)

## Changelog

### v2.3.0-beta (2026-01-27)

- **Diff-based LED skip**: LED detection uses diff on small region (40×320px, threshold 15K)
- **MUTE every frame**: MUTE detection runs every frame (only 0.3ms, instant detection)
- **Fallback protection**: Disables LED diff-skip in fallback mode (region may not cover LEDs)
- **Instant change detection**: Detects LED/MUTE changes on the frame they occur
- **Exception handling**: Detection operations wrapped in try/except for crash resistance
- **Hang protection**: Timeout limits on warmup loops (15s) and drain loop (2s)

### v2.2.0-beta (2026-01-27)

- **Simplified capture**: Removed GStreamer/GOP decode features (code complexity not worth CPU savings)
- **Low-latency option**: Added `--drain N` to grab N frames before read for fresher frames
- **New defaults**: Headless mode, no logging, drain 2 (production-ready out of box)
- **Inverted flags**: Changed `--headless`/`--no-log` to `--display`/`--log` to match defaults
- **Gap detection fix**: New center-outward search algorithm prevents finding valleys inside hollow digits (0, 8)
- **CPU reduced**: Default ~3% headless, ~5% with display (was ~5-7% / ~12%)

### v2.1.0-beta (2026-01-27) - REMOVED

- Hardware decode (`--hwdec`) and GOP filtering (`--gop-decode`) features removed in v2.2.0
- These added complexity without significant benefit over simpler `--drain` approach

### v1.0.5-beta (2026-01-26)

- **Frame skip fix**: Reference now updates when threshold exceeded (was never updating)
- **Threshold tuning**: Changed from 180K to 190K based on exposure cycle analysis
- **Performance validated**: 92% skip rate, 0.33ms skipped vs 3.29ms processed (10x speedup)
- **Slant correction fix**: Reduced panel width to 145px to avoid grey triangle artifacts
- **Bright center LED fallback**: When blob detection fails, finds zone with brightest center (>220, gap >5)

### v1.0.4-beta (2026-01-25)

- **Removed auto-learning**: Auto-learning feature removed (was triggering on false positives)
- **Manual learning only**: Templates saved via `l#/r#` keyboard shortcuts in live_demo.py
- Simplified SegmentReader API (removed `auto_learn` parameter)

### v1.0.3-beta (2026-01-25)

- **Frame skip optimization**: Skip full processing when ROI unchanged (92% skip rate, 10x speedup)
- **Dim digit enhancement**: Normalize blue channel for dim digits (max < 150)
- **Penalty during template selection**: "1" penalty applied per-template, not after (fixes template choice)
- **New template**: `digit_1g.png` for better "1" matching (no penalty, matches on right)
- **LED glitch detection**: Detects B1/B2 flicker patterns (1-3 frame anomalies)
- **DIGIT 1 LOW alert**: iMessage when "1" confidence < 85% with "7" close
- **Hourly summary**: iMessage report at :00 with readings, LED, MUTE, confidence, skip stats
- **Edge case monitoring**: Logs `diff_edge` when 150K-300K for threshold validation
- **Detection CSV**: Added `dim_enhanced`, `frame_skip`, `diff_edge` columns

### v1.0.2-beta (2026-01-23)

- **MUTE_NA status**: When MUTE pixel count > 100, status is "MUTE_NA" (unreliable detection due to glare/external light)
- **Template learning fix**: Properly reloads templates from disk after manual learning (was appending to stale cache)
- **Cache reset after learning**: Forces full template search after learning new digit (was using quick-check with old templates)
- **Position penalty for digit "1"**: Penalizes "1" matches on left side of digit box by 30% to prevent false positives when matching left bars of 0/6/8/P
- **New template**: Added right digit P template (Pc)
- Fixed P→1 glitches (PP misread as 1P, P6 misread as 16)

### v1.0.1-beta (2026-01-23)

- **MUTE LED detection**: Now handles overexposed (white) LEDs in addition to red
- **MUTE threshold**: Lowered from 25 to 15 pixels for stable detection
- **Button search region**: Limited to left of corner to prevent false positives
- **4-button handling**: Uses rightmost 3 buttons (B2, S1, S2) when B1 is visible
- **LED detection**: Also uses rightmost 3 buttons for zone calculation
- Fixed 9 glitches in overnight logging test

### v1.0.0-beta (2026-01-22)

- Initial beta release
- Refactored code for readability (constants, utility functions, docstrings)
- Added design document for project handover
