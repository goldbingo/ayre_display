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
- MQTT publishing for home automation integration
- Landmark tracking for blackout/overexposure recovery (`--track`)

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
│  - Persists state to disk (last_ref.txt)                        │
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
    └── Corner template matching (multiple templates with round-robin)
    └── Button search region: x=0 to corner_x (left of corner only)
    └── Uses rightmost 3 buttons (B2, S1, S2) - skips B1 if 4 detected
    └── Triangulation from known geometry

**Corner Templates:** Uses round-robin with sticky preference - tries current
template first, switches only if it fails (score < 0.85). Three templates stored in
`templates/corner_template.png`, `corner_template_2.png`, `corner_template_3.png`.

Tracking restore (--track): restore_golden()       → 'tracked'
    └── Reuses last known good homography when landmarks disappear
    └── Enabled via --track flag, off by default

Fallback 1: Corner-only detection
    └── Uses _CORNER_TO_PANEL_X/Y offsets

Fallback 2: Brightness-based detection
    └── Thresholds top 3% brightness
    └── Finds contours in valid region
```

**Landmark Tracking (`--track`):** When enabled, stores "golden" landmark positions
(corner + button centers, homography, scale) on first successful landmark detection.
Updates golden state when any landmark moves >5px (camera bump). When landmarks
disappear (blackout, overexposure), restores the golden homography to maintain panel
position. The `'tracked'` method is treated like `'landmark'` for LED zone sizing.

**Key Constants:**
- `_PANEL_WIDTH = 145`, `_PANEL_HEIGHT = 105` (reduced width to avoid slant correction artifacts)
- `_CORNER_TO_PANEL_X = 262`, `_CORNER_TO_PANEL_Y = 90` (from `device_model.json`)

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
4. Check for peak (local max) within ±5 pixels of center
   a. If peak nearby: search both left and right, pick deeper valley
   b. If center is already a valley bottom: use center directly
   c. Otherwise: follow slope direction to find valley bottom
   - Local minimum: value lower than both neighbors
   - Search limited to 35%-65% range (±15% from center)
5. Return gap x-position
```

**Center-Outward Search:** The algorithm starts at the center and searches outward to find the valley between digits. Searching both sides when a peak is nearby prevents picking the wrong valley when the gap isn't centered. This also avoids finding false valleys inside hollow digits like "0" or "8" which have internal gaps.

**Key Constant:** Segment lit threshold = 0.15 (hardcoded in `_resolve_confusing_pair()`)

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

**Position Penalty for "1":** The left vertical bars of digits 0, 6, 8, P can match "1" templates. To prevent false positives, "1" matches on the left 35% of the digit box have their score multiplied by 0.7 **during template comparison**, but only when the 2nd-best digit is a left-bar digit (0/6/8/P). This ensures templates that match on the right side (like `digit_1g.png`) are preferred over those that match on the left and get penalized.

**Manual Template Learning:** (in `--display` mode)

When a digit is misrecognized, capture it as a new template:
1. Press `l` (left) or `r` (right) to select which digit
2. Press the correct digit (`0-9` or `P`)

**Saves to:** `templates/digit_{digit}{letter}.png`
- Letters `a-z` used sequentially (finds next unused)
- Example: `l6` → `digit_6a.png` (or `6b`, `6c`... if `6a` exists)

Templates auto-reload after saving.

## LED Detection

### Button LEDs (`detect_button_leds()`)

Detects which of 4 buttons (B1, B2, S1, S2) has its LED lit:

```
1. Extract button region below panel
2. Detect button rectangles via edge detection
3. Use rightmost 3 buttons (B2, S1, S2) to define zones
   - B1 predicted from button spacing (LED at ~88% of button width)
   - Falls back to cached zones or fixed proportions if <3 buttons
4. Compute all 3 methods independently:
   a. Brightness: max blue channel value per zone (not grayscale)
      - Blue channel gives better contrast for blue LEDs
      - Grayscale dilutes blue signal: 255 blue → ~170 gray
   b. Blob: HSV filtering (H=85-130, S≥150, V≥80)
      - High saturation (S≥150) excludes display glow (S~30)
      - Largest blob inside a button zone wins
   c. Center: mean brightness of center 50% of each zone
5. Agreement-based decision (pick first that matches):
   a. Brightness confident: val >200 and gap >30 → trust brightness
   b. Blob agrees with brightness winner → trust agreement
   c. Center confident: val >220 and gap >5 → trust center
   d. Blob in bright region (val >200) → trust blob
```

This replaces the old cascading fallback (brightness → blob → center) which
could let a noise blob override a correct center detection during auto-exposure
spikes.

**Key Constants:**
- `_BUTTON_REGION_RIGHT_RATIO = 0.65`
- `_BUTTON_REGION_TOP_RATIO = 0.70`
- `_LED_MIN_AREA = 60`, `_LED_MAX_AREA = 1200`
- Blue brightness threshold: >200, gap >30
- Saturation threshold: S≥150

### Mute LED (`detect_red_button()`)

Detects red mute button state using `_detect_red_pixels()`:

```
1. Find corner template position
2. Offset to known red button location
3. Dark-region brightness fallback (night mode):
   - If region mean brightness < 60 and max-mean gap > 100 → LED is lit
   - Bypasses color normalization that kills red signal at night
   - Method tagged as "corner_bright" or "fallback_bright"
4. Color normalization: subtract red bias (median R - median G)
5. Detect LED pixels:
   - Red pixels: HSV H=0-10 or 150-180, S≥50, V≥80
   - White pixels: HSV any H, S≤50, V≥200 (overexposed LED)
6. Filter for bulb-like shapes (area 5-500px, aspect <3, compactness >30%)
7. Threshold: ≥15 LED pixels = lit
```

**Note:** Webcams can overexpose the red LED, causing it to appear white. The detection handles both cases. At night, color normalization can strip the real LED signal; the brightness fallback avoids this.

## Frame Skip Optimization

Skips full processing when frame content unchanged from reference:

```
1. Extract ROI from frame (200:350, 100:350)
2. Compare to reference frame: diff = sum(abs(current - reference))
3. If diff < 100,000: reuse previous reading (skip processing)
4. If diff >= 100,000: update reference to current frame, then full processing
```

**Thresholds:**
- Exposure cycle variation: ~30K swings every 30 frames
- Skip threshold: 100,000 (3-channel mode)
- Digit change: 160K+ permanent increase

**Performance** (500 live frames, `--track --undistort`):
- Skip rate: ~99% when stable
- Skipped frame: 0.24ms
- Processed frame: 13.8ms
- Speedup: 56x per frame

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

The `SegmentReader` class maintains instance-level detection state. The disk file
`last_ref.txt` is the single source of truth for persistence — there is no module-level
global for panel data.

| Instance var | Purpose |
|--------------|---------|
| `_panel_rect` | Panel bounding box |
| `_gap_x` | Digit separator position |
| `_left_box`, `_right_box` | Digit bounding boxes |
| `_left_best_templates` | Quick-check template indices |
| `_prev_frame_roi` | Reference for frame skip |

**Disk Cache (`last_ref.txt`):** A single JSON file with two independent sections:
- `panel` — panel rect, gap, digit boxes, last reading (written by `SegmentReader.save_cache()`)
- `button_zones` — adaptive LED zone positions (written by `detect_button_leds()`)

Each section is preserved when the other is updated. `_save_cache(panel_data=None)`
reads the existing file to preserve the panel section when only button zones change.
`clear_cache()` deletes the file and resets the in-memory button zone cache.

## File Structure

```
├── segment_reader.py          # Core recognition library (~3900 lines)
├── live_demo.py               # Real-time camera monitoring
├── device_geometry.py         # Device geometry model (spatial constants)
├── calibrate_camera.py        # Camera calibration utility
├── watchdog.sh                # Process watchdog (restarts if hung)
├── CLAUDE.md                  # Project instructions for AI assistant
├── DESIGN.md                  # This file
├── .gitignore
│
├── templates/                 # Recognition templates
│   ├── corner_template*.png   # Corner templates for localization (3 variants)
│   └── digit_*.png            # Digit templates (0-9, P, X, multiple variants)
│
├── example/                   # Reference images (42) for batch testing
│
├── calibration/               # Camera/device calibration data
│   ├── camera.json            # Camera intrinsics
│   ├── camera_mount.json      # Camera mount parameters
│   └── device_model.json      # Device geometry model
│
├── scripts/                   # Tests, analysis, and utility scripts
│   ├── test_cache.py          # Cache behaviour tests (13 tests)
│   ├── test_distorted.py      # Perspective distortion tests
│   ├── test_geometry.py       # Device geometry unit tests (62 tests)
│   ├── test_tracking.py       # Landmark tracking stream tests (7 tests)
│   ├── analyze_skip.py        # Frame-skip threshold analysis
│   ├── timing_analysis.py     # Pipeline and skip benchmarking
│   ├── gen_perspective_variants.py  # Generate distorted test images
│   ├── generate_test_views.py # Generate warped test views
│   └── warped_views/          # Generated warped images
│
├── foscam-c2/                 # Reference camera snapshots
├── logs/                      # Runtime logs and issue frames
├── debug/                     # Per-image debug output (generated)
├── distorted/                 # Generated distortion test images
└── legacy/                    # Old versioned files (not tracked)
```

## Key Classes

### `SegmentReader`

Main API for digit reading:

```python
reader = SegmentReader()
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
_CORNER_TO_PANEL_X = 262  # from device_model.json
_CORNER_TO_PANEL_Y = 90   # from device_model.json
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
- **Python 3.8+**
- **paho-mqtt** (optional) - MQTT publishing for home automation

## Logging System

### Detection CSV (`logs/detection.csv`)

Logs every frame with columns:
```
timestamp, panel_x, panel_y, panel_w, panel_h, gap_x,
left_score, right_score, reading, led_status,
corner_score, detection_method, brightness_conf,
mute_status, mute_pixels, dim_enhanced, frame_skip, diff_edge,
diff_mode, led_gap, led_method, proc_ms, issue,
geo_method, geo_scale, geo_rotation, undistort_px
```

- `led_gap`: Brightness difference between brightest and 2nd brightest LED zone
- `led_method`: Which detection method succeeded (brightness/blob/center)
- `proc_ms`: Processing time for the frame
- `geo_method`/`geo_scale`/`geo_rotation`: Geometry transform info
- `undistort_px`: Max pixel shift from lens undistortion

### Issue Frame Capture

```python
log_issue_frame(frame, 'low_conf', confidence=0.75, extra_info='17')
```

**Issue Types:**
- `low_conf` - Recognition below threshold
- `ambiguous` - Close scores between digits
- `led_fail` - LED detection failed
- `led_glitch` - B1/B2 flicker pattern detected
- `reading_glitch` - Single-frame reading change (A→B→A pattern)
- `mute_glitch` - Single-frame mute status flip (A→B→A pattern)
- `led_transition` - LED state changed to B1/B2
- `mute_na` - Abnormal MUTE pixel count (>100)
- `digit_1_penalty` - Digit "1" low confidence with "7" close

**Cooldown:** 30 seconds between saves of same issue type.

### iMessage Alerts

Instant notifications via AppleScript:
- LED FAIL
- MUTE_NA
- LED GLITCH
- READING GLITCH
- MUTE GLITCH
- DIGIT 1 LOW

**Cooldown:** 10 minutes per issue type. Suppressed notifications are counted
and shown when cooldown expires, e.g., "LED FAIL: detection failed (+15 suppressed)"

Config: `.claude/notify_config.json`

### MQTT Publishing

Publishes state to MQTT broker for home automation integration:

```bash
python live_demo.py --mqtt-config .claude/mqtt_config.json
```

**Topics published (with retain flag):**
| Topic | Value | Description |
|-------|-------|-------------|
| `{base}/7seg/num` | "07" | Current reading |
| `{base}/vol` | "07" | Same as reading (alias) |
| `{base}/source` | "S2" | LED status (B1/B2/S1/S2/NA) |
| `{base}/mute` | "off" | "off", "on", or "unknown" |
| `{base}/status` | "online" | "online" or "offline" (Last Will) |

**Publish triggers:** Same as stdout - on state change OR every 60 seconds.

**Config file:** `.claude/mqtt_config.json`
```json
{
    "broker": "mqtt.example.com:8883",
    "base_topic": "home/ayre",
    "user": "username",
    "password": "password",
    "ca_cert": "/path/to/ca.crt"
}
```

- `broker`: Host:port (default port 1883 if not specified)
- `base_topic`: Prefix for all topics
- `user`/`password`: Optional authentication
- `ca_cert`: Optional TLS certificate path

**Note:** Requires `paho-mqtt` package: `pip install paho-mqtt`

## Usage

### `live_demo.py` — Main monitoring script

```bash
# Production (headless, no logging)
python live_demo.py

# Development (display window + logging)
python live_demo.py --display --log

# With landmark tracking (survives blackout/overexposure)
python live_demo.py --display --log --track

# With lens undistortion and tracking
python live_demo.py --display --log --track --undistort

# Adaptive skip: target 1.5 fps
python live_demo.py --target-fps 1.5

# Fixed skip: process every 7th frame
python live_demo.py --skip 7

# Buffer drain for lower latency
python live_demo.py --drain 3

# MQTT publishing for home automation
python live_demo.py --mqtt-config .claude/mqtt_config.json

# Benchmark: process 1000 frames and exit
python live_demo.py --benchmark 1000
```

**Keys** (in `--display` mode): `q` quit, `c` reset cache, `s` save frame, `l#`/`r#` learn digit template (e.g. `l6` learns left digit as 6).

### `segment_reader.py` — Batch test on example images

```bash
# Runs all 42 example/ images through the pipeline
# Expected: 2 XX results (transition images), rest must match filename
python segment_reader.py
```

### `calibrate_camera.py` — Camera calibration

```bash
# Default: calibrate from foscam-c2/ checkerboard images (9x6 pattern)
python calibrate_camera.py

# Custom checkerboard pattern and image directory
python calibrate_camera.py --pattern 7x5 --images /path/to/photos/

# Custom output
python calibrate_camera.py --output calibration/my_camera.json
```

Generates `calibration/camera.json` with intrinsics transformed from native 1920x1080 to the 640x480 RTSP feed. Only re-run if the camera is replaced or repositioned.

### `watchdog.sh` — Process monitor

```bash
# Add to crontab (checks every minute, restarts if hung)
* * * * * /path/to/watchdog.sh
```

### Testing

```bash
# All scripts are in scripts/
python scripts/test_cache.py        # Cache behaviour (13 tests)
python scripts/test_distorted.py    # Perspective distortion
python scripts/test_geometry.py     # Device geometry (62 tests)
python scripts/test_tracking.py     # Landmark tracking (7 tests)

# Analysis tools
python scripts/analyze_skip.py                      # Skip rate from detection.csv
python scripts/timing_analysis.py --live -n 500     # Pipeline breakdown
python scripts/timing_analysis.py --skip -n 500     # Frame skip measurement
python scripts/timing_analysis.py --skip --track --undistort -n 500

# Regenerate test images
python scripts/gen_perspective_variants.py
python scripts/generate_test_views.py
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

## Changelog

### v3.5 (2026-02-03)

- **Debug metadata for all diagnostics**: All four diagnostic log types (gap_ambiguous, gap_wide_valley, reading_glitch, mute_glitch) now save `.txt` metadata files alongside PNG captures via `build_debug_info()`. Includes code version (git short hash), frame_skipped flag, panel/reading/scores, LED/mute state, corner info.
- **Unified diagnostic pipeline stage**: Moved gap diagnostics (gap_ambiguous, gap_wide_valley) to run after LED/mute detection, so all diagnostics use current-frame values instead of stale previous-frame caches.
- **Code version tracking**: `_code_version` computed once at startup from `git rev-parse --short HEAD`, logged in every diagnostic `.txt` file.
- **Fix penalized "1" rejection** (#59): Use unpenalized score for the rejection check so penalty affects ranking only, not ambiguity rejection.
- **U-valley filtering**: Filter expected U-shaped valleys from gap_wide_valley diagnostic for x7 (valley near left peak) and Px (valley near right peak) readings.

### v3.4 (2026-02-02)

- **Agreement-based LED method selection**: Replaced cascading fallback (brightness → blob → center) with independent computation of all 3 methods followed by agreement-based decision. Prevents noise blob from overriding correct center detection during auto-exposure spikes (e.g., blob picks B2 noise while center correctly finds S1).

### v3.3 (2026-02-02)

- **Fix glitch composite off-by-one**: LED glitch, reading glitch, and mute glitch composites were logging the wrong frame due to `frame_history` being appended after glitch detection. Moved `frame_history.append` before glitch checks so indices align with `led_history`. Fixes #57.

### v3.2 (2026-02-01)

- **B1 homography projection**: Added B1 landmark to `device_model.json` and `project_landmark()` to `DeviceGeometry`. B1 button position now uses homography projection instead of pixel-space linear extrapolation, fixing ~18px barrel distortion error at the left frame edge. B1 LED zone uses right half of projected button box. Extrapolation kept as fallback.

### v3.1 (2026-02-01)

- **Eliminate `_panel_cache` global**: Disk file (`last_ref.txt`) is now the single source of truth for panel data. No module-level global for panel cache. `_save_cache()` preserves existing sections when updating only one part. 9 new cache tests added.
- **Repo cleanup**: Moved 35+ old versioned files and stale docs to `legacy/`. Removed `README.md` (redundant with DESIGN.md), `mqtt_config.json.example`, `hourly_summary.py`, `test_segment_reader.py`.
- **Consolidated `scripts/` directory**: Merged `tests/` into `scripts/`. Moved `analyze_skip.py`, `timing_analysis.py`, `test_tracking.py` into `scripts/`. Fixed all relative paths.
- **Timing analysis `--skip` mode**: New streaming mode captures real frames through `SegmentReader.read()` to measure actual skip rate. Added `--track` and `--undistort` flags.
- **Updated DESIGN.md**: Comprehensive file structure, caching strategy rewritten, architecture diagram updated.

### v3.0 (2026-02-01)

- **Camera calibration & geometry model**: New `device_geometry.py` module with `DeviceGeometry` class. Loads device model from `calibration/device_model.json`. Supports homography-based projection, similarity transform (de-rotation + scale normalization), and lens undistortion via camera intrinsics.
- **De-rotation & scale normalization**: Panel crop uses similarity transform derived from homography to correct camera tilt and distance variation. Logged as `geo_method`, `geo_scale`, `geo_rotation` in CSV.
- **Lens undistortion**: `--undistort` flag gates ROI de-warping using camera intrinsics from `calibration/camera.json`. Undistortion logged as `undistort_px` (max pixel shift).
- **Landmark tracking (`--track`)**: Stores golden landmark positions when detected and reuses them during blackout/overexposure. Detection cascade: `landmark` → `tracked` → `corner` → `brightness`. Golden state updates when any landmark moves >5px (camera bump).
- **Corner detection improvements**: Lowered match threshold from 0.90 to 0.85. Skip matching when search region is too dark or overexposed. New `corner_template_3.png`.
- **Fix gap detection false valleys**: `_find_valley` returns whether a true local minimum was found. Falls back to center instead of picking a point on the slope. Fixes `14` → `11` glitches during dim lighting.
- **Adaptive frame diff**: 3-channel diff (100K threshold, ~93% skip) with periodic blue-only probing. Logged as `diff_mode` in CSV.
- **Debug overlay**: `test_on_image` now renders debug overlay matching live_demo display.
- **New template**: `digit_9f` for night-glowy 9 variant.
- **Distortion test suite**: 9 perspective warp variants per source image. Dark images excluded from distortion generation. 100% pass rate.
- **New files**: `device_geometry.py`, `calibrate_camera.py`, `test_tracking.py`

### v2.5.8-beta (2026-02-01)

- **Landmark tracking (`--track`)**: New `--track` option stores golden landmark positions when detected and reuses them when landmarks disappear (blackout, overexposure). Detection cascade: `landmark` → `tracked` → `corner` → `brightness`. Golden state updates when any landmark moves >5px (camera bump). `'tracked'` method treated like `'landmark'` for LED zone sizing.
- **New file**: `test_tracking.py` — stream simulation tests (7 cases: blackout recovery, overexposure, camera bump, value change, multiple blackouts, different sources, disabled control)
- **Fix gap detection false valleys**: `_find_valley` now returns whether a true local minimum was found. If no real valley exists (just a slope), falls back to center instead of picking a point on the side of a digit. Fixes `14` → `11` glitches during dim lighting.

### v2.5.7-beta (2026-02-01)

- **Adaptive frame diff**: Default to 3-channel diff (100K threshold, ~93% skip). Every ~5 min, probe blue-only mode (33K threshold). If skip ratio drops below 88%, revert to 3-channel. Logged as `diff_mode` (3ch/1ch) in CSV.

### v2.5.6-beta (2026-02-01)

- **Improved gap detection**: Search both sides when a peak is near center, pick deeper valley. If center is already a valley, use it directly. Otherwise follow slope direction. Prevents picking wrong valley when gap is off-center.
- **New example**: `14-B2-UNMUTE-gap-bright.png` test case for gap-bright edge case

### v2.5.5-beta (2026-01-31)

- **Night mute brightness fallback**: When mute region is dark (mean < 60), detect LED via brightness gap (max − mean > 100) instead of color analysis. Prevents 16K overnight MUTE/UNMUTE flicker caused by color normalization stripping the real red signal.

### v2.5.4-beta (2026-01-30)

- **Washout LED fallback**: When button region is overexposed (mean brightness > 230), detect lit LED via dark-hole analysis — the lit button has no recessed hole visible, yielding highest min brightness after 5x5 erosion. Requires gap >= 30 to avoid false positives in severe washout.
- **Closed issues**: #45 (white mask false positive — already fixed in v2.5.3), #46 (washout LED detection), #47 (LED glitch — already fixed in v2.4.3)

### v2.5.3-beta (2026-01-30)

- **Reading glitch detection**: Detects single-frame reading changes (A→B→A pattern, excluding XX) and saves composite image with 3 before + glitch + after frames
- **Mute glitch detection**: Same A→B→A pattern for mute status flips (excluding MUTE_NA), saves composite with labeled frames
- **Mute artifact fix**: Narrowed low-hue red detection from H=0-20 to H=0-10 to reject orange camera artifacts (H=16-18) that inflated mute pixel counts
- **Example image fix**: Cropped two composite debug screenshots to raw 640x480 frames for correct panel detection

### v2.5.2-beta (2026-01-30)

- **X digit templates**: Added `digit_Xa.png` and `digit_Xb.png` for transient display states during digit transitions (e.g., 2→3), preventing misreading as valid digits like 8

### v2.5.1-beta (2026-01-29)

- **"1" penalty fix**: Only penalize when 2nd best is left-bar digit (0/6/8/P), not when 2nd is "7"
- **MQTT raw digits**: `7seg/num` publishes raw recognized digits (before XX conversion)
- **MQTT vol filter**: `vol` topic only publishes valid volume readings (00-66)
- **MQTT efficiency**: Only publish changed values on state change, all values on minute heartbeat
- **MQTT skip invalid**: Don't publish when reading contains "X" or LED is "NA"
- **Watchdog script**: `watchdog.sh` monitors heartbeat file, restarts if hung (FreeBSD compatible)
- **New example**: `11-B2-UNMUTE-1-penalty-fix.png` test case

### v2.5.0-beta (2026-01-29)

- **MQTT support**: New `--mqtt-config` option for publishing to MQTT broker
- **Topics**: `{base}/7seg/num`, `{base}/vol`, `{base}/source`, `{base}/mute`, `{base}/status`
- **Last Will**: Broker publishes "offline" to `{base}/status` on unexpected disconnect
- **Same trigger as stdout**: Publishes on state change OR every 60 seconds
- **Mute conversion**: UNMUTE→"off", MUTE→"on", MUTE_NA→"unknown"
- **TLS support**: Optional ca_cert for secure connections
- **Optional dependency**: Requires `paho-mqtt` package when enabled

### v2.4.3-beta (2026-01-29)

- **Blue channel LED detection**: Switch from grayscale to blue channel for brightness detection
- **Fix LED misdetection**: Grayscale diluted blue LED signal (255 blue → 170 gray), causing wrong LED detection
- **100% accuracy**: Tested on 169 LED issue frames, all correctly detected
- **New thresholds**: Blue brightness >200, gap >30 (was grayscale >150, gap >5)

### v2.4.2-beta (2026-01-29)

- **Simplified "1" penalty**: Removed complex segment A and uneven lighting checks
- **0 vs 1 fix**: Always penalize "1" matches on left 35% of digit box by 30%
- **Code reduction**: Removed 28 lines of broken logic that prevented penalty from triggering

### v2.4.1-beta (2026-01-29)

- **CLAUDE.md reminder**: Added prominent auto-compact reminder at top of CLAUDE.md

### v2.4.0-beta (2026-01-28)

- **Adaptive fps control**: New `--target-fps` option for time-based frame skipping
- **Auto-tuning**: Measures actual fps and adjusts skip interval to maintain target
- **Optimized skip**: Uses `grab()` for skipped frames (no decode, lower CPU)
- **Mutual exclusion**: `--skip` and `--target-fps` cannot be used together
- **Default drain 0**: Changed `--drain` default from 2 to 0 (skip clears buffer)

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
