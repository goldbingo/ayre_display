# 7-Segment Display Reader - Design Document

## Index

**How it works**
- [Processing Pipeline](#processing-pipeline) — Panel detection, slant correction, gap detection, digit recognition
- [LED Detection](#led-detection) — Button LEDs, mute LED
- [Frame Skip Optimization](#frame-skip-optimization)
- [LED Skip Optimization](#led-skip-optimization)
- [Dim Digit Enhancement](#dim-digit-enhancement)
- [Caching Strategy](#caching-strategy)

**Reference**
- [File Structure](#file-structure)
- [Key Classes](#key-classes)
- [Configuration Constants](#configuration-constants)
- [Dependencies](#dependencies)

**Operations**
- [Usage](#usage) — live_demo.py, segment_reader.py, calibrate_camera.py
- [Changing Cameras](#changing-cameras) — Calibration, coordinate system, 7-step guide
- [Logging System](#logging-system) — CSV, issue frames, iMessage alerts, MQTT

**Other**
- [Known Limitations](#known-limitations)
- [Changelog](#changelog)

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
    └── Corner template matching (75x75 templates, round-robin)
    └── LED dot detection via Otsu + connectedComponents in each button zone
    └── Requires corner + 3 buttons (B2, S1, S2) for homography
    └── Similarity transform in undistorted space → panel projection

**Corner Templates:** 75x75 grayscale templates (`templates/corner_1.png` etc.).
Round-robin with sticky preference — tries current template first, switches only
if it fails (score < 0.93). Matched in undistorted space via `undistort_roi()`.

**LED Dot Landmarks:** Each button's LED dot is detected using Otsu thresholding +
connectedComponents. Dark dots found first (unlit LEDs); blue mask fallback for
lit LEDs. Positions used as landmarks for homography fitting. Minimum 3 buttons
required to prevent misidentification at dawn/low light.

Tracking restore (--track): restore_golden()       → 'tracked'
    └── Reuses last known good homography when landmarks disappear
    └── Enabled via --track flag, off by default

Fallback: Calibrated position from camera_mount.json    → 'calibrated'
    └── Fixed panel position from camera mount calibration
```

**Homography:** A 2x3 similarity transform (4 DOF: translation, rotation, scale) fitted
from corner + button LED dot positions in undistorted space. With 3+ buttons, the
transform is overdetermined (least squares). An EMA-smoothed version (α=0.03) tracks
gradual camera drift. The homography projects device-space offsets to raw pixel
coordinates via `redistort_points()`.

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
3. **Interactive crop mode** opens — digit shown at 8x zoom with black border:
   - 4 green crop lines show the auto-trim result as initial crop
   - Click near a corner to select it — its H+V lines turn cyan (3px bold)
   - Arrow keys move the selected corner's lines ±1 original pixel
   - For digit `1`: a dashed yellow reference line shows target width (height/2)
   - **ENTER** saves the cropped region, **ESC** cancels without saving

**Saves to:** `templates/digit_{digit}{letter}.png`
- Letters `a-z` used sequentially (finds next unused)
- Example: `l6` → `digit_6a.png` (or `6b`, `6c`... if `6a` exists)

Templates auto-reload after saving.

## LED Detection

### Button LEDs (`detect_button_leds()` / `predict_panel_from_landmarks()`)

Detects which of 4 buttons (B1, B2, S1, S2) has its LED lit. Two paths:

**Primary: LED dot landmarks** (within `predict_panel_from_landmarks()`):
```
1. Find corner via template matching in undistorted space
2. Search button region (left of corner, below panel)
3. For each button zone, detect LED dot:
   a. Otsu threshold + connectedComponents → dark blob (unlit LED)
   b. Blue mask fallback (HSV H=85-130, area≥15) → lit LED
   c. Method tag: 'dark' (dot found) or 'lit' (blue blob, no dark dot)
4. Lit LED = button where method='lit' (by elimination)
   - Fallback: if all 'dark', compare brightness at dot positions
5. With homography, project B1 position and search there too
```

**Fallback cascade** (when landmark_dot unavailable, e.g. night with tracked/calibrated panel):
```
a. landmark_dot (primary — handled above)
b. Brightness: brightest zone by blue channel mean (val>200, gap>30)
c. Blob + brightness agreement: blob detected AND matches brightest zone
d. Center brightness: center-quarter zone mean (val>220, gap>5)
e. Blob in bright: blob found in a bright region (val>200)
```
Fallback methods use `_create_led_mask()` + `connectedComponents` for blob detection,
computed lazily only when landmark_dot fails.

**Frame-skipped frames:** `_refresh_led_dots(frame)` recomputes LED dots using cached button positions when full landmark detection didn't run. All 4 buttons (B2/S1/S2 detected + projected B1) are checked fresh each frame, so LED changes are caught even during frame skip.

**Key Constants:**
- `_BUTTON_REGION_RIGHT_RATIO = 0.65`
- `_BUTTON_REGION_TOP_RATIO = 0.70`
- Corner match threshold: 0.93

### Mute LED (`detect_red_button()`)

Detects red mute button state using local contrast (`_compute_mute_contrast()`):

```
1. Find corner template position
2. Project mute LED and reference patch positions via homography
3. Extract 13x13 patches (radius=6) at LED and reference positions
   - Reference patch is 26px left of LED in device space (same surface, no overlap)
4. Compute two metrics:
   - rr (red ratio): mean_red(LED) / mean_red(REF) — detects bright LED
   - re (red excess): (R-G)_LED - (R-G)_REF — detects red color through tint
5. Decision: MUTE if rr > 1.10 OR re > 10
```

**Why two metrics:** `rr` alone fails on red-tinted artifact frames (both patches get elevated red, suppressing the ratio). `re` subtracts out the tint by comparing R-G difference between patches. Together they catch all cases: `rr` handles bright LEDs, `re` handles dim LEDs on tinted backgrounds. On synthetic distorted images: UNMUTE max rr=1.01, re=7.3; MUTE min caught rr=0.94, re=10.2 — clean separation on both.

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

**Performance** (500 live frames, `--track`):
- Skip rate: ~99% when stable
- Skipped frame: 0.24ms
- Processed frame: 13.8ms
- Speedup: 56x per frame

**Reference Update:** When diff exceeds threshold, reference updates to current frame before processing. This keeps diff stable relative to recent frames rather than drifting from first frame.

## LED Skip Optimization

Skips LED detection when button zone appearance unchanged between frames. Independent of frame skip — both can apply to the same frame.

**Mechanism (`_led_diff_check()`):**
1. Take grayscale snapshots of each button LED zone with 2px hysteresis padding
2. On subsequent frames, compare zone against snapshot sub-region (handles small drift)
3. If max zone diff < threshold: skip LED detection, reuse previous result
4. If diff >= threshold: resnap all zones and run full LED detection

**Resnap triggers:**
- `threshold` — max zone diff exceeds threshold (default 5.0)
- `drift` — zone bounds shifted beyond 2px padding (B1 uses homography projection, inherently less stable)
- `cooldown` — N frames after threshold/drift resnap (default 2), catches LED changes that appear 1 frame after threshold crossing

**Cooldown logic:**
- Always set after threshold or drift resnap (LED may change on next frame)
- Also set when LED actually changes on any resnap frame (settling time)
- Configurable via `--led-skip-cooldown N`

**Leading-edge transitions:** LED changes often register as sub-threshold on frame N, then full diff on frame N+1. These are caught by cooldown and not counted as true misses.

**CLI options:**
- `--no-led-skip` — disable skip, detect LED every frame (enables diff experiment logging)
- `--led-skip-threshold N` — diff threshold (default 5.0)
- `--led-skip-cooldown N` — cooldown frames after resnap (default 2)

**Interaction with frame skip:** Both optimizations are independent. On a frame-skipped frame, `_refresh_led_dots()` provides LED results from cached button positions; LED diff check runs against those results. A frame can be both frame-skipped and LED-skipped simultaneously.

**Diff experiment logging** (`--no-led-skip --log`): Writes `logs/led_diff_experiment.csv` with per-frame zone diffs for threshold tuning:
```
timestamp, frame_n, B1_diff, B2_diff, S1_diff, S2_diff,
max_diff, lit_led, prev_lit, changed, resnap, threshold
```
Visualize with `python scripts/plot_led_diff.py [-m MINUTES]`.

**Performance** (daytime, stable):
- LED skip rate: ~90%
- Frame skip + LED skip combined: ~99% of frames need no full LED detection
- Full detect: ~16ms, LED-skipped: ~2ms

## Dim Digit Enhancement

Enhances dim digits before template matching:

```python
def _enhance_dim_digit(digit_img):
    gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    # Subtract background (5th percentile) to handle uneven lighting
    background = np.percentile(gray, 5)
    if background > 10:
        gray = np.clip(gray.astype(np.int16) - int(background), 0, 255).astype(np.uint8)
    if gray.max() < 150:  # Dim threshold (max only; mean unreliable)
        blue = digit_img[:, :, 0]
        bg_blue = np.percentile(blue, 5)
        if bg_blue > 10:
            blue = np.clip(blue.astype(np.int16) - int(bg_blue), 0, 255).astype(np.uint8)
        enhanced = cv2.normalize(blue, None, 0, 255, cv2.NORM_MINMAX)
        return enhanced, True
    return gray, False
```

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
├── segment_reader.py          # Core recognition library (~4400 lines)
├── live_demo.py               # Real-time camera monitoring
├── device_geometry.py         # Device geometry model (spatial constants)
├── calibrate_camera.py        # Camera calibration utility
├── watchdog.sh                # Process watchdog (restarts if hung)
├── DESIGN.md                  # This file
├── webcam.link               # RTSP/camera URL (gitignored, contains credentials)
├── .gitignore
│
├── templates/                 # Recognition templates
│   ├── corner_*.png           # Corner templates for localization (75x75, 3 variants)
│   └── digit_*.png            # Digit templates (0-9, P, X, multiple variants)
│
├── example/                   # Reference images (53) for batch testing
│
├── calibration/               # Camera/device calibration data
│   ├── camera.json            # Camera intrinsics
│   ├── camera_mount.json      # Camera mount parameters
│   ├── camera_mount_reference.png  # Annotated calibration reference image
│   └── device_model.json      # Device geometry model
│
├── scripts/                   # Tests, analysis, and utility scripts
│   ├── test_cache.py          # Cache behaviour tests (13 tests)
│   ├── test_distorted.py      # Perspective distortion tests (auto-generates images)
│   ├── test_geometry.py       # Device geometry unit tests (62 tests)
│   ├── test_tracking.py       # Landmark tracking stream tests (7 tests)
│   ├── test_mute_zone.py      # Mute zone stream simulation tests
│   ├── analyze_skip.py        # Frame-skip threshold analysis
│   ├── timing_analysis.py     # Pipeline and skip benchmarking
│   ├── update_device_model.py # Compute device_model.json from camera_mount.json
│   ├── gen_annotated.py       # Generate camera_mount_reference.png
│   ├── gen_perspective_variants.py  # Generate distorted test images
│   ├── calibrate_corner.py    # Interactive corner template capture/alignment
│   ├── calibrate_mount.py     # Mount calibration using segment_reader detection
│   └── test_image.py          # Re-test image(s) with current code
│
├── foscam-c2/                 # Reference camera snapshots
├── logs/                      # Runtime logs and issue frames (display mode)
│   └── headless/              # Headless mode logs (when --log passed)
├── debug/                     # Per-image debug output (generated)
├── distorted/                 # Generated distortion test images
└── legacy/                    # Old versioned files (not tracked)
```

## Key Classes

### `SegmentReader`

Main API — `detect()` runs the full detection pipeline in a single call:

```python
reader = SegmentReader()
result = reader.detect(frame)        # FrameResult with all detection outputs
result = reader.detect(frame, debug=True)  # Also generates step-by-step debug images
reading, changed = reader.read(frame)  # Digits only (used internally by detect)
reader.reset_cache()  # Force re-detection
```

**`FrameResult` fields:**
- `reading`, `cache_hit` — digit recognition result
- `led_status`, `mute_status` — LED and mute detection
- `corner_result`, `corner_debug` — corner template match
- `led_debug_info`, `mute_debug_info` — debug info (None during washout)
- `noise_mean`, `washout` — overexposure detection
- `panel_rect`, `detection_method` — panel location and method used
- `last_led_debug`, `last_mute_debug` — cached from last non-washout frame

**Properties:**
- `last_reading` - Most recent successful reading
- `confidence` - (left_score, right_score) tuple
- `digit_debug` - Debug info for last recognition (includes `gap_debug`, `boxes_debug` when debug=True)

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
_LED_MIN_AREA = 60
_LED_MAX_AREA = 1200
_LED_MAX_ASPECT_RATIO = 3

# Digit rejection / ambiguity thresholds
_REJECTION_MIN_SCORE = 0.75             # reject if score below this AND gap small
_REJECTION_MAX_GAP = 0.20               # reject if gap below this AND score low
_REJECTION_EXTREME_GAP = 0.02           # reject if gap below this regardless of score
_AMBIGUOUS_MAX_SCORE = 0.95             # only flag ambiguous if best score below this
_QUICKCHECK_DRIFT = 0.02                # trigger full rescan if score drifts more than this
```

## Dependencies

- **OpenCV** (`cv2`) - Image processing, template matching
- **NumPy** - Array operations
- **Python 3.8+**
- **paho-mqtt** (optional) - MQTT publishing for home automation

## Logging System

### Detection CSV (`logs/detection.csv`)

Logs every frame with 32 columns:

```
timestamp, panel_x, panel_y, panel_w, panel_h, gap_x,
left_score, right_score, reading, led_status,
corner_score, corner_tmpl, detection_method,
mute_status, dim_enhanced, frame_skip, diff_edge, diff_mode,
led_method, proc_ms, issue,
geo_method, geo_scale, geo_rotation, undistort_px,
noise_mean,
mute_rr, mute_re, mute_gr, mute_led_r, mute_ref_r, mute_h_age
```

**Key fields:**
- `corner_tmpl`: Which corner template matched (index)
- `led_method`: Which method detected LED (landmark_dot/brightness/blob/center)
- `mute_rr`: Red ratio (LED_R / REF_R) — MUTE if > 1.10
- `mute_re`: Red excess ((R-G)_LED - (R-G)_REF) — MUTE if > 10
- `mute_led_r`/`mute_ref_r`: Raw red channel means for LED and reference patches
- `mute_gr`: Gray ratio (LED_gray / REF_gray)
- `mute_h_age`: Frames since last homography update
- `proc_ms`: Processing time for the frame
- `geo_method`/`geo_scale`/`geo_rotation`: Geometry transform info
- `undistort_px`: Max pixel shift from lens undistortion

### Issue Frame Capture

Every capture produces two files — raw frame and debug overlay:
- `timestamp_issue.png` — raw frame (640x480 single, or N×640x480 composite)
- `timestamp_issue_overlay.png` — debug overlay with same dimensions
- `timestamp_issue.txt` — metadata (scores, LED/mute state, corner info, etc.)

The overlay is generated every non-skipped frame and stored in `frame_history` alongside the raw frame. This ensures composites (which include past frames) always have overlays available.

```python
_capture_issue(raw_frame, overlay_frame, 'low_conf', debug_info, confidence=0.75)
_capture_composite(raw_composite, overlay_composite, 'led_glitch', debug_info)
```

**Issue Types:**
- `corner_low_score` - Corner match score between 0.85–0.93
- `digit_1_penalty` - Digit "1" low confidence with "7" close
- `gap_ambiguous` - Close gap valley scores (ratio < 1.2)
- `gap_wide_valley` - Gap valley wider than 9px
- `led_fail` - LED detection failed (not during transitions)
- `led_fallback` - Blob or center LED fallback method activated
- `led_glitch` - B1/B2 flicker pattern detected
- `mute_glitch` - Single-frame mute status flip (A→B→A pattern)
- `mute_homography_outlier` - Raw vs smoothed homography mute position >5px
- `mute_na` - MUTE detection returned NA
- `reading_glitch` - Single-frame reading change (A→B→A pattern)

**Cooldown:** 30 seconds between saves of same issue type.

**Image formats:**
- 640x480: single raw frame
- 1280x480: raw|overlay pair side-by-side (unified capture format)
- 3200x480: 5 raw frames side-by-side (each 640x480)
- 7040x480: context composite — 11 frames at 640x480, issue frame at index 5

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
    "broker": "mqtt.example.com:1883",
    "base_topic": "home/ayre"
}
```

Optional fields: `"user"` and `"password"` for authentication, `"ca_cert": "/path/to/ca.crt"` for TLS (use port 8883).

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

# Camera source: direct URL, custom file, or default (webcam.link)
python live_demo.py --camera rtsp://user:pass@192.168.1.100:554/videoMain
python live_demo.py --camera-file /path/to/camera.link
python live_demo.py   # reads from webcam.link (default)
```

**Keys** (in `--display` mode): `q` quit, `c` reset cache, `s` save frame, `l#`/`r#` learn digit template with interactive crop (e.g. `l6` learns left digit as 6).

### `segment_reader.py` — Batch test on example images

```bash
# Runs all 53 example/ images through the pipeline
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

### Changing Cameras

When replacing the camera with a different model, several calibration files need updating. The system has three layers of calibration data, each serving a different purpose:

| File | What it stores | When to update |
|------|----------------|----------------|
| `calibration/camera.json` | Lens intrinsics (focal length, distortion) | New camera model or lens |
| `calibration/camera_mount.json` | Physical mounting position (corner, buttons) | Camera repositioned or replaced |
| `calibration/device_model.json` | Device-space geometry (panel/button offsets) | Only if the physical display hardware changes |
| `templates/corner_*.png` | Corner template images (75x75) for localization | Camera replaced or repositioned significantly |

#### Coordinate system

All pixel coordinates use the standard image convention: **(0, 0) is the top-left corner** of the frame. X increases rightward, Y increases downward.

```
(0,0) ───────── X+ ──────── (639,0)
  │                            │
  │     ┌──panel──┐            │
  Y+    │  7-seg  │    ●corner │
  │     └─────────┘            │
  │   [B1] [B2] [S1] [S2]     │
  │                       ●mute│
(0,479) ──────────────── (639,479)
```

- **Frame size**: 640x480 pixels (from RTSP feed)
- **corner_xy**: Where the corner template matches — the primary reference point. All other positions are measured relative to this
- **panel_offset** in `device_model.json`: `[-262, -90]` means the panel top-left is 262px *left* and 90px *above* the corner (negative = left/up)
- **mute_button_offset**: `[200, 43]` means the mute LED is 200px *right* and 43px *below* the corner (positive = right/down)
- **landmarks**: Button center offsets from corner, e.g. `B2: [-298.0, 108.0]` means B2 is 298px left and 108px below the corner

When measuring positions in an image editor, the coordinates shown (typically in the status bar) follow this same convention — (0,0) at top-left.

#### Step 1: Capture checkerboard images

Print a checkerboard calibration pattern (default: 9x6 inner corners) and photograph it from 10-20 different angles and distances with the new camera at its **native capture resolution**. Requirements:

- Use the camera's full native resolution (not the scaled RTSP feed)
- Cover different angles: tilted left/right, up/down, rotated
- Cover different distances: close, medium, far
- Ensure the full checkerboard is visible in every image
- Good even lighting, no shadows on the pattern
- Save images as PNG or JPG in a dedicated directory

```bash
mkdir new-camera-cal/
# Copy or transfer captured images into new-camera-cal/
```

#### Step 2: Run camera calibration

```bash
python calibrate_camera.py --images new-camera-cal/ --output calibration/camera.json
```

The script will:
1. Detect checkerboard corners in each image (reports which images succeed)
2. Compute camera matrix (focal length `fx`/`fy`, principal point `cx`/`cy`) and distortion coefficients (`k1`, `k2`, `p1`, `p2`, `k3`) at native resolution
3. Transform intrinsics to 640x480 feed resolution
4. Save to `calibration/camera.json`

Check the output for:
- **RMS reprojection error**: Should be below 1.0 pixel (ideally below 0.6). If above 1.0, remove blurry or poorly-lit images and re-run
- **Images used**: At least 10 images with detected corners recommended
- **k1 (barrel distortion)**: Negative = barrel, positive = pincushion. Wide-angle cameras will have strong negative k1

**Important: feed pipeline adjustment.** The script's `transform_intrinsics()` function hardcodes the Foscam C2 feed pipeline:

```
Native 1920x1080 → center crop to 1440x1080 (4:3) → scale to 640x480
```

If the new camera has a different pipeline, you must edit `transform_intrinsics()` in `calibrate_camera.py`:
- **Different native resolution**: The crop and scale math adjusts automatically based on `native_size` and `target_size`, but the center-crop-to-4:3 assumption may not apply
- **No center crop** (camera delivers 640x480 directly): Skip the crop step — set `crop_x = 0`, `crop_w = native_w`
- **Different aspect ratio crop**: Adjust the `crop_w` / `crop_h` calculation to match how your camera/RTSP server transforms the stream

You can verify the pipeline by comparing a frame grabbed at native resolution vs the RTSP feed to see what cropping/scaling is applied.

#### Step 3: Update RTSP stream URL

The stream address is read from `webcam.link` (a single-line file in the project root, gitignored since it may contain credentials):

```bash
echo 'rtsp://user:pass@192.168.1.100:554/videoMain' > webcam.link
```

Supports RTSP URLs, HTTP URLs, or a local camera index (e.g., `0`). The feed must deliver 640x480 frames (or update `frame_size` in `camera.json` and `device_model.json` if using a different resolution).

#### Step 4: Capture new corner templates

The corner template is a small image patch used to locate the display in each frame. With a new camera, the appearance will differ due to lens characteristics, resolution, and viewing angle.

1. Start the live feed in display mode:
   ```bash
   python live_demo.py --display
   ```

2. Press `s` to save the current frame (saved to `logs/`)

3. Open the saved frame and crop a ~75x75 pixel patch with the corner feature at the top-left of the crop (the distinctive visual anchor point near the top-right of the panel). The match position equals the corner position — no offset needed.

4. Save as `templates/corner_1.png`. Optionally capture 2-3 variants under different lighting:
   - `templates/corner_1.png` — primary (normal lighting)
   - `templates/corner_2.png` — variant (dimmer or different exposure)
   - `templates/corner_3.png` — variant (night or bright)

   Alternatively, use the interactive calibration tool:
   ```bash
   python scripts/calibrate_corner.py
   ```

5. Test corner detection:
   ```bash
   python segment_reader.py
   ```
   All example images should find the corner. If using new example images, save several to `example/` with filenames matching the pattern `{reading}-{LED}-{MUTE}.PNG` (e.g., `27-B2-UNMUTE.PNG`).

#### Step 5: Calibrate camera mount positions

Use the interactive calibration tool to click on 8 landmarks. Save a frame first (`s` in display mode), then:

```bash
python scripts/calibrate_mount.py logs/manual_*.png
```

The tool displays the image at 2x zoom with a magnified crosshair. Click on each landmark in order:

| # | Landmark | What to click |
|---|----------|---------------|
| 1 | **corner** | The distinctive feature near the top-right of the panel (knob edge) |
| 2 | **B1** | Center of the B1 button LED |
| 3 | **B2** | Center of the B2 button LED |
| 4 | **S1** | Center of the S1 button LED |
| 5 | **S2** | Center of the S2 button LED |
| 6 | **mute LED** | Center of the red mute LED |
| 7 | **digit top-left** | Top-left corner of the digit display area |
| 8 | **digit bottom-right** | Bottom-right corner of the digit display area |

The tool updates `calibration/camera_mount.json` and recomputes `calibration/device_model.json` automatically.

**Reference image:** `calibration/camera_mount_reference.png` shows all annotated positions. Regenerate with:

```bash
python scripts/gen_annotated.py logs/saved_frame.png
```

![Camera mount calibration reference](calibration/camera_mount_reference.png)

**Note:** All values are in pixel-space at 640x480. If the physical hardware layout hasn't changed (same device, just a different camera), the offsets may only need minor adjustment for field-of-view differences.

#### Step 6: Recapture digit templates (if needed)

If the new camera produces noticeably different digit appearance (different sharpness, color balance, or viewing angle), existing digit templates may not match well. Use manual template learning:

1. Run `python live_demo.py --display`
2. When a digit is misrecognized, press `l` or `r` (left/right digit) then the correct digit key (`0-9` or `P`)
3. Templates save to `templates/digit_{digit}{letter}.png`

#### Step 7: Verify

Run the full test suite:

```bash
# All example images must pass (update example/ images if camera changed)
python segment_reader.py

# Geometry tests (62 assertions)
python scripts/test_geometry.py

# Live test with full pipeline
python live_demo.py --display --log --track```

Check that:
- Corner detection finds the template reliably (score > 0.85)
- Panel position is correct (digits visible in debug overlay)
- LED detection picks the correct button
- Mute LED detection works in both lit and unlit states

#### Summary: what to update for common scenarios

| Scenario | camera.json | camera_mount.json | device_model.json | corner templates | digit templates |
|----------|:-----------:|:-----------------:|:-----------------:|:----------------:|:--------------:|
| Same camera, same position | - | - | - | - | - |
| Same camera, repositioned | - | Maybe¹ | - | Maybe³ | - |
| New camera, same position | Yes | Yes | Maybe² | Yes | Maybe⁴ |
| New camera, new position | Yes | Yes | Yes | Yes | Maybe⁴ |
| New display hardware | - | Yes | Yes | Yes | Yes |

¹ Only if landmarks fail to detect (corner template no longer matches or features fall outside search regions). The homography auto-corrects for small shifts.
² Only if the new lens has a significantly different FOV, causing pixel distances between features to change beyond what undistortion and homography can correct.
³ Only if the corner appearance differs enough at the new angle that template matching drops below 0.85.
⁴ Only if digit appearance differs (sharpness, color balance, viewing angle) enough that template matching fails.

#### Detection threshold tuning (after calibration)

If detection fails after calibration, tune these hardcoded thresholds by priority:

**P1 — Brightness thresholds** (sensor gain/exposure dependent, check CSV log values):

| Threshold | Current | Location | Effect if wrong |
|-----------|---------|----------|-----------------|
| Washout `noise_mean` | 180 | `segment_reader.py` | False washout skips LED/mute |
| Dim digit `raw_max` | 150 | `_enhance_dim_digit()` | Dim digits not enhanced |
| LED brightness | 200/220 | `detect_button_leds()` | Fallback LED detection fails |
| Mute red excess | 10 | mute contrast check | False mute or missed mute |

**P2 — Color response** (only if sensor has different white balance, inspect LED HSV values):

| Threshold | Current | Location |
|-----------|---------|----------|
| Blue hue range | H=85-130 | `_find_led_in_button()`, blob detection |
| Blue saturation (3 tiers) | S=80/100/150/200 | bright/normal/dim LED tiers |
| Blue value (3 tiers) | V=50/80/100/240 | bright/normal/dim LED tiers |
| Segment lit ratio | 0.15 | `_check_segment()` |

**P3 — Resolution dependent** (scale by old/new resolution ratio):

| Threshold | Current | Scaling |
|-----------|---------|---------|
| Blob area range | 60-1200 px² | resolution² |
| Frame diff threshold | 100,000 | resolution² |
| LED diff threshold | 5.0 | zone pixel count |
| Digit min dimensions | 5×10 px | linear |

**Universal (don't touch)**: template match scores/gaps, segment positions A-G, C-B diff thresholds, aspect ratios, EMA alpha, cooldowns, score boost/penalty factors. These are algorithm-level, not camera-dependent.

See [issue #71](https://github.com/goldbingo/ayre_display/issues/71) for the full threshold inventory.

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
python scripts/test_mute_zone.py    # Mute zone stream simulation

# Re-test image(s) with current code (supports composite multi-frame images)
python scripts/test_image.py path/to/image.png              # single image
python scripts/test_image.py logs/                           # all PNGs in dir
python scripts/test_image.py --save --no-display image.png   # save without window

# Analysis tools
python scripts/analyze_skip.py                      # Skip rate from detection.csv
python scripts/timing_analysis.py --live -n 500     # Pipeline breakdown
python scripts/timing_analysis.py --skip -n 500     # Frame skip measurement
python scripts/timing_analysis.py --skip --track -n 500
```

## Known Limitations

1. **Slant angle** - Fixed at 8.0°, not auto-detected
2. **Two digits only** - Hardcoded for 2-digit display
3. **Dawn startup delay** - Requires 3 button LED dots visible for homography; at dawn only corner-based fallback available until enough ambient light reveals buttons
4. **Single camera model** - Camera calibration pipeline hardcoded for Foscam C2 feed (1920x1080 → center crop → 640x480); different cameras need `transform_intrinsics()` adjustment

## Changelog

### v4.3.0 (2026-02-18)

- **Validate detected buttons against projected positions (#83)**: In dawn/dim lighting, contour detection finds shifted button boxes (15px+) which corrupt homography. Reject buttons >8px from projected position, reconstruct from projection via Step 5, with fallback to original detected buttons if all projections also fail.

### v4.2.0 (2026-02-18)

- **Homography quality check with reprojection residuals (#85)**: Compute per-landmark residuals after similarity fit; capture issue frames when max residual exceeds 3px to detect misidentified dots.
- **Skip capture-only triggers without --log**: Guard 6 issue triggers (corner_low_score, gap_ambiguous, gap_wide_valley, homography_quality, mute_homography_outlier, led_fallback) behind `args.log` to avoid wasted CPU.
- **Fix dark dot detection in dawn lighting (#84)**: Percentile-threshold second pass in `_find_led_in_button()` isolates LED dots merged with surrounding dark pixels in low-contrast conditions.

### v4.1.0 (2026-02-18)

- **Exclude unfound LED dots from homography (#82)**: Button center positions (22px off) were corrupting the similarity transform, skewing mute and B1 projections. Only accurately-detected dot positions now feed `compute_homography()`; unfound dots use projected positions for debug overlay only.

### v4.0.9 (2026-02-17)

- **DESIGN.md consistency overhaul**: Update panel detection cascade (remove stale fallbacks), LED detection fallback cascade, mount calibration workflow (calibrate_mount.py), CSV fields, issue types, image formats, LED skip documentation.
- **Remove deprecated `--undistort` flag**: Always-on; removed from all scripts and watchdog.sh.

### v4.0.8 (2026-02-17)

- **Fix stale LED highlight in overlay on frame-skipped frames**: Rebuild `leds` dict to match current `lit_led` when reusing cached debug info.
- **LED diff plot improvements**: Yellow lines from leading-edge transitions to following resnap; leading-edge filter relaxed (next frame resnap sufficient, no need for changed=1).

### v4.0.7 (2026-02-17)

- **LED skip optimization (#70)**: Skip LED detection when button zone appearance unchanged between frames. Grayscale zone snapshots with 2px hysteresis padding, threshold-based resnap with cooldown. ~90% LED skip rate (independent of frame skip). CLI: `--no-led-skip`, `--led-skip-threshold`, `--led-skip-cooldown`.
- **Always cooldown after threshold/drift resnap**: Catches LED changes that appear 1 frame after threshold crossing. Leading-edge transitions excluded from missed count in plot.
- **LED fail transition suppression**: Defer LED-fail capture by 1 frame; ignore NA between different LEDs (normal transition).
- **Capture blob/center LED fallback frames** (#78): Log images when fallback methods activate for analysis.
- **Remove deprecated `--undistort` flag**: Undistortion is always enabled. Removed from all scripts and watchdog.sh.

### v4.0.6 (2026-02-17)

- **Skip LED fallback pre-computation when landmark succeeds (#78)**: Move `_create_led_mask()`, `connectedComponents`, and blob/brightness/center fallback methods inside the landmark-failure branch. ~95% of frames now skip this unnecessary computation.

### v4.0.5 (2026-02-16)

- **Fix B1 LED dot detection**: Use full projected search rect for B1 instead of right-half crop. B1's search area is centered on the homography-projected LED position — cropping to the right half discarded context needed for blob isolation near the frame edge.
- **Fix stale LED on frame-skipped frames**: New `_refresh_led_dots(frame)` recomputes LED dots using cached button positions when `predict_panel_from_landmarks()` didn't run. All 4 buttons (B2/S1/S2 + projected B1) detected fresh each frame.
- **Skip redundant `detect_button_leds()` on frame-skipped frames**: `_refresh_led_dots()` already provides LED results; skipping the redundant call reduces per-frame cost from 0.81ms to 0.37ms.
- **Suppress ambiguous captures during PP↔digit transitions**: Skip ambiguous/low_conf captures when reading history shows PP↔digit mix or current reading is XX.
- **Overlay arrows**: Predicted (yellow) LED arrows point upward from below; detected (green/orange) arrows point downward from above.

### v4.0.4 (2026-02-16)

- **Unified detection flow (#79)**: New `SegmentReader.detect(frame)` method combines digits + corner + LED + mute in a single call, returning a `FrameResult` dataclass. Eliminates duplicated detection logic between `live_demo.py` and `test_on_image()`. Corner detection now cached from `predict_panel_from_landmarks()` instead of running twice per frame.
- **Debug parameter for `read()`**: `read(frame, debug=True)` passes `debug=True` to `find_digit_gap()` and `define_digit_boxes()`, storing debug images in `digit_debug` dict. Zero overhead when `debug=False` (production).

### v4.0.3 (2026-02-16)

- **Landmark-first LED detection**: Flip LED detection priority — landmark dot method (Otsu + connectedComponents) checked first as primary; old brightness/blob/center methods become fallback when landmarks unavailable.
- **Clean CSV fields (55→33)**: Remove 22 obsolete fields from old mute detection (#64/#67/#72 analysis), old red-pixel/clustering method, smoothed/raw position comparison, panel stats. Keep `noise_mean` for dawn light-level correlation.
- **Remove dead code**: `_detect_red_pixels()`, `mute_proj_outlier` capture, `mute_rr_night` capture, MUTE_NA from scattered pixels.

### v4.0.2 (2026-02-16)

- **Remove mute arrow overlay**: Yellow arrow removed from main frame; mute LED info shown in zoom inset only.

### v4.0.1 (2026-02-16)

- **3-button minimum for homography**: Require corner + 3 buttons before computing homography. Prevents bad mute projection at dawn when only 1 LED dot is visible and misidentified (B2 detected as S2). Falls through to corner-only panel detection until enough light reveals all buttons.
- **Fix stream image test crash**: Extract raw 640x480 half from 1280x480 images in `test_on_image()` to avoid undistort map overflow. Crop 4 stream example images to raw frames.

### v4.0.0 (2026-02-16)

- **Merge dev branch** (`feature/65-corner-calibration`): Corner calibration, LED landmarks, undistorted-space homography, local contrast mute detection — all merged to main.
- **Local contrast mute detection** (#72): Replace old red-pixel/clustering detection with contrast-based method. Two metrics: `rr` (red ratio = LED_R / REF_R, threshold 1.10) and `re` (red excess = (R-G)_LED - (R-G)_REF, threshold 10). Decision: MUTE if either exceeds threshold. Validated on 1.4M frames with zero misclassification (MUTE rr 1.03+, UNMUTE rr ≤0.98).
- **Corner calibration** (#65): Interactive corner template capture tool, 75x75 templates, undistorted-space template matching.
- **LED dot landmarks**: LED dot detection with Otsu + connectedComponents, projected positions for homography, method-based lit/unlit determination.
- **Undistorted-space homography**: Similarity transform fitted in undistorted space, redistortion for raw-domain projection.
- **Unified issue capture**: Always store overlay in frame_history, save raw + overlay as separate files for all captures (headless and display).
- **Mute debug overlay**: Zoom inset (4x) showing LED and reference patches with rr/re values.
- **Larger mute patches**: Patch radius 4→6 (9x9 → 13x13), reference offset -18→-26px.
- **Undistorted-domain synthetic warps**: `gen_perspective_variants.py` applies transforms in undistorted space via single-pass remap.
- **Button center fallback**: When `_find_led_in_button()` fails, use button center for homography instead of skipping.

### v3.9.20 (2026-02-12)

- **Local contrast mute detection A/B logging** (#72): Add `_compute_mute_contrast()` that computes LED-vs-reference patch red and gray ratios using homography-projected positions. Reference patch placed 18px left of LED in device space (same button surface, clears glow halo). Values logged to CSV alongside old detection method (11 new fields: `mute_rr`, `mute_gr`, `mute_led_r`, `mute_ref_r`, `mute_led_sx/sy`, `mute_led_rx/ry`, `mute_ref_sx/sy`, `mute_h_age`). Old method remains decision-maker.
- **Smoothed homography** (#72): EMA-smoothed 2x3 affine matrix (α=0.03, ~33 frame time constant) in `DeviceGeometry`. Both raw and smoothed LED projections logged for jitter comparison. Smoothed homography initialized from `camera_mount.json` and resets on golden restore. `increment_homography_age()` tracks staleness.
- **Mute contrast overlay**: Crosshairs at smoothed LED center (red) and reference center (cyan), with red ratio text near mute region. Additive to existing mute overlay.

### v3.9.19 (2026-02-12)

- **Mute proj/det CSV fields for #72 validation**: Log `mute_proj_x,mute_proj_y` (homography-predicted LED center) and `mute_det_x,mute_det_y` (detected LED centroid) per frame. Calibrated `mute_button_offset` from [200,43] to [195,37] based on 119-sample analysis. Added `mute_proj_outlier` issue capture when offset > 5px.
- **Washout overlay improvements**: Draw LED/mute detection zones with dashed lines during washout using cached last-good debug info. Add red "WASHOUT" banner. Suppress panel_fail, gap issue logging during washout. Defer gap issues to display path for raw|display image pairs. Always draw corner search window even when detection fails.
- **Gap ambiguous ratio threshold**: Change from absolute `valley_diff < 1000` to ratio `second_val/best_val < 1.2`. No example images trigger; only flags genuinely ambiguous gap positions.

### v3.9.18 (2026-02-09)

- **Washout guard for overexposure** (#63): Skip LED and MUTE detection when `noise_mean > 180` (neutral housing region saturated). Reports LED=NA, MUTE=MUTE_NA, suppresses `led_fail` and `mute_na` issue logging and notifications. Adds `get_noise_mean()` utility function. Prevents false NA reports during brief headlight/sun flashes (~44 washout frames/day observed across 3 events).

### v3.9.17 (2026-02-08)

- **Mute diagnostic fields for #64 analysis**: Add 6 new CSV fields — `mute_red_mean_v`, `mute_blob_count`, `mute_cluster_density`, `mute_red_bias`, `mute_noise_std`, `mute_noise_mean`. Noise fields measure camera gain noise from a neutral 40x40 device housing region (`DeviceGeometry.get_noise_region()`), free from LED and display segment contamination. Provides clean index for adaptive mute detection thresholds during dawn high-gain conditions.

### v3.9.16 (2026-02-08)

- **Fix 6→8 misread from blue channel saturation**: Replace blue_ratio and `_check_segment_lit` with grayscale B vs C comparison for segment B detection. Blue channel saturates from LED glow, making unlit segments appear lit. Grayscale preserves contrast since glow is narrow-band blue. Check 1 (close scores, gap<0.07): C-B threshold 45. Check 2 (any gap): C-B threshold 38.

### v3.9.15 (2026-02-07)

- **Homography-based mute detection zone** (#69): Load initial homography from `camera_mount.json` at startup so mute detection always uses projected coordinates. Stop resetting `_homography` every frame — persistent homography survives blackouts and overexposure without `--track` mode. Fallback to fixed region only when calibration file is missing.

### v3.9.14 (2026-02-07)

- **Known wide-valley digits expanded**: Suppress gap_wide_valley for right digit `1`, `3`, `7` — all have open segments near the gap boundary producing expected wide valleys.

### v3.9.13 (2026-02-07)

- **Suppress transition noise in issue logging** (#68): Skip reading_glitch, invalid_reading, and gap_wide_valley logs during display transitions (rapid countdowns, value changes). All 14 reading anomalies and 54 gap_wide events in 2.99M rows were display transitions — zero real glitches. Filters: `rh[-4] != rh[-3]` for reading glitches, `prev_reading != reading` for invalid readings, 3+ distinct values for gap_wide.
- **Known wide-valley digits**: Suppress gap_wide_valley for right digit `1` (narrow digit creates expected wide gap). Raise valley width threshold from 8 to 9 pixels.
- **Full-resolution context composites**: Issue context images saved at full 640x480 per frame (was 33% downscaled), enabling re-testing with current code.

### v3.9.12 (2026-02-07)

- **Fix scattered artifact mute glitch** (#67): When clustering check fails due to a small artifact inflating the bounding box but brightness_gap > 100 confirms LED is lit, override to MUTE. Validated against 2.86M rows (0 false positives in 1.49M legit UNMUTE frames) and 225 glitch frames from two dawn bursts (44/44 fixed).
- **Fix dark-region brightness fallback** (#64): Use `medianBlur(gray, 3)` before `max()` to filter single-pixel noise spikes while preserving real LED signal. Replaces raw `max()` which was susceptible to hot pixels.
- **Fix test_on_image crash**: Handle `_find_corner` returning `(None, None, 0.0)` with `return_debug=True` instead of silently catching TypeError.
- **Empty region guard**: `detect_red_button` returns False gracefully when mute region is empty (extreme distortions only).

### v3.9.11 (2026-02-04)

- **Fix dim enhancement false trigger**: Check grayscale max before background subtraction in `_enhance_dim_digit()`. Bright glow-flooded images (high floor, low contrast) were falsely triggering blue channel enhancement after background subtraction dropped max below 150. Now uses raw max to decide.
- **Debug overlay shows grayscale**: Digit crops and gap panel in `draw_display_overlay()` now show the grayscale image that `matchTemplate` actually sees, instead of the raw BGR crop.

### v3.9.10 (2026-02-03)

- **Auto-compute device_model offsets**: New `scripts/update_device_model.py` reads 7 measured pixel positions from `camera_mount.json` and computes all `device_model.json` offsets automatically. Replaces manual subtraction workflow. Supports `--dry-run`.
- **B1 now a mandatory measurement**: Added B1 center and mute_center to `camera_mount.json`. Updated reference image and DESIGN.md to show 7 measurement points (was 6).
- **Camera source CLI options**: Added `--camera URL` and `--camera-file PATH` (mutually exclusive) to `live_demo.py`. Default still reads `webcam.link`.
- **Self-contained reference image generator**: `scripts/gen_annotated.py` reads all positions from `camera_mount.json` — no hardcoded coordinates.
- **Auto-generate distorted test images**: `test_distorted.py` calls `gen_perspective_variants.py` automatically when `distorted/` is missing or empty.
- **DESIGN.md improvements**: Grouped index, calibration update table with footnotes, MQTT minimal config, documented `webcam.link`.

### v3.8 (2026-02-03)

- **Separate log directories**: Display mode logs to `logs/`, headless mode logs to `logs/headless/`. Prevents interleaving of manual-session and production log files.
- **Per-mode PID files**: Each mode writes its own PID file (`/tmp/live_demo_display.pid` or `/tmp/live_demo_headless.pid`). On startup, verifies stale PID via `ps` before killing, with `atexit` cleanup.
- **Watchdog --log removed**: Headless production runs no longer pass `--log`, avoiding unnecessary file logging.

### v3.7 (2026-02-03)

- **Consolidate duplicated thresholds**: Replaced hardcoded magic numbers (0.05, 0.75, 0.20, 0.02, 0.95) with named constants (`_REJECTION_MIN_SCORE`, `_REJECTION_MAX_GAP`, `_REJECTION_EXTREME_GAP`, `_AMBIGUOUS_MAX_SCORE`, `_QUICKCHECK_DRIFT`). Wired up existing but unused `_TEMPLATE_AMBIGUITY_GAP` constant.
- **Extract `_quick_check_digit()` helper**: Deduplicated left/right digit quick-check blocks (~60 lines removed) into a single method on `SegmentReader`.

### v3.6 (2026-02-03)

- **Per-frame debug metadata in glitch logs**: Glitch diagnostics (LED, reading, mute) now include metadata for every frame in the composite image, not just the current frame. Each frame's scores, LED/mute status, reading, panel info etc. are prefixed with role labels (e.g. `before3/left_score`, `glitch/reading`, `after/led_status`), enabling comparison of what changed in the glitch frame vs stable frames.

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
- **Timing analysis `--skip` mode**: New streaming mode captures real frames through `SegmentReader.read()` to measure actual skip rate. Added `--track` flag.
- **Updated DESIGN.md**: Comprehensive file structure, caching strategy rewritten, architecture diagram updated.

### v3.0 (2026-02-01)

- **Camera calibration & geometry model**: New `device_geometry.py` module with `DeviceGeometry` class. Loads device model from `calibration/device_model.json`. Supports homography-based projection, similarity transform (de-rotation + scale normalization), and lens undistortion via camera intrinsics.
- **De-rotation & scale normalization**: Panel crop uses similarity transform derived from homography to correct camera tilt and distance variation. Logged as `geo_method`, `geo_scale`, `geo_rotation` in CSV.
- **Lens undistortion**: Always-on ROI de-warping using camera intrinsics from `calibration/camera.json`. Undistortion logged as `undistort_px` (max pixel shift).
- **Landmark tracking (`--track`)**: Stores golden landmark positions when detected and reuses them during blackout/overexposure. Detection cascade: `landmark` → `tracked` → `corner` → `brightness`. Golden state updates when any landmark moves >5px (camera bump).
- **Corner detection improvements**: Lowered match threshold from 0.90 to 0.85. Skip matching when search region is too dark or overexposed. New `corner_template_3.png`.
- **Fix gap detection false valleys**: `_find_valley` returns whether a true local minimum was found. Falls back to center instead of picking a point on the slope. Fixes `14` → `11` glitches during dim lighting.
- **Adaptive frame diff**: 3-channel diff (100K threshold, ~93% skip) with periodic blue-only probing. Logged as `diff_mode` in CSV.
- **Debug overlay**: `test_on_image` now renders debug overlay matching live_demo display.
- **New template**: `digit_9f` for night-glowy 9 variant.
- **Distortion test suite**: 34 warp variants (17 perspective + 17 affine) per source image. Dark images excluded from distortion generation. 100% pass rate.
- **New files**: `device_geometry.py`, `calibrate_camera.py`, `test_tracking.py`

