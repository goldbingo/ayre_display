# 7-Segment Display Reader - Design Document

## Overview

This system reads 2-digit numbers from a 7-segment LED display via camera feed. It's designed for real-time monitoring of equipment displays (e.g., audio mixers, industrial panels).

**Key Features:**
- Real-time digit recognition from video stream
- Button LED state detection (B1, B2, S1, S2)
- Mute button (red LED) detection
- Auto-learning for new digit variants
- Adaptive caching for performance

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
    └── Triangulation from known geometry

Fallback 1: Corner-only detection
    └── Uses _CORNER_TO_PANEL_X/Y offsets

Fallback 2: Brightness-based detection
    └── Thresholds top 3% brightness
    └── Finds contours in valid region
```

**Key Constants:**
- `_PANEL_WIDTH = 165`, `_PANEL_HEIGHT = 105` (fixed from calibration)
- `_CORNER_TO_PANEL_X = 266`, `_CORNER_TO_PANEL_Y = 86`

### 2. Slant Correction (`correct_slant()`)

The LED digits are italicized. A fixed 8.0° shear transform corrects this:

```python
# Shear matrix for slant correction
M = [[1, -tan(angle), offset], [0, 1, 0]]
```

### 3. Digit Gap Detection (`find_digit_gap()`)

Finds the vertical gap between two digits using column projection:

```
1. Create blue mask of corrected panel
2. Sum pixels per column → projection profile
3. Smooth with Gaussian filter
4. Find deepest valley in center 35-65% region
5. Validate: gap must have < 5 blue pixels
```

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
3. Return digit with highest score

Thresholds:
- _TEMPLATE_CONFIDENCE_THRESHOLD = 0.80
- _TEMPLATE_AMBIGUITY_GAP = 0.05
```

**Auto-Learning:** When confidence is low for multiple frames, automatically saves new template variant.

## LED Detection

### Button LEDs (`detect_button_leds()`)

Detects which of 4 buttons (B1, B2, S1, S2) has its LED lit:

```
1. Extract button region below panel
2. Detect button rectangles via edge detection
3. Create green LED mask (HSV filtering)
4. Find brightest zone among button areas
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
3. Count red pixels in HSV (dual range for hue wrap)
4. Threshold: ≥25 red pixels = lit
```

## Caching Strategy

The `SegmentReader` class maintains frame-to-frame caches:

| Cache | TTL | Purpose |
|-------|-----|---------|
| `_panel_rect` | 100 frames | Panel bounding box |
| `_gap_x` | 100 frames | Digit separator position |
| `_left_box`, `_right_box` | 100 frames | Digit bounding boxes |
| `_left_best_templates` | 100 frames | Quick-check template indices |

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
reader = SegmentReader(cache_ttl=100, auto_learn=False)
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

Automatic issue logging for debugging:

```python
log_issue_frame(frame, 'low_conf', confidence=0.75, extra_info='17')
```

**Issue Types:**
- `low_conf` - Recognition below threshold
- `ambiguous` - Close scores between digits
- `led_fail` - LED detection failed
- `glitch` - Sudden reading change

**Cooldown:** 30 seconds between saves of same issue type.

## Testing

```bash
# Unit tests
python -m pytest test_segment_reader.py -v

# Test on example images
python segment_reader.py

# Live demo
python live_demo.py
python live_demo.py --headless  # No GUI, console output
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
