# Ayre Display Reader

A computer vision system for reading 7-segment displays from Ayre audio equipment via RTSP camera stream.

## Features

- **7-segment digit recognition** using template matching with multiple variants per digit
- **LED button detection** (B1, B2, S1, S2) with automatic button zone detection
- **MUTE indicator detection** via red LED sensing
- **Multi-level panel detection** - landmark, corner, and brightness fallback methods
- **Live camera feed** with debug visualization
- **Auto-logging** of problematic frames (low confidence, ambiguous readings, LED failures)

## Files

- `segment_reader.py` - Core detection library
- `live_demo.py` - Live camera demo application
- `test_segment_reader.py` - Unit tests (54 tests)
- `templates/` - Digit and corner templates for matching
- `example/` - Example images for testing
- `logs/` - Detection logs and captured problem frames

## Usage

### Live Demo

```bash
# Default RTSP stream (640x480, skip every 3 frames)
python live_demo.py

# Custom resolution
python live_demo.py --width 1280 --height 720

# Process every frame (no skip)
python live_demo.py --skip 1

# Headless mode (no display window)
python live_demo.py --headless
```

### Keys (in live demo)
- `q` - Quit
- `c` - Reset cache
- `s` - Save current frame
- `l#` / `r#` - Learn digit template (e.g., `l6` learns left digit as 6)

### Unit Tests

```bash
python -m pytest test_segment_reader.py -v
```

## Requirements

- Python 3.8+
- OpenCV (`cv2`)
- NumPy

## Detection Pipeline

1. **Panel Detection** (3-level fallback)
   - **Landmark** - Corner + button positions (most accurate)
   - **Corner** - Corner template only (threshold 0.7)
   - **Brightness** - Hybrid centroid approach for dim scenes

2. **Slant Correction** - Fixed 8° correction for perspective

3. **Digit Recognition**
   - Gap detection to separate left/right digits
   - Template matching with Otsu thresholding
   - Multiple template variants per digit (e.g., 2a, 2b, 2c)

4. **LED Detection**
   - Button zone detection from visible buttons
   - Enlarged zones in fallback mode
   - Single brightest LED identification

5. **MUTE Detection** - Red pixel counting in MUTE region
