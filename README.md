# Ayre Display Reader

A computer vision system for reading 7-segment displays from Ayre audio equipment via RTSP camera stream.

## Features

- **7-segment digit recognition** using template matching
- **LED button detection** (B1, B2, S1, S2) with automatic button zone detection
- **MUTE indicator detection** via red LED sensing
- **Live camera feed** with debug visualization
- **Auto-save** debug frames when MUTE or B1 is detected

## Files

- `segment_reader.py` - Core detection library
- `live_demo_v10.py` / `live_demo_v11.py` - Live camera demo application
- `test_segment_reader.py` - Unit tests
- `templates/` - Digit and corner templates for matching
- `example/` - Example images for testing
- `proj.md` - Project documentation

## Usage

### Live Demo

```bash
# Local camera
python live_demo_v11.py --camera 0

# RTSP stream
python live_demo_v11.py --camera "rtsp://user:pass@host:port/path"

# Headless mode (no display)
python live_demo_v11.py --camera 0 --headless
```

### Keys (in live demo)
- `q` - Quit
- `r` - Reset cache
- `s` - Save current frame
- `l` - Learn mode (save digit template)

### Unit Tests

```bash
python -m pytest test_segment_reader.py -v
```

## Requirements

- Python 3.8+
- OpenCV (`cv2`)
- NumPy

## Detection Overview

1. **Corner detection** - Template matching to find reference point
2. **Panel detection** - Locate the 7-segment display area
3. **Slant correction** - Correct perspective distortion
4. **Digit recognition** - Template matching against learned digits
5. **LED detection** - Find lit button LEDs (B1, B2, S1, S2)
6. **MUTE detection** - Red pixel counting in MUTE button region
