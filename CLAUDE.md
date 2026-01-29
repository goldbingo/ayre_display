# Project Notes

## Temporary Files
Use `.claude/tmp/` for all temporary files.

## Project: 7-Segment Display Reader
- Main script: `live_demo.py`
- Core library: `segment_reader.py`
- Design docs: `DESIGN.md`
- Logs directory: `logs/`
- Example images: `example/`
- Templates: `templates/`

## User Preferences

### "run live" command
When user says "run live", execute in background:
```bash
pkill -9 -f "live_demo.py"; sleep 1; .venv/bin/python live_demo.py --display --log &
```
- Always run in background (don't block terminal)
- Kill old instance first
- Default: headless, no logging, drain 2
- Use `--display` to show window
- Use `--log` to enable logging

### Before Git Commit (MANDATORY)
1. Run `python segment_reader.py` - **ALL** example/ images must pass
   - Do NOT use `head` or partial output - verify ALL images
   - Expected: 2 XX results (edge case images), rest must match filename
2. Update DESIGN.md if any features changed

### When Changing CSV Log Format
When adding/removing fields in `log_detection()` or changing the CSV header:
1. Archive old CSV: `mv logs/detection.csv logs/detection_archived_YYYYMMDD.csv`
2. New `detection.csv` will be created with correct header on next run
3. This prevents header/data mismatch in the CSV file

### Manual Template Learning
- `l#` - Save left digit as # (e.g., `l6`)
- `r#` - Save right digit as # (e.g., `r8`)

## Key Files
| File | Purpose |
|------|---------|
| `live_demo.py` | Main monitoring script |
| `segment_reader.py` | Core detection library |
| `DESIGN.md` | Full design documentation |
| `.claude/notify_config.json` | iMessage config |

## Debug SOPs

### Gap Detection Debug
When gap_x is wrong, visualize the column brightness histogram:

```python
import sys
sys.path.insert(0, '/Volumes/ExtData/proj/claude')
import cv2
import numpy as np
import subprocess
from segment_reader import correct_slant

# Load frame (use saved issue frame or current frame)
frame = cv2.imread('logs/YYYYMMDD_HHMMSS_low_conf_display_....png')

# Extract panel (get coords from corresponding .txt file)
px, py, pw, ph = 151, 227, 145, 105  # adjust as needed
panel = frame[py:py+ph, px:px+pw]

# Apply slant correction (same as read() method)
corrected_img, _, _ = correct_slant(panel, 8.0)

# Compute column brightness histogram
gray = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
col_sums = np.sum(gray, axis=0).astype(np.float64)
kernel = np.ones(5) / 5
smoothed = np.convolve(col_sums, kernel, mode='same')

h, w = corrected_img.shape[:2]
center = w // 2
search_limit = int(w * 0.15)

# Find local minima
for i in range(1, len(smoothed) - 1):
    if smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]:
        in_range = center - search_limit <= i <= center + search_limit
        print(f"x={i}: brightness={smoothed[i]:.0f} {'<-- IN RANGE' if in_range else ''}")

# Visualize: scale up image, draw histogram bars, mark minima with circles
# Draw green line at expected gap, red line at detected gap
# Open result in Preview
```

Key points:
- Algorithm searches from center outward for first local minimum
- Yellow circles = local minima in search range
- Deep valley (low brightness) = correct gap between digits
- Small dip (high brightness) = false minimum
