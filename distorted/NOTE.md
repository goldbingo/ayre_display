# Distorted Image Test Results

246 images: 6 warp types applied to example/ images.

## Warp Types
- shift_right_30px, shift_down_20px, rotate_3deg, zoom_in_10pct, perspective_tilt, combined_shift_rot

## Results (2026-01-31)

231 pass, 3 fail, 12 expected-1X.

All shift, rotate, perspective, and combined warps pass (100%).

### 3 Remaining Failures (all zoom_in_10pct)

| Image | Cause |
|-------|-------|
| 08-B2-washout_zoom_in_10pct | Corner detected but left digit too washed out (score 0.53) |
| 27-B2-UNMUTE-misread_zoom_in_10pct | Corner undetectable (0.64) even in original — shifted/misaligned frame |
| PP-S1-UNMUTE-dark_zoom_in_10pct | Corner undetectable (0.50) even in original — too dark |

These are inherently difficult images where corner detection or digit recognition fails regardless of zoom.
