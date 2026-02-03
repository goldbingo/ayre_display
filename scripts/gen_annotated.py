#!/usr/bin/env python3
"""Generate annotated reference image for camera calibration docs.

Reads all positions from camera_mount.json — no hardcoded coordinates.

Usage:
    python scripts/gen_annotated.py path/to/frame.png
    python scripts/gen_annotated.py path/to/debug_overlay.png --right-half
"""
import cv2
import json
import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MOUNT_PATH = os.path.join(PROJECT_ROOT, 'calibration', 'camera_mount.json')
OUT_PATH = os.path.join(PROJECT_ROOT, 'calibration', 'camera_mount_reference.png')

# Parse args
right_half = '--right-half' in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith('--')]
if not args:
    print(f"Usage: python {sys.argv[0]} <frame.png> [--right-half]")
    print("  --right-half  Use right half of a debug overlay image")
    sys.exit(1)

# Load source frame
full = cv2.imread(args[0])
if full is None:
    print(f"Error: Cannot read {args[0]}")
    sys.exit(1)

if right_half:
    frame = full[:, full.shape[1] // 2:, :]
else:
    frame = full

# Load positions from camera_mount.json
with open(MOUNT_PATH) as f:
    mount = json.load(f)

corner_xy = tuple(mount['corner_xy'])
button_centers = {k: tuple(v) for k, v in mount['button_centers'].items()}
panel_rect = tuple(mount['panel_rect'])  # x, y, w, h
mute_center = tuple(mount['mute_center'])
mute_region = tuple(mount['mute_region'])  # x1, y1, x2, y2

# Compute offsets for labels
cx, cy = corner_xy
px, py, pw, ph = panel_rect
panel_offset = (px - cx, py - cy)
mute_offset = (mute_center[0] - cx, mute_center[1] - cy)

# Darken frame so annotations stand out
overlay = (frame * 0.50).astype(np.uint8)

CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)
GRAY = (120, 120, 120)

font = cv2.FONT_HERSHEY_SIMPLEX


def put_label(img, text, pos, color=WHITE, size=0.42, thickness=1):
    """Draw text with dark background."""
    (tw, th), _ = cv2.getTextSize(text, font, size, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + 3), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, size, color, thickness, cv2.LINE_AA)


# ── Coordinate axes (top-left) ──
ax_x, ax_y = 15, 22
ax_len = 35
cv2.arrowedLine(overlay, (ax_x, ax_y), (ax_x + ax_len, ax_y), WHITE, 2, tipLength=0.2)
cv2.arrowedLine(overlay, (ax_x, ax_y), (ax_x, ax_y + ax_len), WHITE, 2, tipLength=0.2)
put_label(overlay, 'X+', (ax_x + ax_len + 3, ax_y + 1), WHITE, size=0.35)
put_label(overlay, 'Y+', (ax_x + 6, ax_y + ax_len + 12), WHITE, size=0.35)
put_label(overlay, '(0,0)', (ax_x - 2, ax_y - 8), WHITE, size=0.32)

# Title
put_label(overlay, 'Camera Mount Calibration Reference', (130, 18), WHITE, size=0.5, thickness=1)

# ── MANDATORY: 7 points to measure (bright, solid) ──

# 1. Corner (yellow crosshair)
cross_sz = 20
cv2.line(overlay, (cx - cross_sz, cy), (cx + cross_sz, cy), YELLOW, 2)
cv2.line(overlay, (cx, cy - cross_sz), (cx, cy + cross_sz), YELLOW, 2)
cv2.circle(overlay, (cx, cy), 4, YELLOW, -1)
put_label(overlay, f'MEASURE: corner_xy ({cx}, {cy})', (cx - 105, cy + 22), YELLOW, size=0.38)

# 2. Panel top-left (green dot)
cv2.rectangle(overlay, (px, py), (px + pw, py + ph), GREEN, 2)
cv2.circle(overlay, (px, py), 6, GREEN, -1)
put_label(overlay, f'MEASURE: panel top-left ({px}, {py})', (px - 15, py - 10), GREEN, size=0.38)

# 3. Button centers B1, B2, S1, S2 (cyan)
for name in ('B1', 'B2', 'S1', 'S2'):
    bx, by = button_centers[name]
    cv2.circle(overlay, (bx, by), 10, CYAN, 2)
    cv2.circle(overlay, (bx, by), 3, CYAN, -1)

for name in ('B1', 'B2', 'S1', 'S2'):
    bx, by = button_centers[name]
    label_y = by + 22 if name in ('S1',) else by - 15
    put_label(overlay, f'MEASURE: {name} ({bx}, {by})', (bx - 45, label_y), CYAN, size=0.38)

# 4. Mute LED center (red dot)
mx1, my1, mx2, my2 = mute_region
mcx, mcy = mute_center
cv2.rectangle(overlay, (mx1, my1), (mx2, my2), RED, 2)
cv2.circle(overlay, (mcx, mcy), 6, RED, -1)
put_label(overlay, 'MEASURE: mute center', (mx1 - 120, my1 - 10), RED, size=0.38)
put_label(overlay, f'({mcx}, {mcy})', (mx1 - 60, my1 + 6), RED, size=0.38)

# ── COMPUTED: offsets derived from measurements (dim arrows) ──

# Panel offset arrow
cv2.arrowedLine(overlay, corner_xy, (px, py), MAGENTA, 1, tipLength=0.06)
put_label(overlay, f'computed: panel_offset ({panel_offset[0]:+d}, {panel_offset[1]:+d})',
          (195, 280), MAGENTA, size=0.33)

# Mute offset arrow
cv2.arrowedLine(overlay, corner_xy, (mcx, mcy), ORANGE, 1, tipLength=0.08)
put_label(overlay, f'computed: mute_offset ({mute_offset[0]:+d}, {mute_offset[1]:+d})',
          (420, 390), ORANGE, size=0.33)

# Landmark offset arrows (corner to each button)
for name, (bx, by) in button_centers.items():
    cv2.arrowedLine(overlay, corner_xy, (bx, by), GRAY, 1, tipLength=0.04)

put_label(overlay, 'computed: landmark offsets', (130, 380), GRAY, size=0.33)

# ── Legend (bottom-right) ──
lx, ly = 430, 440
put_label(overlay, 'MEASURE = manual (7 points)', (lx, ly), WHITE, size=0.33)
put_label(overlay, 'computed = derived from above', (lx, ly + 16), GRAY, size=0.33)

cv2.imwrite(OUT_PATH, overlay)
print(f"Saved: {OUT_PATH}")
