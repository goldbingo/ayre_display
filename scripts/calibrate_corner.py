#!/usr/bin/env python3
"""Interactive Corner Template Capture Tool

Captures aligned 75x75 corner templates from saved images for use by
_find_corner() in segment_reader.py. Primarily used with corner_low_score
captures from the issue logger.

Two modes:
  Mode 1 (no existing templates): Click to place corner, nudge with arrows.
  Mode 2 (templates exist): Auto-match best template, show ghost overlay,
          nudge until ghost aligns with frame features.

Usage:
  python scripts/calibrate_corner.py image.png
  python scripts/calibrate_corner.py logs/corner_low_score_*.png
  python scripts/calibrate_corner.py logs/          # all PNGs in directory
  python scripts/calibrate_corner.py --migrate      # convert 150x150 → 75x75
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

# Add parent dir so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device_geometry import DeviceGeometry

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')
TEMPLATE_SIZE = 75
ZOOM = 3
ZOOM_REGION = 250  # region around corner shown zoomed
VIEW_MODES = ['frame', 'diff', 'edges', 'blend', 'ghost']


def load_templates():
    """Load existing corner templates. Returns list of (path, grayscale_img)."""
    paths = sorted(glob.glob(os.path.join(TEMPLATE_DIR, 'corner_*.png')))
    templates = []
    for p in paths:
        if '.bak.' in os.path.basename(p):
            continue
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates.append((p, img))
    return templates


def next_template_path():
    """Find next available template filename (corner_N.png)."""
    existing = sorted(glob.glob(os.path.join(TEMPLATE_DIR, 'corner_*.png')))
    existing = [p for p in existing if '.bak.' not in os.path.basename(p)]
    if not existing:
        return os.path.join(TEMPLATE_DIR, 'corner_1.png')
    # Find highest index
    max_idx = 0
    for p in existing:
        name = os.path.splitext(os.path.basename(p))[0]  # e.g. 'corner_3'
        try:
            idx = int(name.split('_')[-1])
            max_idx = max(max_idx, idx)
        except ValueError:
            pass
    return os.path.join(TEMPLATE_DIR, f'corner_{max_idx + 1}.png')


def extract_frames(img_path):
    """Load image, extract frames from composites.

    Returns list of BGR frames:
    - 640x480: single frame → [frame]
    - 1280x480: raw|display pair → [raw] (left half only)
    - 3200x480: 5-frame composite → [f0, f1, f2, f3, f4]
    - 7040x480: 11-frame ctx composite → [f0, ..., f10]
    Returns empty list on failure.
    """
    img = cv2.imread(img_path)
    if img is None:
        return []
    h, w = img.shape[:2]
    if h == 480 and w % 640 == 0:
        n = w // 640
        if n == 2:
            # raw|display pair: only raw (left half)
            return [img[:, :640].copy()]
        elif n > 2:
            # Multi-frame composite: return each frame
            return [img[:, j*640:(j+1)*640].copy() for j in range(n)]
    return [img]


def match_templates(search_green, templates):
    """Match all templates against green channel region.
    Returns list of (score, loc, template_idx) sorted by score descending."""
    results = []
    for i, (_, tmpl) in enumerate(templates):
        th, tw = tmpl.shape[:2]
        if th > search_green.shape[0] or tw > search_green.shape[1]:
            continue
        result = cv2.matchTemplate(search_green, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        results.append((max_val, max_loc, i))
    results.sort(key=lambda x: x[0], reverse=True)
    return results


def _apply_ghost(zoomed_src, ghost_tmpl, gx, gy, view_mode):
    """Apply ghost overlay onto zoomed_src region based on view mode.

    Args:
        zoomed_src: BGR image (modified in place)
        ghost_tmpl: grayscale 75x75 template
        gx, gy: ghost position in zoomed_src coords
        view_mode: one of VIEW_MODES ('frame' = no overlay)
    """
    if view_mode == 'frame':
        return

    gh, gw = ghost_tmpl.shape[:2]
    src_x1 = max(0, gx)
    src_y1 = max(0, gy)
    src_x2 = min(zoomed_src.shape[1], gx + gw)
    src_y2 = min(zoomed_src.shape[0], gy + gh)
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return

    tmpl_x1 = src_x1 - gx
    tmpl_y1 = src_y1 - gy
    tmpl_x2 = tmpl_x1 + (src_x2 - src_x1)
    tmpl_y2 = tmpl_y1 + (src_y2 - src_y1)
    ghost_patch = ghost_tmpl[tmpl_y1:tmpl_y2, tmpl_x1:tmpl_x2]
    roi = zoomed_src[src_y1:src_y2, src_x1:src_x2]

    if view_mode == 'blend':
        ghost_bgr = np.zeros_like(roi)
        ghost_bgr[:, :, 0] = ghost_patch  # blue
        ghost_bgr[:, :, 1] = ghost_patch  # green (cyan)
        cv2.addWeighted(roi, 0.5, ghost_bgr, 0.5, 0, dst=roi)

    elif view_mode == 'diff':
        # |frame_green - ghost| amplified for visibility
        frame_green = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(frame_green, ghost_patch)
        diff_amp = np.clip(diff.astype(np.int16) * 4, 0, 255).astype(np.uint8)
        roi[:, :, 0] = diff_amp
        roi[:, :, 1] = diff_amp
        roi[:, :, 2] = diff_amp

    elif view_mode == 'edges':
        edges = cv2.Canny(ghost_patch, 50, 150)
        mask = edges > 0
        roi[mask, 0] = 255
        roi[mask, 1] = 255
        roi[mask, 2] = 0

    elif view_mode == 'ghost':
        roi[:, :, 0] = ghost_patch
        roi[:, :, 1] = ghost_patch
        roi[:, :, 2] = ghost_patch


def build_display(frame, corner_x, corner_y, ghost_tmpl=None, ghost_name=None,
                  score=None, status_msg='', view_mode='frame',
                  score_map=None, search_left=0, search_top=0,
                  search_size=0, geometry=None,
                  img_index=0, img_total=0, sub_index=0, sub_total=1):
    """Build the display showing the search region zoomed up.

    Returns the display image.
    """
    # The zoomed view IS the search region
    zx1, zy1 = search_left, search_top

    # Undistort search region (matches what _find_corner sees)
    if geometry is not None:
        zoomed_src = geometry.undistort_roi(frame, zx1, zy1,
                                           search_size, search_size, derotate=False)
    else:
        zoomed_src = frame[zy1:zy1+search_size, zx1:zx1+search_size].copy()

    # Corner position relative to search region
    rx = corner_x - zx1
    ry = corner_y - zy1

    # Clamp to search region
    rx = max(0, min(search_size - TEMPLATE_SIZE, rx))
    ry = max(0, min(search_size - TEMPLATE_SIZE, ry))

    # Apply ghost overlay
    if ghost_tmpl is not None:
        _apply_ghost(zoomed_src, ghost_tmpl, rx, ry, view_mode)

    # Draw template rectangle
    cv2.rectangle(zoomed_src, (rx, ry), (rx + TEMPLATE_SIZE - 1, ry + TEMPLATE_SIZE - 1),
                  (0, 255, 0), 1)

    # Scale up
    zoom_h = search_size * ZOOM
    zoom_w = search_size * ZOOM
    zoomed_display = cv2.resize(zoomed_src, (zoom_w, zoom_h),
                                interpolation=cv2.INTER_NEAREST)

    # View mode label
    mode_label = view_mode.upper()
    cv2.putText(zoomed_display, mode_label, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # --- Score profile graphs centered on green box ---
    if score_map is not None:
        GRAPH_H = 60
        map_h, map_w = score_map.shape[:2]
        sx = rx  # same as corner_x - search_left (clamped)
        sy = ry

        box_left_d = rx * ZOOM
        box_top_d = ry * ZOOM
        box_right_d = (rx + TEMPLATE_SIZE) * ZOOM
        box_cx = box_left_d + TEMPLATE_SIZE * ZOOM // 2
        box_cy = box_top_d + TEMPLATE_SIZE * ZOOM // 2

        # X-axis graph: score vs x at current y (above or below box)
        if 0 <= sy < map_h:
            row = score_map[sy, :]
            vmin, vmax = max(0, row.min() - 0.02), min(1.0, row.max() + 0.02)
            if vmax - vmin < 0.01:
                vmax = vmin + 0.01
            # Try above; if not enough space, put below
            if box_top_d - 4 - GRAPH_H >= 0:
                graph_bottom = box_top_d - 4
                graph_t = graph_bottom - GRAPH_H
            else:
                graph_t = box_top_d + TEMPLATE_SIZE * ZOOM + 4
                graph_bottom = graph_t + GRAPH_H
            if graph_t >= 0 and graph_bottom <= zoom_h:
                half = TEMPLATE_SIZE // 2
                # Build filled polygon: baseline at bottom, curve on top
                fill_pts = []
                curve_pts = []
                for k in range(-half, half + 1):
                    mx = sx + k
                    if 0 <= mx < map_w:
                        val = row[mx]
                        px = box_cx + k * ZOOM
                        py = int(graph_bottom - (val - vmin) / (vmax - vmin) * GRAPH_H)
                        py = max(graph_t, min(graph_bottom, py))
                        if 0 <= px < zoom_w:
                            fill_pts.append((px, py))
                            curve_pts.append((px, py))
                if len(fill_pts) > 1:
                    # Close polygon along baseline
                    fill_pts.append((fill_pts[-1][0], graph_bottom))
                    fill_pts.insert(0, (fill_pts[0][0], graph_bottom))
                    # Draw filled area at 50% opacity
                    overlay = zoomed_display.copy()
                    cv2.fillPoly(overlay, [np.array(fill_pts)], (80, 200, 80))
                    cv2.addWeighted(overlay, 0.5, zoomed_display, 0.5, 0, dst=zoomed_display)
                # Yellow marker at offset=0
                if 0 <= sx < map_w:
                    val = row[sx]
                    cy = int(graph_bottom - (val - vmin) / (vmax - vmin) * GRAPH_H)
                    val_color = (0, 0, 255) if val < 0.93 else (0, 255, 255)
                    cv2.line(zoomed_display, (box_cx, graph_t), (box_cx, graph_bottom), val_color, 1)
                    cv2.circle(zoomed_display, (box_cx, cy), 4, val_color, -1)
                    cv2.putText(zoomed_display, f'{val:.4f}', (box_cx + 6, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, val_color, 1)
                # Circle at peak
                peak_x = int(np.argmax(row))
                peak_k = peak_x - sx
                if abs(peak_k) <= half:
                    peak_px = box_cx + peak_k * ZOOM
                    peak_val = row[peak_x]
                    peak_py = int(graph_bottom - (peak_val - vmin) / (vmax - vmin) * GRAPH_H)
                    peak_py = max(graph_t, min(graph_bottom, peak_py))
                    if 0 <= peak_px < zoom_w:
                        cv2.circle(zoomed_display, (peak_px, peak_py), 6, (0, 255, 255), 1)

        # Y-axis graph: score vs y at current x (right or left of box)
        if 0 <= sx < map_w:
            col = score_map[:, sx]
            vmin, vmax = max(0, col.min() - 0.02), min(1.0, col.max() + 0.02)
            if vmax - vmin < 0.01:
                vmax = vmin + 0.01
            # Try right; if not enough space, put left
            if box_right_d + 4 + GRAPH_H <= zoom_w:
                graph_left = box_right_d + 4
                graph_r = graph_left + GRAPH_H
            else:
                graph_r = box_left_d - 4
                graph_left = graph_r - GRAPH_H
            if graph_left >= 0 and graph_r <= zoom_w:
                half = TEMPLATE_SIZE // 2
                # Build filled polygon: baseline at left, curve to right
                fill_pts = []
                for k in range(-half, half + 1):
                    my = sy + k
                    if 0 <= my < map_h:
                        val = col[my]
                        py = box_cy + k * ZOOM
                        px = int(graph_left + (val - vmin) / (vmax - vmin) * GRAPH_H)
                        px = max(graph_left, min(graph_r, px))
                        if 0 <= py < zoom_h:
                            fill_pts.append((px, py))
                if len(fill_pts) > 1:
                    # Close polygon along baseline (left edge)
                    fill_pts.append((graph_left, fill_pts[-1][1]))
                    fill_pts.insert(0, (graph_left, fill_pts[0][1]))
                    overlay = zoomed_display.copy()
                    cv2.fillPoly(overlay, [np.array(fill_pts)], (80, 200, 80))
                    cv2.addWeighted(overlay, 0.5, zoomed_display, 0.5, 0, dst=zoomed_display)
                if 0 <= sy < map_h:
                    val = col[sy]
                    cx = int(graph_left + (val - vmin) / (vmax - vmin) * GRAPH_H)
                    cv2.line(zoomed_display, (graph_left, box_cy), (graph_r, box_cy), (0, 255, 255), 1)
                    cv2.circle(zoomed_display, (cx, box_cy), 4, (0, 255, 255), -1)
                # Circle at peak
                peak_y = int(np.argmax(col))
                peak_k = peak_y - sy
                if abs(peak_k) <= half:
                    peak_py = box_cy + peak_k * ZOOM
                    peak_val = col[peak_y]
                    peak_px = int(graph_left + (peak_val - vmin) / (vmax - vmin) * GRAPH_H)
                    peak_px = max(graph_left, min(graph_r, peak_px))
                    if 0 <= peak_py < zoom_h:
                        cv2.circle(zoomed_display, (peak_px, peak_py), 6, (0, 255, 255), 1)
                    cv2.circle(zoomed_display, (cx, box_cy), 4, (0, 255, 255), -1)

    # --- Info panel to the right ---
    PANEL_W = 250
    LINE_H = 24
    canvas = np.zeros((zoom_h, zoom_w + PANEL_W, 3), dtype=np.uint8)
    canvas[:zoom_h, :zoom_w] = zoomed_display

    # Score color: red if < 0.93, yellow otherwise
    if score is not None:
        score_color = (0, 0, 255) if score < 0.93 else (0, 255, 255)
        score_text = f'score {score:.4f}'
    else:
        score_color = (0, 255, 255)
        score_text = 'score --'

    # Image index line: (3-2/12) = frame 2 of file 3 out of 12 files
    if img_total > 0:
        if sub_total > 1:
            idx_text = f'({img_index}-{sub_index}/{img_total})'
        else:
            idx_text = f'({img_index}/{img_total})'
    else:
        idx_text = ''

    lines = [
        (idx_text, (220, 220, 220)),
        (status_msg if status_msg else '', (100, 255, 100)),
        ('', None),
        (f'pos ({corner_x},{corner_y})', (220, 220, 220)),
        (score_text, score_color),
        (f'ghost {ghost_name}' if ghost_name is not None else '', (200, 200, 200)),
        (f'view  {view_mode}', (200, 200, 200)),
        ('', None),
        ('arrows  nudge', (160, 160, 160)),
        ('s  save', (160, 160, 160)),
        ('SPACE  next', (160, 160, 160)),
        ('n  next <0.93', (160, 160, 160)),
        ('t  ghost', (160, 160, 160)),
        ('v  view', (160, 160, 160)),
        ('ESC  quit', (160, 160, 160)),
    ]
    y = LINE_H
    for line, color in lines:
        if line:
            cv2.putText(canvas, line, (zoom_w + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += LINE_H

    return canvas


def calibrate_frame(frame, img_path, geometry, img_index=0, img_total=0,
                    sub_index=0, sub_total=1):
    """Interactive calibration for a single frame."""
    templates = load_templates()
    fh, fw = frame.shape[:2]

    # Get undistorted search region (same as _find_corner)
    search_left, search_top, search_size = geometry.get_corner_search_region(fw, fh)
    search_roi = geometry.undistort_roi(frame, search_left, search_top, search_size, search_size,
                                        derotate=False)
    search_green = search_roi[:, :, 1] if search_roi.ndim == 3 else search_roi

    # Initial position
    ghost_tmpl = None
    ghost_idx = None
    score = None

    if templates:
        # Mode 2: match existing templates
        results = match_templates(search_green, templates)
        if results:
            best_score, best_loc, best_tidx = results[0]
            corner_x = search_left + best_loc[0]
            corner_y = search_top + best_loc[1]
            score = best_score
            ghost_tmpl = templates[best_tidx][1]
            ghost_idx = best_tidx
        else:
            corner_x, corner_y = search_left + search_size // 4, search_top + search_size // 4
    else:
        # Mode 1: default from camera_mount
        corner_x = int(geometry.corner_search_position[0] * fw)
        corner_y = int(geometry.corner_search_position[1] * fh)

    status_msg = os.path.basename(img_path)
    view_mode = 'frame'
    score_map = None

    # Clamp limits: corner must stay within search region
    max_cx = search_left + search_size - TEMPLATE_SIZE
    max_cy = search_top + search_size - TEMPLATE_SIZE

    def clamp():
        nonlocal corner_x, corner_y
        corner_x = max(search_left, min(max_cx, corner_x))
        corner_y = max(search_top, min(max_cy, corner_y))

    clamp()

    def compute_score_map():
        nonlocal score_map
        if ghost_tmpl is not None:
            th, tw = ghost_tmpl.shape[:2]
            if th <= search_green.shape[0] and tw <= search_green.shape[1]:
                score_map = cv2.matchTemplate(search_green, ghost_tmpl, cv2.TM_CCOEFF_NORMED)
                return
        score_map = None

    compute_score_map()

    def update_score():
        nonlocal score
        if score_map is not None:
            sx = corner_x - search_left
            sy = corner_y - search_top
            if 0 <= sy < score_map.shape[0] and 0 <= sx < score_map.shape[1]:
                score = float(score_map[sy, sx])
                return
        score = None

    def ghost_label():
        if ghost_idx is not None and ghost_idx < len(templates):
            return os.path.splitext(os.path.basename(templates[ghost_idx][0]))[0]
        return None

    def refresh():
        display = build_display(frame, corner_x, corner_y,
                                ghost_tmpl, ghost_label(), score, status_msg,
                                view_mode, score_map,
                                search_left, search_top,
                                search_size, geometry,
                                img_index, img_total,
                                sub_index, sub_total)
        cv2.imshow('Corner Calibration', display)

    def on_mouse(event, mx, my, flags, param):
        nonlocal corner_x, corner_y, score
        if event == cv2.EVENT_LBUTTONDOWN:
            # Click → convert to frame coords (center box on click)
            zoom_w = search_size * ZOOM
            if mx < zoom_w and my < search_size * ZOOM:
                frame_x = search_left + mx // ZOOM - TEMPLATE_SIZE // 2
                frame_y = search_top + my // ZOOM - TEMPLATE_SIZE // 2
                corner_x = max(search_left, min(max_cx, frame_x))
                corner_y = max(search_top, min(max_cy, frame_y))
                update_score()
                refresh()

    cv2.namedWindow('Corner Calibration', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Corner Calibration', on_mouse)
    refresh()

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            return 'quit'

        elif key == ord(' '):
            return 'next'

        elif key == ord('n'):
            return 'find_low'

        elif key == ord('s'):
            # Save template — extract from search_roi (same undistortion as matching)
            rx = corner_x - search_left
            ry = corner_y - search_top
            template_green = search_green[ry:ry+TEMPLATE_SIZE, rx:rx+TEMPLATE_SIZE]
            if template_green.shape[0] != TEMPLATE_SIZE or template_green.shape[1] != TEMPLATE_SIZE:
                print(f'  Cannot save: position out of search region bounds')
                continue

            save_path = next_template_path()
            # Save as single-channel PNG (same format as migrated templates)
            cv2.imwrite(save_path, template_green)
            print(f'Saved template: {save_path} ({template_green.shape[1]}x{template_green.shape[0]})')

            # Verify: reload and match all templates
            new_templates = load_templates()
            # Force reload by clearing global cache
            import segment_reader
            segment_reader._corner_templates = None

            results = match_templates(search_green, new_templates)
            print(f'Match scores against current frame:')
            for s, loc, tidx in results:
                tpath = os.path.basename(new_templates[tidx][0])
                print(f'  {tpath}: {s:.4f} at ({search_left + loc[0]}, {search_top + loc[1]})')

            status_msg = f'SAVED {os.path.basename(save_path)}'
            # Set score from the saved template's match result
            score = None
            for s_val, loc, tidx in results:
                tpath = new_templates[tidx][0]
                if tpath == save_path:
                    score = s_val
                    break
            # Update templates list and ghost to the new template
            templates = new_templates
            for i, (p, t) in enumerate(templates):
                if p == save_path:
                    ghost_tmpl = t
                    ghost_idx = i
                    break
            compute_score_map()
            refresh()

        elif key == ord('t'):
            # Cycle ghost template (keep current position)
            if templates:
                if ghost_idx is None:
                    ghost_idx = 0
                else:
                    ghost_idx = (ghost_idx + 1) % len(templates)
                ghost_tmpl = templates[ghost_idx][1]
                compute_score_map()
                update_score()
                refresh()

        elif key in (81, 2):  # left arrow
            corner_x = max(search_left, corner_x - 1)
            update_score()
            refresh()
        elif key in (82, 0):  # up arrow
            corner_y = max(search_top, corner_y - 1)
            update_score()
            refresh()
        elif key in (83, 3):  # right arrow
            corner_x = min(max_cx, corner_x + 1)
            update_score()
            refresh()
        elif key in (84, 1):  # down arrow
            corner_y = min(max_cy, corner_y + 1)
            update_score()
            refresh()

        elif key == ord('v'):
            # Cycle view mode
            idx = VIEW_MODES.index(view_mode)
            view_mode = VIEW_MODES[(idx + 1) % len(VIEW_MODES)]
            refresh()


def migrate_templates():
    """Convert existing 150x150 templates to 75x75 (bottom-right quadrant)."""
    paths = sorted(glob.glob(os.path.join(TEMPLATE_DIR, 'corner_*.png')))
    paths = [p for p in paths if '.bak.' not in os.path.basename(p)]

    converted = 0
    skipped = 0
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'  SKIP (unreadable): {path}')
            skipped += 1
            continue
        h, w = img.shape[:2]
        if h == TEMPLATE_SIZE and w == TEMPLATE_SIZE:
            print(f'  OK (already {TEMPLATE_SIZE}x{TEMPLATE_SIZE}): {os.path.basename(path)}')
            skipped += 1
            continue
        if h == 150 and w == 150:
            # Crop bottom-right quadrant
            cropped = img[75:150, 75:150]
            # Back up original
            bak_path = path.replace('.png', '.150x150.bak.png')
            os.rename(path, bak_path)
            cv2.imwrite(path, cropped)
            print(f'  CONVERTED: {os.path.basename(path)} (150x150 → 75x75, backup: {os.path.basename(bak_path)})')
            converted += 1
        else:
            print(f'  SKIP (unexpected size {w}x{h}): {os.path.basename(path)}')
            skipped += 1

    print(f'\nMigration complete: {converted} converted, {skipped} skipped')
    return converted


def main():
    parser = argparse.ArgumentParser(description='Interactive Corner Template Capture Tool')
    parser.add_argument('inputs', nargs='*', help='Image files or directories')
    parser.add_argument('--migrate', action='store_true',
                        help='Convert existing 150x150 templates to 75x75')
    args = parser.parse_args()

    if args.migrate:
        print('Migrating corner templates...')
        migrate_templates()
        return

    if not args.inputs:
        parser.print_help()
        return

    # Collect image paths
    image_paths = []
    for inp in args.inputs:
        if os.path.isdir(inp):
            image_paths.extend(sorted(glob.glob(os.path.join(inp, '*.png'))))
        elif os.path.isfile(inp):
            image_paths.append(inp)
        else:
            # Could be a glob that the shell didn't expand
            expanded = sorted(glob.glob(inp))
            if expanded:
                image_paths.extend(expanded)
            else:
                print(f'Warning: not found: {inp}')

    if not image_paths:
        print('No images found.')
        return

    geometry = DeviceGeometry()

    print(f'Processing {len(image_paths)} image(s)...')
    print('Controls: arrows=nudge, click=jump, s=save, t=ghost, SPACE=next, n=next<0.93, ESC=quit\n')

    def find_best_score(frame_bgr):
        """Quick score check: match all templates, return best score."""
        templates = load_templates()
        if not templates:
            return 0.0
        fh, fw = frame_bgr.shape[:2]
        sl, st, ss = geometry.get_corner_search_region(fw, fh)
        sroi = geometry.undistort_roi(frame_bgr, sl, st, ss, ss, derotate=False)
        sg = sroi[:, :, 1] if sroi.ndim == 3 else sroi
        results = match_templates(sg, templates)
        return results[0][0] if results else 0.0

    i = 0
    while True:
        path = image_paths[i]
        frames = extract_frames(path)
        if not frames:
            print(f'  Could not load image, skipping.')
            i = (i + 1) % len(image_paths)
            continue

        n_frames = len(frames)
        quit_all = False
        find_low = False
        for j, frame in enumerate(frames):
            label = f'[{i+1}-{j+1}/{len(image_paths)}]' if n_frames > 1 else f'[{i+1}/{len(image_paths)}]'
            print(f'{label} {path}')
            result = calibrate_frame(frame, path, geometry, i + 1, len(image_paths),
                                     j + 1, n_frames)
            if result == 'quit':
                quit_all = True
                break
            elif result == 'find_low':
                find_low = True
                break
            elif result == 'next':
                continue  # next sub-frame, or fall through to next file
        if quit_all:
            break
        if find_low:
            # Scan forward for next frame with best score < 0.93
            start_i = (i + 1) % len(image_paths)
            found = False
            k = start_i
            for _ in range(len(image_paths)):
                p = image_paths[k]
                frs = extract_frames(p)
                for fr in frs:
                    sc = find_best_score(fr)
                    if sc < 0.93:
                        print(f'  Found low score {sc:.4f} at [{k+1}/{len(image_paths)}] {p}')
                        i = k
                        found = True
                        break
                if found:
                    break
                k = (k + 1) % len(image_paths)
            if not found:
                print('  No frames with score < 0.93 found.')
                i = (i + 1) % len(image_paths)
        else:
            i = (i + 1) % len(image_paths)

    cv2.destroyAllWindows()
    print('Done.')


if __name__ == '__main__':
    main()
