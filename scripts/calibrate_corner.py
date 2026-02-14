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
VIEW_MODES = ['toggle', 'diff', 'edges', 'blend', 'ghost']


def load_templates():
    """Load existing corner templates. Returns list of (path, grayscale_img)."""
    paths = sorted(glob.glob(os.path.join(TEMPLATE_DIR, 'corner_template*.png')))
    templates = []
    for p in paths:
        if '.bak.' in os.path.basename(p):
            continue
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates.append((p, img))
    return templates


def next_template_path():
    """Find next available template filename."""
    existing = sorted(glob.glob(os.path.join(TEMPLATE_DIR, 'corner_template*.png')))
    existing = [p for p in existing if '.bak.' not in os.path.basename(p)]
    if not existing:
        return os.path.join(TEMPLATE_DIR, 'corner_template.png')
    # Find highest index
    max_idx = 0
    for p in existing:
        name = os.path.splitext(os.path.basename(p))[0]
        if name == 'corner_template':
            max_idx = max(max_idx, 1)
        elif name.startswith('corner_template_'):
            try:
                idx = int(name.split('_')[-1])
                max_idx = max(max_idx, idx)
            except ValueError:
                pass
    return os.path.join(TEMPLATE_DIR, f'corner_template_{max_idx + 1}.png')


def extract_frame(img_path):
    """Load image, extract raw frame from composites. Returns BGR frame or None."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if h == 480 and w % 640 == 0:
        n = w // 640
        if n >= 2:
            # Composite: use left half (raw frame)
            return img[:, :640].copy()
    return img


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


def _apply_ghost(zoomed_src, ghost_tmpl, gx, gy, view_mode, toggle_show_ghost):
    """Apply ghost overlay onto zoomed_src region based on view mode.

    Args:
        zoomed_src: BGR image (modified in place for blend/edges, replaced for toggle/diff)
        ghost_tmpl: grayscale 75x75 template
        gx, gy: ghost position in zoomed_src coords
        view_mode: one of VIEW_MODES
        toggle_show_ghost: if True and mode is 'toggle', show ghost instead of frame
    """
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

    elif view_mode == 'toggle':
        if toggle_show_ghost:
            # Show ghost as cyan on black
            roi[:] = 0
            roi[:, :, 0] = ghost_patch
            roi[:, :, 1] = ghost_patch

    elif view_mode == 'diff':
        # |frame_green - ghost| amplified for visibility
        frame_green = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(frame_green, ghost_patch)
        # Amplify: 4x so small differences are visible
        diff_amp = np.clip(diff.astype(np.int16) * 4, 0, 255).astype(np.uint8)
        # Show as heat: black=aligned, bright=misaligned
        roi[:, :, 0] = diff_amp
        roi[:, :, 1] = diff_amp
        roi[:, :, 2] = diff_amp

    elif view_mode == 'edges':
        # Canny edges of ghost template overlaid as cyan lines
        edges = cv2.Canny(ghost_patch, 50, 150)
        mask = edges > 0
        roi[mask, 0] = 255  # blue
        roi[mask, 1] = 255  # green (cyan)
        roi[mask, 2] = 0

    elif view_mode == 'ghost':
        # 100% ghost template as grayscale
        roi[:, :, 0] = ghost_patch
        roi[:, :, 1] = ghost_patch
        roi[:, :, 2] = ghost_patch


def build_display(frame, corner_x, corner_y, ghost_tmpl=None, ghost_name=None,
                  score=None, status_msg='', view_mode='toggle',
                  toggle_show_ghost=False, zoom_origin=None):
    """Build the display with zoomed view and full frame side by side.

    Returns the combined display image and zoom origin (zx1, zy1).
    zoom_origin: if provided, keeps the view stable; only pans when corner
                 leaves the visible area.
    """
    fh, fw = frame.shape[:2]
    margin = 30  # pan when corner is within this many px of zoom edge

    # --- Zoomed view ---
    if zoom_origin is not None:
        zx1, zy1 = zoom_origin
        # Only pan if corner is near edge or outside
        rx = corner_x - zx1
        ry = corner_y - zy1
        need_pan = (rx < margin or rx + TEMPLATE_SIZE > ZOOM_REGION - margin or
                    ry < margin or ry + TEMPLATE_SIZE > ZOOM_REGION - margin)
        if not need_pan:
            zx2 = zx1 + ZOOM_REGION
            zy2 = zy1 + ZOOM_REGION
        else:
            half = ZOOM_REGION // 2
            zx1 = max(0, corner_x - half)
            zy1 = max(0, corner_y - half)
            zx2 = min(fw, zx1 + ZOOM_REGION)
            zy2 = min(fh, zy1 + ZOOM_REGION)
            zx1 = max(0, zx2 - ZOOM_REGION)
            zy1 = max(0, zy2 - ZOOM_REGION)
    else:
        half = ZOOM_REGION // 2
        zx1 = max(0, corner_x - half)
        zy1 = max(0, corner_y - half)
        zx2 = min(fw, zx1 + ZOOM_REGION)
        zy2 = min(fh, zy1 + ZOOM_REGION)
        zx1 = max(0, zx2 - ZOOM_REGION)
        zy1 = max(0, zy2 - ZOOM_REGION)

    zoomed_src = frame[zy1:zy2, zx1:zx2].copy()

    rx = corner_x - zx1
    ry = corner_y - zy1

    # Apply ghost overlay
    if ghost_tmpl is not None:
        _apply_ghost(zoomed_src, ghost_tmpl, rx, ry, view_mode, toggle_show_ghost)

    # Draw rectangle after ghost so it's always visible
    cv2.rectangle(zoomed_src, (rx, ry), (rx + TEMPLATE_SIZE - 1, ry + TEMPLATE_SIZE - 1),
                  (0, 255, 0), 1)

    # Scale up
    zoom_h = ZOOM_REGION * ZOOM
    zoom_w = ZOOM_REGION * ZOOM
    zoomed_display = cv2.resize(zoomed_src, (zoom_w, zoom_h),
                                interpolation=cv2.INTER_NEAREST)

    # View mode label on zoomed view
    mode_label = view_mode.upper()
    if view_mode == 'toggle':
        mode_label += ' [GHOST]' if toggle_show_ghost else ' [FRAME]'
    cv2.putText(zoomed_display, mode_label, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # --- Full frame view ---
    full_display = frame.copy()
    cv2.rectangle(full_display, (corner_x, corner_y),
                  (corner_x + TEMPLATE_SIZE - 1, corner_y + TEMPLATE_SIZE - 1),
                  (0, 255, 0), 1)
    cv2.circle(full_display, (corner_x, corner_y), 3, (0, 0, 255), -1)

    # --- Combine ---
    canvas_h = max(zoom_h, fh)
    canvas_w = zoom_w + fw + 10
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:zoom_h, :zoom_w] = zoomed_display
    canvas[:fh, zoom_w + 10:zoom_w + 10 + fw] = full_display

    # --- Status bar with semi-transparent background ---
    BAR_H = 28
    bar_y = canvas_h - BAR_H
    # 50% black background
    overlay = canvas[bar_y:canvas_h, :].copy()
    canvas[bar_y:canvas_h, :] = (overlay * 0.5).astype(np.uint8)

    parts = [f'pos=({corner_x},{corner_y})']
    if score is not None:
        parts.append(f'score={score:.4f}')
    if ghost_name is not None:
        parts.append(f'ghost={ghost_name}')
    parts.append(f'v={view_mode}')
    parts.append('arrows  s=save  n=next  t=ghost  v=view  ESC=quit')
    if status_msg:
        parts.append(status_msg)
    cv2.putText(canvas, '  |  '.join(parts), (5, canvas_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    return canvas, (zx1, zy1)


def calibrate_frame(frame, img_path, geometry):
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
    zoom_origin = None  # will be set on first refresh
    view_mode = 'toggle'
    toggle_show_ghost = False

    def ghost_label():
        if ghost_idx is not None and ghost_idx < len(templates):
            return os.path.splitext(os.path.basename(templates[ghost_idx][0]))[0]
        return None

    def refresh():
        nonlocal zoom_origin
        display, zoom_origin = build_display(frame, corner_x, corner_y,
                                             ghost_tmpl, ghost_label(), score, status_msg,
                                             view_mode, toggle_show_ghost,
                                             zoom_origin)
        cv2.imshow('Corner Calibration', display)

    def on_mouse(event, mx, my, flags, param):
        nonlocal corner_x, corner_y, score
        if event == cv2.EVENT_LBUTTONDOWN:
            # Click in zoomed view → convert to frame coords
            zoom_w = ZOOM_REGION * ZOOM
            if mx < zoom_w and my < ZOOM_REGION * ZOOM:
                frame_x = zoom_origin[0] + mx // ZOOM - TEMPLATE_SIZE // 2
                frame_y = zoom_origin[1] + my // ZOOM - TEMPLATE_SIZE // 2
                corner_x = max(0, min(fw - TEMPLATE_SIZE, frame_x))
                corner_y = max(0, min(fh - TEMPLATE_SIZE, frame_y))
                score = None
                refresh()

    cv2.namedWindow('Corner Calibration', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Corner Calibration', on_mouse)
    refresh()

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            return 'quit'

        elif key == ord('n'):
            return 'next'

        elif key == ord('s'):
            # Save template
            # Extract from undistorted frame at current position
            undist_roi = geometry.undistort_roi(frame, corner_x, corner_y,
                                                TEMPLATE_SIZE, TEMPLATE_SIZE, derotate=False)
            if undist_roi.ndim == 3:
                template_green = undist_roi[:, :, 1]
            else:
                template_green = undist_roi

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
            score = None
            # Update ghost to the new template
            for i, (p, t) in enumerate(new_templates):
                if p == save_path:
                    ghost_tmpl = t
                    ghost_idx = i
                    break
            refresh()

        elif key == ord('t'):
            # Cycle ghost template (keep current position)
            if templates:
                if ghost_idx is None:
                    ghost_idx = 0
                else:
                    ghost_idx = (ghost_idx + 1) % len(templates)
                ghost_tmpl = templates[ghost_idx][1]
                score = None
                refresh()

        elif key in (81, 2):  # left arrow
            corner_x = max(0, corner_x - 1)
            score = None
            refresh()
        elif key in (82, 0):  # up arrow
            corner_y = max(0, corner_y - 1)
            score = None
            refresh()
        elif key in (83, 3):  # right arrow
            corner_x = min(fw - TEMPLATE_SIZE, corner_x + 1)
            score = None
            refresh()
        elif key in (84, 1):  # down arrow
            corner_y = min(fh - TEMPLATE_SIZE, corner_y + 1)
            score = None
            refresh()

        elif key == ord('v'):
            # Cycle view mode
            idx = VIEW_MODES.index(view_mode)
            view_mode = VIEW_MODES[(idx + 1) % len(VIEW_MODES)]
            toggle_show_ghost = False
            refresh()

        elif key == ord(' '):
            # Toggle ghost/frame in toggle mode
            if view_mode == 'toggle':
                toggle_show_ghost = not toggle_show_ghost
                refresh()


def migrate_templates():
    """Convert existing 150x150 templates to 75x75 (bottom-right quadrant)."""
    paths = sorted(glob.glob(os.path.join(TEMPLATE_DIR, 'corner_template*.png')))
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
    print('Controls: arrows=nudge, click=jump, s=save, t=cycle_ghost, n=next, ESC=quit\n')

    for i, path in enumerate(image_paths):
        print(f'[{i+1}/{len(image_paths)}] {path}')
        frame = extract_frame(path)
        if frame is None:
            print(f'  Could not load image, skipping.')
            continue

        result = calibrate_frame(frame, path, geometry)
        if result == 'quit':
            break

    cv2.destroyAllWindows()
    print('Done.')


if __name__ == '__main__':
    main()
