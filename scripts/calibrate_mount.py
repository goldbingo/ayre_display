#!/usr/bin/env python3
"""
Interactive camera mount calibration tool.

Click on 8 landmarks (corner, B1, B2, S1, S2, mute LED, digit corners) to
update camera_mount.json (v2 format) and recompute device_model.json.

Usage:
    python scripts/calibrate_mount.py image.png
    python scripts/calibrate_mount.py logs/some_frame.png
"""

import argparse
import cv2
import json
import numpy as np
import os
import re
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MOUNT_PATH = os.path.join(PROJECT_ROOT, 'calibration', 'camera_mount.json')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'calibration', 'device_model.json')
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'templates')

DISPLAY_SCALE = 2
ZOOM_SCALE = 4
ZOOM_RADIUS = 60
CORNER_HALF = 37  # corner_xy in JSON is top-left; template is 75x75

# Landmark definitions: name, color (BGR), description, mode
# mode: 'point' = crosshair only, 'box' = show a box around center
LANDMARKS = [
    ('corner',   (0, 255, 255), 'Corner (top-right device edge)', 'box'),
    ('B1',       (0, 200, 0),   'B1 button (leftmost)',           'point'),
    ('B2',       (0, 255, 0),   'B2 button',                      'point'),
    ('S1',       (255, 200, 0), 'S1 button',                      'point'),
    ('S2',       (255, 100, 0), 'S2 button (rightmost)',           'point'),
    ('mute_led',      (0, 0, 255),   'Mute LED (far right)',            'point'),
    ('digit_left_bl', (255, 0, 255), 'LEFT digit lower-left corner',    'point'),
    ('digit_right_tr',(255, 0, 255), 'RIGHT digit top-right corner',    'point'),
]

# Default box sizes (half-width, half-height) for each box-mode landmark
# These get overridden from device_model if available
DEFAULT_BOX_HALF = {
    'corner': (37, 37),
    'B1': (37, 24), 'B2': (37, 24), 'S1': (38, 23), 'S2': (33, 24),
}


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_json(path, data):
    """Write JSON with compact short arrays."""
    text = json.dumps(data, indent=2)
    V = r'[-\d.e]+'
    text = re.sub(rf'\[\s*\n\s+({V}),\s*\n\s+({V})\s*\n\s*\]',
                  r'[\1, \2]', text)
    text = re.sub(rf'\[\s*\n\s+({V}),\s*\n\s+({V}),\s*\n\s+({V})\s*\n\s*\]',
                  r'[\1, \2, \3]', text)
    text = re.sub(rf'\[\s*\n\s+({V}),\s*\n\s+({V}),\s*\n\s+({V}),\s*\n\s+({V})\s*\n\s*\]',
                  r'[\1, \2, \3, \4]', text)
    with open(path, 'w') as f:
        f.write(text + '\n')


def extract_frame(source):
    """Load a single 640x480 frame from image file or RTSP stream."""
    if source.startswith('rtsp://'):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERROR: Cannot open stream {source}")
            sys.exit(1)
        # Drain a few frames to get a stable one
        for _ in range(5):
            ret, img = cap.read()
        cap.release()
        if not ret or img is None:
            print(f"ERROR: Cannot read frame from {source}")
            sys.exit(1)
    else:
        img = cv2.imread(source)
        if img is None:
            print(f"ERROR: Cannot read {source}")
            sys.exit(1)
    h, w = img.shape[:2]
    if h == 480 and w >= 640:
        return img[:, :640].copy()
    print(f"ERROR: Unexpected image size {w}x{h}")
    sys.exit(1)


def find_corner_template(frame):
    """Try to auto-detect corner using templates. Returns (x, y, score) or None."""
    templates = []
    for name in sorted(os.listdir(TEMPLATE_DIR)):
        if name.startswith('corner_') and name.endswith('.png'):
            path = os.path.join(TEMPLATE_DIR, name)
            tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if tmpl is not None:
                templates.append(tmpl)

    if not templates:
        return None

    # Search the entire frame (no restricted region — this is a calibration tool)
    search_region = frame[:, :, 1] if frame.ndim == 3 else frame

    best_score = 0
    best_corner = None

    for tmpl in templates:
        th, tw = tmpl.shape[:2]
        crop = tmpl[th//2:, tw//2:]
        result = cv2.matchTemplate(search_region, crop, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = max_val
            best_corner = max_loc

    if best_score >= 0.85 and best_corner:
        # best_corner is top-left of template match, convert to center
        return (best_corner[0] + CORNER_HALF, best_corner[1] + CORNER_HALF, best_score)
    return None


def put_text_bg(img, text, pos, scale, color, thickness=1, bg_color=(0, 0, 0), pad=4):
    """Draw text with a dark background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad),
                  bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def get_box_half(name, model):
    """Get (half_w, half_h) for a box-mode landmark from model or defaults."""
    return DEFAULT_BOX_HALF.get(name, (30, 20))


class Calibrator:
    def __init__(self, frame, old_mount, model):
        self.frame_orig = frame
        self.frame = frame.copy()
        self.gamma = 1.0
        self.old_mount = old_mount
        self.model = model
        self.positions = {}   # landmark_name -> (x, y) in frame coords
        self.current_step = 0
        self.zoom_on = False
        self.active_pos = None  # Current click position (frame coords)
        self.window_name = 'Calibrate Mount'

    def _old_pos(self, name):
        """Get old position for a landmark as center coords, or None."""
        if self.old_mount is None:
            return None
        # v2: check landmarks_raw first
        landmarks_raw = self.old_mount.get('landmarks_raw', {})
        if name in landmarks_raw:
            pos = landmarks_raw[name]
            if name == 'corner':
                # landmarks_raw corner is top-left, convert to center
                return (pos[0] + CORNER_HALF, pos[1] + CORNER_HALF)
            return tuple(pos)
        # v1 fallback
        if name == 'corner':
            v = self.old_mount.get('corner_xy')
            if v:
                # corner_xy is top-left in JSON, convert to center
                return (v[0] + CORNER_HALF, v[1] + CORNER_HALF)
            return None
        elif name == 'mute_led':
            v = self.old_mount.get('mute_center')
            return tuple(v) if v else None
        elif name in ('B1', 'B2', 'S1', 'S2'):
            centers = self.old_mount.get('button_centers', {})
            if name in centers:
                return tuple(centers[name])
        return None

    def _adjust_gamma(self, delta):
        """Adjust display gamma by delta. Positive = brighter."""
        self.gamma = max(0.2, min(5.0, self.gamma + delta))
        inv_gamma = 1.0 / self.gamma
        table = ((np.arange(256) / 255.0) ** inv_gamma * 255).astype(np.uint8)
        self.frame = cv2.LUT(self.frame_orig, table)
        print(f"  Gamma: {self.gamma:.1f}")

    def _step_back(self):
        """Go back to previous step."""
        if self.current_step <= 0:
            return False  # Signal: can't go back, quit
        self.current_step -= 1
        name = LANDMARKS[self.current_step][0]
        # Restore the position that was confirmed for this step
        self.active_pos = self.positions.pop(name, None)
        self.zoom_on = False
        self.draw()
        print(f"  << Back to {name}")
        return True

    def draw(self):
        """Draw the current state."""
        S = DISPLAY_SCALE
        fh, fw = self.frame.shape[:2]

        if self.zoom_on and self.active_pos:
            zoom_img = self._make_zoom_view()
            # Side panel on the right
            panel = self._make_side_panel(zoom_img.shape[0])
            vis = np.hstack([zoom_img, panel])
        else:
            vis = cv2.resize(self.frame, (fw * S, fh * S),
                             interpolation=cv2.INTER_NEAREST)
            self._draw_overlays(vis, S)
            self._draw_info_panel(vis)

        cv2.imshow(self.window_name, vis)

    def _draw_overlays(self, vis, scale):
        """Draw markers, boxes, crosses, previews on the scaled image."""
        # Old positions as gray markers (update mode)
        # _old_pos returns center coords for all landmarks
        if self.old_mount:
            gray = (120, 120, 120)
            for lm_name, _, _, mode in LANDMARKS:
                old = self._old_pos(lm_name)
                if old and len(old) == 2:
                    ox, oy = int(old[0]) * scale, int(old[1]) * scale
                    if mode == 'box':
                        hw, hh = get_box_half(lm_name, self.model)
                        hw, hh = hw * scale, hh * scale
                        cv2.rectangle(vis, (ox-hw, oy-hh), (ox+hw, oy+hh), gray, 1, cv2.LINE_AA)
                        put_text_bg(vis, "old", (ox + hw + 4, oy - 4), 0.3, gray)
                    else:
                        cv2.drawMarker(vis, (ox, oy), gray, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
                        put_text_bg(vis, "old", (ox + 6, oy - 6), 0.3, gray)

        # Placed (confirmed) landmarks
        for lm_name, color, _, mode in LANDMARKS:
            if lm_name in self.positions:
                px, py = self.positions[lm_name]
                sx, sy = px * scale, py * scale
                self._draw_marker(vis, sx, sy, lm_name, color, mode, scale, confirmed=True)

        # Active position (current step, not yet confirmed)
        if self.active_pos and self.current_step < len(LANDMARKS):
            name, color, _, mode = LANDMARKS[self.current_step]
            ax, ay = self.active_pos[0] * scale, self.active_pos[1] * scale
            self._draw_marker(vis, ax, ay, name, color, mode, scale, confirmed=False)

        # Live preview overlays
        self._draw_preview(vis, scale)

    def _draw_marker(self, vis, sx, sy, name, color, mode, scale, confirmed):
        """Draw a single landmark marker at display coords (sx, sy)."""
        if mode == 'box':
            hw, hh = get_box_half(name, self.model)
            hw, hh = hw * scale, hh * scale
            cv2.rectangle(vis, (sx-hw, sy-hh), (sx+hw, sy+hh), color, 2, cv2.LINE_AA)
            if name != 'corner':
                cv2.drawMarker(vis, (sx, sy), color, cv2.MARKER_CROSS,
                               10 if confirmed else 14, 1, cv2.LINE_AA)
            if confirmed:
                put_text_bg(vis, name, (sx + hw + 4, sy - 4), 0.5, color)
        elif mode == 'crosshair':
            # Mute-style: crosshair + LED patch box + ref patch box
            model = self.model
            r = (model.get('mute_led_patch_radius', 4) if model else 4) * scale
            cv2.rectangle(vis, (sx-r, sy-r), (sx+r, sy+r), color, 1)
            cv2.line(vis, (sx-r-6, sy), (sx-r-1, sy), color, 1, cv2.LINE_AA)
            cv2.line(vis, (sx+r+1, sy), (sx+r+6, sy), color, 1, cv2.LINE_AA)
            cv2.line(vis, (sx, sy-r-6), (sx, sy-r-1), color, 1, cv2.LINE_AA)
            cv2.line(vis, (sx, sy+r+1), (sx, sy+r+6), color, 1, cv2.LINE_AA)
            if model:
                dx = model.get('mute_ref_offset_dx', -18) * scale
                dy = model.get('mute_ref_offset_dy', 0) * scale
                rx, ry = sx + dx, sy + dy
                cv2.rectangle(vis, (rx-r, ry-r), (rx+r, ry+r), (200, 200, 0), 1)
                if confirmed:
                    put_text_bg(vis, "ref", (rx+r+4, ry+4), 0.35, (200, 200, 0))
            if confirmed:
                put_text_bg(vis, name, (sx+r+8, sy-4), 0.5, color)
        elif mode == 'point':
            # Simple crosshair for point landmarks (digit corners)
            sz = 12 if confirmed else 16
            cv2.drawMarker(vis, (sx, sy), color, cv2.MARKER_CROSS,
                           sz, 1, cv2.LINE_AA)
            if confirmed:
                put_text_bg(vis, name, (sx + 8, sy - 4), 0.4, color)

    def _draw_preview(self, vis, scale):
        """Draw mute search region and panel rect preview."""
        model = self.model

        # Panel rect preview from digit landmarks only (extended for full panel)
        if 'digit_left_bl' in self.positions and 'digit_right_tr' in self.positions:
            bl = self.positions['digit_left_bl']
            tr = self.positions['digit_right_tr']
            w = tr[0] - bl[0]
            h = bl[1] - tr[1]
            cx_panel = (bl[0] + tr[0]) / 2
            cy_panel = (tr[1] + bl[1]) / 2
            ew, eh = w * 1.85, h * 1.7
            x1 = int((cx_panel - ew / 2) * scale)
            y1 = int((cy_panel - eh / 2) * scale)
            x2 = int((cx_panel + ew / 2) * scale)
            y2 = int((cy_panel + eh / 2) * scale)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (100, 255, 100), 2)
            put_text_bg(vis, "panel", (x1 + 4, y1 - 4), 0.4, (100, 255, 100))

    def _make_zoom_view(self):
        """Create zoomed view centered on active position."""
        ax, ay = self.active_pos
        fh, fw = self.frame.shape[:2]
        Z = ZOOM_SCALE
        r = ZOOM_RADIUS

        x1 = max(0, ax - r)
        y1 = max(0, ay - r)
        x2 = min(fw, ax + r)
        y2 = min(fh, ay + r)

        crop = self.frame[y1:y2, x1:x2].copy()
        zoomed = cv2.resize(crop, (crop.shape[1] * Z, crop.shape[0] * Z),
                            interpolation=cv2.INTER_NEAREST)

        if self.current_step < len(LANDMARKS):
            name, color, _, mode = LANDMARKS[self.current_step]
            zx = (ax - x1) * Z + Z // 2
            zy = (ay - y1) * Z + Z // 2
            self._draw_marker(zoomed, zx, zy, name, color, mode, Z, confirmed=False)

        # Old position in zoom (gray)
        if self.old_mount and self.current_step < len(LANDMARKS):
            lm_name, _, _, mode = LANDMARKS[self.current_step]
            old = self._old_pos(lm_name)
            if old and len(old) == 2:
                gray = (120, 120, 120)
                ox = (int(old[0]) - x1) * Z + Z // 2
                oy = (int(old[1]) - y1) * Z + Z // 2
                if mode == 'box':
                    hw, hh = get_box_half(lm_name, self.model)
                    hw, hh = hw * Z, hh * Z
                    cv2.rectangle(zoomed, (ox-hw, oy-hh), (ox+hw, oy+hh), gray, 1, cv2.LINE_AA)
                else:
                    cv2.drawMarker(zoomed, (ox, oy), gray, cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)

        return zoomed

    def _draw_info_panel(self, vis):
        """Draw step list and instructions at top of window."""
        h, w = vis.shape[:2]
        panel_h = 52 + len(LANDMARKS) * 24 + 30
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, vis, 0.2, 0, vis)

        y = 24
        put_text_bg(vis, "MOUNT CALIBRATION", (10, y), 0.7, (255, 255, 255),
                    thickness=2, bg_color=(30, 30, 30))
        help_text = "CLICK=place  ARROWS=nudge  SPACE=confirm  Z=zoom  +/-=bright  ESC=back"
        put_text_bg(vis, help_text, (280, y), 0.45, (180, 180, 180),
                    bg_color=(30, 30, 30))

        y += 32

        for i, (name, color, desc, mode) in enumerate(LANDMARKS):
            if i < self.current_step:
                pos = self.positions.get(name)
                tag = "box" if mode == 'box' else "pt"
                coords = f"({pos[0]}, {pos[1]})" if pos else "?"
                label = f"  OK  {desc}  {coords}"
                text_color = color
            elif i == self.current_step:
                if self.active_pos:
                    label = f"  >>  {desc}  => ({self.active_pos[0]}, {self.active_pos[1]})  [SPACE to confirm]"
                else:
                    action = "Click center" if mode == 'box' else "Click position"
                    label = f"  >>  {desc}  [{action}]"
                text_color = (255, 255, 255)
            else:
                label = f"      {desc}"
                text_color = (100, 100, 100)

            put_text_bg(vis, label, (10, y), 0.45, text_color, bg_color=(30, 30, 30))
            y += 24

        if self.current_step >= len(LANDMARKS):
            y += 6
            put_text_bg(vis, "ALL DONE!  Press S to save,  ESC to go back", (10, y),
                        0.6, (0, 255, 0), thickness=2, bg_color=(30, 30, 30))

        # Status indicators at bottom
        bx = w - 10
        if self.zoom_on:
            put_text_bg(vis, "ZOOM 4x", (bx - 100, h - 18), 0.5,
                        (0, 255, 255), bg_color=(30, 30, 30))
        if self.gamma != 1.0:
            put_text_bg(vis, f"gamma {self.gamma:.1f}", (bx - 230, h - 18), 0.5,
                        (200, 200, 100), bg_color=(30, 30, 30))

    def _make_side_panel(self, height):
        """Create info panel as a separate dark column for zoom mode."""
        pw = 280
        panel = np.zeros((height, pw, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        y = 20
        put_text_bg(panel, "MOUNT CALIBRATION", (8, y), 0.5, (255, 255, 255),
                    thickness=1, bg_color=(30, 30, 30))
        y += 28

        for i, (name, color, desc, mode) in enumerate(LANDMARKS):
            if i < self.current_step:
                pos = self.positions.get(name)
                coords = f"({pos[0]}, {pos[1]})" if pos else "?"
                label = f"OK  {desc}"
                text_color = color
                put_text_bg(panel, label, (8, y), 0.38, text_color, bg_color=(30, 30, 30))
                y += 16
                put_text_bg(panel, f"    {coords}", (8, y), 0.35, text_color, bg_color=(30, 30, 30))
            elif i == self.current_step:
                if self.active_pos:
                    label = f">>  {desc}"
                    coord_label = f"    ({self.active_pos[0]}, {self.active_pos[1]})"
                else:
                    label = f">>  {desc}"
                    coord_label = "    [click on image]"
                text_color = (255, 255, 255)
                put_text_bg(panel, label, (8, y), 0.38, text_color, bg_color=(30, 30, 30))
                y += 16
                put_text_bg(panel, coord_label, (8, y), 0.35, text_color, bg_color=(30, 30, 30))
            else:
                label = f"    {desc}"
                put_text_bg(panel, label, (8, y), 0.38, (100, 100, 100), bg_color=(30, 30, 30))
                y += 16
                # no second line for future steps
                y -= 16  # undo since we skip
            y += 20

        if self.current_step >= len(LANDMARKS):
            y += 4
            put_text_bg(panel, "ALL DONE!", (8, y), 0.5, (0, 255, 0),
                        thickness=1, bg_color=(30, 30, 30))
            y += 22
            put_text_bg(panel, "S=save  ESC=back", (8, y), 0.4, (0, 255, 0), bg_color=(30, 30, 30))

        # Keys help at bottom
        by = height - 60
        put_text_bg(panel, "SPACE = confirm", (8, by), 0.38, (180, 180, 180), bg_color=(30, 30, 30))
        by += 18
        put_text_bg(panel, "ARROWS = nudge", (8, by), 0.38, (180, 180, 180), bg_color=(30, 30, 30))
        by += 18
        put_text_bg(panel, "Z=zoom  +/-=bright", (8, by), 0.38, (180, 180, 180), bg_color=(30, 30, 30))

        if self.gamma != 1.0:
            put_text_bg(panel, f"gamma {self.gamma:.1f}", (pw - 100, 20), 0.4,
                        (200, 200, 100), bg_color=(30, 30, 30))

        return panel

    def _display_to_frame(self, dx, dy):
        """Convert display (window) coords to frame coords. Returns None if outside image."""
        if self.zoom_on and self.active_pos:
            # Zoom image is on the left; side panel on the right
            zoom_w = ZOOM_RADIUS * 2 * ZOOM_SCALE
            if dx >= zoom_w:
                return None  # Click on side panel, ignore
            ax, ay = self.active_pos
            fw = self.frame.shape[1]
            fh = self.frame.shape[0]
            r = ZOOM_RADIUS
            x1 = max(0, ax - r)
            y1 = max(0, ay - r)
            fx = x1 + dx // ZOOM_SCALE
            fy = y1 + dy // ZOOM_SCALE
            return max(0, min(fw-1, fx)), max(0, min(fh-1, fy))
        else:
            return dx // DISPLAY_SCALE, dy // DISPLAY_SCALE

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.current_step >= len(LANDMARKS):
            return
        result = self._display_to_frame(x, y)
        if result is None:
            return
        self.active_pos = result
        self.draw()

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        # Auto-detect corner or pre-place from old mount
        corner_result = find_corner_template(self.frame)
        if corner_result:
            self.active_pos = (corner_result[0], corner_result[1])
            print(f"Corner auto-detected at ({corner_result[0]}, {corner_result[1]}) "
                  f"score={corner_result[2]:.3f}")
        elif self.old_mount:
            old = self._old_pos('corner')
            if old and len(old) == 2:
                self.active_pos = (int(old[0]), int(old[1]))

        self.draw()

        while True:
            raw_key = cv2.waitKeyEx(0)
            key = raw_key & 0xFF

            if key == 27:  # ESC = go back
                if not self._step_back():
                    print("Cancelled.")
                    cv2.destroyAllWindows()
                    return None

            elif key == ord('z'):
                self.zoom_on = not self.zoom_on
                self.draw()

            elif key in (ord('+'), ord('=')):
                self._adjust_gamma(0.3)
                self.draw()

            elif key == ord('-'):
                self._adjust_gamma(-0.3)
                self.draw()

            elif key == ord(' '):
                if self.active_pos and self.current_step < len(LANDMARKS):
                    name = LANDMARKS[self.current_step][0]
                    self.positions[name] = self.active_pos
                    print(f"  {name}: ({self.active_pos[0]}, {self.active_pos[1]})")
                    self.current_step += 1
                    self.active_pos = None
                    self.zoom_on = False

                    # Pre-place next step from old mount
                    if self.current_step < len(LANDMARKS):
                        next_name = LANDMARKS[self.current_step][0]
                        old = self._old_pos(next_name)
                        if old and len(old) == 2:
                            self.active_pos = (int(old[0]), int(old[1]))
                    self.draw()

            elif key == ord('s') and self.current_step >= len(LANDMARKS):
                cv2.destroyAllWindows()
                return self.positions

            elif self.active_pos:
                dx, dy = 0, 0
                if raw_key in (63232, 65362):    dy = -1
                elif raw_key in (63233, 65364):  dy = 1
                elif raw_key in (63234, 65361):  dx = -1
                elif raw_key in (63235, 65363):  dx = 1
                if dx or dy:
                    ax, ay = self.active_pos
                    self.active_pos = (ax + dx, ay + dy)
                    self.draw()

        cv2.destroyAllWindows()
        return None


def _load_undistort():
    """Load camera intrinsics for undistortion. Returns (K, dist) or (None, None)."""
    camera_path = os.path.join(PROJECT_ROOT, 'calibration', 'camera.json')
    if not os.path.exists(camera_path):
        return None, None
    with open(camera_path) as f:
        cal = json.load(f)
    K = np.array(cal['camera_matrix'], dtype=np.float64)
    dist = np.array(cal['dist_coeffs'], dtype=np.float64)
    return K, dist


def _undistort_points(pts_raw, K, dist):
    """Undistort Nx2 raw pixel coords. Returns Nx2 ndarray."""
    pts = np.array(pts_raw, dtype=np.float64).reshape(-1, 1, 2)
    corrected = cv2.undistortPoints(pts, K, dist, P=K)
    return corrected.reshape(-1, 2)


def build_mount(positions, old_mount, model):
    """Build camera_mount.json v2 data from placed positions."""
    mount = {}
    mount['format_version'] = 2

    # positions['corner'] is center; corner_xy in JSON is top-left
    cx, cy = positions['corner']
    corner_tl = [cx - CORNER_HALF, cy - CORNER_HALF]

    # landmarks_raw: all 8 names -> [x, y] (corner stored as top-left)
    landmarks_raw = {}
    for lm_name, _, _, _ in LANDMARKS:
        if lm_name in positions:
            if lm_name == 'corner':
                landmarks_raw[lm_name] = corner_tl
            else:
                landmarks_raw[lm_name] = list(positions[lm_name])
    mount['landmarks_raw'] = landmarks_raw

    # landmarks_undist: undistorted coords
    K, dist = _load_undistort()
    if K is not None:
        names = list(landmarks_raw.keys())
        raw_pts = [landmarks_raw[n] for n in names]
        undist_pts = _undistort_points(raw_pts, K, dist)
        mount['landmarks_undist'] = {
            names[i]: [round(float(undist_pts[i, 0]), 2),
                       round(float(undist_pts[i, 1]), 2)]
            for i in range(len(names))
        }
    else:
        # No camera calibration — undist = raw
        mount['landmarks_undist'] = dict(landmarks_raw)

    # Backward-compat fields
    mount['corner_xy'] = corner_tl

    mount['button_centers'] = {}
    for name in ['B1', 'B2', 'S1', 'S2']:
        mount['button_centers'][name] = list(positions[name])

    mount['mute_center'] = list(positions['mute_led'])

    radius = model.get('mute_search_radius', 40) if model else 40
    mx, my = positions['mute_led']
    mount['mute_region'] = [mx - radius, my - radius, mx + radius, my + radius]

    # Panel rect from digit landmarks, extended to full panel area
    bl = positions['digit_left_bl']
    tr = positions['digit_right_tr']
    w = tr[0] - bl[0]
    h = bl[1] - tr[1]
    cx_panel = (bl[0] + tr[0]) / 2
    cy_panel = (tr[1] + bl[1]) / 2
    ew, eh = w * 1.85, h * 1.7
    mount['panel_rect'] = [int(cx_panel - ew / 2), int(cy_panel - eh / 2),
                           int(ew), int(eh)]

    return mount


def update_device_model(mount):
    """Recompute device_model.json from mount data using undistorted-space offsets."""
    model = load_json(MODEL_PATH)
    if model is None:
        print("WARNING: No device_model.json found, skipping model update.")
        return

    undist = mount.get('landmarks_undist', {})
    if not undist or 'corner' not in undist:
        # Fallback: raw offsets (no undistortion available)
        cx, cy = mount['corner_xy']
        buttons = mount['button_centers']
        new_landmarks = {'corner': [0.0, 0.0]}
        for name in ['B1', 'B2', 'S1', 'S2']:
            bx, by = buttons[name]
            new_landmarks[name] = [float(bx - cx), float(by - cy)]
        model['landmarks'] = new_landmarks
        save_json(MODEL_PATH, model)
        print(f"  Updated device_model.json (raw offsets, no undistortion)")
        return

    # Undistorted corner as origin
    ucx, ucy = undist['corner']

    # Compute undistorted landmark offsets for all 8 landmarks
    new_landmarks = {'corner': [0.0, 0.0]}
    for name in ['B1', 'B2', 'S1', 'S2', 'mute_led', 'digit_left_bl', 'digit_right_tr']:
        if name in undist:
            ux, uy = undist[name]
            new_landmarks[name] = [round(ux - ucx, 2), round(uy - ucy, 2)]

    # Panel offset from digit landmarks, extended to full panel area
    if 'digit_left_bl' in undist and 'digit_right_tr' in undist:
        bl = undist['digit_left_bl']
        tr = undist['digit_right_tr']
        w = tr[0] - bl[0]
        h = bl[1] - tr[1]
        cx_panel = (bl[0] + tr[0]) / 2
        cy_panel = (tr[1] + bl[1]) / 2
        ew, eh = w * 1.85, h * 1.7
        model['panel_offset'] = [round(cx_panel - ew / 2 - ucx),
                                  round(cy_panel - eh / 2 - ucy)]
        model['panel_size'] = [round(ew), round(eh)]

    # Mute offset
    if 'mute_led' in undist:
        model['mute_button_offset'] = [round(undist['mute_led'][0] - ucx),
                                        round(undist['mute_led'][1] - ucy)]

    model['landmarks'] = new_landmarks

    save_json(MODEL_PATH, model)
    print(f"  Updated device_model.json (undistorted offsets)")


def auto_capture_corner_template(frame, corner_pos):
    """If no corner templates exist, capture one from the frame."""
    existing = [f for f in os.listdir(TEMPLATE_DIR)
                if f.startswith('corner_') and f.endswith('.png')]
    if existing:
        return

    cx, cy = corner_pos
    half = 37
    h, w = frame.shape[:2]
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half + 1), min(h, cy + half + 1)

    gray = frame[y1:y2, x1:x2, 1]
    path = os.path.join(TEMPLATE_DIR, 'corner_1.png')
    cv2.imwrite(path, gray)
    print(f"  Auto-captured corner template: {path} ({gray.shape[1]}x{gray.shape[0]})")


def main():
    parser = argparse.ArgumentParser(description='Camera mount calibration')
    parser.add_argument('image', nargs='?', help='Path to a 640x480 image or rtsp:// URL (default: webcam.link)')
    args = parser.parse_args()

    source = args.image
    if source is None:
        link_path = os.path.join(PROJECT_ROOT, 'webcam.link')
        if os.path.exists(link_path):
            with open(link_path) as f:
                source = f.read().strip()
            print(f"Using webcam.link: {source[:20]}...")
        else:
            parser.error('No image specified and webcam.link not found')

    frame = extract_frame(source)
    old_mount = load_json(MOUNT_PATH)
    model = load_json(MODEL_PATH)

    if old_mount:
        print("Update mode: existing camera_mount.json loaded")
        print(f"  Old corner: {old_mount.get('corner_xy')}")
        print(f"  Old mute:   {old_mount.get('mute_center')}")
    else:
        print("From-scratch mode: no existing camera_mount.json")

    print()
    print(f"Place {len(LANDMARKS)} landmarks:")
    print("  CLICK on center  |  ARROWS nudge +/-1px  |  SPACE confirm")
    print("  Z = 4x zoom      |  +/- = brightness     |  ESC = go back")
    print()

    cal = Calibrator(frame, old_mount, model)
    positions = cal.run()

    if positions is None:
        return

    print(f"\n--- Results ---")
    for name, _, _, _ in LANDMARKS:
        pos = positions.get(name)
        old = cal._old_pos(name)
        if pos and old and len(old) == 2:
            dx, dy = pos[0] - int(old[0]), pos[1] - int(old[1])
            print(f"  {name:10s}: ({pos[0]:3d}, {pos[1]:3d})  delta=({dx:+d}, {dy:+d})")
        elif pos:
            print(f"  {name:10s}: ({pos[0]:3d}, {pos[1]:3d})")

    mount = build_mount(positions, old_mount, model)

    print(f"\nSaving...")

    if os.path.exists(MOUNT_PATH):
        date_str = datetime.now().strftime('%Y%m%d')
        bak_path = MOUNT_PATH.replace('.json', f'.{date_str}.bak.json')
        i = 1
        while os.path.exists(bak_path):
            bak_path = MOUNT_PATH.replace('.json', f'.{date_str}_{i}.bak.json')
            i += 1
        os.rename(MOUNT_PATH, bak_path)
        print(f"  Backed up old mount: {os.path.basename(bak_path)}")

    save_json(MOUNT_PATH, mount)
    print(f"  Saved camera_mount.json")

    update_device_model(mount)
    auto_capture_corner_template(frame, positions['corner'])

    print(f"\nDone! Run 'python scripts/update_device_model.py --dry-run' to verify.")


if __name__ == '__main__':
    main()
