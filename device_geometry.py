"""
Device Geometry Model

Maps device-space coordinates to pixel-space coordinates. All spatial
relationships between device features (panel, buttons, corner) are defined
in device-space and projected to pixel-space via a transform.

Transform pipeline:
  Phase 1: Simple translation (corner_pixel + offset)
  Phase 2: + ROI undistortion for template matching
  Phase 3: + Homography from 4 landmarks (corner + 3 buttons)
"""

import json
import os
import cv2
import numpy as np


class DeviceGeometry:
    """Maps device-space coordinates to pixel-space via detected landmarks."""

    def __init__(self, model_path=None):
        """Load device model from JSON.

        Args:
            model_path: Path to device_model.json. If None, uses default.
        """
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), 'calibration', 'device_model.json')

        with open(model_path) as f:
            model = json.load(f)

        # Device-space offsets (origin = corner template center)
        self.panel_offset = tuple(model['panel_offset'])       # (-266, -86)
        self.panel_size = tuple(model['panel_size'])            # (145, 105)
        self.mute_offset = tuple(model['mute_button_offset'])  # (200, 43)
        self.mute_search_radius = model['mute_search_radius']  # 40
        self.button_panel_gap = model['button_panel_gap']       # 65
        self.button_search_top_offset = model['button_search_top_offset']  # 20

        # Corner search parameters
        self.corner_search_size = model['corner_search_size']          # 150
        self.corner_search_position = tuple(model['corner_search_position'])  # (0.58, 0.57)

        # Button region parameters
        self.button_region_right_ratio = model['button_region_right_ratio']  # 0.65
        self.button_region_top_ratio = model['button_region_top_ratio']      # 0.70

        # LED detection parameters
        self.led_min_area = model['led_min_area']              # 60
        self.led_max_area = model['led_max_area']              # 1200
        self.led_max_aspect_ratio = model['led_max_aspect_ratio']  # 3

        # Button zone layout (ratios within button region)
        self.button_zone_centers = model['button_zone_centers']        # B1:0.10, B2:0.33, ...
        self.button_zone_width_ratio = model['button_zone_width_ratio']   # 0.20
        self.button_zone_top_ratio = model['button_zone_top_ratio']       # 0.35
        self.button_zone_bottom_ratio = model['button_zone_bottom_ratio'] # 0.90
        self.zone_enlarge_px = tuple(model['zone_enlarge_px'])            # (20, 30, 20)

        # Frame diff ROI
        self.frame_diff_roi = tuple(model['frame_diff_roi'])

        # Button region geometry
        self.button_region_right_margin = model['button_region_right_margin']  # 20

        # B1 LED prediction parameters
        self.b1_led_position_ratio = model['b1_led_position_ratio']      # 0.88
        self.b1_led_min_visible_px = model['b1_led_min_visible_px']      # 15
        self.b1_led_edge_margin = model['b1_led_edge_margin']            # 18
        self.b1_b2_spacing = model['b1_b2_spacing']                      # 5

        # Button detection parameters
        self.button_min_width = model['button_min_width']                # 20
        self.button_min_height = model['button_min_height']              # 30
        self.button_edge_margin = model['button_edge_margin']            # 5
        self.button_nms_merge_px = model['button_nms_merge_px']          # 20

        # Device-space landmark positions (for homography)
        landmarks = model.get('landmarks', {})
        self.landmark_positions = {
            name: np.array(pos, dtype=np.float64)
            for name, pos in landmarks.items()
        }  # (200, 350, 100, 350)

        # Mute contrast detection parameters (#72)
        self.mute_led_patch_radius = model.get('mute_led_patch_radius', 6)
        self.mute_ref_offset = (model.get('mute_ref_offset_dx', -26),
                                model.get('mute_ref_offset_dy', 0))
        self.mute_contrast_threshold = 1.1

        # Transform state (Phase 1: translation only)
        self._corner_xy = None    # Last known corner position in pixels
        self._homography = None   # Phase 3: 2x3 affine or 3x3 perspective matrix
        self._homography_is_perspective = False
        self._scale = 1.0         # Phase 3: scale from homography
        self._geo_method = 'none' # Tracks panel projection method: homography/offset/none

        # Smoothed homography state (#72: local contrast mute detection)
        self._smoothed_homography = None   # EMA-smoothed 2x3 matrix
        self._smoothed_scale = 1.0
        self._ema_alpha = 0.03             # ~33 frame time constant (2.2s at 15fps)
        self._homography_age = 0           # frames since last compute_homography()
        self._homography_residuals = {}    # per-landmark reprojection residuals (#85)
        self._max_residual = 0.0           # max reprojection residual (#85)

        # Landmark tracking state (--track mode)
        self._tracking_enabled = False
        self._golden_landmarks = None   # (corner_xy, button_centers) from last good frame
        self._golden_homography = None  # 2x3 affine matrix copy
        self._golden_corner_xy = None   # (x, y) copy
        self._golden_scale = None       # scale copy

        # Camera mount calibration reference (for alignment overlay)
        self._calibration_ref = None
        self._calibration_path = os.path.join(
            os.path.dirname(model_path), 'camera_mount.json')
        self._golden_path = os.path.join(
            os.path.dirname(model_path), 'golden_state.json')

        # Intrinsics (Phase 2: distortion correction)
        self._camera_matrix = None
        self._dist_coeffs = None
        self._map_x = None  # Precomputed remap table
        self._map_y = None

        # Auto-load camera intrinsics if available
        camera_path = os.path.join(
            os.path.dirname(model_path), 'camera.json')
        if os.path.exists(camera_path):
            self._load_intrinsics(camera_path)

        # Calibrated button centers from camera_mount.json (raw pixel coords)
        self._calibrated_button_centers = None

        # Auto-load initial homography from camera_mount.json
        self._load_initial_homography()

    def _load_intrinsics(self, camera_path):
        """Load camera intrinsics from JSON and precompute remap tables."""
        with open(camera_path) as f:
            cal = json.load(f)
        frame_size = tuple(cal['frame_size'])
        self.set_intrinsics(cal['camera_matrix'], cal['dist_coeffs'], frame_size)

    def _load_initial_homography(self):
        """Compute initial homography from camera_mount.json calibration data."""
        if not os.path.exists(self._calibration_path):
            print(f"WARNING: {self._calibration_path} not found. "
                  "Run 'python scripts/calibrate_mount.py' to create it. "
                  "Mute detection will be disabled until calibration is done.")
            return
        try:
            with open(self._calibration_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return
        if 'corner_xy' not in data or 'button_centers' not in data:
            return
        corner_xy = tuple(data['corner_xy'])
        button_centers = {k: tuple(v) for k, v in data['button_centers'].items()}
        self._calibrated_button_centers = button_centers
        self.compute_homography(corner_xy, button_centers)

    def set_corner(self, corner_x, corner_y):
        """Update corner position for translation-based projection."""
        self._corner_xy = (corner_x, corner_y)

    # -----------------------------------------------------------------
    # Projection: device-space -> pixel-space
    # -----------------------------------------------------------------

    def redistort_points(self, points):
        """Re-distort points from undistorted space back to raw pixel space.

        Args:
            points: Nx2 array of undistorted coordinates.

        Returns:
            Nx2 array of distorted (raw pixel) coordinates.
        """
        if not self.has_intrinsics():
            return points
        pts = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
        # undistortPoints with no P gives normalized coords; with P=K gives pixel coords
        # To redistort: go normalized → apply distortion → back to pixel
        normalized = cv2.undistortPoints(pts, self._camera_matrix,
                                         np.zeros(5))  # no distortion = just normalize
        # Actually we need the inverse of undistort. OpenCV doesn't have a direct
        # redistort function. Use projectPoints with rvec=0, tvec=0.
        obj_pts = np.zeros((len(points), 3), dtype=np.float64)
        # Convert undistorted pixel coords to normalized coords
        fx = self._camera_matrix[0, 0]
        fy = self._camera_matrix[1, 1]
        cx = self._camera_matrix[0, 2]
        cy = self._camera_matrix[1, 2]
        pts_flat = np.array(points, dtype=np.float64).reshape(-1, 2)
        obj_pts[:, 0] = (pts_flat[:, 0] - cx) / fx
        obj_pts[:, 1] = (pts_flat[:, 1] - cy) / fy
        obj_pts[:, 2] = 1.0
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec,
                                        self._camera_matrix, self._dist_coeffs)
        return img_pts.reshape(-1, 2)

    def _project(self, dx, dy):
        """Project device-space offset to pixel coordinates.

        Phase 1: Simple translation from corner.
        Phase 3: Transform in undistorted space, then re-distort.
        Supports both 2x3 affine and 3x3 perspective transforms.
        """
        if self._homography is not None:
            pt = np.array([[[dx, dy]]], dtype=np.float32)
            if self._homography.shape == (3, 3):
                projected = cv2.perspectiveTransform(pt, self._homography)
            else:
                projected = cv2.transform(pt, self._homography)
            ux, uy = projected[0, 0, 0], projected[0, 0, 1]
            if self.has_intrinsics():
                raw = self.redistort_points(np.array([[ux, uy]]))[0]
                return int(round(raw[0])), int(round(raw[1]))
            return int(round(ux)), int(round(uy))

        if self._corner_xy is None:
            return None
        return (self._corner_xy[0] + dx, self._corner_xy[1] + dy)

    # -----------------------------------------------------------------
    # Region accessors
    # -----------------------------------------------------------------

    def get_panel_rect(self, corner_x=None, corner_y=None):
        """Get panel rectangle in pixel coordinates.

        Uses homography projection if available, else translation from corner.

        Args:
            corner_x, corner_y: Corner position. If None, uses stored corner.

        Returns:
            (x, y, w, h) or None if corner unknown.
        """
        if self._homography is not None:
            # Project panel top-left through similarity transform
            result = self._project(self.panel_offset[0], self.panel_offset[1])
            if result is not None:
                pw = int(round(self.panel_size[0] * self._scale))
                ph = int(round(self.panel_size[1] * self._scale))
                return (result[0], result[1], pw, ph)

        # Fallback: translation from corner
        if corner_x is not None:
            cx, cy = corner_x, corner_y
        elif self._corner_xy is not None:
            cx, cy = self._corner_xy
        else:
            return None

        px = cx + self.panel_offset[0]
        py = cy + self.panel_offset[1]
        return (px, py, self.panel_size[0], self.panel_size[1])

    def get_mute_region(self, corner_x=None, corner_y=None):
        """Get mute button search region.

        Uses homography projection if available, else translation from corner.

        Returns:
            (cx, cy, half) - center and half-size of search box, or None.
        """
        if self._homography is not None:
            result = self._project(self.mute_offset[0], self.mute_offset[1])
            if result is not None:
                half = int(round(self.mute_search_radius * self._scale))
                return (result[0], result[1], half)

        # Fallback: translation from corner
        if corner_x is not None:
            cx, cy = corner_x, corner_y
        elif self._corner_xy is not None:
            cx, cy = self._corner_xy
        else:
            return None

        btn_x = cx + self.mute_offset[0]
        btn_y = cy + self.mute_offset[1]
        return (btn_x, btn_y, self.mute_search_radius)

    def get_noise_region(self):
        """Get neutral region between corner and mute button for noise measurement.

        The device housing between corner and mute LED has no light-emitting
        elements, so green-channel std dev here reflects camera gain noise.

        Returns:
            (cx, cy, half) - center and half-size of region, or None.
        """
        # Neutral housing surface between corner (0,0) and mute button
        mid_x = 80
        mid_y = 75
        half = 20  # 40x40 device-space region

        if self._homography is not None:
            result = self._project(mid_x, mid_y)
            if result is not None:
                scaled_half = int(round(half * self._scale))
                return (result[0], result[1], scaled_half)

        if self._corner_xy is not None:
            cx = self._corner_xy[0] + mid_x
            cy = self._corner_xy[1] + mid_y
            return (cx, cy, half)

        return None

    def get_corner_search_region(self, frame_w, frame_h):
        """Get corner template search region.

        If a corner was previously found, centers the search region on it
        (with margin for camera drift). Otherwise uses default frame-ratio
        position for first-frame detection.

        Returns:
            (x, y, size) - top-left corner and size of search square.
        """
        size = self.corner_search_size
        if self._corner_xy is not None:
            # Center search region on last known corner
            # Use template-aware offset: corner position corresponds to top-left
            # of the template crop, not center of search region. Place the search
            # region so the expected corner position has margin on all sides.
            margin = 40  # Pixels of margin around expected position
            cx, cy = self._corner_xy
            x = max(0, min(frame_w - size, cx - margin))
            y = max(0, min(frame_h - size, cy - margin))
        else:
            # First frame: use default position (right half of frame)
            x = int(frame_w * self.corner_search_position[0])
            y = int(frame_h * self.corner_search_position[1])
        return (x, y, size)

    def get_button_region_from_geometry(self, frame_w, frame_h):
        """Get button region using projected geometry (Phase 4).

        When homography/corner is known, derives button region from the
        projected landmark positions instead of fixed frame ratios.

        Returns:
            (top, bottom, left, right) or None if no geometry available.
        """
        if self._corner_xy is None:
            return None
        cx, cy = self._corner_xy
        # Buttons are below corner, to its left
        top = cy + self.button_search_top_offset
        bottom = frame_h
        left = 0
        right = min(frame_w, cx + self.button_region_right_margin)
        return (top, bottom, left, right)

    def get_button_region_fallback(self, frame_w, frame_h):
        """Get button region when panel_rect is None.

        Uses last-known corner position if available, otherwise frame ratios.

        Returns:
            (top, bottom, left, right) in pixels.
        """
        if self._corner_xy is not None:
            return self.get_button_region_from_geometry(frame_w, frame_h)
        top = int(frame_h * self.button_region_top_ratio)
        bottom = frame_h
        left = 0
        right = int(frame_w * self.button_region_right_ratio)
        return (top, bottom, left, right)

    def get_button_region_from_panel(self, panel_rect, frame_w, frame_h):
        """Get button region based on panel position.

        Returns:
            (top, bottom, left, right) in pixels.
        """
        px, py, pw, ph = panel_rect
        top = py + ph
        bottom = frame_h
        left = 0
        right = int(frame_w * self.button_region_right_ratio)
        return (top, bottom, left, right)

    def get_default_button_zones(self, bw, bh):
        """Get default button zones (ratios within button region).

        Returns:
            List of (left_x, right_x, top_y, bottom_y, name) tuples.
        """
        zone_width = bw * self.button_zone_width_ratio
        zone_top = int(bh * self.button_zone_top_ratio)
        zone_bottom = int(bh * self.button_zone_bottom_ratio)
        return [
            (bw * frac - zone_width / 2, bw * frac + zone_width / 2,
             zone_top, zone_bottom, name)
            for name, frac in self.button_zone_centers.items()
        ]

    def enlarge_zones(self, button_zones, bw, bh):
        """Enlarge button zones for fallback detection.

        Args:
            button_zones: List of (left_x, right_x, top_y, bottom_y, name).
            bw, bh: Button region dimensions.

        Returns:
            Enlarged zones list.
        """
        lr, top, bot = self.zone_enlarge_px
        return [
            (max(0, left_x - lr), min(bw, right_x + lr),
             max(0, top_y - top), min(bh, bottom_y + bot), name)
            for left_x, right_x, top_y, bottom_y, name in button_zones
        ]

    def get_frame_diff_roi(self):
        """Get frame diff ROI coordinates.

        Returns:
            (y1, y2, x1, x2) tuple.
        """
        return self.frame_diff_roi

    # -----------------------------------------------------------------
    # Phase 2 stubs (implemented in Phase 2)
    # -----------------------------------------------------------------

    def set_intrinsics(self, camera_matrix, dist_coeffs, frame_size=(640, 480)):
        """Set camera intrinsics for distortion correction.

        Precomputes remap tables for ROI undistortion.
        """
        self._camera_matrix = np.array(camera_matrix, dtype=np.float64)
        self._dist_coeffs = np.array(dist_coeffs, dtype=np.float64)
        self._map_x, self._map_y = cv2.initUndistortRectifyMap(
            self._camera_matrix, self._dist_coeffs, None,
            self._camera_matrix, frame_size, cv2.CV_32FC1)

    def has_intrinsics(self):
        """Check if camera intrinsics are available."""
        return self._camera_matrix is not None

    def get_undistort_shift(self, x, y, w, h):
        """Get max pixel displacement from undistortion in an ROI.

        Returns:
            Max pixel shift (float), or 0.0 if no intrinsics.
        """
        if self._map_x is None:
            return 0.0
        dx = self._map_x[y:y+h, x:x+w] - np.arange(x, x+w, dtype=np.float32)
        dy = self._map_y[y:y+h, x:x+w] - np.arange(y, y+h, dtype=np.float32).reshape(-1, 1)
        return float(np.sqrt(dx**2 + dy**2).max())

    def undistort_points(self, points):
        """Undistort landmark points for accurate homography.

        Args:
            points: Nx2 array of pixel coordinates.

        Returns:
            Nx2 array of corrected coordinates, or original if no intrinsics.
        """
        if not self.has_intrinsics():
            return points

        pts = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
        corrected = cv2.undistortPoints(pts, self._camera_matrix,
                                        self._dist_coeffs, P=self._camera_matrix)
        return corrected.reshape(-1, 2)

    def undistort_roi(self, frame, x, y, w, h, derotate=True):
        """Undistort and optionally correct geometry of a small ROI.

        When derotate=True and homography is active, applies de-rotation
        (if > 0.5 deg) and scale normalization (if > 2% off) to undo
        camera rotation and zoom so the crop matches template dimensions.

        Args:
            frame: Full BGR frame.
            x, y, w, h: ROI rectangle.
            derotate: If True, correct rotation and scale from homography.

        Returns:
            Processed ROI (undistorted, de-rotated, scale-normalized).
        """
        if self._map_x is None:
            roi = frame[y:y+h, x:x+w]
        else:
            roi = frame[y:y+h, x:x+w]
            map_x_roi = self._map_x[y:y+h, x:x+w]
            map_y_roi = self._map_y[y:y+h, x:x+w]
            roi = cv2.remap(roi, map_x_roi - x, map_y_roi - y, cv2.INTER_LINEAR)

        if derotate and self._homography is not None:
            angle = self.get_rotation_deg()
            need_rotate = abs(angle) > 0.5
            need_scale = abs(self._scale - 1.0) > 0.02

            if need_rotate or need_scale:
                h_roi, w_roi = roi.shape[:2]
                inv_scale = 1.0 / self._scale if need_scale else 1.0
                rot_angle = angle if need_rotate else 0.0
                new_w = int(round(w_roi * inv_scale))
                new_h = int(round(h_roi * inv_scale))
                # Rotate around input center, then shift to output center
                M = cv2.getRotationMatrix2D((w_roi / 2, h_roi / 2), rot_angle, inv_scale)
                M[0, 2] += (new_w - w_roi) / 2
                M[1, 2] += (new_h - h_roi) / 2
                roi = cv2.warpAffine(roi, M, (new_w, new_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
        return roi

    # -----------------------------------------------------------------
    # Phase 3 stubs (implemented in Phase 3)
    # -----------------------------------------------------------------

    def compute_homography(self, corner_xy, button_centers):
        """Compute similarity transform from detected landmarks.

        Uses corner + available button correspondences to fit a similarity
        transform (translation + rotation + uniform scale) mapping
        device-space offsets -> raw pixel coordinates.

        With 1 button (2 points), the similarity transform is exactly
        determined. With 2+ buttons, it's overdetermined (least squares).

        Args:
            corner_xy: (x, y) of detected corner in raw frame pixels.
            button_centers: Dict of button name -> (x, y) in raw frame pixels.
                           Any subset of {'B2', 'S1', 'S2'}.

        Returns:
            True if transform was computed, False on failure.
        """
        if 'corner' not in self.landmark_positions:
            return False

        # Accept any landmarks present in both button_centers and model
        available = [name for name in button_centers
                     if name in self.landmark_positions]
        if len(available) < 3:
            return False

        # Device-space points (offsets from corner in raw pixels)
        src_pts = [self.landmark_positions['corner'].astype(np.float32)]
        # Pixel-space points (destination)
        dst_pts = [np.array(corner_xy, dtype=np.float32)]

        for name in available:
            src_pts.append(self.landmark_positions[name].astype(np.float32))
            dst_pts.append(np.array(button_centers[name], dtype=np.float32))

        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)

        # Undistort destination points so the similarity transform operates
        # in undistorted space — makes extrapolation to B1 accurate
        if self.has_intrinsics():
            dst = self.undistort_points(dst).astype(np.float32)

        # Similarity transform (4 DOF: rotation, scale, translation)
        # Even with 4+ points, similarity is more stable than full homography
        # because the small number of landmarks makes 8-DOF underdetermined
        M, inliers = cv2.estimateAffinePartial2D(src, dst)
        if M is None:
            return False
        self._homography = M  # 2x3 affine matrix
        self._homography_is_perspective = False
        self._corner_xy = tuple(corner_xy)
        self._scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)

        # Reprojection residuals (#85)
        projected = cv2.transform(src.reshape(-1, 1, 2), M).reshape(-1, 2)
        residuals = np.linalg.norm(projected - dst, axis=1)
        self._homography_residuals = dict(zip(['corner'] + available, residuals.tolist()))
        self._max_residual = float(np.max(residuals))

        # Reset age; smoothed homography updated every frame via
        # increment_homography_age() (#72)
        self._homography_age = 0

        return True

    def get_scale(self):
        """Get current scale factor from homography.

        Returns:
            Scale factor (1.0 if no homography).
        """
        return self._scale

    def get_rotation_deg(self):
        """Get rotation angle from similarity transform.

        Returns:
            Rotation in degrees (0.0 if no homography).
        """
        if self._homography is None:
            return 0.0
        return np.degrees(np.arctan2(self._homography[1, 0], self._homography[0, 0]))

    def project_landmark(self, name):
        """Project a named landmark to pixel coordinates.

        Returns:
            (x, y) in pixel coords, or None if landmark unknown or no transform.
        """
        if name not in self.landmark_positions:
            return None
        pos = self.landmark_positions[name]
        return self._project(pos[0], pos[1])

    # -----------------------------------------------------------------
    # Smoothed homography (#72: local contrast mute detection)
    # -----------------------------------------------------------------

    def _update_smoothed_homography(self, M):
        """EMA update on affine/perspective matrix elements."""
        if self._smoothed_homography is None or self._smoothed_homography.shape != M.shape:
            self._smoothed_homography = M.copy()
            self._smoothed_scale = self._scale
        else:
            alpha = self._ema_alpha
            self._smoothed_homography = (
                alpha * M + (1 - alpha) * self._smoothed_homography)
            self._smoothed_scale = (
                alpha * self._scale + (1 - alpha) * self._smoothed_scale)

    def _project_point(self, M, dx, dy):
        """Apply affine or perspective transform to device offset -> raw frame coords (float).

        Projects from device space to undistorted space via M,
        then re-distorts to raw pixel coordinates.
        """
        if M.shape == (3, 3):
            w = M[2, 0] * dx + M[2, 1] * dy + M[2, 2]
            ux = (M[0, 0] * dx + M[0, 1] * dy + M[0, 2]) / w
            uy = (M[1, 0] * dx + M[1, 1] * dy + M[1, 2]) / w
        else:
            ux = M[0, 0] * dx + M[0, 1] * dy + M[0, 2]
            uy = M[1, 0] * dx + M[1, 1] * dy + M[1, 2]
        if self.has_intrinsics():
            raw = self.redistort_points(np.array([[ux, uy]]))[0]
            return (float(raw[0]), float(raw[1]))
        return (float(ux), float(uy))

    def get_mute_led_center(self, smoothed=True):
        """Project mute LED center to frame coords.

        Args:
            smoothed: If True, use smoothed homography; else raw.

        Returns:
            (x, y) as floats, or None if no homography.
        """
        M = self._smoothed_homography if smoothed else self._homography
        if M is None:
            return None
        return self._project_point(M, self.mute_offset[0], self.mute_offset[1])

    def get_mute_ref_center(self, smoothed=True):
        """Project mute reference patch center to frame coords.

        The reference is offset from LED center by mute_ref_offset in device space.

        Returns:
            (x, y) as floats, or None if no homography.
        """
        M = self._smoothed_homography if smoothed else self._homography
        if M is None:
            return None
        dx = self.mute_offset[0] + self.mute_ref_offset[0]
        dy = self.mute_offset[1] + self.mute_ref_offset[1]
        return self._project_point(M, dx, dy)

    def increment_homography_age(self):
        """Increment homography age counter and update smoothed homography.

        Called each frame. The EMA update runs every frame (not just on
        detection frames) so that convergence speed is independent of
        frame-skip rate.
        """
        self._homography_age += 1
        if self._homography is not None:
            self._update_smoothed_homography(self._homography)

    def get_homography_residuals(self):
        """Return per-landmark reprojection residuals from last compute_homography()."""
        return self._homography_residuals

    def get_max_residual(self):
        """Return max reprojection residual from last compute_homography()."""
        return self._max_residual

    def get_homography_age(self):
        """Return frames since last compute_homography() (0 = fresh)."""
        return self._homography_age

    def has_homography(self):
        """Return True if a homography matrix is available."""
        return self._homography is not None

    # -----------------------------------------------------------------
    # Landmark tracking (--track mode)
    # -----------------------------------------------------------------

    def set_tracking(self, enabled):
        """Enable or disable landmark tracking.

        When enabled, loads calibration reference and persisted golden state.
        When disabled, clears all golden state.
        """
        self._tracking_enabled = enabled
        if enabled:
            self.load_calibration_ref()
            self._load_golden_from_disk()
        else:
            self._golden_landmarks = None
            self._golden_homography = None
            self._golden_corner_xy = None
            self._golden_scale = None

    def update_golden(self, corner_xy, button_centers):
        """Store or update golden landmark positions.

        First call: stores baseline. Subsequent calls: updates if any
        landmark moved more than 5px (camera bump).

        Args:
            corner_xy: (x, y) of detected corner.
            button_centers: Dict of button name -> (x, y).
        """
        if not self._tracking_enabled:
            return

        if self._golden_landmarks is None:
            # First time — store baseline
            self._golden_landmarks = (corner_xy, dict(button_centers))
            self._golden_homography = self._homography.copy()
            self._golden_corner_xy = self._corner_xy
            self._golden_scale = self._scale
            self._save_golden_to_disk()
            return

        # Check max landmark displacement vs stored golden
        old_corner, old_buttons = self._golden_landmarks
        max_disp = np.hypot(corner_xy[0] - old_corner[0],
                            corner_xy[1] - old_corner[1])
        for name in button_centers:
            if name in old_buttons:
                dx = button_centers[name][0] - old_buttons[name][0]
                dy = button_centers[name][1] - old_buttons[name][1]
                max_disp = max(max_disp, np.hypot(dx, dy))

        if max_disp > 5.0:
            self._golden_landmarks = (corner_xy, dict(button_centers))
            self._golden_homography = self._homography.copy()
            self._golden_corner_xy = self._corner_xy
            self._golden_scale = self._scale
            self._save_golden_to_disk()

    def restore_golden(self):
        """Restore homography, corner, and scale from golden copies.

        Returns:
            True if golden state was restored, False if no golden available.
        """
        if self._golden_homography is None:
            return False

        self._homography = self._golden_homography.copy()
        self._corner_xy = self._golden_corner_xy
        self._scale = self._golden_scale
        # Reset smoothed to golden on camera bump (#72)
        self._smoothed_homography = self._golden_homography.copy()
        self._smoothed_scale = self._golden_scale
        return True

    # -----------------------------------------------------------------
    # Calibration reference (alignment overlay)
    # -----------------------------------------------------------------

    def load_calibration_ref(self):
        """Load camera_mount.json into _calibration_ref for alignment overlay."""
        if not os.path.exists(self._calibration_path):
            self._calibration_ref = None
            return
        try:
            with open(self._calibration_path) as f:
                self._calibration_ref = json.load(f)
        except (json.JSONDecodeError, IOError):
            self._calibration_ref = None

    def get_calibration_ref(self):
        """Return calibration reference dict, or None if not loaded."""
        return self._calibration_ref

    def get_calibrated_button_centers(self):
        """Return calibrated button centers from camera_mount.json.

        Returns:
            Dict of button name -> (x, y) in raw pixel coords, or None.
        """
        return self._calibrated_button_centers

    # -----------------------------------------------------------------
    # Golden state persistence (disk)
    # -----------------------------------------------------------------

    def _save_golden_to_disk(self):
        """Write current golden landmarks + panel_rect to golden_state.json."""
        if self._golden_landmarks is None:
            return
        corner_xy, button_centers = self._golden_landmarks
        panel_rect = self.get_panel_rect()
        mute_region = self.get_mute_region()
        data = {
            'corner_xy': list(corner_xy),
            'button_centers': {k: list(v) for k, v in button_centers.items()},
        }
        if panel_rect is not None:
            data['panel_rect'] = list(panel_rect)
        if mute_region is not None:
            cx, cy, half = mute_region
            data['mute_region'] = [cx - half, cy - half, cx + half, cy + half]
        try:
            with open(self._golden_path, 'w') as f:
                json.dump(data, f, indent=2)
                f.write('\n')
        except IOError:
            pass

    def _load_golden_from_disk(self):
        """Load golden_state.json into golden landmark fields.

        Falls back to camera_mount.json if golden_state.json doesn't exist.
        """
        path = self._golden_path
        if not os.path.exists(path):
            path = self._calibration_path
            if not os.path.exists(path):
                return
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return
        if 'corner_xy' not in data or 'button_centers' not in data:
            return

        corner_xy = tuple(data['corner_xy'])
        button_centers = {k: tuple(v) for k, v in data['button_centers'].items()}
        self._golden_landmarks = (corner_xy, button_centers)

        # Recompute homography from loaded landmarks
        if self.compute_homography(corner_xy, button_centers):
            self._golden_homography = self._homography.copy()
            self._golden_corner_xy = self._corner_xy
            self._golden_scale = self._scale


# Module-level singleton for convenience
_default_geometry = None


def get_geometry(model_path=None):
    """Get or create the default DeviceGeometry instance."""
    global _default_geometry
    if _default_geometry is None:
        _default_geometry = DeviceGeometry(model_path)
    return _default_geometry
