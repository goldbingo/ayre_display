#!/usr/bin/env python3
"""
7-Segment Display Reader

Steps:
1. Panel Detection - Find the dark panel containing blue LED digits
2. Slant Correction - Correct for italic/slanted digits
3. Digit Separation - Find gap and define digit bounding boxes
4. Adaptive Caching - Cache panel/slant to avoid re-detection every frame
5. Recognition - Intensity-based 7-segment pattern matching
"""

import cv2
import glob
import numpy as np
import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from device_geometry import get_geometry as _get_geometry


@dataclass
class FrameResult:
    """Result of a full single-frame detection (digits + corner + LED + mute)."""
    reading: str
    cache_hit: bool
    led_status: str
    mute_status: str
    corner_result: Optional[tuple] = None       # (x, y, score, tmpl_idx) or None
    corner_debug: Optional[tuple] = None         # (search_rect, match_rect, crop_size) or None
    led_debug_info: Optional[dict] = None        # None during washout
    mute_debug_info: Optional[dict] = None       # None during washout
    noise_mean: Optional[float] = None
    washout: bool = False
    panel_rect: Optional[tuple] = None           # (x, y, w, h) or None
    detection_method: Optional[str] = None       # 'landmark', 'tracked', 'calibrated'
    last_led_debug: Optional[dict] = None        # Cached from last non-washout frame
    last_mute_debug: Optional[dict] = None       # Cached from last non-washout frame


# Corner result cache — set in predict_panel_from_landmarks(), consumed by detect()
_frame_corner_result = None   # (x, y, score, tmpl_idx) or None
_frame_corner_debug = None    # (search_rect, match_rect, crop_size) or None

# Unified cache file for button zones and panel detection
_CACHE_FILE = os.path.join(os.path.dirname(__file__), 'last_ref.txt')
_ZONE_CHANGE_THRESHOLD = 10  # Pixels - only save if zones shift by more than this

# Cache for button zone centers (adaptive from detected buttons)
_button_zone_cache = None
# Track LED detection failures while using cache
_cache_led_fail_count = 0
_CACHE_FAIL_THRESHOLD = 10  # Switch to enlarged zones after this many failures

# Cache for detected buttons from predict_panel_from_landmarks() (#74)
# Reused by detect_button_leds() to avoid redundant _detect_buttons() call
_cached_buttons = None  # (region_bounds_tuple, sorted_buttons_list) or None
_frame_led_dots = None  # Per-frame LED dot info from predict_panel_from_landmarks(), consumed by detect_button_leds()

# --- LED diff experiment ---
_led_diff_snapshots = None   # dict: zone_name → grayscale crop (with 2px padding)
_led_diff_zones = None       # dict: zone_name → (x1, y1, x2, y2) — padded zone bounds at snapshot time
_led_diff_lit = None         # last lit LED name
_led_diff_log = None         # file handle for experiment CSV
_led_diff_frame_n = 0        # frame counter for experiment
_LED_DIFF_PAD = 2            # hysteresis padding in pixels

# Logging configuration
_LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
_LOG_ENABLED = True

def disable_logging():
    """Disable all file logging."""
    global _LOG_ENABLED
    _LOG_ENABLED = False

def set_log_dir(log_dir):
    """Set the log directory path. Must be called before _init_log()."""
    global _LOG_DIR
    _LOG_DIR = log_dir

def set_undistort(use_undistort=True):
    """Deprecated: undistortion is always enabled. Kept for caller compatibility."""
    pass

_TRACKING = False  # When True: store/restore golden landmark positions

def set_tracking(enabled):
    """Enable/disable landmark tracking (--track mode)."""
    global _TRACKING
    _TRACKING = enabled
    _geometry.set_tracking(enabled)

def get_geometry():
    """Return the DeviceGeometry singleton."""
    return _geometry

_LOG_COOLDOWN = 30  # Seconds between saves of same issue type
_LOG_MAX_FRAMES = 1000  # Max issue frames to keep
_log_last_save = {}  # issue_type -> timestamp
_log_file = None  # CSV file handle

# Corner templates for pattern matching (used for red button detection)
_corner_templates = None
_corner_template_idx = 0  # Current preferred template (round-robin with sticky preference)
_CORNER_TEMPLATE_FILES = sorted(glob.glob(
    os.path.join(os.path.dirname(__file__), 'templates', 'corner_*.png')
))
# Device geometry model (all spatial constants derived from here)
_geometry = _get_geometry()

# Red button offset from corner center
_RED_BUTTON_OFFSET = _geometry.mute_offset

# Digit templates for pattern matching
_digit_templates = None
_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')
_TEMPLATE_SIZE = (44, 99)  # (width, height) - matches native digit box size
_DIGIT_PADDING = 15  # Pixels of horizontal padding (left/right) around digit box
_DIGIT_PADDING_V = 10  # Pixels of vertical padding (top/bottom) around digit box
_PANEL_WIDTH = _geometry.panel_size[0]
_PANEL_HEIGHT = _geometry.panel_size[1]

_digit_1_issue = None  # Dict with score_1, score_7, gap when "1" low conf and "7" close

# =============================================================================
# Detection Thresholds
# =============================================================================
_TEMPLATE_CONFIDENCE_THRESHOLD = 0.80
_TEMPLATE_AMBIGUITY_GAP = 0.05          # min gap between 1st and 2nd to be unambiguous
_REJECTION_MIN_SCORE = 0.75             # reject digit if score below this AND gap small
_REJECTION_MAX_GAP = 0.20               # reject digit if gap below this AND score low
_REJECTION_EXTREME_GAP = 0.02           # reject digit if gap below this regardless of score
_AMBIGUOUS_MAX_SCORE = 0.95             # only flag ambiguous if best score below this
_QUICKCHECK_DRIFT = 0.02                # trigger full rescan if score drifts more than this
_MIN_DIGIT_HEIGHT = 10
_MIN_DIGIT_WIDTH = 5

# Panel Detection (from geometry model)
_CORNER_TO_PANEL_X = abs(_geometry.panel_offset[0])
_CORNER_TO_PANEL_Y = abs(_geometry.panel_offset[1])
# Button/LED Detection (from geometry model)
_BUTTON_REGION_RIGHT_RATIO = _geometry.button_region_right_ratio
_BUTTON_REGION_TOP_RATIO = _geometry.button_region_top_ratio
_LED_MIN_AREA = _geometry.led_min_area
_LED_MAX_AREA = _geometry.led_max_area
_LED_MAX_ASPECT_RATIO = _geometry.led_max_aspect_ratio
_WASHOUT_MIN_GAP = 30  # Min brightness gap for dark-hole LED detection

# Digit Recognition

# =============================================================================
# Confusing Digit Resolution
# =============================================================================
# Digits 0, 1, 6, 8, P can confuse template matching due to similar vertical bars.
# This module uses segment analysis to distinguish them.
#
# Segment layout:
#    AAA
#   F   B
#    GGG
#   E   C
#    DDD
#
# Key distinguishing segments:
#   - A (top):          0,6,8,P have it; 1 doesn't
#   - B (top-right):    0,1,8,P have it; 6 doesn't
#   - C (bottom-right): 0,1,6,8 have it; P doesn't
#   - D (bottom):       0,6,8 have it; 1,P don't
#   - G (middle):       6,8,P have it; 0,1 don't

_CONFUSING_DIGITS = {'0', '1', '6', '8', 'P'}

# For each pair, define which segment(s) to check and expected states
# Format: (segment, digit_that_has_it, digit_that_lacks_it)
_DISTINGUISHING_SEGMENTS = {
    frozenset({'6', '8'}): ('B', '8', '6'),  # B lit → 8, B off → 6
    frozenset({'6', 'P'}): ('D', '6', 'P'),  # D lit → 6, D off → P
    frozenset({'0', '8'}): ('G', '8', '0'),  # G lit → 8, G off → 0
    frozenset({'8', 'P'}): ('D', '8', 'P'),  # D lit → 8, D off → P
    frozenset({'0', '6'}): ('G', '6', '0'),  # G lit → 6, G off → 0
    frozenset({'0', 'P'}): ('D', '0', 'P'),  # D lit → 0, D off → P
    # 1 vs others: 1 lacks top (A), bottom (D), middle (G), left bars (E,F)
    frozenset({'0', '1'}): ('A', '0', '1'),  # A lit → 0, A off → 1
    frozenset({'1', '6'}): ('A', '6', '1'),  # A lit → 6, A off → 1
    frozenset({'1', '8'}): ('A', '8', '1'),  # A lit → 8, A off → 1
    frozenset({'1', 'P'}): ('A', 'P', '1'),  # A lit → P, A off → 1
}


def _check_segment_lit(digit_img, segment, match_pos, template_size):
    """Check if a specific segment is lit in the digit image.

    Args:
        digit_img: BGR image of the digit
        segment: Segment name ('A', 'B', 'C', 'D', 'E', 'F', 'G')
        match_pos: (x, y) position of template match
        template_size: (width, height) of matched template

    Returns:
        float: Ratio indicating how "lit" the segment is (0.0 to 1.0)
    """
    match_x, match_y = match_pos
    tmpl_w, tmpl_h = template_size
    img_h, img_w = digit_img.shape[:2]

    # Define segment regions relative to template bounds
    # These are sampling zones at the CENTER of each segment
    # Regions are (x_offset, y_offset, width, height) as ratios
    # IMPORTANT: Avoid overlap with adjacent segments to prevent glow bleeding
    regions = {
        'A': (0.30, 0.02, 0.40, 0.10),   # Top horizontal (narrower)
        'B': (0.75, 0.22, 0.20, 0.20),   # Top-right vertical (moved down, smaller to avoid A glow)
        'C': (0.75, 0.58, 0.20, 0.20),   # Bottom-right vertical (moved up, smaller to avoid D glow)
        'D': (0.30, 0.88, 0.40, 0.10),   # Bottom horizontal (narrower)
        'E': (0.05, 0.58, 0.20, 0.20),   # Bottom-left vertical
        'F': (0.05, 0.22, 0.20, 0.20),   # Top-left vertical
        'G': (0.35, 0.44, 0.30, 0.12),   # Middle horizontal (narrower)
    }

    if segment not in regions:
        return 0.5  # Unknown segment, return neutral

    rx, ry, rw, rh = regions[segment]

    # Calculate absolute coordinates
    x1 = int(match_x + tmpl_w * rx)
    y1 = int(match_y + tmpl_h * ry)
    x2 = min(int(match_x + tmpl_w * (rx + rw)), img_w)
    y2 = min(int(match_y + tmpl_h * (ry + rh)), img_h)

    if x2 <= x1 or y2 <= y1:
        return 0.5  # Invalid region

    # Extract region and analyze
    region = digit_img[y1:y2, x1:x2]
    if region.size == 0:
        return 0.5

    # Use blue channel (digits are blue LEDs)
    blue = region[:, :, 0]

    # Compare to center reference (should be dark if no segment)
    # Use mean intensity normalized by max possible
    intensity = blue.mean() / 255.0

    # Also check for actual blue pixels (saturation > threshold)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    # Blue hue range: 85-130, saturation > 80
    blue_mask = cv2.inRange(hsv, np.array([85, 80, 50]), np.array([130, 255, 255]))
    blue_ratio = np.sum(blue_mask > 0) / blue_mask.size

    # Combine intensity and blue ratio
    # Weight blue_ratio higher as it's more reliable
    return 0.3 * intensity + 0.7 * blue_ratio


def _resolve_confusing_pair(digit_img, digit1, digit2, match_pos, template_size):
    """Resolve confusion between two similar digits using segment analysis.

    Args:
        digit_img: BGR image of the digit
        digit1: First candidate digit
        digit2: Second candidate digit
        match_pos: (x, y) position of template match
        template_size: (width, height) of matched template

    Returns:
        str: The winning digit, or None if can't determine
    """
    pair = frozenset({digit1, digit2})

    if pair not in _DISTINGUISHING_SEGMENTS:
        return None  # Not a known confusing pair

    segment, has_it, lacks_it = _DISTINGUISHING_SEGMENTS[pair]

    lit_ratio = _check_segment_lit(digit_img, segment, match_pos, template_size)

    # Threshold for "lit" determination
    # Use 0.15 as threshold - segment is lit if ratio > 0.15
    if lit_ratio > 0.15:
        return has_it
    else:
        return lacks_it


def resolve_confusing_digits(digit_img, candidates, match_pos, template_size):
    """Resolve confusion between similar-looking digits (0, 6, 8, P).

    Called when template matching returns close scores for confusing digits.
    Uses segment analysis to determine the correct digit.

    Args:
        digit_img: BGR image of the digit
        candidates: List of (digit, score) tuples, sorted by score descending
        match_pos: (x, y) position of best template match
        template_size: (width, height) of matched template

    Returns:
        tuple: (winning_digit, adjusted_score) or (None, None) if not applicable
    """
    if len(candidates) < 2:
        return None, None

    best_digit, best_score = candidates[0]
    second_digit, second_score = candidates[1]

    # Only apply to confusing digit pairs
    if best_digit not in _CONFUSING_DIGITS or second_digit not in _CONFUSING_DIGITS:
        return None, None

    # Resolve the confusion
    winner = _resolve_confusing_pair(digit_img, best_digit, second_digit,
                                      match_pos, template_size)

    if winner is None:
        return None, None

    # Return winner with boosted score (to pass confidence threshold)
    if winner == best_digit:
        return winner, min(best_score * 1.05, 0.99)
    else:
        # Second candidate won - use its score boosted
        return winner, min(second_score * 1.10, 0.99)


# =============================================================================
# Utility Functions
# =============================================================================
def _enhance_dim_digit(digit_img):
    """Enhance dim digit region for better template matching.

    Detects if digit is dim (low brightness) and normalizes the blue channel
    to improve template matching confidence. Also handles uneven background
    by subtracting local minimum.

    Args:
        digit_img: BGR digit image

    Returns:
        (gray_img, was_enhanced): Grayscale image and flag indicating if enhancement was applied
    """
    gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    raw_max = gray.max()  # Check brightness before background subtraction

    # Subtract background (local minimum) to handle uneven lighting
    # Use percentile instead of min to be robust against noise
    background = np.percentile(gray, 5)
    if background > 10:  # Only subtract if significant background brightness
        gray = np.clip(gray.astype(np.int16) - int(background), 0, 255).astype(np.uint8)

    # Check if dim: raw max < 150 (use pre-subtraction value to avoid
    # bright glow-flooded images falsely triggering blue channel enhancement)
    is_dim = raw_max < 150

    if is_dim:
        # Extract blue channel (digits are blue)
        blue = digit_img[:, :, 0]
        # Subtract background from blue channel too
        bg_blue = np.percentile(blue, 5)
        if bg_blue > 10:
            blue = np.clip(blue.astype(np.int16) - int(bg_blue), 0, 255).astype(np.uint8)
        # Normalize to full range for better matching
        enhanced = cv2.normalize(blue, None, 0, 255, cv2.NORM_MINMAX)
        return enhanced, True

    return gray, False


def _is_valid_reading(reading):
    """Check if reading is valid (00-66 or PP).

    Valid readings:
    - PP: special value
    - Numbers 00-66 (as two-digit string)

    Args:
        reading: 2-character string

    Returns:
        True if valid, False otherwise
    """
    if reading == 'PP':
        return True
    if len(reading) != 2:
        return False
    if not reading.isdigit():
        return False
    num = int(reading)
    return 0 <= num <= 66



def reload_templates():
    """Force reload all digit templates from disk."""
    global _digit_templates
    _digit_templates = None
    return _load_digit_templates()


def _load_digit_templates():
    """Load digit templates from templates directory.

    Supports multiple templates per digit with naming: digit_0a.png, digit_0b.png, etc.
    The digit is the character after 'digit_' (0-9 or P), followed by a variant letter.
    Returns dict mapping digit -> list of grayscale templates.
    """
    global _digit_templates
    if _digit_templates is not None:
        return _digit_templates

    _digit_templates = {}
    if not os.path.exists(_TEMPLATES_DIR):
        return _digit_templates

    for f in os.listdir(_TEMPLATES_DIR):
        if f.startswith('digit_') and f.endswith('.png'):
            # Extract digit from filename: digit_0a.png -> "0", digit_Pb.png -> "P"
            name = f.replace('digit_', '').replace('.png', '')
            if len(name) >= 1:
                digit = name[0]  # First character is the digit (0-9 or P)

                template_path = os.path.join(_TEMPLATES_DIR, f)
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    if digit not in _digit_templates:
                        _digit_templates[digit] = []
                    _digit_templates[digit].append(template)

    return _digit_templates


def _extract_digit_with_padding(img, box, padding=None, left_bound=None, right_bound=None):
    """Extract digit region with padding for search tolerance.

    Args:
        img: Source image (corrected panel)
        box: (x, y, w, h) bounding box
        padding: Pixels to add around box (default: _DIGIT_PADDING)
        left_bound: Don't extend left past this x coordinate
        right_bound: Don't extend right past this x coordinate

    Returns:
        Extracted region with padding, clipped to image bounds.
        When bounded by gap, replicates edge pixels to fill target padding.
    """
    if padding is None:
        padding = _DIGIT_PADDING
    padding_v = _DIGIT_PADDING_V

    x, y, w, h = box
    img_h, img_w = img.shape[:2]

    # Add padding, clip to image bounds (horizontal: padding, vertical: padding_v)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding_v)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding_v)

    # Track pixels to replicate at edges
    pad_left = 0
    pad_right = 0

    # Apply additional bounds if specified, track lost pixels
    # Clip at box edge (not at bound) to avoid including gap glow,
    # then replicate the box edge column to fill full padding
    if left_bound is not None and x1 < left_bound:
        pad_left = x - x1
        x1 = x
    if right_bound is not None and x2 > right_bound:
        box_right = x + w
        pad_right = x2 - box_right
        x2 = box_right

    # Extract the region
    region = img[y1:y2, x1:x2]

    # Replicate edge pixels if bounded by gap
    if pad_left > 0 or pad_right > 0:
        if len(region.shape) == 3:
            # Color image
            if pad_left > 0:
                left_col = region[:, 0:1, :]
                left_pad = np.repeat(left_col, pad_left, axis=1)
                region = np.concatenate([left_pad, region], axis=1)
            if pad_right > 0:
                right_col = region[:, -1:, :]
                right_pad = np.repeat(right_col, pad_right, axis=1)
                region = np.concatenate([region, right_pad], axis=1)
        else:
            # Grayscale image
            if pad_left > 0:
                left_col = region[:, 0:1]
                left_pad = np.repeat(left_col, pad_left, axis=1)
                region = np.concatenate([left_pad, region], axis=1)
            if pad_right > 0:
                right_col = region[:, -1:]
                right_pad = np.repeat(right_col, pad_right, axis=1)
                region = np.concatenate([region, right_pad], axis=1)

    return region


def match_single_template(gray_img, digit, template_idx):
    """Match a single specific template and return its score, position, and size.

    Args:
        gray_img: Grayscale image to match against
        digit: The digit character ('0'-'9', 'P')
        template_idx: Index of the specific template to use

    Returns:
        tuple: (score, match_pos, template_size) or (-1.0, None, None) if template not found
            - score: Match confidence (0-1)
            - match_pos: (x, y) position of best match
            - template_size: (width, height) of template
    """
    templates = _load_digit_templates()
    if not templates or digit not in templates:
        return -1.0, None, None

    template_list = templates[digit]
    if template_idx >= len(template_list):
        return -1.0, None, None

    template = template_list[template_idx]
    th, tw = template.shape[:2]

    if gray_img.shape[0] < th or gray_img.shape[1] < tw:
        return -1.0, None, None

    result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_val, max_loc, (tw, th)


def recognize_digit_template(digit_img, return_debug=False):
    """
    Recognize a digit using template matching on grayscale image.

    Converts input to grayscale before matching.
    Matches against all templates for each digit and returns the best match.
    Uses sliding window matching to be less sensitive to exact box position.

    Args:
        digit_img: BGR image of single digit (larger than template for search tolerance)
        return_debug: If True, also return debug info with match position

    Returns:
        digit: Recognized digit character ('0'-'9', 'P', or 'X')
        score: Match confidence (0-1)
        debug_info: (only if return_debug=True) dict with search_size, match_pos, template_size
    """
    templates = _load_digit_templates()
    if not templates:
        if return_debug:
            return 'X', 0.0, None
        return 'X', 0.0

    h, w = digit_img.shape[:2]
    if h < _MIN_DIGIT_HEIGHT or w < _MIN_DIGIT_WIDTH:
        if return_debug:
            return 'X', 0.0, None
        return 'X', 0.0

    # Convert to grayscale with dim enhancement if needed
    gray, _ = _enhance_dim_digit(digit_img)

    # Collect all scores - use sliding window matching
    # Track best template index, position, and size for each digit
    all_scores = []
    digit_1_original_score = None  # Track unpenalized "1" score for potential restoration
    digit_1_penalized = False  # Whether "1" penalty was applied

    for digit, template_list in templates.items():
        best_for_digit = -1.0
        best_idx_for_digit = 0
        best_pos_for_digit = None
        best_size_for_digit = None
        best_original_for_digit = -1.0  # Unpenalized score
        for idx, template in enumerate(template_list):
            th, tw = template.shape[:2]
            # Only match if image is large enough for template
            if gray.shape[0] >= th and gray.shape[1] >= tw:
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                # Track raw score and position for "1" - penalty applied later
                # after we know the 2nd best digit (penalty only if 2nd is 0/6/8/P)
                adjusted_val = max_val

                if adjusted_val > best_for_digit:
                    best_for_digit = adjusted_val
                    best_idx_for_digit = idx
                    best_pos_for_digit = max_loc
                    best_size_for_digit = (tw, th)
                    best_original_for_digit = max_val  # Store unpenalized score

        all_scores.append((digit, best_for_digit, best_idx_for_digit, best_pos_for_digit, best_size_for_digit))
        if digit == '1':
            digit_1_original_score = best_original_for_digit

    # Sort by score descending
    all_scores.sort(key=lambda x: -x[1])

    if not all_scores:
        if return_debug:
            return 'X', 0.0, None
        return 'X', 0.0

    best_digit, best_score, best_template_idx, best_match_pos, best_template_size = all_scores[0]
    second_digit, second_score, second_template_idx = all_scores[1][:3] if len(all_scores) > 1 else ('X', 0.0, 0)

    # Apply "1" penalty only if 2nd best is a "left-bar digit" (0, 6, 8, P)
    # These digits have a left vertical bar that can be mistaken for "1"
    # If 2nd best is something else (like 7), don't penalize - it's likely a real "1"
    digit_width = gray.shape[1]
    left_edge_threshold = digit_width * 0.35
    left_bar_digits = {'0', '6', '8', 'P'}

    if digit_1_original_score is not None and len(all_scores) > 1:
        # Find "1"'s entry
        digit_1_entry = None
        digit_1_idx = None
        for idx, entry in enumerate(all_scores):
            if entry[0] == '1':
                digit_1_entry = entry
                digit_1_idx = idx
                break

        if digit_1_entry is not None:
            digit_1_pos = digit_1_entry[3]  # match_pos
            digit_1_at_left = digit_1_pos is not None and digit_1_pos[0] < left_edge_threshold

            if digit_1_at_left:
                # "1" matched at left edge - check what 2nd best is
                # Get the digit that would be 2nd if "1" were best
                other_digits = [e for e in all_scores if e[0] != '1']
                if other_digits:
                    second_best_digit = other_digits[0][0]

                    if second_best_digit in left_bar_digits:
                        # 2nd best has left bar - apply penalty (might be false positive)
                        digit_1_penalized = True
                        penalized_score = digit_1_original_score * 0.7
                        all_scores[digit_1_idx] = ('1', penalized_score, digit_1_entry[2], digit_1_entry[3], digit_1_entry[4])
                        all_scores.sort(key=lambda x: -x[1])
                        best_digit, best_score, best_template_idx, best_match_pos, best_template_size = all_scores[0]
                        second_digit, second_score, second_template_idx = all_scores[1][:3] if len(all_scores) > 1 else ('X', 0.0, 0)
                        # Penalty was for ranking only — if "1" still won, restore true confidence
                        if best_digit == '1':
                            best_score = digit_1_original_score
                    # else: 2nd best is not a left-bar digit (e.g., 7) - no penalty, keep original score

    # Handle "2" vs "9" confusion with uneven lighting
    # Bright right side can make "9" look like "2" (right side segments appear lit)
    swapped_due_to_lighting = False
    if best_digit == '2' and second_digit == '9' and (best_score - second_score) < _TEMPLATE_AMBIGUITY_GAP:
        # Check for uneven lighting
        digit_width = gray.shape[1]
        left_region = gray[:, :int(digit_width * 0.4)]
        right_region = gray[:, int(digit_width * 0.6):]
        left_mean = left_region.mean()
        right_mean = right_region.mean()
        if left_mean > 5 and (right_mean / max(left_mean, 1)) > 2.0:
            # Severe uneven lighting - swap to "9", find its full info from all_scores
            for item in all_scores:
                if item[0] == '9':
                    best_digit, best_score, best_template_idx, best_match_pos, best_template_size = item
                    # Update second to be the original best ("2")
                    second_digit, second_score = '2', all_scores[0][1]
                    swapped_due_to_lighting = True
                    break

    # Handle "9" vs "5" and "8" vs "6" confusion by checking segment B on grayscale.
    # 9 and 8 have segment B (top-right) lit; 5 and 6 do not.
    # Compare grayscale intensity of segment B vs C: for 5/6 (B off), C is much
    # brighter than B (C-B >= 35). For 8/9 (B on), C and B are similar (C-B < 35).
    # Uses grayscale (not blue channel) because blue saturates from glow, but
    # grayscale preserves contrast since glow is narrow-band blue.
    top_right_lit = {'9', '8'}  # Digits with top-right segment
    top_right_off = {'5', '6'}  # Digits without top-right segment
    is_top_right_confusion = (best_digit in top_right_lit and second_digit in top_right_off) or \
                             (best_digit in top_right_off and second_digit in top_right_lit)
    same_pair = (best_digit in {'9', '5'} and second_digit in {'9', '5'}) or \
                (best_digit in {'8', '6'} and second_digit in {'8', '6'})
    if is_top_right_confusion and same_pair and (best_score - second_score) < 0.07:
        match_x, match_y = best_match_pos
        tmpl_w, tmpl_h = best_template_size
        gh, gw = gray.shape[:2]

        # Segment B (top-right vertical) on grayscale
        bx1 = int(match_x + tmpl_w * 0.75)
        by1 = int(match_y + tmpl_h * 0.22)
        bx2 = min(int(match_x + tmpl_w * 0.95), gw)
        by2 = min(int(match_y + tmpl_h * 0.42), gh)

        # Segment C (bottom-right vertical) on grayscale
        cx1 = int(match_x + tmpl_w * 0.75)
        cy1 = int(match_y + tmpl_h * 0.58)
        cx2 = min(int(match_x + tmpl_w * 0.95), gw)
        cy2 = min(int(match_y + tmpl_h * 0.78), gh)

        if bx2 > bx1 and by2 > by1 and cx2 > cx1 and cy2 > cy1:
            b_gray_val = gray[by1:by2, bx1:bx2].mean()
            c_gray_val = gray[cy1:cy2, cx1:cx2].mean()
            seg_b_diff = c_gray_val - b_gray_val

            lit_digit = '9' if best_digit in {'9', '5'} else '8'
            off_digit = '5' if best_digit in {'9', '5'} else '6'

            if seg_b_diff < 45:
                # B and C similar brightness → B is lit → should be 9 or 8
                if best_digit == lit_digit:
                    best_score = min(best_score * 1.05, 0.99)
                    second_score = second_score * 0.95
                else:
                    for item in all_scores:
                        if item[0] == lit_digit:
                            best_digit, _, best_template_idx, best_match_pos, best_template_size = item
                            best_score = min(item[1] * 1.05, 0.99)
                            second_digit, second_score = off_digit, all_scores[0][1] * 0.95
                            break
            else:
                # C much brighter than B → B is off → should be 5 or 6
                if best_digit == off_digit:
                    best_score = min(best_score * 1.05, 0.99)
                    second_score = second_score * 0.95
                else:
                    for item in all_scores:
                        if item[0] == off_digit:
                            best_digit, _, best_template_idx, best_match_pos, best_template_size = item
                            best_score = min(item[1] * 1.05, 0.99)
                            second_digit, second_score = lit_digit, all_scores[0][1] * 0.95
                            break

    # Special check for 6 vs 8 using grayscale B/C difference (no gap condition).
    # For 8, both B and C are lit: grayscale C-B < 38.
    # For 6, only C is lit, B has only glow: C-B >= 38.
    # Uses grayscale (not blue channel) because blue saturates from LED glow.
    segment_override_6to8 = False
    if best_digit == '6' and best_match_pos is not None and best_template_size is not None:
        _mx, _my = best_match_pos
        _tw, _th = best_template_size
        _gh, _gw = gray.shape[:2]
        _bx1 = int(_mx + _tw * 0.75); _by1 = int(_my + _th * 0.22)
        _bx2 = min(int(_mx + _tw * 0.95), _gw); _by2 = min(int(_my + _th * 0.42), _gh)
        _cx1 = int(_mx + _tw * 0.75); _cy1 = int(_my + _th * 0.58)
        _cx2 = min(int(_mx + _tw * 0.95), _gw); _cy2 = min(int(_my + _th * 0.78), _gh)
        if _bx2 > _bx1 and _by2 > _by1 and _cx2 > _cx1 and _cy2 > _cy1:
            _b_gray = gray[_by1:_by2, _bx1:_bx2].mean()
            _c_gray = gray[_cy1:_cy2, _cx1:_cx2].mean()
            if _c_gray - _b_gray < 38:
                for item in all_scores:
                    if item[0] == '8':
                        old_6_score = best_score
                        best_digit = '8'
                        best_score = max(min(item[1] * 1.10, 0.99), old_6_score)
                        best_template_idx = item[2]
                        best_match_pos = item[3]
                        best_template_size = item[4]
                        second_digit, second_score = '6', old_6_score * 0.95
                        segment_override_6to8 = True
                        break

    # Resolve 0/1/6/8/P confusion using segment analysis when candidates are close
    # These digits share many segments and often have close template scores
    gap = best_score - second_score
    if best_digit in _CONFUSING_DIGITS and second_digit in _CONFUSING_DIGITS and gap < _TEMPLATE_AMBIGUITY_GAP:
        candidates = [(best_digit, best_score), (second_digit, second_score)]
        resolved_digit, resolved_score = resolve_confusing_digits(
            digit_img, candidates, best_match_pos, best_template_size
        )
        if resolved_digit is not None:
            if resolved_digit != best_digit:
                # Segment analysis chose the second candidate - update results
                for item in all_scores:
                    if item[0] == resolved_digit:
                        best_digit = resolved_digit
                        best_score = resolved_score
                        best_template_idx = item[2]
                        best_match_pos = item[3]
                        best_template_size = item[4]
                        second_digit, second_score = candidates[0]  # Original best is now second
                        break
            else:
                # Segment analysis confirmed the best candidate - boost confidence
                best_score = resolved_score
            # Recalculate gap after resolution
            gap = best_score - second_score

    # Reject ambiguous readings: low confidence + close second candidate = transitional frame
    # Also reject if gap is extremely small (top two nearly identical) regardless of score
    # Skip rejection if we intentionally swapped due to uneven lighting
    if not swapped_due_to_lighting and ((best_score < _REJECTION_MIN_SCORE and gap < _REJECTION_MAX_GAP) or gap < _REJECTION_EXTREME_GAP):
        if return_debug:
            debug_info = {
                'search_size': (w, h),
                'match_pos': best_match_pos,
                'template_size': best_template_size,
                'second_digit': second_digit,
                'second_score': second_score,
                'best_template_idx': best_template_idx,
                'second_template_idx': second_template_idx,
                'rejected': f'low_conf_{best_score:.2f}_gap_{gap:.2f}',
                'rejected_digit': best_digit,  # Store actual detected digit for display
            }
            return 'X', best_score, debug_info
        return 'X', best_score

    # Flag issue when "1" has low confidence and "7" is close (penalty issue)
    # Skip penalty if both candidates detected near left edge (thin "1" naturally at left)
    # This will be checked in live_demo.py for logging with full frame
    global _digit_1_issue
    _digit_1_issue = None
    if best_digit == '1' and best_score < 0.85 and second_digit == '7':
        # Check if both matches are near left edge - if so, it's likely a valid "1"
        second_match_pos = all_scores[1][3] if len(all_scores) > 1 and len(all_scores[1]) > 3 else None
        left_edge_threshold = w * 0.25  # Within 25% of left edge
        both_near_left = (best_match_pos[0] < left_edge_threshold and
                          second_match_pos is not None and second_match_pos[0] < left_edge_threshold)
        if not both_near_left:
            gap = best_score - second_score
            _digit_1_issue = {'score_1': best_score, 'score_7': second_score, 'gap': gap}

    if return_debug:
        debug_info = {
            'search_size': (w, h),
            'match_pos': best_match_pos,
            'template_size': best_template_size,
            'second_digit': second_digit,
            'second_score': second_score,
            'best_template_idx': best_template_idx,
            'second_template_idx': second_template_idx,
        }
        return best_digit, best_score, debug_info
    return best_digit, best_score


def _load_cache():
    """Load button zone cache from disk if exists."""
    global _button_zone_cache
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, 'r') as f:
                data = json.load(f)
                if 'button_zones' in data:
                    _button_zone_cache = [(z['left'], z['right'], z['top'], z['bottom'], z['name'])
                                          for z in data['button_zones']]
        except (json.JSONDecodeError, KeyError, IOError):
            _button_zone_cache = None


def _save_cache(panel_data=None):
    """Save unified cache to disk.

    Args:
        panel_data: If provided, write this as the 'panel' section.
                    If None, preserve existing panel section from disk.
    """
    try:
        data = {}
        # When no panel_data given, preserve existing panel section from disk
        if panel_data is None:
            if os.path.exists(_CACHE_FILE):
                try:
                    with open(_CACHE_FILE, 'r') as f:
                        existing = json.load(f)
                    if 'panel' in existing:
                        data['panel'] = existing['panel']
                except (json.JSONDecodeError, IOError):
                    pass
        else:
            data['panel'] = panel_data
        if _button_zone_cache is not None:
            data['button_zones'] = [{'left': left, 'right': right, 'top': top, 'bottom': bottom, 'name': name}
                                    for left, right, top, bottom, name in _button_zone_cache]
        with open(_CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except IOError:
        pass


def _to_native(val):
    """Convert numpy types to Python native types for JSON serialization."""
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return [_to_native(v) for v in val]
    if hasattr(val, 'item'):  # numpy scalar
        return val.item()
    return val


def _load_panel_from_cache():
    """Read panel dict from disk cache file. Returns None if missing/corrupt."""
    if not os.path.exists(_CACHE_FILE):
        return None
    try:
        with open(_CACHE_FILE, 'r') as f:
            data = json.load(f)
        return data.get('panel')
    except (json.JSONDecodeError, IOError):
        return None


def _load_corner_templates():
    """Load corner templates for pattern matching."""
    global _corner_templates
    if _corner_templates is None:
        _corner_templates = []
        for path in _CORNER_TEMPLATE_FILES:
            if os.path.exists(path):
                tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if tmpl is not None:
                    _corner_templates.append(tmpl)
    return _corner_templates


def _find_corner(frame, min_match=0.93, return_debug=False):
    """
    Find the corner in the frame using template matching.

    Optimized: Uses bottom-right 1/4 of template and searches only right portion of frame.
    Uses round-robin with sticky preference - try current template first, switch only if it fails.

    Args:
        frame: BGR image
        min_match: Minimum match score (0-1) to consider valid
        return_debug: If True, return debug info for visualization

    Returns:
        (x, y, score): Corner center coordinates and match score, or None if not found
        If return_debug=True, also returns: search_rect, match_rect, template_crop_size
    """
    global _corner_template_idx
    templates = _load_corner_templates()
    if not templates:
        return (None, None) if return_debug else None

    # Search only in small square region with matched pattern centered
    h_frame, w_frame = frame.shape[:2]
    # search origin is in raw (camera) domain
    search_left_raw, search_top_raw, search_size = _geometry.get_corner_search_region(w_frame, h_frame)
    # undistort_roi produces locally-undistorted image; templates match in this space
    search_roi = _geometry.undistort_roi(
        frame, search_left_raw, search_top_raw, search_size, search_size, derotate=False)
    search_region = search_roi[:, :, 1] if search_roi.ndim == 3 else search_roi  # Green channel only

    # Compute undistorted origin for accurate coordinate conversion later.
    # undistort_roi indexes the map at (search_left_raw, search_top_raw), so
    # ROI pixel (px, py) ≈ undistorted frame position (search_left_ud + px, search_top_ud + py).
    if _geometry.has_intrinsics():
        _origin_ud = _geometry.undistort_points(
            np.array([[float(search_left_raw), float(search_top_raw)]]))
        search_left_ud = float(_origin_ud[0, 0])
        search_top_ud = float(_origin_ud[0, 1])
    else:
        search_left_ud = float(search_left_raw)
        search_top_ud = float(search_top_raw)

    # Skip if search region is too dark or overexposed — template matching is noise
    roi_mean = search_region.mean()
    if roi_mean < 15 or roi_mean > 240:
        if return_debug:
            return (None, None, 0.0), ((search_left_raw, search_top_raw, search_size, search_size), None, (0, 0))
        return None

    # Search region rect for debug visualization (raw domain, drawn on raw frame)
    search_rect_raw = (search_left_raw, search_top_raw, search_size, search_size)

    def try_template(idx):
        """Try matching a single template, return (score, loc) or None.

        Templates are 75x75 bottom-right crops from 150x150 originals.
        Corner feature is at template (0,0), so match position = corner position.
        """
        if idx >= len(templates):
            return None
        template = templates[idx]
        th, tw = template.shape[:2]
        if th > search_size or tw > search_size:
            return None
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        return (max_val, max_loc, (tw, th))

    # Round-robin with sticky preference: try current template first
    best_score = 0
    best_loc = None
    best_crop_size = None
    best_tmpl_idx = _corner_template_idx

    # Try preferred template first
    result = try_template(_corner_template_idx)
    if result and result[0] >= min_match:
        best_score, best_loc, best_crop_size = result
    else:
        # Preferred failed, try others
        if result:
            best_score, best_loc, best_crop_size = result
        for i in range(len(templates)):
            if i == _corner_template_idx:
                continue
            result = try_template(i)
            if result and result[0] >= min_match:
                # Found a working template, switch to it
                _corner_template_idx = i
                best_tmpl_idx = i
                best_score, best_loc, best_crop_size = result
                break
            elif result and result[0] > best_score:
                best_tmpl_idx = i
                best_score, best_loc, best_crop_size = result

    # Retry with larger search region if normal search failed
    if best_score < min_match:
        expanded_size = int(search_size * 1.6)
        # Center expanded region on same midpoint (raw domain)
        mid_x_raw = search_left_raw + search_size // 2
        mid_y_raw = search_top_raw + search_size // 2
        exp_left_raw = max(0, min(w_frame - expanded_size, mid_x_raw - expanded_size // 2))
        exp_top_raw = max(0, min(h_frame - expanded_size, mid_y_raw - expanded_size // 2))
        exp_left_raw, exp_top_raw, expanded_size = int(exp_left_raw), int(exp_top_raw), int(expanded_size)

        exp_roi = _geometry.undistort_roi(
            frame, exp_left_raw, exp_top_raw, expanded_size, expanded_size, derotate=False)
        exp_region = exp_roi[:, :, 1] if exp_roi.ndim == 3 else exp_roi

        exp_mean = exp_region.mean()
        if 15 <= exp_mean <= 240:
            # Compute undistorted origin for expanded region
            if _geometry.has_intrinsics():
                _exp_ud = _geometry.undistort_points(
                    np.array([[float(exp_left_raw), float(exp_top_raw)]]))
                exp_left_ud = float(_exp_ud[0, 0])
                exp_top_ud = float(_exp_ud[0, 1])
            else:
                exp_left_ud = float(exp_left_raw)
                exp_top_ud = float(exp_top_raw)

            for i in range(len(templates)):
                template = templates[i]
                th, tw = template.shape[:2]
                if th > expanded_size or tw > expanded_size:
                    continue
                result = cv2.matchTemplate(exp_region, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                    best_loc = max_loc
                    best_crop_size = (tw, th)
                    # Switch to expanded region origins
                    search_left_raw, search_top_raw = exp_left_raw, exp_top_raw
                    search_left_ud, search_top_ud = exp_left_ud, exp_top_ud
                    search_size = expanded_size
                    search_rect_raw = (exp_left_raw, exp_top_raw, expanded_size, expanded_size)
                    _corner_template_idx = i
                    best_tmpl_idx = i
                    if max_val >= min_match:
                        break

    if best_score < min_match:
        if return_debug:
            # Always return score for logging, even when below threshold
            return (None, None, best_score, best_tmpl_idx), (search_rect_raw, None, best_crop_size or (0, 0))
        return None

    # best_loc is the match position within the undistorted ROI.
    # Templates are 75x75 bottom-right crops from 150x150 originals,
    # so corner feature is at template (0,0). Match position = corner position.
    corner_x_ud = search_left_ud + best_loc[0]
    corner_y_ud = search_top_ud + best_loc[1]

    # Convert from undistorted domain back to raw (camera) domain
    if _geometry.has_intrinsics():
        raw_pts = _geometry.redistort_points(
            np.array([[corner_x_ud, corner_y_ud]]))
        corner_x_raw = int(round(raw_pts[0, 0]))
        corner_y_raw = int(round(raw_pts[0, 1]))
    else:
        corner_x_raw = int(corner_x_ud)
        corner_y_raw = int(corner_y_ud)

    # Update geometry with detected corner (raw domain)
    _geometry.set_corner(corner_x_raw, corner_y_raw)

    # Match rect in raw frame coordinates (full template area)
    th, tw = templates[best_tmpl_idx].shape[:2]
    match_rect_raw = (corner_x_raw, corner_y_raw, tw, th)

    if return_debug:
        return (corner_x_raw, corner_y_raw, best_score, best_tmpl_idx), (search_rect_raw, match_rect_raw, best_crop_size)
    return (corner_x_raw, corner_y_raw, best_score)


def draw_corner_debug(frame, debug_info, corner_score=None):
    """Draw corner search area and match location on frame.

    Args:
        frame: BGR image to draw on (modified in place)
        debug_info: (search_rect, match_rect, template_size) from _find_corner
        corner_score: match score (0-1) to display (optional)
    """
    search_rect, match_rect, template_size = debug_info

    # Draw search area (blue dashed rectangle)
    sx, sy, sw, sh = search_rect
    cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (255, 100, 0), 1)
    cv2.putText(frame, "search", (sx + 2, sy + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

    # Draw match location (yellow rectangle)
    if match_rect is not None:
        mx, my, mw, mh = match_rect
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 255, 255), 2)
        label = f"corner {corner_score:.3f}" if corner_score is not None else "corner"
        cv2.putText(frame, label, (mx + 2, my - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


def _zones_changed_significantly(old_zones, new_zones):
    """Check if zone centers have changed by more than threshold."""
    if old_zones is None or len(old_zones) != len(new_zones):
        return True

    # Compare zone centers (midpoint of left/right and top/bottom)
    old_dict = {name: ((left + right) / 2, (top + bottom) / 2)
                for left, right, top, bottom, name in old_zones}
    for left, right, top, bottom, name in new_zones:
        if name not in old_dict:
            return True
        old_cx, old_cy = old_dict[name]
        new_cx = (left + right) / 2
        new_cy = (top + bottom) / 2
        if abs(new_cx - old_cx) > _ZONE_CHANGE_THRESHOLD or abs(new_cy - old_cy) > _ZONE_CHANGE_THRESHOLD:
            return True
    return False


def clear_cache():
    """Clear all cached data (memory and disk)."""
    global _button_zone_cache, _cached_buttons, _frame_led_dots, _led_diff_snapshots, _led_diff_zones, _led_diff_lit
    _button_zone_cache = None
    _cached_buttons = None
    _frame_led_dots = None
    _led_diff_snapshots = None
    _led_diff_zones = None
    _led_diff_lit = None
    if os.path.exists(_CACHE_FILE):
        try:
            os.remove(_CACHE_FILE)
        except IOError:
            pass


# Load cache from disk on module import
_load_cache()


# =============================================================================
# Logging functions for cache threshold analysis
# =============================================================================

_CSV_HEADER = ('timestamp,panel_x,panel_y,panel_w,panel_h,gap_x,'
               'left_score,right_score,reading,led_status,'
               'corner_score,corner_tmpl,detection_method,'
               'mute_status,dim_enhanced,frame_skip,diff_edge,diff_mode,'
               'led_method,proc_ms,issue,'
               'geo_method,geo_scale,geo_rotation,undistort_px,'
               'noise_mean,'
               'mute_rr,mute_re,mute_gr,mute_led_r,mute_ref_r,'
               'mute_h_age')


def _init_log():
    """Initialize CSV log file with headers.

    If an existing detection.csv has a different header, archive it
    with a timestamped name and start fresh.
    """
    global _log_file
    if not _LOG_ENABLED or _log_file is not None:
        return

    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        log_path = os.path.join(_LOG_DIR, 'detection.csv')

        # Check existing CSV header
        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            with open(log_path, 'r') as f:
                existing_header = f.readline().rstrip('\n')
            if existing_header != _CSV_HEADER:
                # Archive with timestamp
                from datetime import datetime
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                archive_path = os.path.join(_LOG_DIR, f'detection_archived_{ts}.csv')
                os.rename(log_path, archive_path)
                print(f"CSV header changed, archived old CSV to {archive_path}", flush=True)

        write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0

        _log_file = open(log_path, 'a')
        # Register cleanup immediately after opening to prevent leaks
        import atexit
        atexit.register(close_log)

        if write_header:
            _log_file.write(_CSV_HEADER + '\n')
            _log_file.flush()
    except (IOError, OSError) as e:
        print(f"Warning: Failed to initialize log: {e}", flush=True)
        if _log_file is not None:
            _log_file.close()
        _log_file = None


def log_detection(panel_rect=None, gap_x=None, left_score=0, right_score=0,
                  reading=None, led_status=None, corner_score=0, corner_tmpl=None,
                  detection_method=None, mute_status=None,
                  dim_enhanced=None, frame_skip=False, diff_edge=None,
                  diff_mode=None, led_method=None, proc_ms=None, issue=None,
                  geo_method=None, geo_scale=None, geo_rotation=None,
                  undistorted=None, noise_mean=None,
                  mute_rr=None, mute_re=None, mute_gr=None,
                  mute_led_r=None, mute_ref_r=None,
                  mute_h_age=None):
    """Log detection indicators to CSV."""
    if not _LOG_ENABLED:
        return

    _init_log()
    if _log_file is None:
        return

    ts = time.strftime('%Y-%m-%d %H:%M:%S')

    px, py, pw, ph = panel_rect if panel_rect is not None else (0, 0, 0, 0)
    gx = gap_x if gap_x is not None else 0
    rd = reading if reading is not None else ''
    led = led_status if led_status is not None else ''
    method = str(detection_method) if detection_method is not None else ''
    mute = mute_status if mute_status is not None else ''
    dim_enh = dim_enhanced if dim_enhanced is not None else ''
    skip = '1' if frame_skip else ''
    diff_e = str(int(diff_edge)) if diff_edge is not None else ''
    d_mode = diff_mode if diff_mode is not None else ''
    led_m = led_method if led_method is not None else ''
    proc = f'{proc_ms:.1f}' if proc_ms is not None else ''
    iss = issue if issue is not None else ''
    geo_m = geo_method if geo_method is not None else ''
    geo_s = f'{geo_scale:.3f}' if geo_scale is not None else ''
    geo_r = f'{geo_rotation:.2f}' if geo_rotation is not None else ''
    undist = f'{undistorted:.1f}' if undistorted else '0'
    n_mean = f'{noise_mean:.1f}' if noise_mean is not None else ''
    m_rr = f'{mute_rr:.2f}' if mute_rr is not None else ''
    m_re = f'{mute_re:.1f}' if mute_re is not None else ''
    m_gr = f'{mute_gr:.2f}' if mute_gr is not None else ''
    m_led_r = f'{mute_led_r:.1f}' if mute_led_r is not None else ''
    m_ref_r = f'{mute_ref_r:.1f}' if mute_ref_r is not None else ''
    m_h_age = str(int(mute_h_age)) if mute_h_age is not None else ''

    c_tmpl = str(int(corner_tmpl)) if corner_tmpl is not None else ''

    _log_file.write(f'{ts},{px},{py},{pw},{ph},{gx},'
                   f'{left_score:.3f},{right_score:.3f},{rd},{led},'
                   f'{corner_score:.3f},{c_tmpl},{method},'
                   f'{mute},{dim_enh},{skip},{diff_e},{d_mode},'
                   f'{led_m},{proc},{iss},'
                   f'{geo_m},{geo_s},{geo_r},{undist},'
                   f'{n_mean},'
                   f'{m_rr},{m_re},{m_gr},{m_led_r},{m_ref_r},'
                   f'{m_h_age}\n')
    _log_file.flush()


def get_digit_1_issue():
    """Get and clear digit '1' low confidence issue (when '7' is close).

    Returns dict with score_1, score_7, gap or None if no issue.
    """
    global _digit_1_issue
    issue = _digit_1_issue
    _digit_1_issue = None
    return issue


def log_issue_frame(frame, issue_type, confidence=0, extra_info=None, display_frame=None, debug_info=None):
    """Save frame when detection issue occurs (with cooldown).

    Args:
        frame: Raw camera frame
        issue_type: Type of issue (e.g., 'low_conf', 'ambiguous')
        confidence: Confidence score
        extra_info: Additional info for filename
        display_frame: Optional display window frame with overlays
        debug_info: Optional dict with debug/overlay info to write to text file
    """
    if not _LOG_ENABLED or frame is None:
        return None

    now = time.time()

    # Check cooldown
    last = _log_last_save.get(issue_type, 0)
    if now - last < _LOG_COOLDOWN:
        return None
    _log_last_save[issue_type] = now

    # Create filename with timestamp and info
    os.makedirs(_LOG_DIR, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    conf_str = f'_{confidence:.2f}' if confidence else ''
    extra_str = f'_{extra_info}' if extra_info else ''
    base_name = f'{ts}_{issue_type}{conf_str}{extra_str}'

    # Save frame(s)
    filepath = os.path.join(_LOG_DIR, f'{base_name}.png')

    if display_frame is not None:
        # Stitch raw and display frames side by side (raw left, display right)
        combined = np.hstack([frame, display_frame])
        if not cv2.imwrite(filepath, combined):
            print(f"Warning: Failed to write frame {filepath}", flush=True)
            return None
    else:
        # Save raw frame only
        if not cv2.imwrite(filepath, frame):
            print(f"Warning: Failed to write frame {filepath}", flush=True)
            return None

    # Save debug info to text file if provided
    if debug_info:
        txt_path = os.path.join(_LOG_DIR, f'{base_name}.txt')
        try:
            with open(txt_path, 'w') as f:
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Issue: {issue_type}\n")
                if confidence:
                    f.write(f"Confidence: {confidence:.3f}\n")
                if extra_info:
                    f.write(f"Extra: {extra_info}\n")
                f.write("\n")
                for key, value in debug_info.items():
                    f.write(f"{key}: {value}\n")
        except (IOError, OSError) as e:
            print(f"Warning: Failed to write debug info {txt_path}: {e}", flush=True)

    # Cleanup old frames if too many
    _cleanup_old_frames()

    return filepath


def _cleanup_old_frames():
    """Remove oldest frames if exceeding max count."""
    try:
        frames = sorted([f for f in os.listdir(_LOG_DIR) if f.endswith('.png')])
        if len(frames) > _LOG_MAX_FRAMES:
            for f in frames[:-_LOG_MAX_FRAMES]:
                try:
                    os.remove(os.path.join(_LOG_DIR, f))
                except (IOError, OSError) as e:
                    print(f"Warning: Failed to cleanup {f}: {e}", flush=True)
    except (IOError, OSError) as e:
        print(f"Warning: Frame cleanup failed: {e}", flush=True)


def close_log():
    """Close log file."""
    global _log_file
    if _log_file:
        _log_file.close()
        _log_file = None


def predict_panel_from_landmarks(frame):
    """
    Predict panel location using corner template and button detection.

    Uses the corner position and detected buttons (B2, S1, S2) to calculate
    the panel position based on known spatial relationships.

    Args:
        frame: BGR image from camera/file

    Returns:
        panel_rect: (x, y, w, h) of predicted panel, or None if landmarks not found
    """
    global _cached_buttons, _frame_led_dots, _frame_corner_result, _frame_corner_debug
    _cached_buttons = None  # Clear stale cache from previous frame
    _frame_led_dots = None

    h_frame, w_frame = frame.shape[:2]

    # Step 1: Find corner (green channel matching, 0.93 threshold)
    # Always get debug info for caching (consumed by detect())
    corner_raw = _find_corner(frame, min_match=0.93, return_debug=True)
    # corner_raw is (result_tuple, debug_tuple) or (None, None) if no templates
    corner_result_full = corner_raw[0] if corner_raw else None
    corner_debug_info = corner_raw[1] if corner_raw else None
    _frame_corner_result = corner_result_full
    _frame_corner_debug = corner_debug_info

    # Check if corner was actually found (x is not None in result tuple)
    if corner_result_full is None or corner_result_full[0] is None:
        return None

    corner_x, corner_y = corner_result_full[0], corner_result_full[1]
    corner_score = corner_result_full[2]

    # Step 2: Define button search region — use same region as detect_button_leds()
    # so both functions feed identical crops to _detect_buttons() (#74)
    _geometry.set_corner(corner_x, corner_y)
    geo_region = _geometry.get_button_region_from_geometry(w_frame, h_frame)
    if geo_region is None:
        return None
    btn_search_top, btn_search_bottom, btn_search_left, btn_search_right = geo_region

    button_region = frame[btn_search_top:btn_search_bottom, btn_search_left:btn_search_right]
    if button_region.shape[0] < 10 or button_region.shape[1] < 10:
        return None

    # Step 3: Detect buttons in the region
    buttons = _detect_buttons(button_region)
    buttons = sorted(buttons, key=lambda b: b[0])  # Sort left to right

    # Cache for reuse by detect_button_leds() (#74)
    _cached_buttons = ((btn_search_top, btn_search_bottom, btn_search_left, btn_search_right), buttons)

    # Require at least 1 button for landmark-based detection
    if len(buttons) < 1:
        return None

    # Stage 2: Find LED dot in each button for precise landmark
    led_centers = {}  # LED positions for homography
    led_dot_found = {}

    if len(buttons) >= 3:
        names = ['B2', 'S1', 'S2']
        target_buttons = buttons[-3:]
    elif len(buttons) == 2:
        names = ['S1', 'S2']
        target_buttons = buttons[-2:]
    else:
        names = ['S2']
        target_buttons = [buttons[-1]]

    led_methods = {}  # name -> 'dark' or 'lit' (which detector found the dot)
    for name, btn in zip(names, target_buttons):
        x, y, w, h = btn
        btn_cx = x + w // 2
        btn_cy = y + h // 2
        led_result = _find_led_in_button(button_region, btn)
        if led_result is not None:
            lx, ly, method = led_result
            in_right_half = lx >= btn_cx
            vert_ok = abs(ly - btn_cy) < h * 0.6
            if in_right_half and vert_ok:
                led_centers[name] = (btn_search_left + lx,
                                     btn_search_top + ly)
                led_dot_found[name] = True
                led_methods[name] = method
                continue
        # Dot not found or sanity check failed — use button center as fallback
        led_dot_found[name] = False
        led_centers[name] = (btn_search_left + btn_cx,
                             btn_search_top + btn_cy)
        led_methods[name] = 'center'

    if not led_centers:
        return None

    # Determine which LED is lit from detection methods:
    # - 'lit' method (blue blob found, no dark dot) = this LED is lit
    # - 'dark' method (dark dot found) = this LED is unlit
    # If exactly one button was found via 'lit', that's the lit LED.
    # If no 'lit' found but some buttons had no dot at all, use brightness
    # comparison among dark-dot positions as fallback.
    lit_buttons = [name for name, m in led_methods.items() if m == 'lit']
    # Lit LED decision is deferred until after B1 is found (step 5 below)
    lit_led_name = None
    if len(lit_buttons) == 1:
        lit_led_name = lit_buttons[0]

    # Cache LED positions for debug overlay and detect_button_leds fast path
    _frame_led_dots = {name: (led_centers[name], led_dot_found[name])
                        for name in led_centers}

    # Step 4: Initial homography from detected LED positions
    if not _geometry.compute_homography((corner_x, corner_y), led_centers):
        return None

    # Step 5: Search for missing button LEDs at projected positions
    # With initial homography, we can project where undetected buttons should be
    all_button_names = ['B1', 'B2', 'S1', 'S2']
    missing_names = [n for n in all_button_names if n not in led_centers]
    avg_w = sum(b[2] for b in target_buttons) // len(target_buttons)
    avg_h = sum(b[3] for b in target_buttons) // len(target_buttons)
    new_landmarks = {}

    # LED fraction: LED is at ~78% across button width (measured from B2/S1/S2)
    LED_FRAC_X = 0.78

    for name in missing_names:
        proj = _geometry.project_landmark(name)
        if proj is None:
            continue
        px, py = int(proj[0]), int(proj[1])
        # Convert projected LED position to button_region coords
        bx = px - btn_search_left
        by = py - btn_search_top
        # Compute full button rect from projected LED position
        # project_landmark returns LED position, which is at ~78% of button width
        btn_x = int(bx - LED_FRAC_X * avg_w)
        btn_y = by - avg_h // 2
        button_rect = (btn_x, btn_y, avg_w, avg_h)
        # Check button rect is within button_region bounds
        brh, brw = button_region.shape[:2]
        if (btn_x + avg_w < 0 or btn_x >= brw or
                btn_y + avg_h < 0 or btn_y >= brh):
            _frame_led_dots[name] = ((px, py), 'predicted')
            continue
        # Same detection path as B2/S1/S2
        btn_cx = btn_x + avg_w // 2
        btn_cy = btn_y + avg_h // 2
        led_result = _find_led_in_button(button_region, button_rect)
        if led_result is not None:
            lx, ly, method = led_result
            in_right_half = lx >= btn_cx
            vert_ok = abs(ly - btn_cy) < avg_h * 0.6
            if in_right_half and vert_ok:
                frame_pos = (btn_search_left + lx, btn_search_top + ly)
                _frame_led_dots[name] = (frame_pos, True)
                led_methods[name] = method
                if method == 'lit' and lit_led_name is None:
                    lit_led_name = name
                # Add to landmarks for homography refit (except B1 — too close to frame edge)
                if name != 'B1':
                    new_landmarks[name] = frame_pos
                continue
        _frame_led_dots[name] = ((px, py), 'predicted')

    # Step 6: Recompute homography if new landmarks found
    if new_landmarks:
        led_centers.update(new_landmarks)
        _geometry.compute_homography((corner_x, corner_y), led_centers)

    # Deferred lit LED decision: when no single 'lit' winner (0 or 2+ lit),
    # compare brightness at all LED positions (including B1 projected)
    if lit_led_name is None:
        all_dots = {}
        for name, (pos, found) in _frame_led_dots.items():
            if name.startswith('_') or name.endswith('_proj'):
                continue
            # Include dark dots and projected B1 position
            if found == 'predicted':
                all_dots[name] = pos  # B1 projected — still valid for brightness
            elif found is True and led_methods.get(name) == 'dark':
                all_dots[name] = pos
        if len(all_dots) >= 2:
            blue = frame[:, :, 0]
            h_frame, w_frame = frame.shape[:2]
            dot_brightness = []
            for name, (dx, dy) in all_dots.items():
                ix, iy = int(dx), int(dy)
                y1, y2 = max(0, iy - 2), min(h_frame, iy + 3)
                x1, x2 = max(0, ix - 2), min(w_frame, ix + 3)
                patch = blue[y1:y2, x1:x2]
                val = int(np.max(patch)) if patch.size > 0 else 0
                dot_brightness.append((val, name))
            dot_brightness.sort(key=lambda x: -x[0])
            best_val, best_name = dot_brightness[0]
            second_val = dot_brightness[1][0]
            if best_val > second_val + 15:
                lit_led_name = best_name

    if lit_led_name:
        _frame_led_dots['_lit'] = lit_led_name

    # Always show B1 homography projection as yellow arrow for comparison
    b1_proj = _geometry.project_landmark('B1')
    if b1_proj is not None:
        _frame_led_dots['B1_proj'] = ((int(b1_proj[0]), int(b1_proj[1])), 'predicted')

    # Show mute LED projection as yellow arrow
    mute_proj = _geometry.project_landmark('mute_led')
    if mute_proj is not None:
        _frame_led_dots['mute_proj'] = ((int(mute_proj[0]), int(mute_proj[1])), 'predicted')

    # Project panel from the initial homography (corner + B2/S1/S2).
    # B1 is at the frame edge and too imprecise to include in the fit.
    panel_rect = _geometry.get_panel_rect()
    if panel_rect is not None:
        px, py, pw, ph = panel_rect
        px = max(0, px)
        py = max(0, py)
        pw = min(w_frame - px, pw)
        ph = min(h_frame - py, ph)
        if pw >= 50 and ph >= 30:
            _geometry._geo_method = 'homography'
            if _TRACKING:
                _geometry.update_golden((corner_x, corner_y), led_centers)
            return (px, py, pw, ph)

    return None


def _refresh_led_dots(frame):
    """Recompute _frame_led_dots on the current frame using cached button positions.

    Called on frame-skipped frames where predict_panel_from_landmarks() didn't run
    but LED detection still needs fresh dot data.
    """
    global _frame_led_dots

    if _cached_buttons is None:
        _frame_led_dots = None
        return

    (btn_search_top, btn_search_bottom, btn_search_left, btn_search_right), buttons = _cached_buttons
    if len(buttons) < 1:
        _frame_led_dots = None
        return

    h_frame, w_frame = frame.shape[:2]
    button_region = frame[btn_search_top:btn_search_bottom, btn_search_left:btn_search_right]
    if button_region.shape[0] < 10 or button_region.shape[1] < 10:
        _frame_led_dots = None
        return

    # Assign names to buttons (same logic as predict_panel_from_landmarks)
    if len(buttons) >= 3:
        names = ['B2', 'S1', 'S2']
        target_buttons = buttons[-3:]
    elif len(buttons) == 2:
        names = ['S1', 'S2']
        target_buttons = buttons[-2:]
    else:
        names = ['S2']
        target_buttons = [buttons[-1]]

    led_methods = {}
    led_dots = {}
    for name, btn in zip(names, target_buttons):
        x, y, w, h = btn
        btn_cx = x + w // 2
        btn_cy = y + h // 2
        led_result = _find_led_in_button(button_region, btn)
        if led_result is not None:
            lx, ly, method = led_result
            in_right_half = lx >= btn_cx
            vert_ok = abs(ly - btn_cy) < h * 0.6
            if in_right_half and vert_ok:
                led_dots[name] = ((btn_search_left + lx, btn_search_top + ly), True)
                led_methods[name] = method
                continue
        led_dots[name] = ((btn_search_left + btn_cx, btn_search_top + btn_cy), False)
        led_methods[name] = 'center'

    # Detect B1 at projected position (same as predict_panel_from_landmarks step 5)
    # Compute full button rect from projected LED position (LED is at ~78% of button width)
    LED_FRAC_X = 0.78
    b1_proj = _geometry.project_landmark('B1')
    if b1_proj is not None:
        px, py = int(b1_proj[0]), int(b1_proj[1])
        bx = px - btn_search_left
        by = py - btn_search_top
        avg_w = sum(b[2] for b in target_buttons) // len(target_buttons)
        avg_h = sum(b[3] for b in target_buttons) // len(target_buttons)
        btn_x = int(bx - LED_FRAC_X * avg_w)
        btn_y = by - avg_h // 2
        button_rect = (btn_x, btn_y, avg_w, avg_h)
        brh, brw = button_region.shape[:2]
        if (btn_x + avg_w >= 0 and btn_x < brw and
                btn_y + avg_h >= 0 and btn_y < brh):
            btn_cx = btn_x + avg_w // 2
            btn_cy = btn_y + avg_h // 2
            led_result = _find_led_in_button(button_region, button_rect)
            if led_result is not None:
                lx, ly, method = led_result
                in_right_half = lx >= btn_cx
                vert_ok = abs(ly - btn_cy) < avg_h * 0.6
                if in_right_half and vert_ok:
                    led_dots['B1'] = ((btn_search_left + lx, btn_search_top + ly), True)
                    led_methods['B1'] = method
                else:
                    led_dots['B1'] = ((px, py), 'predicted')
            else:
                led_dots['B1'] = ((px, py), 'predicted')
        else:
            led_dots['B1'] = ((px, py), 'predicted')

    # Determine lit LED
    lit_buttons = [name for name, m in led_methods.items() if m == 'lit']
    lit_led_name = lit_buttons[0] if len(lit_buttons) == 1 else None

    # Brightness fallback — when no single 'lit' winner (0 or 2+ lit buttons)
    if lit_led_name is None:
        all_dots = {}
        for name, (pos, found) in led_dots.items():
            if name.startswith('_') or name.endswith('_proj'):
                continue
            if found == 'predicted':
                all_dots[name] = pos  # B1 projected
            elif found is True and led_methods.get(name) == 'dark':
                all_dots[name] = pos
        if len(all_dots) >= 2:
            blue = frame[:, :, 0]
            dot_brightness = []
            for name, (dx, dy) in all_dots.items():
                ix, iy = int(dx), int(dy)
                y1, y2 = max(0, iy - 2), min(h_frame, iy + 3)
                x1, x2 = max(0, ix - 2), min(w_frame, ix + 3)
                patch = blue[y1:y2, x1:x2]
                val = int(np.max(patch)) if patch.size > 0 else 0
                dot_brightness.append((val, name))
            dot_brightness.sort(key=lambda x: -x[0])
            best_val, best_name = dot_brightness[0]
            second_val = dot_brightness[1][0]
            if best_val > second_val + 15:
                lit_led_name = best_name

    if lit_led_name:
        led_dots['_lit'] = lit_led_name

    # Add B1_proj for overlay display
    if b1_proj is not None:
        led_dots['B1_proj'] = ((int(b1_proj[0]), int(b1_proj[1])), 'predicted')
    mute_proj = _geometry.project_landmark('mute_led')
    if mute_proj is not None:
        led_dots['mute_proj'] = ((int(mute_proj[0]), int(mute_proj[1])), 'predicted')

    _frame_led_dots = led_dots


def detect_panel(frame):
    """
    Detect the dark rectangular panel containing blue LED digits.

    Uses landmark-based prediction (corner + buttons) as primary method,
    with calibrated position from camera_mount.json as fallback.

    Args:
        frame: BGR image from camera/file

    Returns:
        panel_rect: (x, y, w, h) of the detected panel, or None if not found
        method: detection method used ('landmark', 'tracked', 'calibrated', or None)
    """
    h_frame, w_frame = frame.shape[:2]

    # Reset geo_method - will be set when landmarks/tracking succeeds
    _geometry._geo_method = 'none'

    # Try landmark-based detection first (corner + buttons)
    landmark_panel = predict_panel_from_landmarks(frame)
    if landmark_panel is not None:
        return landmark_panel, 'landmark'

    # Tracking restore: reuse golden homography when landmarks disappear
    if _TRACKING and _geometry.restore_golden():
        panel_rect = _geometry.get_panel_rect()
        if panel_rect is not None:
            px, py, pw, ph = panel_rect
            px = max(0, px)
            py = max(0, py)
            pw = min(w_frame - px, pw)
            ph = min(h_frame - py, ph)
            if pw >= 50 and ph >= 30:
                _geometry._geo_method = 'homography'
                return (px, py, pw, ph), 'tracked'

    # Fallback: use calibrated panel position from camera_mount.json
    panel_rect = _geometry.get_panel_rect()
    if panel_rect is not None:
        px, py, pw, ph = panel_rect
        px = max(0, px)
        py = max(0, py)
        pw = min(w_frame - px, pw)
        ph = min(h_frame - py, ph)
        if pw >= 50 and ph >= 30:
            _geometry._geo_method = 'calibrated'
            return (px, py, pw, ph), 'calibrated'

    return None, None


def _led_diff_log_only(button_region, button_zones, lit_led):
    """Log per-zone grayscale diffs with hysteresis snapshot.

    Snapshot is taken with 2px padding around each zone. On subsequent frames,
    if the zone drifts within the padding, the matching sub-region is extracted
    from the padded snapshot for comparison. Only re-snapshots when a zone
    exceeds the padding buffer or diff exceeds threshold.
    """
    global _led_diff_snapshots, _led_diff_zones, _led_diff_lit, _led_diff_log, _led_diff_frame_n

    if not _LOG_ENABLED or len(button_zones) < 3:
        return

    _led_diff_frame_n += 1
    pad = _LED_DIFF_PAD
    gray = cv2.cvtColor(button_region, cv2.COLOR_BGR2GRAY)
    gh, gw = gray.shape[:2]

    # Build current zone bounds
    current_bounds = {}
    for left_x, right_x, top_y, bottom_y, name in button_zones:
        x1, x2 = int(left_x), int(right_x)
        y1, y2 = int(top_y), int(bottom_y)
        if x1 < x2 and y1 < y2 and x2 <= gw and y2 <= gh:
            current_bounds[name] = (x1, y1, x2, y2)

    # Compare to snapshots using hysteresis
    diffs = {}
    need_resnap = False
    if _led_diff_snapshots is not None and _led_diff_zones is not None:
        for name, (cx1, cy1, cx2, cy2) in current_bounds.items():
            snap = _led_diff_snapshots.get(name)
            snap_bounds = _led_diff_zones.get(name)
            if snap is None or snap_bounds is None:
                need_resnap = True
                break
            sx1, sy1, sx2, sy2 = snap_bounds  # padded bounds

            # Check if current zone is within the padded snapshot
            if cx1 < sx1 or cy1 < sy1 or cx2 > sx2 or cy2 > sy2:
                need_resnap = True
                break

            # Extract matching sub-region from padded snapshot
            ox = cx1 - sx1  # offset within padded crop
            oy = cy1 - sy1
            cw = cx2 - cx1
            ch = cy2 - cy1
            snap_sub = snap[oy:oy+ch, ox:ox+cw]
            current_crop = gray[cy1:cy2, cx1:cx2]

            if snap_sub.shape != current_crop.shape:
                need_resnap = True
                break

            diffs[name] = float(np.mean(np.abs(
                current_crop.astype(np.int16) - snap_sub.astype(np.int16))))
    else:
        need_resnap = True

    # Check if diff exceeds threshold
    valid_diffs = [v for v in diffs.values() if v >= 0] if diffs else []
    max_diff_val = max(valid_diffs) if valid_diffs else 0
    if max_diff_val >= 5.0:
        need_resnap = True

    # Log — always include per-zone diffs when available
    if _led_diff_log is None:
        log_path = os.path.join(_LOG_DIR, 'led_diff_experiment.csv')
        os.makedirs(_LOG_DIR, exist_ok=True)
        _led_diff_log = open(log_path, 'w')
        _led_diff_log.write('frame_n,B1_diff,B2_diff,S1_diff,S2_diff,max_diff,lit_led,prev_lit,changed,resnap\n')

    b1d = f"{diffs.get('B1', -1):.2f}" if diffs else ""
    b2d = f"{diffs.get('B2', -1):.2f}" if diffs else ""
    s1d = f"{diffs.get('S1', -1):.2f}" if diffs else ""
    s2d = f"{diffs.get('S2', -1):.2f}" if diffs else ""
    md = f"{max_diff_val:.2f}" if valid_diffs else ""
    prev_lit = _led_diff_lit or ''
    changed = '1' if lit_led != _led_diff_lit else '0'
    resnap = '1' if need_resnap else '0'
    _led_diff_log.write(f'{_led_diff_frame_n},{b1d},{b2d},{s1d},{s2d},{md},{lit_led or ""},{prev_lit},{changed},{resnap}\n')
    _led_diff_log.flush()

    # Re-snapshot with padding
    if need_resnap:
        _led_diff_snapshots = {}
        _led_diff_zones = {}
        for name, (cx1, cy1, cx2, cy2) in current_bounds.items():
            # Padded bounds, clipped to image
            px1 = max(0, cx1 - pad)
            py1 = max(0, cy1 - pad)
            px2 = min(gw, cx2 + pad)
            py2 = min(gh, cy2 + pad)
            _led_diff_snapshots[name] = gray[py1:py2, px1:px2].copy()
            _led_diff_zones[name] = (px1, py1, px2, py2)

    _led_diff_lit = lit_led


def detect_button_leds(frame, panel_rect=None, debug=False, return_debug=False, detection_method=None):
    """
    Detect which button LED (B1, B2, S1, S2) is lit.

    The buttons are arranged left to right below the display panel.
    Each button has a small LED indicator that can appear blue or bright white.
    Only one LED is lit at a time.

    Approach: Detect 3 visible buttons (B1 is cut off at left edge), estimate B1
    position from spacing, then find the single brightest LED blob and determine
    which button zone it belongs to.

    Args:
        frame: BGR image from camera/file
        panel_rect: (x, y, w, h) of detected panel, used to locate button region
        debug: If True, return debug image showing detection

    Returns:
        leds: dict mapping LED name to bool (True=lit, False=off)
              e.g., {'B1': False, 'B2': True, 'S1': False, 'S2': False}
        debug_img: (only if debug=True) Image showing LED detection
    """
    h_frame, w_frame = frame.shape[:2]
    debug_img = frame.copy() if debug else None

    # Define button region - prefer geometry-based when corner is known
    geo_region = _geometry.get_button_region_from_geometry(w_frame, h_frame)
    if geo_region is not None:
        btn_top, btn_bottom, btn_left, btn_right = geo_region
    elif panel_rect is not None:
        btn_top, btn_bottom, btn_left, btn_right = _geometry.get_button_region_from_panel(
            panel_rect, w_frame, h_frame)
    else:
        btn_top, btn_bottom, btn_left, btn_right = _geometry.get_button_region_fallback(
            w_frame, h_frame)

    # Extract button region
    button_region = frame[btn_top:btn_bottom, btn_left:btn_right]
    bh, bw = button_region.shape[:2]

    leds = {'B1': False, 'B2': False, 'S1': False, 'S2': False}
    button_names = ['B1', 'B2', 'S1', 'S2']

    global _button_zone_cache, _cache_led_fail_count

    # Check for severe overexposure - when entire button region is blown out,
    # LED detection is unreliable (LED blob merges with ambient brightness)
    gray_check = cv2.cvtColor(button_region, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray_check)
    if mean_brightness > 230:
        # Overexposed — standard LED detection unreliable.
        # Fallback: detect lit LED by absence of "dark hole".
        # Lit LED fills the button hole → highest min brightness.
        washout_zones = _button_zone_cache
        if washout_zones is None:
            # No cache — use default zones from geometry
            washout_zones = _geometry.get_default_button_zones(bw, bh)

        # Min-filter each zone (5x5 erosion = darkest 5x5 block)
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(gray_check, kernel)

        zone_mins = []
        for lx, rx, ty, by_, name in washout_zones:
            lx, rx = max(0, int(lx)), min(bw, int(rx))
            ty, by_ = max(0, int(ty)), min(bh, int(by_))
            zone_eroded = eroded[ty:by_, lx:rx]
            if zone_eroded.size > 0:
                zone_mins.append((int(np.min(zone_eroded)), name))

        lit_led = None
        washout_gap = 0
        if len(zone_mins) >= 2:
            zone_mins.sort(reverse=True)  # highest min first
            best_min, best_name = zone_mins[0]
            second_min, _ = zone_mins[1]
            washout_gap = best_min - second_min
            if washout_gap >= _WASHOUT_MIN_GAP:
                lit_led = best_name
                leds[lit_led] = True

        # Return with debug info
        washout_debug = {
            'region': (btn_left, btn_top, btn_right, btn_bottom),
            'zones': washout_zones,
            'buttons': [],
            'predicted_b1_box': None,
            'led_position': None,
            'lit_led': lit_led,
            'leds': leds,
            'brightness_gap': washout_gap,
            'led_method': 'washout' if lit_led else None,
        }
        if debug:
            # Draw button region boundary
            cv2.rectangle(debug_img, (btn_left, btn_top), (btn_right, btn_bottom),
                          (100, 100, 100), 1)

            # Draw LED zones
            for left_x, right_x, top_y, bottom_y, name in washout_zones:
                lx = int(left_x) + btn_left
                rx = int(right_x) + btn_left
                ty = int(top_y) + btn_top
                by = int(bottom_y) + btn_top
                color = (0, 255, 0) if leds[name] else (128, 128, 128)
                cv2.rectangle(debug_img, (lx, ty), (rx, by), color, 1)
                cv2.putText(debug_img, name, (lx + 5, ty + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # Show method label
            method = "washout" if lit_led else "washout(NA)"
            cv2.putText(debug_img, f"Method: {method}",
                        (btn_left + 5, btn_bottom - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if return_debug:
            return leds, debug_img, washout_debug
        if debug:
            return leds, debug_img
        return leds, None

    # Detect button rectangles (typically finds 3 - B1 is cut off at left edge)
    # Reuse cached buttons from predict_panel_from_landmarks() if region matches (#74)
    region_key = (btn_top, btn_bottom, btn_left, btn_right)
    if _cached_buttons is not None and _cached_buttons[0] == region_key:
        buttons = _cached_buttons[1]
    else:
        buttons = _detect_buttons(button_region)
        buttons = sorted(buttons, key=lambda b: b[0])

    # Create LED mask for detection
    led_mask = _create_led_mask(button_region)

    button_zones = []  # List of (center_x, name) for each button
    used_cache = False

    predicted_b1_box = None  # Will store predicted B1 button box (x, y, w, h)

    if len(buttons) >= 3:
        # We have 3+ buttons - use rightmost 3: B2, S1, S2
        # (B1 is partially cut off at left edge, skip if 4 detected)
        b2_btn, s1_btn, s2_btn = buttons[-3], buttons[-2], buttons[-1]
        widths = [b2_btn[2], s1_btn[2], s2_btn[2]]
        heights = [b2_btn[3], s1_btn[3], s2_btn[3]]
        avg_width = sum(widths) / len(widths)
        avg_height = sum(heights) / len(heights)

        # Get centers of B2, S1, S2 (rightmost 3 buttons)
        b2_center = b2_btn[0] + b2_btn[2] // 2
        s1_center = s1_btn[0] + s1_btn[2] // 2
        s2_center = s2_btn[0] + s2_btn[2] // 2

        # Try homography projection first (accounts for lens distortion)
        # Note: project_landmark('B1') returns LED dot position, not button center
        # LED is at ~78% of button width from left edge (measured from B2/S1/S2)
        LED_FRAC_X = 0.78
        b1_proj = _geometry.project_landmark('B1')
        if b1_proj is not None:
            b1_abs_x, b1_abs_y = b1_proj
            # Convert to button-region-relative coords
            b1_led_x = b1_abs_x - btn_left
            b1_y_rel = b1_abs_y - btn_top
            b1_x = int(b1_led_x - LED_FRAC_X * avg_width)
            b1_y = int(b1_y_rel - avg_height / 2)
            predicted_b1_box = (b1_x, b1_y, int(avg_width), int(avg_height))
        else:
            # Fallback: pixel-space extrapolation (no homography available)
            spacing_b2_s1 = s1_center - b2_center
            spacing_s1_s2 = s2_center - s1_center
            avg_spacing = (spacing_b2_s1 + spacing_s1_s2) / 2

            b1_center = b2_center - avg_spacing
            b1_x = int(b1_center - avg_width / 2)
            b1_y = b2_btn[1]
            predicted_b1_box = (b1_x, b1_y, int(avg_width), int(b2_btn[3]))

        # Build LED zones with boundaries (left_x, right_x, top_y, bottom_y, name)
        # LED is on the right side of each button (50%-100% of button width)
        half_width = avg_width / 2

        # Get Y boundaries from detected buttons (B2, S1, S2)
        b2_top, b2_bottom = b2_btn[1], b2_btn[1] + b2_btn[3]
        s1_top, s1_bottom = s1_btn[1], s1_btn[1] + s1_btn[3]
        s2_top, s2_bottom = s2_btn[1], s2_btn[1] + s2_btn[3]
        # B1: use projected Y when available, else B2's Y
        if b1_proj is not None:
            b1_top = int(b1_y_rel - avg_height / 2)
            b1_bottom = int(b1_y_rel + avg_height / 2)
        else:
            b1_top, b1_bottom = b2_top, b2_bottom

        # LED zone for B1
        if b1_proj is not None:
            # Homography projection: center zone around projected LED position
            led_zone_half = half_width / 2
            b1_led_left = max(0, int(b1_led_x - led_zone_half))
            b1_led_right = min(int(b1_led_x + led_zone_half), b2_center - _geometry.b1_b2_spacing)
            button_zones.append((b1_led_left, b1_led_right, b1_top, b1_bottom, 'B1'))
        else:
            # Extrapolation fallback: offset LED position (B1 partially off-screen)
            b1_led_x = b1_x + avg_width * _geometry.b1_led_position_ratio
            if b1_led_x > _geometry.b1_led_min_visible_px:
                b1_led_left = max(_geometry.b1_led_edge_margin, b1_led_x - half_width / 2)
                b1_led_right = min(b1_led_x + half_width / 2, b2_center - _geometry.b1_b2_spacing)
                button_zones.append((b1_led_left, b1_led_right, b1_top, b1_bottom, 'B1'))
        # B2, S1, S2 (detected)
        button_zones.append((b2_center, b2_center + half_width, b2_top, b2_bottom, 'B2'))
        button_zones.append((s1_center, s1_center + half_width, s1_top, s1_bottom, 'S1'))
        button_zones.append((s2_center, s2_center + half_width, s2_top, s2_bottom, 'S2'))

        # Update cache if zones changed significantly (only if we have 3+ zones)
        if len(button_zones) >= 3:
            if _zones_changed_significantly(_button_zone_cache, button_zones):
                _button_zone_cache = list(button_zones)
                _save_cache()

    if len(button_zones) < 3:
        # Try cached zones first
        if _button_zone_cache is not None:
            button_zones = _button_zone_cache
            used_cache = True
        else:
            # Fallback: use default zones from geometry model
            button_zones = _geometry.get_default_button_zones(bw, bh)

        # When in fallback mode WITHOUT cache, enlarge the LED detection zones
        # Skip enlargement when using cached zones UNLESS cache is failing repeatedly
        cache_seems_stale = used_cache and _cache_led_fail_count >= _CACHE_FAIL_THRESHOLD
        if cache_seems_stale:
            clear_cache()
            _cache_led_fail_count = 0
        if (not used_cache or cache_seems_stale) and detection_method is not None and detection_method not in ('landmark', 'tracked'):
            button_zones = _geometry.enlarge_zones(button_zones, bw, bh)

    # Find the LED blob inside any button zone
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        led_mask, connectivity=8)

    # Collect all valid blobs (basic size/shape filtering)
    valid_blobs = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        blob_w = stats[i, cv2.CC_STAT_WIDTH]
        blob_h = stats[i, cv2.CC_STAT_HEIGHT]
        blob_x = int(centroids[i][0])
        blob_y = int(centroids[i][1])

        # LED should be a compact blob with reasonable size
        if _LED_MIN_AREA <= area < _LED_MAX_AREA:
            aspect = max(blob_w, blob_h) / max(1, min(blob_w, blob_h))
            if aspect < _LED_MAX_ASPECT_RATIO:  # Reasonably compact
                valid_blobs.append((blob_x, blob_y, area))

    # === Compute all 3 methods independently, then pick best ===
    lit_led = None
    led_position = None
    led_method = None  # Track which method detected the LED (brightness/blob/center)

    # Method 1: Blob detection — largest blob inside a button zone
    blob_winner = None
    blob_pos = None
    best_area = 0
    for blob_x, blob_y, area in valid_blobs:
        for left_x, right_x, top_y, bottom_y, name in button_zones:
            if (left_x <= blob_x <= right_x and
                top_y <= blob_y <= bottom_y and
                area > best_area):
                best_area = area
                blob_winner = name
                blob_pos = (blob_x + btn_left, blob_y + btn_top)

    # Method 2: Brightness-based using blue channel
    brightness_winner = None
    brightness_pos = None
    brightest_val = 0
    brightness_gap = 0
    if len(button_zones) > 0:
        blue_channel = button_region[:, :, 0]  # Blue channel for LED detection
        zone_brightness = []
        for left_x, right_x, top_y, bottom_y, name in button_zones:
            x1, x2 = int(left_x), int(right_x)
            y1, y2 = int(top_y), int(bottom_y)
            if x1 < x2 and y1 < y2 and x2 <= blue_channel.shape[1] and y2 <= blue_channel.shape[0]:
                zone = blue_channel[y1:y2, x1:x2]
                if zone.size > 0:
                    max_bright = int(np.max(zone))
                    zone_brightness.append((name, max_bright, (x1 + x2) // 2, (y1 + y2) // 2))

        if zone_brightness:
            zone_brightness.sort(key=lambda x: -x[1])
            brightness_winner = zone_brightness[0][0]
            brightest_val = zone_brightness[0][1]
            brightness_pos = (zone_brightness[0][2] + btn_left, zone_brightness[0][3] + btn_top)
            second_val = zone_brightness[1][1] if len(zone_brightness) > 1 else 0
            brightness_gap = brightest_val - second_val

    # Method 3: Center brightness detection
    center_winner = None
    center_pos = None
    center_gap = 0
    center_val = 0
    if len(button_zones) > 0:
        zone_centers = []
        for left_x, right_x, top_y, bottom_y, name in button_zones:
            x1, x2 = int(left_x), int(right_x)
            y1, y2 = int(top_y), int(bottom_y)
            if x1 < x2 and y1 < y2 and x2 <= blue_channel.shape[1] and y2 <= blue_channel.shape[0]:
                zone = blue_channel[y1:y2, x1:x2]
                if zone.size > 0:
                    h, w = zone.shape
                    center_zone = zone[h//4:3*h//4, w//4:3*w//4]
                    if center_zone.size > 0:
                        zone_centers.append((center_zone.mean(), name, (x1, y1, x2, y2)))
        if len(zone_centers) >= 2:
            zone_centers.sort(key=lambda x: x[0], reverse=True)
            center_winner = zone_centers[0][1]
            center_val = zone_centers[0][0]
            center_gap = center_val - zone_centers[1][0]
            x1, y1, x2, y2 = zone_centers[0][2]
            center_pos = ((x1 + x2) // 2 + btn_left, (y1 + y2) // 2 + btn_top)

    # === Decision: landmark first, then agreement-based fallback ===
    landmark_lit = _frame_led_dots.get('_lit') if _frame_led_dots else None

    if landmark_lit is not None:
        # (a) Landmark LED dot detection — primary method
        lit_led = landmark_lit
        led_method = 'landmark_dot'
        if landmark_lit in _frame_led_dots:
            pos, _ = _frame_led_dots[landmark_lit]
            led_position = (int(pos[0]), int(pos[1]))
    elif brightest_val > 200 and brightness_gap > 30:
        # (b) Brightness confident — fallback
        lit_led, led_position, led_method = brightness_winner, brightness_pos, 'brightness'
    elif blob_winner is not None and blob_winner == brightness_winner:
        # (c) Blob agrees with brightest zone — fallback
        lit_led, led_position, led_method = blob_winner, blob_pos, 'blob'
    elif center_val > 220 and center_gap > 5:
        # (d) Center confident — fallback
        lit_led, led_position, led_method = center_winner, center_pos, 'center'
    elif blob_winner is not None and brightest_val > 200:
        # (e) Blob found something in a bright region — fallback
        lit_led, led_position, led_method = blob_winner, blob_pos, 'blob'

    if lit_led:
        leds[lit_led] = True

    # Track LED detection failures when using cached zones
    if used_cache:
        if lit_led is None:
            _cache_led_fail_count += 1
        else:
            _cache_led_fail_count = 0  # Reset on success

    # --- LED diff experiment: log only ---
    _led_diff_log_only(button_region, button_zones, lit_led)

    # Build debug info for return_debug mode
    led_debug_info = None
    if return_debug:
        led_debug_info = {
            'region': (btn_left, btn_top, btn_right, btn_bottom),
            'zones': button_zones,
            'buttons': buttons,  # Detected buttons (B2, S1, S2)
            'predicted_b1_box': predicted_b1_box,  # Predicted B1 box
            'led_position': led_position,
            'lit_led': lit_led,
            'leds': leds,
            'brightness_gap': brightness_gap,  # Gap between brightest and 2nd brightest zone
            'led_method': led_method,  # Which method detected: brightness/blob/center
            'led_dots': dict(_frame_led_dots) if _frame_led_dots else None,  # Per-button LED dot positions
        }

    if debug:
        # Draw button region boundary
        cv2.rectangle(debug_img, (btn_left, btn_top), (btn_right, btn_bottom),
                      (100, 100, 100), 1)

        # Draw LED zones (boundaries with X and Y constraints)
        for left_x, right_x, top_y, bottom_y, name in button_zones:
            lx = int(left_x) + btn_left
            rx = int(right_x) + btn_left
            ty = int(top_y) + btn_top
            by = int(bottom_y) + btn_top
            color = (0, 255, 0) if leds[name] else (128, 128, 128)
            cv2.rectangle(debug_img, (lx, ty), (rx, by), color, 1)
            cv2.putText(debug_img, name, (lx + 5, ty + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Draw detected buttons
        for bx, by, bw_btn, bh_btn in buttons:
            cv2.rectangle(debug_img,
                          (bx + btn_left, by + btn_top),
                          (bx + btn_left + bw_btn, by + btn_top + bh_btn),
                          (255, 255, 0), 1)

        # Draw LED position
        if led_position:
            cv2.circle(debug_img, led_position, 8, (0, 255, 0), 2)
            cv2.putText(debug_img, f"{lit_led}:ON",
                        (led_position[0] - 20, led_position[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # Show detection method
        if len(buttons) >= 3:
            method = f"3-btn ({len(buttons)} detected)"
        elif used_cache:
            method = "cached"
        else:
            method = "fallback"
        cv2.putText(debug_img, f"Method: {method}",
                    (btn_left + 5, btn_bottom - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if return_debug:
            return leds, debug_img, led_debug_info
        return leds, debug_img

    if return_debug:
        return leds, None, led_debug_info
    return leds, None


def _draw_dashed_rect(frame, pt1, pt2, color, thickness=1, dash_length=8):
    """Draw a dashed rectangle on frame."""
    x1, y1 = pt1
    x2, y2 = pt2

    def draw_dashed_line(start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = max(abs(dx), abs(dy))
        if length == 0:
            return
        num_dashes = max(1, int(length / dash_length))
        for i in range(0, num_dashes, 2):
            t1 = i / num_dashes
            t2 = min((i + 1) / num_dashes, 1.0)
            p1 = (int(start[0] + dx * t1), int(start[1] + dy * t1))
            p2 = (int(start[0] + dx * t2), int(start[1] + dy * t2))
            cv2.line(frame, p1, p2, color, thickness)

    draw_dashed_line((x1, y1), (x2, y1))  # Top
    draw_dashed_line((x2, y1), (x2, y2))  # Right
    draw_dashed_line((x2, y2), (x1, y2))  # Bottom
    draw_dashed_line((x1, y2), (x1, y1))  # Left


def draw_led_debug(frame, led_debug_info, dashed=False):
    """Draw LED detection debug info on frame.

    Args:
        frame: BGR image to draw on (modified in place)
        led_debug_info: Debug info dict from detect_button_leds(return_debug=True)
        dashed: If True, draw all rectangles with dashed lines (e.g. during washout)
    """
    if led_debug_info is None:
        return

    btn_left, btn_top, btn_right, btn_bottom = led_debug_info['region']
    button_zones = led_debug_info.get('zones', [])
    buttons = led_debug_info.get('buttons', [])
    predicted_b1_box = led_debug_info.get('predicted_b1_box')
    led_position = led_debug_info.get('led_position')
    lit_led = led_debug_info.get('lit_led')
    leds = led_debug_info.get('leds', {})

    _rect = lambda f, p1, p2, c, t: _draw_dashed_rect(f, p1, p2, c, t) if dashed else cv2.rectangle(f, p1, p2, c, t)

    # Draw button region boundary
    _rect(frame, (btn_left, btn_top), (btn_right, btn_bottom),
          (100, 100, 100), 1)

    if dashed:
        # During washout, just draw the region boundary — skip zone/button details
        return

    # Draw LED zones (boundaries with X and Y constraints)
    for left_x, right_x, top_y, bottom_y, name in button_zones:
        lx = int(left_x) + btn_left
        rx = int(right_x) + btn_left
        ty = int(top_y) + btn_top
        by = int(bottom_y) + btn_top
        is_lit = leds.get(name)
        color = (0, 255, 0) if is_lit else (128, 128, 128)
        thickness = 2 if is_lit else 1
        cv2.rectangle(frame, (lx, ty), (rx, by), color, thickness)
        cv2.putText(frame, name, (lx + 5, ty + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Draw detected buttons (B2, S1, S2) with solid boxes
    for bx, by, bw_btn, bh_btn in buttons:
        cv2.rectangle(frame,
                      (bx + btn_left, by + btn_top),
                      (bx + btn_left + bw_btn, by + btn_top + bh_btn),
                      (255, 255, 0), 1)


    # Draw LED dot landmarks (from _find_led_in_button)
    if _frame_led_dots:
        for name, val in _frame_led_dots.items():
            if name.startswith('_'):
                continue  # Skip metadata keys like '_lit'
            pos, found = val
            px, py = int(pos[0]), int(pos[1])
            if found == 'predicted':
                # Projected from homography (B1): yellow arrow
                color = (0, 255, 255)
                label = f"{name} pred"
            elif found:
                # LED dot found: green arrow pointing down to it
                color = (0, 255, 0)
                label = f"{name} dot"
            else:
                # Fallback to button center: orange arrow
                color = (0, 200, 255)
                label = f"{name} ctr"
            if found == 'predicted':
                # Yellow arrows: below LED pointing up
                cv2.arrowedLine(frame, (px, py + 20), (px, py + 4),
                                color, 2, tipLength=0.4)
                cv2.putText(frame, label, (px - 15, py + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            else:
                # Green/orange arrows: above LED pointing down
                cv2.arrowedLine(frame, (px, py - 20), (px, py - 4),
                                color, 2, tipLength=0.4)
                cv2.putText(frame, label, (px - 15, py - 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)


def draw_mute_debug(frame, mute_debug_info, dashed=False):
    """Draw MUTE LED detection debug info on frame.

    Args:
        frame: BGR image to draw on (modified in place)
        mute_debug_info: Debug info dict from detect_red_button(return_debug=True)
        dashed: If True, draw region with dashed lines (e.g. during washout)
    """
    if mute_debug_info is None:
        return

    # Mute LED position info is shown in the zoom inset overlay;
    # no additional markers needed on the main frame.
    return


def draw_digit_debug(frame, panel_rect, digit_debug):
    """Draw digit matching debug info on frame.

    Shows the search area (padded digit box) and where the template matched.

    Args:
        frame: BGR image to draw on (modified in place)
        panel_rect: (x, y, w, h) of the panel in frame coordinates
        digit_debug: Debug info dict from SegmentReader.digit_debug
    """
    if digit_debug is None or panel_rect is None:
        return

    px, py, pw, ph = panel_rect
    padding = _DIGIT_PADDING
    padding_v = _DIGIT_PADDING_V

    # Get corrected image size for scaling (box coords are in corrected space)
    corrected_size = digit_debug.get('corrected_size')
    if corrected_size:
        cw, ch = corrected_size
        scale_x = pw / cw
        scale_y = ph / ch
    else:
        scale_x = scale_y = 1.0

    # Get gap_x for bounding (in corrected coords)
    gap_x = digit_debug.get('gap_x')
    gap_x_scaled = int(gap_x * scale_x) if gap_x is not None else None

    for side, box_key, match_key, color in [
        ('L', 'left_box', 'left_match', (255, 0, 255)),   # Magenta for left
        ('R', 'right_box', 'right_match', (0, 255, 255)), # Yellow for right
    ]:
        box = digit_debug.get(box_key)
        match_info = digit_debug.get(match_key)
        if box is None:
            continue

        bx, by, bw, bh = box

        # Scale box coordinates from corrected space to display space
        bx_scaled = int(bx * scale_x)
        by_scaled = int(by * scale_y)
        bw_scaled = int(bw * scale_x)
        bh_scaled = int(bh * scale_y)
        padding_x = int(padding * scale_x)
        padding_y = int(padding_v * scale_y)

        # Search area (padded box) in frame coordinates
        # Apply bounds like _extract_digit_with_padding does
        search_x = px + bx_scaled - padding_x
        search_y = py + by_scaled - padding_y
        search_x2 = px + bx_scaled + bw_scaled + padding_x
        search_y2 = py + by_scaled + bh_scaled + padding_y

        # Apply gap bounds: left digit bounded on right, right digit bounded on left
        if gap_x_scaled is not None:
            if side == 'L':
                # Left digit: right_bound = gap_x
                search_x2 = min(search_x2, px + gap_x_scaled)
            else:
                # Right digit: left_bound = gap_x
                search_x = max(search_x, px + gap_x_scaled)

        search_w = search_x2 - search_x
        search_h = search_y2 - search_y

        # Draw search area
        cv2.rectangle(frame, (search_x, search_y),
                      (search_x + search_w, search_y + search_h), color, 1)

        # Draw match position if available
        if match_info and match_info.get('match_pos') and match_info.get('template_size'):
            mx, my = match_info['match_pos']
            tw, th = match_info['template_size']
            # Scale match position and template size
            mx_scaled = int(mx * scale_x)
            my_scaled = int(my * scale_y)
            tw_scaled = int(tw * scale_x)
            th_scaled = int(th * scale_y)
            # Match position is relative to search area
            match_x = search_x + mx_scaled
            match_y = search_y + my_scaled
            cv2.rectangle(frame, (match_x, match_y),
                          (match_x + tw_scaled, match_y + th_scaled), color, 2)


def get_noise_mean(frame):
    """Get noise_mean from neutral housing region. Returns float or None."""
    noise_region_info = _geometry.get_noise_region()
    if noise_region_info is None:
        return None
    nr_cx, nr_cy, nr_half = noise_region_info
    h_frame, w_frame = frame.shape[:2]
    nr_left = max(0, nr_cx - nr_half)
    nr_right = min(w_frame, nr_cx + nr_half)
    nr_top = max(0, nr_cy - nr_half)
    nr_bottom = min(h_frame, nr_cy + nr_half)
    nr_patch = frame[nr_top:nr_bottom, nr_left:nr_right]
    if nr_patch.size == 0:
        return None
    return float(np.mean(nr_patch[:, :, 1]))


def _compute_mute_contrast(frame, geometry):
    """Compute local contrast between mute LED patch and reference patch.

    A/B phase (#72): values are logged but don't affect detection decisions.

    Args:
        frame: BGR image from camera.
        geometry: DeviceGeometry instance with homography.

    Returns:
        Dict with contrast values and patch centers, or None if unavailable.
    """
    # Prefer smoothed positions, fall back to raw
    led_s = geometry.get_mute_led_center(smoothed=True)
    if led_s is None:
        led_s = geometry.get_mute_led_center(smoothed=False)
    if led_s is None:
        return None

    ref_s = geometry.get_mute_ref_center(smoothed=True)
    if ref_s is None:
        ref_s = geometry.get_mute_ref_center(smoothed=False)
    led_r = geometry.get_mute_led_center(smoothed=False)

    h_frame, w_frame = frame.shape[:2]
    radius = geometry.mute_led_patch_radius  # 4 → 9x9 patch

    # Check ref patch fits within frame (LED patch can be clipped at edge)
    if ref_s is None:
        return None
    rx, ry = ref_s
    if (rx - radius < 0 or rx + radius >= w_frame or
            ry - radius < 0 or ry + radius >= h_frame):
        return None

    # Extract patches (9x9 at default radius=4), clipped to frame bounds
    def _extract_patch(center):
        cx, cy = int(round(center[0])), int(round(center[1]))
        y1 = max(0, cy - radius)
        y2 = min(h_frame, cy + radius + 1)
        x1 = max(0, cx - radius)
        x2 = min(w_frame, cx + radius + 1)
        return frame[y1:y2, x1:x2]

    led_patch = _extract_patch(led_s)
    ref_patch = _extract_patch(ref_s)

    if led_patch.size == 0 or ref_patch.size == 0:
        return None

    # Red channel means
    led_red = float(np.mean(led_patch[:, :, 2]))
    ref_red = float(np.mean(ref_patch[:, :, 2]))

    # Green channel means (for red excess)
    led_green = float(np.mean(led_patch[:, :, 1]))
    ref_green = float(np.mean(ref_patch[:, :, 1]))

    # Gray means
    led_gray = float(np.mean(cv2.cvtColor(led_patch, cv2.COLOR_BGR2GRAY)))
    ref_gray = float(np.mean(cv2.cvtColor(ref_patch, cv2.COLOR_BGR2GRAY)))

    # Ratios (LED / reference, clamped to avoid div-by-zero)
    red_ratio = led_red / max(ref_red, 1.0)
    gray_ratio = led_gray / max(ref_gray, 1.0)

    # Red excess: (R-G)_LED - (R-G)_REF
    # Measures red color difference independent of overall brightness/tint
    red_excess = (led_red - led_green) - (ref_red - ref_green)

    return {
        'mute_rr': round(red_ratio, 2),
        'mute_re': round(red_excess, 1),
        'mute_gr': round(gray_ratio, 2),
        'mute_led_r': round(led_red, 1),
        'mute_ref_r': round(ref_red, 1),
        'mute_led_sx': round(led_s[0], 1),
        'mute_led_sy': round(led_s[1], 1),
        'mute_led_rx': round(led_r[0], 1) if led_r else None,
        'mute_led_ry': round(led_r[1], 1) if led_r else None,
        'mute_ref_sx': round(ref_s[0], 1),
        'mute_ref_sy': round(ref_s[1], 1),
        'mute_h_age': geometry.get_homography_age(),
    }


def detect_red_button(frame, debug=False, return_debug=False, corner_result=None):
    """
    Detect if the red button LED (MUTE indicator) is lit.

    Uses template matching to find the corner, then searches for red LED
    in a region relative to the corner position. Falls back to fixed region
    if corner not found.

    Args:
        frame: BGR image from camera/file
        corner_result: Optional pre-computed corner result (x, y, score) to avoid redundant detection
        debug: If True, return debug image showing detection
        return_debug: If True, return debug info dict for visualization

    Returns:
        is_lit: bool, True if red LED is lit
        debug_img: (only if debug=True) Image showing detection
        debug_info: (only if return_debug=True) Dict with detection region info
    """
    h_frame, w_frame = frame.shape[:2]
    debug_img = frame.copy() if debug else None

    # Use provided corner result or find corner using template matching
    if corner_result is None:
        corner_result = _find_corner(frame)

    if corner_result is not None:
        corner_x, corner_y, match_score = corner_result[0], corner_result[1], corner_result[2]
        mute = _geometry.get_mute_region(corner_x, corner_y)
    else:
        mute = _geometry.get_mute_region()  # Uses persistent homography

    if mute is None:
        # camera_mount.json is required for v4.0 architecture
        if return_debug:
            return False, debug_img, {'region': (0, 0, 0, 0), 'method': 'none',
                                       'is_lit': False, 'led_center': None,
                                       'noise_mean': None, 'mute_proj': None}
        if debug:
            return False, debug_img
        return False, None

    btn_x, btn_y, region_half = mute
    region_left = max(0, btn_x - region_half)
    region_right = min(w_frame, btn_x + region_half)
    region_top = max(0, btn_y - region_half)
    region_bottom = min(h_frame, btn_y + region_half)
    method = "corner" if corner_result else "homography"
    mute_proj = (btn_x, btn_y)

    # Extract search region
    region = frame[region_top:region_bottom, region_left:region_right]
    if region.size == 0:
        if return_debug:
            return False, debug_img, {'region': (region_left, region_top, region_right, region_bottom),
                                       'method': method, 'is_lit': False, 'led_center': None,
                                       'noise_mean': None, 'mute_proj': mute_proj}
        if debug:
            return False, debug_img
        return False, None

    # Neutral region noise measurement (between corner and mute button)
    noise_std = None
    noise_mean = None
    noise_region_info = _geometry.get_noise_region()
    if noise_region_info is not None:
        nr_cx, nr_cy, nr_half = noise_region_info
        nr_left = max(0, nr_cx - nr_half)
        nr_right = min(w_frame, nr_cx + nr_half)
        nr_top = max(0, nr_cy - nr_half)
        nr_bottom = min(h_frame, nr_cy + nr_half)
        nr_patch = frame[nr_top:nr_bottom, nr_left:nr_right]
        if nr_patch.size > 0:
            nr_green = nr_patch[:, :, 1]
            noise_std = float(np.std(nr_green))
            noise_mean = float(np.mean(nr_green))

    # Local contrast detection (#72): compare red channel of LED patch vs reference patch
    mute_contrast = _compute_mute_contrast(frame, _geometry)

    rr = mute_contrast.get('mute_rr') if mute_contrast else None
    re = mute_contrast.get('mute_re') if mute_contrast else None
    led_r = mute_contrast.get('mute_led_r') if mute_contrast else None
    # Combined metric: rr detects bright LED, re detects red color through tint
    # rr alone has false positives on uneven lighting; re catches red through tint
    # In dark scenes (both patches near black), rr is just noise — require minimum
    # absolute red brightness. Real mute LED has red ~60+, noise is ~10.
    rr_hit = (rr is not None and rr > _geometry.mute_contrast_threshold
              and led_r is not None and led_r >= 25)
    re_hit = re is not None and re > 10
    is_lit = rr_hit or re_hit

    # Build debug info for return_debug mode
    debug_info = None
    if return_debug:
        led_center = None
        if is_lit and mute_contrast:
            led_sx = mute_contrast.get('mute_led_sx')
            led_sy = mute_contrast.get('mute_led_sy')
            if led_sx is not None and led_sy is not None:
                led_center = (int(round(led_sx)), int(round(led_sy)))

        debug_info = {
            'region': (region_left, region_top, region_right, region_bottom),
            'method': method,
            'is_lit': is_lit,
            'led_center': led_center,
            'noise_mean': round(noise_mean, 1) if noise_mean is not None else None,
            'mute_proj': mute_proj,
        }
        if mute_contrast:
            debug_info.update(mute_contrast)

    if debug:
        # Draw corner if found
        if corner_result is not None:
            cv2.circle(debug_img, (corner_x, corner_y), 75, (0, 255, 255), 2)
            cv2.putText(debug_img, f"CORNER {match_score:.2f}",
                        (corner_x - 30, corner_y - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Draw search region boundary
        cv2.rectangle(debug_img, (region_left, region_top),
                      (region_right, region_bottom), (0, 100, 100), 2)

        # Show mute status via local contrast
        color = (0, 255, 0) if is_lit else (128, 128, 128)
        label = f"MUTE:{'ON' if is_lit else 'OFF'} ({method})"
        cv2.putText(debug_img, label,
                    (region_left, region_top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if return_debug:
            return is_lit, debug_img, debug_info
        return is_lit, debug_img

    if return_debug:
        return is_lit, None, debug_info
    return is_lit, None


def _detect_buttons(button_region):
    """
    Detect button rectangles in the button region.

    Args:
        button_region: BGR image of the button area

    Returns:
        List of (x, y, w, h) tuples for detected buttons
    """
    gray = cv2.cvtColor(button_region, cv2.COLOR_BGR2GRAY)
    bh, bw = gray.shape[:2]

    # Use adaptive thresholding to find button boundaries
    # Buttons appear as darker rectangles against lighter background
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate edges to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    buttons = []
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Filter by size and aspect ratio
        # Buttons are wider than tall, reasonable size
        aspect_ratio = w / h if h > 0 else 0
        area = w * h

        # Button criteria:
        # - Width > Height (aspect ratio > 1.2)
        # - Reasonable size (not too small, not too large)
        # - Not too close to edges
        min_area = (bw * bh) * 0.01  # At least 1% of region
        max_area = (bw * bh) * 0.15  # At most 15% of region

        if (aspect_ratio > 1.2 and aspect_ratio < 5.0 and
            area > min_area and area < max_area and
            w > _geometry.button_min_width and h > _geometry.button_min_height and
            x > _geometry.button_edge_margin and x + w < bw - _geometry.button_edge_margin and
            y > _geometry.button_edge_margin):
            buttons.append((x, y, w, h))

    # If we found too many, filter by y-position similarity (buttons should be aligned)
    if len(buttons) > 4:
        # Group buttons by similar y-position
        buttons = sorted(buttons, key=lambda b: b[1])
        # Find the most common y-level
        y_positions = [b[1] for b in buttons]
        if y_positions:
            median_y = np.median(y_positions)
            y_tolerance = bh * 0.2
            buttons = [b for b in buttons if abs(b[1] - median_y) < y_tolerance]

    # Merge overlapping buttons (Non-Maximum Suppression)
    # Buttons too close together (x within 20px) are likely double-detections
    if len(buttons) > 1:
        buttons = sorted(buttons, key=lambda b: b[0])  # Sort by x
        merged = [buttons[0]]
        for btn in buttons[1:]:
            last = merged[-1]
            if btn[0] < last[0] + last[2] + _geometry.button_nms_merge_px:
                # Merge: take union of bounding boxes
                new_x = min(last[0], btn[0])
                new_y = min(last[1], btn[1])
                new_x2 = max(last[0] + last[2], btn[0] + btn[2])
                new_y2 = max(last[1] + last[3], btn[1] + btn[3])
                merged[-1] = (new_x, new_y, new_x2 - new_x, new_y2 - new_y)
            else:
                merged.append(btn)
        buttons = merged

    return buttons


def _find_led_in_button(button_region, button_rect):
    """Find LED dot center within a button rectangle.

    Searches the right portion of the button for a circular LED dot.
    Tries dark blob detection first (shape-based, reliable for unlit LEDs),
    then falls back to blue mask for lit LEDs (stricter criteria).

    Args:
        button_region: BGR image of the button search area
        button_rect: (x, y, w, h) of button in button_region coords

    Returns:
        (cx, cy, method) where method is 'dark' or 'lit', or None if not found.
    """
    x, y, w, h = button_rect
    brh, brw = button_region.shape[:2]
    # Search right portion of button (LED is on the right side)
    # Start at w//2 to skip button text labels (e.g. "S2")
    margin = max(2, w // 8)
    rx = max(0, x + w // 2)
    # Vertical padding to avoid edge artifacts from button border
    pad_y = max(2, h // 6)
    cy1 = max(0, y + pad_y)
    cy2 = min(brh, y + h - pad_y)
    crop = button_region[cy1:cy2, rx:min(brw, x + w + margin)]
    if crop.size == 0:
        return None

    # Try dark dot first (threshold + connectedComponents)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    ch, cw = gray.shape[:2]
    _, dark_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask)
    if nlabels > 1:
        edge_margin = 4
        best_dot = None
        best_score = -1
        for lbl in range(1, nlabels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            bw2 = stats[lbl, cv2.CC_STAT_WIDTH]
            bh2 = stats[lbl, cv2.CC_STAT_HEIGHT]
            cx2, cy2 = centroids[lbl]
            # Area filter: LED dot should be ~20-150 px
            if area < 15 or area > 200:
                continue
            # Aspect ratio: should be roughly circular (0.5-2.0)
            aspect = bw2 / max(1, bh2)
            if aspect < 0.5 or aspect > 2.0:
                continue
            # Reject blobs touching crop edge
            if cx2 < edge_margin or cx2 > cw - edge_margin:
                continue
            if cy2 < edge_margin or cy2 > ch - edge_margin:
                continue
            # Compactness: area / bounding box area (circle ~ 0.78)
            bbox_area = bw2 * bh2
            compactness = area / max(1, bbox_area)
            if compactness < 0.4:
                continue
            # Score: prefer larger, rounder dots
            score = area * compactness
            if score > best_score:
                best_score = score
                best_dot = (int(rx + cx2), int(cy1 + cy2))
        if best_dot is not None:
            return (best_dot[0], best_dot[1], 'dark')

    # Fallback: lit LED (blue blob) — dark blob fails when LED is lit
    led_mask = _create_led_mask(crop)
    blue_px = cv2.countNonZero(led_mask)
    if blue_px >= 5:
        # In dark areas, a lit LED is unmistakably bright — reject dim glow
        # In bright areas, the LED just looks blue (may not be brighter)
        crop_mean = gray.mean()
        if crop_mean < 100:
            blue_brightness = gray[led_mask > 0].mean()
            if blue_brightness < 150:
                return None  # dim glow in dark area, not a real LED
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(led_mask)
        if nlabels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            best = 1 + np.argmax(areas)
            area = areas[best - 1]
            if area >= 5:
                cx, cy = centroids[best]
                return (int(rx + cx), int(cy1 + cy), 'lit')

    return None


def _create_led_mask(button_region):
    """
    Create a binary mask of potential LED pixels (blue or bright white).

    Args:
        button_region: BGR image of the button area

    Returns:
        Binary mask where LED pixels are white
    """
    hsv = cv2.cvtColor(button_region, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(button_region, cv2.COLOR_BGR2GRAY)

    # Detect blue/cyan LED pixels (high saturation to exclude display glow)
    # LED: S>200, Display: S~30 - use S>=150 to separate
    lower_blue = np.array([85, 150, 80])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Detect bright white pixels (overexposed LED)
    bright_mask = (gray > 215).astype(np.uint8) * 255

    # Combine masks
    led_mask = cv2.bitwise_or(blue_mask, bright_mask)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    led_mask = cv2.morphologyEx(led_mask, cv2.MORPH_OPEN, kernel)

    return led_mask


def get_blue_mask(image, tight=False, very_tight=False):
    """Get binary mask of blue LED pixels in an image.

    Args:
        image: BGR image
        tight: If True, use stricter thresholds to reduce glow effects
        very_tight: If True, use very strict thresholds for glowing images
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if very_tight:
        # Very strict thresholds for glowing/bright images
        # S>=200 and V>=240 to only get the brightest core pixels
        lower_blue = np.array([85, 200, 240])
        upper_blue = np.array([125, 255, 255])
    elif tight:
        # Stricter thresholds - only bright core of LEDs
        # S>=95 to allow slight saturation variation, V>=200 to reject glow
        lower_blue = np.array([85, 95, 200])
        upper_blue = np.array([115, 255, 255])
    else:
        # Standard range for blue LEDs
        lower_blue = np.array([85, 100, 100])
        upper_blue = np.array([115, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if tight and not very_tight:
        # Erode to shrink further
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask


def preprocess_glowing_image(image):
    """Preprocess a glowing/bright image to reduce glow artifacts.

    Converts to greyscale, reduces brightness, and enhances contrast.
    This helps separate the actual lit segments from the glow.

    Args:
        image: BGR image with excessive glow

    Returns:
        processed: BGR image with reduced glow
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # For glowing images, use a simple high threshold to keep only brightest pixels
    # This separates the actual lit segments from the glow
    _, thresholded = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Slight erosion to separate close segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresholded = cv2.erode(thresholded, kernel, iterations=1)

    # Convert back to BGR (as cyan to preserve blue detection)
    processed = np.zeros_like(image)
    processed[:, :, 0] = thresholded  # Blue channel
    processed[:, :, 1] = thresholded  # Green channel
    processed[:, :, 2] = 0  # No red

    return processed


def correct_slant(panel_img, angle=None):
    """
    Correct the slant of digits using affine shear transform.

    Args:
        panel_img: BGR image of the panel region
        angle: Slant angle in degrees (if None, will be estimated)
               Positive = digits lean right, negative = lean left

    Returns:
        corrected_img: De-skewed image
        angle: The slant angle used
        debug_img: Debug visualization
    """
    # Always use fixed slant angle of 8.0 degrees for stability
    if angle is None:
        angle = 8.0

    h, w = panel_img.shape[:2]
    debug_img = panel_img.copy()

    # Draw detected slant line on debug image
    cx, cy = w // 2, h // 2
    line_len = min(w, h) // 3
    dx = int(line_len * np.sin(np.radians(angle)))
    dy = int(line_len * np.cos(np.radians(angle)))
    cv2.line(debug_img, (cx - dx, cy + dy), (cx + dx, cy - dy), (0, 0, 255), 2)
    cv2.putText(debug_img, f"Angle: {angle:.1f} deg",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # If angle is very small, no correction needed
    if abs(angle) < 1.0:
        return panel_img.copy(), angle, debug_img

    # Calculate shear factor from angle
    # If digits lean right (positive angle), top is right of bottom
    # To straighten: move top LEFT, bottom RIGHT
    # Shear transforms: x' = x + shear * (y - center_y)
    # At y=0 (top): x' = x - shear*h/2 (moves left if shear > 0)
    # At y=h (bottom): x' = x + shear*h/2 (moves right if shear > 0)
    # So positive shear corrects rightward lean
    shear = np.tan(np.radians(angle))

    # Calculate extra width needed
    extra_w = int(abs(shear) * h)
    new_w = w + extra_w

    # Build affine transform matrix for shear around vertical center
    # We want x' = x + shear * (y - h/2) + offset
    # Which is: x' = x + shear*y - shear*h/2 + offset
    center_y = h / 2

    # Offset to keep image centered in new canvas
    offset_x = extra_w / 2

    M = np.array([
        [1, shear, -shear * center_y + offset_x],
        [0, 1, 0]
    ], dtype=np.float32)

    # Apply shear transform
    corrected_img = cv2.warpAffine(panel_img, M, (new_w, h),
                                    borderMode=cv2.BORDER_REPLICATE)

    # Crop invalid triangular edges from slant correction
    # Shear creates invalid triangles: bottom-left and top-right
    # Crop aggressively from both sides
    crop_left = extra_w + 2
    crop_right = extra_w + 2
    corrected_img = corrected_img[:, crop_left:-crop_right]

    return corrected_img, angle, debug_img


def recognize_digit(digit_img, debug=False, **_kwargs):
    """
    Recognize a single digit using template matching.

    Thin wrapper around recognize_digit_template for backward compatibility.

    Args:
        digit_img: BGR image of a single digit box
        debug: If True, return debug image with match annotation

    Returns:
        digit: Recognized character ('0'-'9', 'P') or 'X' if unknown
        debug_img: (only if debug=True) Annotated image, else None
    """
    digit, score = recognize_digit_template(digit_img)
    if debug:
        h, w = digit_img.shape[:2]
        debug_img = digit_img.copy()
        cv2.putText(debug_img, f'{digit}({score:.2f})', (5, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return digit, debug_img
    return digit, None


def find_digit_gap(corrected_img, debug=False):
    """
    Step 3-1: Find the gap between the two digits.

    Uses brightness histogram (column sum of grayscale) and searches from
    center outward for the first local minimum (U-shape valley).

    Args:
        corrected_img: De-skewed panel image
        debug: If True, return debug image showing gap detection

    Returns:
        gap_x: X-coordinate of the gap center
        debug_img: (only if debug=True) Image showing column projection and gap
    """
    h, w = corrected_img.shape[:2]

    # Use grayscale brightness for gap detection
    gray = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
    col_sums = np.sum(gray, axis=0).astype(np.float64)

    # Smooth to reduce noise
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(col_sums, kernel, mode='same')

    # Search from center to find valley bottom (U-shape)
    center = len(smoothed) // 2
    search_limit = int(len(smoothed) * 0.15)  # Don't go beyond 35%-65% range

    def _find_valley(start, direction, limit):
        """Search in one direction for a real valley (local minimum).
        Returns (x, value, is_valley) where is_valley indicates a true local minimum."""
        best_x = start
        for x in range(start + direction, start + direction * limit, direction):
            if x <= 0 or x >= len(smoothed) - 1:
                break
            if smoothed[x] <= smoothed[x - 1] and smoothed[x] <= smoothed[x + 1] and (smoothed[x] < smoothed[x - 1] or smoothed[x] < smoothed[x + 1]):
                # Found local minimum - real valley (handles flat-bottom valleys)
                return x, smoothed[x], True
            if smoothed[x] < smoothed[best_x]:
                best_x = x
        return best_x, smoothed[best_x], False

    # Check if a peak (local max) is nearby center
    peak_range = max(3, kernel_size)
    left_bound = max(1, center - peak_range)
    right_bound = min(len(smoothed) - 2, center + peak_range)
    peak_nearby = any(
        smoothed[x] > smoothed[x - 1] and smoothed[x] > smoothed[x + 1]
        for x in range(left_bound, right_bound + 1)
    )

    if peak_nearby:
        # Peak nearby: search both sides, pick deeper real valley
        left_x, left_val, left_real = _find_valley(center, -1, search_limit)
        right_x, right_val, right_real = _find_valley(center, 1, search_limit)
        if left_real and right_real:
            gap_x = left_x if left_val <= right_val else right_x
        elif left_real:
            gap_x = left_x
        elif right_real:
            gap_x = right_x
        else:
            gap_x = center  # No real valley found, use center
    elif smoothed[center] <= smoothed[center - 1] and smoothed[center] <= smoothed[center + 1]:
        # Center is already at valley bottom
        gap_x = center
    else:
        # Follow slope direction to find valley bottom
        if smoothed[center - 1] < smoothed[center + 1]:
            x, _, found = _find_valley(center, -1, search_limit)
        else:
            x, _, found = _find_valley(center, 1, search_limit)
        gap_x = x if found else center

    if debug:
        # Create debug visualization
        debug_img = corrected_img.copy()

        # Draw column projection as a graph at the bottom
        proj_height = 60
        max_sum = smoothed.max() if smoothed.max() > 0 else 1

        # Draw projection bars (smoothed brightness histogram)
        for x in range(w):
            bar_height = int((smoothed[x] / max_sum) * (proj_height - 5))
            if bar_height > 0:
                cv2.line(debug_img, (x, h - 5), (x, h - 5 - bar_height), (0, 255, 0), 1)

        # Draw center line (gray)
        cv2.line(debug_img, (center, h - proj_height), (center, h), (100, 100, 100), 1)

        # Draw gap line (yellow, full height, 50% transparent)
        line_layer = debug_img.copy()
        cv2.line(line_layer, (gap_x, 0), (gap_x, h), (0, 255, 255), 2)
        cv2.addWeighted(line_layer, 0.5, debug_img, 0.5, 0, dst=debug_img)

        # Add label
        cv2.putText(debug_img, f"Gap: x={gap_x}", (gap_x + 5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return gap_x, debug_img

    return gap_x, None


def define_digit_boxes(corrected_img, gap_x, debug=False):
    """
    Step 3-2: Define digit boxes based on the gap position.

    Creates boxes constrained to their respective sides of the gap.
    Left box: from left edge to gap_x
    Right box: from gap_x to right edge

    Args:
        corrected_img: De-skewed panel image
        gap_x: X-coordinate of the gap between digits
        debug: If True, return debug image showing boxes

    Returns:
        left_box: (x, y, w, h) of left digit box
        right_box: (x, y, w, h) of right digit box
        debug_img: (only if debug=True) Image showing boxes
    """
    h, w = corrected_img.shape[:2]

    # Check if this is a glowing image
    gray = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
    mean_gray = np.mean(gray)
    std_gray = np.std(gray)
    very_bright_ratio = np.sum(gray > 200) / gray.size
    # Glowing detection: high mean brightness OR (high std AND some bright pixels)
    is_glowing = mean_gray > 55 or (std_gray > 50 and very_bright_ratio > 0.03)

    # Use blue mask for accurate digit bounds (avoids glow artifacts)
    # For glowing images, use preprocessed mask to reduce glow
    if is_glowing:
        preprocessed = preprocess_glowing_image(corrected_img)
        binary = get_blue_mask(preprocessed, tight=False)
    else:
        binary = get_blue_mask(corrected_img, tight=True)

    # If tight mask has too few pixels, try loose mask
    if np.sum(binary > 0) < 50:
        binary = get_blue_mask(corrected_img, tight=False)

    # Find bounds for LEFT digit (left of gap)
    left_region = binary[:, :gap_x]
    left_pixels = np.where(left_region > 0)

    if len(left_pixels[0]) > 0:
        left_top = left_pixels[0].min()
        left_bottom = left_pixels[0].max()
        left_left = left_pixels[1].min()
        left_right = left_pixels[1].max()
    else:
        left_top, left_bottom = 0, h
        left_left, left_right = 0, gap_x

    # Find bounds for RIGHT digit (right of gap)
    right_region = binary[:, gap_x:]
    right_pixels = np.where(right_region > 0)

    if len(right_pixels[0]) > 0:
        right_top = right_pixels[0].min()
        right_bottom = right_pixels[0].max()
        right_left = right_pixels[1].min() + gap_x  # Offset by gap_x
        right_right = right_pixels[1].max() + gap_x
    else:
        right_top, right_bottom = 0, h
        right_left, right_right = gap_x, w

    # Calculate each digit's dimensions
    left_w = left_right - left_left
    left_h = left_bottom - left_top
    right_w = right_right - right_left
    right_h = right_bottom - right_top

    # Use MAX height for both (same height for proper segment detection)
    box_h = max(left_h, right_h)

    # Add vertical padding (20% of height)
    v_pad = int(box_h * 0.2)
    box_h += 2 * v_pad

    # Use same y position for both (aligned)
    box_y = min(left_top, right_top) - v_pad
    box_y = max(0, box_y)
    if box_y + box_h > h:
        box_h = h - box_y

    # Use MAX width for both boxes (same size for consistency)
    box_w = max(left_w, right_w)

    # Add horizontal padding (25% of width)
    h_pad = int(box_w * 0.25)
    box_w += 2 * h_pad

    # Ensure minimum width is at least height / 2.2
    min_width = int(box_h / 2.2)
    box_w = max(box_w, min_width)

    margin = 2  # Small margin from gap

    # Left box: right-aligned to gap (ends at gap - margin)
    # Content is right-aligned within box (narrow "1" will be near gap)
    left_box_x = gap_x - margin - box_w
    left_box_x = max(0, left_box_x)
    # Adjust if box would extend past left edge
    if left_box_x == 0:
        box_w_left = gap_x - margin
    else:
        box_w_left = box_w

    # Right box: left-aligned to gap (starts at gap + margin)
    # This ensures we capture the full left side of the right digit
    right_box_x = gap_x + margin
    # Ensure box doesn't go past right edge
    if right_box_x + box_w > w:
        box_w_right = w - right_box_x
    else:
        box_w_right = box_w

    # Final box width: use the smaller to ensure same size
    final_box_w = min(box_w_left, box_w_right)

    # Recalculate left box x with final width
    left_box_x = gap_x - margin - final_box_w
    left_box_x = max(0, left_box_x)

    left_box = (int(left_box_x), int(box_y), int(final_box_w), int(box_h))
    right_box = (int(right_box_x), int(box_y), int(final_box_w), int(box_h))

    if debug:
        debug_img = corrected_img.copy()

        # Draw left box (green)
        lx, ly, lw, lh = left_box
        cv2.rectangle(debug_img, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 2)
        cv2.putText(debug_img, "L", (lx + 5, ly + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw right box (blue)
        rx, ry, rw, rh = right_box
        cv2.rectangle(debug_img, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
        cv2.putText(debug_img, "R", (rx + 5, ry + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Draw gap line (yellow, 50% transparent)
        line_layer = debug_img.copy()
        cv2.line(line_layer, (gap_x, 0), (gap_x, h), (0, 255, 255), 1)
        cv2.addWeighted(line_layer, 0.5, debug_img, 0.5, 0, dst=debug_img)

        return left_box, right_box, debug_img

    return left_box, right_box, None


class SegmentReader:
    """
    Step 4: Adaptive caching for efficient frame processing.

    Caches panel detection and slant angle to avoid recomputing on every frame.
    Only updates cache when scene changes significantly.
    """

    def __init__(self):
        """Initialize SegmentReader with empty cache state."""

        # Cached values
        self._panel_rect = None
        self._gap_x = None  # Gap position between digits
        self._left_box = None  # Left digit bounding box
        self._right_box = None  # Right digit bounding box
        self._last_reading = None  # Last successful reading
        self._last_scores = (0.0, 0.0)  # Last match scores (left, right)
        self._last_second = (('X', 0.0), ('X', 0.0))  # Second best candidates ((digit, score), (digit, score))
        self._last_digit_debug = None  # Debug info for digit matching
        self._detection_method = None  # Panel detection method used
        self._dim_enhanced = None  # Dim digit enhancement status (L/R/LR/None)

        # Cached detection results for washout overlay
        self._last_led_debug = None   # Last non-washout LED debug info
        self._last_mute_debug = None  # Last non-washout mute debug info

        # Frame diff optimization: skip processing if frame unchanged
        self._prev_frame_roi = None  # Previous frame ROI for diff comparison
        self._prev_reading = None  # Previous reading to reuse
        self._prev_panel_rect = None  # Previous panel rect
        self._frame_skipped = False  # Whether current frame was skipped
        self._frame_diff_threshold = 100000  # Diff threshold for skip (3-channel)
        self._frame_diff_edge = None  # Diff value for monitoring

        # Pending issue for deferred logging (allows caller to add display frame)
        self._pending_issue = None  # (issue_type, confidence, extra_info)

        # Quick-check optimization: track best templates for 1st and 2nd candidates
        # Format: ((1st_digit, 1st_template_idx, 1st_score), (2nd_digit, 2nd_template_idx, 2nd_score))
        self._left_best_templates = None  # For left digit position
        self._right_best_templates = None  # For right digit position
        self._last_full_scan = 0  # Timestamp of last full template scan
        self._full_scan_interval = 180  # Force full scan every 3 minutes

        # Load cache from unified cache file
        self.load_cache()

    def save_cache(self):
        """Save current cache to unified cache file."""
        panel_data = {
            'panel_rect': _to_native(self._panel_rect),
            'gap_x': _to_native(self._gap_x),
            'left_box': _to_native(self._left_box),
            'right_box': _to_native(self._right_box),
            'last_reading': self._last_reading
        }
        _save_cache(panel_data=panel_data)

    def load_cache(self):
        """Load cache from unified cache file if it exists."""
        cache_data = _load_panel_from_cache()
        if cache_data is None:
            return False

        try:
            self._panel_rect = tuple(cache_data['panel_rect']) if cache_data.get('panel_rect') else None
            self._gap_x = cache_data.get('gap_x')
            self._left_box = tuple(cache_data['left_box']) if cache_data.get('left_box') else None
            self._right_box = tuple(cache_data['right_box']) if cache_data.get('right_box') else None
            self._last_reading = cache_data.get('last_reading')
            return True
        except (KeyError, TypeError):
            return False

    def _update_cache(self, frame):
        """
        Re-detect panel, update cache.

        Args:
            frame: BGR frame to analyze

        Returns:
            True if detection successful
        """
        panel_rect, _ = detect_panel(frame)

        if panel_rect is None:
            return False

        x, y, w, h = panel_rect
        panel_img = _geometry.undistort_roi(frame, x, y, w, h, derotate=True)

        # Compute boxes (slant is always fixed at 8.0 degrees)
        corrected_img, _, _ = correct_slant(panel_img, 8.0)
        gap_x, _ = find_digit_gap(corrected_img)
        left_box, right_box, _ = define_digit_boxes(corrected_img, gap_x)

        # Update cache
        self._panel_rect = panel_rect
        self._gap_x = gap_x
        self._left_box = left_box
        self._right_box = right_box
        # Persist cache to file
        self.save_cache()

        return True

    def _quick_check_digit(self, digit_img, gray, best_templates, force_full):
        """Quick-check a single digit against cached best 2 templates.
        Returns (digit, score, debug_info)."""
        if force_full or best_templates is None:
            return recognize_digit_template(digit_img, return_debug=True)

        (d1, idx1, score1), (d2, idx2, score2) = best_templates
        new_score1, match_pos1, template_size1 = match_single_template(gray, d1, idx1)
        new_score2, _, _ = match_single_template(gray, d2, idx2)

        if score1 - new_score1 > _QUICKCHECK_DRIFT or abs(new_score2 - score2) > _QUICKCHECK_DRIFT:
            return recognize_digit_template(digit_img, return_debug=True)

        return d1, new_score1, {
            'second_digit': d2, 'second_score': new_score2,
            'best_template_idx': idx1, 'second_template_idx': idx2,
            'match_pos': match_pos1, 'template_size': template_size1,
        }

    def read(self, frame, debug=False):
        """
        Read the 2-digit value from frame - all fresh detection, no caching.

        Args:
            frame: BGR image from camera/file
            debug: If True, call find_digit_gap/define_digit_boxes with debug=True
                   and store gap_debug/boxes_debug in digit_debug dict

        Returns:
            reading: 2-character string (e.g., "10", "PP", "XX" if detection fails)
            cache_hit: Always False (no caching)
        """
        # Handle invalid frame
        if frame is None or frame.size == 0:
            return "XX", False

        self._frame_skipped = False  # Reset skip flag
        self._frame_diff_edge = None  # Reset edge case tracking

        # Frame diff optimization: skip if frame ROI unchanged from reference
        # Compare to reference frame (not previous) to avoid drift
        roi_y1, roi_y2, roi_x1, roi_x2 = _geometry.get_frame_diff_roi()
        h_frame, w_frame = frame.shape[:2]
        if roi_y2 <= h_frame and roi_x2 <= w_frame:
            current_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]  # 3-channel
            if self._prev_frame_roi is not None and self._prev_reading is not None:
                if current_roi.shape == self._prev_frame_roi.shape:
                    pixel_diff = np.abs(current_roi.astype(np.int16) - self._prev_frame_roi.astype(np.int16))
                    # Ignore noise (0-4), only count significant changes
                    diff = np.sum(pixel_diff[pixel_diff >= 5])
                    self._frame_diff_edge = diff

                    if diff < self._frame_diff_threshold:
                        # Frame unchanged from reference, reuse reading
                        self._frame_skipped = True
                        return self._prev_reading, False
                    else:
                        # Diff exceeded threshold: update reference to current frame
                        self._prev_frame_roi = current_roi.copy()

        # Always detect panel fresh
        panel_rect, detection_method = detect_panel(frame)
        self._detection_method = detection_method  # Store for logging
        if panel_rect is None:
            nm = get_noise_mean(frame)
            if nm is None or nm <= 180:  # Skip during washout
                log_issue_frame(frame, 'panel_fail')
            return "XX", False

        x, y, w, h = panel_rect
        panel_img = _geometry.undistort_roi(frame, x, y, w, h, derotate=True)

        # Process with fixed 8.0 degree slant
        corrected_img, _, slant_debug_img = correct_slant(panel_img, 8.0)
        gap_x, gap_debug = find_digit_gap(corrected_img, debug=debug)
        left_box, right_box, boxes_debug = define_digit_boxes(corrected_img, gap_x, debug=debug)

        left_digit_img = _extract_digit_with_padding(corrected_img, left_box, right_bound=gap_x)
        right_digit_img = _extract_digit_with_padding(corrected_img, right_box, left_bound=gap_x)

        # Check if periodic full scan is needed (every 3 minutes)
        current_time = time.time()
        force_full_scan = (current_time - self._last_full_scan) >= self._full_scan_interval

        # Convert images to grayscale with dim enhancement if needed
        left_gray, left_enhanced = _enhance_dim_digit(left_digit_img)
        right_gray, right_enhanced = _enhance_dim_digit(right_digit_img)

        # Track enhancement status: L, R, LR, or None
        if left_enhanced and right_enhanced:
            self._dim_enhanced = 'LR'
        elif left_enhanced:
            self._dim_enhanced = 'L'
        elif right_enhanced:
            self._dim_enhanced = 'R'
        else:
            self._dim_enhanced = None

        # Quick-check optimization: check only the best 2 templates first
        # Full search triggered if score drifts > _QUICKCHECK_DRIFT or periodic rescan
        left_digit, left_score, left_debug = self._quick_check_digit(
            left_digit_img, left_gray, self._left_best_templates, force_full_scan)
        right_digit, right_score, right_debug = self._quick_check_digit(
            right_digit_img, right_gray, self._right_best_templates, force_full_scan)

        # Update last full scan timestamp if we did a full scan
        if force_full_scan:
            self._last_full_scan = current_time

        reading = left_digit + right_digit

        # Update best templates for next frame
        if left_debug:
            self._left_best_templates = (
                (left_digit, left_debug.get('best_template_idx', 0), left_score),
                (left_debug.get('second_digit', 'X'), left_debug.get('second_template_idx', 0), left_debug.get('second_score', 0.0)),
            )
        if right_debug:
            self._right_best_templates = (
                (right_digit, right_debug.get('best_template_idx', 0), right_score),
                (right_debug.get('second_digit', 'X'), right_debug.get('second_template_idx', 0), right_debug.get('second_score', 0.0)),
            )

        # Store for display and cache
        self._panel_rect = panel_rect
        self._gap_x = gap_x
        self._left_box = left_box
        self._right_box = right_box
        # Store raw digits for display (use rejected_digit if available, for showing actual detection)
        raw_left = left_debug.get('rejected_digit', left_digit) if left_debug and left_digit == 'X' else left_digit
        raw_right = right_debug.get('rejected_digit', right_digit) if right_debug and right_digit == 'X' else right_digit
        self._raw_digits = (raw_left, raw_right)
        self._last_reading = reading
        self._last_scores = (left_score, right_score)
        self._last_second = (
            (left_debug.get('second_digit', 'X'), left_debug.get('second_score', 0.0)) if left_debug else ('X', 0.0),
            (right_debug.get('second_digit', 'X'), right_debug.get('second_score', 0.0)) if right_debug else ('X', 0.0),
        )
        self._last_digit_debug = {
            'left_box': left_box,
            'right_box': right_box,
            'left_match': left_debug,
            'right_match': right_debug,
            'corrected_size': (corrected_img.shape[1], corrected_img.shape[0]),
            'gap_x': gap_x,
            'left_img': left_digit_img,
            'right_img': right_digit_img,
            'corrected_img': corrected_img,  # For gap debug visualization
            'gap_debug': gap_debug,           # Only populated when debug=True
            'boxes_debug': boxes_debug,       # Only populated when debug=True
            'slant_debug': slant_debug_img,   # Slant correction debug image
        }

        # Save to unified cache file
        self.save_cache()

        # Track issues for deferred logging (caller can add display frame)
        self._pending_issue = None

        # Check for low confidence
        if left_score < 0.7 or right_score < 0.7:
            self._pending_issue = ('low_conf', min(left_score, right_score), reading)

        # Check for ambiguous recognition (1st < 95% AND 2nd within 5%)
        left_2nd_score = left_debug.get('second_score', 0) if left_debug else 0
        right_2nd_score = right_debug.get('second_score', 0) if right_debug else 0

        left_ambiguous = left_score < _AMBIGUOUS_MAX_SCORE and (left_score - left_2nd_score) < _TEMPLATE_AMBIGUITY_GAP
        right_ambiguous = right_score < _AMBIGUOUS_MAX_SCORE and (right_score - right_2nd_score) < _TEMPLATE_AMBIGUITY_GAP

        if left_ambiguous or right_ambiguous:
            gap = min(left_score - left_2nd_score, right_score - right_2nd_score)
            self._pending_issue = ('ambiguous', min(left_score, right_score), f'{reading}_gap{gap:.2f}')

        # PP always appears as a pair - if single P detected with high confidence, report PP
        if reading != 'PP':
            if left_digit == 'P' and left_score >= 0.85:
                reading = 'PP'
            elif right_digit == 'P' and right_score >= 0.85:
                reading = 'PP'

        # Check for invalid reading (outside 00-66, PP range)
        # Invalid readings (like "88") are transitional frames - report as "XX"
        if not _is_valid_reading(reading):
            # Only log if not a display transition (previous reading was same)
            if self._prev_reading is not None and self._prev_reading == reading:
                self._pending_issue = ('invalid_reading', min(left_score, right_score), reading)
            reading = 'XX'

        # Store reading for frame diff optimization
        old_reading = self._prev_reading
        self._prev_reading = reading

        # Initialize reference ROI on first frame (subsequent updates happen when diff exceeds threshold)
        roi_y1, roi_y2, roi_x1, roi_x2 = _geometry.get_frame_diff_roi()
        h_frame, w_frame = frame.shape[:2]

        if roi_y2 <= h_frame and roi_x2 <= w_frame:
            if self._prev_frame_roi is None:
                self._prev_frame_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()

        return reading, False

    def detect(self, frame, debug=False):
        """Full single-frame detection: digits + corner + LED + mute.

        Args:
            frame: BGR image (640x480)
            debug: If True, generate step-by-step debug images in digit_debug

        Returns:
            FrameResult with all detection outputs.
        """
        # 1. Digits (existing read() logic, with debug passthrough)
        reading, cache_hit = self.read(frame, debug=debug)

        # Recompute LED dots on frame-skipped frames (predict_panel didn't run)
        if self._frame_skipped:
            _refresh_led_dots(frame)

        # 2. Corner — reuse from predict_panel_from_landmarks() cache
        corner_result = _frame_corner_result
        corner_debug = _frame_corner_debug
        if corner_result is None and not self._frame_skipped:
            # calibrated fallback path — predict_panel didn't find corner
            corner_raw = _find_corner(frame, return_debug=True)
            if corner_raw:
                corner_result = corner_raw[0]
                corner_debug = corner_raw[1]

        # 3. Washout
        noise_mean = get_noise_mean(frame)
        washout = noise_mean is not None and noise_mean > 180

        # 4. LED (skip if washout)
        if washout:
            led_status = "NA"
            led_debug_info = None
        elif self._frame_skipped and _frame_led_dots:
            # On skipped frames, _refresh_led_dots() already detected all 4 buttons
            lit_name = _frame_led_dots.get('_lit')
            led_status = lit_name if lit_name else "NA"
            # Reuse previous debug info but update led_method and dots
            led_debug_info = dict(self._last_led_debug) if self._last_led_debug else {}
            led_debug_info['led_method'] = 'landmark_dot'
            led_debug_info['lit_led'] = lit_name
            led_debug_info['led_dots'] = dict(_frame_led_dots)
        else:
            try:
                leds, _, led_debug_info = detect_button_leds(
                    frame, self._panel_rect, return_debug=True,
                    detection_method=self._detection_method)
                lit = [k for k, v in leds.items() if v]
                led_status = lit[0] if lit else "NA"
                self._last_led_debug = led_debug_info
            except Exception as e:
                print(f"Error in LED detection: {e}", flush=True)
                led_status = "NA"
                led_debug_info = None

        # 5. Mute (skip if washout)
        if washout:
            mute_status = "MUTE_NA"
            mute_debug_info = None
        else:
            valid_corner = corner_result if (corner_result and corner_result[0] is not None) else None
            try:
                is_muted, _, mute_debug_info = detect_red_button(
                    frame, return_debug=True, corner_result=valid_corner)
                mute_status = "MUTE" if is_muted else "UNMUTE"
                self._last_mute_debug = mute_debug_info
            except Exception as e:
                print(f"Error in MUTE detection: {e}", flush=True)
                mute_status = "UNMUTE"
                mute_debug_info = None

        return FrameResult(
            reading=reading,
            cache_hit=cache_hit,
            led_status=led_status,
            mute_status=mute_status,
            corner_result=corner_result,
            corner_debug=corner_debug,
            led_debug_info=led_debug_info,
            mute_debug_info=mute_debug_info,
            noise_mean=noise_mean,
            washout=washout,
            panel_rect=self._panel_rect,
            detection_method=self._detection_method,
            last_led_debug=self._last_led_debug,
            last_mute_debug=self._last_mute_debug,
        )

    def reset_cache(self, keep_last_reading=False):
        """
        Clear cached values, forcing re-detection on next read.

        Args:
            keep_last_reading: If True, preserve the last successful reading
        """
        self._panel_rect = None
        self._gap_x = None
        self._left_box = None
        self._right_box = None
        # Reset quick-check templates (forces full search on next frame)
        self._left_best_templates = None
        self._right_best_templates = None
        if not keep_last_reading:
            self._last_reading = None

    @property
    def panel_rect(self):
        """Get cached panel rectangle (x, y, w, h) or None."""
        return self._panel_rect

    @property
    def pending_issue(self):
        """Get pending issue tuple (issue_type, confidence, extra_info) or None."""
        return self._pending_issue

    def clear_pending_issue(self):
        """Clear pending issue after logging."""
        self._pending_issue = None

    @property
    def slant_angle(self):
        """Get fixed slant angle (always 8.0 degrees)."""
        return 8.0

    @property
    def gap_x(self):
        """Get detected gap X position within panel, or None."""
        return self._gap_x

    @property
    def last_reading(self):
        """Get last successful reading or None."""
        return self._last_reading

    @property
    def last_scores(self):
        """Get last match scores (left, right) as tuple of floats 0.0-1.0."""
        return self._last_scores

    @property
    def raw_digits(self):
        """Get raw detected digits (left, right) before PP/XX conversion."""
        return getattr(self, '_raw_digits', ('X', 'X'))

    @property
    def last_second(self):
        """Get second best candidates ((left_digit, left_score), (right_digit, right_score))."""
        return self._last_second

    @property
    def digit_debug(self):
        """Get last digit matching debug info."""
        return self._last_digit_debug

    @property
    def detection_method(self):
        """Get last panel detection method used."""
        return getattr(self, '_detection_method', None)

    @property
    def dim_enhanced(self):
        """Get dim digit enhancement status: 'L', 'R', 'LR', or None."""
        return getattr(self, '_dim_enhanced', None)

    @property
    def frame_skipped(self):
        """Get whether frame was skipped due to unchanged content."""
        return getattr(self, '_frame_skipped', False)

    @property
    def frame_diff_edge(self):
        """Get frame diff value when near threshold (150K-300K), else None."""
        return getattr(self, '_frame_diff_edge', None)

    @property
    def geo_method(self):
        """Get geometry projection method: 'homography', 'offset', or 'none'."""
        return _geometry._geo_method

    @property
    def geo_scale(self):
        """Get similarity transform scale factor, or None if no homography."""
        return _geometry._scale if _geometry._homography is not None else None

    @property
    def geo_rotation(self):
        """Get rotation angle from similarity transform in degrees, or None."""
        return _geometry.get_rotation_deg() if _geometry._homography is not None else None

    @property
    def undistorted(self):
        """Get max pixel shift from undistortion in panel ROI, or 0."""
        if not _geometry.has_intrinsics() or self._panel_rect is None:
            return 0.0
        x, y, w, h = self._panel_rect
        return _geometry.get_undistort_shift(x, y, w, h)


def draw_display_overlay(frame, panel_rect, corrected_img, gap_x,
                         left_digit_img, right_digit_img,
                         left_digit, right_digit, left_score, right_score,
                         left_match, right_match,
                         left_second, left_second_score,
                         right_second, right_second_score,
                         reading, led_status, mute_status,
                         corner_debug=None, corner_score=None,
                         led_debug_info=None,
                         mute_debug_info=None, frame_skipped=False,
                         washout=False):
    """Draw full debug overlay on frame (same layout as live_demo --display).

    Args:
        frame: Raw BGR frame (not modified; a copy is used)
        panel_rect: (x, y, w, h) of the panel
        corrected_img: Slant-corrected panel image
        gap_x: Gap position in corrected image
        left_digit_img, right_digit_img: Extracted digit images with padding
        left_digit, right_digit: Recognized digit characters
        left_score, right_score: Match confidence scores
        left_match, right_match: Debug dicts from recognize_digit_template
        left_second, left_second_score: 2nd best digit and score (left)
        right_second, right_second_score: 2nd best digit and score (right)
        reading: Combined reading string (e.g. "27")
        led_status: LED status string (e.g. "B2")
        mute_status: Mute status string (e.g. "UNMUTE")
        corner_debug: Corner debug info dict (optional)
        corner_score: Corner match score (optional)
        led_debug_info: LED debug info dict (optional)
        mute_debug_info: Mute debug info dict (optional)

    Returns:
        overlay: BGR image with all debug overlays drawn
    """
    overlay = frame.copy()
    frame_h, frame_w = overlay.shape[:2]
    x, y, w, h = panel_rect

    # Panel rectangle (green; dashed when frame was skipped)
    if frame_skipped:
        _draw_dashed_rect(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2, dash_length=10)
    else:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Corner debug, LED zones, MUTE zone
    if corner_debug:
        draw_corner_debug(overlay, corner_debug, corner_score=corner_score)
    if led_debug_info:
        draw_led_debug(overlay, led_debug_info, dashed=washout)
    if mute_debug_info:
        draw_mute_debug(overlay, mute_debug_info, dashed=washout)
    # Status below camera clock text
    status_text = f"LED:{led_status}  {mute_status}"
    bg_x2 = 10 + cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0] + 10
    st_y1, st_y2 = 25, 60
    roi = overlay[st_y1:st_y2, 5:bg_x2]
    overlay[st_y1:st_y2, 5:bg_x2] = (roi * 0.5).astype(roi.dtype)
    cv2.putText(overlay, status_text, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Washout indicator (red banner below status)
    if washout:
        wo_text = "WASHOUT"
        wo_size = cv2.getTextSize(wo_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        wo_x2 = 10 + wo_size[0] + 10
        wo_y1, wo_y2 = 62, 90
        if wo_y2 <= frame_h and wo_x2 <= frame_w:
            roi_wo = overlay[wo_y1:wo_y2, 5:wo_x2]
            overlay[wo_y1:wo_y2, 5:wo_x2] = (roi_wo * 0.3).astype(roi_wo.dtype)
            overlay[wo_y1:wo_y2, 5:wo_x2, 2] = np.clip(overlay[wo_y1:wo_y2, 5:wo_x2, 2].astype(np.int16) + 80, 0, 255).astype(np.uint8)
        cv2.putText(overlay, wo_text, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Digit images at top-right
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_scale = 0.7
    label_thick = 2
    img_y = 5
    x_offset = frame_w - 10

    # Label colors: colored when fresh, white when frame skipped (stale data)
    if frame_skipped:
        right_color1 = left_color1 = (255, 255, 255)
        right_color2 = left_color2 = (100, 100, 100)
    else:
        right_color1, right_color2 = (0, 255, 255), (128, 255, 255)
        left_color1, left_color2 = (255, 0, 255), (255, 128, 255)

    # Right digit image (cyan border)
    # Show grayscale after _enhance_dim_digit — same image that matchTemplate sees
    if right_digit_img is not None:
        rimg_gray, _ = _enhance_dim_digit(right_digit_img) if len(right_digit_img.shape) == 3 else (right_digit_img, False)
        rimg = cv2.cvtColor(rimg_gray, cv2.COLOR_GRAY2BGR)
        if right_match and right_match.get('match_pos') and right_match.get('template_size'):
            mx, my = right_match['match_pos']
            tw, th = right_match['template_size']
            cv2.rectangle(rimg, (mx, my), (mx + tw, my + th), (0, 255, 0), 1)
        rh_img, rw_img = rimg.shape[:2]
        x_offset -= rw_img
        if x_offset >= 0:
            overlay[img_y:img_y+rh_img, x_offset:x_offset+rw_img] = rimg
            cv2.rectangle(overlay, (x_offset, img_y), (x_offset+rw_img, img_y+rh_img), (0, 255, 255), 1)
            label1 = f"{right_digit}:{int(right_score*100)}%"
            label2 = f"{right_second}:{int(right_second_score*100)}%"
            ts1 = cv2.getTextSize(label1, label_font, label_scale, label_thick)[0]
            ts2 = cv2.getTextSize(label2, label_font, label_scale, label_thick)[0]
            max_tw = max(ts1[0], ts2[0])
            bg_x1, bg_y1 = max(0, x_offset - 3), img_y + rh_img + 3
            bg_x2, bg_y2 = x_offset + max_tw + 3, min(frame_h, img_y + rh_img + 48)
            if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                roi = overlay[bg_y1:bg_y2, bg_x1:bg_x2]
                overlay[bg_y1:bg_y2, bg_x1:bg_x2] = (roi * 0.5).astype(roi.dtype)
            cv2.putText(overlay, label1, (x_offset, img_y+rh_img+20), label_font, label_scale, right_color1, label_thick)
            cv2.putText(overlay, label2, (x_offset, img_y+rh_img+42), label_font, label_scale, right_color2, label_thick)
    x_offset -= 5

    # Left digit image (magenta border)
    # Show grayscale after _enhance_dim_digit — same image that matchTemplate sees
    if left_digit_img is not None:
        limg_gray, _ = _enhance_dim_digit(left_digit_img) if len(left_digit_img.shape) == 3 else (left_digit_img, False)
        limg = cv2.cvtColor(limg_gray, cv2.COLOR_GRAY2BGR)
        if left_match and left_match.get('match_pos') and left_match.get('template_size'):
            mx, my = left_match['match_pos']
            tw, th = left_match['template_size']
            cv2.rectangle(limg, (mx, my), (mx + tw, my + th), (0, 255, 0), 1)
        lh_img, lw_img = limg.shape[:2]
        x_offset -= lw_img
        if x_offset >= 0:
            overlay[img_y:img_y+lh_img, x_offset:x_offset+lw_img] = limg
            cv2.rectangle(overlay, (x_offset, img_y), (x_offset+lw_img, img_y+lh_img), (255, 0, 255), 1)
            label1 = f"{left_digit}:{int(left_score*100)}%"
            label2 = f"{left_second}:{int(left_second_score*100)}%"
            ts1 = cv2.getTextSize(label1, label_font, label_scale, label_thick)[0]
            ts2 = cv2.getTextSize(label2, label_font, label_scale, label_thick)[0]
            max_tw = max(ts1[0], ts2[0])
            bg_x1, bg_y1 = max(0, x_offset - 3), img_y + lh_img + 3
            bg_x2, bg_y2 = x_offset + max_tw + 3, min(frame_h, img_y + lh_img + 48)
            if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                roi = overlay[bg_y1:bg_y2, bg_x1:bg_x2]
                overlay[bg_y1:bg_y2, bg_x1:bg_x2] = (roi * 0.5).astype(roi.dtype)
            cv2.putText(overlay, label1, (x_offset, img_y+lh_img+20), label_font, label_scale, left_color1, label_thick)
            cv2.putText(overlay, label2, (x_offset, img_y+lh_img+42), label_font, label_scale, left_color2, label_thick)

    # Reading (large text, right-aligned)
    reading_font_scale = 1.5
    reading_thick = 3
    reading_size = cv2.getTextSize(reading, cv2.FONT_HERSHEY_SIMPLEX, reading_font_scale, reading_thick)[0]
    # Use last digit image height (left overrides right, matching original draw order)
    last_digit_h = 100
    if right_digit_img is not None:
        last_digit_h = right_digit_img.shape[0]
    if left_digit_img is not None:
        last_digit_h = left_digit_img.shape[0]
    reading_y = img_y + last_digit_h + 95
    reading_x = frame_w - reading_size[0] - 10
    bg_x1 = reading_x - 5
    bg_y1 = reading_y - reading_size[1] - 5
    bg_x2 = frame_w - 5
    bg_y2 = reading_y + 8
    if bg_x1 >= 0 and bg_y1 >= 0 and bg_y2 <= frame_h:
        roi = overlay[bg_y1:bg_y2, bg_x1:bg_x2]
        overlay[bg_y1:bg_y2, bg_x1:bg_x2] = (roi * 0.5).astype(roi.dtype)
    reading_color = (0, 255, 0)
    cv2.putText(overlay, reading, (reading_x, reading_y), cv2.FONT_HERSHEY_SIMPLEX, reading_font_scale, reading_color, reading_thick)

    # Gap debug: corrected panel + brightness histogram (left of digit images)
    # Show grayscale — same image that find_digit_gap uses for column sums
    if corrected_img is not None and gap_x is not None:
        gray_corr = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
        cimg = cv2.cvtColor(gray_corr, cv2.COLOR_GRAY2BGR)
        col_sums = np.sum(gray_corr, axis=0).astype(np.float64)
        kernel = np.ones(5) / 5
        smoothed = np.convolve(col_sums, kernel, mode='same')
        corr_h, corr_w = corrected_img.shape[:2]
        hist_h = 30
        hist_img = np.zeros((hist_h, corr_w, 3), dtype=np.uint8)
        max_val = max(smoothed) if max(smoothed) > 0 else 1
        for gx in range(corr_w):
            bar_h = int(smoothed[gx] / max_val * (hist_h - 2))
            cv2.line(hist_img, (gx, hist_h), (gx, hist_h - bar_h), (80, 80, 80), 1)
        line_layer = hist_img.copy()
        cv2.line(line_layer, (gap_x, 0), (gap_x, hist_h), (0, 255, 255), 2)
        cv2.addWeighted(line_layer, 0.5, hist_img, 0.5, 0, dst=hist_img)
        # Mark local minima
        center = corr_w // 2
        search_limit = int(corr_w * 0.15)
        for i in range(max(1, center - search_limit), min(len(smoothed) - 1, center + search_limit)):
            if smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]:
                bar_h_i = int(smoothed[i] / max_val * (hist_h - 2))
                cv2.circle(hist_img, (i, hist_h - bar_h_i), 2, (0, 255, 255), -1)
        gap_debug_img = np.vstack([cimg, hist_img])
        debug_h, debug_w = gap_debug_img.shape[:2]
        debug_x = x_offset - debug_w - 10
        debug_y = img_y
        if debug_x >= 0 and debug_y + debug_h <= frame_h:
            overlay[debug_y:debug_y+debug_h, debug_x:debug_x+debug_w] = gap_debug_img
            cv2.rectangle(overlay, (debug_x, debug_y), (debug_x+debug_w, debug_y+debug_h), (100, 100, 100), 1)

    # Mute zone zoom inset (4x, placed under gap debug)
    if mute_debug_info is not None:
        led_sx = mute_debug_info.get('mute_led_sx')
        led_sy = mute_debug_info.get('mute_led_sy')
        ref_sx = mute_debug_info.get('mute_ref_sx')
        ref_sy = mute_debug_info.get('mute_ref_sy')
        mute_rr = mute_debug_info.get('mute_rr')
        mute_re = mute_debug_info.get('mute_re')
        is_lit = mute_debug_info.get('is_lit', False)
        if led_sx is not None and ref_sx is not None:
            # Crop region centered on midpoint of LED and ref
            mid_x = (led_sx + ref_sx) / 2
            mid_y = (led_sy + ref_sy) / 2
            crop_half = 16  # 32x32 crop -> 128x128 at 4x
            zoom = 4
            cx1 = max(0, int(mid_x - crop_half))
            cy1 = max(0, int(mid_y - crop_half))
            cx2 = min(frame_w, int(mid_x + crop_half))
            cy2 = min(frame_h, int(mid_y + crop_half))
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                zoomed = cv2.resize(crop, (crop.shape[1] * zoom, crop.shape[0] * zoom),
                                    interpolation=cv2.INTER_NEAREST)
                zh, zw = zoomed.shape[:2]
                # Position: under gap debug if available, else top-right area
                if corrected_img is not None and gap_x is not None:
                    zx = debug_x
                    zy = debug_y + debug_h + 5
                else:
                    zx = frame_w - zw - 10
                    zy = 150
                # Draw patch boxes on zoomed image
                r = _geometry.mute_led_patch_radius
                for sx, sy, color in [(led_sx, led_sy, (0, 255, 0)),
                                      (ref_sx, ref_sy, (255, 255, 0))]:
                    if sx is not None and sy is not None:
                        cv2.rectangle(zoomed,
                                      (int((sx - r - cx1) * zoom), int((sy - r - cy1) * zoom)),
                                      (int((sx + r + 1 - cx1) * zoom), int((sy + r + 1 - cy1) * zoom)),
                                      color, 1)
                # MUTE:ON label in zoom view
                if is_lit:
                    cv2.putText(zoomed, "MUTE:ON", (4, 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(zoomed, "MUTE:ON", (4, 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # rr and re labels inside zoom (each colored by own threshold)
                rr_hit = mute_rr is not None and mute_rr > _geometry.mute_contrast_threshold
                re_hit = mute_re is not None and mute_re > 10
                rr_text = f"rr={mute_rr:.2f}" if mute_rr is not None else "rr=N/A"
                re_text = f"re={mute_re:.0f}" if mute_re is not None else "re=N/A"
                rr_color = (0, 0, 255) if rr_hit else (0, 200, 200)
                re_color = (0, 0, 255) if re_hit else (0, 200, 200)
                # rr on left, re on right
                cv2.putText(zoomed, rr_text, (4, zh - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
                cv2.putText(zoomed, rr_text, (4, zh - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, rr_color, 1)
                rr_w = cv2.getTextSize(rr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
                cv2.putText(zoomed, re_text, (4 + rr_w + 6, zh - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
                cv2.putText(zoomed, re_text, (4 + rr_w + 6, zh - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, re_color, 1)
                # Blit onto overlay
                if zx >= 0 and zy >= 0 and zx + zw <= frame_w and zy + zh <= frame_h:
                    overlay[zy:zy+zh, zx:zx+zw] = zoomed
                    cv2.rectangle(overlay, (zx, zy), (zx+zw, zy+zh), (100, 100, 100), 1)

    return overlay


def test_on_image(image_path):
    """Test panel detection and digit recognition pipeline on a single image.

    Uses SegmentReader.detect() for unified detection flow:
    digits + corner + LED + mute in a single call.

    Saves debug images for each step to the debug/ directory.

    Args:
        image_path: Path to the input image file
    """
    print(f"Testing: {image_path}")

    # Reset all detection state so unrelated images don't pollute each other
    global _button_zone_cache, _cached_buttons, _frame_led_dots, _corner_template_idx
    _geometry._corner_xy = None
    _geometry._smoothed_homography = None
    _geometry._golden_homography = None
    _geometry._scale = 1.0
    _geometry._homography_age = 0
    _geometry._geo_method = 'none'
    _geometry._homography_is_perspective = False
    _geometry._homography = None
    _geometry._load_initial_homography()  # Reload calibrated position
    _corner_template_idx = 0
    _button_zone_cache = None
    _cached_buttons = None
    _frame_led_dots = None

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  ERROR: Could not load image")
        return

    # Extract raw half from 1280x480 raw|overlay pairs
    h_img, w_img = frame.shape[:2]
    if w_img == 1280 and h_img == 480:
        frame = frame[:, :640]

    # Unified detection: digits + corner + LED + mute
    reader = SegmentReader()
    result = reader.detect(frame, debug=True)

    if result.panel_rect is None:
        print(f"  Panel NOT detected")
        return

    x, y, w, h = result.panel_rect
    print(f"  Panel detected: x={x}, y={y}, w={w}, h={h}")

    # Extract results from FrameResult
    led_status = result.led_status
    mute_status = result.mute_status
    corner_result = result.corner_result
    corner_debug = result.corner_debug
    led_debug_info = result.led_debug_info
    mute_debug_info = result.mute_debug_info

    print(f"  LED: {led_status}")
    print(f"  MUTE: {mute_status}")

    # Extract digit debug info
    dd = reader.digit_debug
    corrected_img = dd['corrected_img']
    gap_x = dd['gap_x']
    left_box = dd['left_box']
    right_box = dd['right_box']
    left_digit_img = dd['left_img']
    right_digit_img = dd['right_img']
    left_match = dd['left_match']
    right_match = dd['right_match']
    gap_debug = dd['gap_debug']
    boxes_debug = dd['boxes_debug']

    print(f"  Slant angle: 8.0 degrees (fixed)")
    print(f"  Gap position: x={gap_x}")

    lx, ly, lw, lh = left_box
    rx, ry, rw, rh = right_box
    print(f"  Left box: x={lx}-{lx+lw}, size {lw}x{lh}")
    print(f"  Right box: x={rx}-{rx+rw}, size {rw}x{rh}")

    # Get digit recognition results from reader state
    left_digit, right_digit = reader.raw_digits
    left_score, right_score = reader.last_scores
    (left_second, left_second_score), (right_second, right_second_score) = reader.last_second

    reading_raw = left_digit + right_digit
    print(f"  Recognition: {reading_raw}")

    # Build debug images with score annotation
    left_debug_img = left_digit_img.copy()
    h_l = left_debug_img.shape[0]
    cv2.putText(left_debug_img, f'{left_digit}({left_score:.2f})', (5, h_l - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    right_debug_img = right_digit_img.copy()
    h_r = right_debug_img.shape[0]
    cv2.putText(right_debug_img, f'{right_digit}({right_score:.2f})', (5, h_r - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Save debug images to debug directory
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    debug_dir = "debug"
    os.makedirs(debug_dir, exist_ok=True)

    cv2.imwrite(f"{debug_dir}/{base_name}_step3_1_gap.png", gap_debug)
    cv2.imwrite(f"{debug_dir}/{base_name}_step3_2_boxes.png", boxes_debug)
    cv2.imwrite(f"{debug_dir}/{base_name}_step5_left.png", left_debug_img)
    cv2.imwrite(f"{debug_dir}/{base_name}_step5_right.png", right_debug_img)

    # Generate overlay image (same layout as live_demo --display)
    overlay = draw_display_overlay(frame, (x, y, w, h), corrected_img, gap_x,
                                   left_digit_img, right_digit_img,
                                   left_digit, right_digit, left_score, right_score,
                                   left_match, right_match,
                                   left_second, left_second_score,
                                   right_second, right_second_score,
                                   reading_raw, led_status, mute_status,
                                   corner_debug=corner_debug,
                                   corner_score=corner_result[2] if corner_result else None,
                                   led_debug_info=led_debug_info,
                                   mute_debug_info=mute_debug_info,
                                   washout=result.washout)

    cv2.imwrite(f"{debug_dir}/{base_name}_overlay.png", overlay)

    print(f"  Debug images saved")


def main():
    """Run digit recognition tests on all example images.

    Finds all PNG images in the example/ directory and runs the
    full recognition pipeline on each, printing results to stdout.
    Debug images are saved to the debug/ directory.
    """
    import glob
    example_images = sorted(glob.glob("example/*.png") + glob.glob("example/*.PNG"))

    print("Steps 1-5: Panel + Slant + Boxes + Recognition")
    print("=" * 60)
    print(f"Found {len(example_images)} example images\n")

    for img_path in example_images:
        test_on_image(img_path)
        print()


if __name__ == "__main__":
    main()
