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
import numpy as np
import os
import sys
import json
import time

from device_geometry import get_geometry as _get_geometry


# Unified cache file for button zones and panel detection
_CACHE_FILE = os.path.join(os.path.dirname(__file__), 'last_ref.txt')
_ZONE_CHANGE_THRESHOLD = 10  # Pixels - only save if zones shift by more than this

# Cache for button zone centers (adaptive from detected buttons)
_button_zone_cache = None
# Cache for panel detection (shared with SegmentReader)
_panel_cache = None
# Track LED detection failures while using cache
_cache_led_fail_count = 0
_CACHE_FAIL_THRESHOLD = 10  # Switch to enlarged zones after this many failures

# Logging configuration
_LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
_LOG_ENABLED = True

def disable_logging():
    """Disable all file logging."""
    global _LOG_ENABLED
    _LOG_ENABLED = False
_LOG_COOLDOWN = 30  # Seconds between saves of same issue type
_LOG_MAX_FRAMES = 1000  # Max issue frames to keep
_log_last_save = {}  # issue_type -> timestamp
_log_file = None  # CSV file handle

# Corner templates for pattern matching (used for red button detection)
_corner_templates = None
_corner_template_idx = 0  # Current preferred template (round-robin with sticky preference)
_CORNER_TEMPLATE_FILES = [
    os.path.join(os.path.dirname(__file__), 'templates', 'corner_template.png'),
    os.path.join(os.path.dirname(__file__), 'templates', 'corner_template_2.png'),
    os.path.join(os.path.dirname(__file__), 'templates', 'corner_template_3.png'),
]
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
_last_auto_learned = None  # Tuple (digit, filename) of last manually saved template

# =============================================================================
# Detection Thresholds
# =============================================================================
_TEMPLATE_CONFIDENCE_THRESHOLD = 0.80
_TEMPLATE_AMBIGUITY_GAP = 0.05
_MIN_DIGIT_HEIGHT = 10
_MIN_DIGIT_WIDTH = 5

# Panel Detection (from geometry model)
_CORNER_TO_PANEL_X = abs(_geometry.panel_offset[0])
_CORNER_TO_PANEL_Y = abs(_geometry.panel_offset[1])
_BRIGHTNESS_PERCENTILE = _geometry.brightness_percentile
_MIN_BRIGHTNESS_THRESHOLD = _geometry.min_brightness_threshold
_PANEL_MARGIN_TOP_RATIO = _geometry.panel_margin_top_ratio
_PANEL_MARGIN_BOTTOM_RATIO = _geometry.panel_margin_bottom_ratio

# Button/LED Detection (from geometry model)
_BUTTON_REGION_RIGHT_RATIO = _geometry.button_region_right_ratio
_BUTTON_REGION_TOP_RATIO = _geometry.button_region_top_ratio
_LED_MIN_AREA = _geometry.led_min_area
_LED_MAX_AREA = _geometry.led_max_area
_LED_MAX_ASPECT_RATIO = _geometry.led_max_aspect_ratio
_WASHOUT_MIN_GAP = 30  # Min brightness gap for dark-hole LED detection

# Digit Recognition
_NO_DIGIT_MAX_INTENSITY = 50
_NO_DIGIT_MAX_BLUE_RATIO = 0.1
_SEGMENT_LIT_THRESHOLD = 0.5

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
def _get_content_bounds(mask, min_pixels=10):
    """Get bounding box of non-zero pixels in mask.

    Args:
        mask: Binary mask image
        min_pixels: Minimum pixels required to return valid bounds

    Returns:
        (x_min, y_min, x_max, y_max) tuple or None if insufficient pixels
    """
    coords = np.where(mask > 0)
    if len(coords[0]) < min_pixels:
        return None
    return (np.min(coords[1]), np.min(coords[0]),
            np.max(coords[1]), np.max(coords[0]))


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

    # Subtract background (local minimum) to handle uneven lighting
    # Use percentile instead of min to be robust against noise
    background = np.percentile(gray, 5)
    if background > 10:  # Only subtract if significant background brightness
        gray = np.clip(gray.astype(np.int16) - int(background), 0, 255).astype(np.uint8)

    # Check if dim: max < 150
    is_dim = gray.max() < 150  # Only use max brightness, mean is unreliable due to black background

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


def _calculate_brightness_confidence(detected_w, detected_h, content_x, content_y,
                                      frame_w, frame_h, bright_pixels, contours):
    """Calculate confidence score for brightness fallback panel detection.

    Scoring factors:
    - Size match (30%): How close detected size is to expected 165x105
    - Aspect ratio (25%): How close to expected 1.57 (165/105)
    - Content fill (20%): Ratio of bright pixels to bounding box
    - Position validity (15%): Distance from frame edges
    - Contour solidity (10%): Contour area / convex hull area

    Args:
        detected_w, detected_h: Detected content bounding box size
        content_x, content_y: Content position in frame
        frame_w, frame_h: Frame dimensions
        bright_pixels: Count of bright pixels in detected region
        contours: List of contours in the detection group

    Returns:
        Confidence score (0.0 to 1.0)
    """
    # 1. Size match (30%) - compare to expected panel content size
    # Content should be slightly smaller than panel (165x105)
    expected_w, expected_h = 140, 90  # Expected content size (smaller than panel)
    size_ratio = (detected_w * detected_h) / (expected_w * expected_h)
    size_score = 1.0 - min(abs(1.0 - size_ratio), 1.0)

    # 2. Aspect ratio (25%) - expected ~1.57
    expected_aspect = 165 / 105  # 1.57
    actual_aspect = detected_w / detected_h if detected_h > 0 else 0
    aspect_score = 1.0 - min(abs(expected_aspect - actual_aspect) / expected_aspect, 1.0)

    # 3. Content fill (20%) - bright pixels / bounding box
    bbox_area = detected_w * detected_h
    fill_ratio = bright_pixels / bbox_area if bbox_area > 0 else 0
    # Good fill is 0.3-0.7 (not too sparse, not solid blob)
    if 0.3 <= fill_ratio <= 0.7:
        fill_score = 1.0
    elif fill_ratio < 0.3:
        fill_score = fill_ratio / 0.3
    else:
        fill_score = max(0, 1.0 - (fill_ratio - 0.7) / 0.3)

    # 4. Position validity (15%) - should not be at edges
    margin_left = content_x
    margin_right = frame_w - (content_x + detected_w)
    margin_top = content_y
    margin_bottom = frame_h - (content_y + detected_h)
    min_margin = min(margin_left, margin_right, margin_top, margin_bottom)
    position_score = min(min_margin / 50.0, 1.0)

    # 5. Contour solidity (10%) - compactness of shape
    if contours:
        total_contour_area = sum(cv2.contourArea(c) for c in contours)
        # Merge contours for hull calculation
        all_points = np.vstack([c for c in contours])
        hull = cv2.convexHull(all_points)
        hull_area = cv2.contourArea(hull)
        solidity = total_contour_area / hull_area if hull_area > 0 else 0
    else:
        solidity = 0.5  # Default if no contours

    # Combined score with weights
    confidence = (0.30 * size_score +
                  0.25 * aspect_score +
                  0.20 * fill_score +
                  0.15 * position_score +
                  0.10 * solidity)

    return confidence


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


def _detect_red_pixels(image):
    """Detect LED pixels - both red and saturated white (overexposed).

    Handles two cases:
    1. Red LED: normal exposure shows red color
    2. White LED: overexposure causes red LED to appear white/saturated

    Also validates that detected pixels form bulb-like shapes.

    Args:
        image: BGR image

    Returns:
        Binary mask where LED pixels are white (255)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red detection: H: 0-10 or 150-180 (red hue wraps around 0)
    # S: ≥50 (reasonably saturated, not grayish)
    # V: ≥80 (bright enough to be a lit LED, filters dark noise)
    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 80]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([150, 50, 80]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # White/saturated detection: overexposed LED appears white
    # Only near-red hues (H: 0-10 or 150-180) - rejects bright non-red surfaces
    # S: low (0-50) - desaturated = whitish
    # V: very high (200-255) - bright saturated blob
    white_mask1 = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([10, 50, 255]))
    white_mask2 = cv2.inRange(hsv, np.array([150, 0, 200]), np.array([180, 50, 255]))
    white_mask = cv2.bitwise_or(white_mask1, white_mask2)

    # Combine red and white masks
    combined = cv2.bitwise_or(red_mask, white_mask)

    # Validate bulb-like shapes using connected components
    # Filter out noise and keep only compact, round-ish blobs
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, connectivity=8)

    result = np.zeros_like(combined)
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        # Filter by size: LED blob should be reasonably sized (5-500 pixels)
        if area < 5 or area > 500:
            continue

        # Filter by aspect ratio: bulb should be roughly round (aspect ratio < 3)
        aspect = max(width, height) / max(min(width, height), 1)
        if aspect > 3:
            continue

        # Filter by compactness: bulb is compact, not a thin line
        # Compactness = area / (width * height), higher is more compact
        compactness = area / max(width * height, 1)
        if compactness < 0.3:  # At least 30% filled
            continue

        # This blob passes all filters - add to result
        result[labels == i] = 255

    return result


def _cleanup_mask(binary, kernel_size=3, operation='close'):
    """Apply morphological cleanup to binary mask.

    Args:
        binary: Binary image to clean
        kernel_size: Size of morphological kernel
        operation: 'close' to fill gaps, 'open' to remove noise

    Returns:
        Cleaned binary image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if operation == 'close':
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    elif operation == 'open':
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


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
    if left_bound is not None and x1 < left_bound:
        pad_left = left_bound - x1
        x1 = left_bound
    if right_bound is not None and x2 > right_bound:
        pad_right = x2 - right_bound
        x2 = right_bound

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
                        penalized_score = digit_1_original_score * 0.7
                        all_scores[digit_1_idx] = ('1', penalized_score, digit_1_entry[2], digit_1_entry[3], digit_1_entry[4])
                        all_scores.sort(key=lambda x: -x[1])
                        best_digit, best_score, best_template_idx, best_match_pos, best_template_size = all_scores[0]
                        second_digit, second_score, second_template_idx = all_scores[1][:3] if len(all_scores) > 1 else ('X', 0.0, 0)
                    # else: 2nd best is not a left-bar digit (e.g., 7) - no penalty, keep original score

    # Handle "2" vs "9" confusion with uneven lighting
    # Bright right side can make "9" look like "2" (right side segments appear lit)
    swapped_due_to_lighting = False
    if best_digit == '2' and second_digit == '9' and (best_score - second_score) < 0.05:
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

    # Handle "9" vs "5" and "8" vs "6" confusion by checking top-right segment
    # 9 and 8 have top-right segment lit (blue), 5 and 6 do not
    top_right_lit = {'9', '8'}  # Digits with top-right segment
    top_right_off = {'5', '6'}  # Digits without top-right segment
    is_top_right_confusion = (best_digit in top_right_lit and second_digit in top_right_off) or \
                             (best_digit in top_right_off and second_digit in top_right_lit)
    # Only check within same digit pair (9/5 or 8/6)
    same_pair = (best_digit in {'9', '5'} and second_digit in {'9', '5'}) or \
                (best_digit in {'8', '6'} and second_digit in {'8', '6'})
    if is_top_right_confusion and same_pair and (best_score - second_score) < 0.07:
        # Compare middle-right (segment b area) to center (dark reference)
        # Segment b is lit if middle-right is significantly brighter than center
        match_x, match_y = best_match_pos
        tmpl_w, tmpl_h = best_template_size
        img_h, img_w = digit_img.shape[:2]

        # Middle-right region (segment b vertical part, avoiding corners)
        mr_x = match_x + int(tmpl_w * 0.7)
        mr_y = match_y + int(tmpl_h * 0.20)
        mr_x2 = min(match_x + tmpl_w, img_w)
        mr_y2 = min(match_y + int(tmpl_h * 0.50), img_h)

        # Center region (dark reference - inside digit, no segments)
        cx = match_x + int(tmpl_w * 0.35)
        cy = match_y + int(tmpl_h * 0.15)
        cx2 = min(match_x + int(tmpl_w * 0.65), img_w)
        cy2 = min(match_y + int(tmpl_h * 0.35), img_h)

        # Get blue channel values
        if mr_x2 > mr_x and mr_y2 > mr_y and cx2 > cx and cy2 > cy:
            mid_right = digit_img[mr_y:mr_y2, mr_x:mr_x2]
            center = digit_img[cy:cy2, cx:cx2]
            mr_blue = mid_right[:, :, 0].mean()
            center_blue = center[:, :, 0].mean()

            # Ratio: how much brighter is middle-right vs center?
            blue_ratio = mr_blue / max(center_blue, 1)

            # Determine which digit pair we're dealing with
            lit_digit = '9' if best_digit in {'9', '5'} else '8'
            off_digit = '5' if best_digit in {'9', '5'} else '6'

            if blue_ratio > 1.2:
                # Segment b is lit → should be 9 or 8
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
                # Segment b is off → should be 5 or 6
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

    # Special check for 6 vs 8 using B/C segment ratio
    # For 8, both B and C are lit equally (ratio ~1.0)
    # For 6, only C is lit, B has only glow (ratio < 0.90)
    # Only apply when C is reliably detected (C > 0.9) to avoid false positives
    segment_override_6to8 = False
    if best_digit == '6' and best_match_pos is not None and best_template_size is not None:
        b_lit = _check_segment_lit(digit_img, 'B', best_match_pos, best_template_size)
        c_lit = _check_segment_lit(digit_img, 'C', best_match_pos, best_template_size)
        # Only trust the ratio when C is clearly detected (> 0.9)
        if c_lit > 0.9:
            bc_ratio = b_lit / c_lit
            # If B and C are equally bright (ratio > 0.97), it's 8, not 6
            if bc_ratio > 0.97:
                for item in all_scores:
                    if item[0] == '8':
                        old_6_score = best_score
                        best_digit = '8'
                        # Use the higher of: boosted 8 score OR 6's score (to avoid negative gap)
                        best_score = max(min(item[1] * 1.10, 0.99), old_6_score)
                        best_template_idx = item[2]
                        best_match_pos = item[3]
                        best_template_size = item[4]
                        second_digit, second_score = '6', old_6_score * 0.95  # Penalize 6
                        segment_override_6to8 = True
                        break

    # Resolve 0/1/6/8/P confusion using segment analysis when candidates are close
    # These digits share many segments and often have close template scores
    gap = best_score - second_score
    if best_digit in _CONFUSING_DIGITS and second_digit in _CONFUSING_DIGITS and gap < 0.05:
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
    if not swapped_due_to_lighting and ((best_score < 0.75 and gap < 0.20) or gap < 0.02):
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


def _auto_save_template(digit, template_img, reason=""):
    """Auto-save a new template for learning.

    Uses naming convention: digit_0a_learn.png, digit_0b_learn.png, etc.
    """
    global _digit_templates
    from datetime import datetime

    if not os.path.exists(_TEMPLATES_DIR):
        os.makedirs(_TEMPLATES_DIR)

    # Find next available letter suffix (a-z) for auto-learned templates
    existing = [f for f in os.listdir(_TEMPLATES_DIR)
                if f.startswith(f'digit_{digit}') and f.endswith('_learn.png')]

    # Extract used suffix letters
    used_letters = set()
    for f in existing:
        # digit_0a_learn.png -> extract 'a'
        name = f.replace('digit_', '').replace('_learn.png', '')
        if len(name) >= 2:
            used_letters.add(name[1])

    # Find next available letter
    next_letter = None
    for c in 'abcdefghijklmnopqrstuvwxyz':
        if c not in used_letters:
            next_letter = c
            break

    # Limit templates per digit to avoid bloat (a-z = 26 max)
    if next_letter is None:
        return

    global _last_auto_learned

    filename = f'digit_{digit}{next_letter}_learn.png'
    filepath = os.path.join(_TEMPLATES_DIR, filename)

    try:
        # Save template image
        if not cv2.imwrite(filepath, template_img):
            print(f"Warning: Failed to write template {filepath}", flush=True)
            return

        # Log to learn.log
        log_file = os.path.join(os.path.dirname(_TEMPLATES_DIR), 'learn.log')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file, 'a') as f:
            f.write(f'{timestamp} - Learned: digit={digit}, file={filename}, reason={reason}\n')

        # Add to in-memory cache
        if _digit_templates is not None:
            if digit not in _digit_templates:
                _digit_templates[digit] = []
            _digit_templates[digit].append(template_img)

        # Signal for display notification
        _last_auto_learned = (digit, filename)
    except (IOError, OSError) as e:
        print(f"Warning: Failed to save template {filename}: {e}", flush=True)


def _recognize_digit_segments(digit_img):
    """
    Recognize digit using segment-based analysis (for verification).

    Uses 7-segment pattern matching based on blue pixel ratios.

    Args:
        digit_img: BGR image of single digit

    Returns:
        digit: Recognized character ('0'-'9', 'P') or 'X' if unknown
    """
    h, w = digit_img.shape[:2]
    if h < _MIN_DIGIT_HEIGHT or w < _MIN_DIGIT_WIDTH:
        return 'X'

    # Get blue mask
    mask = get_blue_mask(digit_img, tight=True)
    coords = np.where(mask > 0)

    if len(coords[0]) < 20:
        return 'X'

    # Get content bounds
    content_top = np.min(coords[0])
    content_bottom = np.max(coords[0])
    content_left = np.min(coords[1])
    content_right = np.max(coords[1])
    content_w = content_right - content_left
    content_h = content_bottom - content_top

    if content_w < 3 or content_h < 10:
        return 'X'

    # Check for narrow "1"
    if content_w < content_h * 0.25:
        return '1'

    # Define segment zones (relative to content bounds)
    def get_zone_ratio(rx, ry, rw, rh):
        x1 = int(content_left + rx * content_w)
        y1 = int(content_top + ry * content_h)
        x2 = int(content_left + (rx + rw) * content_w)
        y2 = int(content_top + (ry + rh) * content_h)
        zone = mask[y1:y2, x1:x2]
        return np.sum(zone > 0) / zone.size if zone.size > 0 else 0

    # Segment zones
    segments = {
        'A': get_zone_ratio(0.15, 0.00, 0.70, 0.15),  # top
        'B': get_zone_ratio(0.60, 0.05, 0.40, 0.42),  # upper right
        'C': get_zone_ratio(0.60, 0.53, 0.40, 0.42),  # lower right
        'D': get_zone_ratio(0.15, 0.85, 0.70, 0.15),  # bottom
        'E': get_zone_ratio(0.00, 0.53, 0.40, 0.42),  # lower left
        'F': get_zone_ratio(0.00, 0.05, 0.40, 0.42),  # upper left
        'G': get_zone_ratio(0.15, 0.42, 0.70, 0.16),  # middle
    }

    # Segment patterns for each digit
    PATTERNS = {
        '0': 'ABCDEF',
        '1': 'BC',
        '2': 'ABDEG',
        '3': 'ABCDG',
        '4': 'BCFG',
        '5': 'ACDFG',
        '6': 'ACDEFG',
        '7': 'ABC',
        '8': 'ABCDEFG',
        '9': 'ABCDFG',
        'P': 'ABEFG',
    }

    # Score each pattern
    best_digit = 'X'
    best_score = -100

    threshold = 0.15  # Minimum ratio to consider segment "on"

    for digit, pattern in PATTERNS.items():
        score = 0
        for seg in 'ABCDEFG':
            ratio = segments[seg]
            if seg in pattern:
                # Segment should be ON
                score += ratio
            else:
                # Segment should be OFF
                score += (1 - ratio) * 0.5  # Less penalty for off segments

        if score > best_score:
            best_score = score
            best_digit = digit

    return best_digit


def _load_cache():
    """Load unified cache from disk if exists."""
    global _button_zone_cache, _panel_cache
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, 'r') as f:
                data = json.load(f)
                # Load button zones
                if 'button_zones' in data:
                    _button_zone_cache = [(z['left'], z['right'], z['top'], z['bottom'], z['name'])
                                          for z in data['button_zones']]
                # Load panel cache
                if 'panel' in data:
                    _panel_cache = data['panel']
        except (json.JSONDecodeError, KeyError, IOError):
            _button_zone_cache = None
            _panel_cache = None


def _save_cache():
    """Save unified cache to disk."""
    global _button_zone_cache, _panel_cache
    try:
        data = {}
        if _button_zone_cache is not None:
            data['button_zones'] = [{'left': left, 'right': right, 'top': top, 'bottom': bottom, 'name': name}
                                    for left, right, top, bottom, name in _button_zone_cache]
        if _panel_cache is not None:
            data['panel'] = _panel_cache
        with open(_CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except IOError:
        pass


def _update_panel_cache(panel_rect=None, gap_x=None, left_box=None, right_box=None, last_reading=None):
    """Update panel cache and save to disk."""
    global _panel_cache

    # Convert numpy types to Python native types
    def to_native(val):
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            return [to_native(v) for v in val]
        if hasattr(val, 'item'):  # numpy scalar
            return val.item()
        return val

    _panel_cache = {
        'panel_rect': to_native(panel_rect),
        'gap_x': to_native(gap_x),
        'left_box': to_native(left_box),
        'right_box': to_native(right_box),
        'last_reading': last_reading
    }
    _save_cache()


def _get_panel_cache():
    """Get panel cache data."""
    return _panel_cache


def _load_corner_templates():
    """Load corner templates for pattern matching."""
    global _corner_templates
    if _corner_templates is None:
        _corner_templates = []
        for path in _CORNER_TEMPLATE_FILES:
            if os.path.exists(path):
                tmpl = cv2.imread(path)
                if tmpl is not None:
                    _corner_templates.append(tmpl)
    return _corner_templates


def _find_corner(frame, min_match=0.90, return_debug=False):
    """
    Find the corner in the frame using template matching.

    Optimized: Uses center 1/4 of template and searches only right portion of frame.
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
    search_left, search_top, search_size = _geometry.get_corner_search_region(w_frame, h_frame)
    search_region = _geometry.undistort_roi(
        frame, search_left, search_top, search_size, search_size, derotate=False)

    # Search region rect for debug visualization
    search_rect = (search_left, search_top, search_size, search_size)

    def try_template(idx):
        """Try matching a single template, return (score, loc, crop_size) or None."""
        if idx >= len(templates):
            return None
        template = templates[idx]
        th, tw = template.shape[:2]
        crop_h, crop_w = th // 2, tw // 2
        crop_y, crop_x = th // 2, tw // 2
        template_crop = template[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        if crop_h > search_size or crop_w > search_size:
            return None
        result = cv2.matchTemplate(search_region, template_crop, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        return (max_val, max_loc, (crop_w, crop_h))

    # Round-robin with sticky preference: try current template first
    best_score = 0
    best_loc = None
    best_crop_size = None

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
                best_score, best_loc, best_crop_size = result
                break
            elif result and result[0] > best_score:
                best_score, best_loc, best_crop_size = result

    if best_score < min_match:
        if return_debug:
            # Always return score for logging, even when below threshold
            return (None, None, best_score), (search_rect, None, best_crop_size or (0, 0))
        return None

    # Match location is top-left of cropped template in search region
    # Convert to center of full template in full frame
    # crop is from right-lower quadrant, so corner center is at crop origin
    corner_x = search_left + best_loc[0]
    corner_y = search_top + best_loc[1]

    # Update geometry with detected corner for adaptive search regions
    _geometry.set_corner(corner_x, corner_y)

    # Match rect in frame coordinates (where template was matched)
    match_rect = (search_left + best_loc[0], search_top + best_loc[1], best_crop_size[0], best_crop_size[1])

    if return_debug:
        return (corner_x, corner_y, best_score), (search_rect, match_rect, best_crop_size)
    return (corner_x, corner_y, best_score)


def draw_corner_debug(frame, debug_info):
    """Draw corner search area and match location on frame.

    Args:
        frame: BGR image to draw on (modified in place)
        debug_info: (search_rect, match_rect, template_size) from _find_corner
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
        cv2.putText(frame, "corner", (mx + 2, my - 5),
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
    global _button_zone_cache, _panel_cache
    _button_zone_cache = None
    _panel_cache = None
    if os.path.exists(_CACHE_FILE):
        try:
            os.remove(_CACHE_FILE)
        except IOError:
            pass


# Alias for backwards compatibility
clear_button_zone_cache = clear_cache


# Load cache from disk on module import
_load_cache()


# =============================================================================
# Logging functions for cache threshold analysis
# =============================================================================

def _init_log():
    """Initialize CSV log file with headers."""
    global _log_file
    if not _LOG_ENABLED or _log_file is not None:
        return

    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        log_path = os.path.join(_LOG_DIR, 'detection.csv')
        write_header = not os.path.exists(log_path)

        _log_file = open(log_path, 'a')
        # Register cleanup immediately after opening to prevent leaks
        import atexit
        atexit.register(close_log)

        if write_header:
            _log_file.write('timestamp,panel_x,panel_y,panel_w,panel_h,gap_x,'
                           'left_score,right_score,reading,led_status,'
                           'corner_score,detection_method,brightness_conf,mute_status,mute_pixels,dim_enhanced,frame_skip,diff_edge,led_gap,led_method,proc_ms,issue,'
                           'geo_method,geo_scale,geo_rotation,undistort_px\n')
            _log_file.flush()
    except (IOError, OSError) as e:
        print(f"Warning: Failed to initialize log: {e}", flush=True)
        if _log_file is not None:
            _log_file.close()
        _log_file = None


def log_detection(panel_rect=None, gap_x=None, left_score=0, right_score=0,
                  reading=None, led_status=None, corner_score=0,
                  detection_method=None, brightness_conf=None, mute_status=None,
                  mute_pixels=0, dim_enhanced=None, frame_skip=False, diff_edge=None,
                  led_gap=None, led_method=None, proc_ms=None, issue=None,
                  geo_method=None, geo_scale=None, geo_rotation=None,
                  undistorted=None):
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
    br_conf = f'{brightness_conf:.3f}' if brightness_conf is not None else ''
    mute = mute_status if mute_status is not None else ''
    mute_px = int(mute_pixels) if mute_pixels else 0
    dim_enh = dim_enhanced if dim_enhanced is not None else ''
    skip = '1' if frame_skip else ''
    diff_e = str(int(diff_edge)) if diff_edge is not None else ''
    led_g = str(int(led_gap)) if led_gap is not None else ''
    led_m = led_method if led_method is not None else ''
    proc = f'{proc_ms:.1f}' if proc_ms is not None else ''
    iss = issue if issue is not None else ''
    geo_m = geo_method if geo_method is not None else ''
    geo_s = f'{geo_scale:.3f}' if geo_scale is not None else ''
    geo_r = f'{geo_rotation:.2f}' if geo_rotation is not None else ''
    undist = f'{undistorted:.1f}' if undistorted else '0'

    _log_file.write(f'{ts},{px},{py},{pw},{ph},{gx},'
                   f'{left_score:.3f},{right_score:.3f},{rd},{led},'
                   f'{corner_score:.3f},{method},{br_conf},{mute},{mute_px},{dim_enh},{skip},{diff_e},{led_g},{led_m},{proc},{iss},'
                   f'{geo_m},{geo_s},{geo_r},{undist}\n')
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


def _detect_dark_panel(frame, margin_top, margin_bottom):
    """
    Fallback panel detection using intensity ratio to find black/gray boundary.
    Finds the black display region within the darker slot area.

    Args:
        frame: BGR image
        margin_top: Top margin to exclude
        margin_bottom: Bottom margin to exclude

    Returns:
        (x, y, w, h) of detected panel, or None
    """
    h_frame, w_frame = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # First, find the general dark slot area using threshold
    _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    dark_mask = _cleanup_mask(dark_mask, kernel_size=5, operation='close')

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the main dark slot
    slot_rect = None
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if y < margin_top or (y + h) > margin_bottom:
            continue
        aspect = w / h if h > 0 else 0
        if aspect < 1.0 or aspect > 5.0:
            continue
        if area > max_area and area > 5000:
            max_area = area
            slot_rect = (x, y, w, h)

    if slot_rect is None:
        return None

    sx, sy, sw, sh = slot_rect
    slot_roi = gray[sy:sy+sh, sx:sx+sw]

    # Within the slot, find display by ratio-based edge detection
    # Scan middle row to find left/right transitions
    mid_row = slot_roi[sh//2, :].astype(float)
    smooth = np.convolve(mid_row, np.ones(5)/5, mode='same')

    # Find left edge: where intensity drops significantly
    left_edge = 0
    for x in range(sw//4, sw//2):
        left_avg = np.mean(smooth[max(0,x-10):x]) if x > 10 else smooth[0]
        right_avg = np.mean(smooth[x:min(sw,x+10)])
        if right_avg < left_avg * 0.7:
            left_edge = x
            break

    # Find right edge: where intensity rises significantly
    right_edge = sw
    for x in range(sw*3//4, sw//2, -1):
        left_avg = np.mean(smooth[max(0,x-10):x])
        right_avg = np.mean(smooth[x:min(sw,x+10)]) if x < sw-10 else smooth[-1]
        if left_avg < right_avg * 0.7:
            right_edge = x
            break

    # Find top/bottom using same ratio approach
    mid_x = (left_edge + right_edge) // 2
    mid_col = slot_roi[:, mid_x].astype(float)
    smooth_col = np.convolve(mid_col, np.ones(5)/5, mode='same')

    top_edge = 0
    for y in range(sh//4, sh//2):
        top_avg = np.mean(smooth_col[max(0,y-10):y]) if y > 10 else smooth_col[0]
        bot_avg = np.mean(smooth_col[y:min(sh,y+10)])
        if bot_avg < top_avg * 0.7:
            top_edge = y
            break

    bottom_edge = sh
    for y in range(sh*3//4, sh//2, -1):
        top_avg = np.mean(smooth_col[max(0,y-10):y])
        bot_avg = np.mean(smooth_col[y:min(sh,y+10)]) if y < sh-10 else smooth_col[-1]
        if top_avg < bot_avg * 0.7:
            bottom_edge = y
            break

    # Convert to frame coordinates with padding
    pad_x = int((right_edge - left_edge) * 0.1)
    pad_y = int((bottom_edge - top_edge) * 0.3)  # More vertical padding

    panel_x = max(0, sx + left_edge - pad_x)
    panel_y = max(0, sy + top_edge - pad_y)
    panel_w = min(w_frame - panel_x, right_edge - left_edge + 2 * pad_x)
    panel_h = min(h_frame - panel_y, bottom_edge - top_edge + 2 * pad_y)

    # Validate
    if panel_w <= 0 or panel_h <= 0:
        return None
    if panel_w < 30 or panel_h < 30:
        return None
    aspect = panel_w / panel_h
    if aspect < 0.5 or aspect > 3.0:
        return None

    return (panel_x, panel_y, panel_w, panel_h)


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
    h_frame, w_frame = frame.shape[:2]

    # Step 1: Find corner
    # Threshold lowered to 0.85 to use corner detection more often (revisit after more data logged)
    corner_result = _find_corner(frame, min_match=0.85)
    if corner_result is None:
        return None

    corner_x, corner_y, corner_score = corner_result

    # Step 2: Define button search region based on corner
    # Buttons are to the left of the corner, BELOW the corner position
    # Corner is at top-right of device, buttons are at bottom
    btn_search_top = corner_y + _geometry.button_search_top_offset  # Buttons start below corner
    btn_search_bottom = h_frame
    btn_search_left = 0
    btn_search_right = corner_x  # Buttons are LEFT of corner, don't search past it

    button_region = frame[btn_search_top:btn_search_bottom, btn_search_left:btn_search_right]
    if button_region.shape[0] < 10 or button_region.shape[1] < 10:
        return None

    # Step 3: Detect buttons in the region
    buttons = _detect_buttons(button_region)
    buttons = sorted(buttons, key=lambda b: b[0])  # Sort left to right

    # Require at least 3 buttons for reliable landmark-based detection
    # If < 3 buttons, fall back to corner-only detection (more reliable)
    if len(buttons) < 3:
        return None

    # Use rightmost 3 buttons: B2, S1, S2
    # (B1 is sometimes visible at left edge, skip it if 4 buttons found)
    b2_x, b2_y, b2_w, b2_h = buttons[-3]
    s1_x, s1_y, s1_w, s1_h = buttons[-2]
    s2_x, s2_y, s2_w, s2_h = buttons[-1]

    # Convert button centers to frame coordinates
    button_centers = {
        'B2': (btn_search_left + b2_x + b2_w // 2,
               btn_search_top + b2_y + b2_h // 2),
        'S1': (btn_search_left + s1_x + s1_w // 2,
               btn_search_top + s1_y + s1_h // 2),
        'S2': (btn_search_left + s2_x + s2_w // 2,
               btn_search_top + s2_y + s2_h // 2),
    }

    # Step 4: Compute homography and project panel position
    if _geometry.compute_homography((corner_x, corner_y), button_centers):
        panel_rect = _geometry.get_panel_rect()
        if panel_rect is not None:
            px, py, pw, ph = panel_rect
            # Clamp to frame bounds
            px = max(0, px)
            py = max(0, py)
            pw = min(w_frame - px, pw)
            ph = min(h_frame - py, ph)
            if pw >= 50 and ph >= 30:
                _geometry._geo_method = 'homography'
                return (px, py, pw, ph)

    # Fallback: manual offset calculation (same as original code)
    _geometry._geo_method = 'offset'
    b2_x_frame = btn_search_left + b2_x
    s2_x_frame = btn_search_left + s2_x

    btn_top_in_frame = btn_search_top + min(b2_y, s1_y, s2_y)

    panel_bottom = btn_top_in_frame - _geometry.button_panel_gap
    panel_top = max(0, panel_bottom - _PANEL_HEIGHT)
    panel_height = panel_bottom - panel_top

    panel_center = (b2_x_frame + s2_x_frame + s2_w) // 2
    panel_width = _PANEL_WIDTH
    panel_left = max(0, panel_center - panel_width // 2)

    if panel_width < 50 or panel_width > 200 or panel_height < 30:
        return None

    return (panel_left, panel_top, panel_width, panel_height)


def detect_panel(frame, return_confidence=False):
    """
    Detect the dark rectangular panel containing blue LED digits.

    Uses landmark-based prediction (corner + buttons) as primary method,
    with blue LED color detection as fallback.

    Args:
        frame: BGR image from camera/file
        return_confidence: If True, return confidence score for brightness method

    Returns:
        panel_rect: (x, y, w, h) of the detected panel, or None if not found
        method: detection method used ('landmark', 'corner', 'brightness', or None)
        confidence: (only if return_confidence=True) confidence score (0-1) for
                   brightness method, None for other methods
    """
    h_frame, w_frame = frame.shape[:2]

    # Reset stale homography - will be recomputed if landmarks found
    _geometry._homography = None
    _geometry._geo_method = 'none'

    # Try landmark-based detection first (corner + buttons)
    landmark_panel = predict_panel_from_landmarks(frame)
    if landmark_panel is not None:
        if return_confidence:
            return landmark_panel, 'landmark', None
        return landmark_panel, 'landmark'

    # Fallback 1: Corner-only detection (if corner found but buttons failed)
    # Use fixed spatial relationship from corner to panel
    # Threshold lowered to 0.85 to use corner detection more often (revisit after more data logged)
    corner_result = _find_corner(frame, min_match=0.85)
    if corner_result is not None:
        corner_x, corner_y, _ = corner_result
        # Known offsets from calibration:
        # Panel x ≈ corner_x - 266 (centered between B2 and S2)
        # Panel y ≈ corner_y - 86
        panel_x = corner_x - _CORNER_TO_PANEL_X
        panel_y = corner_y - _CORNER_TO_PANEL_Y

        # Validate bounds
        if panel_x >= 0 and panel_y >= 0:
            if return_confidence:
                return (panel_x, panel_y, _PANEL_WIDTH, _PANEL_HEIGHT), 'corner', None
            return (panel_x, panel_y, _PANEL_WIDTH, _PANEL_HEIGHT), 'corner'

    # Fallback 2: brightness-based detection (if corner not found)
    # Find bright regions (the glowing digits)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold at top 3% brightness
    thresh_val = np.percentile(gray, _BRIGHTNESS_PERCENTILE)
    _, binary = cv2.threshold(gray, max(thresh_val, _MIN_BRIGHTNESS_THRESHOLD), 255, cv2.THRESH_BINARY)

    # Clean up
    binary = _cleanup_mask(binary, kernel_size=3, operation='close')

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        if return_confidence:
            return None, None, None
        return None, None

    # Filter contours by position and size
    margin_top = int(h_frame * _PANEL_MARGIN_TOP_RATIO)
    margin_bottom = int(h_frame * _PANEL_MARGIN_BOTTOM_RATIO)
    min_area = 50
    max_area = h_frame * w_frame * 0.025  # Increased for dark scenarios

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)

        # Must be in vertical middle, reasonable size
        if y < margin_top or (y + h) > margin_bottom:
            continue
        if area < min_area or area > max_area:
            continue

        candidates.append((x, y, w, h, area))

    if not candidates:
        if return_confidence:
            return None, None, None
        return None, None

    # Group nearby candidates horizontally (merge digit contours)
    candidates.sort(key=lambda c: c[0])  # Sort by x

    groups = []
    current_group = [candidates[0]]

    for c in candidates[1:]:
        last = current_group[-1]
        # If close horizontally and similar y, add to group
        if c[0] - (last[0] + last[2]) < 50 and abs(c[1] - last[1]) < 50:
            current_group.append(c)
        else:
            groups.append(current_group)
            current_group = [c]
    groups.append(current_group)

    # Find best group (highest total area, good aspect ratio)
    best_group = None
    best_score = 0

    for group in groups:
        xs = [c[0] for c in group]
        ys = [c[1] for c in group]
        x2s = [c[0] + c[2] for c in group]
        y2s = [c[1] + c[3] for c in group]

        gx, gy = min(xs), min(ys)
        gw, gh = max(x2s) - gx, max(y2s) - gy

        total_area = sum(c[4] for c in group)
        aspect = gw / gh if gh > 0 else 0

        # Prefer 2-digit aspect ratio (1.0 - 2.5)
        if 0.8 < aspect < 3.0:
            score = total_area * (1 + aspect)
        else:
            score = total_area * 0.3

        if score > best_score:
            best_score = score
            best_group = group

    if not best_group:
        if return_confidence:
            return None, None, None
        return None, None

    # Get bounding box of best group
    xs = [c[0] for c in best_group]
    ys = [c[1] for c in best_group]
    x2s = [c[0] + c[2] for c in best_group]
    y2s = [c[1] + c[3] for c in best_group]

    x, y = min(xs), min(ys)
    w, h = max(x2s) - x, max(y2s) - y

    # Calculate weighted center (centroid) of bright pixels in the region
    # Use centroid for X (horizontal), but top edge for Y (vertical)
    # This handles uneven glow: X centering is stable, Y anchors to top
    region_mask = binary[y:y+h, x:x+w]
    moments = cv2.moments(region_mask)
    if moments['m00'] > 0:
        # Centroid X relative to region, convert to frame coordinates
        cx = int(moments['m10'] / moments['m00']) + x
    else:
        # Fallback to bounding box center
        cx = x + w // 2

    # Horizontal: center on centroid X
    # Vertical: center content within panel height
    panel_x = max(0, cx - _PANEL_WIDTH // 2)
    vertical_padding = (_PANEL_HEIGHT - h) // 2  # Dynamic padding to center content
    panel_y = max(0, y - vertical_padding)
    panel_w = min(w_frame - panel_x, _PANEL_WIDTH)
    panel_h = min(h_frame - panel_y, _PANEL_HEIGHT)

    panel_rect = (panel_x, panel_y, panel_w, panel_h)

    if return_confidence:
        # Calculate confidence score for brightness detection
        bright_pixels = cv2.countNonZero(region_mask)
        # Get contours from the region for solidity calculation
        region_contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        confidence = _calculate_brightness_confidence(
            w, h, x, y, w_frame, h_frame, bright_pixels, region_contours
        )
        return panel_rect, 'brightness', confidence

    return panel_rect, 'brightness'


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

    # Define button region - below the panel
    if panel_rect is not None:
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

    if bh < 10 or bw < 10:
        if return_debug:
            return leds, debug_img, None
        if debug:
            return leds, debug_img
        return leds, None

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
    buttons = _detect_buttons(button_region)

    # Create LED mask for detection
    led_mask = _create_led_mask(button_region)

    # Build button zones from detected buttons
    # Sort by x position (left to right)
    buttons = sorted(buttons, key=lambda b: b[0])

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

        # Calculate spacing from B2, S1, S2
        spacing_b2_s1 = s1_center - b2_center
        spacing_s1_s2 = s2_center - s1_center
        avg_spacing = (spacing_b2_s1 + spacing_s1_s2) / 2

        # Predict B1 center as B2 center minus average spacing
        b1_center = b2_center - avg_spacing

        # Predict B1 X position
        b1_x = int(b1_center - avg_width / 2)

        # B1 is on the same row as B2, so use B2's Y directly
        b2_y = b2_btn[1]
        b2_height = b2_btn[3]
        b1_y = b2_y

        predicted_b1_box = (b1_x, b1_y, int(avg_width), int(b2_height))

        # Build LED zones with boundaries (left_x, right_x, top_y, bottom_y, name)
        # LED is on the right side of each button (50%-100% of button width)
        half_width = avg_width / 2

        # Get Y boundaries from detected buttons (B2, S1, S2)
        b2_top, b2_bottom = b2_btn[1], b2_btn[1] + b2_btn[3]
        s1_top, s1_bottom = s1_btn[1], s1_btn[1] + s1_btn[3]
        s2_top, s2_bottom = s2_btn[1], s2_btn[1] + s2_btn[3]
        # B1 uses B2's Y (same row)
        b1_top, b1_bottom = b2_top, b2_bottom

        # LED zone: from button center to right edge, within button Y bounds
        # B1 (predicted) - LED is at ~88% of button width from left edge
        # (tuned from 0.75: actual LED at x=32.6, old estimate was 23.25)
        b1_led_x = b1_x + avg_width * 0.88  # Expected LED X position
        if b1_led_x > 15:  # LED must be at least 15px into visible area
            # B1 zone: shifted right to center on actual LED position (~32px)
            # Min 18px from edge to avoid display contamination
            b1_led_left = max(18, b1_led_x - half_width / 2)
            b1_led_right = min(b1_led_x + half_width / 2, b2_center - 5)
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
        if (not used_cache or cache_seems_stale) and detection_method is not None and detection_method != 'landmark':
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

    # Find the LED by checking which button zone contains the blob
    # Pick the largest blob that falls within a button boundary (X and Y)
    lit_led = None
    led_position = None
    led_method = None  # Track which method detected the LED (brightness/blob/center)
    best_area = 0

    for blob_x, blob_y, area in valid_blobs:
        # Check if blob is inside any button zone (both X and Y)
        for left_x, right_x, top_y, bottom_y, name in button_zones:
            if (left_x <= blob_x <= right_x and
                top_y <= blob_y <= bottom_y and
                area > best_area):
                best_area = area
                lit_led = name
                led_position = (blob_x + btn_left, blob_y + btn_top)

    # Primary detection: brightness-based using blue channel
    # LEDs are blue - using blue channel instead of grayscale gives much better contrast
    # (grayscale dilutes blue LED signal: 255 blue -> ~170 gray, causing wrong detections)
    brightness_gap = 0  # Track gap between brightest and second brightest zone for logging
    if len(button_zones) > 0:
        blue_channel = button_region[:, :, 0]  # Blue channel for LED detection
        zone_brightness = []
        for left_x, right_x, top_y, bottom_y, name in button_zones:
            # Extract zone region
            x1, x2 = int(left_x), int(right_x)
            y1, y2 = int(top_y), int(bottom_y)
            if x1 < x2 and y1 < y2 and x2 <= blue_channel.shape[1] and y2 <= blue_channel.shape[0]:
                zone = blue_channel[y1:y2, x1:x2]
                if zone.size > 0:
                    # Use max brightness in zone (LED is a bright spot)
                    max_bright = int(np.max(zone))
                    zone_brightness.append((name, max_bright, (x1 + x2) // 2, (y1 + y2) // 2))

        # Find the brightest zone - must be significantly brighter than others
        brightness_gap = 0  # Track for logging
        if zone_brightness:
            zone_brightness.sort(key=lambda x: -x[1])  # Sort by brightness descending
            brightest_name, brightest_val, bx, by = zone_brightness[0]
            second_val = zone_brightness[1][1] if len(zone_brightness) > 1 else 0
            brightness_gap = brightest_val - second_val

            # Use brightness detection if clearly bright and gap is significant
            # Thresholds tuned for blue channel: >200 brightness, >30 gap
            # (higher than grayscale thresholds to avoid digit glow false positives)
            if brightest_val > 200 and brightness_gap > 30:
                lit_led = brightest_name
                led_position = (bx + btn_left, by + btn_top)
                led_method = 'brightness'

    # Fallback to blob detection if brightness didn't find anything
    if lit_led is None and best_area > 0:
        # Use the blob detection result
        for blob_x, blob_y, area in valid_blobs:
            for left_x, right_x, top_y, bottom_y, name in button_zones:
                if (left_x <= blob_x <= right_x and
                    top_y <= blob_y <= bottom_y and
                    area == best_area):
                    lit_led = name
                    led_position = (blob_x + btn_left, blob_y + btn_top)
                    led_method = 'blob'
                    break
            if lit_led:
                break

    # Fallback to bright center detection if still nothing found
    # Lit LED buttons have bright center (LED glow) - find zone with brightest center
    if lit_led is None and len(button_zones) > 0:
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
            zone_centers.sort(key=lambda x: x[0], reverse=True)  # Sort by center brightness (brightest first)
            brightest_center, brightest_name, brightest_coords = zone_centers[0]
            second_center = zone_centers[1][0]
            gap = brightest_center - second_center
            # Require gap (>5) and bright center (>220) to detect
            if gap > 5 and brightest_center > 220:
                lit_led = brightest_name
                x1, y1, x2, y2 = brightest_coords
                led_position = ((x1 + x2) // 2 + btn_left, (y1 + y2) // 2 + btn_top)
                led_method = 'center'

    if lit_led:
        leds[lit_led] = True

    # Track LED detection failures when using cached zones
    if used_cache:
        if lit_led is None:
            _cache_led_fail_count += 1
        else:
            _cache_led_fail_count = 0  # Reset on success

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


def draw_led_debug(frame, led_debug_info):
    """Draw LED detection debug info on frame.

    Args:
        frame: BGR image to draw on (modified in place)
        led_debug_info: Debug info dict from detect_button_leds(return_debug=True)
    """
    if led_debug_info is None:
        return

    btn_left, btn_top, btn_right, btn_bottom = led_debug_info['region']
    button_zones = led_debug_info['zones']
    buttons = led_debug_info['buttons']
    predicted_b1_box = led_debug_info.get('predicted_b1_box')
    led_position = led_debug_info['led_position']
    lit_led = led_debug_info['lit_led']
    leds = led_debug_info['leds']

    # Draw button region boundary
    cv2.rectangle(frame, (btn_left, btn_top), (btn_right, btn_bottom),
                  (100, 100, 100), 1)

    # Draw LED zones (boundaries with X and Y constraints)
    for left_x, right_x, top_y, bottom_y, name in button_zones:
        lx = int(left_x) + btn_left
        rx = int(right_x) + btn_left
        ty = int(top_y) + btn_top
        by = int(bottom_y) + btn_top
        color = (0, 255, 0) if leds[name] else (128, 128, 128)
        cv2.rectangle(frame, (lx, ty), (rx, by), color, 1)
        cv2.putText(frame, name, (lx + 5, ty + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Draw detected buttons (B2, S1, S2) with solid boxes
    for bx, by, bw_btn, bh_btn in buttons:
        cv2.rectangle(frame,
                      (bx + btn_left, by + btn_top),
                      (bx + btn_left + bw_btn, by + btn_top + bh_btn),
                      (255, 255, 0), 1)

    # Draw predicted B1 box with dashed line
    if predicted_b1_box:
        bx, by, bw_btn, bh_btn = predicted_b1_box[:4]
        _draw_dashed_rect(frame,
                          (bx + btn_left, by + btn_top),
                          (bx + btn_left + bw_btn, by + btn_top + bh_btn),
                          (0, 255, 255), 1)  # Yellow dashed for predicted B1

    # Draw LED position
    if led_position:
        cv2.circle(frame, led_position, 8, (0, 255, 0), 2)
        cv2.putText(frame, f"{lit_led}:ON",
                    (led_position[0] - 20, led_position[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)


def draw_mute_debug(frame, mute_debug_info):
    """Draw MUTE LED detection debug info on frame.

    Args:
        frame: BGR image to draw on (modified in place)
        mute_debug_info: Debug info dict from detect_red_button(return_debug=True)
    """
    if mute_debug_info is None:
        return

    region_left, region_top, region_right, region_bottom = mute_debug_info['region']
    is_lit = mute_debug_info['is_lit']
    led_center = mute_debug_info['led_center']
    red_pixels = mute_debug_info['red_pixels']

    # Draw search region boundary
    color = (0, 0, 255) if is_lit else (0, 0, 128)  # Bright red if lit, dark red otherwise
    cv2.rectangle(frame, (region_left, region_top),
                  (region_right, region_bottom), color, 1)

    # Only draw circle and MUTE:ON when actually lit (matches MUTE detection behavior)
    if is_lit and led_center:
        cv2.circle(frame, led_center, 8, (0, 0, 255), 2)
        cv2.putText(frame, "MUTE:ON",
                    (led_center[0] - 25, led_center[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    else:
        # Show pixel count at region top when not lit
        cv2.putText(frame, f"MUTE({red_pixels}px)",
                    (region_left, region_top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 128), 1)


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
        corner_x, corner_y, match_score = corner_result
        # Calculate red button region relative to corner using geometry model
        mute = _geometry.get_mute_region(corner_x, corner_y)
        btn_x, btn_y, region_half = mute

        # Define search region around expected button location
        region_left = max(0, btn_x - region_half)
        region_right = min(w_frame, btn_x + region_half)
        region_top = max(0, btn_y - region_half)
        region_bottom = min(h_frame, btn_y + region_half)
        method = "corner"
    else:
        # Fallback: use fixed region in lower right from geometry model
        region_left, region_right, region_top, region_bottom = _geometry.get_mute_fallback_region(
            w_frame, h_frame)
        method = "fallback"

    # Extract search region
    region = frame[region_top:region_bottom, region_left:region_right]

    # Compute channel medians (used by both brightness fallback and color normalization)
    med_r = np.median(region[:, :, 2])
    med_g = np.median(region[:, :, 1])

    # Dark-region brightness fallback: when mute region is dark,
    # a lit LED creates an obvious bright spot (high max vs low mean).
    # This avoids color normalization issues that kill the red signal at night.
    if med_g < 60:
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        region_mean = np.mean(gray_region)
        region_max = int(np.max(gray_region))
        brightness_gap = region_max - region_mean

        if brightness_gap > 100:
            # Dark region with bright spot — LED is lit
            is_lit = True
            red_pixels = int(np.sum(gray_region > (region_mean + brightness_gap * 0.5)))
            is_single_red_blob = True  # Bypass MUTE_NA check in live_demo.py
            bright_method = method + "_bright"

            if return_debug:
                led_center = None
                bright_mask = (gray_region > (region_mean + brightness_gap * 0.5)).astype(np.uint8)
                coords = np.where(bright_mask > 0)
                if len(coords[0]) > 0:
                    cy = int(np.mean(coords[0])) + region_top
                    cx = int(np.mean(coords[1])) + region_left
                    led_center = (cx, cy)
                debug_info = {
                    'region': (region_left, region_top, region_right, region_bottom),
                    'method': bright_method,
                    'red_pixels': red_pixels,
                    'is_lit': is_lit,
                    'is_single_red_blob': is_single_red_blob,
                    'led_center': led_center,
                    'red_bias': 0,
                    'cluster_density': 1.0,
                    'is_clustered': True
                }
                return is_lit, debug_img, debug_info
            if debug:
                return is_lit, debug_img
            return is_lit, None

    # Color normalization: remove red tint from video artifacts
    # Always correct if red channel is higher than green (common artifact)
    # Real LED is bright enough to survive aggressive correction
    red_bias = 0
    if med_r > med_g:
        # Subtract excess red from entire region
        red_bias = med_r - med_g
        region_corrected = region.copy()
        region_corrected[:, :, 2] = np.clip(region[:, :, 2].astype(np.int16) - int(red_bias), 0, 255).astype(np.uint8)
    else:
        region_corrected = region

    # Detect LED pixels (red or white/saturated for overexposed LEDs)
    # Also filters for bulb-like shapes
    led_mask = _detect_red_pixels(region_corrected)

    # Count LED pixels
    red_pixels = np.sum(led_mask > 0)

    # Spatial clustering check: real LED is a tight cluster, artifact is scattered
    is_clustered = True
    cluster_density = 1.0
    if red_pixels > 15:
        coords = np.where(led_mask > 0)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
            cluster_density = red_pixels / bbox_area
            # Real LED: density > 0.3 (tight cluster)
            # Artifact: density < 0.3 (scattered across region)
            is_clustered = cluster_density > 0.3

    # Threshold: need at least 15 pixels AND clustered to consider LED lit
    # (LED typically 20-40 pixels, lowered to 15 for stable detection with fluctuation)
    is_lit = red_pixels >= 15 and is_clustered

    # Single red blob check: real LED is one compact blob with red hue (H=150-180)
    # Used by live_demo.py to avoid MUTE_NA for real LED at high pixel counts
    is_single_red_blob = False
    if red_pixels >= 15:
        hsv_region = cv2.cvtColor(region_corrected, cv2.COLOR_BGR2HSV)
        num_labels, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(led_mask, connectivity=8)
        blob_count = num_labels - 1  # exclude background
        if blob_count == 1:
            # Single blob — check if hue is red (H=150-180)
            blob_hues = hsv_region[led_mask > 0, 0]
            red_hue_ratio = np.sum((blob_hues >= 150) & (blob_hues <= 180)) / len(blob_hues)
            is_single_red_blob = red_hue_ratio > 0.6
        elif blob_count > 1:
            # Multiple blobs — check largest blob
            largest_label = 1 + np.argmax(stats_cc[1:, cv2.CC_STAT_AREA])
            largest_mask = (labels_cc == largest_label)
            blob_hues = hsv_region[largest_mask, 0]
            red_hue_ratio = np.sum((blob_hues >= 150) & (blob_hues <= 180)) / len(blob_hues)
            largest_area = stats_cc[largest_label, cv2.CC_STAT_AREA]
            # Largest blob must dominate (>70% of total) and be red
            is_single_red_blob = (largest_area / red_pixels > 0.7) and (red_hue_ratio > 0.6)

    # Build debug info for return_debug mode
    debug_info = None
    if return_debug:
        # Find center of LED pixels if any
        led_center = None
        if red_pixels > 0:
            coords = np.where(led_mask > 0)
            cy = int(np.mean(coords[0])) + region_top
            cx = int(np.mean(coords[1])) + region_left
            led_center = (cx, cy)

        debug_info = {
            'region': (region_left, region_top, region_right, region_bottom),
            'method': method,
            'red_pixels': red_pixels,
            'is_lit': is_lit,
            'is_single_red_blob': is_single_red_blob,
            'led_center': led_center,
            'red_bias': round(red_bias, 1),
            'cluster_density': round(cluster_density, 3),
            'is_clustered': is_clustered
        }

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

        # Find center of LED pixels if any
        if red_pixels > 0:
            coords = np.where(led_mask > 0)
            cy = int(np.mean(coords[0])) + region_top
            cx = int(np.mean(coords[1])) + region_left

            color = (0, 255, 0) if is_lit else (0, 0, 255)
            cv2.circle(debug_img, (cx, cy), 10, color, 2)
            cv2.putText(debug_img, f"MUTE:{'ON' if is_lit else 'OFF'}",
                        (cx - 30, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Show method and pixel count
        cv2.putText(debug_img, f"{method} led_px={red_pixels}",
                    (region_left, region_top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

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
            w > 20 and h > 30 and  # Real buttons are ~45-50px tall
            x > 5 and x + w < bw - 5 and
            y > 5):  # Exclude detections at top edge
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
            # Check if overlapping or too close (within 20px)
            if btn[0] < last[0] + last[2] + 20:
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


def is_glowing_panel(corrected_img):
    """Detect if panel image has excessive glow (too bright).

    Returns True if tight blue pixel ratio > 35% of panel area.
    """
    blue_mask = get_blue_mask(corrected_img, tight=True)
    panel_area = corrected_img.shape[0] * corrected_img.shape[1]
    blue_ratio = np.sum(blue_mask > 0) / panel_area
    return blue_ratio > 0.35


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


# 7-segment display patterns
# Each digit maps to the set of segments that should be ON
#
#  Segment layout:
#     AAA
#    F   B
#     GGG
#    E   C
#     DDD
#
SEGMENT_PATTERNS = {
    '0': {'A', 'B', 'C', 'D', 'E', 'F'},
    '1': {'B', 'C'},
    '2': {'A', 'B', 'D', 'E', 'G'},
    '3': {'A', 'B', 'C', 'D', 'G'},
    '4': {'B', 'C', 'F', 'G'},
    '5': {'A', 'C', 'D', 'F', 'G'},
    '6': {'A', 'C', 'D', 'E', 'F', 'G'},
    '7': {'A', 'B', 'C'},
    '8': {'A', 'B', 'C', 'D', 'E', 'F', 'G'},
    '9': {'A', 'B', 'C', 'D', 'F', 'G'},
    'P': {'A', 'B', 'E', 'F', 'G'},
}


def get_segment_zones(box_w, box_h):
    """
    Define the 7 segment sampling zones within a digit box.

    Samples small regions at the CENTER of where each segment should be,
    not the full segment area. This avoids false positives from adjacent segments.

    Args:
        box_w, box_h: Dimensions of the digit box

    Returns:
        dict mapping segment name to (x, y, w, h) sampling region
    """
    # Sample sizes for HORIZONTAL segments (A, G, D) - wide and short
    horiz_w = max(4, int(box_w * 0.20))
    horiz_h = max(4, int(box_h * 0.08))

    # Sample sizes for VERTICAL segments (B, C, E, F) - narrow and tall
    # Make taller to catch vertical segments better
    vert_w = max(4, int(box_w * 0.15))
    vert_h = max(6, int(box_h * 0.18))

    # Key positions - centered zones
    center_x = box_w // 2 - horiz_w // 2
    left_x = int(box_w * 0.08)
    # Right side: mirror of left position (0.08 margin from right edge)
    right_x = box_w - int(box_w * 0.08) - vert_w

    # Vertical positions for horizontal segments
    top_y = int(box_h * 0.05)
    mid_y = box_h // 2 - horiz_h // 2
    # Position D at the very bottom
    bottom_y = box_h - horiz_h - 2

    # Vertical positions for vertical segments (at 1/4 and 3/4 height)
    upper_y = int(box_h * 0.25) - vert_h // 2
    lower_y = int(box_h * 0.75) - vert_h // 2

    # G zone needs to be narrower to avoid glow from vertical segments B, C, E, F
    # When G is off (like in "0"), glow from vertical segments can bleed into the edges
    g_w = max(4, int(box_w * 0.12))  # Narrower than A/D
    g_x = box_w // 2 - g_w // 2

    zones = {
        # Horizontal segments - sample at center (wide, short)
        'A': (center_x, top_y, horiz_w, horiz_h),
        'G': (g_x, mid_y, g_w, horiz_h),  # Narrower zone to avoid edge glow
        'D': (center_x, bottom_y, horiz_w, horiz_h),

        # Vertical segments - left side (narrow, tall)
        'F': (left_x, upper_y, vert_w, vert_h),
        'E': (left_x, lower_y, vert_w, vert_h),

        # Vertical segments - right side (narrow, tall)
        'B': (right_x, upper_y, vert_w, vert_h),
        'C': (right_x, lower_y, vert_w, vert_h),
    }

    return zones


def recognize_digit(digit_img, debug=False):
    """
    Step 5: Recognize a single digit using template matching.
    Falls back to 7-segment analysis if template matching fails.

    Args:
        digit_img: BGR image of a single digit box
        debug: If True, return debug image showing segment zones

    Returns:
        digit: Recognized character ('0'-'9', 'P') or 'X' if unknown
        debug_img: (only if debug=True) Image showing segment analysis
    """
    # Try template matching first (more robust)
    digit, score = recognize_digit_template(digit_img)
    if score > 0.6:  # High confidence threshold
        if debug:
            h, w = digit_img.shape[:2]
            debug_img = digit_img.copy()
            cv2.putText(debug_img, f'{digit}({score:.2f})', (5, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            return digit, debug_img
        return digit, None

    # Fall back to segment-based analysis
    h, w = digit_img.shape[:2]

    # Convert to grayscale for intensity analysis
    gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)

    # Check if this is a glowing image using brightness analysis
    # Glowing images have very high brightness and/or many very bright pixels
    mean_gray = np.mean(gray)
    std_gray = np.std(gray)
    very_bright_ratio = np.sum(gray > 200) / gray.size
    is_glowing = mean_gray > 100 or very_bright_ratio > 0.20

    # Get masks - for glowing images, use original for bounds, preprocessed for scoring
    blue_mask_tight = get_blue_mask(digit_img, tight=True)
    blue_mask_loose = get_blue_mask(digit_img, tight=False)

    if is_glowing:
        # Use original tight mask for content bounds (preserves digit extent)
        blue_mask = blue_mask_tight if np.sum(blue_mask_tight > 0) >= 20 else blue_mask_loose
        # Use preprocessed mask for segment scoring (reduces glow interference)
        preprocessed = preprocess_glowing_image(digit_img)
        blue_mask_full = get_blue_mask(preprocessed, tight=False)
        blue_mask_loose = blue_mask_full
    else:
        blue_mask = blue_mask_tight
        if np.sum(blue_mask_tight > 0) < 20:
            blue_mask = blue_mask_loose
            blue_mask_full = blue_mask_loose
        else:
            blue_mask_full = blue_mask_tight

    # Find content bounds using appropriate mask
    coords = np.where(blue_mask > 0)

    if len(coords[0]) < 10:
        # Try with loose mask if current mask finds nothing
        blue_mask = get_blue_mask(digit_img, tight=False)
        coords = np.where(blue_mask > 0)

    if len(coords[0]) < 10:
        # No significant content
        if debug:
            debug_img = digit_img.copy()
            cv2.putText(debug_img, 'X', (5, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return 'X', debug_img
        return 'X', None

    # Get content bounding box from blue mask
    content_top = max(0, np.min(coords[0]))
    content_bottom = min(h, np.max(coords[0]))
    content_left = max(0, np.min(coords[1]))
    content_right = min(w, np.max(coords[1]))

    content_w = content_right - content_left
    content_h = content_bottom - content_top

    # Special case: very narrow content is likely "1"
    # Use threshold 0.20 - narrow enough to catch "1" but not "3"
    if content_w < content_h * 0.20:
        if debug:
            debug_img = digit_img.copy()
            cv2.rectangle(debug_img, (content_left, content_top),
                          (content_right, content_bottom), (0, 255, 255), 1)
            cv2.putText(debug_img, '1', (5, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return '1', debug_img
        return '1', None

    # For glowing images, recalculate content bounds from preprocessed mask
    # This ensures zones align with where the preprocessed mask has content
    if is_glowing:
        pp_coords = np.where(blue_mask_full > 0)
        if len(pp_coords[0]) >= 10:
            content_top = max(0, np.min(pp_coords[0]))
            content_bottom = min(h, np.max(pp_coords[0]))
            content_left = max(0, np.min(pp_coords[1]))
            content_right = min(w, np.max(pp_coords[1]))
            content_w = content_right - content_left
            content_h = content_bottom - content_top

    # Get segment zones based on content dimensions
    zones = get_segment_zones(content_w, content_h)

    # Offset zones to content position
    offset_zones = {}
    for seg_name, (sx, sy, sw, sh) in zones.items():
        offset_zones[seg_name] = (sx + content_left, sy + content_top, sw, sh)
    zones = offset_zones

    # Calculate blue pixel ratio for each segment zone (both tight and loose)
    segment_intensities = {}
    segment_blue_ratios = {}
    segment_blue_ratios_loose = {}  # For B/C fallback check
    g_center_ratio = 0.0  # Special check for G segment
    for seg_name, (sx, sy, sw, sh) in zones.items():
        # Ensure zone is within bounds
        sx, sy = max(0, int(sx)), max(0, int(sy))
        sw = min(int(sw), w - sx)
        sh = min(int(sh), h - sy)

        if sw > 0 and sh > 0:
            zone_gray = gray[sy:sy+sh, sx:sx+sw]
            zone_blue = blue_mask_full[sy:sy+sh, sx:sx+sw]
            zone_blue_loose = blue_mask_loose[sy:sy+sh, sx:sx+sw]
            segment_intensities[seg_name] = np.mean(zone_gray)
            # Ratio of blue pixels in zone
            zone_area = sw * sh if sw * sh > 0 else 1
            segment_blue_ratios[seg_name] = np.sum(zone_blue > 0) / zone_area
            segment_blue_ratios_loose[seg_name] = np.sum(zone_blue_loose > 0) / zone_area

            # Special check for G segment: verify blue pixels are in center, not edges
            # Edge glow from vertical segments would concentrate on left/right edges
            if seg_name == 'G' and sw > 4:
                center_third = sw // 3
                center_zone = zone_blue[:, center_third:sw-center_third]
                edge_left = zone_blue[:, :center_third]
                edge_right = zone_blue[:, sw-center_third:]
                center_pixels = np.sum(center_zone > 0)
                edge_pixels = np.sum(edge_left > 0) + np.sum(edge_right > 0)
                total_pixels = center_pixels + edge_pixels
                if total_pixels > 0:
                    g_center_ratio = center_pixels / total_pixels
                else:
                    g_center_ratio = 0.0
        else:
            segment_intensities[seg_name] = 0
            segment_blue_ratios[seg_name] = 0
            segment_blue_ratios_loose[seg_name] = 0

    # Find the intensity range to set adaptive threshold
    intensities = list(segment_intensities.values())
    max_intensity = max(intensities) if intensities else 0
    min_intensity = min(intensities) if intensities else 0

    # Also check blue ratios
    blue_ratios = list(segment_blue_ratios.values())
    max_blue_ratio = max(blue_ratios) if blue_ratios else 0

    # If all segments have similar low intensity AND no blue pixels, no digit present
    if max_intensity < 50 and max_blue_ratio < 0.1:
        if debug:
            debug_img = digit_img.copy()
            cv2.putText(debug_img, 'X', (5, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return 'X', debug_img
        return 'X', None

    # Soft scoring: combine intensity and tight blue ratio
    # Intensity gives base signal, tight ratio confirms if it's real segment vs glow
    intensity_range = max_intensity - min_intensity

    segment_scores = {}
    for seg in segment_intensities:
        # Normalize intensity to 0-1
        if intensity_range > 0:
            int_score = (segment_intensities[seg] - min_intensity) / intensity_range
        else:
            int_score = 0.5

        tight_ratio = segment_blue_ratios[seg]
        loose_ratio = segment_blue_ratios_loose[seg]

        # Special handling for G segment: check if blue pixels are in center vs edges
        # Edge glow from vertical segments B/C/E/F would concentrate on edges
        if seg == 'G' and g_center_ratio < 0.3 and tight_ratio < 0.5:
            # Blue pixels mostly on edges - likely glow from vertical segments
            score = min(int_score, 0.2)
        # Adjust based on tight ratio
        elif tight_ratio > 0.8:
            # Very high tight ratio - segment is definitely lit
            # Intensity variation is just due to lighting/angle, not glow
            score = 0.9
        elif tight_ratio > 0.5:
            # High tight ratio - likely lit
            score = max(int_score, 0.75)
        elif tight_ratio > 0.25:
            # Medium-high tight ratio - definitely lit, boost significantly
            score = max(int_score, 0.7)
        elif tight_ratio > 0.10:
            # Medium tight ratio - possibly lit, boost to help detection
            # Lowered from 0.15 to 0.10 to handle lower-contrast camera setups
            score = max(int_score, 0.55)
        elif tight_ratio < 0.02:
            # Very low tight ratio - segment is off
            score = min(int_score, 0.2)
        else:
            # Use intensity as-is
            score = int_score

        segment_scores[seg] = score

    # Score each digit pattern using soft matching
    best_match = 'X'
    best_score = -100
    second_score = -100

    for digit, pattern in SEGMENT_PATTERNS.items():
        score = 0
        for seg_name in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            seg_value = segment_scores[seg_name]
            if seg_name in pattern:
                # Segment should be LIT: higher value = higher score
                score += seg_value
            else:
                # Segment should be OFF: lower value = higher score
                score += (1.0 - seg_value)

        if score > best_score:
            second_score = best_score
            best_score = score
            best_match = digit
        elif score > second_score:
            second_score = score

    # Determine lit segments for debug visualization (threshold at 0.5)
    threshold = 0.5
    lit_segments = {seg for seg, val in segment_scores.items() if val > threshold}

    if debug:
        debug_img = digit_img.copy()

        # Draw content bounds (yellow)
        cv2.rectangle(debug_img, (content_left, content_top),
                      (content_right, content_bottom), (0, 255, 255), 1)

        # Draw segment zones with color indicating lit/unlit
        for seg_name, (sx, sy, sw, sh) in zones.items():
            sx, sy, sw, sh = int(sx), int(sy), int(sw), int(sh)
            is_lit = seg_name in lit_segments
            intensity = segment_intensities[seg_name]
            color = (0, 255, 0) if is_lit else (0, 0, 255)  # Green=lit, Red=off
            cv2.rectangle(debug_img, (sx, sy), (sx + sw, sy + sh), color, 1)
            # Show intensity value
            cv2.putText(debug_img, f"{int(intensity)}", (sx+2, sy+sh-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

        # Add recognized digit and threshold info
        cv2.putText(debug_img, f"{best_match} t={int(threshold)}", (5, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return best_match, debug_img

    return best_match, None


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

    # Search from center following the slope to find valley bottom (U-shape)
    center = len(smoothed) // 2
    search_limit = int(len(smoothed) * 0.15)  # Don't go beyond 35%-65% range

    # Determine search direction based on slope at center
    # Follow the downward slope to find the true valley bottom
    search_left = smoothed[center - 1] < smoothed[center + 1]

    # Search in one direction only, following the slope
    gap_x = center
    if search_left:
        for x in range(center - 1, center - search_limit, -1):
            if x > 0 and smoothed[x] < smoothed[x - 1] and smoothed[x] < smoothed[x + 1]:
                # Found local minimum - this is valley bottom
                gap_x = x
                break
            # Track darkest point in case no local minimum found
            if smoothed[x] < smoothed[gap_x]:
                gap_x = x
    else:
        for x in range(center + 1, center + search_limit):
            if x < len(smoothed) - 1 and smoothed[x] < smoothed[x - 1] and smoothed[x] < smoothed[x + 1]:
                # Found local minimum - this is valley bottom
                gap_x = x
                break
            # Track darkest point in case no local minimum found
            if smoothed[x] < smoothed[gap_x]:
                gap_x = x

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

        # Draw gap line (yellow, full height)
        cv2.line(debug_img, (gap_x, 0), (gap_x, h), (0, 255, 255), 2)

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

        # Draw gap line (yellow)
        cv2.line(debug_img, (gap_x, 0), (gap_x, h), (0, 255, 255), 1)

        return left_box, right_box, debug_img

    return left_box, right_box, None


class SegmentReader:
    """
    Step 4: Adaptive caching for efficient frame processing.

    Caches panel detection and slant angle to avoid recomputing on every frame.
    Only updates cache when scene changes significantly.
    """

    def __init__(self, cache_ttl=100):
        """
        Args:
            cache_ttl: Maximum frames before forcing cache refresh
        """
        self.cache_ttl = cache_ttl

        # Cached values
        self._panel_rect = None
        self._gap_x = None  # Gap position between digits
        self._left_box = None  # Left digit bounding box
        self._right_box = None  # Right digit bounding box
        self._frames_since_update = 0
        self._last_reading = None  # Last successful reading
        self._last_scores = (0.0, 0.0)  # Last match scores (left, right)
        self._last_second = (('X', 0.0), ('X', 0.0))  # Second best candidates ((digit, score), (digit, score))
        self._last_digit_debug = None  # Debug info for digit matching
        self._detection_method = None  # Panel detection method used
        self._brightness_conf = None  # Brightness fallback confidence score
        self._dim_enhanced = None  # Dim digit enhancement status (L/R/LR/None)

        # Frame diff optimization: skip processing if frame unchanged
        self._prev_frame_roi = None  # Previous frame ROI for diff comparison
        self._prev_reading = None  # Previous reading to reuse
        self._prev_panel_rect = None  # Previous panel rect
        self._frame_skipped = False  # Whether current frame was skipped
        self._frame_diff_threshold = 100000  # Diff threshold for skip (ignores noise 0-4)
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
        _update_panel_cache(
            panel_rect=self._panel_rect,
            gap_x=self._gap_x,
            left_box=self._left_box,
            right_box=self._right_box,
            last_reading=self._last_reading
        )

    def load_cache(self):
        """Load cache from unified cache file if it exists."""
        cache_data = _get_panel_cache()
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
        panel_img = _geometry.undistort_roi(frame, x, y, w, h)

        # Compute boxes (slant is always fixed at 8.0 degrees)
        corrected_img, _, _ = correct_slant(panel_img, 8.0)
        gap_x, _ = find_digit_gap(corrected_img)
        left_box, right_box, _ = define_digit_boxes(corrected_img, gap_x)

        # Update cache
        self._panel_rect = panel_rect
        self._gap_x = gap_x
        self._left_box = left_box
        self._right_box = right_box
        self._frames_since_update = 0

        # Persist cache to file
        self.save_cache()

        return True

    def _try_read(self, frame, verify_panel=False):
        """
        Try to read digits using cached panel rect and slant angle.

        Args:
            frame: BGR image
            verify_panel: If True, also check that fresh detection gives similar panel

        Returns:
            reading: 2-character string, or None if cache invalid
        """
        if self._panel_rect is None:
            return None

        x, y, w, h = self._panel_rect

        # Check bounds (including negative coords and zero area)
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return None
        if y + h > frame.shape[0] or x + w > frame.shape[1]:
            return None

        # Verify cached panel matches fresh detection
        if verify_panel:
            fresh_panel, _ = detect_panel(frame)
            if fresh_panel is None:
                return None
            fx, fy, fw, fh = fresh_panel
            # Check if panels are similar (within 20% size and 10% position)
            size_diff = abs(fw * fh - w * h) / (w * h)
            pos_diff = (abs(fx - x) + abs(fy - y)) / max(w, h)
            if size_diff > 0.2 or pos_diff > 0.1:
                return None

        panel_img = _geometry.undistort_roi(frame, x, y, w, h)
        corrected_img, _, _ = correct_slant(panel_img, 8.0)

        gap_x, _ = find_digit_gap(corrected_img)
        left_box, right_box, _ = define_digit_boxes(corrected_img, gap_x)

        left_digit_img = _extract_digit_with_padding(corrected_img, left_box, right_bound=gap_x)
        right_digit_img = _extract_digit_with_padding(corrected_img, right_box, left_bound=gap_x)

        left_digit, left_score = recognize_digit_template(left_digit_img)
        right_digit, right_score = recognize_digit_template(right_digit_img)
        self._last_scores = (left_score, right_score)

        reading = left_digit + right_digit

        # If recognition failed, cache is invalid
        if 'X' in reading:
            return None

        return reading

    def read(self, frame):
        """
        Read the 2-digit value from frame - all fresh detection, no caching.

        Args:
            frame: BGR image from camera/file

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
            current_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
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
        panel_rect, detection_method, brightness_conf = detect_panel(frame, return_confidence=True)
        self._detection_method = detection_method  # Store for logging
        self._brightness_conf = brightness_conf  # Store brightness confidence
        if panel_rect is None:
            log_issue_frame(frame, 'panel_fail')
            return "XX", False

        x, y, w, h = panel_rect
        panel_img = _geometry.undistort_roi(frame, x, y, w, h)

        # Process with fixed 8.0 degree slant
        corrected_img, _, _ = correct_slant(panel_img, 8.0)
        gap_x, _ = find_digit_gap(corrected_img)
        left_box, right_box, _ = define_digit_boxes(corrected_img, gap_x)

        left_digit_img = _extract_digit_with_padding(corrected_img, left_box, right_bound=gap_x)
        right_digit_img = _extract_digit_with_padding(corrected_img, right_box, left_bound=gap_x)

        # Quick-check optimization: check only the best 2 templates first
        # Full search triggered if: 1st drops >2% OR 2nd changes >2% OR 3 min elapsed
        CONFIDENCE_THRESHOLD = 0.02

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

        # Left digit recognition with quick-check
        if force_full_scan or self._left_best_templates is None:
            # Full search: periodic rescan or first frame
            left_digit, left_score, left_debug = recognize_digit_template(
                left_digit_img, return_debug=True)
        else:
            (d1, idx1, score1), (d2, idx2, score2) = self._left_best_templates
            # Quick check: match only the 2 specific templates
            new_score1, match_pos1, template_size1 = match_single_template(left_gray, d1, idx1)
            new_score2, _, _ = match_single_template(left_gray, d2, idx2)

            # Check if stable: 1st not dropped >2%, 2nd not changed >2%
            need_full = False
            if score1 - new_score1 > CONFIDENCE_THRESHOLD:
                need_full = True  # 1st dropped
            if abs(new_score2 - score2) > CONFIDENCE_THRESHOLD:
                need_full = True  # 2nd changed

            if need_full:
                left_digit, left_score, left_debug = recognize_digit_template(
                    left_digit_img, return_debug=True)
            else:
                # Use quick-check result
                left_digit, left_score = d1, new_score1
                left_debug = {
                    'second_digit': d2, 'second_score': new_score2,
                    'best_template_idx': idx1, 'second_template_idx': idx2,
                    'match_pos': match_pos1, 'template_size': template_size1,
                }

        # Right digit recognition with quick-check
        if force_full_scan or self._right_best_templates is None:
            # Full search: periodic rescan or first frame
            right_digit, right_score, right_debug = recognize_digit_template(
                right_digit_img, return_debug=True)
        else:
            (d1, idx1, score1), (d2, idx2, score2) = self._right_best_templates
            # Quick check: match only the 2 specific templates
            new_score1, match_pos1, template_size1 = match_single_template(right_gray, d1, idx1)
            new_score2, _, _ = match_single_template(right_gray, d2, idx2)

            # Check if stable: 1st not dropped >2%, 2nd not changed >2%
            need_full = False
            if score1 - new_score1 > CONFIDENCE_THRESHOLD:
                need_full = True  # 1st dropped
            if abs(new_score2 - score2) > CONFIDENCE_THRESHOLD:
                need_full = True  # 2nd changed

            if need_full:
                right_digit, right_score, right_debug = recognize_digit_template(
                    right_digit_img, return_debug=True)
            else:
                # Use quick-check result
                right_digit, right_score = d1, new_score1
                right_debug = {
                    'second_digit': d2, 'second_score': new_score2,
                    'best_template_idx': idx1, 'second_template_idx': idx2,
                    'match_pos': match_pos1, 'template_size': template_size1,
                }

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

        left_ambiguous = left_score < 0.95 and (left_score - left_2nd_score) < 0.05
        right_ambiguous = right_score < 0.95 and (right_score - right_2nd_score) < 0.05

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
        self._frames_since_update = 0
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
    def brightness_conf(self):
        """Get brightness fallback confidence score, or None if not using brightness method."""
        return getattr(self, '_brightness_conf', None)

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


def test_on_image(image_path):
    """Test panel detection and digit recognition pipeline on a single image.

    Runs the complete recognition pipeline:
    1. Panel detection (corner-based or brightness fallback)
    2. LED state detection
    3. Slant correction (fixed 8.0 degrees)
    4. Digit gap detection and box definition
    5. Template-based digit recognition

    Saves debug images for each step to the debug/ directory.

    Args:
        image_path: Path to the input image file
    """
    print(f"Testing: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  ERROR: Could not load image")
        return

    # Step 1: Panel detection
    panel_rect, debug_img = detect_panel(frame)

    if not panel_rect:
        print(f"  Panel NOT detected")
        return

    x, y, w, h = panel_rect
    print(f"  Panel detected: x={x}, y={y}, w={w}, h={h}")

    # LED detection
    leds, led_debug = detect_button_leds(frame, panel_rect, debug=True)
    lit_leds = [k for k, v in leds.items() if v]
    print(f"  LED: {lit_leds[0] if lit_leds else 'None'}")

    # Extract panel region (with distortion correction if available)
    panel_img = _geometry.undistort_roi(frame, x, y, w, h)

    # Step 2: Slant correction (fixed at 8.0 degrees)
    angle = 8.0
    corrected_img, _, slant_debug_img = correct_slant(panel_img, angle)
    print(f"  Slant angle: {angle:.1f} degrees (fixed)")

    # Step 3-1: Find digit gap
    gap_x, gap_debug = find_digit_gap(corrected_img, debug=True)
    print(f"  Gap position: x={gap_x}")

    # Step 3-2: Define digit boxes
    left_box, right_box, boxes_debug = define_digit_boxes(corrected_img, gap_x, debug=True)
    lx, ly, lw, lh = left_box
    rx, ry, rw, rh = right_box
    print(f"  Left box: x={lx}-{lx+lw}, size {lw}x{lh}")
    print(f"  Right box: x={rx}-{rx+rw}, size {rw}x{rh}")

    # Step 5: Recognize digits (with padding for search tolerance)
    left_digit_img = _extract_digit_with_padding(corrected_img, left_box, right_bound=gap_x)
    right_digit_img = _extract_digit_with_padding(corrected_img, right_box, left_bound=gap_x)

    left_digit, left_debug = recognize_digit(left_digit_img, debug=True)
    right_digit, right_debug = recognize_digit(right_digit_img, debug=True)

    reading = left_digit + right_digit
    print(f"  Recognition: {reading}")

    # Save debug images to debug directory
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    debug_dir = "debug"
    os.makedirs(debug_dir, exist_ok=True)

    cv2.imwrite(f"{debug_dir}/{base_name}_step3_1_gap.png", gap_debug)
    cv2.imwrite(f"{debug_dir}/{base_name}_step3_2_boxes.png", boxes_debug)
    cv2.imwrite(f"{debug_dir}/{base_name}_step5_left.png", left_debug)
    cv2.imwrite(f"{debug_dir}/{base_name}_step5_right.png", right_debug)
    cv2.imwrite(f"{debug_dir}/{base_name}_led.png", led_debug)

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
