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


# Unified cache file for button zones and panel detection
_CACHE_FILE = os.path.join(os.path.dirname(__file__), 'last_ref.txt')
_ZONE_CHANGE_THRESHOLD = 10  # Pixels - only save if zones shift by more than this

# Cache for button zone centers (adaptive from detected buttons)
_button_zone_cache = None
# Cache for panel detection (shared with SegmentReader)
_panel_cache = None

# Logging configuration
_LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
_LOG_ENABLED = True
_LOG_COOLDOWN = 30  # Seconds between saves of same issue type
_LOG_MAX_FRAMES = 1000  # Max issue frames to keep
_log_last_save = {}  # issue_type -> timestamp
_log_file = None  # CSV file handle

# Corner template for pattern matching (used for red button detection)
_corner_template = None
_CORNER_TEMPLATE_FILE = os.path.join(os.path.dirname(__file__), 'templates', 'corner_template.png')
# Red button offset from corner center (determined empirically)
_RED_BUTTON_OFFSET = (200, 43)  # (dx, dy) pixels from corner center

# Digit templates for pattern matching
_digit_templates = None
_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')
_TEMPLATE_SIZE = (44, 99)  # (width, height) - matches native digit box size
_DIGIT_PADDING = 15  # Pixels of horizontal padding (left/right) around digit box
_DIGIT_PADDING_V = 10  # Pixels of vertical padding (top/bottom) around digit box
_PANEL_WIDTH = 165  # Fixed panel width from landmark calibration
_PANEL_HEIGHT = 105  # Fixed panel height from landmark calibration

# Auto-learning state (temporal stability)
import threading
_learning_buffer = {}  # digit -> (count, template_img, reason, timestamp)
_learning_lock = threading.Lock()  # Thread safety for _learning_buffer
_LEARNING_THRESHOLD = 60  # Frames of low confidence before learning
_LEARNING_TIMEOUT = 30.0  # Seconds before learning buffer entry expires
_last_auto_learned = None  # (digit, filename) when auto-learning occurs, cleared after display

# =============================================================================
# Detection Thresholds
# =============================================================================
_TEMPLATE_CONFIDENCE_THRESHOLD = 0.80
_TEMPLATE_AMBIGUITY_GAP = 0.05
_MIN_DIGIT_HEIGHT = 10
_MIN_DIGIT_WIDTH = 5

# Panel Detection
_CORNER_TO_PANEL_X = 266
_CORNER_TO_PANEL_Y = 86
_BRIGHTNESS_PERCENTILE = 97
_MIN_BRIGHTNESS_THRESHOLD = 100
_PANEL_MARGIN_TOP_RATIO = 0.15
_PANEL_MARGIN_BOTTOM_RATIO = 0.85

# Button/LED Detection
_BUTTON_REGION_RIGHT_RATIO = 0.65
_BUTTON_REGION_TOP_RATIO = 0.70
_LED_MIN_AREA = 100
_LED_MAX_AREA = 1200
_LED_MAX_ASPECT_RATIO = 3

# Digit Recognition
_NO_DIGIT_MAX_INTENSITY = 50
_NO_DIGIT_MAX_BLUE_RATIO = 0.1
_SEGMENT_LIT_THRESHOLD = 0.5


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


def _detect_red_pixels(image):
    """Detect red pixels using dual HSV ranges.

    Red wraps around hue=0 in HSV, so we need two ranges.

    Args:
        image: BGR image

    Returns:
        Binary mask where red pixels are white (255)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([20, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([150, 30, 30]), np.array([180, 255, 255]))
    return cv2.bitwise_or(mask1, mask2)


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


def get_last_auto_learned():
    """Get and clear the last auto-learned template info.

    Returns:
        (digit, filename) tuple if a template was recently learned, None otherwise.
        Clears the value after returning so each learning event is only returned once.
    """
    global _last_auto_learned
    result = _last_auto_learned
    _last_auto_learned = None
    return result


def _load_digit_templates():
    """Load digit templates from templates directory.

    Supports multiple templates per digit with naming:
    - Manual: digit_0a.png, digit_0b.png, etc.
    - Auto-learned: digit_0a_learn.png, digit_0b_learn.png, etc.
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


def recognize_digit_template(digit_img, auto_learn=False, return_debug=False):
    """
    Recognize a digit using template matching on grayscale image.

    Converts input to grayscale before matching.
    Matches against all templates for each digit and returns the best match.
    Uses sliding window matching to be less sensitive to exact box position.

    Args:
        digit_img: BGR image of single digit (larger than template for search tolerance)
        auto_learn: If True, auto-add templates when confidence is low
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

    # Convert to grayscale (no scaling - match at original size)
    gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)

    # Collect all scores - use sliding window matching
    # Track best template index for each digit
    all_scores = []
    best_match_pos = None
    best_template_size = None
    best_overall_score = -1.0

    for digit, template_list in templates.items():
        best_for_digit = -1.0
        best_idx_for_digit = 0
        for idx, template in enumerate(template_list):
            th, tw = template.shape[:2]
            # Only match if image is large enough for template
            if gray.shape[0] >= th and gray.shape[1] >= tw:
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_for_digit:
                    best_for_digit = max_val
                    best_idx_for_digit = idx
                if max_val > best_overall_score:
                    best_overall_score = max_val
                    best_match_pos = max_loc
                    best_template_size = (tw, th)
        all_scores.append((digit, best_for_digit, best_idx_for_digit))

    # Sort by score descending
    all_scores.sort(key=lambda x: -x[1])

    if not all_scores:
        if return_debug:
            return 'X', 0.0, None
        return 'X', 0.0

    best_digit, best_score, best_template_idx = all_scores[0]
    second_digit, second_score, second_template_idx = all_scores[1] if len(all_scores) > 1 else ('X', 0.0, 0)

    # Check if we need auto-learning
    needs_learning = False
    learn_reason = ""
    if best_score < _TEMPLATE_CONFIDENCE_THRESHOLD:
        needs_learning = True
        learn_reason = f"low_confidence(score={best_score:.3f}<{_TEMPLATE_CONFIDENCE_THRESHOLD})"
    elif (best_score - second_score) < _TEMPLATE_AMBIGUITY_GAP:
        needs_learning = True
        learn_reason = f"ambiguous(best={best_digit}:{best_score:.3f}, second={second_digit}:{second_score:.3f}, gap={best_score-second_score:.3f})"

    if needs_learning and auto_learn:
        # Use temporal stability for learning (no segment-based verification)
        # The best_digit from template matching is used directly
        # Requires CONSECUTIVE frames with same digit
        global _learning_buffer

        with _learning_lock:
            now = time.time()
            # Clear buffer for other digits and expired entries
            for d in list(_learning_buffer.keys()):
                if d != best_digit:
                    del _learning_buffer[d]
                elif len(_learning_buffer[d]) >= 4 and now - _learning_buffer[d][3] > _LEARNING_TIMEOUT:
                    del _learning_buffer[d]  # Expired

            if best_digit in _learning_buffer:
                count, _, _, _ = _learning_buffer[best_digit]
                _learning_buffer[best_digit] = (count + 1, digit_img.copy(), learn_reason, now)

                # Only learn after consistent detection for N consecutive frames
                if count + 1 >= _LEARNING_THRESHOLD:
                    _auto_save_template(best_digit, digit_img, learn_reason)
                    del _learning_buffer[best_digit]  # Reset counter
            else:
                # Start counting
                _learning_buffer[best_digit] = (1, digit_img.copy(), learn_reason, now)
    # Note: Don't clear buffer when confidence is good - left/right digits share the buffer
    # Buffer entries are cleared when a different digit is seen with low confidence

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


def _load_corner_template():
    """Load corner template for pattern matching."""
    global _corner_template
    if _corner_template is None and os.path.exists(_CORNER_TEMPLATE_FILE):
        _corner_template = cv2.imread(_CORNER_TEMPLATE_FILE)
    return _corner_template


def _find_corner(frame, min_match=0.7, return_debug=False):
    """
    Find the corner in the frame using template matching.

    Optimized: Uses center 1/4 of template and searches only right portion of frame.

    Args:
        frame: BGR image
        min_match: Minimum match score (0-1) to consider valid
        return_debug: If True, return debug info for visualization

    Returns:
        (x, y, score): Corner center coordinates and match score, or None if not found
        If return_debug=True, also returns: search_rect, match_rect, template_crop_size
    """
    template = _load_corner_template()
    if template is None:
        return (None, None) if return_debug else None

    th, tw = template.shape[:2]

    # Use right lower 1/4 of template (half width, half height)
    crop_h, crop_w = th // 2, tw // 2
    crop_y, crop_x = th // 2, tw // 2
    template_crop = template[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

    # Search only in 150x150 square region with matched pattern centered
    h_frame, w_frame = frame.shape[:2]
    search_size = 150
    search_left = int(w_frame * 0.58)
    search_top = int(h_frame * 0.57)
    search_region = frame[search_top:search_top+search_size, search_left:search_left+search_size]

    # Search region rect for debug visualization
    search_rect = (search_left, search_top, search_size, search_size)

    result = cv2.matchTemplate(search_region, template_crop, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val < min_match:
        if return_debug:
            return None, (search_rect, None, (crop_w, crop_h))
        return None

    # Match location is top-left of cropped template in search region
    # Convert to center of full template in full frame
    # crop is from right-lower quadrant, so corner center is at crop origin
    corner_x = search_left + max_loc[0]
    corner_y = search_top + max_loc[1]

    # Match rect in frame coordinates (where template was matched)
    match_rect = (search_left + max_loc[0], search_top + max_loc[1], crop_w, crop_h)

    if return_debug:
        return (corner_x, corner_y, max_val), (search_rect, match_rect, (crop_w, crop_h))
    return (corner_x, corner_y, max_val)


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
                           'corner_score,detection_method,mute_status,mute_pixels,issue\n')
            _log_file.flush()
    except (IOError, OSError) as e:
        print(f"Warning: Failed to initialize log: {e}", flush=True)
        if _log_file is not None:
            _log_file.close()
        _log_file = None


def log_detection(panel_rect=None, gap_x=None, left_score=0, right_score=0,
                  reading=None, led_status=None, corner_score=0,
                  detection_method=None, mute_status=None, mute_pixels=0,
                  issue=None):
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
    mute_px = int(mute_pixels) if mute_pixels else 0
    iss = issue if issue is not None else ''

    _log_file.write(f'{ts},{px},{py},{pw},{ph},{gx},'
                   f'{left_score:.3f},{right_score:.3f},{rd},{led},'
                   f'{corner_score:.3f},{method},{mute},{mute_px},{iss}\n')
    _log_file.flush()


def log_issue_frame(frame, issue_type, confidence=0, extra_info=None, display_frame=None):
    """Save frame when detection issue occurs (with cooldown).

    Args:
        frame: Raw camera frame
        issue_type: Type of issue (e.g., 'low_conf', 'ambiguous')
        confidence: Confidence score
        extra_info: Additional info for filename
        display_frame: Optional display window frame with overlays
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

    # Save raw frame
    filepath = os.path.join(_LOG_DIR, f'{base_name}.png')
    if not cv2.imwrite(filepath, frame):
        print(f"Warning: Failed to write frame {filepath}", flush=True)
        return None

    # Save display frame if provided
    if display_frame is not None:
        display_path = os.path.join(_LOG_DIR, f'{base_name}_display.png')
        if not cv2.imwrite(display_path, display_frame):
            print(f"Warning: Failed to write display frame {display_path}", flush=True)

    # Cleanup old frames if too many
    _cleanup_old_frames()

    return filepath


def _cleanup_old_frames():
    """Remove oldest frame pairs if exceeding max count.

    Keeps raw frames and their display counterparts together as pairs.
    """
    try:
        # Count only raw frames (exclude _display.png files)
        raw_frames = sorted([f for f in os.listdir(_LOG_DIR)
                            if f.endswith('.png') and not f.endswith('_display.png')])
        if len(raw_frames) > _LOG_MAX_FRAMES:
            # Remove oldest pairs (raw + display)
            for f in raw_frames[:-_LOG_MAX_FRAMES]:
                try:
                    # Remove raw frame
                    os.remove(os.path.join(_LOG_DIR, f))
                    # Remove corresponding display frame if exists
                    display_f = f.replace('.png', '_display.png')
                    display_path = os.path.join(_LOG_DIR, display_f)
                    if os.path.exists(display_path):
                        os.remove(display_path)
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
    corner_result = _find_corner(frame, min_match=0.7)
    if corner_result is None:
        return None

    corner_x, corner_y, corner_score = corner_result

    # Step 2: Define button search region based on corner
    # Buttons are to the left of the corner, BELOW the corner position
    # Corner is at top-right of device, buttons are at bottom
    btn_search_top = corner_y + 20  # Buttons start below corner
    btn_search_bottom = h_frame
    btn_search_left = 0
    btn_search_right = min(w_frame, corner_x + 50)

    button_region = frame[btn_search_top:btn_search_bottom, btn_search_left:btn_search_right]
    if button_region.shape[0] < 10 or button_region.shape[1] < 10:
        return None

    # Step 3: Detect buttons in the region
    buttons = _detect_buttons(button_region)
    buttons = sorted(buttons, key=lambda b: b[0])  # Sort left to right

    if len(buttons) < 3:
        return None

    # We have B2, S1, S2 (B1 is usually cut off at left edge)
    # Get their positions (in button_region coordinates)
    b2_x, b2_y, b2_w, b2_h = buttons[0]
    s1_x, s1_y, s1_w, s1_h = buttons[1]
    s2_x, s2_y, s2_w, s2_h = buttons[2]

    # Calculate spacing between buttons
    b2_center = b2_x + b2_w // 2
    s1_center = s1_x + s1_w // 2
    s2_center = s2_x + s2_w // 2
    spacing = ((s1_center - b2_center) + (s2_center - s1_center)) / 2

    # Step 4: Predict panel position
    # Panel (7-segment display) is above the buttons, roughly centered over S1
    # Convert button positions to frame coordinates
    b2_x_frame = btn_search_left + b2_x
    s1_x_frame = btn_search_left + s1_x
    s2_x_frame = btn_search_left + s2_x

    btn_top_in_frame = btn_search_top + min(b2_y, s1_y, s2_y)
    btn_height = max(b2_h, s1_h, s2_h)

    # Panel is above the buttons - use fixed offset from button top
    # Display bottom is ~65px above button tops, display is ~105px tall
    panel_bottom = btn_top_in_frame - 65
    panel_top = max(0, panel_bottom - 105)
    panel_height = panel_bottom - panel_top

    # Panel is roughly centered between B2 and S2
    # Display is narrower than button span
    panel_center = (b2_x_frame + s2_x_frame + s2_w) // 2
    panel_width = int(spacing * 1.5)  # Panel width ~1.5 button spacings
    panel_left = max(0, panel_center - panel_width // 2)
    panel_right = min(w_frame, panel_left + panel_width)

    # Validate dimensions
    # Reject if too small (<50), too large (>200), or wrong height
    if panel_width < 50 or panel_width > 200 or panel_height < 30:
        return None

    return (panel_left, panel_top, panel_width, panel_height)


def detect_panel(frame):
    """
    Detect the dark rectangular panel containing blue LED digits.

    Uses landmark-based prediction (corner + buttons) as primary method,
    with blue LED color detection as fallback.

    Args:
        frame: BGR image from camera/file

    Returns:
        panel_rect: (x, y, w, h) of the detected panel, or None if not found
        method: detection method used ('landmark', 'brightness', or None)
    """
    h_frame, w_frame = frame.shape[:2]

    # Try landmark-based detection first (corner + buttons)
    landmark_panel = predict_panel_from_landmarks(frame)
    if landmark_panel is not None:
        return landmark_panel, 'landmark'

    # Fallback 1: Corner-only detection (if corner found but buttons failed)
    # Use fixed spatial relationship from corner to panel
    corner_result = _find_corner(frame, min_match=0.7)
    if corner_result is not None:
        corner_x, corner_y, _ = corner_result
        # Known offsets from calibration:
        # Panel x ≈ corner_x - 266 (centered between B2 and S2)
        # Panel y ≈ corner_y - 86
        panel_x = corner_x - _CORNER_TO_PANEL_X
        panel_y = corner_y - _CORNER_TO_PANEL_Y

        # Validate bounds
        if panel_x >= 0 and panel_y >= 0:
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
        return None, None

    # Filter contours by position and size
    margin_top = int(h_frame * _PANEL_MARGIN_TOP_RATIO)
    margin_bottom = int(h_frame * _PANEL_MARGIN_BOTTOM_RATIO)
    min_area = 50
    max_area = h_frame * w_frame * 0.02

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
    # Vertical: anchor to top of bounding box with small padding
    panel_x = max(0, cx - _PANEL_WIDTH // 2)
    panel_y = max(0, y - 26)  # 26px padding above top edge (calibrated)
    panel_w = min(w_frame - panel_x, _PANEL_WIDTH)
    panel_h = min(h_frame - panel_y, _PANEL_HEIGHT)

    panel_rect = (panel_x, panel_y, panel_w, panel_h)

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
        px, py, pw, ph = panel_rect
        btn_top = py + ph
        btn_bottom = h_frame
        btn_left = 0
        btn_right = int(w_frame * _BUTTON_REGION_RIGHT_RATIO)
    else:
        btn_top = int(h_frame * _BUTTON_REGION_TOP_RATIO)
        btn_bottom = h_frame
        btn_left = 0
        btn_right = int(w_frame * _BUTTON_REGION_RIGHT_RATIO)

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

    global _button_zone_cache

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
        # We have 3+ buttons - these are B2, S1, S2 (B1 is partially cut off at left edge)
        # Use B2, S1, S2 positions to predict B1 location
        widths = [b[2] for b in buttons[:3]]
        heights = [b[3] for b in buttons[:3]]
        avg_width = sum(widths) / len(widths)
        avg_height = sum(heights) / len(heights)

        # Get centers of B2, S1, S2 (first 3 detected buttons, left to right)
        b2_center = buttons[0][0] + buttons[0][2] // 2
        s1_center = buttons[1][0] + buttons[1][2] // 2
        s2_center = buttons[2][0] + buttons[2][2] // 2

        # Calculate spacing from B2, S1, S2
        spacing_b2_s1 = s1_center - b2_center
        spacing_s1_s2 = s2_center - s1_center
        avg_spacing = (spacing_b2_s1 + spacing_s1_s2) / 2

        # Predict B1 center as B2 center minus average spacing
        b1_center = b2_center - avg_spacing

        # Predict B1 X position
        b1_x = int(b1_center - avg_width / 2)

        # B1 is on the same row as B2, so use B2's Y directly
        # buttons[0] is B2 (first detected, leftmost visible button)
        b2_y = buttons[0][1]
        b2_height = buttons[0][3]
        b1_y = b2_y

        predicted_b1_box = (b1_x, b1_y, int(avg_width), int(b2_height))

        # Build LED zones with boundaries (left_x, right_x, top_y, bottom_y, name)
        # LED is on the right side of each button (50%-100% of button width)
        half_width = avg_width / 2

        # Get Y boundaries from detected buttons (B2, S1, S2)
        b2_top, b2_bottom = buttons[0][1], buttons[0][1] + buttons[0][3]
        s1_top, s1_bottom = buttons[1][1], buttons[1][1] + buttons[1][3]
        s2_top, s2_bottom = buttons[2][1], buttons[2][1] + buttons[2][3]
        # B1 uses B2's Y (same row)
        b1_top, b1_bottom = b2_top, b2_bottom

        # LED zone: from button center to right edge, within button Y bounds
        # B1 (predicted) - LED is at ~75% of button width from left edge
        # Only detect B1 if the LED zone is visible (LED X position > 0)
        b1_led_x = b1_x + avg_width * 0.75  # Expected LED X position
        if b1_led_x > 15:  # LED must be at least 15px into visible area
            # Tighter zone for B1: just around the expected LED position
            # Zone starts at LED position - 10px (not less than 20px from edge to avoid noise)
            b1_led_left = max(20, b1_led_x - 10)
            b1_led_right = b1_led_x + 15
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
            # Fallback: use fixed zones with boundaries (left_x, right_x, top_y, bottom_y, name)
            # Approximate button dimensions (relative to button region)
            zone_width = bw * 0.20
            zone_top = int(bh * 0.35)
            zone_bottom = int(bh * 0.90)
            zone_centers = {
                'B1': 0.10,   # ~10% (partially visible)
                'B2': 0.33,   # ~33%
                'S1': 0.60,   # ~60%
                'S2': 0.86,   # ~86%
            }
            button_zones = [
                (bw * frac - zone_width/2, bw * frac + zone_width/2, zone_top, zone_bottom, name)
                for name, frac in zone_centers.items()
            ]

        # When in fallback mode, enlarge the LED detection zones
        if detection_method is not None and detection_method != 'landmark':
            # Enlarge zones: extend left/right by 20px, top by 30px, bottom by 20px
            enlarged_zones = []
            for left_x, right_x, top_y, bottom_y, name in button_zones:
                new_left = max(0, left_x - 20)
                new_right = min(bw, right_x + 20)
                new_top = max(0, top_y - 30)
                new_bottom = min(bh, bottom_y + 20)
                enlarged_zones.append((new_left, new_right, new_top, new_bottom, name))
            button_zones = enlarged_zones

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
        if _LED_MIN_AREA < area < _LED_MAX_AREA:
            aspect = max(blob_w, blob_h) / max(1, min(blob_w, blob_h))
            if aspect < _LED_MAX_ASPECT_RATIO:  # Reasonably compact
                valid_blobs.append((blob_x, blob_y, area))

    # Find the LED by checking which button zone contains the blob
    # Pick the largest blob that falls within a button boundary (X and Y)
    lit_led = None
    led_position = None
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

    # Primary detection: brightness-based (compare max brightness across zones)
    # This is more reliable than blob detection for bright LEDs
    if len(button_zones) > 0:
        gray = cv2.cvtColor(button_region, cv2.COLOR_BGR2GRAY)
        zone_brightness = []
        for left_x, right_x, top_y, bottom_y, name in button_zones:
            # Extract zone region
            x1, x2 = int(left_x), int(right_x)
            y1, y2 = int(top_y), int(bottom_y)
            if x1 < x2 and y1 < y2 and x2 <= gray.shape[1] and y2 <= gray.shape[0]:
                zone = gray[y1:y2, x1:x2]
                if zone.size > 0:
                    # Use max brightness in zone (LED is a bright spot)
                    max_bright = int(np.max(zone))
                    zone_brightness.append((name, max_bright, (x1 + x2) // 2, (y1 + y2) // 2))

        # Find the brightest zone - must be significantly brighter than others
        if zone_brightness:
            zone_brightness.sort(key=lambda x: -x[1])  # Sort by brightness descending
            brightest_name, brightest_val, bx, by = zone_brightness[0]
            # Use brightness detection if clearly bright and brighter than others
            if brightest_val > 150:
                second_val = zone_brightness[1][1] if len(zone_brightness) > 1 else 0
                if brightest_val - second_val > 20:
                    lit_led = brightest_name
                    led_position = (bx + btn_left, by + btn_top)

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
                    break
            if lit_led:
                break

    if lit_led:
        leds[lit_led] = True

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


def detect_red_button(frame, debug=False, return_debug=False):
    """
    Detect if the red button LED (MUTE indicator) is lit.

    Uses template matching to find the corner, then searches for red LED
    in a region relative to the corner position. Falls back to fixed region
    if corner not found.

    Args:
        frame: BGR image from camera/file
        debug: If True, return debug image showing detection
        return_debug: If True, return debug info dict for visualization

    Returns:
        is_lit: bool, True if red LED is lit
        debug_img: (only if debug=True) Image showing detection
        debug_info: (only if return_debug=True) Dict with detection region info
    """
    h_frame, w_frame = frame.shape[:2]
    debug_img = frame.copy() if debug else None

    # Try to find corner using template matching
    corner_result = _find_corner(frame)

    if corner_result is not None:
        corner_x, corner_y, match_score = corner_result
        # Calculate red button region relative to corner
        # Red button is at offset (200, 43) from corner center
        btn_x = corner_x + _RED_BUTTON_OFFSET[0]
        btn_y = corner_y + _RED_BUTTON_OFFSET[1]

        # Define search region around expected button location
        region_half = 40
        region_left = max(0, btn_x - region_half)
        region_right = min(w_frame, btn_x + region_half)
        region_top = max(0, btn_y - region_half)
        region_bottom = min(h_frame, btn_y + region_half)
        method = "corner"
    else:
        # Fallback: use fixed region in lower right
        # Based on analysis: red button at ~95% x, ~74% y
        region_left = int(w_frame * 0.90)
        region_right = w_frame
        region_top = int(h_frame * 0.65)
        region_bottom = int(h_frame * 0.85)
        method = "fallback"

    # Extract search region
    region = frame[region_top:region_bottom, region_left:region_right]

    # Detect red pixels using dual HSV ranges (red wraps around hue=0)
    red_mask = _detect_red_pixels(region)

    # Count red pixels
    red_pixels = np.sum(red_mask > 0)

    # Threshold: need at least 25 red pixels to consider LED lit
    # (LED is small ~20-50 pixels, lowered from 50 for consistent detection)
    is_lit = red_pixels >= 25

    # Build debug info for return_debug mode
    debug_info = None
    if return_debug:
        # Find center of red pixels if any
        led_center = None
        if red_pixels > 0:
            coords = np.where(red_mask > 0)
            cy = int(np.mean(coords[0])) + region_top
            cx = int(np.mean(coords[1])) + region_left
            led_center = (cx, cy)

        debug_info = {
            'region': (region_left, region_top, region_right, region_bottom),
            'method': method,
            'red_pixels': red_pixels,
            'is_lit': is_lit,
            'led_center': led_center
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

        # Find center of red pixels if any
        if red_pixels > 0:
            coords = np.where(red_mask > 0)
            cy = int(np.mean(coords[0])) + region_top
            cx = int(np.mean(coords[1])) + region_left

            color = (0, 255, 0) if is_lit else (0, 0, 255)
            cv2.circle(debug_img, (cx, cy), 10, color, 2)
            cv2.putText(debug_img, f"MUTE:{'ON' if is_lit else 'OFF'}",
                        (cx - 30, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Show method and pixel count
        cv2.putText(debug_img, f"{method} red_px={red_pixels}",
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

    # Detect blue pixels (normal LED appearance)
    lower_blue = np.array([85, 80, 80])
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


def recognize_digit(digit_img, debug=False, auto_learn=False):
    """
    Step 5: Recognize a single digit using template matching.
    Falls back to 7-segment analysis if template matching fails.

    Args:
        digit_img: BGR image of a single digit box
        debug: If True, return debug image showing segment zones
        auto_learn: If True, enable auto-learning of new templates

    Returns:
        digit: Recognized character ('0'-'9', 'P') or 'X' if unknown
        debug_img: (only if debug=True) Image showing segment analysis
    """
    # Try template matching first (more robust)
    digit, score = recognize_digit_template(digit_img, auto_learn=auto_learn)
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

    # Gap must be in center region (35%-65% of width) to avoid finding
    # local minima within digit segments caused by LED flicker
    center = len(smoothed) // 2
    min_gap_x = int(len(smoothed) * 0.35)
    max_gap_x = int(len(smoothed) * 0.65)

    # Strategy: Find the two digit peaks, then find the valley between them
    # This correctly identifies the gap even when there's no clear local minimum

    # Find local maxima (peaks) in the valid range - these are digit segments
    valid_region = smoothed[min_gap_x:max_gap_x + 1]
    region_min = valid_region.min()
    region_max = valid_region.max()
    peak_threshold = region_min + (region_max - region_min) * 0.3  # Peaks must be 30% above min

    peaks = []
    for x in range(min_gap_x + 1, max_gap_x):
        if smoothed[x] > smoothed[x - 1] and smoothed[x] > smoothed[x + 1]:
            if smoothed[x] > peak_threshold:  # Only significant peaks
                peaks.append((x, smoothed[x]))

    # Need two well-separated peaks (at least 20% of width apart)
    min_peak_separation = int(len(smoothed) * 0.15)

    if len(peaks) >= 2:
        # Sort by brightness to find the two main digit peaks
        peaks_sorted = sorted(peaks, key=lambda p: -p[1])

        # Find two peaks that are sufficiently separated
        left_peak = right_peak = None
        for i, (px1, pv1) in enumerate(peaks_sorted):
            for px2, pv2 in peaks_sorted[i + 1:]:
                if abs(px1 - px2) >= min_peak_separation:
                    left_peak = min(px1, px2)
                    right_peak = max(px1, px2)
                    break
            if left_peak is not None:
                break

        if left_peak is not None and right_peak is not None and right_peak > left_peak:
            # Find the minimum between the two peaks (the valley = gap)
            valley_region = smoothed[left_peak:right_peak + 1]
            if len(valley_region) < 2:
                # Peaks too close, use center between them
                gap_x = (left_peak + right_peak) // 2
            else:
                valley_min_idx = np.argmin(valley_region)
                gap_x = left_peak + valley_min_idx

                # Find flat bottom around the minimum (within 5% of min value)
                valley_min = valley_region[valley_min_idx]
                threshold = valley_min * 1.05
                flat_indices = np.where(valley_region <= threshold)[0]
                if len(flat_indices) > 1:
                    # Use center of flat region
                    gap_x = left_peak + (flat_indices[0] + flat_indices[-1]) // 2
        else:
            # Peaks not well-separated, use center
            gap_x = center
    else:
        # Fallback: use center (no clear peaks found)
        gap_x = center

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

    def __init__(self, cache_ttl=100, auto_learn=False):
        """
        Args:
            cache_ttl: Maximum frames before forcing cache refresh
            auto_learn: If True, auto-learn new templates when confidence is low
        """
        self.cache_ttl = cache_ttl
        self.auto_learn = auto_learn

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
        panel_img = frame[y:y+h, x:x+w]

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

        panel_img = frame[y:y+h, x:x+w]
        corrected_img, _, _ = correct_slant(panel_img, 8.0)

        gap_x, _ = find_digit_gap(corrected_img)
        left_box, right_box, _ = define_digit_boxes(corrected_img, gap_x)

        left_digit_img = _extract_digit_with_padding(corrected_img, left_box, right_bound=gap_x)
        right_digit_img = _extract_digit_with_padding(corrected_img, right_box, left_bound=gap_x)

        left_digit, left_score = recognize_digit_template(left_digit_img, auto_learn=self.auto_learn)
        right_digit, right_score = recognize_digit_template(right_digit_img, auto_learn=self.auto_learn)
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

        # Always detect panel fresh
        panel_rect, detection_method = detect_panel(frame)
        self._detection_method = detection_method  # Store for logging
        if panel_rect is None:
            log_issue_frame(frame, 'panel_fail')
            return "XX", False

        x, y, w, h = panel_rect
        panel_img = frame[y:y+h, x:x+w]

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

        # Convert images to grayscale once for quick-check
        left_gray = cv2.cvtColor(left_digit_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_digit_img, cv2.COLOR_BGR2GRAY)

        # Left digit recognition with quick-check
        if force_full_scan or self._left_best_templates is None:
            # Full search: periodic rescan or first frame
            left_digit, left_score, left_debug = recognize_digit_template(
                left_digit_img, auto_learn=self.auto_learn, return_debug=True)
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
                    left_digit_img, auto_learn=self.auto_learn, return_debug=True)
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
                right_digit_img, auto_learn=self.auto_learn, return_debug=True)
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
                    right_digit_img, auto_learn=self.auto_learn, return_debug=True)
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

    # Extract panel region
    panel_img = frame[y:y+h, x:x+w]

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
