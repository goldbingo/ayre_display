#!/usr/bin/env python3
"""Unit tests for 7-segment display reader."""

import pytest
import cv2
import numpy as np
import os
import json
import tempfile
from segment_reader import (
    detect_panel,
    correct_slant,
    find_digit_gap,
    define_digit_boxes,
    recognize_digit,
    get_blue_mask,
    SegmentReader,
    detect_button_leds,
    clear_button_zone_cache,
    detect_red_button,
)


# Test images and expected results (image, digit, led)
TEST_CASES = [
    ("example/08-S2-MUTE.PNG", "08", "S2"),
    ("example/09-B2-UNMUTE.PNG", "09", "B2"),
    ("example/10-B2-UNMUTE.PNG", "10", "B2"),
    ("example/11-B2-UNMUTE.PNG", "11", "B2"),
    ("example/16-B2-UNMUTE.PNG", "16", "B2"),
    ("example/17-B2-UNMUTE.PNG", "17", "B2"),
    ("example/19-B2-UNMUTE.PNG", "19", "B2"),
    ("example/25-B2-UNMUTE.PNG", "25", "B2"),
    ("example/34-S2-UNMUTE.PNG", "34", "S2"),
    ("example/42-S2-MUTE.PNG", "42", "S2"),
    ("example/PP-S1-UNMUTE.PNG", "PP", "S1"),
]

# For tests that only need image and digit
TEST_CASES_DIGIT = [(img, digit) for img, digit, led in TEST_CASES]


class TestPanelDetection:
    """Tests for panel detection."""

    @pytest.mark.parametrize("image_path,expected", TEST_CASES_DIGIT)
    def test_panel_detected(self, image_path, expected):
        """Panel should be detected for all test images."""
        frame = cv2.imread(image_path)
        assert frame is not None, f"Could not load {image_path}"

        panel_rect, _ = detect_panel(frame)
        assert panel_rect is not None, f"Panel not detected in {image_path}"

        x, y, w, h = panel_rect
        assert w > 0 and h > 0, "Panel dimensions should be positive"
        assert x >= 0 and y >= 0, "Panel position should be non-negative"

    def test_no_panel_in_blank_image(self):
        """No panel should be detected in a blank image."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        panel_rect, _ = detect_panel(blank)
        assert panel_rect is None


class TestSlantEstimation:
    """Tests for slant angle estimation."""

class TestSlantCorrection:
    """Tests for slant correction."""

    def test_correct_slant_output_shape(self):
        """Corrected image should have valid dimensions."""
        frame = cv2.imread("example/10-B2-UNMUTE.PNG")
        panel_rect, _ = detect_panel(frame)
        x, y, w, h = panel_rect
        panel_img = frame[y:y+h, x:x+w]

        corrected, angle, _ = correct_slant(panel_img)
        assert corrected.shape[0] == panel_img.shape[0], "Height should be preserved"
        assert corrected.shape[2] == 3, "Should be BGR image"

    def test_zero_angle_no_change(self):
        """Zero angle should produce similar output."""
        frame = cv2.imread("example/10-B2-UNMUTE.PNG")
        panel_rect, _ = detect_panel(frame)
        x, y, w, h = panel_rect
        panel_img = frame[y:y+h, x:x+w]

        corrected, _, _ = correct_slant(panel_img, angle=0.0)
        # Width should be same or very close for zero angle
        assert abs(corrected.shape[1] - panel_img.shape[1]) <= 2


class TestDigitGap:
    """Tests for digit gap detection."""

    @pytest.mark.parametrize("image_path,expected", TEST_CASES_DIGIT)
    def test_gap_in_middle(self, image_path, expected):
        """Gap should be roughly in the middle of the panel."""
        frame = cv2.imread(image_path)
        panel_rect, _ = detect_panel(frame)
        x, y, w, h = panel_rect
        panel_img = frame[y:y+h, x:x+w]

        corrected, _, _ = correct_slant(panel_img)
        gap_x, _ = find_digit_gap(corrected)

        # Gap should be in middle third of image
        img_w = corrected.shape[1]
        assert img_w * 0.25 < gap_x < img_w * 0.75, f"Gap at {gap_x} not in middle of {img_w}"


class TestDigitBoxes:
    """Tests for digit box definition."""

    @pytest.mark.parametrize("image_path,expected", TEST_CASES_DIGIT)
    def test_boxes_non_overlapping(self, image_path, expected):
        """Left and right boxes should not overlap."""
        frame = cv2.imread(image_path)
        panel_rect, _ = detect_panel(frame)
        x, y, w, h = panel_rect
        panel_img = frame[y:y+h, x:x+w]

        corrected, _, _ = correct_slant(panel_img)
        gap_x, _ = find_digit_gap(corrected)
        left_box, right_box, _ = define_digit_boxes(corrected, gap_x)

        lx, ly, lw, lh = left_box
        rx, ry, rw, rh = right_box

        # Left box should end before right box starts
        assert lx + lw <= rx, "Boxes should not overlap"

    @pytest.mark.parametrize("image_path,expected", TEST_CASES_DIGIT)
    def test_boxes_have_content(self, image_path, expected):
        """Boxes should have positive dimensions."""
        frame = cv2.imread(image_path)
        panel_rect, _ = detect_panel(frame)
        x, y, w, h = panel_rect
        panel_img = frame[y:y+h, x:x+w]

        corrected, _, _ = correct_slant(panel_img)
        gap_x, _ = find_digit_gap(corrected)
        left_box, right_box, _ = define_digit_boxes(corrected, gap_x)

        for box in [left_box, right_box]:
            bx, by, bw, bh = box
            assert bw > 0 and bh > 0, "Box dimensions should be positive"


class TestBlueMask:
    """Tests for blue mask generation."""

    def test_tight_mask_subset_of_loose(self):
        """Tight mask should have fewer or equal pixels than loose mask."""
        frame = cv2.imread("example/10-B2-UNMUTE.PNG")
        panel_rect, _ = detect_panel(frame)
        x, y, w, h = panel_rect
        panel_img = frame[y:y+h, x:x+w]

        tight = get_blue_mask(panel_img, tight=True)
        loose = get_blue_mask(panel_img, tight=False)

        tight_pixels = np.sum(tight > 0)
        loose_pixels = np.sum(loose > 0)

        assert tight_pixels <= loose_pixels, "Tight mask should have fewer pixels"

    def test_mask_finds_blue_pixels(self):
        """Mask should find blue pixels in test images."""
        frame = cv2.imread("example/10-B2-UNMUTE.PNG")
        panel_rect, _ = detect_panel(frame)
        x, y, w, h = panel_rect
        panel_img = frame[y:y+h, x:x+w]

        mask = get_blue_mask(panel_img, tight=False)
        assert np.sum(mask > 0) > 100, "Should find significant blue pixels"


class TestDigitRecognition:
    """Tests for individual digit recognition."""

    @pytest.mark.parametrize("image_path,expected", TEST_CASES_DIGIT)
    def test_recognize_digits(self, image_path, expected):
        """Should correctly recognize both digits."""
        frame = cv2.imread(image_path)
        panel_rect, _ = detect_panel(frame)
        x, y, w, h = panel_rect
        panel_img = frame[y:y+h, x:x+w]

        corrected, _, _ = correct_slant(panel_img)
        gap_x, _ = find_digit_gap(corrected)
        left_box, right_box, _ = define_digit_boxes(corrected, gap_x)

        lx, ly, lw, lh = left_box
        rx, ry, rw, rh = right_box

        left_img = corrected[ly:ly+lh, lx:lx+lw]
        right_img = corrected[ry:ry+rh, rx:rx+rw]

        left_digit, _ = recognize_digit(left_img)
        right_digit, _ = recognize_digit(right_img)

        result = left_digit + right_digit
        assert result == expected, f"Expected {expected}, got {result}"


class TestLEDDetection:
    """Tests for button LED detection."""

    @pytest.mark.parametrize("image_path,expected_digit,expected_led", TEST_CASES)
    def test_led_detection(self, image_path, expected_digit, expected_led):
        """Should correctly detect which LED is lit."""
        frame = cv2.imread(image_path)
        assert frame is not None, f"Could not load {image_path}"

        panel_rect, _ = detect_panel(frame)
        assert panel_rect is not None

        leds, _ = detect_button_leds(frame, panel_rect)

        lit_leds = [k for k, v in leds.items() if v]
        assert len(lit_leds) == 1, f"Expected exactly 1 LED lit, got {len(lit_leds)}"
        assert lit_leds[0] == expected_led, f"Expected {expected_led}, got {lit_leds[0]}"

    def test_only_one_led_lit(self):
        """Only one LED should be lit at a time."""
        for image_path, _, _ in TEST_CASES:
            frame = cv2.imread(image_path)
            panel_rect, _ = detect_panel(frame)
            leds, _ = detect_button_leds(frame, panel_rect)

            lit_count = sum(1 for v in leds.values() if v)
            assert lit_count <= 1, f"Multiple LEDs lit in {image_path}"

    def test_led_returns_all_buttons(self):
        """LED detection should return status for all 4 buttons."""
        frame = cv2.imread("example/10-B2-UNMUTE.PNG")
        panel_rect, _ = detect_panel(frame)
        leds, _ = detect_button_leds(frame, panel_rect)

        assert set(leds.keys()) == {'B1', 'B2', 'S1', 'S2'}

    def test_button_zone_cache_persistence(self):
        """Button zone cache should persist to disk."""
        import segment_reader as sr

        # Clear cache first
        clear_button_zone_cache()
        assert sr._button_zone_cache is None

        # Run detection
        frame = cv2.imread("example/10-B2-UNMUTE.PNG")
        panel_rect, _ = detect_panel(frame)
        detect_button_leds(frame, panel_rect)

        # Check cache is populated
        assert sr._button_zone_cache is not None
        assert len(sr._button_zone_cache) == 4

        # Check file exists
        assert os.path.exists(sr._BUTTON_ZONE_CACHE_FILE)

    def test_no_led_in_blank_image(self):
        """No LED should be detected in blank image."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        leds, _ = detect_button_leds(blank, None)

        lit_count = sum(1 for v in leds.values() if v)
        assert lit_count == 0, "No LEDs should be lit in blank image"


class TestRedButtonDetection:
    """Tests for red button (MUTE) LED detection."""

    # Test cases: (image_path, expected_mute_status)
    MUTE_TEST_CASES = [
        ("example/08-S2-MUTE.PNG", True),
        ("example/09-B2-UNMUTE.PNG", False),
        ("example/10-B2-UNMUTE.PNG", False),
        ("example/11-B2-UNMUTE.PNG", False),
        ("example/16-B2-UNMUTE.PNG", False),
        ("example/17-B2-UNMUTE.PNG", False),
        ("example/19-B2-UNMUTE.PNG", False),
        ("example/25-B2-UNMUTE.PNG", False),
        ("example/34-S2-UNMUTE.PNG", False),
        ("example/42-S2-MUTE.PNG", True),
        ("example/PP-S1-UNMUTE.PNG", False),
    ]

    @pytest.mark.parametrize("image_path,expected_mute", MUTE_TEST_CASES)
    def test_red_button_detection(self, image_path, expected_mute):
        """Should correctly detect MUTE status for all test images."""
        frame = cv2.imread(image_path)
        assert frame is not None, f"Could not load {image_path}"

        is_lit, _ = detect_red_button(frame)
        assert is_lit == expected_mute, f"Expected MUTE={expected_mute}, got {is_lit}"

    def test_no_red_in_blank_image(self):
        """No red button should be detected in blank image."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        is_lit, _ = detect_red_button(blank)
        assert is_lit == False, "No red button should be lit in blank image"

    def test_debug_mode_returns_image(self):
        """Debug mode should return an image."""
        frame = cv2.imread("example/10-B2-UNMUTE.PNG")
        is_lit, debug_img = detect_red_button(frame, debug=True)

        assert debug_img is not None
        assert debug_img.shape == frame.shape


class TestSegmentReader:
    """Tests for the SegmentReader class."""

    @pytest.mark.parametrize("image_path,expected", TEST_CASES_DIGIT)
    def test_read_all_test_images(self, image_path, expected):
        """SegmentReader should correctly read all test images."""
        frame = cv2.imread(image_path)
        reader = SegmentReader()

        reading, _ = reader.read(frame)
        assert reading == expected, f"Expected {expected}, got {reading}"

    def test_cache_hit_on_second_read(self):
        """Second read of same image should be cache hit."""
        frame = cv2.imread("example/10-B2-UNMUTE.PNG")
        reader = SegmentReader()

        reading1, hit1 = reader.read(frame)
        reading2, hit2 = reader.read(frame)

        assert reading1 == reading2 == "10"
        assert hit2 == True, "Second read should be cache hit"

    def test_reset_cache(self):
        """Reset should clear cached values."""
        frame = cv2.imread("example/10-B2-UNMUTE.PNG")
        reader = SegmentReader()

        reader.read(frame)
        assert reader.panel_rect is not None

        reader.reset_cache()
        assert reader._panel_rect is None
        assert reader._left_box is None
        assert reader._right_box is None

    def test_invalid_frame_returns_last_reading(self):
        """Invalid frame should return last successful reading."""
        frame = cv2.imread("example/10-B2-UNMUTE.PNG")
        reader = SegmentReader()

        reader.read(frame)

        reading, _ = reader.read(None)
        assert reading == "10"


class TestCachePersistence:
    """Tests for cache file persistence."""

    def test_save_and_load_cache(self):
        """Cache should be saveable and loadable."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            cache_file = f.name
        # Delete empty temp file so reader starts fresh
        os.remove(cache_file)

        try:
            # Create reader and read an image
            frame = cv2.imread("example/10-B2-UNMUTE.PNG")
            reader1 = SegmentReader(cache_file=cache_file)
            reader1.read(frame)

            # Verify cache file exists
            assert os.path.exists(cache_file)

            # Load cache in new reader
            reader2 = SegmentReader(cache_file=cache_file)

            assert reader2._panel_rect == reader1._panel_rect
            assert reader2._left_box == reader1._left_box
            assert reader2._right_box == reader1._right_box
        finally:
            if os.path.exists(cache_file):
                os.remove(cache_file)

    def test_cache_file_valid_json(self):
        """Cache file should be valid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            cache_file = f.name
        # Delete empty temp file so reader starts fresh
        os.remove(cache_file)

        try:
            frame = cv2.imread("example/10-B2-UNMUTE.PNG")
            reader = SegmentReader(cache_file=cache_file)
            reader.read(frame)

            with open(cache_file, 'r') as f:
                data = json.load(f)

            assert 'panel_rect' in data
            assert 'left_box' in data
            assert 'right_box' in data
        finally:
            if os.path.exists(cache_file):
                os.remove(cache_file)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_image(self):
        """Should handle empty/black image gracefully."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        reader = SegmentReader()

        reading, _ = reader.read(blank)
        assert reading == "XX"

    def test_none_frame(self):
        """Should handle None frame."""
        reader = SegmentReader()
        reading, _ = reader.read(None)
        assert reading == "XX"

    def test_small_image(self):
        """Should handle very small images."""
        small = np.zeros((10, 10, 3), dtype=np.uint8)
        reader = SegmentReader()

        reading, _ = reader.read(small)
        assert reading == "XX"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
