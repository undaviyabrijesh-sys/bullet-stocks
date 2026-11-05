import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image

from stockmind_telegram_mvp.app import overlay_engine
from stockmind_telegram_mvp.app.overlay_engine import (
    _zone_overlay_label,
    render_overlay,
    validate_chart_image,
)


def sample_image(size=(640, 360), image_format="PNG"):
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format=image_format)
    return output.getvalue()


class OverlayRendererTests(unittest.TestCase):
    def test_detects_actual_supported_image_format(self):
        self.assertEqual(validate_chart_image(sample_image()), "image/png")
        self.assertEqual(validate_chart_image(sample_image(image_format="JPEG")), "image/jpeg")
        self.assertEqual(validate_chart_image(sample_image(image_format="WEBP")), "image/webp")

    def test_rejects_unsupported_or_corrupt_image_bytes(self):
        output = BytesIO()
        Image.new("RGB", (32, 32), "white").save(output, format="GIF")
        with self.assertRaisesRegex(ValueError, "JPEG, PNG or WEBP"):
            validate_chart_image(output.getvalue())
        with self.assertRaises(ValueError):
            validate_chart_image(b"not an image")

    def test_rejects_animated_images(self):
        output = BytesIO()
        frames = [Image.new("RGB", (32, 32), color) for color in ("white", "black")]
        frames[0].save(output, format="WEBP", save_all=True, append_images=frames[1:], duration=100, loop=0)
        with self.assertRaisesRegex(ValueError, "Animated images"):
            validate_chart_image(output.getvalue())

    def test_rejects_images_over_pixel_limit(self):
        with patch.object(overlay_engine, "MAX_CHART_IMAGE_PIXELS", 100):
            with self.assertRaisesRegex(ValueError, "too many pixels"):
                validate_chart_image(sample_image(size=(11, 10)))

    def test_draws_a_valid_zone(self):
        zones = [{
            "zone_type": "demand",
            "label": "DEMAND 100-105",
            "x_start": 0.2,
            "x_end": 0.92,
            "y_top": 0.55,
            "y_bottom": 0.68,
            "draw_overlay": True,
        }]
        result, count = render_overlay(sample_image(), zones)
        self.assertEqual(count, 1)
        self.assertIsNotNone(result)
        with Image.open(BytesIO(result)) as rendered:
            self.assertEqual(rendered.format, "JPEG")
            self.assertEqual(rendered.size, (640, 360))

    def test_skips_unapproved_invalid_and_non_numeric_zones(self):
        zones = [
            {"zone_type": "supply", "draw_overlay": False},
            {
                "zone_type": "demand", "draw_overlay": True,
                "x_start": 0.8, "x_end": 0.2, "y_top": 0.3, "y_bottom": 0.4,
            },
            {
                "zone_type": "other", "draw_overlay": True,
                "x_start": 0.1, "x_end": 0.9, "y_top": 0.3, "y_bottom": 0.4,
            },
            {
                "zone_type": "demand", "draw_overlay": True,
                "x_start": True, "x_end": 0.9, "y_top": 0.3, "y_bottom": 0.4,
            },
            {
                "zone_type": "supply", "draw_overlay": True,
                "x_start": 0.1, "x_end": 0.9, "y_top": float("nan"), "y_bottom": 0.4,
            },
            {
                "zone_type": "demand", "draw_overlay": True,
                "x_start": 0.2, "x_end": 0.2001, "y_top": 0.3, "y_bottom": 0.4,
            },
            {
                "zone_type": "supply", "draw_overlay": True,
                "x_start": 0.1, "x_end": 0.9, "y_top": 0.3, "y_bottom": 0.301,
            },
        ]
        result, count = render_overlay(sample_image(), zones)
        self.assertIsNone(result)
        self.assertEqual(count, 0)

    def test_skips_coordinates_too_large_to_convert_to_float(self):
        zones = [{
            "zone_type": "demand",
            "draw_overlay": True,
            "x_start": 10 ** 400,
            "x_end": 0.9,
            "y_top": 0.3,
            "y_bottom": 0.4,
        }]

        result, count = render_overlay(sample_image(), zones)

        self.assertIsNone(result)
        self.assertEqual(count, 0)

    def test_non_list_zones_are_ignored(self):
        result, count = render_overlay(sample_image(), None)
        self.assertIsNone(result)
        self.assertEqual(count, 0)

    def test_overlay_label_preserves_zone_origin_and_safely_falls_back(self):
        self.assertEqual(
            _zone_overlay_label({"label": "USER DEMAND"}, "demand"),
            "USER DEMAND",
        )
        self.assertEqual(
            _zone_overlay_label({"label": "AI SUPPLY – 10,000"}, "supply"),
            "AI SUPPLY  10,000",
        )
        self.assertEqual(_zone_overlay_label({"label": "ગુજરાતી"}, "demand"), "DEMAND")
        self.assertEqual(_zone_overlay_label({"label": " "}, "supply"), "SUPPLY")
        self.assertEqual(_zone_overlay_label({"label": "X" * 80}, "demand"), "X" * 32)

    def test_rejects_invalid_image_bytes(self):
        with self.assertRaises(ValueError):
            render_overlay(b"not an image", [])


if __name__ == "__main__":
    unittest.main()
