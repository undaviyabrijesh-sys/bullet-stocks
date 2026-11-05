import unittest
from io import BytesIO

from PIL import Image

from app.overlay_engine import render_overlay


def sample_image():
    output = BytesIO()
    Image.new("RGB", (640, 360), "white").save(output, format="PNG")
    return output.getvalue()


class OverlayRendererTests(unittest.TestCase):
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

    def test_skips_unapproved_or_invalid_zones(self):
        zones = [
            {"zone_type": "supply", "draw_overlay": False},
            {
                "zone_type": "demand",
                "draw_overlay": True,
                "x_start": 0.8,
                "x_end": 0.2,
                "y_top": 0.3,
                "y_bottom": 0.4,
            },
            {
                "zone_type": "other",
                "draw_overlay": True,
                "x_start": 0.1,
                "x_end": 0.9,
                "y_top": 0.3,
                "y_bottom": 0.4,
            },
        ]
        result, count = render_overlay(sample_image(), zones)
        self.assertIsNone(result)
        self.assertEqual(count, 0)

    def test_rejects_invalid_image_bytes(self):
        with self.assertRaises(Exception):
            render_overlay(b"not an image", [])


if __name__ == "__main__":
    unittest.main()
