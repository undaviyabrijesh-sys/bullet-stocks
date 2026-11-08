import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image

from stockmind_telegram_mvp.app import overlay_engine
from stockmind_telegram_mvp.app.overlay_engine import (
    _zone_overlay_label,
    normalized_plot_bounds_to_pixel_box,
    normalized_zone_to_pixel_box,
    prepare_chart_image,
    render_overlay,
    validate_chart_image,
)


def sample_image(size=(640, 360), image_format="PNG"):
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format=image_format)
    return output.getvalue()


def oriented_sample_image(image_format="PNG", orientation=6):
    output = BytesIO()
    exif = Image.Exif()
    exif[274] = orientation
    Image.new("RGB", (64, 48), "white").save(
        output, format=image_format, exif=exif
    )
    return output.getvalue()


class OverlayRendererTests(unittest.TestCase):
    def test_detects_actual_supported_image_format(self):
        self.assertEqual(validate_chart_image(sample_image()), "image/png")
        self.assertEqual(validate_chart_image(sample_image(image_format="JPEG")), "image/jpeg")
        self.assertEqual(validate_chart_image(sample_image(image_format="WEBP")), "image/webp")

    def test_exif_orientation_is_normalized_for_every_supported_format(self):
        expected_mime = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }
        for image_format in expected_mime:
            with self.subTest(image_format=image_format):
                original = oriented_sample_image(image_format=image_format, orientation=6)
                normalized, mime_type = prepare_chart_image(original)
                with Image.open(BytesIO(normalized)) as image:
                    self.assertEqual(image.size, (48, 64))
                    self.assertNotIn(274, image.getexif())
                self.assertEqual(mime_type, expected_mime[image_format])
                self.assertEqual(validate_chart_image(normalized), mime_type)

    def test_renderer_normalizes_exif_orientation_for_direct_callers(self):
        original = oriented_sample_image(image_format="PNG", orientation=6)
        zone = {
            "zone_type": "demand",
            "draw_overlay": True,
            "x_start": 0.2,
            "x_end": 0.8,
            "y_top": 0.2,
            "y_bottom": 0.4,
        }
        bounds = {
            "x_start": 0.05,
            "x_end": 0.95,
            "y_top": 0.05,
            "y_bottom": 0.95,
        }

        result, count = render_overlay(
            original, [zone], plot_bounds=bounds, require_plot_bounds=True
        )

        self.assertEqual(count, 1)
        with Image.open(BytesIO(result)) as rendered:
            # EXIF orientation 6 rotates the source from 64x48 to 48x64.
            # Normalize before applying normalized coordinates or zones shift.
            self.assertEqual(rendered.size, (48, 64))

    def test_exif_orientation_one_preserves_original_bytes(self):
        original = oriented_sample_image(orientation=1)
        normalized, mime_type = prepare_chart_image(original)
        self.assertEqual(normalized, original)
        self.assertEqual(mime_type, "image/png")

    def test_rejects_invalid_exif_orientation_instead_of_guessing(self):
        original = oriented_sample_image(orientation=9)
        with self.assertRaisesRegex(ValueError, "invalid EXIF orientation"):
            prepare_chart_image(original)

    def test_rejects_orientation_normalization_that_exceeds_byte_limit(self):
        exif = Image.Exif()
        exif[274] = 6
        output = BytesIO()
        Image.effect_noise((320, 240), 100).convert("RGB").save(
            output, format="JPEG", quality=20, exif=exif
        )
        original = output.getvalue()
        with self.assertRaisesRegex(ValueError, "Orientation-corrected chart exceeds"):
            prepare_chart_image(original, max_output_bytes=len(original))

    def test_rejects_invalid_normalized_output_limit(self):
        original = sample_image()
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaisesRegex(
                ValueError, "max_output_bytes must be a positive integer"
            ):
                prepare_chart_image(original, max_output_bytes=limit)

    def test_rejects_truncated_pixel_data_or_container_before_vision_request(self):
        # Pillow's verify() alone accepts this truncated JPEG; full pixel decoding
        # detects the missing image data before the bot sends it to the vision API.
        # A truncated PNG container is rejected by the separate integrity check.
        for image_format in ("JPEG", "PNG"):
            with self.subTest(image_format=image_format):
                truncated = sample_image(image_format=image_format)[:-10]
                with self.assertRaisesRegex(ValueError, "not a valid"):
                    validate_chart_image(truncated)

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

    def test_plot_bounds_convert_to_half_open_pixel_coordinates(self):
        self.assertEqual(
            normalized_plot_bounds_to_pixel_box(
                {"x_start": 0.05, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9},
                640, 360,
            ),
            (32, 36, 576, 324),
        )
        self.assertIsNone(normalized_plot_bounds_to_pixel_box(
            {"x_start": 0.8, "x_end": 0.2, "y_top": 0.1, "y_bottom": 0.9},
            640, 360,
        ))

    def test_required_plot_bounds_suppress_overlays_when_missing(self):
        zone = {
            "zone_type": "demand", "draw_overlay": True,
            "x_start": 0.2, "x_end": 0.8, "y_top": 0.3, "y_bottom": 0.5,
        }
        result, count = render_overlay(sample_image(), [zone], require_plot_bounds=True)
        self.assertIsNone(result)
        self.assertEqual(count, 0)

    def test_zone_outside_price_plot_is_not_drawn(self):
        zone = {
            "zone_type": "demand", "draw_overlay": True,
            "x_start": 0.2, "x_end": 0.95, "y_top": 0.3, "y_bottom": 0.5,
        }
        bounds = {"x_start": 0.05, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9}
        result, count = render_overlay(
            sample_image(), [zone], plot_bounds=bounds, require_plot_bounds=True
        )
        self.assertIsNone(result)
        self.assertEqual(count, 0)

    def test_zone_and_label_are_constrained_to_price_plot(self):
        zone = {
            "zone_type": "supply", "label": "AI SUPPLY", "draw_overlay": True,
            "x_start": 0.2, "x_end": 0.8, "y_top": 0.3, "y_bottom": 0.5,
        }
        bounds = {"x_start": 0.05, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9}
        with patch.object(overlay_engine.ImageDraw.ImageDraw, "rectangle", autospec=True) as rect, patch.object(
            overlay_engine.ImageDraw.ImageDraw, "text", autospec=True
        ) as text_draw:
            result, count = render_overlay(
                sample_image(), [zone], plot_bounds=bounds, require_plot_bounds=True
            )
        self.assertIsNotNone(result)
        self.assertEqual(count, 1)
        self.assertEqual(rect.call_args.args[1], (128, 108, 511, 179))
        label_x, label_y = text_draw.call_args.args[1]
        self.assertGreaterEqual(label_x, 32)
        self.assertLess(label_x, 576)
        self.assertGreaterEqual(label_y, 36)
        self.assertLess(label_y, 324)

    def test_long_label_is_omitted_when_it_cannot_fit_inside_plot(self):
        zone = {
            "zone_type": "demand", "label": "X" * 32, "draw_overlay": True,
            "x_start": 0.22, "x_end": 0.28, "y_top": 0.3, "y_bottom": 0.5,
        }
        bounds = {"x_start": 0.2, "x_end": 0.3, "y_top": 0.1, "y_bottom": 0.9}
        with patch.object(overlay_engine.ImageDraw.ImageDraw, "text", autospec=True) as text_draw:
            result, count = render_overlay(
                sample_image(), [zone], plot_bounds=bounds, require_plot_bounds=True
            )
        self.assertIsNotNone(result)
        self.assertEqual(count, 1)
        text_draw.assert_not_called()

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

    def test_pillow_rectangle_uses_half_open_pixel_bounds(self):
        zones = [
            {
                "zone_type": "demand", "draw_overlay": True,
                "x_start": 0.2, "x_end": 0.4,
                "y_top": 0.2, "y_bottom": 0.4,
            },
            {
                "zone_type": "supply", "draw_overlay": True,
                "x_start": 0.0, "x_end": 1.0,
                "y_top": 0.0, "y_bottom": 0.45,
            },
        ]

        with patch.object(
            overlay_engine.ImageDraw.ImageDraw, "rectangle", autospec=True
        ) as draw_rectangle:
            _, count = render_overlay(sample_image(), zones)

        self.assertEqual(count, 2)
        self.assertEqual(
            [call.args[1] for call in draw_rectangle.call_args_list],
            [
                (128, 72, 255, 143),
                (0, 0, 639, 161),
            ],
        )

    def test_invalid_early_candidates_do_not_consume_overlay_limit(self):
        invalid = {
            "zone_type": "demand", "draw_overlay": True,
            "x_start": 0.8, "x_end": 0.2, "y_top": 0.3, "y_bottom": 0.4,
        }
        valid = {
            "zone_type": "supply", "draw_overlay": True,
            "x_start": 0.2, "x_end": 0.6, "y_top": 0.3, "y_bottom": 0.4,
        }

        with patch.object(
            overlay_engine.ImageDraw.ImageDraw, "rectangle", autospec=True
        ) as draw_rectangle:
            result, count = render_overlay(sample_image(), [invalid] * 8 + [valid])

        self.assertIsNotNone(result)
        self.assertEqual(count, 1)
        self.assertEqual(draw_rectangle.call_count, 1)

    def test_renderer_caps_successfully_drawn_zones_at_eight(self):
        zones = [
            {
                "zone_type": "demand" if index % 2 == 0 else "supply",
                "draw_overlay": True,
                "x_start": 0.02 + index * 0.1,
                "x_end": 0.08 + index * 0.1,
                "y_top": 0.2,
                "y_bottom": 0.3,
            }
            for index in range(9)
        ]

        with patch.object(
            overlay_engine.ImageDraw.ImageDraw, "rectangle", autospec=True
        ) as draw_rectangle:
            result, count = render_overlay(sample_image(), zones)

        self.assertIsNotNone(result)
        self.assertEqual(count, 8)
        self.assertEqual(draw_rectangle.call_count, 8)

    def test_renderer_pixel_geometry_is_shared_and_rejects_subminimum_boxes(self):
        zone = {
            "zone_type": "demand",
            "draw_overlay": True,
            "x_start": 0.2,
            "x_end": 0.4,
            "y_top": 0.2,
            "y_bottom": 0.4,
        }
        self.assertEqual(
            normalized_zone_to_pixel_box(zone, 640, 360),
            (128, 72, 256, 144),
        )
        self.assertIsNone(normalized_zone_to_pixel_box(zone, 10, 10))

        result, count = render_overlay(sample_image(size=(10, 10)), [zone])
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

    def test_public_renderer_always_prepares_image_before_drawing(self):
        canonical = sample_image()
        zones = [{
            "zone_type": "demand", "draw_overlay": True,
            "x_start": 0.2, "x_end": 0.6, "y_top": 0.3, "y_bottom": 0.5,
        }]
        bounds = {
            "x_start": 0.05, "x_end": 0.95,
            "y_top": 0.05, "y_bottom": 0.95,
        }

        with patch.object(
            overlay_engine, "prepare_chart_image",
            return_value=(canonical, "image/png"),
        ) as prepare, patch.object(
            overlay_engine, "_render_overlay_prepared",
            return_value=(b"overlay-jpeg", 1),
        ) as render:
            result, count = render_overlay(
                b"untrusted-input",
                zones,
                plot_bounds=bounds,
                require_plot_bounds=True,
            )

        self.assertEqual((result, count), (b"overlay-jpeg", 1))
        prepare.assert_called_once_with(b"untrusted-input")
        render.assert_called_once_with(
            canonical,
            zones,
            plot_bounds=bounds,
            require_plot_bounds=True,
        )

    def test_rejects_invalid_image_bytes(self):
        with self.assertRaises(ValueError):
            render_overlay(b"not an image", [])


if __name__ == "__main__":
    unittest.main()
