import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from stockmind_telegram_mvp.app.annotation_intake import (
    _strip_annotation_metadata,
    prepare_annotation_draft,
)


def _png_bytes(size=(64, 48)):
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format="PNG")
    return output.getvalue()


class AnnotationIntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "source.png"
        self.image_bytes = _png_bytes()
        self.source.write_bytes(self.image_bytes)
        self.output = self.root / "inbox"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_copies_original_and_records_hash_dimensions_without_fabricating_labels(self):
        result = prepare_annotation_draft(
            self.source,
            self.output,
            "btc-1h-001",
            market_context={"asset_class": "Crypto", "symbol": "btcusdt", "exchange": "Binance", "timeframe": "1H"},
        )
        copied = self.output / result["image_file"]
        draft_path = self.output / result["draft_file"]
        draft = json.loads(draft_path.read_text(encoding="utf-8"))

        self.assertEqual(copied.read_bytes(), self.image_bytes)
        self.assertEqual(result["image_sha256"], hashlib.sha256(self.image_bytes).hexdigest())
        self.assertEqual((draft["image_width"], draft["image_height"]), (64, 48))
        self.assertEqual(draft["market_context"]["symbol"], "BTCUSDT")
        self.assertEqual(draft["market_context"]["exchange"], "binance")
        self.assertFalse(draft["benchmark_ready"])
        self.assertEqual(draft["reference_zones"], [])
        self.assertIsNone(draft["reference_plot_bounds"])
        self.assertEqual(draft["price_axis_anchors"], [])
        self.assertIsNone(draft["price_axis_scale"])
        self.assertIsNone(draft["price_axis_plot_bounds"])
        self.assertEqual(draft["annotation_metadata"]["review_status"], "pending")
        self.assertIsNone(draft["annotation_metadata"]["annotator_id"])
        self.assertIsNone(draft["prediction_metadata"])
        self.assertNotIn(str(self.source), draft_path.read_text(encoding="utf-8"))

    def test_exif_oriented_screenshot_is_normalized_before_hash_and_storage(self):
        exif = Image.Exif()
        exif[274] = 6
        output = BytesIO()
        Image.new("RGB", (64, 48), "white").save(output, format="PNG", exif=exif)
        original = output.getvalue()
        self.source.write_bytes(original)

        result = prepare_annotation_draft(
            self.source, self.output, "rotated-btc-1h"
        )
        stored = (self.output / result["image_file"]).read_bytes()
        draft = json.loads(
            (self.output / result["draft_file"]).read_text(encoding="utf-8")
        )
        with Image.open(BytesIO(stored)) as image:
            self.assertEqual(image.size, (48, 64))
            self.assertNotIn(274, image.getexif())
        self.assertEqual(draft["image_width"], 48)
        self.assertEqual(draft["image_height"], 64)
        self.assertEqual(result["image_sha256"], hashlib.sha256(stored).hexdigest())
        self.assertNotEqual(stored, original)


    def test_strips_png_exif_and_descriptive_text_metadata(self):
        from PIL.PngImagePlugin import PngInfo

        exif = Image.Exif()
        exif[315] = "Private annotator name"
        exif[270] = "Private screenshot description"
        pnginfo = PngInfo()
        pnginfo.add_text("Author", "Private annotator name")
        pnginfo.add_text("Comment", "Internal note")
        output = BytesIO()
        original_image = Image.new("RGB", (64, 48), (24, 80, 140))
        original_image.save(
            output, format="PNG", exif=exif, pnginfo=pnginfo
        )
        original = output.getvalue()
        self.source.write_bytes(original)

        result = prepare_annotation_draft(
            self.source, self.output, "metadata-btc-1h"
        )
        stored = (self.output / result["image_file"]).read_bytes()
        with Image.open(BytesIO(stored)) as image:
            self.assertEqual(image.size, (64, 48))
            self.assertEqual(image.getpixel((10, 10)), (24, 80, 140))
            self.assertFalse(image.getexif())
            self.assertNotIn("Author", image.info)
            self.assertNotIn("Comment", image.info)
        self.assertNotEqual(stored, original)
        self.assertEqual(
            result["image_sha256"], hashlib.sha256(stored).hexdigest()
        )

    def test_strips_jpeg_exif_without_changing_dimensions(self):
        exif = Image.Exif()
        exif[315] = "Private annotator name"
        output = BytesIO()
        Image.new("RGB", (80, 60), (90, 120, 150)).save(
            output, format="JPEG", quality=90, exif=exif
        )
        self.source.write_bytes(output.getvalue())

        result = prepare_annotation_draft(
            self.source, self.output, "metadata-jpeg-1h"
        )
        stored = (self.output / result["image_file"]).read_bytes()
        with Image.open(BytesIO(stored)) as image:
            self.assertEqual(image.size, (80, 60))
            self.assertFalse(image.getexif())
        self.assertEqual(
            result["image_sha256"], hashlib.sha256(stored).hexdigest()
        )

    def test_strips_webp_exif_with_lossless_pixel_preservation(self):
        exif = Image.Exif()
        exif[315] = "Private annotator name"
        output = BytesIO()
        original_image = Image.new("RGB", (32, 24), (12, 34, 56))
        original_image.save(
            output, format="WEBP", quality=80, exif=exif
        )
        original = output.getvalue()
        with Image.open(BytesIO(original)) as image:
            original_pixels = image.convert("RGB").tobytes()
        self.source.write_bytes(original)

        result = prepare_annotation_draft(
            self.source, self.output, "metadata-webp-1h"
        )
        stored = (self.output / result["image_file"]).read_bytes()
        with Image.open(BytesIO(stored)) as image:
            self.assertEqual(image.size, (32, 24))
            self.assertFalse(image.getexif())
            self.assertEqual(image.convert("RGB").tobytes(), original_pixels)
        self.assertEqual(
            result["image_sha256"], hashlib.sha256(stored).hexdigest()
        )

    def test_metadata_cleaning_respects_output_byte_limit(self):
        from PIL.PngImagePlugin import PngInfo

        pnginfo = PngInfo()
        pnginfo.add_text("Author", "Private annotator name")
        output = BytesIO()
        Image.new("RGB", (64, 48), "white").save(
            output, format="PNG", pnginfo=pnginfo
        )
        with self.assertRaisesRegex(ValueError, "Metadata-cleaned chart image exceeds"):
            _strip_annotation_metadata(output.getvalue(), max_image_bytes=1)
        self.assertFalse(self.output.exists())

    def test_rejects_oversized_image_before_creating_outputs(self):
        with self.assertRaisesRegex(ValueError, "exceeds max_image_bytes"):
            prepare_annotation_draft(self.source, self.output, "chart-001", max_image_bytes=10)
        self.assertFalse(self.output.exists())

    def test_rejects_invalid_image_bytes(self):
        self.source.write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "valid JPEG, PNG or WEBP"):
            prepare_annotation_draft(self.source, self.output, "chart-001")
        self.assertFalse(self.output.exists())

    def test_rejects_animated_image(self):
        output = BytesIO()
        first = Image.new("RGB", (8, 8), "white")
        second = Image.new("RGB", (8, 8), "black")
        first.save(output, format="GIF", save_all=True, append_images=[second], duration=100, loop=0)
        self.source.write_bytes(output.getvalue())
        with self.assertRaisesRegex(ValueError, "JPEG, PNG or WEBP"):
            prepare_annotation_draft(self.source, self.output, "chart-001")
        self.assertFalse(self.output.exists())

    def test_rejects_path_like_fixture_ids(self):
        with self.assertRaisesRegex(ValueError, "fixture_id"):
            prepare_annotation_draft(self.source, self.output, "../chart")
        self.assertFalse(self.output.exists())

    def test_rejects_unknown_or_control_character_market_context(self):
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            prepare_annotation_draft(self.source, self.output, "chart-001", market_context={"name": "private"})
        with self.assertRaisesRegex(ValueError, "control characters"):
            prepare_annotation_draft(self.source, self.output, "chart-002", market_context={"symbol": "BTC\nUSDT"})
        self.assertFalse(self.output.exists())

    def test_refuses_to_overwrite_existing_draft_or_image(self):
        prepare_annotation_draft(self.source, self.output, "chart-001")
        with self.assertRaises(FileExistsError):
            prepare_annotation_draft(self.source, self.output, "chart-001")

    def test_removes_partial_image_when_image_write_fails(self):
        destination = self.output / "images" / "partial-write.png"
        real_open = open

        class PartialWriter:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                self.stream.close()

            def write(self, value):
                self.stream.write(value[: max(1, len(value) // 2)])
                raise OSError("simulated disk full")

        def failing_open(path, mode="r", *args, **kwargs):
            stream = real_open(path, mode, *args, **kwargs)
            if path == destination and mode == "xb":
                return PartialWriter(stream)
            return stream

        with patch.object(Path, "open", autospec=True, side_effect=failing_open):
            with self.assertRaisesRegex(OSError, "simulated disk full"):
                prepare_annotation_draft(self.source, self.output, "partial-write")

        self.assertFalse(destination.exists())
        self.assertFalse((self.output / "partial-write.draft.json").exists())

    def test_rejects_boolean_or_non_positive_byte_limit(self):
        for invalid in (True, 0, -1, 1.5):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "max_image_bytes must be a positive integer"):
                    prepare_annotation_draft(self.source, self.output, "chart-001", max_image_bytes=invalid)
        self.assertFalse(self.output.exists())

    def test_rejects_output_images_symlink_that_escapes_inbox(self):
        self.output.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (self.output / "images").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "must not resolve outside"):
            prepare_annotation_draft(self.source, self.output, "chart-001")
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
