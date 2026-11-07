import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from stockmind_telegram_mvp.app.annotation_intake import prepare_annotation_draft


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
        self.assertEqual(draft["annotation_metadata"]["review_status"], "pending")
        self.assertIsNone(draft["annotation_metadata"]["annotator_id"])
        self.assertIsNone(draft["prediction_metadata"])
        self.assertNotIn(str(self.source), draft_path.read_text(encoding="utf-8"))

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
