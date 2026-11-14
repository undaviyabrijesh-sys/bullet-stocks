import hashlib
import json
import struct
import tempfile
import unittest
import zlib
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from stockmind_telegram_mvp.app.annotation_intake import (
    _jpeg_has_untrusted_container_tail,
    _strip_annotation_metadata,
    _webp_contains_unexposed_metadata,
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

    def test_strips_png_text_fields_named_like_nonessential_format_metadata(self):
        from PIL import PngImagePlugin

        pnginfo = PngImagePlugin.PngInfo()
        private_values = {
            "timestamp": "2026-11-13T14:00:00+05:30",
            "duration": "Private capture duration",
            "loop": "Private capture note",
        }
        for key, value in private_values.items():
            pnginfo.add_text(key, value)
        output = BytesIO()
        Image.new("RGB", (64, 48), (24, 80, 140)).save(
            output, format="PNG", pnginfo=pnginfo
        )
        self.source.write_bytes(output.getvalue())

        result = prepare_annotation_draft(
            self.source, self.output, "metadata-key-collision"
        )
        stored = (self.output / result["image_file"]).read_bytes()
        with Image.open(BytesIO(stored)) as image:
            for key in private_values:
                self.assertNotIn(key, image.info)
            self.assertEqual(image.getpixel((10, 10)), (24, 80, 140))
        self.assertEqual(
            result["image_sha256"], hashlib.sha256(stored).hexdigest()
        )

    def test_strips_png_time_chunk_not_exposed_by_pillow(self):
        original = _png_bytes()
        timestamp = struct.pack(">HBBBBB", 2026, 11, 13, 12, 34, 56)
        chunk_type = b"tIME"
        chunk = (
            struct.pack(">I", len(timestamp))
            + chunk_type
            + timestamp
            + struct.pack(">I", zlib.crc32(chunk_type + timestamp) & 0xFFFFFFFF)
        )
        insert_at = original.rfind(b"IEND") - 4
        original_with_time = original[:insert_at] + chunk + original[insert_at:]
        self.source.write_bytes(original_with_time)

        # Pillow does not expose the native PNG modification-time chunk through
        # image.info, so the intake path must inspect the chunk stream itself.
        with Image.open(BytesIO(original_with_time)) as image:
            self.assertNotIn("timestamp", image.info)

        result = prepare_annotation_draft(
            self.source, self.output, "png-time-metadata"
        )
        stored = (self.output / result["image_file"]).read_bytes()
        self.assertNotIn(b"tIME", stored)
        self.assertNotEqual(stored, original_with_time)
        with Image.open(BytesIO(stored)) as image:
            self.assertEqual(image.size, (64, 48))
            self.assertEqual(image.getpixel((10, 10)), (255, 255, 255))
        self.assertEqual(
            result["image_sha256"], hashlib.sha256(stored).hexdigest()
        )

    def test_strips_unrecognized_private_png_chunk(self):
        original = _png_bytes()
        chunk_type = b"vpAg"  # Unknown private ancillary chunk.
        private_payload = b"PRIVATE-ANNOTATOR-ID=ann-secret"
        chunk = (
            struct.pack(">I", len(private_payload))
            + chunk_type
            + private_payload
            + struct.pack(">I", zlib.crc32(chunk_type + private_payload) & 0xFFFFFFFF)
        )
        insert_at = original.rfind(b"IEND") - 4
        original_with_private_chunk = original[:insert_at] + chunk + original[insert_at:]

        # Pillow accepts and ignores this valid ancillary chunk.
        with Image.open(BytesIO(original_with_private_chunk)) as image:
            image.load()
            self.assertEqual(image.getpixel((10, 10)), (255, 255, 255))

        sanitized = _strip_annotation_metadata(
            original_with_private_chunk, 5 * 1024 * 1024
        )
        self.assertNotEqual(sanitized, original_with_private_chunk)
        self.assertNotIn(chunk_type, sanitized)
        with Image.open(BytesIO(sanitized)) as image:
            self.assertEqual(image.size, (64, 48))
            self.assertEqual(image.getpixel((10, 10)), (255, 255, 255))

    def test_strips_private_bytes_appended_after_png_iend(self):
        original = _png_bytes()
        private_suffix = b"PRIVATE-ANNOTATOR-ID=ann-secret"
        image_with_suffix = original + private_suffix

        # Pillow decodes the image and ignores the bytes after IEND.
        with Image.open(BytesIO(image_with_suffix)) as image:
            image.load()
            self.assertEqual(image.getpixel((10, 10)), (255, 255, 255))

        sanitized = _strip_annotation_metadata(
            image_with_suffix, 5 * 1024 * 1024
        )
        self.assertNotEqual(sanitized, image_with_suffix)
        self.assertFalse(sanitized.endswith(private_suffix))
        with Image.open(BytesIO(sanitized)) as image:
            self.assertEqual(image.size, (64, 48))
            self.assertEqual(image.getpixel((10, 10)), (255, 255, 255))

    def test_strips_unexposed_jpeg_application_segment(self):
        output = BytesIO()
        Image.new("RGB", (80, 60), (90, 120, 150)).save(
            output, format="JPEG", quality=90
        )
        plain_jpeg = output.getvalue()

        # Ordinary JFIF density metadata is retained without re-encoding.
        self.assertEqual(
            _strip_annotation_metadata(plain_jpeg, 5 * 1024 * 1024),
            plain_jpeg,
        )

        private_payload = b"PRIVATE-ANNOTATOR-ID=ann-secret"
        segment = (
            b"\xff\xe3"
            + struct.pack(">H", len(private_payload) + 2)
            + private_payload
        )
        original = plain_jpeg[:2] + segment + plain_jpeg[2:]
        with Image.open(BytesIO(original)) as image:
            self.assertNotIn("APP3", image.info)
            self.assertIn(("APP3", private_payload), image.applist)

        sanitized = _strip_annotation_metadata(original, 5 * 1024 * 1024)
        self.assertNotEqual(sanitized, original)
        with Image.open(BytesIO(sanitized)) as image:
            self.assertEqual(image.size, (80, 60))
            self.assertFalse(image.getexif())
            self.assertNotIn("APP3", [marker for marker, _ in image.applist])


    def test_jpeg_marker_parser_skips_app_payload_and_stuffed_scan_bytes(self):
        # The minimal framed stream includes FF 00 (a stuffed data byte),
        # FF D0 (a restart marker), and FF D9 inside an APP payload.
        scan = bytes.fromhex(
            "ffd8 ffda 0008 010100003f00 11ff00d9 22ffd0 33 ffd9"
        )
        self.assertFalse(_jpeg_has_untrusted_container_tail(scan))

        payload = b"private-marker:" + bytes.fromhex("ffd9") + b":still-in-segment"
        app3 = b"\xff\xe3" + struct.pack(">H", len(payload) + 2) + payload
        with_app_payload = scan[:2] + app3 + scan[2:]
        self.assertFalse(_jpeg_has_untrusted_container_tail(with_app_payload))

    def test_strips_private_bytes_appended_after_jpeg_eoi(self):
        output = BytesIO()
        Image.new("RGB", (80, 60), (90, 120, 150)).save(
            output, format="JPEG", quality=90
        )
        original = output.getvalue()
        self.assertFalse(_jpeg_has_untrusted_container_tail(original))

        private_suffix = b"PRIVATE-ANNOTATOR-ID=ann-secret"
        image_with_suffix = original + private_suffix
        self.assertTrue(_jpeg_has_untrusted_container_tail(image_with_suffix))

        sanitized = _strip_annotation_metadata(
            image_with_suffix, 5 * 1024 * 1024
        )
        self.assertNotEqual(sanitized, image_with_suffix)
        self.assertNotIn(private_suffix, sanitized)
        self.assertFalse(_jpeg_has_untrusted_container_tail(sanitized))
        with Image.open(BytesIO(sanitized)) as image:
            image.load()
            self.assertEqual(image.size, (80, 60))

        # Exercise the complete intake path, including EXIF normalization,
        # canonical storage and the hash recorded in the pending draft.
        self.source.write_bytes(image_with_suffix)
        result = prepare_annotation_draft(
            self.source, self.output, "jpeg-trailing-data"
        )
        stored = (self.output / result["image_file"]).read_bytes()
        self.assertNotIn(private_suffix, stored)
        self.assertFalse(_jpeg_has_untrusted_container_tail(stored))
        self.assertEqual(
            result["image_sha256"], hashlib.sha256(stored).hexdigest()
        )

    def test_jpeg_marker_parser_accepts_progressive_multiscan_images(self):
        output = BytesIO()
        Image.effect_noise((128, 96), 100).convert("RGB").save(
            output, format="JPEG", quality=88, progressive=True, optimize=True
        )
        self.assertFalse(
            _jpeg_has_untrusted_container_tail(output.getvalue())
        )

    def test_jpeg_marker_parser_rejects_missing_eoi_and_truncated_segments(self):
        output = BytesIO()
        Image.new("RGB", (80, 60), (90, 120, 150)).save(
            output, format="JPEG", quality=90
        )
        original = output.getvalue()
        self.assertTrue(_jpeg_has_untrusted_container_tail(original[:-2]))

        truncated_segment = b"\xff\xd8\xff\xe1\x00\x20\x01\x02"
        self.assertTrue(_jpeg_has_untrusted_container_tail(truncated_segment))

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

    def test_webp_container_detector_accepts_plain_static_image(self):
        output = BytesIO()
        Image.new("RGB", (32, 24), (12, 34, 56)).save(
            output, format="WEBP", lossless=True
        )
        self.assertFalse(_webp_contains_unexposed_metadata(output.getvalue()))

    def test_webp_container_detector_rejects_trailing_and_truncated_chunks(self):
        output = BytesIO()
        Image.new("RGB", (32, 24), (12, 34, 56)).save(
            output, format="WEBP", lossless=True
        )
        original = output.getvalue()
        self.assertTrue(
            _webp_contains_unexposed_metadata(
                original + b"PRIVATE-ANNOTATOR-ID=ann-secret"
            )
        )

        malformed = bytearray(original)
        struct.pack_into("<I", malformed, 16, len(original))
        self.assertTrue(_webp_contains_unexposed_metadata(bytes(malformed)))

    def test_strips_private_riff_chunk_from_webp(self):
        output = BytesIO()
        Image.new("RGB", (32, 24), (12, 34, 56)).save(
            output, format="WEBP", lossless=True
        )
        original = output.getvalue()
        private_payload = b"PRIVATE-ANNOTATOR-ID=ann-secret"
        chunk = (
            b"JUNK"
            + struct.pack("<I", len(private_payload))
            + private_payload
            + (bytes((0,)) if len(private_payload) & 1 else b"")
        )
        with_private_chunk = bytearray(original + chunk)
        struct.pack_into("<I", with_private_chunk, 4, len(with_private_chunk) - 8)
        with_private_chunk = bytes(with_private_chunk)

        with Image.open(BytesIO(with_private_chunk)) as image:
            image.load()
            original_pixels = image.convert("RGB").tobytes()

        sanitized = _strip_annotation_metadata(
            with_private_chunk, 5 * 1024 * 1024
        )
        self.assertNotEqual(sanitized, with_private_chunk)
        self.assertNotIn(b"JUNK", sanitized)
        self.assertNotIn(private_payload, sanitized)
        with Image.open(BytesIO(sanitized)) as image:
            self.assertEqual(image.size, (32, 24))
            self.assertEqual(image.convert("RGB").tobytes(), original_pixels)

    def test_strips_bytes_appended_after_webp_riff_container(self):
        output = BytesIO()
        Image.new("RGB", (32, 24), (12, 34, 56)).save(
            output, format="WEBP", lossless=True
        )
        original = output.getvalue()
        private_suffix = b"PRIVATE-ANNOTATOR-ID=ann-secret"
        image_with_suffix = original + private_suffix

        with Image.open(BytesIO(image_with_suffix)) as image:
            image.load()
            original_pixels = image.convert("RGB").tobytes()

        sanitized = _strip_annotation_metadata(
            image_with_suffix, 5 * 1024 * 1024
        )
        self.assertNotEqual(sanitized, image_with_suffix)
        self.assertFalse(sanitized.endswith(private_suffix))
        with Image.open(BytesIO(sanitized)) as image:
            self.assertEqual(image.size, (32, 24))
            self.assertEqual(image.convert("RGB").tobytes(), original_pixels)

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
