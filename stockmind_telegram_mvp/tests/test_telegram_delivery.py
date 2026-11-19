import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiogram.types import BufferedInputFile
from PIL import Image

from stockmind_telegram_mvp.app import telegram_delivery
from stockmind_telegram_mvp.app.overlay_engine import render_overlay
from stockmind_telegram_mvp.app.telegram_delivery import (
    send_analysis_text,
    send_overlay_photo,
    split_telegram_text,
)


class AnalysisDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_analysis_is_not_sent_or_used_to_edit_status(self):
        message = SimpleNamespace(answer=AsyncMock())
        status = SimpleNamespace(edit_text=AsyncMock())

        delivered = await send_analysis_text(message, status, "")

        self.assertFalse(delivered)
        message.answer.assert_not_awaited()
        status.edit_text.assert_not_awaited()

    async def test_empty_analysis_without_status_does_not_send_empty_reply(self):
        message = SimpleNamespace(answer=AsyncMock())

        delivered = await send_analysis_text(message, None, "")

        self.assertFalse(delivered)
        message.answer.assert_not_awaited()

    async def test_successful_status_edit_returns_true(self):
        message = SimpleNamespace(answer=AsyncMock())
        status = SimpleNamespace(edit_text=AsyncMock())

        delivered = await send_analysis_text(message, status, "Analysis complete")

        self.assertTrue(delivered)
        status.edit_text.assert_awaited_once_with("Analysis complete")
        message.answer.assert_not_awaited()

    async def test_missing_status_message_falls_back_to_reply(self):
        message = SimpleNamespace(answer=AsyncMock())
        status = SimpleNamespace(
            edit_text=AsyncMock(side_effect=RuntimeError("Bad Request: message to edit not found"))
        )

        delivered = await send_analysis_text(message, status, "Analysis complete")

        self.assertTrue(delivered)
        message.answer.assert_awaited_once_with("Analysis complete")

    async def test_unchanged_status_is_treated_as_delivered_without_duplicate(self):
        message = SimpleNamespace(answer=AsyncMock())
        status = SimpleNamespace(
            edit_text=AsyncMock(side_effect=RuntimeError("Bad Request: message is not modified"))
        )

        delivered = await send_analysis_text(message, status, "Analysis complete")

        self.assertTrue(delivered)
        message.answer.assert_not_awaited()

    async def test_ambiguous_transport_failure_does_not_send_duplicate(self):
        message = SimpleNamespace(answer=AsyncMock())
        status = SimpleNamespace(edit_text=AsyncMock(side_effect=TimeoutError("request timed out")))

        delivered = await send_analysis_text(message, status, "Analysis complete")

        self.assertFalse(delivered)
        message.answer.assert_not_awaited()

    async def test_missing_status_uses_reply_and_reports_failure(self):
        message = SimpleNamespace(answer=AsyncMock())
        message.answer.side_effect = RuntimeError("Telegram unavailable")

        delivered = await send_analysis_text(message, None, "Analysis complete")

        self.assertFalse(delivered)
        message.answer.assert_awaited_once_with("Analysis complete")


    def test_split_preserves_every_character_and_prefers_paragraph_boundaries(self):
        text = ("First paragraph.\n\nSecond paragraph.\n\n" * 300)

        chunks = split_telegram_text(text)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 4000 for chunk in chunks))
        self.assertEqual("".join(chunks), text)
        self.assertTrue(chunks[0].endswith("\n\n"))

    def test_split_handles_long_unbroken_tokens_without_data_loss(self):
        text = "X" * 8001

        chunks = split_telegram_text(text)

        self.assertEqual([len(chunk) for chunk in chunks], [4000, 4000, 1])
        self.assertEqual("".join(chunks), text)

    def test_split_counts_supplementary_unicode_as_two_utf16_units(self):
        text = "😀" * 2501

        chunks = split_telegram_text(text)

        self.assertEqual([len(chunk) for chunk in chunks], [2000, 501])
        self.assertTrue(all(
            len(chunk.encode("utf-16-le")) // 2 <= 4000 for chunk in chunks
        ))
        self.assertEqual("".join(chunks), text)

    def test_split_combines_surrogate_pairs_and_replaces_unpaired_surrogates(self):
        text = (
            "A"
            + chr(0xD83D)
            + chr(0xDE00)
            + "B"
            + chr(0xD800)
            + "C"
            + chr(0xDC00)
        )

        chunks = split_telegram_text(text)

        expected = "A😀B�C�"
        self.assertEqual(chunks, [expected])
        self.assertEqual("".join(chunks), expected)
        self.assertEqual(chunks[0].encode("utf-8").decode("utf-8"), expected)

    async def test_delivery_normalizes_malformed_surrogates_before_telegram_encoding(self):
        message = SimpleNamespace(answer=AsyncMock())
        malformed = "Chart" + chr(0xD800) + " zone"

        delivered = await send_analysis_text(message, None, malformed)

        self.assertTrue(delivered)
        message.answer.assert_awaited_once_with("Chart� zone")

    def test_split_scans_large_responses_linearly(self):
        text = "X" * 400_000
        original_counter = telegram_delivery._telegram_text_units
        scanned_characters = 0

        def count_scanned(value):
            nonlocal scanned_characters
            scanned_characters += len(value)
            return original_counter(value)

        with patch.object(
            telegram_delivery,
            "_telegram_text_units",
            side_effect=count_scanned,
        ):
            chunks = split_telegram_text(text)

        self.assertEqual(len(chunks), 100)
        self.assertEqual("".join(chunks), text)
        # One full-input count plus one count per emitted chunk is linear.
        # The old implementation repeatedly counted every remaining suffix.
        self.assertLessEqual(scanned_characters, 2 * len(text))

    def test_split_rejects_invalid_limits_and_non_text(self):
        with self.assertRaises(ValueError):
            split_telegram_text("text", limit=0)
        with self.assertRaises(ValueError):
            split_telegram_text("text", limit=True)
        with self.assertRaises(ValueError):
            split_telegram_text("😀", limit=1)
        with self.assertRaises(TypeError):
            split_telegram_text(b"text")


class OverlayDeliveryTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def sample_image(size=(640, 360), image_format="JPEG"):
        output = BytesIO()
        Image.new("RGB", size, "white").save(
            output, format=image_format, optimize=True
        )
        return output.getvalue()

    async def test_successful_photo_delivery_returns_true(self):
        message = SimpleNamespace(
            answer_photo=AsyncMock(),
            answer_document=AsyncMock(),
        )
        image_bytes = self.sample_image()

        delivered = await send_overlay_photo(
            message, image_bytes, caption="Estimated overlay"
        )

        self.assertTrue(delivered)
        message.answer_photo.assert_awaited_once()
        message.answer_document.assert_not_awaited()
        photo = message.answer_photo.await_args.args[0]
        self.assertIsInstance(photo, BufferedInputFile)
        self.assertEqual(photo.filename, "stockmind-chart-zones.jpg")
        self.assertEqual(
            message.answer_photo.await_args.kwargs["caption"],
            "Estimated overlay",
        )

    async def test_png_overlay_uses_png_filename(self):
        message = SimpleNamespace(
            answer_photo=AsyncMock(),
            answer_document=AsyncMock(),
        )
        png_bytes = self.sample_image(image_format="PNG")

        delivered = await send_overlay_photo(
            message, png_bytes, caption="Estimated overlay"
        )

        self.assertTrue(delivered)
        photo = message.answer_photo.await_args.args[0]
        self.assertIsInstance(photo, BufferedInputFile)
        self.assertEqual(photo.filename, "stockmind-chart-zones.png")
        message.answer_document.assert_not_awaited()

    async def test_transparent_png_render_reaches_telegram_upload_unchanged(self):
        source = Image.new("RGBA", (64, 48), (0, 0, 0, 0))
        source.putpixel((2, 2), (20, 40, 60, 255))
        source_bytes = BytesIO()
        source.save(source_bytes, format="PNG")
        zone = {
            "zone_type": "demand",
            "draw_overlay": True,
            "confidence": "high",
            "x_start": 0.25,
            "x_end": 0.75,
            "y_top": 0.25,
            "y_bottom": 0.65,
        }

        rendered, zone_count = render_overlay(source_bytes.getvalue(), [zone])
        self.assertEqual(zone_count, 1)
        self.assertIsNotNone(rendered)

        with Image.open(BytesIO(rendered)) as output:
            self.assertEqual(output.format, "PNG")
            self.assertEqual(output.mode, "RGBA")
            self.assertEqual(output.getpixel((5, 5))[3], 0)
            self.assertEqual(output.getpixel((32, 24))[3], 60)

        message = SimpleNamespace(
            answer_photo=AsyncMock(),
            answer_document=AsyncMock(),
        )
        delivered = await send_overlay_photo(
            message, rendered, caption="Estimated overlay"
        )

        self.assertTrue(delivered)
        message.answer_photo.assert_awaited_once()
        message.answer_document.assert_not_awaited()
        upload = message.answer_photo.await_args.args[0]
        self.assertIsInstance(upload, BufferedInputFile)
        self.assertEqual(upload.filename, "stockmind-chart-zones.png")
        self.assertEqual(upload.data, rendered)
        with Image.open(BytesIO(upload.data)) as uploaded_image:
            self.assertEqual(uploaded_image.mode, "RGBA")
            self.assertEqual(uploaded_image.getpixel((5, 5))[3], 0)
            self.assertEqual(uploaded_image.getpixel((32, 24))[3], 60)

    async def test_photo_byte_limit_routes_exact_image_as_document(self):
        message = SimpleNamespace(
            answer_photo=AsyncMock(),
            answer_document=AsyncMock(),
        )
        png_bytes = self.sample_image(image_format="PNG")

        with patch.object(telegram_delivery, "TELEGRAM_PHOTO_MAX_BYTES", 1):
            delivered = await send_overlay_photo(
                message, png_bytes, caption="Estimated overlay"
            )

        self.assertTrue(delivered)
        message.answer_photo.assert_not_awaited()
        message.answer_document.assert_awaited_once()
        document = message.answer_document.await_args.args[0]
        self.assertEqual(document.filename, "stockmind-chart-zones.png")
        self.assertEqual(document.data, png_bytes)

    async def test_extreme_aspect_ratio_routes_image_as_document(self):
        message = SimpleNamespace(
            answer_photo=AsyncMock(),
            answer_document=AsyncMock(),
        )
        png_bytes = self.sample_image(size=(1000, 20), image_format="PNG")

        delivered = await send_overlay_photo(
            message, png_bytes, caption="Estimated overlay"
        )

        self.assertTrue(delivered)
        message.answer_photo.assert_not_awaited()
        message.answer_document.assert_awaited_once()

    async def test_photo_limit_helper_checks_size_dimension_sum_and_ratio(self):
        self.assertTrue(
            telegram_delivery._fits_telegram_photo_limits(
                5000, 5000, telegram_delivery.TELEGRAM_PHOTO_MAX_BYTES
            )
        )
        self.assertFalse(
            telegram_delivery._fits_telegram_photo_limits(5001, 5000, 1)
        )
        self.assertFalse(
            telegram_delivery._fits_telegram_photo_limits(1000, 20, 1)
        )
        self.assertFalse(
            telegram_delivery._fits_telegram_photo_limits(
                100, 100, telegram_delivery.TELEGRAM_PHOTO_MAX_BYTES + 1
            )
        )
        self.assertFalse(
            telegram_delivery._fits_telegram_photo_limits(True, 100, 1)
        )

    async def test_document_above_safety_limit_is_not_sent(self):
        message = SimpleNamespace(
            answer_photo=AsyncMock(),
            answer_document=AsyncMock(),
        )
        png_bytes = self.sample_image(image_format="PNG")

        with patch.object(telegram_delivery, "TELEGRAM_DOCUMENT_MAX_BYTES", 1):
            delivered = await send_overlay_photo(
                message, png_bytes, caption="Estimated overlay"
            )

        self.assertFalse(delivered)
        message.answer_photo.assert_not_awaited()
        message.answer_document.assert_not_awaited()

    async def test_invalid_overlay_bytes_are_not_sent(self):
        message = SimpleNamespace(
            answer_photo=AsyncMock(),
            answer_document=AsyncMock(),
        )

        delivered = await send_overlay_photo(
            message, b"not-an-image", caption="Estimated overlay"
        )

        self.assertFalse(delivered)
        message.answer_photo.assert_not_awaited()
        message.answer_document.assert_not_awaited()

    async def test_truncated_jpeg_pixel_data_is_rejected_before_upload(self):
        message = SimpleNamespace(
            answer_photo=AsyncMock(),
            answer_document=AsyncMock(),
        )
        truncated = self.sample_image(image_format="JPEG")[:-10]

        delivered = await send_overlay_photo(
            message, truncated, caption="Estimated overlay"
        )

        self.assertFalse(delivered)
        message.answer_photo.assert_not_awaited()
        message.answer_document.assert_not_awaited()

    async def test_animated_png_is_rejected_before_upload(self):
        message = SimpleNamespace(
            answer_photo=AsyncMock(),
            answer_document=AsyncMock(),
        )
        output = BytesIO()
        frames = [
            Image.new("RGBA", (32, 32), color)
            for color in ("red", "blue")
        ]
        frames[0].save(
            output,
            format="PNG",
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )

        delivered = await send_overlay_photo(
            message, output.getvalue(), caption="Estimated overlay"
        )

        self.assertFalse(delivered)
        message.answer_photo.assert_not_awaited()
        message.answer_document.assert_not_awaited()

    async def test_overlay_pixel_limit_is_checked_before_upload(self):
        message = SimpleNamespace(
            answer_photo=AsyncMock(),
            answer_document=AsyncMock(),
        )
        png_bytes = self.sample_image(size=(11, 10), image_format="PNG")

        with patch.object(telegram_delivery, "TELEGRAM_OVERLAY_MAX_PIXELS", 100):
            delivered = await send_overlay_photo(
                message, png_bytes, caption="Estimated overlay"
            )

        self.assertFalse(delivered)
        message.answer_photo.assert_not_awaited()
        message.answer_document.assert_not_awaited()

    async def test_telegram_photo_failure_returns_false_without_document_retry(self):
        message = SimpleNamespace(
            answer_photo=AsyncMock(side_effect=RuntimeError("Telegram unavailable")),
            answer_document=AsyncMock(),
        )

        delivered = await send_overlay_photo(
            message, self.sample_image(), caption="Estimated overlay"
        )

        self.assertFalse(delivered)
        message.answer_photo.assert_awaited_once()
        message.answer_document.assert_not_awaited()

    async def test_telegram_document_failure_returns_false(self):
        message = SimpleNamespace(
            answer_photo=AsyncMock(),
            answer_document=AsyncMock(
                side_effect=RuntimeError("Telegram unavailable")
            ),
        )
        png_bytes = self.sample_image(image_format="PNG")

        with patch.object(telegram_delivery, "TELEGRAM_PHOTO_MAX_BYTES", 1):
            delivered = await send_overlay_photo(
                message, png_bytes, caption="Estimated overlay"
            )

        self.assertFalse(delivered)
        message.answer_document.assert_awaited_once()

if __name__ == "__main__":
    unittest.main()
