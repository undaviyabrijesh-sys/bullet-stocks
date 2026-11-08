import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiogram.types import BufferedInputFile

from stockmind_telegram_mvp.app.telegram_delivery import (
    send_analysis_text,
    send_overlay_photo,
    split_telegram_text,
)


class AnalysisDeliveryTests(unittest.IsolatedAsyncioTestCase):
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

    def test_split_rejects_invalid_limits_and_non_text(self):
        with self.assertRaises(ValueError):
            split_telegram_text("text", limit=0)
        with self.assertRaises(ValueError):
            split_telegram_text("text", limit=True)
        with self.assertRaises(TypeError):
            split_telegram_text(b"text")

    async def test_long_analysis_edits_first_chunk_then_replies_with_remainder(self):
        text = ("BTC structure and invalidation.\n\n" * 300)
        chunks = split_telegram_text(text)
        message = SimpleNamespace(answer=AsyncMock())
        status = SimpleNamespace(edit_text=AsyncMock())

        delivered = await send_analysis_text(message, status, text)

        self.assertTrue(delivered)
        status.edit_text.assert_awaited_once_with(chunks[0])
        self.assertEqual(
            [call.args[0] for call in message.answer.await_args_list],
            chunks[1:],
        )
        self.assertEqual("".join([status.edit_text.await_args.args[0]] + [
            call.args[0] for call in message.answer.await_args_list
        ]), text)

    async def test_failed_followup_chunk_reports_partial_delivery_without_retry(self):
        text = "A" * 8500
        message = SimpleNamespace(answer=AsyncMock(side_effect=RuntimeError("Telegram unavailable")))
        status = SimpleNamespace(edit_text=AsyncMock())

        delivered = await send_analysis_text(message, status, text)

        self.assertFalse(delivered)
        status.edit_text.assert_awaited_once_with("A" * 4000)
        message.answer.assert_awaited_once_with("A" * 4000)


class OverlayDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_photo_delivery_returns_true(self):
        message = SimpleNamespace(answer_photo=AsyncMock())

        delivered = await send_overlay_photo(message, b"jpeg-bytes", caption="Estimated overlay")

        self.assertTrue(delivered)
        message.answer_photo.assert_awaited_once()
        photo = message.answer_photo.await_args.args[0]
        self.assertIsInstance(photo, BufferedInputFile)
        self.assertEqual(message.answer_photo.await_args.kwargs["caption"], "Estimated overlay")

    async def test_telegram_failure_returns_false_without_raising(self):
        message = SimpleNamespace(answer_photo=AsyncMock(side_effect=RuntimeError("Telegram unavailable")))

        delivered = await send_overlay_photo(message, b"jpeg-bytes", caption="Estimated overlay")

        self.assertFalse(delivered)
        message.answer_photo.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
