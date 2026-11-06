import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiogram.types import BufferedInputFile

from stockmind_telegram_mvp.app.telegram_delivery import send_overlay_photo


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
