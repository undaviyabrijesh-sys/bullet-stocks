"""Integration-style tests for the StockMind research-to-Telegram flow.

The bot module is imported with dummy configuration only. All provider and
Telegram operations are mocked; these tests never contact external services.
"""
import asyncio
import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from unittest.mock import patch as environment_patch


_TEST_ENV = {
    "TELEGRAM_BOT_TOKEN": "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi",
    "OPENAI_API_KEY": "stockmind-test-key",
    "TAPETIDE_TOKEN": "stockmind-test-tapetide-token",
    "OPENAI_MODEL": "gpt-6-luna-test",
    "OPENAI_VISION_MODEL": "gpt-6-luna-test",
}
with environment_patch.dict(os.environ, _TEST_ENV):
    bot_module = importlib.import_module("stockmind_telegram_mvp.app.bot")


class ResearchToTelegramFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        bot_module.histories.clear()
        bot_module.usage.clear()
        bot_module.usage_day.clear()
        bot_module.beta_users.clear()

        self.user_id = 987654321
        self.status = SimpleNamespace(edit_text=AsyncMock())
        self.message = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id),
            answer=AsyncMock(return_value=self.status),
            answer_photo=AsyncMock(),
        )

    def use_research(self, **methods):
        research = SimpleNamespace(**methods)
        self.enterContext(patch.object(bot_module, "research", research))
        return research

    async def test_text_answer_is_saved_and_delivered(self):
        research = self.use_research(answer=AsyncMock(return_value="BTC structure is unclear."))

        await bot_module.run_research(self.message, "Analyse BTC")

        research.answer.assert_awaited_once()
        self.status.edit_text.assert_awaited_once_with("BTC structure is unclear.")
        self.assertEqual(
            list(bot_module.histories[self.user_id]),
            [
                {"role": "user", "content": "Analyse BTC"},
                {"role": "assistant", "content": "BTC structure is unclear."},
            ],
        )

    async def test_chart_analysis_is_saved_even_when_overlay_delivery_fails(self):
        result = {
            "analysis": "Visible demand structure; verify the price scale.",
            "plot_bounds": {"x_start": 0.08, "x_end": 0.94, "y_top": 0.12, "y_bottom": 0.9},
            "zones": [{"zone_type": "demand", "draw_overlay": True}],
        }
        research = self.use_research(analyze_chart_image=AsyncMock(return_value=result))
        self.message.answer_photo.side_effect = RuntimeError("Telegram photo delivery failed")

        with patch.object(bot_module, "validate_chart_image", return_value="image/png"), patch.object(
            bot_module, "render_overlay", return_value=(b"overlay-jpeg", 1)
        ) as render:
            await bot_module.run_research(self.message, "Map BTC 1H DNS", b"chart-bytes", "image/png")

        research.analyze_chart_image.assert_awaited_once()
        render.assert_called_once()
        self.status.edit_text.assert_awaited_once_with(
            "Visible demand structure; verify the price scale."
        )
        self.message.answer_photo.assert_awaited_once()
        self.assertEqual(bot_module.histories[self.user_id][-1]["content"], result["analysis"])

    async def test_successful_answer_is_kept_when_text_delivery_is_ambiguous(self):
        research = self.use_research(answer=AsyncMock(return_value="Research completed."))
        self.status.edit_text.side_effect = TimeoutError("Telegram request timed out")

        await bot_module.run_research(self.message, "Check TCS")

        research.answer.assert_awaited_once()
        self.status.edit_text.assert_awaited_once_with("Research completed.")
        self.message.answer.assert_awaited_once()
        self.assertEqual(bot_module.histories[self.user_id][-1]["content"], "Research completed.")

    async def test_progress_message_failure_does_not_prevent_research_or_final_reply(self):
        research = self.use_research(answer=AsyncMock(return_value="Research completed."))
        self.message.answer.side_effect = [RuntimeError("progress unavailable"), None]

        await bot_module.run_research(self.message, "Check Reliance")

        research.answer.assert_awaited_once()
        self.message.answer.assert_any_await("Research completed.")
        self.assertEqual(bot_module.histories[self.user_id][-1]["content"], "Research completed.")

    async def test_provider_failure_does_not_append_a_successful_assistant_turn(self):
        research = self.use_research(answer=AsyncMock(side_effect=RuntimeError("provider unavailable")))

        await bot_module.run_research(self.message, "Analyse Nifty")

        research.answer.assert_awaited_once()
        self.status.edit_text.assert_awaited_once()
        self.assertIn("couldn't complete this analysis", self.status.edit_text.await_args.args[0])
        self.assertEqual(list(bot_module.histories[self.user_id]), [])


class BotModuleResourceCleanup(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        async def close_async_clients():
            await bot_module.market.close()
            await bot_module.crypto_market.close()
            await bot_module.bot.session.close()

        asyncio.run(close_async_clients())
        bot_module.research.client.close()


if __name__ == "__main__":
    unittest.main()
