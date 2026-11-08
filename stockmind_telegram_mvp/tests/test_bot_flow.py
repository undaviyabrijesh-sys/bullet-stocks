"""Integration-style tests for the StockMind research-to-Telegram flow.

The bot module is imported with dummy configuration only. All provider and
Telegram operations are mocked; these tests never contact external services.
"""
import asyncio
import importlib
import os
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from unittest.mock import patch as environment_patch


from stockmind_telegram_mvp.app.ai import ResearchTurnDeadlineExceeded


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
        bot_module.user_locks.clear()

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

    async def test_same_user_requests_are_serialized_and_second_sees_first_turn(self):
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        calls = []

        async def answer(history, user_text):
            calls.append((list(history), user_text))
            if user_text == "first request":
                first_started.set()
                await release_first.wait()
            return f"answer: {user_text}"

        research = self.use_research(answer=AsyncMock(side_effect=answer))
        first_task = asyncio.create_task(bot_module.run_research(self.message, "first request"))
        await asyncio.wait_for(first_started.wait(), timeout=1)

        second_task = asyncio.create_task(bot_module.run_research(self.message, "second request"))
        await asyncio.sleep(0)
        self.assertFalse(second_task.done(), "same-user request should wait for the active turn")

        release_first.set()
        await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=1)

        research.answer.assert_awaited()
        self.assertEqual([user_text for _, user_text in calls], ["first request", "second request"])
        self.assertEqual(
            calls[1][0],
            [
                {"role": "user", "content": "first request"},
                {"role": "assistant", "content": "answer: first request"},
            ],
        )
        self.assertEqual(
            list(bot_module.histories[self.user_id]),
            [
                {"role": "user", "content": "first request"},
                {"role": "assistant", "content": "answer: first request"},
                {"role": "user", "content": "second request"},
                {"role": "assistant", "content": "answer: second request"},
            ],
        )

    async def test_reset_waits_for_active_same_user_research(self):
        bot_module.histories[self.user_id].append(
            {"role": "assistant", "content": "previous context"}
        )
        research_started = asyncio.Event()
        release_research = asyncio.Event()

        async def answer(history, user_text):
            research_started.set()
            await release_research.wait()
            return "completed answer"

        self.use_research(answer=AsyncMock(side_effect=answer))
        research_task = asyncio.create_task(bot_module.run_research(self.message, "new question"))
        await asyncio.wait_for(research_started.wait(), timeout=1)

        reset_message = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id),
            answer=AsyncMock(),
        )
        reset_task = asyncio.create_task(bot_module.reset(reset_message))
        await asyncio.sleep(0)
        self.assertFalse(reset_task.done(), "reset should wait until the active turn commits")

        release_research.set()
        await asyncio.wait_for(asyncio.gather(research_task, reset_task), timeout=1)
        self.assertEqual(list(bot_module.histories[self.user_id]), [])
        reset_message.answer.assert_awaited_once_with(
            "Conversation context cleared. Ask a new research question."
        )

    async def test_chart_deadline_is_reported_without_saving_a_successful_turn(self):
        research = self.use_research(
            analyze_chart_image=AsyncMock(
                side_effect=ResearchTurnDeadlineExceeded(0.02)
            )
        )

        with patch.object(
            bot_module,
            "prepare_chart_image",
            return_value=(b"normalized-chart", "image/png"),
        ):
            await bot_module.run_research(
                self.message, "Map BTC 1H DNS", b"chart-bytes", "image/png"
            )

        research.analyze_chart_image.assert_awaited_once()
        self.status.edit_text.assert_awaited_once()
        self.assertIn("0.02-second time budget expired", self.status.edit_text.await_args.args[0])
        self.assertEqual(list(bot_module.histories[self.user_id]), [])

    async def test_text_deadline_is_reported_without_saving_a_successful_turn(self):
        research = self.use_research(
            answer=AsyncMock(side_effect=ResearchTurnDeadlineExceeded(0.02))
        )

        await bot_module.run_research(self.message, "Analyse BTC")

        research.answer.assert_awaited_once()
        self.status.edit_text.assert_awaited_once()
        failure_message = self.status.edit_text.await_args.args[0]
        self.assertIn("research turn", failure_message)
        self.assertNotIn("chart analysis", failure_message)
        self.assertIn("0.02-second time budget expired", failure_message)
        self.assertEqual(list(bot_module.histories[self.user_id]), [])

    async def test_chart_analysis_is_saved_even_when_overlay_delivery_fails(self):
        result = {
            "analysis": "Visible demand structure; verify the price scale.",
            "plot_bounds": {"x_start": 0.08, "x_end": 0.94, "y_top": 0.12, "y_bottom": 0.9},
            "zones": [{"zone_type": "demand", "draw_overlay": True}],
        }
        research = self.use_research(analyze_chart_image=AsyncMock(return_value=result))
        self.message.answer_photo.side_effect = RuntimeError("Telegram photo delivery failed")

        with patch.object(
            bot_module, "prepare_chart_image",
            return_value=(b"normalized-chart", "image/png"),
        ) as prepare, patch.object(
            bot_module, "_render_overlay_prepared", return_value=(b"overlay-jpeg", 1)
        ) as render:
            await bot_module.run_research(self.message, "Map BTC 1H DNS", b"chart-bytes", "image/png")

        prepare.assert_called_once_with(
            b"chart-bytes", max_output_bytes=bot_module.MAX_CHART_IMAGE_BYTES
        )
        research.analyze_chart_image.assert_awaited_once()
        self.assertEqual(research.analyze_chart_image.await_args.args[2], b"normalized-chart")
        self.assertEqual(research.analyze_chart_image.await_args.args[3], "image/png")
        render.assert_called_once()
        self.assertEqual(render.call_args.args[0], b"normalized-chart")
        self.status.edit_text.assert_awaited_once_with(
            "Visible demand structure; verify the price scale."
        )
        self.message.answer_photo.assert_awaited_once()
        self.assertEqual(bot_module.histories[self.user_id][-1]["content"], result["analysis"])

    async def test_chart_analysis_reports_overlay_render_failure_separately(self):
        result = {
            "analysis": "Visible demand structure; verify the price scale.",
            "plot_bounds": {"x_start": 0.08, "x_end": 0.94, "y_top": 0.12, "y_bottom": 0.9},
            "zones": [{"zone_type": "demand", "draw_overlay": True}],
        }
        research = self.use_research(analyze_chart_image=AsyncMock(return_value=result))

        with patch.object(
            bot_module, "prepare_chart_image",
            return_value=(b"normalized-chart", "image/png"),
        ), patch.object(
            bot_module, "_render_overlay_prepared",
            side_effect=RuntimeError("renderer failed"),
        ):
            await bot_module.run_research(
                self.message, "Map BTC 1H DNS", b"chart-bytes", "image/png"
            )

        answer = self.status.edit_text.await_args.args[0]
        self.assertIn("chart analysis completed", answer)
        self.assertIn("couldn't render the overlay", answer)
        self.assertNotIn("no zone location was clear enough", answer)
        self.message.answer_photo.assert_not_awaited()
        self.assertEqual(bot_module.histories[self.user_id][-1]["content"], answer)

    async def test_missing_drawable_zones_keep_the_uncertainty_message(self):
        result = {
            "analysis": "The chart is too unclear to locate a reliable zone.",
            "plot_bounds": None,
            "zones": [],
        }
        research = self.use_research(analyze_chart_image=AsyncMock(return_value=result))

        with patch.object(
            bot_module, "prepare_chart_image",
            return_value=(b"normalized-chart", "image/png"),
        ), patch.object(
            bot_module, "_render_overlay_prepared", return_value=(None, 0)
        ):
            await bot_module.run_research(
                self.message, "Map BTC 1H DNS", b"chart-bytes", "image/png"
            )

        answer = self.status.edit_text.await_args.args[0]
        self.assertIn("no zone location was clear enough", answer)
        self.assertNotIn("couldn't render the overlay", answer)
        self.message.answer_photo.assert_not_awaited()
        self.assertEqual(bot_module.histories[self.user_id][-1]["content"], answer)

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

    async def test_reset_for_unknown_user_does_not_allocate_history_or_keep_lock(self):
        unknown_user_id = self.user_id + 1
        reset_message = SimpleNamespace(
            from_user=SimpleNamespace(id=unknown_user_id),
            answer=AsyncMock(),
        )

        await bot_module.reset(reset_message)

        self.assertNotIn(unknown_user_id, bot_module.histories)
        self.assertEqual(len(bot_module.user_locks), 0)
        reset_message.answer.assert_awaited_once_with(
            "Conversation context cleared. Ask a new research question."
        )

    async def test_concurrent_start_requests_respect_beta_capacity(self):
        first = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id),
            answer=AsyncMock(),
        )
        second = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id + 1),
            answer=AsyncMock(),
        )

        with patch.object(bot_module, "BETA_USER_LIMIT", 1):
            await asyncio.gather(bot_module.start(first), bot_module.start(second))

        self.assertEqual(len(bot_module.beta_users), 1)
        replies = [first.answer.await_args.args[0], second.answer.await_args.args[0]]
        self.assertEqual(replies.count(bot_module.WELCOME), 1)
        self.assertEqual(
            replies.count("The 10-person beta is currently full. Please try again later."),
            1,
        )

    async def test_research_and_start_share_the_same_beta_capacity(self):
        research = self.use_research(answer=AsyncMock(return_value="should not run"))
        admitted = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id),
            answer=AsyncMock(),
        )
        rejected = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id + 1),
            answer=AsyncMock(),
            answer_photo=AsyncMock(),
        )

        with patch.object(bot_module, "BETA_USER_LIMIT", 1):
            await bot_module.start(admitted)
            await bot_module.run_research(rejected, "Analyse BTC")

        research.answer.assert_not_awaited()
        rejected.answer.assert_awaited_once_with(
            "The 10-person beta is currently full. Please try again later."
        )
        self.assertNotIn(self.user_id + 1, bot_module.usage)

    async def test_concurrent_same_user_requests_cannot_exceed_daily_limit(self):
        research = self.use_research(answer=AsyncMock(return_value="Research completed."))

        with patch.object(bot_module, "DAILY_QUERY_LIMIT", 1):
            await asyncio.wait_for(
                asyncio.gather(
                    bot_module.run_research(self.message, "first request"),
                    bot_module.run_research(self.message, "second request"),
                ),
                timeout=2,
            )

        research.answer.assert_awaited_once()
        self.assertEqual(bot_module.usage[self.user_id], 1)
        self.assertTrue(
            any(
                call.args == ("You've reached today's beta research limit. It resets at midnight IST.",)
                for call in self.message.answer.await_args_list
            )
        )

    async def test_daily_usage_resets_for_a_new_ist_date(self):
        user_id = self.user_id + 2
        with patch.object(bot_module, "DAILY_QUERY_LIMIT", 2):
            self.assertTrue(bot_module._consume_daily_query(user_id, "2026-11-06"))
            self.assertTrue(bot_module._consume_daily_query(user_id, "2026-11-06"))
            self.assertFalse(bot_module._consume_daily_query(user_id, "2026-11-06"))
            self.assertTrue(bot_module._consume_daily_query(user_id, "2026-11-07"))

        self.assertEqual(bot_module.usage_day[user_id], "2026-11-07")
        self.assertEqual(bot_module.usage[user_id], 1)

    async def test_full_beta_rejects_chart_before_image_normalization(self):
        bot_module.beta_users.add(self.user_id + 1)
        with patch.object(bot_module, "BETA_USER_LIMIT", 1), patch.object(
            bot_module, "prepare_chart_image"
        ) as prepare:
            await bot_module.run_research(
                self.message, "Analyse this chart", b"untrusted-image", "image/png"
            )

        prepare.assert_not_called()
        self.message.answer.assert_awaited_once_with(
            "The 10-person beta is currently full. Please try again later."
        )
        self.assertNotIn(self.user_id, bot_module.beta_users)
        self.assertNotIn(self.user_id, bot_module.usage)

    async def test_daily_limit_rejects_chart_before_image_normalization(self):
        today = datetime.now(bot_module.IST).date().isoformat()
        bot_module.usage_day[self.user_id] = today
        bot_module.usage[self.user_id] = 1

        with patch.object(bot_module, "DAILY_QUERY_LIMIT", 1), patch.object(
            bot_module, "prepare_chart_image"
        ) as prepare:
            await bot_module.run_research(
                self.message, "Analyse this chart", b"untrusted-image", "image/png"
            )

        prepare.assert_not_called()
        self.message.answer.assert_awaited_once_with(
            "You've reached today's beta research limit. It resets at midnight IST."
        )
        self.assertEqual(bot_module.usage[self.user_id], 1)

    async def test_full_beta_photo_is_rejected_before_telegram_download(self):
        bot_module.beta_users.add(self.user_id + 1)
        photo_message = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id),
            photo=[SimpleNamespace(file_id="telegram-file-id", file_size=1024)],
            caption="BTC 1H",
            answer=AsyncMock(),
        )

        with patch.object(bot_module, "BETA_USER_LIMIT", 1), patch.object(
            bot_module.bot, "get_file", new=AsyncMock()
        ) as get_file:
            await bot_module.handle_chart_photo(photo_message)

        get_file.assert_not_awaited()
        photo_message.answer.assert_awaited_once_with(
            "The 10-person beta is currently full. Please try again later."
        )

    async def test_daily_limit_document_is_rejected_before_telegram_download(self):
        today = datetime.now(bot_module.IST).date().isoformat()
        bot_module.usage_day[self.user_id] = today
        bot_module.usage[self.user_id] = 1
        document_message = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id),
            document=SimpleNamespace(
                mime_type="image/png",
                file_size=1024,
                file_id="telegram-file-id",
            ),
            caption="BTC 1H",
            answer=AsyncMock(),
        )

        with patch.object(bot_module, "DAILY_QUERY_LIMIT", 1), patch.object(
            bot_module.bot, "get_file", new=AsyncMock()
        ) as get_file:
            await bot_module.handle_chart_document(document_message)

        get_file.assert_not_awaited()
        document_message.answer.assert_awaited_once_with(
            "You've reached today's beta research limit. It resets at midnight IST."
        )


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
