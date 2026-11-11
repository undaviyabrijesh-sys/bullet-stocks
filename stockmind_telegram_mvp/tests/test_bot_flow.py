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
from stockmind_telegram_mvp.app.telegram_delivery import split_telegram_text


_TEST_ENV = {
    "TELEGRAM_BOT_TOKEN": "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi",
    "OPENAI_API_KEY": "stockmind-test-key",
    "TAPETIDE_TOKEN": "stockmind-test-tapetide-token",
    "OPENAI_MODEL": "gpt-6-luna-test",
    "OPENAI_VISION_MODEL": "gpt-6-luna-test",
    "STOCKMIND_PREFERENCES_DB": ":memory:",
}
with environment_patch.dict(os.environ, _TEST_ENV):
    bot_module = importlib.import_module("stockmind_telegram_mvp.app.bot")


class OverlayOutcomeMessageTests(unittest.TestCase):
    def test_drawn_overlay_needs_no_unavailable_note(self):
        self.assertEqual(
            bot_module._overlay_unavailable_note(
                {"plot_bounds": {"x_start": 0.1}, "zones": []}, 1
            ),
            "",
        )

    def test_missing_plot_boundary_is_distinguished_from_missing_zones(self):
        note = bot_module._overlay_unavailable_note(
            {"plot_bounds": None, "zones": [{"zone_type": "demand"}]}, 0
        )
        self.assertIn("main price-plot boundary", note)
        self.assertIn("withheld all overlays", note)

    def test_empty_zone_list_does_not_claim_that_no_zone_exists(self):
        note = bot_module._overlay_unavailable_note(
            {
                "plot_bounds": {
                    "x_start": 0.1, "x_end": 0.9,
                    "y_top": 0.1, "y_bottom": 0.9,
                },
                "zones": [],
            },
            0,
        )
        self.assertIn("No distinct supply/demand zone was returned", note)
        self.assertIn("does not prove that no zone exists", note)

    def test_model_declined_geometry_is_distinguished_from_renderer_rejection(self):
        note = bot_module._overlay_unavailable_note(
            {
                "plot_bounds": {
                    "x_start": 0.1, "x_end": 0.9,
                    "y_top": 0.1, "y_bottom": 0.9,
                },
                "zones": [{"zone_type": "demand", "draw_overlay": False}],
            },
            0,
        )
        self.assertIn("candidate areas", note)
        self.assertIn("none had geometry approved", note)

    def test_final_pixel_or_plot_rejection_is_reported_separately(self):
        note = bot_module._overlay_unavailable_note(
            {
                "plot_bounds": {
                    "x_start": 0.1, "x_end": 0.9,
                    "y_top": 0.1, "y_bottom": 0.9,
                },
                "zones": [{"zone_type": "demand", "draw_overlay": True}],
            },
            0,
        )
        self.assertIn("final pixel-size and plot-boundary checks", note)
        self.assertIn("sharper, uncropped screenshot", note)


class ResearchToTelegramFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        bot_module.histories.clear()
        bot_module.usage.clear()
        bot_module.usage_day.clear()
        bot_module.beta_users.clear()
        bot_module.user_locks.clear()
        bot_module.user_languages.clear()
        if bot_module.language_preferences is not None:
            bot_module.language_preferences.clear()
        if bot_module.admission_store is not None:
            bot_module.admission_store.clear()
        if bot_module.conversation_history_store is not None:
            bot_module.conversation_history_store.clear()
        bot_module.history_storage_unavailable_users.clear()

        self.user_id = 987654321
        self.status = SimpleNamespace(edit_text=AsyncMock())
        self.message = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id),
            answer=AsyncMock(return_value=self.status),
            answer_photo=AsyncMock(),
        )

    async def test_language_command_sets_explicit_gujarati_preference(self):
        language_message = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id, language_code="en"),
            text="/language gu",
            answer=AsyncMock(),
        )

        await bot_module.set_language(language_message)

        self.assertEqual(bot_module.user_languages[self.user_id], "gu")
        reply = language_message.answer.await_args.args[0]
        self.assertIn("Interface ભાષા", reply)
        self.assertIn("ગુજરાતી", reply)

    async def test_saved_language_is_loaded_after_process_cache_is_cleared(self):
        language_message = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id, language_code="en"),
            text="/language gu",
            answer=AsyncMock(),
        )
        await bot_module.set_language(language_message)
        bot_module.user_languages.clear()

        resolved = bot_module._language_for(
            SimpleNamespace(from_user=SimpleNamespace(id=self.user_id, language_code="en"))
        )

        self.assertEqual(resolved, "gu")

    async def test_language_command_warns_when_persistence_is_unavailable(self):
        class BrokenPreferenceStore:
            def get(self, user_id):
                return None

            def set(self, user_id, language):
                raise OSError("read-only test database")

        language_message = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id, language_code="en"),
            text="/language hi",
            answer=AsyncMock(),
        )
        with patch.object(bot_module, "language_preferences", BrokenPreferenceStore()):
            await bot_module.set_language(language_message)

        self.assertEqual(bot_module.user_languages[self.user_id], "hi")
        reply = language_message.answer.await_args.args[0]
        self.assertIn("इस सत्र", reply)
        self.assertIn("सहेजा नहीं जा सका", reply)

    async def test_language_command_help_uses_telegram_locale(self):
        language_message = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id, language_code="hi-IN"),
            text="/language",
            answer=AsyncMock(),
        )

        await bot_module.set_language(language_message)

        reply = language_message.answer.await_args.args[0]
        self.assertIn("इंटरफ़ेस भाषा", reply)
        self.assertIn("/language gu", reply)

    async def test_start_uses_explicit_language_before_telegram_locale(self):
        start_message = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id, language_code="hi-IN"),
            answer=AsyncMock(),
        )
        bot_module.user_languages[self.user_id] = "gu"

        await bot_module.start(start_message)

        reply = start_message.answer.await_args.args[0]
        self.assertIn("બીટામાં આપનું સ્વાગત છે", reply)
        self.assertNotIn("बीटा में आपका स्वागत है", reply)

    async def test_admission_error_uses_requested_ui_language(self):
        bot_module.beta_users.add(self.user_id + 1)
        with patch.object(bot_module, "BETA_USER_LIMIT", 1):
            error = bot_module._research_admission_error(self.user_id, "hi")
        self.assertIn("बीटा सीमा", error)

    async def test_research_fails_closed_when_shared_lock_store_is_unavailable(self):
        research = self.use_research(answer=AsyncMock(return_value="must not run"))

        with patch.object(bot_module, "user_locks", None):
            await bot_module.run_research(self.message, "Analyse BTC")

        research.answer.assert_not_awaited()
        self.message.answer.assert_awaited_once()
        self.assertIn("storage", self.message.answer.await_args.args[0].lower())

    async def test_reset_fails_closed_when_shared_lock_store_is_unavailable(self):
        reset_message = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id),
            answer=AsyncMock(),
        )

        with patch.object(bot_module, "user_locks", None):
            await bot_module.reset(reset_message)

        reset_message.answer.assert_awaited_once()
        self.assertIn("storage", reset_message.answer.await_args.args[0].lower())

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

    def test_completed_turn_bounds_persisted_and_process_local_context(self):
        answer = "BTC setup with confirmation and invalidation. " * 1200

        bot_module._record_conversation_turn(self.user_id, "Analyse BTC", answer)

        local_history = list(bot_module.histories[self.user_id])
        persisted_history = bot_module.conversation_history_store.get(
            self.user_id,
            bot_module.MAX_HISTORY_MESSAGES,
            bot_module.MAX_HISTORY_BYTES,
        )
        local_bytes = sum(
            len(item["content"].encode("utf-8")) for item in local_history
        )
        persisted_bytes = sum(
            len(item["content"].encode("utf-8")) for item in persisted_history
        )

        self.assertLessEqual(local_bytes, bot_module.MAX_HISTORY_BYTES)
        self.assertLessEqual(persisted_bytes, bot_module.MAX_HISTORY_BYTES)
        self.assertEqual(local_history, persisted_history)
        self.assertEqual(local_history[-1]["role"], "assistant")
        self.assertIn("[message shortened", local_history[-1]["content"])

    async def test_conversation_history_survives_process_cache_reset(self):
        first_research = SimpleNamespace(
            answer=AsyncMock(return_value="First answer with the BTC context.")
        )
        with patch.object(bot_module, "research", first_research):
            await bot_module.run_research(self.message, "Analyse BTC")

        bot_module.histories.clear()
        second_research = SimpleNamespace(
            answer=AsyncMock(return_value="Second answer.")
        )
        with patch.object(bot_module, "research", second_research):
            await bot_module.run_research(self.message, "What changed?")

        self.assertEqual(
            second_research.answer.await_args.args[0],
            [
                {"role": "user", "content": "Analyse BTC"},
                {"role": "assistant", "content": "First answer with the BTC context."},
            ],
        )

    async def test_reset_deletes_persisted_conversation_history(self):
        research = SimpleNamespace(answer=AsyncMock(return_value="Saved answer."))
        with patch.object(bot_module, "research", research):
            await bot_module.run_research(self.message, "Remember this context")

        reset_message = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id),
            answer=AsyncMock(),
        )
        await bot_module.reset(reset_message)
        bot_module.histories.clear()

        self.assertEqual(bot_module._load_conversation_history(self.user_id), [])
        self.assertIn("cleared", reset_message.answer.await_args.args[0].lower())

    async def test_reset_warns_when_persisted_history_cannot_be_deleted(self):
        bot_module.histories[self.user_id].append(
            {"role": "assistant", "content": "local context"}
        )
        reset_message = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id),
            answer=AsyncMock(),
        )
        with patch.object(bot_module, "conversation_history_store", None):
            await bot_module.reset(reset_message)

        self.assertEqual(list(bot_module.histories[self.user_id]), [])
        self.assertIn("could not be deleted", reset_message.answer.await_args.args[0])

    async def test_long_research_answer_is_not_truncated_before_telegram_delivery(self):
        answer = ("BTC setup: confirmation, invalidation and risk notes.\n\n" * 180)
        chunks = split_telegram_text(answer)
        research = self.use_research(answer=AsyncMock(return_value=answer))

        await bot_module.run_research(self.message, "Analyse BTC")

        research.answer.assert_awaited_once()
        self.status.edit_text.assert_awaited_once_with(chunks[0])
        # message.answer is first used for the progress status; subsequent
        # calls are the additional answer chunks.
        delivered_chunks = [self.status.edit_text.await_args.args[0]] + [
            call.args[0] for call in self.message.answer.await_args_list[1:]
        ]
        self.assertEqual(delivered_chunks, chunks)
        self.assertEqual("".join(delivered_chunks), answer)
        self.assertEqual(bot_module.histories[self.user_id][-1]["content"], answer)

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
        self.assertIn("main price-plot boundary", answer)
        self.assertIn("withheld all overlays", answer)
        self.assertNotIn("No distinct supply/demand zone was returned", answer)
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

    async def test_beta_membership_and_daily_quota_survive_process_cache_reset(self):
        first = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id),
            answer=AsyncMock(),
        )
        await bot_module.start(first)
        self.assertIn(bot_module.WELCOME, first.answer.await_args.args)

        today = datetime.now(bot_module.IST).date().isoformat()
        with patch.object(bot_module, "DAILY_QUERY_LIMIT", 1):
            self.assertTrue(bot_module._consume_daily_query(self.user_id, today))
            # Simulate a fresh process by discarding all process-local mirrors.
            bot_module.beta_users.clear()
            bot_module.usage.clear()
            bot_module.usage_day.clear()
            self.assertEqual(
                bot_module._research_admission_error(self.user_id, "en"),
                "You've reached today's beta research limit. It resets at midnight IST.",
            )

        second = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id + 1),
            answer=AsyncMock(),
        )
        with patch.object(bot_module, "BETA_USER_LIMIT", 1):
            await bot_module.start(second)
        self.assertIn("beta is currently full", second.answer.await_args.args[0])

    async def test_quota_storage_failure_fails_closed_before_image_processing(self):
        with patch.object(bot_module, "admission_store", None), patch.object(
            bot_module, "prepare_chart_image"
        ) as prepare:
            await bot_module.run_research(
                self.message, "Analyse this chart", b"untrusted-image", "image/png"
            )
        prepare.assert_not_called()
        self.message.answer.assert_awaited_once()
        self.assertIn("quota storage is unavailable", self.message.answer.await_args.args[0])

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

    async def test_photo_preflight_uses_telegram_language_for_beta_rejection(self):
        bot_module.beta_users.add(self.user_id + 1)
        photo_message = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id, language_code="hi-IN"),
            photo=[SimpleNamespace(file_id="telegram-file-id", file_size=1024)],
            caption="BTC 1H",
            answer=AsyncMock(),
        )

        with patch.object(bot_module, "BETA_USER_LIMIT", 1), patch.object(
            bot_module.bot, "get_file", new=AsyncMock()
        ) as get_file:
            await bot_module.handle_chart_photo(photo_message)

        get_file.assert_not_awaited()
        reply = photo_message.answer.await_args.args[0]
        self.assertIn("बीटा सीमा", reply)
        self.assertNotIn("The 10-person beta", reply)

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

    async def test_photo_download_overflow_uses_telegram_interface_language(self):
        photo_message = SimpleNamespace(
            from_user=SimpleNamespace(id=self.user_id, language_code="hi-IN"),
            photo=[SimpleNamespace(file_id="telegram-file-id", file_size=None)],
            caption="BTC 1H",
            answer=AsyncMock(),
        )

        async def overflow_download(file_path, destination):
            destination.write(b"12345")

        with patch.object(bot_module, "MAX_CHART_IMAGE_BYTES", 4), patch.object(
            bot_module.bot,
            "get_file",
            new=AsyncMock(return_value=SimpleNamespace(file_path="photos/test.jpg")),
        ) as get_file, patch.object(
            bot_module.bot,
            "download_file",
            new=AsyncMock(side_effect=overflow_download),
        ) as download_file:
            await bot_module.handle_chart_photo(photo_message)

        get_file.assert_awaited_once_with("telegram-file-id")
        download_file.assert_awaited_once()
        reply = photo_message.answer.await_args.args[0]
        self.assertIn("यह image बहुत बड़ी है", reply)
        self.assertNotIn("That screenshot is too large", reply)


class BotModuleResourceCleanup(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        async def close_async_clients():
            await bot_module.market.close()
            await bot_module.crypto_market.close()
            await bot_module.bot.session.close()

        asyncio.run(close_async_clients())
        bot_module.research.client.close()
        if bot_module.language_preferences is not None:
            bot_module.language_preferences.close()


if __name__ == "__main__":
    unittest.main()
