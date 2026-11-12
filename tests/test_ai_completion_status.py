import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from stockmind_telegram_mvp.app.ai import ResearchAI, ResearchResponseIncomplete


class CompletionStatusTests(unittest.IsolatedAsyncioTestCase):
    def make_research(self, response, market_data=None):
        research = object.__new__(ResearchAI)
        research.model = "gpt-4o-mini"
        research.vision_model = "gpt-4o-mini"
        research.turn_timeout_seconds = 2.0
        create = AsyncMock(return_value=response)
        research.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        research.market_data = market_data
        research.crypto_data = None
        return research, create

    async def test_text_response_with_no_choices_is_rejected(self):
        research, create = self.make_research(SimpleNamespace(choices=[]))

        with self.assertRaisesRegex(ResearchResponseIncomplete, "no completion choice"):
            await research.answer([], "Analyse BTC")

        create.assert_awaited_once()

    async def test_truncated_tool_call_is_rejected_before_market_data_dispatch(self):
        call = SimpleNamespace(
            id="call-truncated",
            function=SimpleNamespace(
                name="get_quote",
                arguments='{"symbol":"RELIANCE"}',
            ),
        )
        message = SimpleNamespace(
            content="partial tool response",
            tool_calls=[call],
            model_dump=lambda **kwargs: {"role": "assistant", "tool_calls": []},
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="length")]
        )
        quote = AsyncMock(return_value={"data": {"symbol": "RELIANCE"}})
        research, create = self.make_research(
            response, market_data=SimpleNamespace(quote=quote)
        )

        with self.assertRaisesRegex(ResearchResponseIncomplete, "truncated"):
            await research.answer([], "Get Reliance quote")

        create.assert_awaited_once()
        quote.assert_not_awaited()

    async def test_content_filtered_text_is_not_returned_as_a_complete_answer(self):
        message = SimpleNamespace(
            content="partial content that must not be presented",
            tool_calls=[],
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="content_filter")]
        )
        research, _ = self.make_research(response)

        with self.assertRaisesRegex(ResearchResponseIncomplete, "content filter"):
            await research.answer([], "Analyse this request")

    async def test_chart_response_truncated_by_output_limit_is_rejected(self):
        message = SimpleNamespace(content='{"analysis":"partial JSON')
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="length")]
        )
        research, create = self.make_research(response)

        with self.assertRaisesRegex(ResearchResponseIncomplete, "truncated"):
            await research.analyze_chart_image(
                [], "Map BTC demand and supply", b"chart-bytes", "image/png"
            )

        create.assert_awaited_once()

    async def test_chart_response_with_no_choices_is_rejected(self):
        research, create = self.make_research(SimpleNamespace(choices=[]))

        with self.assertRaisesRegex(ResearchResponseIncomplete, "no completion choice"):
            await research.analyze_chart_image(
                [], "Map BTC demand and supply", b"chart-bytes", "image/png"
            )

        create.assert_awaited_once()

    async def test_tool_calls_finish_reason_remains_a_valid_intermediate_response(self):
        call = SimpleNamespace(
            id="call-valid",
            function=SimpleNamespace(
                name="get_quote",
                arguments='{"symbol":"RELIANCE"}',
            ),
        )
        first_message = SimpleNamespace(
            content=None,
            tool_calls=[call],
            model_dump=lambda **kwargs: {"role": "assistant", "tool_calls": []},
        )
        final_message = SimpleNamespace(
            content="The quote lookup completed.",
            tool_calls=[],
        )
        responses = [
            SimpleNamespace(
                choices=[SimpleNamespace(message=first_message, finish_reason="tool_calls")]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=final_message, finish_reason="stop")]
            ),
        ]
        create = AsyncMock(side_effect=responses)
        research, _ = self.make_research(
            responses[0],
            market_data=SimpleNamespace(
                quote=AsyncMock(return_value={"data": {"symbol": "RELIANCE"}})
            ),
        )
        research.client.chat.completions.create = create
        quote = research.market_data.quote

        answer = await research.answer([], "Get Reliance quote")

        self.assertEqual(answer, "The quote lookup completed.")
        quote.assert_awaited_once_with("RELIANCE")
        self.assertEqual(create.await_count, 2)


if __name__ == "__main__":
    unittest.main()
