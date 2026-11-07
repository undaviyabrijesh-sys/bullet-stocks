import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from stockmind_telegram_mvp.app.ai import (
    MAX_TOOL_CALLS_PER_TURN,
    ResearchAI,
    _parse_tool_arguments,
    _serialize_tool_result,
)


class ToolArgumentParsingTests(unittest.IsolatedAsyncioTestCase):
    def test_parses_json_object(self):
        self.assertEqual(_parse_tool_arguments('{"symbol":"RELIANCE"}'), {"symbol": "RELIANCE"})

    def test_empty_arguments_default_to_empty_object(self):
        self.assertEqual(_parse_tool_arguments(None), {})
        self.assertEqual(_parse_tool_arguments(""), {})

    def test_rejects_malformed_json(self):
        for raw in ('{"symbol":', "{not-json", "["):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "valid JSON"):
                    _parse_tool_arguments(raw)

    def test_rejects_valid_json_that_is_not_an_object(self):
        for raw in ("[]", '"RELIANCE"', "42", "true", "null"):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "JSON object"):
                    _parse_tool_arguments(raw)

    def test_rejects_non_string_argument_payload(self):
        for raw in ({"symbol": "RELIANCE"}, 0, False, []):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "JSON string"):
                    _parse_tool_arguments(raw)

    async def test_malformed_tool_call_is_returned_to_model_and_turn_continues(self):
        malformed_call = SimpleNamespace(
            id="call-malformed",
            function=SimpleNamespace(name="get_quote", arguments='{"symbol":'),
        )
        first_message = SimpleNamespace(
            tool_calls=[malformed_call],
            model_dump=lambda **kwargs: {"role": "assistant", "tool_calls": []},
            content=None,
        )
        second_message = SimpleNamespace(tool_calls=[], content="I recovered and can answer safely.")
        responses = [
            SimpleNamespace(choices=[SimpleNamespace(message=first_message)]),
            SimpleNamespace(choices=[SimpleNamespace(message=second_message)]),
        ]
        create = AsyncMock(side_effect=responses)

        research = object.__new__(ResearchAI)
        research.model = "gpt-4o-mini"
        research.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        research.market_data = None
        research.crypto_data = None

        answer = await research.answer([], "Get Reliance quote")

        self.assertEqual(answer, "I recovered and can answer safely.")
        self.assertEqual(create.await_count, 2)
        tool_message = create.await_args_list[1].kwargs["messages"][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "call-malformed")
        self.assertEqual(json.loads(tool_message["content"])["type"], "InvalidToolArguments")


class ToolResultSerializationTests(unittest.TestCase):
    def test_time_series_sampling_keeps_provider_order_tail_and_marks_partial_data(self):
        rows = [{"open_time": index} for index in range(10)]
        with patch("stockmind_telegram_mvp.app.ai.MAX_TOOL_DATA_ITEMS", 3):
            serialized = _serialize_tool_result(
                {"source": "Binance", "data": rows},
                "get_crypto_klines",
                max_bytes=4096,
            )

        result = json.loads(serialized)
        self.assertEqual([row["open_time"] for row in result["data"]], [7, 8, 9])
        self.assertEqual(result["truncation"]["data_items_total"], 10)
        self.assertEqual(result["truncation"]["data_items_returned"], 3)
        self.assertEqual(result["truncation"]["selection"], "last_in_provider_order")
        self.assertIn("Partial result", result["truncation"]["warning"])

    def test_stock_search_sampling_keeps_first_matches(self):
        rows = [{"symbol": f"STOCK{index}"} for index in range(5)]
        with patch("stockmind_telegram_mvp.app.ai.MAX_TOOL_DATA_ITEMS", 2):
            serialized = _serialize_tool_result(
                {"data": rows}, "search_stocks", max_bytes=4096
            )

        result = json.loads(serialized)
        self.assertEqual(
            [row["symbol"] for row in result["data"]], ["STOCK0", "STOCK1"]
        )
        self.assertEqual(
            result["truncation"]["selection"], "first_in_provider_order"
        )

    def test_oversized_non_list_payload_is_withheld_not_partially_forwarded(self):
        sensitive_value = "x" * 2048
        serialized = _serialize_tool_result(
            {"data": {"statement": sensitive_value}},
            "get_financials",
            max_bytes=512,
        )

        result = json.loads(serialized)
        self.assertEqual(result["type"], "ToolResultTooLarge")
        self.assertNotIn(sensitive_value, serialized)

    def test_serialized_market_data_never_exceeds_the_requested_byte_cap(self):
        rows = [{"value": "x" * 160} for _ in range(50)]
        serialized = _serialize_tool_result(
            {"data": rows}, "get_crypto_klines", max_bytes=512
        )

        self.assertLessEqual(len(serialized.encode("utf-8")), 512)
        self.assertIn("truncation", json.loads(serialized))


class ToolBudgetExecutionTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def make_call(index, name="get_quote", arguments='{"symbol":"RELIANCE"}'):
        return SimpleNamespace(
            id=f"call-{index}",
            function=SimpleNamespace(
                name=name,
                arguments=arguments,
            ),
        )

    @staticmethod
    def make_response(message):
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    @staticmethod
    def assistant_dump(calls):
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in calls
            ],
        }

    def make_research(self, create, market_data):
        research = object.__new__(ResearchAI)
        research.model = "gpt-4o-mini"
        research.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        research.market_data = market_data
        research.crypto_data = None
        return research

    async def test_identical_tool_requests_in_one_batch_execute_provider_once(self):
        calls = [
            self.make_call(
                1,
                name="search_stocks",
                arguments='{"query":"Reliance","limit":10,"sector":"Energy"}',
            ),
            self.make_call(
                2,
                name="search_stocks",
                arguments='{"sector":"Energy","limit":10,"query":"Reliance"}',
            ),
        ]
        first_message = SimpleNamespace(
            tool_calls=calls,
            content=None,
            model_dump=lambda **kwargs: self.assistant_dump(calls),
        )
        final_message = SimpleNamespace(
            tool_calls=[], content="I reused the stock-search result."
        )
        create = AsyncMock(
            side_effect=[
                self.make_response(first_message),
                self.make_response(final_message),
            ]
        )
        search = AsyncMock(return_value={"data": [{"symbol": "RELIANCE"}]})
        research = self.make_research(create, SimpleNamespace(search_stocks=search))

        answer = await research.answer([], "Find Reliance")

        self.assertEqual(answer, "I reused the stock-search result.")
        search.assert_awaited_once_with(
            "Reliance", limit=10, sector="Energy", industry=None
        )
        tool_messages = [
            item for item in create.await_args_list[1].kwargs["messages"]
            if item.get("role") == "tool"
        ]
        self.assertEqual(len(tool_messages), 2)
        self.assertEqual(
            json.loads(tool_messages[0]["content"])["data"][0]["symbol"],
            "RELIANCE",
        )
        duplicate = json.loads(tool_messages[1]["content"])
        self.assertTrue(duplicate["deduplicated"])
        self.assertEqual(duplicate["reused_tool_call_id"], "call-1")
        self.assertEqual(duplicate["scope"], "current_research_turn")

    async def test_identical_tool_request_is_deduplicated_across_model_rounds(self):
        first_call = self.make_call(1)
        second_call = self.make_call(
            2, arguments='{ "symbol" : "RELIANCE" }'
        )
        first_message = SimpleNamespace(
            tool_calls=[first_call],
            content=None,
            model_dump=lambda **kwargs: self.assistant_dump([first_call]),
        )
        second_message = SimpleNamespace(
            tool_calls=[second_call],
            content=None,
            model_dump=lambda **kwargs: self.assistant_dump([second_call]),
        )
        final_message = SimpleNamespace(
            tool_calls=[], content="The repeated quote request was reused."
        )
        create = AsyncMock(
            side_effect=[
                self.make_response(first_message),
                self.make_response(second_message),
                self.make_response(final_message),
            ]
        )
        quote = AsyncMock(return_value={"data": {"symbol": "RELIANCE", "price": "100"}})
        research = self.make_research(create, SimpleNamespace(quote=quote))

        answer = await research.answer([], "Get Reliance quote")

        self.assertEqual(answer, "The repeated quote request was reused.")
        quote.assert_awaited_once_with("RELIANCE")
        final_messages = create.await_args_list[2].kwargs["messages"]
        tool_messages = [item for item in final_messages if item.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 2)
        duplicate = json.loads(tool_messages[1]["content"])
        self.assertTrue(duplicate["deduplicated"])
        self.assertEqual(duplicate["reused_tool_call_id"], "call-1")

    async def test_different_arguments_are_not_deduplicated(self):
        calls = [
            self.make_call(1, arguments='{"symbol":"RELIANCE"}'),
            self.make_call(2, arguments='{"symbol":"TCS"}'),
        ]
        first_message = SimpleNamespace(
            tool_calls=calls,
            content=None,
            model_dump=lambda **kwargs: self.assistant_dump(calls),
        )
        final_message = SimpleNamespace(tool_calls=[], content="Both quotes were fetched.")
        create = AsyncMock(
            side_effect=[
                self.make_response(first_message),
                self.make_response(final_message),
            ]
        )
        quote = AsyncMock(
            side_effect=[
                {"data": {"symbol": "RELIANCE"}},
                {"data": {"symbol": "TCS"}},
            ]
        )
        research = self.make_research(create, SimpleNamespace(quote=quote))

        answer = await research.answer([], "Compare Reliance and TCS")

        self.assertEqual(answer, "Both quotes were fetched.")
        self.assertEqual(quote.await_count, 2)
        tool_messages = [
            item for item in create.await_args_list[1].kwargs["messages"]
            if item.get("role") == "tool"
        ]
        self.assertEqual(
            [json.loads(item["content"])["data"]["symbol"] for item in tool_messages],
            ["RELIANCE", "TCS"],
        )

    async def test_rejects_tool_batch_above_per_turn_call_limit_before_provider_calls(self):
        calls = [self.make_call(index) for index in range(MAX_TOOL_CALLS_PER_TURN + 1)]
        message = SimpleNamespace(
            tool_calls=calls,
            content=None,
            model_dump=lambda **kwargs: self.assistant_dump(calls),
        )
        create = AsyncMock(return_value=self.make_response(message))
        quote = AsyncMock(return_value={"data": []})
        research = self.make_research(create, SimpleNamespace(quote=quote))

        answer = await research.answer([], "Compare several stocks")

        self.assertIn(str(MAX_TOOL_CALLS_PER_TURN), answer)
        self.assertEqual(create.await_count, 1)
        quote.assert_not_awaited()

    async def test_turn_wide_byte_budget_skips_later_market_data_request(self):
        calls = [self.make_call(1), self.make_call(2)]
        first_message = SimpleNamespace(
            tool_calls=calls,
            content=None,
            model_dump=lambda **kwargs: self.assistant_dump(calls),
        )
        final_message = SimpleNamespace(
            tool_calls=[],
            content="I used only the market data available within the context budget.",
        )
        create = AsyncMock(
            side_effect=[
                self.make_response(first_message),
                self.make_response(final_message),
            ]
        )
        quote = AsyncMock(
            return_value={
                "data": [{"value": "x" * 160} for _ in range(20)]
            }
        )
        research = self.make_research(create, SimpleNamespace(quote=quote))

        with patch("stockmind_telegram_mvp.app.ai.MAX_TOOL_RESULT_BYTES", 300), patch(
            "stockmind_telegram_mvp.app.ai.MAX_TOOL_RESULT_BYTES_PER_TURN", 300
        ):
            answer = await research.answer([], "Get two quotes")

        self.assertIn("context budget", answer)
        self.assertEqual(quote.await_count, 1)
        tool_messages = [
            item for item in create.await_args_list[1].kwargs["messages"]
            if item.get("role") == "tool"
        ]
        self.assertEqual(len(tool_messages), 2)
        self.assertEqual(
            json.loads(tool_messages[1]["content"])["type"], "ToolBudgetExceeded"
        )


if __name__ == "__main__":
    unittest.main()
