import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from stockmind_telegram_mvp.app.ai import ResearchAI, _parse_tool_arguments


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
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            _parse_tool_arguments({"symbol": "RELIANCE"})

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


if __name__ == "__main__":
    unittest.main()
