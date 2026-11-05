import unittest

from stockmind_telegram_mvp.app.ai import _parse_tool_arguments


class ToolArgumentParsingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
