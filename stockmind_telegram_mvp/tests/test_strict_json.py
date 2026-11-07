import unittest

from stockmind_telegram_mvp.app.strict_json import StrictJSONError, loads
from stockmind_telegram_mvp.app.ai import _parse_tool_arguments


class StrictJSONTests(unittest.TestCase):
    def test_accepts_nested_standard_json(self):
        self.assertEqual(
            loads('{"outer":{"items":[1,2,3],"label":"NaN"}}'),
            {"outer": {"items": [1, 2, 3], "label": "NaN"}},
        )

    def test_rejects_duplicate_top_level_keys(self):
        with self.assertRaisesRegex(StrictJSONError, "duplicate JSON object key 'fixture_id'"):
            loads('{"fixture_id":"first","fixture_id":"second"}')

    def test_rejects_duplicate_nested_keys(self):
        with self.assertRaisesRegex(StrictJSONError, "duplicate JSON object key 'model'"):
            loads('{"prediction":{"model":"first","model":"second"}}')

    def test_rejects_non_standard_numeric_constants(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token), self.assertRaisesRegex(
                StrictJSONError, "non-standard JSON constant"
            ):
                loads('{"score":' + token + '}')

    def test_rejects_numeric_exponents_that_overflow_to_infinity(self):
        for token in ("1e999", "-1e999", "1.0e999"):
            with self.subTest(token=token), self.assertRaisesRegex(
                StrictJSONError, "supported finite numeric range"
            ):
                loads('{"score":' + token + '}')

    def test_accepts_finite_exponent_numbers(self):
        self.assertEqual(
            loads('{"large":1e3,"small":-2.5e-3}'),
            {"large": 1000.0, "small": -0.0025},
        )

    def test_model_tool_arguments_reject_exponent_overflow_before_dispatch(self):
        for raw in (
            '{"limit":1e999}',
            '{"filters":{"threshold":-1e999}}',
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(
                ValueError, "valid JSON"
            ):
                _parse_tool_arguments(raw)

    def test_rejects_duplicate_keys_even_when_first_value_is_null(self):
        with self.assertRaisesRegex(StrictJSONError, "duplicate JSON object key 'prediction'"):
            loads('{"prediction":null,"prediction":{"zones":[]}}')


if __name__ == "__main__":
    unittest.main()
