import unittest

from stockmind_telegram_mvp.app.ai import CHART_RESPONSE_FORMAT, sanitize_chart_analysis


def zone(**overrides):
    value = {
        "zone_type": "demand",
        "label": "Demand base",
        "price_low": "100",
        "price_high": "105",
        "x_start": 0.2,
        "x_end": 0.92,
        "y_top": 0.55,
        "y_bottom": 0.68,
        "draw_overlay": True,
    }
    value.update(overrides)
    return value


class ChartResponseValidationTests(unittest.TestCase):
    def test_preserves_valid_zone_and_normalizes_analysis(self):
        result = sanitize_chart_analysis({
            "analysis": "  Visible demand base near the prior impulse.  ",
            "zones": [zone()],
        })

        self.assertEqual(result["analysis"], "Visible demand base near the prior impulse.")
        self.assertEqual(len(result["zones"]), 1)
        self.assertTrue(result["zones"][0]["draw_overlay"])
        self.assertEqual(result["zones"][0]["x_start"], 0.2)

    def test_disables_overlay_for_out_of_range_or_non_finite_coordinates(self):
        invalid_zones = [
            zone(x_start=-0.1),
            zone(x_end=1.01),
            zone(y_top=float("nan")),
            zone(y_bottom=float("inf")),
            zone(x_start=True),
            zone(x_start=0.8, x_end=0.2),
            zone(y_top=0.7, y_bottom=0.6),
            zone(y_top=0.1, y_bottom=0.7),
        ]

        result = sanitize_chart_analysis({"analysis": "Zones are uncertain.", "zones": invalid_zones})

        self.assertEqual(len(result["zones"]), len(invalid_zones))
        for item in result["zones"]:
            self.assertFalse(item["draw_overlay"])
            self.assertEqual(
                [item["x_start"], item["x_end"], item["y_top"], item["y_bottom"]],
                [0.0, 0.0, 0.0, 0.0],
            )

    def test_non_drawable_zone_has_zero_geometry(self):
        result = sanitize_chart_analysis({
            "analysis": "No precise box can be located.",
            "zones": [zone(draw_overlay=False, x_start=0.2, x_end=0.9)],
        })

        item = result["zones"][0]
        self.assertFalse(item["draw_overlay"])
        self.assertEqual(item["x_start"], 0.0)
        self.assertEqual(item["x_end"], 0.0)

    def test_ignores_unknown_zone_types_and_non_object_items(self):
        result = sanitize_chart_analysis({
            "analysis": "One supported zone.",
            "zones": [None, "supply", zone(zone_type="resistance"), zone()],
        })

        self.assertEqual(len(result["zones"]), 1)
        self.assertEqual(result["zones"][0]["zone_type"], "demand")

    def test_caps_zone_payload_to_eight_items(self):
        result = sanitize_chart_analysis({
            "analysis": "Many visible areas; only the first eight are retained.",
            "zones": [zone(label=f"Zone {index}") for index in range(10)],
        })

        self.assertEqual(len(result["zones"]), 8)

    def test_sanitizes_untrusted_label_and_price_fields(self):
        result = sanitize_chart_analysis({
            "analysis": "A readable map.",
            "zones": [zone(label="L" * 100, price_low={"unexpected": "object"}, price_high="H" * 40)],
        })

        item = result["zones"][0]
        self.assertEqual(len(item["label"]), 64)
        self.assertIsNone(item["price_low"])
        self.assertEqual(len(item["price_high"]), 32)

    def test_rejects_missing_or_wrongly_typed_response_fields(self):
        for payload in (
            None,
            {"analysis": 42, "zones": []},
            {"analysis": "Readable", "zones": None},
            {"analysis": "   ", "zones": []},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    sanitize_chart_analysis(payload)

    def test_structured_schema_bounds_coordinates_and_zone_count(self):
        schema = CHART_RESPONSE_FORMAT["json_schema"]["schema"]
        zone_schema = schema["properties"]["zones"]
        self.assertEqual(zone_schema["maxItems"], 8)
        properties = zone_schema["items"]["properties"]
        for coordinate in ("x_start", "x_end", "y_top", "y_bottom"):
            with self.subTest(coordinate=coordinate):
                self.assertEqual(properties[coordinate]["minimum"], 0)
                self.assertEqual(properties[coordinate]["maximum"], 1)


if __name__ == "__main__":
    unittest.main()
