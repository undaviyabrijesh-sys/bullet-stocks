import unittest

from stockmind_telegram_mvp.app.ai import (
    CHART_IMAGE_INSTRUCTIONS,
    CHART_IMAGE_PROMPT,
    CHART_PROMPT_VERSION,
    CHART_RESPONSE_FORMAT,
    sanitize_chart_analysis,
)

PLOT_BOUNDS = {
    "x_start": 0.08, "x_end": 0.94, "y_top": 0.12, "y_bottom": 0.9,
}


def payload(zones, **overrides):
    value = {
        "analysis": "Visible chart structure.",
        "plot_bounds": dict(PLOT_BOUNDS),
        "zones": zones,
    }
    value.update(overrides)
    return value


def zone(**overrides):
    value = {
        "zone_type": "demand",
        "origin": "ai_inferred",
        "confidence": "high",
        "label": "USER SUPPLY",
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
        result = sanitize_chart_analysis(payload(
            [zone()],
            analysis="  Visible demand base near the prior impulse.  ",
        ))

        self.assertEqual(result["analysis"], "Visible demand base near the prior impulse.")
        self.assertEqual(len(result["zones"]), 1)
        self.assertTrue(result["zones"][0]["draw_overlay"])
        self.assertEqual(result["zones"][0]["origin"], "ai_inferred")
        self.assertEqual(result["zones"][0]["label"], "AI DEMAND")
        self.assertEqual(result["zones"][0]["x_start"], 0.2)
        self.assertEqual(result["plot_bounds"], PLOT_BOUNDS)

    def test_low_confidence_candidate_is_kept_but_never_drawn(self):
        result = sanitize_chart_analysis(payload([zone(confidence="low")]))
        item = result["zones"][0]
        self.assertEqual(item["confidence"], "low")
        self.assertFalse(item["draw_overlay"])
        self.assertEqual(
            [item["x_start"], item["x_end"], item["y_top"], item["y_bottom"]],
            [0.0, 0.0, 0.0, 0.0],
        )

    def test_missing_or_invalid_confidence_fails_closed(self):
        missing = zone()
        missing.pop("confidence")
        for candidate in (missing, zone(confidence="certain"), zone(confidence=[])):
            with self.subTest(candidate=candidate):
                item = sanitize_chart_analysis(payload([candidate]))["zones"][0]
                self.assertEqual(item["confidence"], "low")
                self.assertFalse(item["draw_overlay"])

    def test_user_marked_origin_generates_a_user_label(self):
        item = sanitize_chart_analysis(payload([
            zone(origin="user_marked", label="AI SUPPLY")
        ]))["zones"][0]

        self.assertEqual(item["origin"], "user_marked")
        self.assertEqual(item["label"], "USER DEMAND")

    def test_missing_or_invalid_origin_defaults_to_ai_inferred(self):
        missing = zone()
        missing.pop("origin")
        for candidate in (missing, zone(origin="user"), zone(origin=True)):
            with self.subTest(candidate=candidate):
                item = sanitize_chart_analysis(payload([candidate]))["zones"][0]
                self.assertEqual(item["origin"], "ai_inferred")
                self.assertEqual(item["label"], "AI DEMAND")

    def test_medium_confidence_candidate_remains_drawable(self):
        item = sanitize_chart_analysis(payload([zone(confidence="medium")]))["zones"][0]
        self.assertEqual(item["confidence"], "medium")
        self.assertTrue(item["draw_overlay"])

    def test_disables_overlay_for_out_of_range_or_non_finite_coordinates(self):
        invalid_zones = [
            zone(x_start=-0.1),
            zone(x_start=10 ** 10000),
            zone(x_end=1.01),
            zone(y_top=float("nan")),
            zone(y_bottom=float("inf")),
            zone(x_start=True),
            zone(x_start=0.8, x_end=0.2),
            zone(y_top=0.7, y_bottom=0.6),
            zone(y_top=0.1, y_bottom=0.7),
        ]

        result = sanitize_chart_analysis(payload(invalid_zones, analysis="Zones are uncertain."))

        self.assertEqual(len(result["zones"]), 8)
        for item in result["zones"]:
            self.assertFalse(item["draw_overlay"])
            self.assertEqual(
                [item["x_start"], item["x_end"], item["y_top"], item["y_bottom"]],
                [0.0, 0.0, 0.0, 0.0],
            )

    def test_zone_crossing_price_scale_or_adjacent_panel_is_not_drawn(self):
        result = sanitize_chart_analysis(payload([
            zone(x_end=0.96),
            zone(y_top=0.08),
            zone(),
        ]))
        self.assertEqual([item["draw_overlay"] for item in result["zones"]], [False, False, True])
        for item in result["zones"][:2]:
            self.assertEqual(
                [item["x_start"], item["x_end"], item["y_top"], item["y_bottom"]],
                [0.0, 0.0, 0.0, 0.0],
            )

    def test_missing_or_invalid_plot_bounds_disable_all_overlays(self):
        for bounds in (
            None,
            {"x_start": 0.8, "x_end": 0.2, "y_top": 0.1, "y_bottom": 0.9},
            {"x_start": 0.1, "x_end": float("nan"), "y_top": 0.1, "y_bottom": 0.9},
        ):
            with self.subTest(bounds=bounds):
                result = sanitize_chart_analysis(payload([zone()], plot_bounds=bounds))
                self.assertIsNone(result["plot_bounds"])
                self.assertFalse(result["zones"][0]["draw_overlay"])

    def test_non_drawable_zone_has_zero_geometry(self):
        result = sanitize_chart_analysis(payload(
            [zone(draw_overlay=False, x_start=0.2, x_end=0.9)],
            analysis="No precise box can be located.",
        ))

        item = result["zones"][0]
        self.assertFalse(item["draw_overlay"])
        self.assertEqual(item["x_start"], 0.0)
        self.assertEqual(item["x_end"], 0.0)

    def test_ignores_unknown_zone_types_and_non_object_items(self):
        result = sanitize_chart_analysis(payload(
            [None, "supply", zone(zone_type="resistance"), zone(zone_type=[]), zone()],
            analysis="One supported zone.",
        ))

        self.assertEqual(len(result["zones"]), 1)
        self.assertEqual(result["zones"][0]["zone_type"], "demand")

    def test_caps_zone_payload_to_eight_items(self):
        result = sanitize_chart_analysis(payload(
            [zone(label=f"Zone {index}") for index in range(10)],
            analysis="Many visible areas; only the first eight are retained.",
        ))

        self.assertEqual(len(result["zones"]), 8)

    def test_invalid_early_items_do_not_consume_valid_zone_limit(self):
        zones = [
            None,
            "not-a-zone",
            *[zone(zone_type="resistance") for _ in range(6)],
            zone(label="Late valid demand"),
        ]

        result = sanitize_chart_analysis(payload(zones))

        self.assertEqual(len(result["zones"]), 1)
        self.assertEqual(result["zones"][0]["label"], "AI DEMAND")
        self.assertTrue(result["zones"][0]["draw_overlay"])

    def test_sanitizes_untrusted_label_and_price_fields(self):
        result = sanitize_chart_analysis(payload(
            [zone(label="L" * 100, price_low={"unexpected": "object"}, price_high="H" * 40)],
            analysis="A readable map.",
        ))

        item = result["zones"][0]
        self.assertEqual(item["label"], "AI DEMAND")
        self.assertEqual(item["origin"], "ai_inferred")
        self.assertIsNone(item["price_low"])
        self.assertIsNone(item["price_high"])
        self.assertIn("Numeric price bounds were withheld", result["analysis"])

    def test_preserves_a_complete_positive_decimal_price_pair(self):
        result = sanitize_chart_analysis(payload([
            zone(price_low=" 100.25 ", price_high="105.75")
        ]))
        item = result["zones"][0]
        self.assertEqual(item["price_low"], "100.25")
        self.assertEqual(item["price_high"], "105.75")
        self.assertTrue(item["draw_overlay"])
        self.assertNotIn("Numeric price bounds were withheld", result["analysis"])

    def test_withholds_incomplete_non_numeric_non_positive_or_reversed_price_pairs(self):
        invalid_pairs = [
            {"price_low": "105", "price_high": "100"},
            {"price_low": "100", "price_high": "100"},
            {"price_low": "0", "price_high": "10"},
            {"price_low": "-1", "price_high": "10"},
            {"price_low": "1e3", "price_high": "2e3"},
            {"price_low": "₹100", "price_high": "105"},
            {"price_low": "100", "price_high": None},
            {"price_low": 100, "price_high": "105"},
            {"price_low": "1" * 33, "price_high": "200"},
        ]
        for pair in invalid_pairs:
            with self.subTest(pair=pair):
                result = sanitize_chart_analysis(payload([zone(**pair)]))
                item = result["zones"][0]
                self.assertIsNone(item["price_low"])
                self.assertIsNone(item["price_high"])
                # Price-label validation is independent of normalized overlay geometry.
                self.assertTrue(item["draw_overlay"])
                self.assertIn("Numeric price bounds were withheld", result["analysis"])

    def test_missing_price_pair_is_not_reported_as_a_validation_error(self):
        result = sanitize_chart_analysis(payload([
            zone(price_low=None, price_high=None)
        ]))
        self.assertIsNone(result["zones"][0]["price_low"])
        self.assertIsNone(result["zones"][0]["price_high"])
        self.assertNotIn("Numeric price bounds were withheld", result["analysis"])

    def test_chart_prompt_requires_ordered_price_bounds_and_bumps_version(self):
        self.assertEqual(CHART_PROMPT_VERSION, "dns-chart-v8")
        combined = CHART_IMAGE_PROMPT + CHART_IMAGE_INSTRUCTIONS
        self.assertIn("price_high", combined)
        self.assertIn("price_low < price_high", combined)
        self.assertIn("both null", combined)
        self.assertIn("qualitative", combined)
        self.assertIn("low-confidence candidate", combined)
        self.assertIn("user_marked", combined)
        self.assertIn("ai_inferred", combined)
        self.assertIn("application derives", combined)
        self.assertIn("compact base", combined)
        self.assertIn("directional displacement", combined)
        self.assertIn("isolated wick", combined)
        self.assertIn("empty list", combined)
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
        self.assertIn("confidence", zone_schema["items"]["required"])
        self.assertIn("origin", zone_schema["items"]["required"])
        self.assertEqual(properties["confidence"]["enum"], ["high", "medium", "low"])
        self.assertEqual(properties["origin"]["enum"], ["user_marked", "ai_inferred"])
        self.assertNotIn("label", properties)
        self.assertIn("plot_bounds", schema["required"])
        plot_schema = schema["properties"]["plot_bounds"]
        self.assertEqual(plot_schema["type"], ["object", "null"])
        self.assertEqual(set(plot_schema["required"]), {"x_start", "x_end", "y_top", "y_bottom"})
        for coordinate in ("x_start", "x_end", "y_top", "y_bottom"):
            with self.subTest(coordinate=coordinate):
                self.assertEqual(properties[coordinate]["minimum"], 0)
                self.assertEqual(properties[coordinate]["maximum"], 1)


if __name__ == "__main__":
    unittest.main()
