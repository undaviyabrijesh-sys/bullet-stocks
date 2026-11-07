import unittest

from stockmind_telegram_mvp.app.ai import CHART_RESPONSE_FORMAT, sanitize_chart_analysis


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
        result = sanitize_chart_analysis(payload(
            [zone()],
            analysis="  Visible demand base near the prior impulse.  ",
        ))

        self.assertEqual(result["analysis"], "Visible demand base near the prior impulse.")
        self.assertEqual(len(result["zones"]), 1)
        self.assertTrue(result["zones"][0]["draw_overlay"])
        self.assertEqual(result["zones"][0]["x_start"], 0.2)
        self.assertEqual(result["plot_bounds"], PLOT_BOUNDS)

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
        self.assertEqual(result["zones"][0]["label"], "Late valid demand")
        self.assertTrue(result["zones"][0]["draw_overlay"])

    def test_sanitizes_untrusted_label_and_price_fields(self):
        result = sanitize_chart_analysis(payload(
            [zone(label="L" * 100, price_low={"unexpected": "object"}, price_high="H" * 40)],
            analysis="A readable map.",
        ))

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
        self.assertIn("plot_bounds", schema["required"])
        plot_schema = schema["properties"]["plot_bounds"]
        self.assertEqual(plot_schema["type"], ["object", "null"])
        self.assertEqual(set(plot_schema["required"]), {"x_start", "x_end", "y_top", "y_bottom"})
        properties = zone_schema["items"]["properties"]
        for coordinate in ("x_start", "x_end", "y_top", "y_bottom"):
            with self.subTest(coordinate=coordinate):
                self.assertEqual(properties[coordinate]["minimum"], 0)
                self.assertEqual(properties[coordinate]["maximum"], 1)


if __name__ == "__main__":
    unittest.main()
