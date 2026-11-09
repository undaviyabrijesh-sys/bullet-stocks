import math
import sys
import unittest

from decimal import Decimal

from stockmind_telegram_mvp.app.price_axis import (
    PriceAxisCalibration,
    calibrate_price_axis,
    parse_zone_price_bounds,
    zone_price_bounds,
)

class PriceAxisCalibrationTests(unittest.TestCase):
    def test_parses_only_complete_positive_ordered_decimal_price_pairs(self):
        self.assertEqual(
            parse_zone_price_bounds({"price_low": "100.25", "price_high": "105.75"}),
            (Decimal("100.25"), Decimal("105.75")),
        )
        for pair in (
            {"price_low": "105", "price_high": "100"},
            {"price_low": "100", "price_high": "100"},
            {"price_low": "0", "price_high": "10"},
            {"price_low": "-1", "price_high": "10"},
            {"price_low": "1e3", "price_high": "2e3"},
            {"price_low": "₹100", "price_high": "105"},
            {"price_low": "100", "price_high": None},
            {"price_low": "1" * 33, "price_high": "200"},
        ):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                parse_zone_price_bounds(pair)
    def test_linear_axis_maps_coordinates_to_price_without_extrapolation(self):
        calibration = calibrate_price_axis([
            {"y": 0.2, "price": 120},
            {"y": 0.5, "price": 90},
            {"y": 0.8, "price": 60},
        ])
        self.assertAlmostEqual(calibration.price_at_y(0.4), 100)
        self.assertAlmostEqual(calibration.max_anchor_relative_error, 0)
        self.assertEqual(calibration.anchor_count, 3)

    def test_linear_calibration_keeps_valid_region_when_y_zero_intercept_overflows(self):
        anchors = [
            {"y": 0.9, "price": 1.7e308},
            {"y": 0.95, "price": 1.65e308},
            {"y": 1.0, "price": 1.6e308},
        ]

        calibration = calibrate_price_axis(anchors, max_relative_error=0.02)

        self.assertIsNone(calibration.intercept)
        self.assertTrue(math.isfinite(calibration.slope))
        self.assertAlmostEqual(calibration.reference_y, 0.95)
        self.assertTrue(math.isclose(
            calibration.reference_value, 1.65e308, rel_tol=1e-12
        ))
        for anchor in anchors:
            self.assertTrue(math.isclose(
                calibration.price_at_y(anchor["y"]),
                anchor["price"],
                rel_tol=1e-12,
            ))
        bounds = zone_price_bounds(
            {"y_top": 0.92, "y_bottom": 0.98},
            calibration,
        )
        self.assertGreater(bounds["price_high"], bounds["price_low"])
        self.assertTrue(math.isfinite(bounds["price_high"]))
        self.assertTrue(math.isfinite(bounds["price_low"]))

    def test_legacy_positional_calibration_constructor_still_works(self):
        calibration = PriceAxisCalibration(
            "linear", -100.0, 140.0, 0.2, 0.8, 3, 0.0
        )
        self.assertAlmostEqual(calibration.price_at_y(0.4), 100.0)

    def test_rejects_malformed_direct_calibration_state(self):
        invalid_calibrations = (
            PriceAxisCalibration("percent", -100.0, 140.0, 0.2, 0.8, 3, 0.0),
            PriceAxisCalibration("linear", 100.0, 140.0, 0.2, 0.8, 3, 0.0),
            PriceAxisCalibration("linear", -100.0, 140.0, float("nan"), 0.8, 3, 0.0),
            PriceAxisCalibration("linear", -100.0, 140.0, 0.2, float("inf"), 3, 0.0),
            PriceAxisCalibration("linear", -100.0, 140.0, 0.8, 0.2, 3, 0.0),
            PriceAxisCalibration("linear", -100.0, float("nan"), 0.2, 0.8, 3, 0.0),
        )
        for calibration in invalid_calibrations:
            with self.subTest(calibration=calibration), self.assertRaises(ValueError):
                calibration.price_at_y(0.4)

    def test_rejects_centered_reference_outside_anchor_range(self):
        calibration = PriceAxisCalibration(
            "linear", -100.0, 140.0, 0.2, 0.8, 3, 0.0,
            reference_y=0.9, reference_value=100.0,
        )
        with self.assertRaisesRegex(ValueError, "reference_y"):
            calibration.price_at_y(0.4)

    def test_rounding_drift_at_anchor_edges_is_clamped_not_extrapolated(self):
        calibration = calibrate_price_axis([
            {"y": 0.2, "price": 120},
            {"y": 0.5, "price": 90},
            {"y": 0.8, "price": 60},
        ])
        self.assertAlmostEqual(
            calibration.price_at_y(0.2 - 0.5e-9),
            calibration.price_at_y(0.2),
        )
        self.assertAlmostEqual(
            calibration.price_at_y(0.8 + 0.5e-9),
            calibration.price_at_y(0.8),
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            calibration.price_at_y(0.2 - 2e-9)

    def test_rejects_incomplete_centered_reference_pair(self):
        for reference_fields in (
            {"reference_y": 0.5},
            {"reference_value": 100.0},
        ):
            with self.subTest(reference_fields=reference_fields):
                calibration = PriceAxisCalibration(
                    "linear", -100.0, 140.0, 0.2, 0.8, 3, 0.0,
                    **reference_fields,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "requires both reference_y and reference_value",
                ):
                    calibration.price_at_y(0.4)

    def test_log_axis_maps_exponential_price_scale(self):
        anchors = [
            {"y": y, "price": 1000 * math.exp(-2 * y)}
            for y in (0.1, 0.45, 0.9)
        ]
        calibration = calibrate_price_axis(anchors, scale="log")
        self.assertAlmostEqual(calibration.price_at_y(0.6), 1000 * math.exp(-1.2))
        self.assertEqual(calibration.scale, "log")

    def test_log_axis_zone_bounds_use_centered_reference_when_zero_intercept_cannot_be_exponentiated(self):
        anchors = [
            {"y": 0.9, "price": 1e308},
            {"y": 0.95, "price": 1e307},
            {"y": 1.0, "price": 1e306},
        ]
        calibration = calibrate_price_axis(anchors, scale="log")

        # The fitted log-intercept is finite, but exp(intercept) is outside
        # float range. In-range prices must still use the centered reference.
        self.assertGreater(calibration.intercept, math.log(sys.float_info.max))
        bounds = zone_price_bounds(
            {"y_top": 0.91, "y_bottom": 0.99},
            calibration,
        )

        self.assertTrue(math.isfinite(bounds["price_high"]))
        self.assertTrue(math.isfinite(bounds["price_low"]))
        self.assertGreater(bounds["price_high"], bounds["price_low"])
        self.assertTrue(math.isclose(
            bounds["price_high"], 6.30957344480203e307, rel_tol=1e-12
        ))
        self.assertTrue(math.isclose(
            bounds["price_low"], 1.584893192461162e306, rel_tol=1e-12
        ))

    def test_linear_calibration_scales_large_finite_anchor_prices_before_fitting(self):
        anchors = [
            {"y": 0.1, "price": 9e307},
            {"y": 0.5, "price": 7e307},
            {"y": 0.9, "price": 5e307},
        ]
        calibration = calibrate_price_axis(anchors)

        self.assertTrue(math.isclose(
            calibration.price_at_y(0.3), 8e307, rel_tol=1e-12
        ))
        self.assertTrue(math.isclose(
            calibration.price_at_y(0.7), 6e307, rel_tol=1e-12
        ))
        self.assertTrue(math.isfinite(calibration.max_anchor_relative_error))

    def test_log_calibration_handles_largest_and_smallest_finite_prices(self):
        highest_price = sys.float_info.max
        lowest_price = math.ulp(0.0)
        midpoint_price = math.exp(
            (math.log(highest_price) + math.log(lowest_price)) / 2
        )
        anchors = [
            {"y": 0.1, "price": highest_price},
            {"y": 0.5, "price": midpoint_price},
            {"y": 0.9, "price": lowest_price},
        ]

        calibration = calibrate_price_axis(anchors, scale="log")

        for anchor in anchors:
            self.assertTrue(math.isclose(
                calibration.price_at_y(anchor["y"]),
                anchor["price"],
                rel_tol=1e-12,
            ))
        self.assertTrue(math.isfinite(calibration.max_anchor_relative_error))

    def test_rejects_anchor_residual_that_overflows_relative_error(self):
        anchors = [
            {"y": 0.1, "price": 1e307},
            {"y": 0.4, "price": 9e306},
            {"y": 0.5, "price": math.ulp(0.0)},
        ]
        with self.assertRaisesRegex(ValueError, "calibration residual exceeds"):
            calibrate_price_axis(anchors)

    def test_rejects_linear_fit_that_cannot_be_represented_as_finite_slope(self):
        anchors = [
            {"y": 0.1, "price": 9e307},
            {"y": 0.100000002, "price": 5e307},
            {"y": 0.100000004, "price": 1e307},
        ]
        with self.assertRaisesRegex(ValueError, "exceeds the supported numeric range"):
            calibrate_price_axis(anchors)

    def test_zone_bounds_return_high_at_top_and_low_at_bottom(self):
        calibration = calibrate_price_axis([
            {"y": 0.2, "price": 120},
            {"y": 0.5, "price": 90},
            {"y": 0.8, "price": 60},
        ])
        bounds = zone_price_bounds({"y_top": 0.4, "y_bottom": 0.6}, calibration)
        self.assertAlmostEqual(bounds["price_high"], 100)
        self.assertAlmostEqual(bounds["price_low"], 80)

    def test_requires_three_anchors_to_check_axis_consistency(self):
        with self.assertRaisesRegex(ValueError, "between 3 and"):
            calibrate_price_axis([
                {"y": 0.2, "price": 120},
                {"y": 0.8, "price": 60},
            ])

    def test_rejects_non_monotonic_or_duplicate_anchors(self):
        with self.assertRaisesRegex(ValueError, "price must decrease"):
            calibrate_price_axis([
                {"y": 0.2, "price": 120},
                {"y": 0.5, "price": 125},
                {"y": 0.8, "price": 60},
            ])
        with self.assertRaisesRegex(ValueError, "distinct y"):
            calibrate_price_axis([
                {"y": 0.2, "price": 120},
                {"y": 0.2, "price": 100},
                {"y": 0.8, "price": 60},
            ])

    def test_rejects_distorted_axis_instead_of_forcing_a_fit(self):
        with self.assertRaisesRegex(ValueError, "distorted or inconsistent"):
            calibrate_price_axis([
                {"y": 0.1, "price": 150},
                {"y": 0.5, "price": 90},
                {"y": 0.9, "price": 20},
            ])

    def test_rejects_invalid_scale_and_tolerance(self):
        anchors = [
            {"y": 0.2, "price": 120},
            {"y": 0.5, "price": 90},
            {"y": 0.8, "price": 60},
        ]
        for kwargs in (
            {"scale": "percent"},
            {"scale": []},
            {"max_relative_error": 0},
            {"max_relative_error": 0.2},
            {"max_relative_error": True},
            {"max_relative_error": 10 ** 10000},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    calibrate_price_axis(anchors, **kwargs)

    def test_rejects_invalid_anchor_values(self):
        base = [
            {"y": 0.2, "price": 120},
            {"y": 0.5, "price": 90},
            {"y": 0.8, "price": 60},
        ]
        invalid = [
            [{"y": True, "price": 120}, *base[1:]],
            [{**base[0], "y": float("nan")}, *base[1:]],
            [{**base[0], "price": float("inf")}, *base[1:]],
            [{**base[0], "price": 10 ** 10000}, *base[1:]],
            [{**base[0], "price": 0}, *base[1:]],
            [None, *base[1:]],
        ]
        for anchors in invalid:
            with self.subTest(anchors=anchors):
                with self.assertRaises(ValueError):
                    calibrate_price_axis(anchors)

    def test_does_not_extrapolate_outside_anchor_range(self):
        calibration = calibrate_price_axis([
            {"y": 0.2, "price": 120},
            {"y": 0.5, "price": 90},
            {"y": 0.8, "price": 60},
        ])
        with self.assertRaisesRegex(ValueError, "outside"):
            calibration.price_at_y(0.1)

    def test_rejects_zone_outside_calibration_range(self):
        calibration = calibrate_price_axis([
            {"y": 0.2, "price": 120},
            {"y": 0.5, "price": 90},
            {"y": 0.8, "price": 60},
        ])
        with self.assertRaisesRegex(ValueError, "outside"):
            zone_price_bounds({"y_top": 0.1, "y_bottom": 0.3}, calibration)


if __name__ == "__main__":
    unittest.main()
