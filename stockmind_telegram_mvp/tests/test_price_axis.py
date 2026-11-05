import math
import unittest

from stockmind_telegram_mvp.app.price_axis import (
    calibrate_price_axis,
    zone_price_bounds,
)


class PriceAxisCalibrationTests(unittest.TestCase):
    def test_linear_axis_maps_coordinates_to_price_without_extrapolation(self):
        calibration = calibrate_price_axis([
            {"y": 0.2, "price": 120},
            {"y": 0.5, "price": 90},
            {"y": 0.8, "price": 60},
        ])
        self.assertAlmostEqual(calibration.price_at_y(0.4), 100)
        self.assertAlmostEqual(calibration.max_anchor_relative_error, 0)
        self.assertEqual(calibration.anchor_count, 3)

    def test_log_axis_maps_exponential_price_scale(self):
        anchors = [
            {"y": y, "price": 1000 * math.exp(-2 * y)}
            for y in (0.1, 0.45, 0.9)
        ]
        calibration = calibrate_price_axis(anchors, scale="log")
        self.assertAlmostEqual(calibration.price_at_y(0.6), 1000 * math.exp(-1.2))
        self.assertEqual(calibration.scale, "log")

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
