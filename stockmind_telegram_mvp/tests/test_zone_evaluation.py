import unittest

from stockmind_telegram_mvp.app.zone_evaluation import evaluate_zone_predictions

def evaluate(predicted_zones, reference_zones, **kwargs):
    return evaluate_zone_predictions(
        predicted_zones,
        reference_zones,
        image_width=640,
        image_height=360,
        **kwargs,
    )




def zone(zone_type="demand", **overrides):
    value = {
        "zone_type": zone_type,
        "x_start": 0.2,
        "x_end": 0.6,
        "y_top": 0.4,
        "y_bottom": 0.6,
    }
    value.update(overrides)
    return value


def prediction(zone_type="demand", **overrides):
    value = zone(zone_type, **overrides)
    value.setdefault("draw_overlay", True)
    return value


PLOT_BOUNDS = {"x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9}


AXIS_ANCHORS = [
    {"y": 0.2, "price": 120},
    {"y": 0.5, "price": 90},
    {"y": 0.8, "price": 60},
]


class ZoneEvaluationTests(unittest.TestCase):
    def test_exact_match_has_perfect_scores(self):
        result = evaluate([prediction()], [zone()])
        self.assertEqual(result["overall"]["true_positive"], 1)
        self.assertEqual(result["overall"]["precision"], 1.0)
        self.assertEqual(result["overall"]["recall"], 1.0)
        self.assertEqual(result["overall"]["f1"], 1.0)
        self.assertEqual(result["overall"]["mean_iou"], 1.0)
        self.assertIsNone(result["price_accuracy"])

    def test_prediction_too_small_for_renderer_is_ignored_not_scored_as_match(self):
        tiny = prediction(
            x_start=0.0, x_end=0.2,
            y_top=0.2, y_bottom=0.4,
        )
        reference = zone(
            x_start=0.0, x_end=0.2,
            y_top=0.2, y_bottom=0.4,
        )
        result = evaluate_zone_predictions(
            [tiny],
            [reference],
            image_width=10,
            image_height=10,
        )
        self.assertEqual(result["ignored_prediction_count"], 1)
        self.assertEqual(result["overall"]["predicted"], 0)
        self.assertEqual(result["overall"]["true_positive"], 0)
        self.assertEqual(result["overall"]["false_negative"], 1)

    def test_evaluator_uses_the_same_pixel_rounding_as_renderer(self):
        predicted = prediction(
            x_start=0.2001, x_end=0.6001,
            y_top=0.4001, y_bottom=0.6001,
        )
        result = evaluate([predicted], [zone()])
        self.assertEqual(result["overall"]["mean_iou"], 1.0)

    def test_image_dimensions_must_be_positive_integers(self):
        for width, height in ((0, 360), (640, 0), (True, 360), (640, 360.0)):
            with self.subTest(width=width, height=height):
                with self.assertRaises(ValueError):
                    evaluate_zone_predictions(
                        [], [],
                        image_width=width,
                        image_height=height,
                    )

    def test_supply_does_not_match_demand_at_same_coordinates(self):
        result = evaluate([prediction("supply")], [zone("demand")])
        self.assertEqual(result["overall"]["true_positive"], 0)
        self.assertEqual(result["overall"]["false_positive"], 1)
        self.assertEqual(result["overall"]["false_negative"], 1)

    def test_duplicate_predictions_only_match_one_reference(self):
        result = evaluate([prediction(), prediction()], [zone()])
        self.assertEqual(result["overall"]["true_positive"], 1)
        self.assertEqual(result["overall"]["false_positive"], 1)
        self.assertEqual(result["overall"]["false_negative"], 0)

    def test_iou_threshold_rejects_weak_overlap(self):
        result = evaluate(
            [prediction(x_start=0.5, x_end=0.9)], [zone()], iou_threshold=0.5
        )
        self.assertEqual(result["overall"]["true_positive"], 0)

    def test_invalid_or_non_drawable_predictions_are_reported_as_ignored(self):
        result = evaluate(
            [
                prediction(draw_overlay=False),
                prediction(x_start=float("nan")),
                prediction(x_start=0.8, x_end=0.2),
            ],
            [zone()],
        )
        self.assertEqual(result["ignored_prediction_count"], 3)
        self.assertEqual(result["overall"]["predicted"], 0)
        self.assertEqual(result["overall"]["false_negative"], 1)

    def test_predictions_beyond_renderer_eight_zone_cap_are_ignored(self):
        result = evaluate([prediction() for _ in range(9)], [zone()])
        self.assertEqual(result["overall"]["predicted"], 8)
        self.assertEqual(result["ignored_prediction_count"], 1)

    def test_malformed_zone_type_is_ignored_for_prediction_and_rejected_for_reference(self):
        malformed = prediction(zone_type=["demand"])
        result = evaluate([malformed], [zone()])
        self.assertEqual(result["ignored_prediction_count"], 1)
        with self.assertRaisesRegex(ValueError, "Invalid manual reference zone"):
            evaluate([prediction()], [zone(zone_type=["demand"])])

    def test_invalid_manual_annotation_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "Invalid manual reference zone"):
            evaluate([prediction()], [zone(x_start=0.8, x_end=0.2)])

    def test_empty_comparison_uses_undefined_metrics_not_fake_perfect_score(self):
        result = evaluate([], [])
        self.assertIsNone(result["overall"]["precision"])
        self.assertIsNone(result["overall"]["recall"])
        self.assertIsNone(result["overall"]["f1"])
        self.assertIsNone(result["overall"]["mean_iou"])

    def test_invalid_threshold_is_rejected(self):
        for threshold in (0, 1.1, float("nan"), True):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    evaluate([], [], iou_threshold=threshold)

    def test_price_axis_error_is_reported_separately_from_iou(self):
        result = evaluate(
            [prediction(y_top=0.41, y_bottom=0.61)],
            [zone()],
            price_axis_anchors=AXIS_ANCHORS,
            price_axis_plot_bounds=PLOT_BOUNDS,
            predicted_plot_bounds=PLOT_BOUNDS,
        )
        self.assertEqual(result["overall"]["true_positive"], 1)
        self.assertGreater(result["overall"]["mean_iou"], 0.8)
        accuracy = result["price_accuracy"]
        self.assertEqual(accuracy["scored_match_count"], 1)
        self.assertAlmostEqual(accuracy["mean_boundary_abs_error"], 10 / 9)
        self.assertAlmostEqual(accuracy["mean_boundary_error_pct"], 1.25)
        comparison = result["matches"][0]["price_comparison"]
        self.assertAlmostEqual(comparison["predicted_price_bounds"]["price_high"], 98.8888888889)
        self.assertAlmostEqual(comparison["reference_price_bounds"]["price_low"], 80)

    def test_price_comparison_is_unscored_instead_of_extrapolating(self):
        result = evaluate(
            [prediction(y_top=0.19, y_bottom=0.4)],
            [zone(y_top=0.2, y_bottom=0.4)],
            price_axis_anchors=AXIS_ANCHORS,
            price_axis_plot_bounds=PLOT_BOUNDS,
            predicted_plot_bounds=PLOT_BOUNDS,
        )
        self.assertEqual(result["overall"]["true_positive"], 1)
        self.assertEqual(result["price_accuracy"]["scored_match_count"], 0)
        self.assertEqual(result["price_accuracy"]["unscored_match_count"], 1)
        self.assertEqual(result["matches"][0]["price_comparison"]["status"], "unscored")


    def test_price_axis_anchors_require_a_manually_marked_plot(self):
        with self.assertRaisesRegex(ValueError, "price_axis_plot_bounds is required"):
            evaluate([prediction()], [zone()], price_axis_anchors=AXIS_ANCHORS)

    def test_price_axis_anchors_outside_main_plot_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "falls outside the manually marked main price plot"):
            evaluate(
                [prediction()],
                [zone()],
                price_axis_anchors=[
                    {"y": 0.05, "price": 130},
                    {"y": 0.5, "price": 90},
                    {"y": 0.8, "price": 60},
                ],
                price_axis_plot_bounds=PLOT_BOUNDS,
            predicted_plot_bounds=PLOT_BOUNDS,
            )

    def test_plot_bounds_without_price_anchors_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires price_axis_anchors"):
            evaluate([prediction()], [zone()], price_axis_plot_bounds=PLOT_BOUNDS)

    def test_distorted_axis_fails_loudly_for_price_scoring(self):
        with self.assertRaisesRegex(ValueError, "distorted or inconsistent"):
            evaluate(
                [prediction()],
                [zone()],
                price_axis_anchors=[
                    {"y": 0.1, "price": 150},
                    {"y": 0.5, "price": 90},
                    {"y": 0.9, "price": 20},
                ],
                price_axis_plot_bounds=PLOT_BOUNDS,
            predicted_plot_bounds=PLOT_BOUNDS,
            )

    def test_plot_boundary_metrics_measure_edges_and_zone_spill(self):
        result = evaluate(
            [prediction(x_start=0.05, x_end=0.95, y_top=0.6, y_bottom=0.95)],
            [zone()],
            reference_plot_bounds=PLOT_BOUNDS,
            predicted_plot_bounds={
                "x_start": 0.04, "x_end": 0.96, "y_top": 0.08, "y_bottom": 0.96
            },
        )
        accuracy = result["plot_boundary_accuracy"]
        self.assertEqual(accuracy["status"], "scored")
        self.assertLess(accuracy["plot_iou"], 1.0)
        self.assertGreater(accuracy["edge_displacement_px"]["right"], 0)
        self.assertGreater(accuracy["edge_displacement_px"]["bottom"], 0)
        self.assertEqual(accuracy["predicted_zone_count"], 1)
        self.assertEqual(accuracy["zones_outside_reference_plot_count"], 1)
        self.assertEqual(accuracy["zones_outside_reference_plot_pct"], 100.0)
        self.assertGreater(accuracy["predicted_zone_area_outside_reference_pct"], 0)
        self.assertGreater(accuracy["right_margin_zone_area_pct"], 0)
        self.assertGreater(accuracy["bottom_panel_zone_area_pct"], 0)

    def test_exact_plot_boundary_has_perfect_iou_and_zero_edge_displacement(self):
        result = evaluate(
            [prediction()],
            [zone()],
            reference_plot_bounds=PLOT_BOUNDS,
            predicted_plot_bounds=PLOT_BOUNDS,
        )
        accuracy = result["plot_boundary_accuracy"]
        self.assertEqual(accuracy["plot_iou"], 1.0)
        self.assertEqual(
            accuracy["edge_displacement_px"],
            {"left": 0, "right": 0, "top": 0, "bottom": 0},
        )
        self.assertEqual(accuracy["zones_outside_reference_plot_count"], 0)

    def test_missing_and_invalid_plot_estimates_are_not_scored_as_perfect(self):
        missing_result = evaluate(
            [prediction()], [zone()], reference_plot_bounds=PLOT_BOUNDS
        )
        missing = missing_result["plot_boundary_accuracy"]
        self.assertEqual(missing["status"], "missing_prediction")
        self.assertIsNone(missing["plot_iou"])
        self.assertEqual(missing_result["ignored_prediction_count"], 1)
        self.assertEqual(missing_result["overall"]["predicted"], 0)
        self.assertEqual(missing_result["overall"]["false_negative"], 1)

        invalid_result = evaluate(
            [prediction()],
            [zone()],
            reference_plot_bounds=PLOT_BOUNDS,
            predicted_plot_bounds={
                "x_start": 0.8, "x_end": 0.2, "y_top": 0.1, "y_bottom": 0.9
            },
        )
        invalid = invalid_result["plot_boundary_accuracy"]
        self.assertEqual(invalid["status"], "invalid_prediction")
        self.assertIsNone(invalid["plot_iou"])
        self.assertEqual(invalid_result["ignored_prediction_count"], 1)
        self.assertEqual(invalid_result["overall"]["predicted"], 0)
        self.assertEqual(invalid_result["overall"]["false_negative"], 1)

    def test_zone_outside_model_plot_is_ignored_like_production_renderer(self):
        result = evaluate(
            [prediction(x_start=0.05, x_end=0.6)],
            [zone()],
            reference_plot_bounds=PLOT_BOUNDS,
            predicted_plot_bounds=PLOT_BOUNDS,
        )
        self.assertEqual(result["ignored_prediction_count"], 1)
        self.assertEqual(result["overall"]["predicted"], 0)
        self.assertEqual(result["overall"]["false_positive"], 0)
        self.assertEqual(result["overall"]["false_negative"], 1)
        self.assertEqual(
            result["plot_boundary_accuracy"]["predicted_zone_count"], 0
        )

    def test_predicted_plot_requires_independent_manual_reference(self):
        with self.assertRaisesRegex(ValueError, "reference_plot_bounds is required"):
            evaluate([prediction()], [zone()], predicted_plot_bounds=PLOT_BOUNDS)

    def test_invalid_manual_reference_plot_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference_plot_bounds must be a valid"):
            evaluate(
                [prediction()],
                [zone()],
                reference_plot_bounds={
                    "x_start": 0.8, "x_end": 0.2, "y_top": 0.1, "y_bottom": 0.9
                },
            )

    def test_manual_reference_zones_outside_reference_plot_are_rejected(self):
        outside_zones = (
            zone(x_start=0.05),
            zone(x_end=0.95),
            zone(y_top=0.05),
            zone(y_bottom=0.95),
        )
        for reference in outside_zones:
            with self.subTest(reference=reference):
                with self.assertRaisesRegex(
                    ValueError, "falls outside the manually marked main price plot"
                ):
                    evaluate(
                        [prediction()],
                        [reference],
                        reference_plot_bounds=PLOT_BOUNDS,
                    )

    def test_manual_reference_zone_touching_plot_edge_is_allowed(self):
        edge_zone = zone(x_start=PLOT_BOUNDS["x_start"])
        result = evaluate(
            [prediction(x_start=PLOT_BOUNDS["x_start"])],
            [edge_zone],
            reference_plot_bounds=PLOT_BOUNDS,
            predicted_plot_bounds=PLOT_BOUNDS,
        )
        self.assertEqual(result["overall"]["true_positive"], 1)

    def test_price_axis_plot_also_constrains_manual_reference_zones(self):
        with self.assertRaisesRegex(
            ValueError, "falls outside the manually marked main price plot"
        ):
            evaluate(
                [prediction()],
                [zone(x_start=0.05)],
                price_axis_anchors=AXIS_ANCHORS,
                price_axis_plot_bounds=PLOT_BOUNDS,
            )


if __name__ == "__main__":
    unittest.main()
