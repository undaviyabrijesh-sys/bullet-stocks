import unittest

from stockmind_telegram_mvp.app.annotation_editor import (
    apply_manual_annotations,
    review_annotation_draft,
    validate_reference_zone_pixel_geometry,
)


PLOT = {"x_start": 0.05, "x_end": 0.95, "y_top": 0.05, "y_bottom": 0.95}


def draft(width=100, height=100):
    return {
        "draft_schema": "stockmind.annotation-draft.v1",
        "benchmark_ready": False,
        "status": "pending_annotation",
        "fixture_id": "btc-1h-pixel-check",
        "image_file": "images/chart.png",
        "image_sha256": "a" * 64,
        "image_width": width,
        "image_height": height,
        "market_context": {
            "asset_class": "crypto",
            "symbol": "BTCUSDT",
            "exchange": "binance",
            "timeframe": "1h",
        },
        "reference_plot_bounds": None,
        "reference_zones": [],
        "price_axis_anchors": [],
        "price_axis_scale": None,
        "price_axis_plot_bounds": None,
        "annotation_metadata": {
            "source": "manual",
            "annotator_id": None,
            "reviewer_id": None,
            "guideline_version": "dns-manual-v1",
            "review_status": "pending",
            "annotated_at": None,
            "reviewed_at": None,
        },
        "prediction_status": "not_captured",
        "predicted_plot_bounds": None,
        "predicted_zones": None,
        "prediction_metadata": None,
    }


def zone(**overrides):
    value = {
        "zone_type": "demand",
        "x_start": 0.2,
        "x_end": 0.8,
        "y_top": 0.3,
        "y_bottom": 0.4,
    }
    value.update(overrides)
    return value


class AnnotationBenchmarkCompatibilityTests(unittest.TestCase):
    def test_accepts_reference_zone_that_rounds_to_one_pixel(self):
        one_pixel = zone(x_start=0.10, x_end=0.11, y_top=0.20, y_bottom=0.21)
        validate_reference_zone_pixel_geometry([one_pixel], 100, 100)

    def test_rejects_zone_that_collapses_after_pixel_rounding(self):
        subpixel = zone(x_start=0.10, x_end=0.104)
        with self.assertRaisesRegex(ValueError, "not scoreable"):
            validate_reference_zone_pixel_geometry([subpixel], 100, 100)

    def test_rejects_zone_taller_than_evaluator_limit(self):
        tall = zone(y_top=0.20, y_bottom=0.70)
        with self.assertRaisesRegex(ValueError, "45%"):
            validate_reference_zone_pixel_geometry([tall], 100, 100)

    def test_rejects_invalid_image_dimensions(self):
        for width, height in ((True, 100), (100, False), (0, 100), (100, 0)):
            with self.subTest(width=width, height=height), self.assertRaisesRegex(
                ValueError, "positive integer"
            ):
                validate_reference_zone_pixel_geometry([], width, height)

    def test_annotation_rejects_unscoreable_reference_before_review(self):
        with self.assertRaisesRegex(ValueError, "not scoreable"):
            apply_manual_annotations(
                draft(),
                PLOT,
                [zone(y_top=0.20, y_bottom=0.70)],
                "ann-001",
                annotated_at="2026-11-07T12:00:00Z",
            )

    def test_review_revalidates_pixel_geometry_after_draft_tampering(self):
        annotated = apply_manual_annotations(
            draft(),
            PLOT,
            [zone()],
            "ann-001",
            annotated_at="2026-11-07T12:00:00Z",
        )
        annotated["reference_zones"][0]["y_top"] = 0.20
        annotated["reference_zones"][0]["y_bottom"] = 0.70

        with self.assertRaisesRegex(ValueError, "not scoreable"):
            review_annotation_draft(
                annotated,
                "rev-002",
                approve=True,
                reviewed_at="2026-11-07T12:05:00Z",
            )


if __name__ == "__main__":
    unittest.main()
