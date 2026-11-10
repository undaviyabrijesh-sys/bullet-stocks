"""Synthetic tests for pairwise manual annotation agreement.

The generated image is only a persistence/geometry fixture. These tests do not
measure StockMind chart-detection accuracy or market performance.
"""

import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from stockmind_telegram_mvp.app.annotation_agreement import (
    _price_axis_agreement,
    compare_reviewed_annotations,
)
from stockmind_telegram_mvp.app.annotation_editor import apply_manual_annotations, review_annotation_draft
from stockmind_telegram_mvp.app.annotation_intake import prepare_annotation_draft


PLOT = {"x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9}
ZONE = {
    "zone_type": "demand",
    "x_start": 0.2,
    "x_end": 0.8,
    "y_top": 0.6,
    "y_bottom": 0.8,
}


def _png_bytes(color="white"):
    output = BytesIO()
    Image.new("RGB", (100, 80), color).save(output, format="PNG")
    return output.getvalue()


class AnnotationAgreementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.context = {
            "asset_class": "crypto",
            "symbol": "BTCUSDT",
            "exchange": "binance",
            "timeframe": "1h",
        }
        self.path_a = self._make_reviewed(
            "a", _png_bytes(), "ann-001", "rev-101", [ZONE], PLOT
        )
        self.path_b = self._make_reviewed(
            "b", _png_bytes(), "ann-002", "rev-102", [dict(ZONE)], PLOT
        )

    def tearDown(self):
        self.temp.cleanup()

    def _make_reviewed(
        self, folder, image_bytes, annotator_id, reviewer_id, zones, plot,
        anchors=None, scale=None,
    ):
        root = self.root / folder
        root.mkdir()
        image_path = root / "source.png"
        image_path.write_bytes(image_bytes)
        intake = prepare_annotation_draft(
            image_path,
            root / "inbox",
            "btc-1h-agreement",
            market_context=self.context,
            guideline_version="dns-manual-v1",
        )
        draft_path = root / "inbox" / intake["draft_file"]
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
        annotated = apply_manual_annotations(
            draft,
            plot,
            zones,
            annotator_id,
            annotated_at="2026-11-10T04:00:00Z",
            price_axis_anchors=anchors,
            price_axis_scale=scale,
        )
        approved = review_annotation_draft(
            annotated,
            reviewer_id,
            approve=True,
            reviewed_at="2026-11-10T04:05:00Z",
        )
        reviewed_path = root / "inbox" / "reviewed.json"
        reviewed_path.write_text(
            json.dumps(approved, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        return reviewed_path

    def test_identical_annotations_have_full_pairwise_agreement(self):
        report = compare_reviewed_annotations(self.path_a, self.path_b)
        self.assertEqual(report["report_schema"], "stockmind.annotation-agreement.v1")
        self.assertEqual(report["zone_agreement"]["matched_zone_count"], 1)
        self.assertEqual(report["zone_agreement"]["agreement_f1"], 1.0)
        self.assertEqual(report["zone_agreement"]["mean_matched_iou"], 1.0)
        self.assertEqual(report["plot_agreement"]["iou"], 1.0)
        self.assertEqual(report["price_axis_agreement"]["status"], "not_provided")

    def test_reports_partial_geometry_disagreement_without_adjudicating(self):
        shifted = {
            **ZONE,
            "x_start": 0.3,
            "x_end": 0.9,
        }
        path_b = self._make_reviewed(
            "shifted", _png_bytes(), "ann-003", "rev-103", [shifted], PLOT
        )
        report = compare_reviewed_annotations(self.path_a, path_b)
        self.assertEqual(report["zone_agreement"]["matched_zone_count"], 1)
        self.assertLess(report["zone_agreement"]["mean_matched_iou"], 1.0)
        self.assertIn("does not select or adjudicate", report["interpretation"])

    def test_does_not_match_supply_against_demand(self):
        supply = {**ZONE, "zone_type": "supply"}
        path_b = self._make_reviewed(
            "supply", _png_bytes(), "ann-004", "rev-104", [supply], PLOT
        )
        report = compare_reviewed_annotations(self.path_a, path_b)
        self.assertEqual(report["zone_agreement"]["matched_zone_count"], 0)
        self.assertEqual(report["zone_agreement"]["agreement_f1"], 0.0)

    def test_rejects_different_screenshot_hashes(self):
        path_b = self._make_reviewed(
            "different-image", _png_bytes("black"), "ann-005", "rev-105", [ZONE], PLOT
        )
        with self.assertRaisesRegex(ValueError, "same screenshot SHA-256"):
            compare_reviewed_annotations(self.path_a, path_b)

    def test_rejects_reused_annotator_identity(self):
        draft = json.loads(self.path_b.read_text(encoding="utf-8"))
        draft["annotation_metadata"]["annotator_id"] = "ann-001"
        self.path_b.write_text(json.dumps(draft), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different annotator_id"):
            compare_reviewed_annotations(self.path_a, self.path_b)

    def test_rejects_different_guideline_versions(self):
        draft = json.loads(self.path_b.read_text(encoding="utf-8"))
        draft["annotation_metadata"]["guideline_version"] = "dns-manual-v2"
        self.path_b.write_text(json.dumps(draft), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "same guideline_version"):
            compare_reviewed_annotations(self.path_a, self.path_b)


    def test_rejects_cross_role_identity_token_reuse(self):
        draft = json.loads(self.path_b.read_text(encoding="utf-8"))
        draft["annotation_metadata"]["reviewer_id"] = "rev-001"
        self.path_b.write_text(json.dumps(draft), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity tokens must not overlap"):
            compare_reviewed_annotations(self.path_a, self.path_b)

    def test_empty_annotation_sets_are_not_reported_as_perfect_agreement(self):
        path_a = self._make_reviewed(
            "empty-a", _png_bytes(), "ann-006", "rev-106", [], PLOT
        )
        path_b = self._make_reviewed(
            "empty-b", _png_bytes(), "ann-007", "rev-107", [], PLOT
        )
        report = compare_reviewed_annotations(path_a, path_b)
        self.assertIsNone(report["zone_agreement"]["agreement_f1"])
        self.assertIsNone(report["zone_agreement"]["mean_matched_iou"])


    def test_reports_matching_manual_price_axis_calibrations(self):
        anchors_a = [
            {"y": 0.2, "price": 70000},
            {"y": 0.5, "price": 60000},
            {"y": 0.8, "price": 50000},
        ]
        anchors_b = [
            {"y": 0.2, "price": 70700},
            {"y": 0.5, "price": 60600},
            {"y": 0.8, "price": 50500},
        ]
        path_a = self._make_reviewed(
            "axis-a", _png_bytes(), "ann-008", "rev-108", [], PLOT,
            anchors_a, "linear",
        )
        path_b = self._make_reviewed(
            "axis-b", _png_bytes(), "ann-009", "rev-109", [], PLOT,
            anchors_b, "linear",
        )
        agreement = compare_reviewed_annotations(path_a, path_b)["price_axis_agreement"]
        self.assertEqual(agreement["status"], "compared")
        self.assertTrue(agreement["scale_match"])
        self.assertEqual(agreement["sample_count"], 11)
        self.assertGreater(agreement["mean_relative_price_disagreement"], 0.009)
        self.assertLess(agreement["mean_relative_price_disagreement"], 0.011)

    def test_reports_price_axis_missing_on_one_side(self):
        anchors = [
            {"y": 0.2, "price": 70000},
            {"y": 0.5, "price": 60000},
            {"y": 0.8, "price": 50000},
        ]
        path_a = self._make_reviewed(
            "axis-present", _png_bytes(), "ann-010", "rev-110", [], PLOT,
            anchors, "linear",
        )
        agreement = compare_reviewed_annotations(path_a, self.path_b)["price_axis_agreement"]
        self.assertEqual(agreement["status"], "missing_on_one_side")
        self.assertEqual(agreement["annotator_a_anchor_count"], 3)
        self.assertEqual(agreement["annotator_b_anchor_count"], 0)

    def test_reports_non_overlapping_price_axis_ranges_without_extrapolation(self):
        anchors_a = [
            {"y": 0.15, "price": 70000},
            {"y": 0.2, "price": 60000},
            {"y": 0.25, "price": 50000},
        ]
        anchors_b = [
            {"y": 0.75, "price": 70000},
            {"y": 0.8, "price": 60000},
            {"y": 0.85, "price": 50000},
        ]
        path_a = self._make_reviewed(
            "axis-range-a", _png_bytes(), "ann-011", "rev-111", [], PLOT,
            anchors_a, "linear",
        )
        path_b = self._make_reviewed(
            "axis-range-b", _png_bytes(), "ann-012", "rev-112", [], PLOT,
            anchors_b, "linear",
        )
        agreement = compare_reviewed_annotations(path_a, path_b)["price_axis_agreement"]
        self.assertEqual(agreement["status"], "no_overlap")
        self.assertIsNone(agreement.get("sample_count"))
        self.assertEqual(agreement["overlap_fraction_of_annotator_a_range"], 0.0)
        self.assertEqual(agreement["overlap_fraction_of_annotator_b_range"], 0.0)
        self.assertEqual(agreement["overlap_fraction_of_union_range"], 0.0)

    def test_reports_how_much_of_each_price_axis_range_is_shared(self):
        anchors_a = [
            {"y": 0.2, "price": 98000},
            {"y": 0.5, "price": 95000},
            {"y": 0.8, "price": 92000},
        ]
        anchors_b = [
            {"y": 0.5, "price": 95000},
            {"y": 0.7, "price": 93000},
            {"y": 0.9, "price": 91000},
        ]
        path_a = self._make_reviewed(
            "axis-overlap-a", _png_bytes(), "ann-015", "rev-115", [], PLOT,
            anchors_a, "linear",
        )
        path_b = self._make_reviewed(
            "axis-overlap-b", _png_bytes(), "ann-016", "rev-116", [], PLOT,
            anchors_b, "linear",
        )

        agreement = compare_reviewed_annotations(path_a, path_b)["price_axis_agreement"]

        self.assertEqual(agreement["status"], "compared")
        self.assertAlmostEqual(agreement["overlap_span"], 0.3)
        self.assertAlmostEqual(agreement["overlap_fraction_of_annotator_a_range"], 0.5)
        self.assertAlmostEqual(agreement["overlap_fraction_of_annotator_b_range"], 0.75)
        self.assertAlmostEqual(agreement["overlap_fraction_of_union_range"], 3 / 7)

    def test_rejects_invalid_price_axis_sample_counts(self):
        for sample_count in (True, 1, 1.5, 102):
            with self.subTest(sample_count=sample_count):
                with self.assertRaisesRegex(ValueError, "sample_count"):
                    _price_axis_agreement({}, {}, sample_count=sample_count)

    def test_compares_different_manual_axis_scales_without_conflating_them(self):
        linear_anchors = [
            {"y": 0.2, "price": 70000},
            {"y": 0.5, "price": 60000},
            {"y": 0.8, "price": 50000},
        ]
        log_anchors = [
            {"y": 0.2, "price": 72000},
            {"y": 0.5, "price": 60000},
            {"y": 0.8, "price": 50000},
        ]
        path_a = self._make_reviewed(
            "axis-linear", _png_bytes(), "ann-013", "rev-113", [], PLOT,
            linear_anchors, "linear",
        )
        path_b = self._make_reviewed(
            "axis-log", _png_bytes(), "ann-014", "rev-114", [], PLOT,
            log_anchors, "log",
        )
        agreement = compare_reviewed_annotations(path_a, path_b)["price_axis_agreement"]
        self.assertEqual(agreement["status"], "compared")
        self.assertFalse(agreement["scale_match"])
        self.assertGreater(agreement["max_relative_price_disagreement"], 0)

    def test_rejects_unhashable_zone_type_as_validation_error(self):
        from stockmind_telegram_mvp.app.zone_evaluation import evaluate_manual_annotation_agreement

        with self.assertRaisesRegex(ValueError, "invalid zone_type"):
            evaluate_manual_annotation_agreement(
                [{"zone_type": []}],
                [],
                image_width=100,
                image_height=80,
            )

if __name__ == "__main__":
    unittest.main()
