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

from stockmind_telegram_mvp.app.annotation_agreement import compare_reviewed_annotations
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
            "a", _png_bytes(), "ann-001", "rev-001", [ZONE], PLOT
        )
        self.path_b = self._make_reviewed(
            "b", _png_bytes(), "ann-002", "rev-002", [dict(ZONE)], PLOT
        )

    def tearDown(self):
        self.temp.cleanup()

    def _make_reviewed(self, folder, image_bytes, annotator_id, reviewer_id, zones, plot):
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

    def test_reports_partial_geometry_disagreement_without_adjudicating(self):
        shifted = {
            **ZONE,
            "x_start": 0.3,
            "x_end": 0.9,
        }
        path_b = self._make_reviewed(
            "shifted", _png_bytes(), "ann-003", "rev-003", [shifted], PLOT
        )
        report = compare_reviewed_annotations(self.path_a, path_b)
        self.assertEqual(report["zone_agreement"]["matched_zone_count"], 1)
        self.assertLess(report["zone_agreement"]["mean_matched_iou"], 1.0)
        self.assertIn("does not select or adjudicate", report["interpretation"])

    def test_does_not_match_supply_against_demand(self):
        supply = {**ZONE, "zone_type": "supply"}
        path_b = self._make_reviewed(
            "supply", _png_bytes(), "ann-004", "rev-004", [supply], PLOT
        )
        report = compare_reviewed_annotations(self.path_a, path_b)
        self.assertEqual(report["zone_agreement"]["matched_zone_count"], 0)
        self.assertEqual(report["zone_agreement"]["agreement_f1"], 0.0)

    def test_rejects_different_screenshot_hashes(self):
        path_b = self._make_reviewed(
            "different-image", _png_bytes("black"), "ann-005", "rev-005", [ZONE], PLOT
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

if __name__ == "__main__":
    unittest.main()
