import copy
import hashlib
import json
import sys
import tempfile
import types
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from stockmind_telegram_mvp.app.annotation_editor import (
    apply_manual_annotations,
    load_draft_and_image,
    review_annotation_draft,
    validate_manual_geometry,
)


PLOT = {"x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9}
ZONE = {
    "zone_type": "demand",
    "x_start": 0.2,
    "x_end": 0.8,
    "y_top": 0.6,
    "y_bottom": 0.8,
}


def _draft():
    return {
        "draft_schema": "stockmind.annotation-draft.v1",
        "benchmark_ready": False,
        "status": "pending_annotation",
        "fixture_id": "btc-1h-001",
        "image_file": "images/btc-1h-001.png",
        "image_sha256": "a" * 64,
        "image_width": 1600,
        "image_height": 900,
        "market_context": {"asset_class": "crypto", "symbol": "BTCUSDT", "exchange": "binance", "timeframe": "1h"},
        "reference_plot_bounds": None,
        "reference_zones": [],
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


class AnnotationGeometryTests(unittest.TestCase):
    def test_accepts_plot_with_no_visible_zones(self):
        plot, zones = validate_manual_geometry(PLOT, [])
        self.assertEqual(plot, PLOT)
        self.assertEqual(zones, [])

    def test_normalizes_numeric_coordinates_to_floats(self):
        plot, zones = validate_manual_geometry(
            {"x_start": 0, "x_end": 1, "y_top": 0, "y_bottom": 1},
            [{"zone_type": "supply", "x_start": 0, "x_end": 1, "y_top": 0, "y_bottom": 1}],
        )
        self.assertEqual(plot["x_start"], 0.0)
        self.assertEqual(zones[0]["y_bottom"], 1.0)

    def test_rejects_boolean_nan_infinite_and_out_of_range_coordinates(self):
        invalid_values = (True, float("nan"), float("inf"), -0.01, 1.01, "0.5")
        for value in invalid_values:
            with self.subTest(value=value):
                bad = dict(PLOT, x_start=value)
                with self.assertRaises(ValueError):
                    validate_manual_geometry(bad, [])

    def test_rejects_zero_or_negative_width_and_height(self):
        with self.assertRaisesRegex(ValueError, "positive width"):
            validate_manual_geometry(dict(PLOT, x_end=PLOT["x_start"]), [])
        with self.assertRaisesRegex(ValueError, "positive height"):
            validate_manual_geometry(dict(PLOT, y_bottom=PLOT["y_top"]), [])

    def test_rejects_zone_outside_plot(self):
        outside = dict(ZONE, x_start=0.05)
        with self.assertRaisesRegex(ValueError, "fully inside"):
            validate_manual_geometry(PLOT, [outside])

    def test_rejects_unknown_zone_type_and_extra_fields(self):
        with self.assertRaisesRegex(ValueError, "zone_type"):
            validate_manual_geometry(PLOT, [dict(ZONE, zone_type="neutral")])
        with self.assertRaisesRegex(ValueError, "exactly four"):
            validate_manual_geometry(PLOT, [dict(ZONE, note="guess")])

    def test_rejects_non_list_or_more_than_eight_zones(self):
        with self.assertRaisesRegex(ValueError, "must be a list"):
            validate_manual_geometry(PLOT, ())
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            validate_manual_geometry(PLOT, [ZONE] * 9)


class AnnotationWorkflowTests(unittest.TestCase):
    def test_annotation_creates_pending_review_without_mutating_input(self):
        draft = _draft()
        original = copy.deepcopy(draft)
        result = apply_manual_annotations(
            draft, PLOT, [ZONE], "ann-001", annotated_at="2026-11-07T12:00:00Z"
        )
        self.assertEqual(draft, original)
        self.assertEqual(result["status"], "pending_review")
        self.assertFalse(result["benchmark_ready"])
        self.assertEqual(result["reference_zones"], [ZONE])
        self.assertEqual(result["annotation_metadata"]["annotator_id"], "ann-001")
        self.assertIsNone(result["annotation_metadata"]["reviewer_id"])
        self.assertEqual(result["annotation_metadata"]["review_status"], "pending")

    def test_allows_no_zone_annotation_when_plot_is_valid(self):
        result = apply_manual_annotations(_draft(), PLOT, [], "ann-001", annotated_at="2026-11-07T12:00:00Z")
        self.assertEqual(result["reference_zones"], [])
        self.assertEqual(result["status"], "pending_review")

    def test_rejects_invalid_annotator_id(self):
        for value in ("ann-", "person@example.com", "../ann-1", "rev-001"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                apply_manual_annotations(_draft(), PLOT, [], value, annotated_at="2026-11-07T12:00:00Z")

    def test_rejects_unannotated_draft_with_existing_provenance(self):
        draft = _draft()
        draft["annotation_metadata"]["annotator_id"] = "ann-001"
        with self.assertRaisesRegex(ValueError, "must not contain"):
            apply_manual_annotations(draft, PLOT, [], "ann-002")

    def test_rejects_intake_draft_with_preexisting_geometry_or_prediction(self):
        draft_with_geometry = _draft()
        draft_with_geometry["reference_plot_bounds"] = PLOT
        with self.assertRaisesRegex(ValueError, "manual geometry"):
            apply_manual_annotations(draft_with_geometry, PLOT, [], "ann-001")

        draft_with_prediction = _draft()
        draft_with_prediction["prediction_status"] = "captured"
        draft_with_prediction["predicted_zones"] = []
        with self.assertRaisesRegex(ValueError, "AI prediction"):
            apply_manual_annotations(draft_with_prediction, PLOT, [], "ann-001")

    def test_requires_timezone_aware_annotation_timestamp(self):
        for value in ("2026-11-07T12:00:00", "not-a-time"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                apply_manual_annotations(_draft(), PLOT, [], "ann-001", annotated_at=value)

    def test_review_requires_reviewer_role_and_valid_time_order(self):
        annotated = apply_manual_annotations(_draft(), PLOT, [ZONE], "ann-001", annotated_at="2026-11-07T12:00:00Z")
        with self.assertRaisesRegex(ValueError, "reviewer_id"):
            review_annotation_draft(annotated, "ann-001", approve=True, reviewed_at="2026-11-07T12:01:00Z")
        with self.assertRaisesRegex(ValueError, "earlier"):
            review_annotation_draft(annotated, "rev-001", approve=True, reviewed_at="2026-11-07T11:59:00Z")

    def test_approval_is_a_separate_non_benchmark_ready_transition(self):
        annotated = apply_manual_annotations(_draft(), PLOT, [ZONE], "ann-001", annotated_at="2026-11-07T12:00:00Z")
        approved = review_annotation_draft(annotated, "rev-002", approve=True, reviewed_at="2026-11-07T12:05:00Z")
        self.assertEqual(approved["status"], "approved_pending_prediction")
        self.assertEqual(approved["annotation_metadata"]["review_status"], "approved")
        self.assertEqual(approved["annotation_metadata"]["reviewer_id"], "rev-002")
        self.assertFalse(approved["benchmark_ready"])
        self.assertEqual(annotated["status"], "pending_review")

    def test_rejection_is_recorded_and_cannot_be_mistaken_for_approval(self):
        annotated = apply_manual_annotations(_draft(), PLOT, [ZONE], "ann-001", annotated_at="2026-11-07T12:00:00Z")
        rejected = review_annotation_draft(annotated, "rev-002", approve=False, reviewed_at="2026-11-07T12:05:00Z")
        self.assertEqual(rejected["status"], "rejected_annotation")
        self.assertEqual(rejected["annotation_metadata"]["review_status"], "rejected")
        self.assertFalse(rejected["benchmark_ready"])

    def test_reviewer_cannot_approve_an_unreviewed_intake_draft(self):
        with self.assertRaisesRegex(ValueError, "status"):
            review_annotation_draft(_draft(), "rev-002", approve=True, reviewed_at="2026-11-07T12:05:00Z")

    def test_approve_flag_must_be_a_real_boolean(self):
        annotated = apply_manual_annotations(_draft(), PLOT, [], "ann-001", annotated_at="2026-11-07T12:00:00Z")
        with self.assertRaisesRegex(ValueError, "boolean"):
            review_annotation_draft(annotated, "rev-002", approve=1, reviewed_at="2026-11-07T12:05:00Z")


class DraftImageVerificationTests(unittest.TestCase):
    def _png(self, size=(32, 24)):
        output = BytesIO()
        Image.new("RGB", size, "white").save(output, format="PNG")
        return output.getvalue()

    def _draft_file(self, root, image_bytes, image_file="images/chart.png"):
        image_path = root / image_file
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)
        draft = _draft()
        draft["image_file"] = image_file
        draft["image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
        with Image.open(BytesIO(image_bytes)) as image:
            draft["image_width"], draft["image_height"] = image.size
        draft_path = root / "chart.draft.json"
        draft_path.write_text(json.dumps(draft), encoding="utf-8")
        return draft_path, image_path

    def test_loads_image_only_after_hash_and_dimensions_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = self._png()
            draft_path, _ = self._draft_file(root, data)
            stub = types.ModuleType("stockmind_telegram_mvp.app.overlay_engine")
            stub.validate_chart_image = lambda image_bytes: "image/png"
            with patch.dict(sys.modules, {stub.__name__: stub}):
                draft, image = load_draft_and_image(draft_path)
            self.assertEqual(draft["fixture_id"], "btc-1h-001")
            self.assertEqual(image.size, (32, 24))

    def test_rejects_screenshot_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft_path, image_path = self._draft_file(root, self._png())
            image_path.write_bytes(self._png((33, 24)))
            stub = types.ModuleType("stockmind_telegram_mvp.app.overlay_engine")
            stub.validate_chart_image = lambda image_bytes: "image/png"
            with patch.dict(sys.modules, {stub.__name__: stub}), self.assertRaisesRegex(ValueError, "SHA-256"):
                load_draft_and_image(draft_path)

    def test_rejects_unsafe_relative_image_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft = _draft()
            draft["image_file"] = "../outside.png"
            draft_path = root / "chart.draft.json"
            draft_path.write_text(json.dumps(draft), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "safe relative path"):
                load_draft_and_image(draft_path)


if __name__ == "__main__":
    unittest.main()
