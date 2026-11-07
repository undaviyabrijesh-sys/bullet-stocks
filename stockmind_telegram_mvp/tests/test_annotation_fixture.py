import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from stockmind_telegram_mvp.app.annotation_editor import (
    apply_manual_annotations,
    review_annotation_draft,
)
from stockmind_telegram_mvp.app.annotation_fixture import (
    assemble_benchmark_fixture,
    build_prediction_capture,
    write_prediction_capture,
)
from stockmind_telegram_mvp.app.annotation_intake import prepare_annotation_draft
from stockmind_telegram_mvp.app.zone_benchmark import evaluate_benchmark_manifest


PLOT = {"x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9}
ZONE = {
    "zone_type": "demand",
    "x_start": 0.2,
    "x_end": 0.8,
    "y_top": 0.6,
    "y_bottom": 0.8,
}


def _png_bytes():
    output = BytesIO()
    Image.new("RGB", (64, 48), "white").save(output, format="PNG")
    return output.getvalue()


def _prediction_metadata(**overrides):
    value = {
        "provider": "openai",
        "model": "vision-model-2026-11-07",
        "requested_model": "vision-model",
        "prompt_version": "dns-chart-v4",
        "prompt_sha256": "a" * 64,
        "generated_at": "2026-11-07T12:10:00Z",
        "system_fingerprint": "fp-test",
        "settings": {
            "temperature": 0.1,
            "image_detail": "high",
            "response_format": "json_schema:chart_zone_map",
            "conversation_context": "caption_only",
            "plot_bounds_policy": "required_for_overlay",
        },
    }
    value.update(overrides)
    return value


class AnnotationFixtureAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inbox = self.root / "inbox"
        self.source = self.root / "source.png"
        self.image_bytes = _png_bytes()
        self.source.write_bytes(self.image_bytes)
        intake = prepare_annotation_draft(
            self.source,
            self.inbox,
            "btc-1h-001",
            market_context={
                "asset_class": "crypto",
                "symbol": "BTCUSDT",
                "exchange": "binance",
                "timeframe": "1h",
            },
        )
        draft_path = self.inbox / intake["draft_file"]
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
        annotated = apply_manual_annotations(
            draft,
            PLOT,
            [ZONE],
            "ann-001",
            annotated_at="2026-11-07T12:00:00Z",
        )
        approved = review_annotation_draft(
            annotated,
            "rev-002",
            approve=True,
            reviewed_at="2026-11-07T12:05:00Z",
        )
        self.reviewed_path = self.inbox / "btc-1h-001.reviewed.json"
        self.reviewed_path.write_text(json.dumps(approved), encoding="utf-8")
        self.image_hash = hashlib.sha256(self.image_bytes).hexdigest()
        self.result = {
            "analysis": "Visible demand structure; verify against source chart.",
            "plot_bounds": dict(PLOT),
            "zones": [{
                "zone_type": "demand",
                "label": "AI DEMAND",
                "price_low": None,
                "price_high": None,
                **ZONE,
                "draw_overlay": True,
            }],
            "prediction_metadata": _prediction_metadata(),
            "source_image_sha256": self.image_hash,
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_builds_capture_bound_to_exact_reviewed_screenshot(self):
        capture = build_prediction_capture(self.reviewed_path, self.result)
        self.assertEqual(capture["fixture_id"], "btc-1h-001")
        self.assertEqual(capture["image_sha256"], self.image_hash)
        self.assertEqual(capture["image_width"], 64)
        self.assertEqual(capture["image_height"], 48)
        self.assertEqual(
            capture["prediction"]["prediction_metadata"]["model"],
            "vision-model-2026-11-07",
        )

    def test_rejects_prediction_from_different_screenshot(self):
        altered = dict(self.result, source_image_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_prediction_capture(self.reviewed_path, altered)

    def test_rejects_result_without_source_image_fingerprint(self):
        altered = dict(self.result)
        altered.pop("source_image_sha256")
        with self.assertRaisesRegex(ValueError, "missing source_image_sha256"):
            build_prediction_capture(self.reviewed_path, altered)

    def test_rejects_unsanitized_or_tampered_prediction_geometry(self):
        altered = json.loads(json.dumps(self.result))
        altered["zones"][0]["x_start"] = -0.1
        with self.assertRaisesRegex(ValueError, "sanitized ResearchAI result"):
            build_prediction_capture(self.reviewed_path, altered)

    def test_rejects_missing_provider_model_or_timestamp_instead_of_guessing(self):
        for field in ("model", "generated_at"):
            altered = json.loads(json.dumps(self.result))
            altered["prediction_metadata"][field] = None
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                build_prediction_capture(self.reviewed_path, altered)

    def test_rejects_annotation_that_has_not_been_independently_approved(self):
        draft = json.loads(self.reviewed_path.read_text(encoding="utf-8"))
        draft["status"] = "pending_review"
        self.reviewed_path.write_text(json.dumps(draft), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "independently approved"):
            build_prediction_capture(self.reviewed_path, self.result)

    def test_writes_capture_and_assembles_evaluator_valid_jsonl(self):
        capture_info = write_prediction_capture(self.reviewed_path, self.result)
        capture_path = Path(capture_info["capture_file"])
        self.assertFalse(capture_info["benchmark_ready"])
        assembled = assemble_benchmark_fixture(self.reviewed_path, capture_path)
        manifest = Path(assembled["manifest_file"])
        self.assertTrue(assembled["benchmark_ready"])
        self.assertEqual(assembled["validation_fixture_count"], 1)

        scored = evaluate_benchmark_manifest(manifest)
        self.assertEqual(scored["fixture_count"], 1)
        self.assertEqual(scored["overall"]["true_positive"], 1)
        self.assertEqual(scored["overall"]["false_positive"], 0)
        self.assertEqual(scored["overall"]["false_negative"], 0)
        self.assertEqual(scored["fixtures"][0]["image_sha256"], self.image_hash)

    def test_copies_manual_price_axis_calibration_into_fixture(self):
        draft = json.loads(self.reviewed_path.read_text(encoding="utf-8"))
        draft.update({
            "status": "pending_annotation",
            "annotation_metadata": {
                "source": "manual",
                "annotator_id": None,
                "reviewer_id": None,
                "guideline_version": "dns-manual-v1",
                "review_status": "pending",
                "annotated_at": None,
                "reviewed_at": None,
            },
            "reference_plot_bounds": None,
            "reference_zones": [],
            "price_axis_anchors": [],
            "price_axis_scale": None,
            "price_axis_plot_bounds": None,
        })
        annotated = apply_manual_annotations(
            draft,
            PLOT,
            [ZONE],
            "ann-003",
            annotated_at="2026-11-07T12:00:00Z",
            price_axis_anchors=[
                {"y": 0.2, "price": 120},
                {"y": 0.5, "price": 90},
                {"y": 0.8, "price": 60},
            ],
            price_axis_scale="linear",
        )
        approved = review_annotation_draft(
            annotated, "rev-004", approve=True, reviewed_at="2026-11-07T12:05:00Z"
        )
        approved["fixture_id"] = "btc-1h-axis"
        axis_path = self.inbox / "btc-1h-axis.reviewed.json"
        axis_path.write_text(json.dumps(approved), encoding="utf-8")
        capture_obj = build_prediction_capture(axis_path, self.result)
        capture_path = self.inbox / "btc-1h-axis.prediction.json"
        capture_path.write_text(json.dumps(capture_obj), encoding="utf-8")
        output = assemble_benchmark_fixture(axis_path, capture_path)
        record = json.loads(Path(output["manifest_file"]).read_text(encoding="utf-8"))
        self.assertEqual(record["price_axis_scale"], "linear")
        self.assertEqual(len(record["price_axis_anchors"]), 3)
        self.assertEqual(record["price_axis_plot_bounds"], PLOT)

    def test_rejects_capture_for_another_fixture(self):
        capture = build_prediction_capture(self.reviewed_path, self.result)
        capture["fixture_id"] = "other-chart"
        capture_path = self.inbox / "tampered.prediction.json"
        capture_path.write_text(json.dumps(capture), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fixture_id"):
            assemble_benchmark_fixture(self.reviewed_path, capture_path)

    def test_refuses_to_overwrite_existing_capture_or_manifest(self):
        capture_info = write_prediction_capture(self.reviewed_path, self.result)
        capture_path = Path(capture_info["capture_file"])
        original_capture = capture_path.read_bytes()
        with self.assertRaises(FileExistsError):
            write_prediction_capture(self.reviewed_path, self.result)
        self.assertEqual(capture_path.read_bytes(), original_capture)

        manifest_info = assemble_benchmark_fixture(self.reviewed_path, capture_path)
        manifest_path = Path(manifest_info["manifest_file"])
        original_manifest = manifest_path.read_bytes()
        with self.assertRaises(FileExistsError):
            assemble_benchmark_fixture(self.reviewed_path, capture_path)
        self.assertEqual(manifest_path.read_bytes(), original_manifest)

    def test_manifest_must_be_next_to_reviewed_draft(self):
        capture_info = write_prediction_capture(self.reviewed_path, self.result)
        outside = self.root / "outside" / "fixture.jsonl"
        with self.assertRaisesRegex(ValueError, "beside the reviewed annotation"):
            assemble_benchmark_fixture(
                self.reviewed_path, capture_info["capture_file"], outside
            )



def test_rejects_duplicate_keys_in_prediction_capture_json(self):
        capture = build_prediction_capture(self.reviewed_path, self.result)
        capture_path = self.inbox / "duplicate.prediction.json"
        raw = json.dumps(capture)
        raw = raw.replace(
            '"fixture_id": "btc-1h-001"',
            '"fixture_id": "shadow", "fixture_id": "btc-1h-001"',
            1,
        )
        capture_path.write_text(raw, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            assemble_benchmark_fixture(self.reviewed_path, capture_path)

    def test_rejects_non_standard_constants_in_prediction_capture_json(self):
        capture = build_prediction_capture(self.reviewed_path, self.result)
        capture_path = self.inbox / "nonfinite.prediction.json"
        raw = json.dumps(capture).replace('"image_width": 64', '"image_width": NaN', 1)
        capture_path.write_text(raw, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-standard JSON constant"):
            assemble_benchmark_fixture(self.reviewed_path, capture_path)

if __name__ == "__main__":
    unittest.main()
