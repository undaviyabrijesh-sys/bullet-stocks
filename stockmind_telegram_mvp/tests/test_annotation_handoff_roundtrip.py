"""End-to-end persistence checks for the manual annotation handoff.

These tests use a synthetic image only. They exercise the same exclusive JSON
writer used by the desktop editor, then reload each state before review and
benchmark assembly. They do not measure chart-detection accuracy.
"""

import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from stockmind_telegram_mvp.app.annotation_editor import (
    _write_new_json,
    apply_manual_annotations,
    load_draft_and_image,
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


def _prediction_metadata():
    return {
        "provider": "openai",
        "model": "vision-model-test",
        "requested_model": "vision-model",
        "prompt_version": "dns-chart-v4",
        "prompt_sha256": "a" * 64,
        "generated_at": "2026-11-09T10:00:00Z",
        "system_fingerprint": "fp-test",
        "settings": {
            "temperature": 0.1,
            "image_detail": "high",
            "response_format": "json_schema:chart_zone_map",
            "conversation_context": "caption_only",
            "plot_bounds_policy": "required_for_overlay",
        },
    }


class AnnotationHandoffRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inbox = self.root / "inbox"
        self.source = self.root / "source.png"
        self.image_bytes = _png_bytes()
        self.source.write_bytes(self.image_bytes)
        self.image_sha256 = hashlib.sha256(self.image_bytes).hexdigest()
        intake = prepare_annotation_draft(
            self.source,
            self.inbox,
            "btc-1h-roundtrip",
            market_context={
                "asset_class": "crypto",
                "symbol": "BTCUSDT",
                "exchange": "binance",
                "timeframe": "1h",
            },
        )
        self.draft_path = self.inbox / intake["draft_file"]

    def tearDown(self):
        self.temp.cleanup()

    def _prediction(self):
        return {
            "analysis": "Synthetic fixture for annotation persistence testing.",
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
            "source_image_sha256": self.image_sha256,
        }

    def test_editor_json_roundtrip_survives_review_capture_and_benchmark(self):
        draft, _ = load_draft_and_image(self.draft_path)
        annotated = apply_manual_annotations(
            draft,
            PLOT,
            [ZONE],
            "ann-roundtrip",
            annotated_at="2026-11-09T10:01:00Z",
        )
        annotated_path = self.inbox / "btc-1h-roundtrip.annotated.json"
        _write_new_json(annotated_path, annotated)

        reloaded_annotation, _ = load_draft_and_image(annotated_path)
        self.assertEqual(
            reloaded_annotation["reference_zones"],
            annotated["reference_zones"],
        )
        self.assertEqual(
            reloaded_annotation["reference_plot_bounds"],
            annotated["reference_plot_bounds"],
        )
        self.assertEqual(reloaded_annotation["image_sha256"], self.image_sha256)

        approved = review_annotation_draft(
            reloaded_annotation,
            "rev-roundtrip",
            approve=True,
            reviewed_at="2026-11-09T10:02:00Z",
        )
        reviewed_path = self.inbox / "btc-1h-roundtrip.reviewed.json"
        _write_new_json(reviewed_path, approved)

        reloaded_review, _ = load_draft_and_image(reviewed_path)
        self.assertEqual(reloaded_review["status"], "approved_pending_prediction")
        self.assertEqual(
            reloaded_review["annotation_metadata"]["annotator_id"],
            "ann-roundtrip",
        )
        self.assertEqual(
            reloaded_review["annotation_metadata"]["reviewer_id"],
            "rev-roundtrip",
        )

        capture_info = write_prediction_capture(reviewed_path, self._prediction())
        capture_path = Path(capture_info["capture_file"])
        capture = build_prediction_capture(reviewed_path, self._prediction())
        self.assertEqual(capture["image_sha256"], self.image_sha256)
        captured_zone = capture["prediction"]["zones"][0]
        self.assertEqual(captured_zone["zone_type"], ZONE["zone_type"])
        for field in ("x_start", "x_end", "y_top", "y_bottom"):
            self.assertEqual(captured_zone[field], ZONE[field])

        assembled = assemble_benchmark_fixture(reviewed_path, capture_path)
        manifest_path = Path(assembled["manifest_file"])
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(record["reference_zones"], reloaded_review["reference_zones"])
        self.assertEqual(record["image_sha256"], self.image_sha256)

        scored = evaluate_benchmark_manifest(manifest_path)
        self.assertEqual(scored["fixture_count"], 1)
        self.assertEqual(scored["overall"]["true_positive"], 1)
        self.assertEqual(scored["overall"]["false_positive"], 0)
        self.assertEqual(scored["overall"]["false_negative"], 0)

    def test_changed_screenshot_after_review_and_capture_blocks_handoff(self):
        draft, _ = load_draft_and_image(self.draft_path)
        annotated = apply_manual_annotations(
            draft,
            PLOT,
            [ZONE],
            "ann-mutated-image",
            annotated_at="2026-11-09T10:01:00Z",
        )
        annotated_path = self.inbox / "mutated.annotated.json"
        _write_new_json(annotated_path, annotated)
        approved = review_annotation_draft(
            load_draft_and_image(annotated_path)[0],
            "rev-mutated-image",
            approve=True,
            reviewed_at="2026-11-09T10:02:00Z",
        )
        reviewed_path = self.inbox / "mutated.reviewed.json"
        _write_new_json(reviewed_path, approved)

        capture_info = write_prediction_capture(reviewed_path, self._prediction())
        capture_path = Path(capture_info["capture_file"])

        intake = json.loads(self.draft_path.read_text(encoding="utf-8"))
        image_path = self.inbox / intake["image_file"]
        Image.new("RGB", (64, 48), "black").save(image_path, format="PNG")

        with self.assertRaisesRegex(ValueError, "SHA-256"):
            build_prediction_capture(reviewed_path, self._prediction())
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            assemble_benchmark_fixture(reviewed_path, capture_path)

    def test_rejected_annotation_remains_ineligible_after_json_roundtrip(self):
        draft, _ = load_draft_and_image(self.draft_path)
        annotated = apply_manual_annotations(
            draft,
            PLOT,
            [ZONE],
            "ann-rejected",
            annotated_at="2026-11-09T10:01:00Z",
        )
        annotated_path = self.inbox / "rejected.annotated.json"
        _write_new_json(annotated_path, annotated)
        reloaded_annotation, _ = load_draft_and_image(annotated_path)

        rejected = review_annotation_draft(
            reloaded_annotation,
            "rev-rejected",
            approve=False,
            reviewed_at="2026-11-09T10:02:00Z",
        )
        rejected_path = self.inbox / "rejected.reviewed.json"
        _write_new_json(rejected_path, rejected)
        reloaded_rejection, _ = load_draft_and_image(rejected_path)

        self.assertEqual(reloaded_rejection["status"], "rejected_annotation")
        self.assertEqual(
            reloaded_rejection["annotation_metadata"]["review_status"],
            "rejected",
        )
        with self.assertRaisesRegex(ValueError, "independently approved"):
            build_prediction_capture(rejected_path, self._prediction())


if __name__ == "__main__":
    unittest.main()
