import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from stockmind_telegram_mvp.app.zone_benchmark import evaluate_benchmark_manifest
from stockmind_telegram_mvp.app.zone_benchmark_compare import compare_benchmark_reports


def _image():
    output = BytesIO()
    Image.new("RGB", (40, 30), "white").save(output, format="PNG")
    return output.getvalue()


def _zone(**changes):
    zone = {
        "zone_type": "demand", "x_start": 0.2, "x_end": 0.6,
        "y_top": 0.4, "y_bottom": 0.6
    }
    zone.update(changes)
    return zone


def _annotation():
    return {
        "source": "manual", "annotator_id": "ann-001", "reviewer_id": "rev-002",
        "guideline_version": "dns-manual-v1", "review_status": "approved",
        "annotated_at": "2026-11-05T19:00:00Z", "reviewed_at": "2026-11-05T19:15:00Z"
    }


class PairedBenchmarkComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image_bytes = _image()
        (self.root / "chart.png").write_bytes(self.image_bytes)
        self.image_hash = hashlib.sha256(self.image_bytes).hexdigest()
        self.context = {
            "asset_class": "crypto", "symbol": "BTCUSDT",
            "exchange": "binance", "timeframe": "1h"
        }

    def tearDown(self):
        self.temp.cleanup()

    def _report(self, model, predicted, *, fixture_id="btc-1h-001",
                reference=None, context=None, anchors=None, threshold=0.5):
        metadata = {
            "provider": "openai", "model": model, "requested_model": model,
            "prompt_version": "dns-chart-v1", "prompt_sha256": "a" * 64,
            "generated_at": "2026-11-06T03:00:00Z", "system_fingerprint": "fp-test",
            "settings": {"temperature": 0.1, "image_detail": "high"}
        }
        record = {
            "fixture_id": fixture_id, "image_file": "chart.png",
            "image_sha256": self.image_hash, "image_width": 40, "image_height": 30,
            "reference_zones": reference if reference is not None else [_zone()],
            "predicted_zones": predicted, "prediction_metadata": metadata,
            "annotation_metadata": _annotation(),
            "market_context": context if context is not None else self.context,
            "iou_threshold": threshold
        }
        if anchors is not None:
            record["price_axis_anchors"] = anchors
            record["price_axis_scale"] = "linear"
        manifest = self.root / f"{model}-{fixture_id}.jsonl"
        manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return evaluate_benchmark_manifest(manifest)

    def test_compares_models_on_identical_reviewed_fixtures(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [])
        result = compare_benchmark_reports(baseline, candidate)
        self.assertEqual(result["fixture_count"], 1)
        self.assertFalse(result["same_prediction_profile"])
        self.assertEqual(result["overall"]["recall"]["delta"], -1.0)
        self.assertEqual(result["overall"]["false_negative"]["delta"], 1)
        self.assertEqual(result["by_symbol"][0]["value"], "BTCUSDT")

    def test_rejects_different_screenshot_hashes(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [])
        candidate["fixtures"][0]["image_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "image_sha256"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_different_reference_annotations(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [], reference=[_zone(y_top=0.42)])
        with self.assertRaisesRegex(ValueError, "reference_zones_sha256"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_different_price_axis_calibration(self):
        anchors_a = [{"y": 0.2, "price": 120}, {"y": 0.5, "price": 90}, {"y": 0.8, "price": 60}]
        anchors_b = [{"y": 0.2, "price": 125}, {"y": 0.5, "price": 90}, {"y": 0.8, "price": 60}]
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}], anchors=anchors_a)
        candidate = self._report("candidate", [{**_zone(), "draw_overlay": True}], anchors=anchors_b)
        with self.assertRaisesRegex(ValueError, "price_axis_calibration_sha256"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_different_iou_thresholds(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}], threshold=0.5)
        candidate = self._report("candidate", [{**_zone(), "draw_overlay": True}], threshold=0.6)
        with self.assertRaisesRegex(ValueError, "iou_threshold"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_different_fixture_sets(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [{**_zone(), "draw_overlay": True}], fixture_id="other")
        with self.assertRaisesRegex(ValueError, "exactly the same fixture IDs"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_different_market_context(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [], context={**self.context, "timeframe": "4h"})
        with self.assertRaisesRegex(ValueError, "market_context"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_duplicate_fixture_ids(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [])
        candidate["fixtures"].append(dict(candidate["fixtures"][0]))
        candidate["fixture_count"] = 2
        with self.assertRaisesRegex(ValueError, "duplicate fixture_id"):
            compare_benchmark_reports(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
