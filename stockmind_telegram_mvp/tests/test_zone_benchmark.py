import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from stockmind_telegram_mvp.app.zone_benchmark import evaluate_benchmark_manifest


def _png_bytes(size=(40, 30)):
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format="PNG")
    return output.getvalue()


def _zone(zone_type="demand", **overrides):
    value = {
        "zone_type": zone_type,
        "x_start": 0.2,
        "x_end": 0.6,
        "y_top": 0.4,
        "y_bottom": 0.6,
    }
    value.update(overrides)
    return value


def _prediction_metadata(**overrides):
    value = {
        "provider": "openai",
        "model": "vision-model-2026-11-05",
        "requested_model": "vision-model",
        "prompt_version": "dns-chart-v1",
        "prompt_sha256": "a" * 64,
        "generated_at": "2026-11-05T20:24:00Z",
        "system_fingerprint": "fp-test",
        "settings": {
            "temperature": 0.1,
            "image_detail": "high",
            "response_format": "json_schema:chart_zone_map",
        },
    }
    value.update(overrides)
    return value


def _prediction(zone_type="demand", **overrides):
    value = _zone(zone_type, **overrides)
    value["draw_overlay"] = True
    return value


class ZoneBenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_bytes = _png_bytes()
        (self.root / "chart.png").write_bytes(self.image_bytes)
        self.manifest = self.root / "fixtures.jsonl"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _record(self, fixture_id="chart-001", **overrides):
        record = {
            "fixture_id": fixture_id,
            "image_file": "chart.png",
            "image_sha256": hashlib.sha256(self.image_bytes).hexdigest(),
            "image_width": 40,
            "image_height": 30,
            "reference_zones": [_zone()],
            "predicted_zones": [_prediction()],
            "prediction_metadata": _prediction_metadata(),
        }
        record.update(overrides)
        return record

    def _write(self, *records):
        self.manifest.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

    def test_scores_multiple_fixtures_and_aggregates_micro_metrics(self):
        self._write(
            self._record("exact"),
            self._record("missed", predicted_zones=[]),
        )

        result = evaluate_benchmark_manifest(self.manifest)

        self.assertEqual(result["fixture_count"], 2)
        self.assertEqual(result["overall"]["true_positive"], 1)
        self.assertEqual(result["overall"]["false_positive"], 0)
        self.assertEqual(result["overall"]["false_negative"], 1)
        self.assertEqual(result["overall"]["precision"], 1.0)
        self.assertEqual(result["overall"]["recall"], 0.5)
        self.assertAlmostEqual(result["overall"]["f1"], 2 / 3)
        self.assertEqual(result["overall"]["mean_iou"], 1.0)
        self.assertEqual(result["by_zone_type"]["demand"]["reference"], 2)
        self.assertEqual(result["prediction_profile"]["model"], "vision-model-2026-11-05")
        self.assertEqual(len(result["prediction_profile_sha256"]), 64)
        self.assertEqual(
            result["fixtures"][0]["prediction_metadata"]["generated_at"],
            "2026-11-05T20:24:00Z",
        )

    def test_allows_different_generation_times_for_the_same_profile(self):
        self._write(
            self._record("first"),
            self._record(
                "second",
                prediction_metadata=_prediction_metadata(
                    generated_at="2026-11-05T20:25:00+00:00"
                ),
            ),
        )

        result = evaluate_benchmark_manifest(self.manifest)

        self.assertEqual(result["fixture_count"], 2)
        self.assertEqual(
            result["fixtures"][1]["prediction_metadata"]["generated_at"],
            "2026-11-05T20:25:00+00:00",
        )

    def test_rejects_mixed_prediction_profiles_in_one_manifest(self):
        self._write(
            self._record("first"),
            self._record(
                "second",
                prediction_metadata=_prediction_metadata(model="another-model"),
            ),
        )

        with self.assertRaisesRegex(ValueError, "mixed prediction profiles"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_timezone_naive_generation_timestamp(self):
        self._write(
            self._record(
                prediction_metadata=_prediction_metadata(
                    generated_at="2026-11-05T20:24:00"
                )
            )
        )

        with self.assertRaisesRegex(ValueError, "must include a timezone"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_non_finite_model_settings(self):
        self._write(
            self._record(
                prediction_metadata=_prediction_metadata(
                    settings={"temperature": float("nan")}
                )
            )
        )

        with self.assertRaisesRegex(ValueError, "finite JSON values"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_missing_prediction_metadata(self):
        record = self._record()
        del record["prediction_metadata"]
        self._write(record)

        with self.assertRaisesRegex(ValueError, "prediction_metadata"):
            evaluate_benchmark_manifest(self.manifest)

    def test_price_axis_metrics_are_aggregated_without_inventing_missing_values(self):
        self._write(
            self._record(
                price_axis_anchors=[
                    {"y": 0.2, "price": 120},
                    {"y": 0.5, "price": 90},
                    {"y": 0.8, "price": 60},
                ],
                price_axis_scale="linear",
            )
        )

        result = evaluate_benchmark_manifest(self.manifest)

        self.assertEqual(result["price_accuracy"]["calibrated_fixture_count"], 1)
        self.assertEqual(result["price_accuracy"]["scored_zone_count"], 1)
        self.assertEqual(result["price_accuracy"]["unscored_zone_count"], 0)
        self.assertEqual(result["price_accuracy"]["mean_boundary_abs_error"], 0.0)

    def test_prediction_below_renderer_pixel_minimum_is_a_missed_zone(self):
        self._write(
            self._record(
                predicted_zones=[
                    _prediction(x_start=0.2, x_end=0.25)
                ]
            )
        )

        result = evaluate_benchmark_manifest(self.manifest)

        self.assertEqual(result["overall"]["predicted"], 0)
        self.assertEqual(result["overall"]["true_positive"], 0)
        self.assertEqual(result["overall"]["false_negative"], 1)
        self.assertEqual(result["ignored_prediction_count"], 1)

    def test_rejects_screenshot_larger_than_configured_byte_limit(self):
        self._write(self._record())

        with self.assertRaisesRegex(ValueError, "exceeds max_image_bytes"):
            evaluate_benchmark_manifest(
                self.manifest, max_image_bytes=len(self.image_bytes) - 1
            )

    def test_rejects_non_positive_image_byte_limit(self):
        self._write(self._record())

        with self.assertRaisesRegex(ValueError, "max_image_bytes must be a positive integer"):
            evaluate_benchmark_manifest(self.manifest, max_image_bytes=0)

    def test_rejects_screenshot_hash_mismatch(self):
        self._write(self._record(image_sha256="0" * 64))

        with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_screenshot_dimension_mismatch(self):
        self._write(self._record(image_width=41))

        with self.assertRaisesRegex(ValueError, "dimensions"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_paths_that_escape_manifest_directory(self):
        self._write(self._record(image_file="../chart.png"))

        with self.assertRaisesRegex(ValueError, "inside the manifest directory"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_duplicate_fixture_ids(self):
        self._write(self._record("duplicate"), self._record("duplicate"))

        with self.assertRaisesRegex(ValueError, "duplicate fixture_id"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_malformed_jsonl(self):
        self.manifest.write_text('{"fixture_id":\n', encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_empty_manifest_instead_of_reporting_fake_zero_accuracy(self):
        self.manifest.write_text("\n  \n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "contains no fixture records"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_axis_scale_without_manual_anchors(self):
        self._write(self._record(price_axis_scale="log"))

        with self.assertRaisesRegex(ValueError, "requires price_axis_anchors"):
            evaluate_benchmark_manifest(self.manifest)


if __name__ == "__main__":
    unittest.main()
