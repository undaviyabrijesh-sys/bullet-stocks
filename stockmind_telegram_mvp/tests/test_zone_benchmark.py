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


def _annotation_metadata(**overrides):
    value = {
        "source": "manual",
        "annotator_id": "ann-001",
        "reviewer_id": "rev-002",
        "guideline_version": "dns-manual-v1",
        "review_status": "approved",
        "annotated_at": "2026-11-05T19:00:00Z",
        "reviewed_at": "2026-11-05T19:15:00Z",
    }
    value.update(overrides)
    return value


def _market_context(**overrides):
    value = {
        "asset_class": "crypto",
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "timeframe": "1h",
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
            "annotation_metadata": _annotation_metadata(),
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
        self.assertEqual(result["annotation_guideline_version"], "dns-manual-v1")
        self.assertEqual(len(result["fixtures"][0]["reference_zones_sha256"]), 64)
        self.assertIsNone(result["fixtures"][0]["price_axis_calibration_sha256"])
        self.assertEqual(result["fixtures"][0]["iou_threshold"], 0.5)
        self.assertEqual(
            result["fixtures"][0]["annotation_metadata"]["reviewer_id"], "rev-002"
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

    def test_rejects_non_manual_annotation_source(self):
        self._write(
            self._record(
                annotation_metadata=_annotation_metadata(source="ai_generated")
            )
        )

        with self.assertRaisesRegex(ValueError, "source must be 'manual'"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_annotations_that_are_not_approved(self):
        self._write(
            self._record(
                annotation_metadata=_annotation_metadata(review_status="pending")
            )
        )

        with self.assertRaisesRegex(ValueError, "review_status must be 'approved'"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_non_anonymized_annotator_or_reviewer_ids(self):
        self._write(
            self._record(
                annotation_metadata=_annotation_metadata(
                    annotator_id="person@example.com"
                )
            )
        )

        with self.assertRaisesRegex(ValueError, "anonymized ID"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_reviewer_id_that_is_not_a_separate_reviewer_id(self):
        self._write(
            self._record(
                annotation_metadata=_annotation_metadata(reviewer_id="ann-001")
            )
        )

        with self.assertRaisesRegex(ValueError, "separate anonymized ID"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_annotation_metadata_with_unapproved_extra_fields(self):
        metadata = _annotation_metadata()
        metadata["reviewer_email"] = "reviewer@example.com"
        self._write(self._record(annotation_metadata=metadata))

        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_timezone_naive_annotation_timestamps(self):
        self._write(
            self._record(
                annotation_metadata=_annotation_metadata(
                    reviewed_at="2026-11-05T19:15:00"
                )
            )
        )

        with self.assertRaisesRegex(ValueError, "reviewed_at must include a timezone"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_review_timestamp_before_annotation_timestamp(self):
        self._write(
            self._record(
                annotation_metadata=_annotation_metadata(
                    annotated_at="2026-11-05T19:20:00Z",
                    reviewed_at="2026-11-05T19:15:00Z",
                )
            )
        )

        with self.assertRaisesRegex(ValueError, "cannot be earlier"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_mixed_annotation_guideline_versions(self):
        self._write(
            self._record("first"),
            self._record(
                "second",
                annotation_metadata=_annotation_metadata(
                    guideline_version="dns-manual-v2"
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "mixed annotation guideline versions"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_missing_annotation_provenance(self):
        record = self._record()
        del record["annotation_metadata"]
        self._write(record)

        with self.assertRaisesRegex(ValueError, "annotation_metadata"):
            evaluate_benchmark_manifest(self.manifest)

    def test_aggregates_metrics_by_asset_symbol_exchange_and_timeframe(self):
        self._write(
            self._record("btc-exact", market_context=_market_context()),
            self._record(
                "nifty-missed",
                predicted_zones=[],
                market_context=_market_context(
                    asset_class="index", symbol="NIFTY", exchange="nse",
                    timeframe="15m"
                ),
            ),
        )

        result = evaluate_benchmark_manifest(self.manifest)

        symbols = {item["value"]: item for item in result["by_symbol"]}
        self.assertEqual(symbols["BTCUSDT"]["fixture_count"], 1)
        self.assertEqual(symbols["BTCUSDT"]["metrics"]["recall"], 1.0)
        self.assertEqual(symbols["NIFTY"]["metrics"]["recall"], 0.0)
        self.assertEqual(
            {item["value"] for item in result["by_asset_class"]},
            {"crypto", "index"},
        )
        self.assertEqual(
            {item["value"] for item in result["by_exchange"]},
            {"binance", "nse"},
        )
        self.assertEqual(
            {item["value"] for item in result["by_timeframe"]},
            {"1h", "15m"},
        )

    def test_missing_market_context_is_kept_in_null_group(self):
        self._write(self._record())

        result = evaluate_benchmark_manifest(self.manifest)

        self.assertIsNone(result["by_symbol"][0]["value"])
        self.assertEqual(result["by_symbol"][0]["fixture_count"], 1)

    def test_normalizes_market_context_case_before_grouping(self):
        self._write(
            self._record(
                "mixed-case",
                market_context=_market_context(
                    asset_class=" Crypto ",
                    symbol=" btcusdt ",
                    exchange=" Binance ",
                    timeframe="1H",
                ),
            ),
            self._record("canonical-case", market_context=_market_context()),
        )

        result = evaluate_benchmark_manifest(self.manifest)

        for field in ("by_asset_class", "by_symbol", "by_exchange", "by_timeframe"):
            self.assertEqual(len(result[field]), 1)
            self.assertEqual(result[field][0]["fixture_count"], 2)
        self.assertEqual(result["by_symbol"][0]["value"], "BTCUSDT")
        self.assertEqual(result["by_exchange"][0]["value"], "binance")
        self.assertEqual(result["by_timeframe"][0]["value"], "1h")

    def test_rejects_oversized_or_control_character_market_context(self):
        cases = (
            ({"asset_class": "x" * 33}, "asset_class cannot exceed 32"),
            ({"symbol": "x" * 65}, "symbol cannot exceed 64"),
            ({"exchange": "x" * 65}, "exchange cannot exceed 64"),
            ({"timeframe": "x" * 17}, "timeframe cannot exceed 16"),
            ({"symbol": "BTC" + chr(10) + "USDT"}, "symbol cannot contain control characters"),
        )
        for context, expected in cases:
            with self.subTest(context=context):
                self._write(self._record(market_context=context))
                with self.assertRaisesRegex(ValueError, expected):
                    evaluate_benchmark_manifest(self.manifest)

    def test_rejects_invalid_market_context_shape_and_values(self):
        cases = (
            ([], "must be a JSON object"),
            ({"symbol": 42}, "symbol must be a non-empty string or null"),
            ({"symbol": "BTC", "account_id": "private"}, "unsupported fields"),
        )
        for context, expected in cases:
            with self.subTest(context=context):
                self._write(self._record(market_context=context))
                with self.assertRaisesRegex(ValueError, expected):
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
                price_axis_plot_bounds={"x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9},
                predicted_plot_bounds={"x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9},
            )
        )

        result = evaluate_benchmark_manifest(self.manifest)

        self.assertEqual(result["price_accuracy"]["calibrated_fixture_count"], 1)
        self.assertEqual(result["price_accuracy"]["scored_zone_count"], 1)
        self.assertEqual(result["price_accuracy"]["unscored_zone_count"], 0)
        self.assertEqual(result["price_accuracy"]["mean_boundary_abs_error"], 0.0)


    def test_price_axis_anchors_require_manual_plot_bounds(self):
        self._write(self._record(
            price_axis_anchors=[
                {"y": 0.2, "price": 120},
                {"y": 0.5, "price": 90},
                {"y": 0.8, "price": 60},
            ],
            price_axis_scale="linear",
        ))
        with self.assertRaisesRegex(ValueError, "price_axis_plot_bounds is required"):
            evaluate_benchmark_manifest(self.manifest)

    def test_price_axis_calibration_fingerprint_includes_plot_bounds(self):
        anchors = [
            {"y": 0.2, "price": 120},
            {"y": 0.5, "price": 90},
            {"y": 0.8, "price": 60},
        ]
        bounds_a = {"x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9}
        bounds_b = {"x_start": 0.1, "x_end": 0.9, "y_top": 0.15, "y_bottom": 0.9}
        self._write(
            self._record(
                "plot-a", price_axis_anchors=anchors,
                price_axis_plot_bounds=bounds_a, predicted_plot_bounds=bounds_a,
            ),
            self._record(
                "plot-b", price_axis_anchors=anchors,
                price_axis_plot_bounds=bounds_b, predicted_plot_bounds=bounds_b,
            ),
        )
        result = evaluate_benchmark_manifest(self.manifest)
        hashes = [fixture["price_axis_calibration_sha256"] for fixture in result["fixtures"]]
        self.assertNotEqual(hashes[0], hashes[1])

    def test_price_axis_anchors_outside_manual_plot_are_rejected(self):
        self._write(self._record(
            price_axis_anchors=[
                {"y": 0.05, "price": 130},
                {"y": 0.5, "price": 90},
                {"y": 0.8, "price": 60},
            ],
            price_axis_scale="linear",
            price_axis_plot_bounds={"x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9},
        ))
        with self.assertRaisesRegex(ValueError, "falls outside the manually marked main price plot"):
            evaluate_benchmark_manifest(self.manifest)

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

    def test_reports_plot_boundary_iou_and_zone_containment(self):
        plot = {
            "x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9
        }
        self._write(self._record(
            reference_plot_bounds=plot,
            predicted_plot_bounds=plot,
        ))

        result = evaluate_benchmark_manifest(self.manifest)
        fixture = result["fixtures"][0]
        self.assertEqual(fixture["plot_boundary_accuracy"]["status"], "scored")
        self.assertEqual(fixture["plot_boundary_accuracy"]["plot_iou"], 1.0)
        self.assertEqual(result["plot_boundary_accuracy"]["mean_plot_iou"], 1.0)
        self.assertEqual(fixture["image_width"], 40)
        self.assertEqual(len(fixture["reference_plot_bounds_sha256"]), 64)

    def test_missing_predicted_plot_boundary_is_counted_as_missing(self):
        plot = {
            "x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9
        }
        self._write(self._record(reference_plot_bounds=plot))

        result = evaluate_benchmark_manifest(self.manifest)
        self.assertEqual(
            result["fixtures"][0]["plot_boundary_accuracy"]["status"],
            "missing_prediction",
        )
        self.assertEqual(result["plot_boundary_accuracy"]["missing_prediction_count"], 1)

    def test_predicted_plot_boundary_requires_manual_reference(self):
        plot = {
            "x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9
        }
        self._write(self._record(predicted_plot_bounds=plot))

        with self.assertRaisesRegex(ValueError, "reference_plot_bounds is required"):
            evaluate_benchmark_manifest(self.manifest)


if __name__ == "__main__":
    unittest.main()
