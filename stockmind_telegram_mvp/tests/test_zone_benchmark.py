import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from stockmind_telegram_mvp.app.zone_benchmark import (
    _aggregate_price_accuracy,
    evaluate_benchmark_manifest,
)


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


def _adjudication_metadata(fixture_id, image_hash, width, height, context, annotation):
    annotation_a = {
        "file_sha256": "c" * 64,
        "annotator_id": annotation["annotator_id"],
        "reviewer_id": annotation["reviewer_id"],
        "reviewed_at": annotation["reviewed_at"],
    }
    annotation_b = {
        "file_sha256": "d" * 64,
        "annotator_id": "ann-003",
        "reviewer_id": "rev-004",
        "reviewed_at": "2026-11-05T19:20:00Z",
    }
    report = {
        "report_schema": "stockmind.annotation-agreement.v1",
        "fixture_id": fixture_id,
        "image_sha256": image_hash,
        "image_width": width,
        "image_height": height,
        "market_context": context,
        "guideline_version": annotation["guideline_version"],
        "annotator_ids": [annotation_a["annotator_id"], annotation_b["annotator_id"]],
        "reviewer_ids": [annotation_a["reviewer_id"], annotation_b["reviewer_id"]],
    }
    report_hash = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "stockmind.annotation-adjudication.v1",
        "decision": "accept_a",
        "selected_annotation": "a",
        "adjudicator_id": "adj-999",
        "adjudicated_at": "2026-11-05T19:30:00Z",
        "rationale": "The selected annotation follows the agreed zone-boundary guideline after review.",
        "annotation_a": annotation_a,
        "annotation_b": annotation_b,
        "agreement_report": report,
        "agreement_report_sha256": report_hash,
    }


def _market_context(**overrides):
    value = {
        "asset_class": "crypto",
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "timeframe": "1h",
    }
    value.update(overrides)
    return value


def _normalized_market_context(context):
    fields = ("asset_class", "symbol", "exchange", "timeframe")
    if context is None:
        return {field: None for field in fields}
    if not isinstance(context, dict):
        return context
    normalized = {}
    for field in fields:
        value = context.get(field)
        if isinstance(value, str):
            cleaned = value.strip()
            normalized[field] = cleaned.upper() if field == "symbol" else cleaned.casefold()
        else:
            normalized[field] = value
    return normalized


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

    def _fixture_image(self, fixture_id):
        """Create deterministic, distinct screenshot bytes for each fixture ID."""
        digest = hashlib.sha256(fixture_id.encode("utf-8")).digest()
        image = Image.new("RGB", (40, 30), "white")
        for index, value in enumerate(digest):
            image.putpixel(
                (index, 0),
                (value, (value + 53) % 256, (value + 107) % 256),
            )
        output = BytesIO()
        image.save(output, format="PNG")
        image_bytes = output.getvalue()
        image_file = f"chart-{digest.hex()[:12]}.png"
        (self.root / image_file).write_bytes(image_bytes)
        return image_file, image_bytes

    def _record(self, fixture_id="chart-001", **overrides):
        image_file, image_bytes = self._fixture_image(fixture_id)
        record = {
            "fixture_id": fixture_id,
            "image_file": image_file,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "image_width": 40,
            "image_height": 30,
            "reference_zones": [_zone()],
            "predicted_zones": [_prediction()],
            "prediction_metadata": _prediction_metadata(),
            "annotation_metadata": _annotation_metadata(),
        }
        record.update(overrides)
        record.setdefault("market_context", None)
        # Only ordinary fixtures get a convenient full-image plot. Preserve
        # one-sided explicit bounds so tests can exercise missing-model and
        # missing-manual-reference validation paths.
        explicit_plot_fields = {
            "reference_plot_bounds", "predicted_plot_bounds", "price_axis_plot_bounds"
        }
        if not explicit_plot_fields.intersection(overrides):
            full_image = {
                "x_start": 0.0, "x_end": 1.0, "y_top": 0.0, "y_bottom": 1.0
            }
            record["reference_plot_bounds"] = full_image
            record["predicted_plot_bounds"] = full_image
        elif (
            "reference_plot_bounds" not in record
            and "price_axis_plot_bounds" in record
        ):
            record["reference_plot_bounds"] = record["price_axis_plot_bounds"]
        if "adjudication_metadata" not in overrides:
            record["adjudication_metadata"] = _adjudication_metadata(
                fixture_id,
                record["image_sha256"],
                record["image_width"],
                record["image_height"],
                _normalized_market_context(record["market_context"]),
                record["annotation_metadata"],
            )
        return record

    def _refresh_adjudication_image_hash(self, record):
        report = record["adjudication_metadata"]["agreement_report"]
        report["image_sha256"] = record["image_sha256"]
        canonical = json.dumps(
            report, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        record["adjudication_metadata"]["agreement_report_sha256"] = hashlib.sha256(
            canonical
        ).hexdigest()

    def _write(self, *records):
        self.manifest.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )


    def test_report_aggregates_confidence_strata_without_scoring_suppressed_low_zones(self):
        self._write(
            self._record("drawable"),
            self._record("empty", predicted_zones=[]),
        )

        result = evaluate_benchmark_manifest(self.manifest)
        summary = result["confidence_stratified_results"]["not_reported_or_invalid"]

        self.assertEqual(result["fixture_count"], 2)
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["drawable_count"], 1)
        self.assertEqual(summary["matched_count"], 1)
        self.assertEqual(summary["precision"], 1.0)
        self.assertEqual(summary["fixture_count_with_candidates"], 1)
        self.assertEqual(summary["fixture_count_with_drawable_predictions"], 1)
        self.assertEqual(summary["fixture_count_with_matches"], 1)
        empty_summary = result["fixtures"][1]["confidence_stratified_results"]["not_reported_or_invalid"]
        self.assertEqual(empty_summary["fixture_count_with_candidates"], 0)
        self.assertEqual(empty_summary["fixture_count_with_drawable_predictions"], 0)
        self.assertEqual(empty_summary["fixture_count_with_matches"], 0)
        self.assertEqual(result["confidence_stratified_results"]["low"]["candidate_count"], 0)
        self.assertEqual(
            result["fixtures"][0]["confidence_stratified_results"]["not_reported_or_invalid"]["matched_count"],
            1,
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

    def test_macro_metrics_weight_each_chart_equally_not_each_zone(self):
        dense_zones = [
            _zone(x_start=0.05 + index * 0.11, x_end=0.13 + index * 0.11)
            for index in range(8)
        ]
        dense_predictions = [
            _prediction(x_start=zone["x_start"], x_end=zone["x_end"])
            for zone in dense_zones
        ]
        self._write(
            self._record(
                "dense-chart",
                reference_zones=dense_zones,
                predicted_zones=dense_predictions,
            ),
            self._record("missed-chart", predicted_zones=[]),
        )

        result = evaluate_benchmark_manifest(self.manifest)

        # Micro recall weights all nine reference zones equally. Macro recall
        # gives the dense and missed charts equal weight.
        self.assertAlmostEqual(result["overall"]["recall"], 8 / 9)
        self.assertAlmostEqual(result["macro_overall"]["recall"], 0.5)
        self.assertEqual(result["macro_overall"]["fixture_count"], 2)
        self.assertEqual(result["macro_overall"]["recall_fixture_count"], 2)
        self.assertAlmostEqual(result["macro_by_zone_type"]["demand"]["recall"], 0.5)
        self.assertAlmostEqual(result["macro_by_zone_type"]["demand"]["f1"], 0.5)

    def test_macro_metrics_exclude_undefined_charts_and_report_coverage(self):
        self._write(
            self._record(
                "empty-chart",
                reference_zones=[],
                predicted_zones=[],
            ),
            self._record("scored-chart"),
        )

        result = evaluate_benchmark_manifest(self.manifest)
        macro = result["macro_overall"]

        self.assertEqual(macro["fixture_count"], 2)
        expected = {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "mean_iou": 1.0,
            "mean_center_error_normalized": 0.0,
        }
        for field, value in expected.items():
            self.assertAlmostEqual(macro[field], value)
            self.assertEqual(macro[f"{field}_fixture_count"], 1)
            self.assertAlmostEqual(macro[f"{field}_coverage"], 0.5)

        supply = result["macro_by_zone_type"]["supply"]
        self.assertEqual(supply["fixture_count"], 2)
        self.assertIsNone(supply["precision"])
        self.assertEqual(supply["precision_fixture_count"], 0)
        self.assertEqual(supply["precision_coverage"], 0.0)

        demand = result["macro_by_zone_type"]["demand"]
        self.assertEqual(demand["fixture_count"], 2)
        self.assertEqual(demand["recall_fixture_count"], 1)
        self.assertEqual(demand["recall_coverage"], 0.5)

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

        with self.assertRaisesRegex(ValueError, "non-standard JSON constant"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_missing_prediction_metadata(self):
        record = self._record()
        del record["prediction_metadata"]
        self._write(record)

        with self.assertRaisesRegex(ValueError, "prediction_metadata"):
            evaluate_benchmark_manifest(self.manifest)


    def test_rejects_missing_independent_adjudication_metadata(self):
        record = self._record()
        del record["adjudication_metadata"]
        self._write(record)
        with self.assertRaisesRegex(ValueError, "adjudication_metadata"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_tampered_pairwise_agreement_report(self):
        record = self._record()
        record["adjudication_metadata"]["agreement_report"]["fixture_id"] = "other-chart"
        record["adjudication_metadata"]["agreement_report_sha256"] = hashlib.sha256(
            json.dumps(
                record["adjudication_metadata"]["agreement_report"],
                sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        self._write(record)
        with self.assertRaisesRegex(ValueError, "agreement_report.fixture_id"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_adjudicator_reusing_an_annotation_identity(self):
        record = self._record()
        record["adjudication_metadata"]["adjudicator_id"] = "adj-001"
        self._write(record)
        with self.assertRaisesRegex(ValueError, "distinct from both annotators"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_same_reviewer_for_both_annotations(self):
        record = self._record()
        record["adjudication_metadata"]["annotation_b"]["reviewer_id"] = "rev-002"
        self._write(record)
        with self.assertRaisesRegex(ValueError, "two distinct reviewer identities"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_second_reviewer_reusing_own_annotator_identity(self):
        record = self._record()
        record["adjudication_metadata"]["annotation_b"]["reviewer_id"] = "rev-003"
        self._write(record)
        with self.assertRaisesRegex(ValueError, "each reviewer identity must differ from its own annotator"):
            evaluate_benchmark_manifest(self.manifest)


    def test_rejects_malformed_adjudication_decision_without_type_error(self):
        for decision in ([], {}):
            with self.subTest(decision=decision):
                record = self._record()
                record["adjudication_metadata"]["decision"] = decision
                self._write(record)
                with self.assertRaisesRegex(ValueError, "decision must be accept_a or accept_b"):
                    evaluate_benchmark_manifest(self.manifest)

    def test_rejects_adjudication_timestamp_before_both_reviews(self):
        record = self._record()
        record["adjudication_metadata"]["adjudicated_at"] = "2026-11-05T19:17:00Z"
        self._write(record)
        with self.assertRaisesRegex(ValueError, "cannot precede either annotation review"):
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

    def test_rejects_reviewer_reusing_annotator_pseudonym_suffix(self):
        for annotator_id, reviewer_id in (
            ("ann-001", "rev-001"),
            ("ann-Case", "rev-case"),
        ):
            with self.subTest(annotator_id=annotator_id, reviewer_id=reviewer_id):
                self._write(
                    self._record(
                        annotation_metadata=_annotation_metadata(
                            annotator_id=annotator_id,
                            reviewer_id=reviewer_id,
                        )
                    )
                )
                with self.assertRaisesRegex(
                    ValueError, "different pseudonymous identity token"
                ):
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


    def test_reported_price_consistency_is_aggregated_separately_from_geometry_error(self):
        self._write(self._record(
            predicted_zones=[_prediction(price_low="80", price_high="100")],
            price_axis_anchors=[
                {"y": 0.2, "price": 120},
                {"y": 0.5, "price": 90},
                {"y": 0.8, "price": 60},
            ],
            price_axis_scale="linear",
            price_axis_plot_bounds={
                "x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9
            },
            reference_plot_bounds={
                "x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9
            },
            predicted_plot_bounds={
                "x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9
            },
        ))

        result = evaluate_benchmark_manifest(self.manifest)
        consistency = result["price_accuracy"]["reported_price_consistency"]
        self.assertEqual(consistency["scored_match_count"], 1)
        self.assertEqual(consistency["unscored_match_count"], 0)
        self.assertEqual(consistency["mean_boundary_abs_disagreement"], 0.0)
        self.assertEqual(
            result["fixtures"][0]["price_accuracy"]["reported_price_consistency"][
                "scored_match_count"
            ],
            1,
        )

    def test_price_accuracy_aggregate_uses_overflow_safe_means(self):
        comparison = {
            "status": "scored",
            "price_high_abs_error": 1e308,
            "price_low_abs_error": 1e308,
            "price_high_error_pct": 142.85714285714286,
            "price_low_error_pct": 166.66666666666666,
        }
        fixtures = [{
            "price_accuracy": {},
            "matches": [
                {
                    "price_comparison": dict(comparison),
                    "reported_price_consistency": {
                        "status": "unscored",
                        "reason": "reported labels unavailable",
                    },
                },
                {
                    "price_comparison": dict(comparison),
                    "reported_price_consistency": {
                        "status": "unscored",
                        "reason": "reported labels unavailable",
                    },
                },
            ],
        }]

        result = _aggregate_price_accuracy(fixtures)

        self.assertEqual(result["mean_boundary_abs_error"], 1e308)
        self.assertEqual(result["max_boundary_abs_error"], 1e308)
        self.assertAlmostEqual(
            result["mean_boundary_error_pct"],
            (142.85714285714286 + 166.66666666666666) / 2,
        )

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

    def test_predictions_without_model_plot_bounds_are_ignored_not_scored(self):
        self._write(self._record(predicted_plot_bounds=None))

        result = evaluate_benchmark_manifest(self.manifest)

        self.assertEqual(result["overall"]["predicted"], 0)
        self.assertEqual(result["overall"]["true_positive"], 0)
        self.assertEqual(result["overall"]["false_negative"], 1)
        self.assertEqual(result["ignored_prediction_count"], 1)
        self.assertIsNone(result["fixtures"][0]["plot_boundary_accuracy"])

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

    def test_rejects_noncanonical_exif_oriented_screenshot(self):
        record = self._record("exif-rotated")
        image_file = self.root / record["image_file"].replace(".png", ".jpg")
        exif = Image.Exif()
        exif[274] = 6
        output = BytesIO()
        Image.new("RGB", (40, 30), "white").save(
            output, format="JPEG", quality=95, exif=exif
        )
        image_bytes = output.getvalue()
        image_file.write_bytes(image_bytes)
        record["image_file"] = image_file.name
        record["image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
        self._refresh_adjudication_image_hash(record)
        # The encoded dimensions remain 40x30, but the canonical visual frame
        # becomes 30x40 after EXIF orientation is applied.
        self._write(record)

        with self.assertRaisesRegex(ValueError, "not EXIF-normalized"):
            evaluate_benchmark_manifest(self.manifest)

    def test_accepts_exif_orientation_one_without_reencoding(self):
        record = self._record("exif-normal")
        image_file = self.root / record["image_file"].replace(".png", ".jpg")
        exif = Image.Exif()
        exif[274] = 1
        output = BytesIO()
        Image.new("RGB", (40, 30), "white").save(
            output, format="JPEG", quality=95, exif=exif
        )
        image_bytes = output.getvalue()
        image_file.write_bytes(image_bytes)
        record["image_file"] = image_file.name
        record["image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
        self._refresh_adjudication_image_hash(record)
        self._write(record)

        result = evaluate_benchmark_manifest(self.manifest)

        self.assertEqual(result["fixture_count"], 1)
        self.assertEqual(result["overall"]["true_positive"], 1)

    def test_rejects_paths_that_escape_manifest_directory(self):
        self._write(self._record(image_file="../chart.png"))

        with self.assertRaisesRegex(ValueError, "inside the manifest directory"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_duplicate_fixture_ids(self):
        self._write(self._record("duplicate"), self._record("duplicate"))

        with self.assertRaisesRegex(ValueError, "duplicate fixture_id"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_duplicate_screenshot_hashes_with_distinct_paths_and_ids(self):
        first = self._record("chart-copy-a")
        first_bytes = (self.root / first["image_file"]).read_bytes()
        (self.root / "chart-copy.png").write_bytes(first_bytes)
        second = self._record(
            "chart-copy-b",
            image_file="chart-copy.png",
            image_sha256=first["image_sha256"],
            image_width=first["image_width"],
            image_height=first["image_height"],
        )
        self._write(first, second)

        with self.assertRaisesRegex(ValueError, "duplicate screenshot image_sha256"):
            evaluate_benchmark_manifest(self.manifest)

    def _v6_metadata(self, include_policy=True):
        settings = {
            "temperature": 0.1,
            "image_detail": "high",
            "response_format": "json_schema:chart_zone_map",
            "conversation_context": "caption_only",
            "plot_bounds_policy": "required_for_overlay",
        }
        if include_policy:
            settings["confidence_policy"] = "high_medium_only"
        return _prediction_metadata(prompt_version="dns-chart-v6", settings=settings)

    def test_v6_metadata_requires_confidence_policy(self):
        record = self._record(prediction_metadata=self._v6_metadata(include_policy=False))
        self._write(record)
        with self.assertRaisesRegex(ValueError, "confidence_policy"):
            evaluate_benchmark_manifest(self.manifest)

    def test_v6_manifest_rejects_missing_zone_confidence(self):
        record = self._record(prediction_metadata=self._v6_metadata())
        self._write(record)
        with self.assertRaisesRegex(ValueError, r"predicted_zones\[0\].confidence"):
            evaluate_benchmark_manifest(self.manifest)

    def test_v6_manifest_rejects_drawable_low_confidence_zone(self):
        record = self._record(
            prediction_metadata=self._v6_metadata(),
            predicted_zones=[_prediction(confidence="low")],
        )
        self._write(record)
        with self.assertRaisesRegex(ValueError, "cannot draw a low-confidence"):
            evaluate_benchmark_manifest(self.manifest)

    def test_v6_manifest_rejects_nonzero_coordinates_for_non_drawable_zone(self):
        low = _prediction(confidence="low")
        low["draw_overlay"] = False
        record = self._record(
            prediction_metadata=self._v6_metadata(), predicted_zones=[low]
        )
        self._write(record)
        with self.assertRaisesRegex(ValueError, "must zero all coordinates"):
            evaluate_benchmark_manifest(self.manifest)

    def test_v6_manifest_accepts_sanitized_confidence_gate(self):
        high = _prediction(confidence="high")
        low = _prediction(confidence="low")
        low.update({
            "draw_overlay": False,
            "x_start": 0.0, "x_end": 0.0, "y_top": 0.0, "y_bottom": 0.0,
        })
        record = self._record(
            prediction_metadata=self._v6_metadata(),
            predicted_zones=[high, low],
        )
        self._write(record)
        result = evaluate_benchmark_manifest(self.manifest)
        self.assertEqual(result["fixture_count"], 1)
        self.assertEqual(result["overall"]["true_positive"], 1)
        self.assertEqual(result["ignored_prediction_count"], 1)

    def _v7_metadata(self, include_origin_policy=True):
        settings = {
            "temperature": 0.1,
            "image_detail": "high",
            "response_format": "json_schema:chart_zone_map",
            "conversation_context": "caption_only",
            "plot_bounds_policy": "required_for_overlay",
            "confidence_policy": "high_medium_only",
        }
        if include_origin_policy:
            settings["zone_origin_policy"] = "user_marked_only_when_visibly_marked"
        return _prediction_metadata(prompt_version="dns-chart-v7", settings=settings)

    def _v8_metadata(self, include_evidence_policy=True, include_origin_policy=True):
        metadata = self._v7_metadata(include_origin_policy=include_origin_policy)
        metadata["prompt_version"] = "dns-chart-v8"
        if include_evidence_policy:
            metadata["settings"]["zone_evidence_policy"] = (
                "compact_base_plus_directional_displacement_v1"
            )
        return metadata

    def test_v7_metadata_requires_explicit_origin_policy(self):
        record = self._record(prediction_metadata=self._v7_metadata(include_origin_policy=False))
        self._write(record)
        with self.assertRaisesRegex(ValueError, "zone_origin_policy"):
            evaluate_benchmark_manifest(self.manifest)

    def test_v7_manifest_rejects_missing_zone_origin(self):
        record = self._record(
            prediction_metadata=self._v7_metadata(),
            predicted_zones=[_prediction(confidence="high")],
        )
        self._write(record)
        with self.assertRaisesRegex(ValueError, r"predicted_zones\[0\].origin"):
            evaluate_benchmark_manifest(self.manifest)

    def test_v7_manifest_rejects_label_that_conflicts_with_origin(self):
        zone = _prediction(confidence="high", origin="ai_inferred", label="USER DEMAND")
        record = self._record(
            prediction_metadata=self._v7_metadata(),
            predicted_zones=[zone],
        )
        self._write(record)
        with self.assertRaisesRegex(ValueError, "label must match its origin"):
            evaluate_benchmark_manifest(self.manifest)

    def test_v7_manifest_accepts_derived_user_and_ai_labels(self):
        zones = [
            _prediction(confidence="high", origin="ai_inferred", label="AI DEMAND"),
            _prediction("supply", confidence="medium", origin="user_marked", label="USER SUPPLY"),
        ]
        record = self._record(
            prediction_metadata=self._v7_metadata(),
            predicted_zones=zones,
        )
        self._write(record)
        result = evaluate_benchmark_manifest(self.manifest)
        self.assertEqual(result["fixture_count"], 1)
        self.assertEqual(result["overall"]["predicted"], 2)
        self.assertEqual(result["overall"]["true_positive"], 1)
        self.assertEqual(result["overall"]["false_positive"], 1)


    def test_v8_metadata_requires_explicit_zone_evidence_policy(self):
        record = self._record(
            prediction_metadata=self._v8_metadata(include_evidence_policy=False)
        )
        self._write(record)
        with self.assertRaisesRegex(ValueError, "zone_evidence_policy"):
            evaluate_benchmark_manifest(self.manifest)

    def test_v8_manifest_rejects_missing_zone_origin(self):
        record = self._record(
            prediction_metadata=self._v8_metadata(),
            predicted_zones=[_prediction(confidence="high")],
        )
        self._write(record)
        with self.assertRaisesRegex(ValueError, r"predicted_zones\[0\].origin"):
            evaluate_benchmark_manifest(self.manifest)

    def test_v8_manifest_rejects_label_that_conflicts_with_origin(self):
        zone = _prediction(confidence="high", origin="ai_inferred", label="USER DEMAND")
        record = self._record(
            prediction_metadata=self._v8_metadata(), predicted_zones=[zone]
        )
        self._write(record)
        with self.assertRaisesRegex(ValueError, "label must match its origin"):
            evaluate_benchmark_manifest(self.manifest)

    def test_v8_manifest_accepts_evidence_policy_and_consistent_zone_provenance(self):
        zone = _prediction(confidence="high", origin="ai_inferred", label="AI DEMAND")
        record = self._record(
            prediction_metadata=self._v8_metadata(), predicted_zones=[zone]
        )
        self._write(record)
        result = evaluate_benchmark_manifest(self.manifest)
        self.assertEqual(result["fixture_count"], 1)
        self.assertEqual(result["overall"]["predicted"], 1)

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


    def test_rejects_duplicate_object_keys_in_jsonl_manifest(self):
        record = self._record("duplicate-json-key")
        raw = json.dumps(record).replace(
            '"fixture_id": "duplicate-json-key"',
            '"fixture_id": "shadow", "fixture_id": "duplicate-json-key"',
            1,
        )
        self.manifest.write_text(raw + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key 'fixture_id'"):
            evaluate_benchmark_manifest(self.manifest)

    def test_rejects_non_standard_numeric_constants_in_jsonl_manifest(self):
        record = self._record("nonfinite-json")
        raw = json.dumps(record).replace('"image_width": 40', '"image_width": NaN', 1)
        self.manifest.write_text(raw + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-standard JSON constant 'NaN'"):
            evaluate_benchmark_manifest(self.manifest)

if __name__ == "__main__":
    unittest.main()
