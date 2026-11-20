import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from stockmind_telegram_mvp.app.zone_benchmark import evaluate_benchmark_manifest
from stockmind_telegram_mvp.app.zone_benchmark_compare import (
    _paired_telemetry_delta,
    _read_report,
    _telemetry_stat,
    _validate_price_accuracy,
    _validate_report,
    compare_benchmark_reports,
)


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


def _adjudication_metadata(fixture_id, image_hash, context, annotation):
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
        "image_width": 40,
        "image_height": 30,
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
                reference=None, context=None, anchors=None, threshold=0.5,
                reference_plot_bounds=None, predicted_plot_bounds=None,
                usage=None, request_latency_ms=None):
        metadata = {
            "provider": "openai", "model": model, "requested_model": model,
            "prompt_version": "dns-chart-v1", "prompt_sha256": "a" * 64,
            "generated_at": "2026-11-06T03:00:00Z", "system_fingerprint": "fp-test",
            "settings": {"temperature": 0.1, "image_detail": "high"}
        }
        if usage is not None:
            metadata["usage"] = usage
        if request_latency_ms is not None:
            metadata["request_latency_ms"] = request_latency_ms
        axis_bounds = {
            "x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9
        }
        if reference_plot_bounds is None:
            reference_plot_bounds = (
                axis_bounds if anchors is not None else
                {"x_start": 0.0, "x_end": 1.0, "y_top": 0.0, "y_bottom": 1.0}
            )
        if predicted_plot_bounds is None:
            predicted_plot_bounds = reference_plot_bounds
        normalized_predictions = []
        for zone in predicted:
            if isinstance(zone, dict):
                zone = dict(zone)
                # Most comparison fixtures exercise drawable predictions.
                # Preserve explicit None/low/invalid values for suppression tests.
                zone.setdefault("confidence", "high")
            normalized_predictions.append(zone)
        record = {
            "fixture_id": fixture_id, "image_file": "chart.png",
            "image_sha256": self.image_hash, "image_width": 40, "image_height": 30,
            "reference_zones": reference if reference is not None else [_zone()],
            "predicted_zones": normalized_predictions, "prediction_metadata": metadata,
            "annotation_metadata": _annotation(),
            "market_context": context if context is not None else self.context,
            "adjudication_metadata": _adjudication_metadata(
                fixture_id,
                self.image_hash,
                context if context is not None else self.context,
                _annotation(),
            ),
            "iou_threshold": threshold,
            "reference_plot_bounds": reference_plot_bounds,
            "predicted_plot_bounds": predicted_plot_bounds,
        }
        if anchors is not None:
            record["price_axis_anchors"] = anchors
            record["price_axis_plot_bounds"] = {
                "x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9
            }
            record["price_axis_scale"] = "linear"
        manifest = self.root / f"{model}-{fixture_id}.jsonl"
        manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return evaluate_benchmark_manifest(manifest)



    def test_rejects_tampered_adjudication_report_hash_in_benchmark_report(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [{**_zone(), "draw_overlay": True}])
        baseline["fixtures"][0]["adjudication_metadata"]["agreement_report_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "agreement_report_sha256 does not match"):
            compare_benchmark_reports(baseline, candidate)

    def test_paired_models_must_use_identical_adjudication_evidence(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [{**_zone(), "draw_overlay": True}])
        candidate["fixtures"][0]["adjudication_metadata"]["adjudicator_id"] = "adj-008"
        with self.assertRaisesRegex(ValueError, "paired reports differ.*adjudication_metadata"):
            compare_benchmark_reports(baseline, candidate)

    def test_compares_and_revalidates_confidence_stratified_results(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [])
        result = compare_benchmark_reports(baseline, candidate)

        confidence = result["confidence_stratified_results"]["high"]
        self.assertEqual(confidence["precision"]["baseline"], 1.0)
        self.assertIsNone(confidence["precision"]["candidate"])
        self.assertEqual(confidence["drawable_count"]["delta"], -1)
        self.assertEqual(confidence["fixture_count_with_candidates"]["baseline"], 1)
        self.assertEqual(confidence["fixture_count_with_candidates"]["candidate"], 0)
        self.assertEqual(confidence["fixture_count_with_candidates"]["delta"], -1)
        self.assertEqual(confidence["fixture_count_with_matches"]["baseline"], 1)
        self.assertEqual(confidence["fixture_count_with_matches"]["candidate"], 0)
        self.assertEqual(
            result["fixtures"][0]["confidence_stratified_results"]["high"][
                "matched_count"
            ]["baseline"],
            1,
        )

        baseline["fixtures"][0]["confidence_stratified_results"][
            "high"
        ]["matched_count"] = 0
        baseline["fixtures"][0]["confidence_stratified_results"][
            "high"
        ]["false_positive_count"] = 1
        baseline["fixtures"][0]["confidence_stratified_results"][
            "high"
        ]["fixture_count_with_matches"] = 0
        with self.assertRaisesRegex(ValueError, "matched count does not match match rows"):
            compare_benchmark_reports(baseline, candidate)

    def test_comparison_keeps_missing_confidence_predictions_suppressed(self):
        missing = {**_zone(), "draw_overlay": True, "confidence": None}
        baseline = self._report("baseline", [missing])
        candidate = self._report("candidate", [])
        result = compare_benchmark_reports(baseline, candidate)

        row = result["confidence_stratified_results"]["not_reported_or_invalid"]
        self.assertEqual(row["candidate_count"]["baseline"], 1)
        self.assertEqual(row["drawable_count"]["baseline"], 0)
        self.assertEqual(row["suppressed_or_invalid_count"]["baseline"], 1)
        self.assertIsNone(row["precision"]["baseline"])

    def test_rejects_confidence_fixture_coverage_counts_that_do_not_match_evidence(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [])
        baseline["fixtures"][0]["confidence_stratified_results"][
            "high"
        ]["fixture_count_with_candidates"] = 0

        with self.assertRaisesRegex(
            ValueError, "fixture_count_with_candidates does not match candidate_count"
        ):
            compare_benchmark_reports(baseline, candidate)

    def test_compares_models_on_identical_reviewed_fixtures(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [])
        result = compare_benchmark_reports(baseline, candidate)
        self.assertEqual(result["fixture_count"], 1)
        self.assertFalse(result["same_prediction_profile"])
        self.assertEqual(result["overall"]["recall"]["delta"], -1.0)
        self.assertEqual(result["overall"]["false_negative"]["delta"], 1)
        self.assertEqual(result["by_symbol"][0]["value"], "BTCUSDT")

    def test_comparison_reports_macro_metrics_and_deltas(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [])
        result = compare_benchmark_reports(baseline, candidate)

        self.assertEqual(
            result["macro_overall"]["metrics"]["recall"]["delta"], -1.0
        )
        self.assertEqual(
            result["macro_overall"]["metrics"]["f1"]["baseline_defined_fixture_count"],
            1,
        )
        precision = result["macro_overall"]["metrics"]["precision"]
        self.assertEqual(precision["baseline_coverage"], 1.0)
        self.assertEqual(precision["candidate_coverage"], 0.0)
        self.assertEqual(precision["coverage_delta"], -1.0)
        self.assertEqual(
            result["macro_by_zone_type"]["demand"]["metrics"]["recall"]["delta"],
            -1.0,
        )

    def test_accepts_legacy_macro_aggregates_without_coverage_fields(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [])
        for report in (baseline, candidate):
            aggregates = [report["macro_overall"], *report["macro_by_zone_type"].values()]
            for aggregate in aggregates:
                for key in list(aggregate):
                    if key.endswith("_coverage"):
                        del aggregate[key]

        result = compare_benchmark_reports(baseline, candidate)
        self.assertEqual(result["fixture_count"], 1)
        precision = result["macro_overall"]["metrics"]["precision"]
        self.assertEqual(precision["baseline_coverage"], 1.0)
        self.assertEqual(precision["candidate_coverage"], 0.0)

    def test_rejects_tampered_macro_aggregate(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [])
        candidate["macro_overall"]["recall"] = 0.25

        with self.assertRaisesRegex(ValueError, "macro_overall metrics do not match"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_tampered_macro_coverage(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [])
        candidate["macro_overall"]["precision_coverage"] = 0.5

        with self.assertRaisesRegex(ValueError, "macro_overall metrics do not match"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_boolean_macro_values_and_non_integer_counts(self):
        cases = (
            ("macro_overall", None, "precision", True),
            ("macro_overall", None, "precision_coverage", True),
            ("macro_overall", None, "fixture_count", 1.0),
            ("macro_overall", None, "precision_fixture_count", 1.0),
            ("macro_by_zone_type", "demand", "recall", False),
        )
        for aggregate_name, zone_type, field, invalid_value in cases:
            with self.subTest(aggregate=aggregate_name, zone_type=zone_type, field=field):
                baseline = self._report(
                    "baseline", [{**_zone(), "draw_overlay": True}]
                )
                candidate = self._report("candidate", [])
                aggregate = (
                    candidate[aggregate_name][zone_type]
                    if zone_type is not None
                    else baseline[aggregate_name]
                )
                aggregate[field] = invalid_value

                with self.assertRaisesRegex(
                    ValueError, "macro_.*metrics do not match"
                ):
                    compare_benchmark_reports(baseline, candidate)

    def test_rejects_drawn_zone_evidence_outside_model_plot_bounds(self):
        bounds = {
            "x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9
        }
        report = self._report(
            "baseline",
            [{**_zone(), "draw_overlay": True}],
            reference_plot_bounds=bounds,
            predicted_plot_bounds=bounds,
        )
        report["fixtures"][0]["plot_boundary_accuracy"]["zone_boxes_px"][0]["box_px"][0] = 0
        with self.assertRaisesRegex(ValueError, "zone_boxes_px contains a zone outside predicted plot bounds"):
            compare_benchmark_reports(report, report)

    def test_rejects_profile_hash_that_does_not_match_profile(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [])
        candidate["prediction_profile"]["model"] = "tampered-model"
        with self.assertRaisesRegex(ValueError, "prediction_profile_sha256 does not match"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_fixture_prediction_metadata_outside_report_profile(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [])
        candidate["fixtures"][0]["prediction_metadata"]["model"] = "tampered-model"
        with self.assertRaisesRegex(ValueError, "does not match report prediction_profile"):
            compare_benchmark_reports(baseline, candidate)

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

    def test_rejects_metric_values_that_do_not_match_confusion_counts(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [])
        candidate["fixtures"][0]["overall"]["precision"] = 0.25

        with self.assertRaisesRegex(ValueError, "overall.precision does not agree"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_supply_demand_counts_that_do_not_reconcile_with_overall(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [])
        row = candidate["fixtures"][0]["by_zone_type"]["supply"]
        row["predicted"] = 1
        row["false_positive"] = 1
        row["precision"] = 0.0
        row["recall"] = None
        row["f1"] = 0.0

        with self.assertRaisesRegex(ValueError, "by_zone_type counts do not reconcile"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_duplicate_prediction_match_indices(self):
        zones = [_zone(), _zone(x_start=0.3, x_end=0.7)]
        baseline = self._report(
            "baseline",
            [{**zones[0], "draw_overlay": True}, {**zones[1], "draw_overlay": True}],
            reference=zones,
        )
        candidate = self._report(
            "candidate",
            [{**zones[0], "draw_overlay": True}, {**zones[1], "draw_overlay": True}],
            reference=zones,
        )
        candidate["fixtures"][0]["matches"][1]["prediction_index"] = 0

        with self.assertRaisesRegex(ValueError, "duplicate matched prediction_index"):
            compare_benchmark_reports(baseline, candidate)

    def test_compares_reported_price_labels_against_calibrated_overlay_coordinates(self):
        anchors = [
            {"y": 0.2, "price": 120},
            {"y": 0.5, "price": 90},
            {"y": 0.8, "price": 60},
        ]
        baseline = self._report(
            "baseline",
            [{**_zone(), "draw_overlay": True, "price_low": "80", "price_high": "100"}],
            anchors=anchors,
        )
        candidate = self._report(
            "candidate",
            [{**_zone(), "draw_overlay": True, "price_low": "70", "price_high": "110"}],
            anchors=anchors,
        )

        result = compare_benchmark_reports(baseline, candidate)
        consistency = result["price_accuracy"]["reported_price_consistency"]
        self.assertEqual(
            consistency["mean_boundary_abs_disagreement"]["baseline"], 0.0
        )
        self.assertAlmostEqual(
            consistency["mean_boundary_abs_disagreement"]["candidate"], 10.0
        )
        self.assertAlmostEqual(
            consistency["mean_boundary_abs_disagreement"]["delta"], 10.0
        )
        self.assertAlmostEqual(
            result["fixtures"][0]["price_accuracy"][
                "reported_price_consistency"
            ]["mean_boundary_abs_disagreement"]["delta"],
            10.0,
        )

    def test_rejects_boolean_or_float_reported_price_consistency_counts(self):
        anchors = [
            {"y": 0.2, "price": 120},
            {"y": 0.5, "price": 90},
            {"y": 0.8, "price": 60},
        ]
        for invalid_count in (True, 1.0):
            with self.subTest(invalid_count=invalid_count):
                report = self._report(
                    "baseline",
                    [{**_zone(), "draw_overlay": True, "price_low": "80", "price_high": "100"}],
                    anchors=anchors,
                )
                report["fixtures"][0]["price_accuracy"]["reported_price_consistency"][
                    "scored_match_count"
                ] = invalid_count
                with self.assertRaisesRegex(
                    ValueError,
                    "reported_price_consistency.scored_match_count must be a non-negative integer",
                ):
                    compare_benchmark_reports(report, report)

    def test_rejects_tampered_reported_price_consistency_summary(self):
        anchors = [
            {"y": 0.2, "price": 120},
            {"y": 0.5, "price": 90},
            {"y": 0.8, "price": 60},
        ]
        report = self._report(
            "baseline",
            [{**_zone(), "draw_overlay": True, "price_low": "80", "price_high": "100"}],
            anchors=anchors,
        )
        report["fixtures"][0]["price_accuracy"]["reported_price_consistency"][
            "mean_boundary_abs_disagreement"
        ] = 999

        with self.assertRaisesRegex(
            ValueError, "reported_price_consistency.mean_boundary_abs_disagreement"
        ):
            compare_benchmark_reports(report, report)

    def test_rejects_price_summary_that_disagrees_with_matched_zone_errors(self):
        anchors = [
            {"y": 0.2, "price": 120},
            {"y": 0.5, "price": 90},
            {"y": 0.8, "price": 60},
        ]
        baseline = self._report(
            "baseline", [{**_zone(), "draw_overlay": True}], anchors=anchors
        )
        candidate = self._report(
            "candidate", [{**_zone(y_top=0.42), "draw_overlay": True}], anchors=anchors
        )
        candidate["fixtures"][0]["price_accuracy"]["mean_boundary_abs_error"] = 999

        with self.assertRaisesRegex(ValueError, "mean_boundary_abs_error does not match"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_boolean_and_float_report_fixture_counts(self):
        for invalid_count in (True, 1.0):
            with self.subTest(invalid_count=invalid_count):
                report = self._report("baseline", [{**_zone(), "draw_overlay": True}])
                report["fixture_count"] = invalid_count
                with self.assertRaisesRegex(ValueError, "fixture_count does not match"):
                    compare_benchmark_reports(report, report)

    def test_rejects_boolean_values_in_micro_and_context_aggregates(self):
        mutations = (
            ("overall", lambda report: report["overall"].__setitem__("precision", True)),
            ("by_zone_type", lambda report: report["by_zone_type"]["demand"].__setitem__("recall", True)),
            ("by_symbol", lambda report: report["by_symbol"][0]["metrics"].__setitem__("precision", True)),
        )
        for label, mutate in mutations:
            with self.subTest(aggregate=label):
                report = self._report("baseline", [{**_zone(), "draw_overlay": True}])
                mutate(report)
                with self.assertRaisesRegex(ValueError, "fixture results"):
                    compare_benchmark_reports(report, report)

    def test_prediction_profile_settings_use_exact_json_types(self):
        report = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        report["prediction_profile"]["settings"]["temperature"] = 1
        canonical = json.dumps(
            report["prediction_profile"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        report["prediction_profile_sha256"] = hashlib.sha256(canonical).hexdigest()
        report["fixtures"][0]["prediction_metadata"]["settings"] = {
            **report["fixtures"][0]["prediction_metadata"]["settings"],
            "temperature": True,
        }

        with self.assertRaisesRegex(ValueError, "does not match report prediction_profile"):
            compare_benchmark_reports(report, report)

    def test_rejects_tampered_report_level_aggregate(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [])
        candidate["overall"]["recall"] = 0.9

        with self.assertRaisesRegex(ValueError, "overall metrics do not match"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_integer_too_large_for_finite_float_metrics(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [])
        candidate["fixtures"][0]["overall"]["mean_center_error_normalized"] = 10 ** 1000

        with self.assertRaisesRegex(ValueError, "must be finite or null"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_non_string_match_zone_type_without_crashing(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [{**_zone(), "draw_overlay": True}])
        candidate["fixtures"][0]["matches"][0]["zone_type"] = []

        with self.assertRaisesRegex(ValueError, "invalid shape or zone_type"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_non_string_price_axis_scale_without_crashing(self):
        anchors = [
            {"y": 0.2, "price": 120},
            {"y": 0.5, "price": 90},
            {"y": 0.8, "price": 60},
        ]
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}], anchors=anchors)
        candidate = self._report("candidate", [{**_zone(), "draw_overlay": True}], anchors=anchors)
        candidate["fixtures"][0]["price_accuracy"]["scale"] = []

        with self.assertRaisesRegex(ValueError, "scale must be linear or log"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_match_iou_below_declared_threshold(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [{**_zone(), "draw_overlay": True}])
        candidate["fixtures"][0]["matches"][0]["iou"] = 0.4
        candidate["fixtures"][0]["overall"]["mean_iou"] = 0.4
        candidate["fixtures"][0]["by_zone_type"]["demand"]["mean_iou"] = 0.4
        candidate["overall"]["mean_iou"] = 0.4
        candidate["by_zone_type"]["demand"]["mean_iou"] = 0.4

        with self.assertRaisesRegex(ValueError, "below the fixture threshold"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_duplicate_fixture_ids(self):
        baseline = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate", [])
        candidate["fixtures"].append(dict(candidate["fixtures"][0]))
        candidate["fixture_count"] = 2
        with self.assertRaisesRegex(ValueError, "duplicate fixture_id"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_duplicate_screenshot_hashes_under_distinct_fixture_ids(self):
        report = self._report("baseline", [{**_zone(), "draw_overlay": True}])
        duplicate = dict(report["fixtures"][0])
        duplicate["fixture_id"] = "btc-1h-duplicate"
        duplicate["adjudication_metadata"] = _adjudication_metadata(
            duplicate["fixture_id"],
            duplicate["image_sha256"],
            duplicate["market_context"],
            duplicate["annotation_metadata"],
        )
        report["fixtures"].append(duplicate)
        report["fixture_count"] += 1

        with self.assertRaisesRegex(ValueError, "duplicate screenshot image_sha256"):
            _validate_report(report, "test")

    def test_compares_plot_boundary_accuracy_on_identical_manual_plot(self):
        reference_plot = {
            "x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9
        }
        wider_candidate = {
            "x_start": 0.1, "x_end": 0.95, "y_top": 0.1, "y_bottom": 0.95
        }
        baseline = self._report(
            "baseline", [{**_zone(), "draw_overlay": True}],
            reference_plot_bounds=reference_plot,
            predicted_plot_bounds=reference_plot,
        )
        candidate = self._report(
            "candidate", [{**_zone(), "draw_overlay": True}],
            reference_plot_bounds=reference_plot,
            predicted_plot_bounds=wider_candidate,
        )

        result = compare_benchmark_reports(baseline, candidate)
        plot_delta = result["plot_boundary_accuracy"]["delta"]
        self.assertLess(plot_delta["mean_plot_iou"], 0)
        self.assertGreater(
            plot_delta["mean_predicted_plot_outside_reference_pct"], 0
        )
        self.assertLess(
            result["fixtures"][0]["plot_boundary_accuracy"]["delta"]["plot_iou"],
            0,
        )

    def test_rejects_different_manual_plot_boundaries(self):
        reference_a = {
            "x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9
        }
        reference_b = {
            "x_start": 0.12, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9
        }
        baseline = self._report(
            "baseline", [{**_zone(), "draw_overlay": True}],
            reference_plot_bounds=reference_a, predicted_plot_bounds=reference_a,
        )
        candidate = self._report(
            "candidate", [{**_zone(), "draw_overlay": True}],
            reference_plot_bounds=reference_b, predicted_plot_bounds=reference_b,
        )
        with self.assertRaisesRegex(ValueError, "reference_plot_bounds_sha256"):
            compare_benchmark_reports(baseline, candidate)

    def test_rejects_tampered_plot_boundary_metrics(self):
        plot = {
            "x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9
        }
        baseline = self._report(
            "baseline", [{**_zone(), "draw_overlay": True}],
            reference_plot_bounds=plot, predicted_plot_bounds=plot,
        )
        candidate = self._report(
            "candidate", [{**_zone(), "draw_overlay": True}],
            reference_plot_bounds=plot, predicted_plot_bounds=plot,
        )
        candidate["fixtures"][0]["plot_boundary_accuracy"]["plot_iou"] = 0.25
        with self.assertRaisesRegex(ValueError, "does not agree with its plot bounds"):
            compare_benchmark_reports(baseline, candidate)



    def test_rejects_boolean_substitutions_in_plot_boundary_metrics(self):
        mutations = (
            ("plot_iou", lambda metrics: metrics.__setitem__("plot_iou", True)),
            ("predicted_zone_count", lambda metrics: metrics.__setitem__("predicted_zone_count", True)),
            ("zones_outside_reference_plot_count", lambda metrics: metrics.__setitem__("zones_outside_reference_plot_count", False)),
            ("edge_displacement_px.left", lambda metrics: metrics["edge_displacement_px"].__setitem__("left", False)),
            ("edge_displacement_normalized.left", lambda metrics: metrics["edge_displacement_normalized"].__setitem__("left", False)),
        )
        for field, mutate in mutations:
            with self.subTest(field=field):
                plot = {
                    "x_start": 0.1, "x_end": 0.9, "y_top": 0.1, "y_bottom": 0.9
                }
                report = self._report(
                    "baseline", [{**_zone(), "draw_overlay": True}],
                    reference_plot_bounds=plot, predicted_plot_bounds=plot,
                )
                metrics = report["fixtures"][0]["plot_boundary_accuracy"]
                mutate(metrics)

                with self.assertRaisesRegex(
                    ValueError, "does not agree with its plot bounds"
                ):
                    compare_benchmark_reports(report, report)

    def test_price_accuracy_validator_rejects_overflowing_scored_percentage(self):
        price = {
            "scale": "linear",
            "anchor_count": 3,
            "max_anchor_relative_error_pct": 0.0,
            "scored_match_count": 1,
            "unscored_match_count": 0,
            "mean_boundary_abs_error": 1e308,
            "max_boundary_abs_error": 1e308,
            "mean_boundary_error_pct": 1.0,
            "reported_price_consistency": {
                "scored_match_count": 0,
                "unscored_match_count": 1,
                "mean_boundary_abs_disagreement": None,
                "max_boundary_abs_disagreement": None,
                "mean_boundary_disagreement_pct": None,
            },
        }
        matches = [{
            "price_comparison": {
                "status": "scored",
                "predicted_price_bounds": {
                    "price_high": 1e308, "price_low": 5e307,
                },
                "reference_price_bounds": {
                    "price_high": 1.0, "price_low": 0.5,
                },
                "price_high_abs_error": 1e308,
                "price_low_abs_error": 5e307,
                "price_high_error_pct": 1.0,
                "price_low_error_pct": 1.0,
            },
            "reported_price_consistency": {
                "status": "unscored",
                "reason": "reported labels unavailable",
            },
        }]

        with self.assertRaisesRegex(ValueError, "outside the supported numeric range"):
            _validate_price_accuracy(price, matches, "fixture")

    def test_report_loader_rejects_duplicate_object_keys(self):
        report_path = self.root / "duplicate-report.json"
        report_path.write_text(
            '{"fixture_count":1,"fixture_count":2}', encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key 'fixture_count'"):
            _read_report(report_path, "baseline")

    def test_report_loader_rejects_non_standard_numeric_constants(self):
        report_path = self.root / "nonfinite-report.json"
        report_path.write_text('{"score":NaN}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-standard JSON constant 'NaN'"):
            _read_report(report_path, "candidate")


    def test_compares_inference_usage_and_latency_with_paired_deltas(self):
        baseline_usage = {
            "prompt_tokens": 1200,
            "completion_tokens": 300,
            "total_tokens": 1500,
            "image_tokens": 850,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 40,
        }
        candidate_usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 250,
            "total_tokens": 1250,
            "image_tokens": 700,
            "cached_prompt_tokens": 100,
            "reasoning_tokens": 25,
        }
        baseline = self._report(
            "same-model", [{**_zone(), "draw_overlay": True}],
            usage=baseline_usage, request_latency_ms=1250,
        )
        candidate = self._report(
            "same-model", [{**_zone(), "draw_overlay": True}],
            usage=candidate_usage, request_latency_ms=900,
        )

        result = compare_benchmark_reports(baseline, candidate)
        telemetry = result["inference_telemetry"]

        self.assertTrue(result["same_prediction_profile"])
        self.assertEqual(
            telemetry["baseline"]["usage"]["total_tokens"]["total"], 1500
        )
        self.assertEqual(
            telemetry["candidate"]["usage"]["total_tokens"]["total"], 1250
        )
        self.assertEqual(
            telemetry["baseline"]["request_latency_ms"]["mean"], 1250.0
        )
        self.assertEqual(
            telemetry["paired_deltas"]["total_tokens"]["total_delta"], -250
        )
        self.assertEqual(
            telemetry["paired_deltas"]["total_tokens"]["paired_fixture_count"], 1
        )
        self.assertEqual(
            telemetry["paired_deltas"]["request_latency_ms"]["mean_delta"], -350.0
        )
        self.assertEqual(
            result["fixtures"][0]["inference_telemetry"]["baseline"]["usage"],
            baseline_usage,
        )
        self.assertEqual(
            result["fixtures"][0]["inference_telemetry"]["candidate"]["request_latency_ms"],
            900,
        )

    def test_missing_inference_telemetry_is_not_treated_as_zero(self):
        baseline = self._report(
            "same-model", [{**_zone(), "draw_overlay": True}]
        )
        candidate = self._report(
            "same-model", [{**_zone(), "draw_overlay": True}],
            usage={
                "prompt_tokens": 1000,
                "completion_tokens": 250,
                "total_tokens": 1250,
            },
            request_latency_ms=900,
        )

        telemetry = compare_benchmark_reports(
            baseline, candidate
        )["inference_telemetry"]

        self.assertEqual(
            telemetry["baseline"]["usage"]["reported_fixture_count"], 0
        )
        self.assertEqual(
            telemetry["baseline"]["usage"]["missing_fixture_count"], 1
        )
        self.assertIsNone(
            telemetry["baseline"]["usage"]["total_tokens"]["total"]
        )
        self.assertEqual(
            telemetry["paired_deltas"]["total_tokens"]["paired_fixture_count"], 0
        )
        self.assertEqual(
            telemetry["paired_deltas"]["total_tokens"]["missing_pair_count"], 1
        )
        self.assertIsNone(
            telemetry["paired_deltas"]["total_tokens"]["total_delta"]
        )
        self.assertIsNone(
            telemetry["paired_deltas"]["request_latency_ms"]["mean_delta"]
        )

    def test_telemetry_summary_reports_median_and_nearest_rank_p95(self):
        summary = _telemetry_stat([100, 200, 300, 400, 500], 6)

        self.assertEqual(summary["reported_fixture_count"], 5)
        self.assertEqual(summary["missing_fixture_count"], 1)
        self.assertEqual(summary["median"], 300)
        self.assertEqual(summary["p95"], 500)

        empty = _telemetry_stat([], 2)
        self.assertIsNone(empty["median"])
        self.assertIsNone(empty["p95"])
        self.assertEqual(empty["missing_fixture_count"], 2)

    def test_optional_date_bound_pricing_estimates_paired_model_costs(self):
        baseline_usage = {
            "prompt_tokens": 1000, "completion_tokens": 250, "total_tokens": 1250,
            "image_tokens": 400, "cached_prompt_tokens": 200, "reasoning_tokens": 50,
        }
        candidate_usage = {
            "prompt_tokens": 800, "completion_tokens": 200, "total_tokens": 1000,
            "image_tokens": 300, "cached_prompt_tokens": 100, "reasoning_tokens": 40,
        }
        baseline = self._report(
            "baseline-model", [{**_zone(), "draw_overlay": True}], usage=baseline_usage
        )
        candidate = self._report(
            "candidate-model", [{**_zone(), "draw_overlay": True}], usage=candidate_usage
        )
        pricing_card = {
            "schema": "stockmind.usage-pricing.v1",
            "version": "test-rate-card-2026-11",
            "currency": "USD",
            "effective_from": "2026-11-01",
            "effective_until": "2026-12-01",
            "source": "Synthetic unit-test rates; not provider pricing",
            "models": {
                "baseline-model": {
                    "input_usd_per_million_tokens": "10",
                    "cached_input_usd_per_million_tokens": "2",
                    "output_usd_per_million_tokens": "20",
                },
                "candidate-model": {
                    "input_usd_per_million_tokens": "10",
                    "cached_input_usd_per_million_tokens": "2",
                    "output_usd_per_million_tokens": "20",
                },
            },
        }

        result = compare_benchmark_reports(
            baseline, candidate, pricing_card=pricing_card
        )
        cost = result["currency_cost_estimate"]
        self.assertEqual(cost["baseline"]["total_cost_usd"], "0.013400000000000000")
        self.assertEqual(cost["candidate"]["total_cost_usd"], "0.011200000000000000")
        self.assertEqual(
            cost["paired_deltas"]["total_delta_usd"], "-0.002200000000000000"
        )
        self.assertEqual(cost["paired_deltas"]["paired_fixture_count"], 1)
        fixture_cost = result["fixtures"][0]["estimated_cost"]
        self.assertEqual(fixture_cost["baseline"]["status"], "priced")
        self.assertEqual(fixture_cost["baseline"]["image_tokens_included_in_input"], 400)
        self.assertEqual(fixture_cost["baseline"]["reasoning_tokens_included_in_output"], 50)

    def test_cost_estimate_preserves_sub_picodollar_per_token_precision(self):
        baseline = self._report(
            "baseline-model",
            [{**_zone(), "draw_overlay": True}],
            usage={"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
        )
        candidate = self._report(
            "candidate-model",
            [{**_zone(), "draw_overlay": True}],
            usage={"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2},
        )
        card = {
            "schema": "stockmind.usage-pricing.v1",
            "version": "sub-picodollar-precision-test",
            "currency": "USD",
            "effective_from": "2026-11-01",
            "effective_until": None,
            "source": "Synthetic unit-test rates; not provider pricing",
            "models": {
                "baseline-model": {
                    "input_usd_per_million_tokens": "0.000000000001",
                    "output_usd_per_million_tokens": "0",
                },
                "candidate-model": {
                    "input_usd_per_million_tokens": "0.000000000001",
                    "output_usd_per_million_tokens": "0",
                },
            },
        }

        result = compare_benchmark_reports(baseline, candidate, pricing_card=card)
        cost = result["currency_cost_estimate"]
        self.assertEqual(cost["baseline"]["total_cost_usd"], "0.000000000000000001")
        self.assertEqual(cost["candidate"]["total_cost_usd"], "0.000000000000000002")
        self.assertEqual(cost["paired_deltas"]["total_delta_usd"], "0.000000000000000001")
        self.assertEqual(
            result["fixtures"][0]["estimated_cost"]["baseline"]["total_cost_usd"],
            "0.000000000000000001",
        )


    def test_cost_estimate_is_unpriced_when_discounted_cached_usage_is_missing(self):
        baseline = self._report(
            "baseline-model", [{**_zone(), "draw_overlay": True}],
            usage={"prompt_tokens": 1000, "completion_tokens": 250, "total_tokens": 1250},
        )
        candidate = self._report(
            "candidate-model", [{**_zone(), "draw_overlay": True}],
            usage={"prompt_tokens": 1000, "completion_tokens": 250, "total_tokens": 1250},
        )
        pricing_card = {
            "schema": "stockmind.usage-pricing.v1",
            "version": "test-rate-card",
            "currency": "USD",
            "effective_from": "2026-11-01",
            "effective_until": None,
            "source": "Synthetic unit-test rates; not provider pricing",
            "models": {
                model: {
                    "input_usd_per_million_tokens": "10",
                    "cached_input_usd_per_million_tokens": "2",
                    "output_usd_per_million_tokens": "20",
                }
                for model in ("baseline-model", "candidate-model")
            },
        }
        result = compare_benchmark_reports(
            baseline, candidate, pricing_card=pricing_card
        )
        cost = result["currency_cost_estimate"]
        self.assertEqual(cost["baseline"]["priced_fixture_count"], 0)
        self.assertEqual(cost["baseline"]["unpriced_reasons"], {
            "cached_token_usage_missing": 1
        })
        self.assertIsNone(cost["paired_deltas"]["total_delta_usd"])

    def test_pricing_card_rejects_float_rates_and_unknown_model_rate_fields(self):
        report = self._report("baseline-model", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate-model", [{**_zone(), "draw_overlay": True}])
        card = {
            "schema": "stockmind.usage-pricing.v1",
            "version": "test-rate-card",
            "currency": "USD",
            "effective_from": "2026-11-01",
            "effective_until": None,
            "source": "Synthetic unit-test rates; not provider pricing",
            "models": {
                "baseline-model": {
                    "input_usd_per_million_tokens": 10.0,
                    "output_usd_per_million_tokens": "20",
                },
                "candidate-model": {
                    "input_usd_per_million_tokens": "10",
                    "output_usd_per_million_tokens": "20",
                },
            },
        }
        with self.assertRaisesRegex(ValueError, "decimal string"):
            compare_benchmark_reports(report, candidate, pricing_card=card)

    def test_pricing_card_accepts_decimal_string_rates(self):
        baseline = self._report("baseline-model", [{**_zone(), "draw_overlay": True}])
        candidate = self._report("candidate-model", [{**_zone(), "draw_overlay": True}])
        card = {
            "schema": "stockmind.usage-pricing.v1",
            "version": "decimal-rate-test",
            "currency": "USD",
            "effective_from": "2026-11-01",
            "effective_until": None,
            "source": "Synthetic unit-test rates; not provider pricing",
            "models": {
                model: {
                    "input_usd_per_million_tokens": "10.25",
                    "cached_input_usd_per_million_tokens": "2.5",
                    "output_usd_per_million_tokens": "20.75",
                }
                for model in ("baseline-model", "candidate-model")
            },
        }
        result = compare_benchmark_reports(baseline, candidate, pricing_card=card)
        self.assertEqual(
            result["currency_cost_estimate"]["pricing_card"]["version"],
            "decimal-rate-test",
        )

    def test_cost_is_not_estimated_outside_rate_card_effective_dates(self):
        usage = {"prompt_tokens": 1000, "completion_tokens": 250, "total_tokens": 1250}
        baseline = self._report(
            "baseline-model", [{**_zone(), "draw_overlay": True}], usage=usage
        )
        candidate = self._report(
            "candidate-model", [{**_zone(), "draw_overlay": True}], usage=usage
        )
        card = {
            "schema": "stockmind.usage-pricing.v1",
            "version": "future-rate-card",
            "currency": "USD",
            "effective_from": "2026-11-07",
            "effective_until": None,
            "source": "Synthetic unit-test rates; not provider pricing",
            "models": {
                model: {
                    "input_usd_per_million_tokens": "10",
                    "output_usd_per_million_tokens": "20",
                }
                for model in ("baseline-model", "candidate-model")
            },
        }
        result = compare_benchmark_reports(baseline, candidate, pricing_card=card)
        self.assertEqual(
            result["currency_cost_estimate"]["baseline"]["unpriced_reasons"],
            {"outside_pricing_effective_period": 1},
        )
        self.assertIsNone(
            result["currency_cost_estimate"]["paired_deltas"]["total_delta_usd"]
        )

    def test_paired_telemetry_delta_reports_median_and_p95(self):
        baseline_values = [100, 200, 300, 400, 500]
        candidate_values = [150, 180, 450, 350, 700]
        baseline = [
            {
                "fixture_id": f"fixture-{index}",
                "prediction_metadata": {"request_latency_ms": value},
            }
            for index, value in enumerate(baseline_values)
        ]
        candidate = [
            {
                "fixture_id": f"fixture-{index}",
                "prediction_metadata": {"request_latency_ms": value},
            }
            for index, value in enumerate(candidate_values)
        ]

        result = _paired_telemetry_delta(
            baseline, candidate, "request_latency_ms"
        )

        self.assertEqual(result["paired_fixture_count"], 5)
        self.assertEqual(result["missing_pair_count"], 0)
        self.assertEqual(result["median_delta"], 50)
        self.assertEqual(result["p95_delta"], 200)


    def test_large_token_costs_preserve_precision_in_totals_and_paired_deltas(self):
        baseline_tokens = 10**20
        candidate_tokens = baseline_tokens + 1
        baseline = self._report(
            "baseline-model",
            [{**_zone(), "draw_overlay": True}],
            usage={
                "prompt_tokens": baseline_tokens,
                "completion_tokens": 0,
                "total_tokens": baseline_tokens,
            },
        )
        candidate = self._report(
            "candidate-model",
            [{**_zone(), "draw_overlay": True}],
            usage={
                "prompt_tokens": candidate_tokens,
                "completion_tokens": 0,
                "total_tokens": candidate_tokens,
            },
        )
        card = {
            "schema": "stockmind.usage-pricing.v1",
            "version": "large-token-precision-test",
            "currency": "USD",
            "effective_from": "2026-11-01",
            "effective_until": None,
            "source": "Synthetic unit-test rates; not provider pricing",
            "models": {
                model: {
                    "input_usd_per_million_tokens": "1",
                    "output_usd_per_million_tokens": "0",
                }
                for model in ("baseline-model", "candidate-model")
            },
        }

        result = compare_benchmark_reports(baseline, candidate, pricing_card=card)
        cost = result["currency_cost_estimate"]

        self.assertEqual(
            cost["baseline"]["total_cost_usd"],
            "100000000000000.000000000000000000",
        )
        self.assertEqual(
            cost["candidate"]["total_cost_usd"],
            "100000000000000.000001000000000000",
        )
        self.assertEqual(
            cost["paired_deltas"]["total_delta_usd"],
            "0.000001000000000000",
        )
        self.assertEqual(
            cost["paired_deltas"]["mean_delta_usd"],
            "0.000001000000000000",
        )


if __name__ == "__main__":
    unittest.main()
