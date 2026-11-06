"""Paired comparison of StockMind chart-zone benchmark reports."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from .zone_benchmark import (
    _aggregate_by_context,
    _aggregate_metrics,
    _aggregate_price_accuracy,
    _validate_annotation_metadata,
    _validate_market_context,
    _validate_prediction_metadata,
)
from .plot_evaluation import (
    aggregate_plot_boundary_accuracy,
    compare_plot_boundary_accuracy,
    validate_plot_boundary_accuracy,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COUNT_FIELDS = ("predicted", "reference", "true_positive", "false_positive", "false_negative")
_METRIC_FIELDS = _COUNT_FIELDS + (
    "precision", "recall", "f1", "mean_iou", "mean_center_error_normalized"
)
_CONTEXT_FIELDS = ("asset_class", "symbol", "exchange", "timeframe")


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _validate_metrics(row, label):
    if not isinstance(row, dict):
        raise ValueError(f"{label} must be a JSON object")
    for field in _COUNT_FIELDS:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label}.{field} must be a non-negative integer")
    if row["predicted"] != row["true_positive"] + row["false_positive"]:
        raise ValueError(f"{label} predicted count does not equal TP + FP")
    if row["reference"] != row["true_positive"] + row["false_negative"]:
        raise ValueError(f"{label} reference count does not equal TP + FN")
    for field in _METRIC_FIELDS[len(_COUNT_FIELDS):]:
        if field not in row:
            raise ValueError(f"{label}.{field} is missing")
        value = row[field]
        if value is not None and not _finite(value):
            raise ValueError(f"{label}.{field} must be finite or null")
    for field in ("precision", "recall", "f1", "mean_iou"):
        value = row.get(field)
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"{label}.{field} must be between 0 and 1 or null")
    value = row.get("mean_center_error_normalized")
    if value is not None and value < 0:
        raise ValueError(f"{label}.mean_center_error_normalized cannot be negative")


def _same_metric(actual, expected):
    if expected is None:
        return actual is None
    return _finite(actual) and math.isclose(
        float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12
    )


def _validate_metric_consistency(row, matches, label):
    """Recompute every reported metric from its counts and matched-zone rows."""
    predicted = row["predicted"]
    reference = row["reference"]
    true_positive = row["true_positive"]
    false_positive = row["false_positive"]
    false_negative = row["false_negative"]
    denominator = 2 * true_positive + false_positive + false_negative
    expected = {
        "precision": true_positive / predicted if predicted else None,
        "recall": true_positive / reference if reference else None,
        "f1": 2 * true_positive / denominator if denominator else None,
        "mean_iou": sum(match["iou"] for match in matches) / len(matches) if matches else None,
        "mean_center_error_normalized": (
            sum(match["center_error_normalized"] for match in matches) / len(matches)
            if matches else None
        ),
    }
    for field, value in expected.items():
        if not _same_metric(row[field], value):
            raise ValueError(f"{label}.{field} does not agree with its counts and matches")


def _validate_price_accuracy(price, matches, label):
    """Validate per-match price errors and their fixture-level summary."""
    if price is None:
        if any(match.get("price_comparison") is not None for match in matches):
            raise ValueError(f"{label} omits price_accuracy but contains price comparisons")
        return

    if not isinstance(price, dict):
        raise ValueError(f"{label} must be an object or null")
    required = {
        "scale", "anchor_count", "max_anchor_relative_error_pct",
        "scored_match_count", "unscored_match_count",
        "mean_boundary_abs_error", "max_boundary_abs_error",
        "mean_boundary_error_pct",
    }
    missing = sorted(required - set(price))
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(missing)}")
    if not isinstance(price["scale"], str) or price["scale"] not in {"linear", "log"}:
        raise ValueError(f"{label}.scale must be linear or log")
    anchor_count = price["anchor_count"]
    if isinstance(anchor_count, bool) or not isinstance(anchor_count, int) or not 3 <= anchor_count <= 20:
        raise ValueError(f"{label}.anchor_count must be an integer from 3 to 20")
    residual = price["max_anchor_relative_error_pct"]
    if not _finite(residual) or not 0 <= residual <= 10:
        raise ValueError(f"{label}.max_anchor_relative_error_pct must be between 0 and 10")
    for field in ("scored_match_count", "unscored_match_count"):
        value = price[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label}.{field} must be a non-negative integer")
    if price["scored_match_count"] + price["unscored_match_count"] != len(matches):
        raise ValueError(f"{label} scored and unscored counts do not equal matched zones")

    absolute_errors = []
    percentage_errors = []
    scored_count = 0
    unscored_count = 0
    for index, match in enumerate(matches):
        comparison = match.get("price_comparison")
        if not isinstance(comparison, dict):
            raise ValueError(f"{label} match {index} is missing its price comparison")
        status = comparison.get("status")
        if status == "unscored":
            if not isinstance(comparison.get("reason"), str) or not comparison["reason"].strip():
                raise ValueError(f"{label} match {index} has no unscored reason")
            unscored_count += 1
            continue
        if status != "scored":
            raise ValueError(f"{label} match {index} has an invalid price-comparison status")

        bounds = {}
        for side in ("predicted_price_bounds", "reference_price_bounds"):
            value = comparison.get(side)
            if not isinstance(value, dict):
                raise ValueError(f"{label} match {index}.{side} must be an object")
            high, low = value.get("price_high"), value.get("price_low")
            if not _finite(high) or not _finite(low) or low <= 0 or high < low:
                raise ValueError(f"{label} match {index}.{side} has invalid positive price bounds")
            bounds[side] = (float(high), float(low))

        predicted_high, predicted_low = bounds["predicted_price_bounds"]
        reference_high, reference_low = bounds["reference_price_bounds"]
        expected_errors = {
            "price_high_abs_error": abs(predicted_high - reference_high),
            "price_low_abs_error": abs(predicted_low - reference_low),
            "price_high_error_pct": 100 * abs(predicted_high - reference_high) / reference_high,
            "price_low_error_pct": 100 * abs(predicted_low - reference_low) / reference_low,
        }
        for field, expected in expected_errors.items():
            if not _same_metric(comparison.get(field), expected):
                raise ValueError(f"{label} match {index}.{field} does not match its price bounds")
        absolute_errors.extend((
            expected_errors["price_high_abs_error"],
            expected_errors["price_low_abs_error"],
        ))
        percentage_errors.extend((
            expected_errors["price_high_error_pct"],
            expected_errors["price_low_error_pct"],
        ))
        scored_count += 1

    if scored_count != price["scored_match_count"] or unscored_count != price["unscored_match_count"]:
        raise ValueError(f"{label} scored/unscored counts do not match per-zone results")
    expected_summary = {
        "mean_boundary_abs_error": sum(absolute_errors) / len(absolute_errors) if absolute_errors else None,
        "max_boundary_abs_error": max(absolute_errors) if absolute_errors else None,
        "mean_boundary_error_pct": sum(percentage_errors) / len(percentage_errors) if percentage_errors else None,
    }
    for field, expected in expected_summary.items():
        if not _same_metric(price[field], expected):
            raise ValueError(f"{label}.{field} does not match per-zone price errors")


def _validate_aggregate_report(report, fixtures, label):
    """Ensure report-level aggregates are derived from the included fixtures."""
    all_matches = [match for fixture in fixtures for match in fixture["matches"]]
    expected_overall = _aggregate_metrics(
        [fixture["overall"] for fixture in fixtures], all_matches
    )
    if report.get("overall") != expected_overall:
        raise ValueError(f"{label} overall metrics do not match its fixture results")

    expected_by_type = {}
    for zone_type in ("supply", "demand"):
        rows = [fixture["by_zone_type"][zone_type] for fixture in fixtures]
        matches = [match for match in all_matches if match["zone_type"] == zone_type]
        expected_by_type[zone_type] = _aggregate_metrics(rows, matches)
    if report.get("by_zone_type") != expected_by_type:
        raise ValueError(f"{label} by_zone_type metrics do not match its fixture results")

    expected_ignored = sum(fixture["ignored_prediction_count"] for fixture in fixtures)
    if report.get("ignored_prediction_count") != expected_ignored:
        raise ValueError(f"{label} ignored_prediction_count does not match its fixtures")

    if report.get("price_accuracy") != _aggregate_price_accuracy(fixtures):
        raise ValueError(f"{label} price_accuracy does not match its fixture results")

    if report.get("plot_boundary_accuracy") != aggregate_plot_boundary_accuracy(fixtures):
        raise ValueError(f"{label} plot_boundary_accuracy does not match its fixture results")

    for field, key in (
        ("asset_class", "by_asset_class"),
        ("symbol", "by_symbol"),
        ("exchange", "by_exchange"),
        ("timeframe", "by_timeframe"),
    ):
        if report.get(key) != _aggregate_by_context(fixtures, field):
            raise ValueError(f"{label} {key} does not match its fixture results")


def _validate_report(report, label):
    if not isinstance(report, dict):
        raise ValueError(f"{label} report must be a JSON object")
    profile_hash = report.get("prediction_profile_sha256")
    if not isinstance(profile_hash, str) or not _SHA256_RE.fullmatch(profile_hash):
        raise ValueError(f"{label} report has an invalid prediction_profile_sha256")
    profile = report.get("prediction_profile")
    if not isinstance(profile, dict):
        raise ValueError(f"{label} report is missing prediction_profile")
    try:
        canonical_profile = json.dumps(
            profile, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} prediction_profile must contain finite JSON values") from exc
    if hashlib.sha256(canonical_profile).hexdigest() != profile_hash:
        raise ValueError(f"{label} prediction_profile_sha256 does not match prediction_profile")
    guideline = report.get("annotation_guideline_version")
    if not isinstance(guideline, str) or not guideline:
        raise ValueError(f"{label} report is missing annotation_guideline_version")
    fixtures = report.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError(f"{label} report must contain a non-empty fixtures array")
    if report.get("fixture_count") != len(fixtures):
        raise ValueError(f"{label} fixture_count does not match its fixtures array")

    indexed = {}
    for index, fixture in enumerate(fixtures):
        prefix = f"{label} fixture at index {index}"
        if not isinstance(fixture, dict):
            raise ValueError(f"{prefix} must be a JSON object")
        fixture_id = fixture.get("fixture_id")
        if not isinstance(fixture_id, str) or not fixture_id.strip():
            raise ValueError(f"{prefix} has an invalid fixture_id")
        if fixture_id in indexed:
            raise ValueError(f"{label} report contains duplicate fixture_id {fixture_id!r}")
        for field in ("image_sha256", "reference_zones_sha256"):
            value = fixture.get(field)
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise ValueError(f"{prefix} has an invalid {field}")
        calibration_hash = fixture.get("price_axis_calibration_sha256")
        if calibration_hash is not None and (
            not isinstance(calibration_hash, str) or not _SHA256_RE.fullmatch(calibration_hash)
        ):
            raise ValueError(f"{prefix} has an invalid price_axis_calibration_sha256")
        try:
            _, fixture_profile, _ = _validate_prediction_metadata(
                fixture.get("prediction_metadata"), index + 1, fixture_id
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{prefix} has invalid prediction provenance: {exc}") from exc
        if fixture_profile != profile:
            raise ValueError(f"{prefix} prediction_metadata does not match report prediction_profile")
        try:
            annotation = _validate_annotation_metadata(
                fixture.get("annotation_metadata"), index + 1, fixture_id
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{prefix} has invalid annotation provenance: {exc}") from exc
        if annotation["guideline_version"] != guideline:
            raise ValueError(f"{prefix} annotation guideline does not match report")
        try:
            _validate_market_context(fixture.get("market_context"), index + 1, fixture_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{prefix} has invalid market_context: {exc}") from exc
        threshold = fixture.get("iou_threshold")
        if not _finite(threshold) or not 0 < threshold <= 1:
            raise ValueError(f"{prefix} has an invalid iou_threshold")

        plot_fields = (
            "image_width", "image_height", "reference_plot_bounds",
            "predicted_plot_bounds", "reference_plot_bounds_sha256",
            "plot_boundary_accuracy",
        )
        if not any(field in fixture for field in plot_fields):
            # Preserve compatibility with reports created before plot-boundary scoring.
            fixture.update({
                "image_width": None,
                "image_height": None,
                "reference_plot_bounds": None,
                "predicted_plot_bounds": None,
                "reference_plot_bounds_sha256": None,
                "plot_boundary_accuracy": None,
            })
        else:
            for dimension in ("image_width", "image_height"):
                value = fixture.get(dimension)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{prefix}.{dimension} must be a positive integer")
            reference_bounds = fixture.get("reference_plot_bounds")
            predicted_bounds = fixture.get("predicted_plot_bounds")
            reference_hash = fixture.get("reference_plot_bounds_sha256")
            if reference_bounds is None:
                if reference_hash is not None:
                    raise ValueError(f"{prefix} has a reference plot hash without reference bounds")
            else:
                if not isinstance(reference_hash, str) or not _SHA256_RE.fullmatch(reference_hash):
                    raise ValueError(f"{prefix} has an invalid reference_plot_bounds_sha256")
                try:
                    canonical_reference = json.dumps(
                        reference_bounds, sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode("utf-8")
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{prefix} reference_plot_bounds must contain finite JSON values") from exc
                if hashlib.sha256(canonical_reference).hexdigest() != reference_hash:
                    raise ValueError(f"{prefix} reference_plot_bounds_sha256 does not match reference_plot_bounds")
            try:
                validate_plot_boundary_accuracy(
                    fixture.get("plot_boundary_accuracy"),
                    reference_bounds,
                    predicted_bounds,
                    fixture["image_width"],
                    fixture["image_height"],
                    f"{prefix}.plot_boundary_accuracy",
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(str(exc)) from exc

        _validate_metrics(fixture.get("overall"), f"{prefix}.overall")
        by_type = fixture.get("by_zone_type")
        if not isinstance(by_type, dict) or set(by_type) != {"supply", "demand"}:
            raise ValueError(f"{prefix} by_zone_type must contain supply and demand")
        for zone_type in ("supply", "demand"):
            _validate_metrics(by_type[zone_type], f"{prefix}.by_zone_type.{zone_type}")
        matches = fixture.get("matches")
        if not isinstance(matches, list):
            raise ValueError(f"{prefix} matches must be an array")
        if len(matches) != fixture["overall"]["true_positive"]:
            raise ValueError(f"{prefix} match count does not equal true_positive")
        prediction_indices = set()
        reference_indices = set()
        for match_index, match in enumerate(matches):
            if (
                not isinstance(match, dict)
                or not isinstance(match.get("zone_type"), str)
                or match.get("zone_type") not in {"supply", "demand"}
            ):
                raise ValueError(f"{prefix} match {match_index} has an invalid shape or zone_type")
            for field, seen in (
                ("prediction_index", prediction_indices),
                ("reference_index", reference_indices),
            ):
                value = match.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{prefix} match {match_index}.{field} must be a non-negative integer")
                if value in seen:
                    raise ValueError(f"{prefix} contains duplicate matched {field} {value}")
                seen.add(value)
            for field in ("iou", "center_error_normalized"):
                value = match.get(field)
                if not _finite(value) or value < 0:
                    raise ValueError(f"{prefix} match {match_index}.{field} is invalid")
            if match["iou"] <= 0 or match["iou"] + 1e-12 < threshold:
                raise ValueError(f"{prefix} match {match_index}.iou is below the fixture threshold")
            if match["iou"] > 1:
                raise ValueError(f"{prefix} match {match_index}.iou cannot exceed 1")
            if match["center_error_normalized"] > math.sqrt(2) + 1e-12:
                raise ValueError(f"{prefix} match {match_index}.center_error_normalized exceeds image bounds")

        _validate_metric_consistency(fixture["overall"], matches, f"{prefix}.overall")
        for zone_type in ("supply", "demand"):
            type_matches = [match for match in matches if match["zone_type"] == zone_type]
            _validate_metric_consistency(
                by_type[zone_type], type_matches, f"{prefix}.by_zone_type.{zone_type}"
            )
        for field in _COUNT_FIELDS:
            if by_type["supply"][field] + by_type["demand"][field] != fixture["overall"][field]:
                raise ValueError(f"{prefix} by_zone_type counts do not reconcile with overall")
        ignored = fixture.get("ignored_prediction_count")
        if isinstance(ignored, bool) or not isinstance(ignored, int) or ignored < 0:
            raise ValueError(f"{prefix} ignored_prediction_count must be non-negative")
        _validate_price_accuracy(
            fixture.get("price_accuracy"), matches, f"{prefix}.price_accuracy"
        )
        indexed[fixture_id] = fixture

    _validate_aggregate_report(report, list(indexed.values()), label)
    return indexed


def _delta(baseline, candidate):
    if baseline is None or candidate is None or not _finite(baseline) or not _finite(candidate):
        return None
    return candidate - baseline


def _compare_metrics(baseline, candidate):
    return {
        field: {
            "baseline": baseline.get(field),
            "candidate": candidate.get(field),
            "delta": _delta(baseline.get(field), candidate.get(field)),
        }
        for field in _METRIC_FIELDS
    }


def _compare_price(baseline, candidate):
    fields = (
        "calibrated_fixture_count", "scored_zone_count", "unscored_zone_count",
        "mean_boundary_abs_error", "max_boundary_abs_error", "mean_boundary_error_pct"
    )
    if baseline is None and candidate is None:
        return None
    baseline, candidate = baseline or {}, candidate or {}
    return {
        field: {
            "baseline": baseline.get(field),
            "candidate": candidate.get(field),
            "delta": _delta(baseline.get(field), candidate.get(field)),
        }
        for field in fields
    }


def _compare_context(baseline_fixtures, candidate_fixtures, field):
    base_groups = {g["value"]: g for g in _aggregate_by_context(baseline_fixtures, field)}
    candidate_groups = {g["value"]: g for g in _aggregate_by_context(candidate_fixtures, field)}
    if set(base_groups) != set(candidate_groups):
        raise ValueError(f"paired reports have different {field} groups")
    output = []
    for value in sorted(base_groups, key=lambda item: (item is not None, (item or "").casefold())):
        base, candidate = base_groups[value], candidate_groups[value]
        output.append({
            "value": value,
            "fixture_count": base["fixture_count"],
            "metrics": _compare_metrics(base["metrics"], candidate["metrics"]),
            "price_accuracy": _compare_price(base["price_accuracy"], candidate["price_accuracy"]),
        })
    return output


def compare_benchmark_reports(baseline_report, candidate_report):
    """Compare reports only when every fixture and scoring input is identical."""
    baseline = _validate_report(baseline_report, "baseline")
    candidate = _validate_report(candidate_report, "candidate")
    if baseline_report["annotation_guideline_version"] != candidate_report["annotation_guideline_version"]:
        raise ValueError("paired reports use different annotation guideline versions")
    base_ids, candidate_ids = set(baseline), set(candidate)
    if base_ids != candidate_ids:
        raise ValueError(
            "paired reports must contain exactly the same fixture IDs "
            f"(missing from candidate: {sorted(base_ids - candidate_ids)}; "
            f"extra in candidate: {sorted(candidate_ids - base_ids)})"
        )

    for fixture_id in sorted(base_ids):
        base, cand = baseline[fixture_id], candidate[fixture_id]
        for field in (
            "image_sha256", "image_width", "image_height",
            "reference_zones_sha256", "reference_plot_bounds_sha256",
            "price_axis_calibration_sha256", "iou_threshold",
            "annotation_metadata", "market_context"
        ):
            if base.get(field) != cand.get(field):
                raise ValueError(f"paired reports differ for fixture {fixture_id!r}: {field}")

    base_fixtures = [baseline[key] for key in sorted(baseline)]
    candidate_fixtures = [candidate[key] for key in sorted(candidate)]
    base_matches = [m for f in base_fixtures for m in f["matches"]]
    candidate_matches = [m for f in candidate_fixtures for m in f["matches"]]
    base_overall = _aggregate_metrics([f["overall"] for f in base_fixtures], base_matches)
    candidate_overall = _aggregate_metrics(
        [f["overall"] for f in candidate_fixtures], candidate_matches
    )

    by_zone_type = {}
    for zone_type in ("supply", "demand"):
        base_rows = [f["by_zone_type"][zone_type] for f in base_fixtures]
        candidate_rows = [f["by_zone_type"][zone_type] for f in candidate_fixtures]
        base_type_matches = [m for m in base_matches if m["zone_type"] == zone_type]
        candidate_type_matches = [m for m in candidate_matches if m["zone_type"] == zone_type]
        by_zone_type[zone_type] = _compare_metrics(
            _aggregate_metrics(base_rows, base_type_matches),
            _aggregate_metrics(candidate_rows, candidate_type_matches),
        )

    fixture_results = []
    for fixture_id in sorted(baseline):
        base, cand = baseline[fixture_id], candidate[fixture_id]
        fixture_results.append({
            "fixture_id": fixture_id,
            "market_context": base["market_context"],
            "metrics": _compare_metrics(base["overall"], cand["overall"]),
            "ignored_prediction_count": {
                "baseline": base["ignored_prediction_count"],
                "candidate": cand["ignored_prediction_count"],
                "delta": cand["ignored_prediction_count"] - base["ignored_prediction_count"],
            },
            "price_accuracy": _compare_price(base["price_accuracy"], cand["price_accuracy"]),
            "plot_boundary_accuracy": compare_plot_boundary_accuracy(
                base["plot_boundary_accuracy"], cand["plot_boundary_accuracy"]
            ),
        })

    base_ignored = sum(f["ignored_prediction_count"] for f in base_fixtures)
    candidate_ignored = sum(f["ignored_prediction_count"] for f in candidate_fixtures)
    return {
        "fixture_count": len(base_fixtures),
        "annotation_guideline_version": baseline_report["annotation_guideline_version"],
        "baseline_prediction_profile": baseline_report["prediction_profile"],
        "baseline_prediction_profile_sha256": baseline_report["prediction_profile_sha256"],
        "candidate_prediction_profile": candidate_report["prediction_profile"],
        "candidate_prediction_profile_sha256": candidate_report["prediction_profile_sha256"],
        "same_prediction_profile": baseline_report["prediction_profile_sha256"] == candidate_report["prediction_profile_sha256"],
        "overall": _compare_metrics(base_overall, candidate_overall),
        "by_zone_type": by_zone_type,
        "by_asset_class": _compare_context(base_fixtures, candidate_fixtures, "asset_class"),
        "by_symbol": _compare_context(base_fixtures, candidate_fixtures, "symbol"),
        "by_exchange": _compare_context(base_fixtures, candidate_fixtures, "exchange"),
        "by_timeframe": _compare_context(base_fixtures, candidate_fixtures, "timeframe"),
        "ignored_prediction_count": {
            "baseline": base_ignored, "candidate": candidate_ignored,
            "delta": candidate_ignored - base_ignored
        },
        "price_accuracy": _compare_price(
            _aggregate_price_accuracy(base_fixtures),
            _aggregate_price_accuracy(candidate_fixtures),
        ),
        "plot_boundary_accuracy": compare_plot_boundary_accuracy(
            aggregate_plot_boundary_accuracy(base_fixtures),
            aggregate_plot_boundary_accuracy(candidate_fixtures),
        ),
        "fixtures": fixture_results,
    }


def _read_report(path, label):
    try:
        with Path(path).open("r", encoding="utf-8") as report_file:
            return json.load(report_file)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} report must be UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} report contains invalid JSON: {exc.msg}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare two DNS benchmark reports on identical reviewed chart fixtures."
    )
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("candidate_report", type=Path)
    args = parser.parse_args(argv)
    try:
        result = compare_benchmark_reports(
            _read_report(args.baseline_report, "baseline"),
            _read_report(args.candidate_report, "candidate"),
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
