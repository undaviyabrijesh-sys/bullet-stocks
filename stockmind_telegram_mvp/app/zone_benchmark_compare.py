"""Paired comparison of StockMind chart-zone benchmark reports."""
import argparse
import json
import math
import re
from pathlib import Path

from .zone_benchmark import (
    _aggregate_by_context,
    _aggregate_metrics,
    _aggregate_price_accuracy,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COUNT_FIELDS = ("predicted", "reference", "true_positive", "false_positive", "false_negative")
_METRIC_FIELDS = _COUNT_FIELDS + (
    "precision", "recall", "f1", "mean_iou", "mean_center_error_normalized"
)
_CONTEXT_FIELDS = ("asset_class", "symbol", "exchange", "timeframe")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


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
        value = row.get(field)
        if value is not None and not _finite(value):
            raise ValueError(f"{label}.{field} must be finite or null")
    for field in ("precision", "recall", "f1", "mean_iou"):
        value = row.get(field)
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"{label}.{field} must be between 0 and 1 or null")
    value = row.get("mean_center_error_normalized")
    if value is not None and value < 0:
        raise ValueError(f"{label}.mean_center_error_normalized cannot be negative")


def _validate_report(report, label):
    if not isinstance(report, dict):
        raise ValueError(f"{label} report must be a JSON object")
    profile_hash = report.get("prediction_profile_sha256")
    if not isinstance(profile_hash, str) or not _SHA256_RE.fullmatch(profile_hash):
        raise ValueError(f"{label} report has an invalid prediction_profile_sha256")
    if not isinstance(report.get("prediction_profile"), dict):
        raise ValueError(f"{label} report is missing prediction_profile")
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
        annotation = fixture.get("annotation_metadata")
        if not isinstance(annotation, dict):
            raise ValueError(f"{prefix} annotation_metadata must be a JSON object")
        if annotation.get("guideline_version") != guideline:
            raise ValueError(f"{prefix} annotation guideline does not match report")
        context = fixture.get("market_context")
        if not isinstance(context, dict) or set(context) != set(_CONTEXT_FIELDS):
            raise ValueError(f"{prefix} market_context has an invalid shape")
        threshold = fixture.get("iou_threshold")
        if not _finite(threshold) or not 0 < threshold <= 1:
            raise ValueError(f"{prefix} has an invalid iou_threshold")

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
        for match_index, match in enumerate(matches):
            if not isinstance(match, dict) or match.get("zone_type") not in {"supply", "demand"}:
                raise ValueError(f"{prefix} match {match_index} has an invalid shape or zone_type")
            for field in ("iou", "center_error_normalized"):
                value = match.get(field)
                if not _finite(value) or value < 0:
                    raise ValueError(f"{prefix} match {match_index}.{field} is invalid")
            if match["iou"] > 1:
                raise ValueError(f"{prefix} match {match_index}.iou cannot exceed 1")
        ignored = fixture.get("ignored_prediction_count")
        if isinstance(ignored, bool) or not isinstance(ignored, int) or ignored < 0:
            raise ValueError(f"{prefix} ignored_prediction_count must be non-negative")
        if fixture.get("price_accuracy") is not None and not isinstance(fixture["price_accuracy"], dict):
            raise ValueError(f"{prefix} price_accuracy must be an object or null")
        indexed[fixture_id] = fixture
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
            "image_sha256", "reference_zones_sha256", "price_axis_calibration_sha256",
            "iou_threshold", "annotation_metadata", "market_context"
        ):
            if base[field] != cand[field]:
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
