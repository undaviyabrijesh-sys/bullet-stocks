"""Reproducible offline runner for manually annotated StockMind chart fixtures.

Each JSONL record points to the exact screenshot, its SHA-256 and dimensions,
manual reference zones, and the AI zones returned for that same image. This
module never calls a model or market-data provider.
"""
import argparse
import hashlib
import json
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path, PurePosixPath

from PIL import Image

from .overlay_engine import validate_chart_image
from .zone_evaluation import evaluate_zone_predictions
from .plot_evaluation import aggregate_plot_boundary_accuracy


_REQUIRED_FIELDS = {
    "fixture_id",
    "image_file",
    "image_sha256",
    "image_width",
    "image_height",
    "reference_zones",
    "predicted_zones",
    "prediction_metadata",
    "annotation_metadata",
}
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ANNOTATOR_ID_RE = re.compile(r"^ann-[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_REVIEWER_ID_RE = re.compile(r"^rev-[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_GUIDELINE_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
DEFAULT_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_MARKET_CONTEXT_FIELDS = ("asset_class", "symbol", "exchange", "timeframe")


def _positive_integer(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _validate_prediction_metadata(metadata, line_number, fixture_id):
    if not isinstance(metadata, dict):
        raise ValueError(
            f"line {line_number} ({fixture_id}): prediction_metadata must be a JSON object"
        )

    string_fields = (
        "provider",
        "model",
        "requested_model",
        "prompt_version",
        "prompt_sha256",
        "generated_at",
    )
    normalized = {}
    for field in string_fields:
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"line {line_number} ({fixture_id}): prediction_metadata.{field} "
                "must be a non-empty string"
            )
        normalized[field] = value.strip()

    if not _SHA256_RE.fullmatch(normalized["prompt_sha256"]):
        raise ValueError(
            f"line {line_number} ({fixture_id}): prediction_metadata.prompt_sha256 "
            "must be a 64-character SHA-256 hex digest"
        )
    normalized["prompt_sha256"] = normalized["prompt_sha256"].lower()

    try:
        generated_at = datetime.fromisoformat(
            normalized["generated_at"].replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(
            f"line {line_number} ({fixture_id}): prediction_metadata.generated_at "
            "must be an ISO-8601 timestamp with a timezone"
        ) from exc
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError(
            f"line {line_number} ({fixture_id}): prediction_metadata.generated_at "
            "must include a timezone"
        )

    fingerprint = metadata.get("system_fingerprint")
    if fingerprint is not None and (
        not isinstance(fingerprint, str) or not fingerprint.strip()
    ):
        raise ValueError(
            f"line {line_number} ({fixture_id}): system_fingerprint must be a non-empty string or null"
        )
    normalized["system_fingerprint"] = fingerprint.strip() if fingerprint else None

    settings = metadata.get("settings")
    if not isinstance(settings, dict):
        raise ValueError(
            f"line {line_number} ({fixture_id}): prediction_metadata.settings must be a JSON object"
        )
    normalized["settings"] = settings

    # Generated time is fixture-specific; the remaining fields define the
    # inference profile and must be identical within one aggregate report.
    profile = {
        "provider": normalized["provider"],
        "model": normalized["model"],
        "requested_model": normalized["requested_model"],
        "prompt_version": normalized["prompt_version"],
        "prompt_sha256": normalized["prompt_sha256"],
        "system_fingerprint": normalized["system_fingerprint"],
        "settings": settings,
    }
    try:
        canonical_profile = json.dumps(
            profile, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"line {line_number} ({fixture_id}): prediction metadata settings "
            "must contain finite JSON values"
        ) from exc
    profile_sha256 = hashlib.sha256(canonical_profile.encode("utf-8")).hexdigest()
    return normalized, profile, profile_sha256


def _validate_annotation_metadata(metadata, line_number, fixture_id):
    """Require approved, independently reviewed annotations with pseudonymous IDs.

    The allow-listed schema intentionally excludes names, email addresses and
    other personal details from benchmark reports.
    """
    prefix = f"line {line_number} ({fixture_id}): annotation_metadata"
    if not isinstance(metadata, dict):
        raise ValueError(f"{prefix} must be a JSON object")

    required = {
        "source",
        "annotator_id",
        "reviewer_id",
        "guideline_version",
        "review_status",
        "annotated_at",
        "reviewed_at",
    }
    missing = sorted(required - set(metadata))
    extra = sorted(set(metadata) - required)
    if missing:
        raise ValueError(f"{prefix} is missing fields: {', '.join(missing)}")
    if extra:
        raise ValueError(
            f"{prefix} contains unsupported fields: {', '.join(extra)}; "
            "store only anonymized annotation provenance"
        )

    if metadata["source"] != "manual":
        raise ValueError(f"{prefix}.source must be 'manual'")

    annotator_id = metadata["annotator_id"]
    if not isinstance(annotator_id, str) or not _ANNOTATOR_ID_RE.fullmatch(annotator_id):
        raise ValueError(
            f"{prefix}.annotator_id must be an anonymized ID like 'ann-001'"
        )
    reviewer_id = metadata["reviewer_id"]
    if not isinstance(reviewer_id, str) or not _REVIEWER_ID_RE.fullmatch(reviewer_id):
        raise ValueError(
            f"{prefix}.reviewer_id must be a separate anonymized ID like 'rev-001'"
        )
    if annotator_id == reviewer_id:
        raise ValueError(f"{prefix} requires a reviewer distinct from the annotator")

    guideline_version = metadata["guideline_version"]
    if (
        not isinstance(guideline_version, str)
        or not _GUIDELINE_VERSION_RE.fullmatch(guideline_version)
    ):
        raise ValueError(
            f"{prefix}.guideline_version must be a non-empty version token"
        )
    if metadata["review_status"] != "approved":
        raise ValueError(f"{prefix}.review_status must be 'approved'")

    parsed_times = {}
    for field in ("annotated_at", "reviewed_at"):
        value = metadata[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{prefix}.{field} must be an ISO-8601 timestamp")
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"{prefix}.{field} must be an ISO-8601 timestamp with a timezone"
            ) from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"{prefix}.{field} must include a timezone")
        parsed_times[field] = parsed
    if parsed_times["reviewed_at"] < parsed_times["annotated_at"]:
        raise ValueError(f"{prefix}.reviewed_at cannot be earlier than annotated_at")

    return {
        "source": "manual",
        "annotator_id": annotator_id,
        "reviewer_id": reviewer_id,
        "guideline_version": guideline_version,
        "review_status": "approved",
        "annotated_at": metadata["annotated_at"].strip(),
        "reviewed_at": metadata["reviewed_at"].strip(),
    }


def _image_path(manifest_dir, image_file):
    if not isinstance(image_file, str) or not image_file.strip():
        raise ValueError("image_file must be a non-empty relative path")
    if "\\" in image_file:
        raise ValueError("image_file must use portable forward-slash path separators")
    relative = PurePosixPath(image_file)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("image_file must stay inside the manifest directory")
    root = manifest_dir.resolve()
    resolved = (root / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("image_file resolves outside the manifest directory") from exc
    if not resolved.is_file():
        raise ValueError(f"fixture image does not exist: {image_file}")
    return resolved


def _load_and_verify_image(record, manifest_dir, line_number, max_image_bytes):
    fixture_id = record.get("fixture_id")
    if not isinstance(fixture_id, str) or not fixture_id.strip():
        raise ValueError(f"line {line_number}: fixture_id must be a non-empty string")

    expected_hash = record.get("image_sha256")
    if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(expected_hash):
        raise ValueError(f"line {line_number}: image_sha256 must be a 64-character SHA-256 hex digest")

    expected_width = _positive_integer(record.get("image_width"), "image_width")
    expected_height = _positive_integer(record.get("image_height"), "image_height")
    image_path = _image_path(manifest_dir, record.get("image_file"))

    # Bound the read itself, not just a later size check, so a malformed or
    # accidentally huge local fixture cannot exhaust memory before validation.
    with image_path.open("rb") as image_file:
        image_bytes = image_file.read(max_image_bytes + 1)
    if len(image_bytes) > max_image_bytes:
        raise ValueError(
            f"line {line_number} ({fixture_id}): screenshot exceeds max_image_bytes "
            f"({max_image_bytes})"
        )

    actual_hash = hashlib.sha256(image_bytes).hexdigest()
    if actual_hash.lower() != expected_hash.lower():
        raise ValueError(f"line {line_number} ({fixture_id}): screenshot SHA-256 does not match")

    # Reuse the production upload validator so the benchmark accepts only the
    # same static image formats and decoded-pixel ceiling as the Telegram bot.
    validate_chart_image(image_bytes)
    # Inspect the same bounded bytes that were hashed and validated; do not
    # reopen a path that could have changed between those operations.
    with Image.open(BytesIO(image_bytes)) as image:
        actual_width, actual_height = image.size
    if (actual_width, actual_height) != (expected_width, expected_height):
        raise ValueError(
            f"line {line_number} ({fixture_id}): screenshot dimensions "
            f"{actual_width}x{actual_height} do not match the manifest "
            f"{expected_width}x{expected_height}"
        )
    return actual_hash.lower()


def _canonical_json_sha256(value):
    """Hash finite JSON with stable key ordering for paired benchmark checks."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_market_context(context, line_number, fixture_id):
    """Validate optional chart identity fields used for stratified metrics."""
    prefix = f"line {line_number} ({fixture_id}): market_context"
    if context is None:
        context = {}
    if not isinstance(context, dict):
        raise ValueError(f"{prefix} must be a JSON object")

    extra = sorted(set(context) - set(_MARKET_CONTEXT_FIELDS))
    if extra:
        raise ValueError(f"{prefix} contains unsupported fields: {', '.join(extra)}")

    normalized = {}
    for field in _MARKET_CONTEXT_FIELDS:
        value = context.get(field)
        if value is None:
            normalized[field] = None
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{prefix}.{field} must be a non-empty string or null")
        else:
            cleaned = value.strip()
            max_length = {
                "asset_class": 32,
                "symbol": 64,
                "exchange": 64,
                "timeframe": 16,
            }[field]
            if len(cleaned) > max_length:
                raise ValueError(
                    f"{prefix}.{field} cannot exceed {max_length} characters"
                )
            if any(not character.isprintable() for character in cleaned):
                raise ValueError(
                    f"{prefix}.{field} cannot contain control characters"
                )
            # Canonicalize case so BTCUSDT/btcusdt, Binance/BINANCE and 1H/1h
            # do not split one real-world chart category into separate groups.
            if field == "symbol":
                cleaned = cleaned.upper()
            else:
                cleaned = cleaned.casefold()
            normalized[field] = cleaned
    return normalized


def _validate_record(record, manifest_dir, line_number, max_image_bytes):
    if not isinstance(record, dict):
        raise ValueError(f"line {line_number}: each fixture record must be a JSON object")
    missing = sorted(_REQUIRED_FIELDS - set(record))
    if missing:
        raise ValueError(f"line {line_number}: missing required fields: {', '.join(missing)}")

    fixture_id = record["fixture_id"]
    prediction_metadata, prediction_profile, prediction_profile_sha256 = (
        _validate_prediction_metadata(
            record["prediction_metadata"], line_number, fixture_id
        )
    )
    annotation_metadata = _validate_annotation_metadata(
        record["annotation_metadata"], line_number, fixture_id
    )
    market_context = _validate_market_context(
        record.get("market_context"), line_number, fixture_id
    )
    image_hash = _load_and_verify_image(record, manifest_dir, line_number, max_image_bytes)

    for field in ("reference_zones", "predicted_zones"):
        if not isinstance(record[field], list):
            raise ValueError(f"line {line_number} ({fixture_id}): {field} must be a JSON array")

    anchors = record.get("price_axis_anchors")
    axis_plot_bounds = record.get("price_axis_plot_bounds")
    scale = record.get("price_axis_scale")
    if scale is not None and anchors is None:
        raise ValueError(
            f"line {line_number} ({fixture_id}): price_axis_scale requires price_axis_anchors"
        )
    if anchors is not None and not isinstance(anchors, list):
        raise ValueError(f"line {line_number} ({fixture_id}): price_axis_anchors must be a JSON array")
    if anchors is not None and axis_plot_bounds is None:
        raise ValueError(
            f"line {line_number} ({fixture_id}): price_axis_plot_bounds is required with price_axis_anchors"
        )
    if anchors is None and axis_plot_bounds is not None:
        raise ValueError(
            f"line {line_number} ({fixture_id}): price_axis_plot_bounds requires price_axis_anchors"
        )

    reference_plot_bounds = record.get("reference_plot_bounds")
    if reference_plot_bounds is None and axis_plot_bounds is not None:
        reference_plot_bounds = axis_plot_bounds
    predicted_plot_bounds = record.get("predicted_plot_bounds")

    try:
        result = evaluate_zone_predictions(
            record["predicted_zones"],
            record["reference_zones"],
            image_width=record["image_width"],
            image_height=record["image_height"],
            iou_threshold=record.get("iou_threshold", 0.5),
            price_axis_anchors=anchors,
            price_axis_plot_bounds=axis_plot_bounds,
            price_axis_scale=scale or "linear",
            max_axis_relative_error=record.get("max_axis_relative_error", 0.02),
            predicted_plot_bounds=predicted_plot_bounds,
            reference_plot_bounds=reference_plot_bounds,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line_number} ({fixture_id}): {exc}") from exc

    try:
        reference_zones_sha256 = _canonical_json_sha256(record["reference_zones"])
        reference_plot_bounds_sha256 = (
            _canonical_json_sha256(reference_plot_bounds)
            if reference_plot_bounds is not None else None
        )
        price_axis_calibration_sha256 = (
            _canonical_json_sha256({
                "anchors": anchors,
                "plot_bounds": axis_plot_bounds,
                "scale": scale or "linear",
                "max_axis_relative_error": record.get("max_axis_relative_error", 0.02),
            })
            if anchors is not None else None
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"line {line_number} ({fixture_id}): reference zones or price-axis "
            "calibration must contain finite JSON values"
        ) from exc

    return {
        "fixture_id": fixture_id,
        "image_sha256": image_hash,
        "image_width": record["image_width"],
        "image_height": record["image_height"],
        "reference_zones_sha256": reference_zones_sha256,
        "reference_plot_bounds": reference_plot_bounds,
        "predicted_plot_bounds": predicted_plot_bounds,
        "reference_plot_bounds_sha256": reference_plot_bounds_sha256,
        "price_axis_calibration_sha256": price_axis_calibration_sha256,
        "iou_threshold": result["iou_threshold"],
        "prediction_metadata": prediction_metadata,
        "annotation_metadata": annotation_metadata,
        "market_context": market_context,
        "_prediction_profile": prediction_profile,
        "_prediction_profile_sha256": prediction_profile_sha256,
        "overall": result["overall"],
        "by_zone_type": result["by_zone_type"],
        "ignored_prediction_count": result["ignored_prediction_count"],
        "matches": result["matches"],
        "price_accuracy": result["price_accuracy"],
        "plot_boundary_accuracy": result["plot_boundary_accuracy"],
    }


def _aggregate_metrics(rows, matches):
    predicted = sum(row["predicted"] for row in rows)
    reference = sum(row["reference"] for row in rows)
    true_positive = sum(row["true_positive"] for row in rows)
    false_positive = sum(row["false_positive"] for row in rows)
    false_negative = sum(row["false_negative"] for row in rows)
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "predicted": predicted,
        "reference": reference,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": true_positive / predicted if predicted else None,
        "recall": true_positive / reference if reference else None,
        "f1": 2 * true_positive / denominator if denominator else None,
        "mean_iou": (
            sum(match["iou"] for match in matches) / len(matches) if matches else None
        ),
        "mean_center_error_normalized": (
            sum(match["center_error_normalized"] for match in matches) / len(matches)
            if matches else None
        ),
    }


def _aggregate_price_accuracy(fixtures):
    calibrated_fixtures = [
        fixture for fixture in fixtures if fixture["price_accuracy"] is not None
    ]
    comparisons = [
        match["price_comparison"]
        for fixture in calibrated_fixtures
        for match in fixture["matches"]
        if match["price_comparison"] is not None
    ]
    scored = [item for item in comparisons if item["status"] == "scored"]
    unscored_count = sum(item["status"] == "unscored" for item in comparisons)
    absolute_errors = [
        item[field]
        for item in scored
        for field in ("price_high_abs_error", "price_low_abs_error")
    ]
    percentage_errors = [
        item[field]
        for item in scored
        for field in ("price_high_error_pct", "price_low_error_pct")
    ]
    if not calibrated_fixtures:
        return None
    return {
        "calibrated_fixture_count": len(calibrated_fixtures),
        "scored_zone_count": len(scored),
        "unscored_zone_count": unscored_count,
        "mean_boundary_abs_error": (
            sum(absolute_errors) / len(absolute_errors) if absolute_errors else None
        ),
        "max_boundary_abs_error": max(absolute_errors) if absolute_errors else None,
        "mean_boundary_error_pct": (
            sum(percentage_errors) / len(percentage_errors) if percentage_errors else None
        ),
    }


def _aggregate_by_context(fixtures, field):
    """Return micro-aggregated metrics grouped by one chart-context dimension.

    Missing context is retained as a null group rather than silently dropped.
    """
    groups = {}
    for fixture in fixtures:
        value = fixture["market_context"][field]
        groups.setdefault(value, []).append(fixture)

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (item[0] is not None, (item[0] or "").casefold()),
    )
    output = []
    for value, group in ordered_groups:
        matches = [match for fixture in group for match in fixture["matches"]]
        rows = [fixture["overall"] for fixture in group]
        output.append({
            "value": value,
            "fixture_count": len(group),
            "metrics": _aggregate_metrics(rows, matches),
            "price_accuracy": _aggregate_price_accuracy(group),
        })
    return output


def evaluate_benchmark_manifest(manifest_path, *, max_image_bytes=DEFAULT_MAX_IMAGE_BYTES):
    """Validate screenshots and score every JSONL fixture in a manifest.

    The default image-byte ceiling matches the Telegram bot's 5 MiB default.
    Pass an explicit limit when the bot is configured with a different ceiling.
    """
    max_image_bytes = _positive_integer(max_image_bytes, "max_image_bytes")
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"benchmark manifest does not exist: {manifest_path}")

    fixtures = []
    seen_ids = set()
    seen_image_hashes = set()
    prediction_profile = None
    prediction_profile_sha256 = None
    annotation_guideline_version = None
    try:
        with manifest_path.open("r", encoding="utf-8") as manifest:
            for line_number, raw_line in enumerate(manifest, start=1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"line {line_number}: invalid JSON ({exc.msg})"
                    ) from exc
                fixture = _validate_record(
                    record, manifest_path.parent, line_number, max_image_bytes
                )
                fixture_guideline = fixture["annotation_metadata"]["guideline_version"]
                if annotation_guideline_version is None:
                    annotation_guideline_version = fixture_guideline
                elif fixture_guideline != annotation_guideline_version:
                    raise ValueError(
                        f"line {line_number} ({fixture['fixture_id']}): mixed annotation "
                        "guideline versions; split this manifest by guideline version"
                    )
                if fixture["fixture_id"] in seen_ids:
                    raise ValueError(
                        f"line {line_number}: duplicate fixture_id {fixture['fixture_id']!r}"
                    )
                seen_ids.add(fixture["fixture_id"])

                # A manifest represents distinct source screenshots. Reusing
                # identical bytes under another fixture_id silently overweights
                # that chart in micro-aggregated benchmark metrics.
                image_hash = fixture["image_sha256"]
                if image_hash in seen_image_hashes:
                    raise ValueError(
                        f"line {line_number}: duplicate screenshot image_sha256 "
                        f"{image_hash!r}; each fixture must use a unique screenshot"
                    )
                seen_image_hashes.add(image_hash)
                if prediction_profile_sha256 is None:
                    prediction_profile = fixture.pop("_prediction_profile")
                    prediction_profile_sha256 = fixture.pop("_prediction_profile_sha256")
                else:
                    fixture_profile = fixture.pop("_prediction_profile")
                    fixture_profile_sha256 = fixture.pop("_prediction_profile_sha256")
                    if fixture_profile_sha256 != prediction_profile_sha256:
                        raise ValueError(
                            f"line {line_number} ({fixture['fixture_id']}): mixed prediction "
                            "profiles; split this manifest by provider, model, prompt and settings"
                        )
                fixtures.append(fixture)
    except UnicodeDecodeError as exc:
        raise ValueError("benchmark manifest must be UTF-8 JSONL") from exc

    if not fixtures:
        raise ValueError("benchmark manifest contains no fixture records")

    all_matches = [match for fixture in fixtures for match in fixture["matches"]]
    overall_rows = [fixture["overall"] for fixture in fixtures]
    by_zone_type = {}
    for zone_type in ("supply", "demand"):
        rows = [fixture["by_zone_type"][zone_type] for fixture in fixtures]
        matches = [match for match in all_matches if match["zone_type"] == zone_type]
        by_zone_type[zone_type] = _aggregate_metrics(rows, matches)

    return {
        "fixture_count": len(fixtures),
        "prediction_profile": prediction_profile,
        "prediction_profile_sha256": prediction_profile_sha256,
        "annotation_guideline_version": annotation_guideline_version,
        "ignored_prediction_count": sum(
            fixture["ignored_prediction_count"] for fixture in fixtures
        ),
        "overall": _aggregate_metrics(overall_rows, all_matches),
        "by_zone_type": by_zone_type,
        "by_asset_class": _aggregate_by_context(fixtures, "asset_class"),
        "by_symbol": _aggregate_by_context(fixtures, "symbol"),
        "by_exchange": _aggregate_by_context(fixtures, "exchange"),
        "by_timeframe": _aggregate_by_context(fixtures, "timeframe"),
        "price_accuracy": _aggregate_price_accuracy(fixtures),
        "plot_boundary_accuracy": aggregate_plot_boundary_accuracy(fixtures),
        "fixtures": fixtures,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score StockMind AI DNS overlays against manually annotated chart fixtures."
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="JSONL manifest; image_file paths are relative to this file's directory",
    )
    parser.add_argument(
        "--max-image-bytes",
        type=int,
        default=DEFAULT_MAX_IMAGE_BYTES,
        help="maximum bytes read per screenshot (default: 5 MiB)",
    )
    args = parser.parse_args(argv)
    try:
        result = evaluate_benchmark_manifest(
            args.manifest, max_image_bytes=args.max_image_bytes
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
