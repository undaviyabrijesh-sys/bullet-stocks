"""Reproducible offline runner for manually annotated StockMind chart fixtures.

Each JSONL record points to the exact screenshot, its SHA-256 and dimensions,
manual reference zones, and the AI zones returned for that same image. This
module never calls a model or market-data provider.
"""
import argparse
import hashlib
import json
import re
from io import BytesIO
from pathlib import Path, PurePosixPath

from PIL import Image

from .overlay_engine import validate_chart_image
from .zone_evaluation import evaluate_zone_predictions


_REQUIRED_FIELDS = {
    "fixture_id",
    "image_file",
    "image_sha256",
    "image_width",
    "image_height",
    "reference_zones",
    "predicted_zones",
}
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
DEFAULT_MAX_IMAGE_BYTES = 5 * 1024 * 1024


def _positive_integer(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


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


def _validate_record(record, manifest_dir, line_number, max_image_bytes):
    if not isinstance(record, dict):
        raise ValueError(f"line {line_number}: each fixture record must be a JSON object")
    missing = sorted(_REQUIRED_FIELDS - set(record))
    if missing:
        raise ValueError(f"line {line_number}: missing required fields: {', '.join(missing)}")

    fixture_id = record["fixture_id"]
    image_hash = _load_and_verify_image(record, manifest_dir, line_number, max_image_bytes)

    for field in ("reference_zones", "predicted_zones"):
        if not isinstance(record[field], list):
            raise ValueError(f"line {line_number} ({fixture_id}): {field} must be a JSON array")

    anchors = record.get("price_axis_anchors")
    scale = record.get("price_axis_scale")
    if scale is not None and anchors is None:
        raise ValueError(
            f"line {line_number} ({fixture_id}): price_axis_scale requires price_axis_anchors"
        )
    if anchors is not None and not isinstance(anchors, list):
        raise ValueError(f"line {line_number} ({fixture_id}): price_axis_anchors must be a JSON array")

    try:
        result = evaluate_zone_predictions(
            record["predicted_zones"],
            record["reference_zones"],
            iou_threshold=record.get("iou_threshold", 0.5),
            price_axis_anchors=anchors,
            price_axis_scale=scale or "linear",
            max_axis_relative_error=record.get("max_axis_relative_error", 0.02),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line_number} ({fixture_id}): {exc}") from exc

    return {
        "fixture_id": fixture_id,
        "image_sha256": image_hash,
        "overall": result["overall"],
        "by_zone_type": result["by_zone_type"],
        "ignored_prediction_count": result["ignored_prediction_count"],
        "matches": result["matches"],
        "price_accuracy": result["price_accuracy"],
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
                if fixture["fixture_id"] in seen_ids:
                    raise ValueError(
                        f"line {line_number}: duplicate fixture_id {fixture['fixture_id']!r}"
                    )
                seen_ids.add(fixture["fixture_id"])
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
        "ignored_prediction_count": sum(
            fixture["ignored_prediction_count"] for fixture in fixtures
        ),
        "overall": _aggregate_metrics(overall_rows, all_matches),
        "by_zone_type": by_zone_type,
        "price_accuracy": _aggregate_price_accuracy(fixtures),
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
