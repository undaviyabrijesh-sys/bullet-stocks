"""Pair an independently reviewed annotation with an actual chart-model result.

The capture and assembly steps are offline. They never call a model, market-data
provider, Telegram or a network service. The source screenshot is referenced by
hash and is never copied into the prediction capture.
"""

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

from .strict_json import loads as strict_json_loads
from .ai import sanitize_chart_analysis
from .annotation_editor import (
    load_draft_and_image,
    validate_manual_geometry,
    validate_price_axis_annotations,
    validate_reference_zone_pixel_geometry,
)
from .zone_benchmark import (
    DEFAULT_MAX_IMAGE_BYTES,
    _validate_annotation_metadata,
    _validate_market_context,
    _validate_prediction_metadata,
    evaluate_benchmark_manifest,
)

_CAPTURE_SCHEMA = "stockmind.chart-prediction-capture.v1"
_FIXTURE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_MAX_JSON_BYTES = 2 * 1024 * 1024


def _canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _read_json(path, *, max_bytes=_MAX_JSON_BYTES):
    path = Path(path).expanduser()
    with path.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"JSON input exceeds the {max_bytes}-byte limit")
    try:
        value = strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(
            "JSON input must be valid UTF-8 JSON with unique object keys and standard numeric constants: "
            f"{exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("JSON input must contain an object")
    return value


def _validate_reviewed_draft(draft):
    if not isinstance(draft, dict):
        raise ValueError("reviewed annotation must be a JSON object")
    if draft.get("draft_schema") != "stockmind.annotation-draft.v1":
        raise ValueError("unsupported annotation draft schema")
    if draft.get("benchmark_ready") is not False:
        raise ValueError("reviewed annotation must remain benchmark_ready=false")
    if draft.get("status") != "approved_pending_prediction":
        raise ValueError("annotation must be independently approved before prediction capture")
    fixture_id = draft.get("fixture_id")
    if not isinstance(fixture_id, str) or not _FIXTURE_ID_RE.fullmatch(fixture_id):
        raise ValueError("reviewed annotation fixture_id is invalid")
    if draft.get("prediction_status") != "not_captured":
        raise ValueError("reviewed annotation must not already contain prediction data")
    if any(draft.get(key) is not None for key in (
        "predicted_plot_bounds", "predicted_zones", "prediction_metadata"
    )):
        raise ValueError("reviewed annotation must not contain pre-existing AI prediction data")

    annotation_metadata = _validate_annotation_metadata(
        draft.get("annotation_metadata"), 1, fixture_id
    )
    market_context = _validate_market_context(
        draft.get("market_context"), 1, fixture_id
    )
    plot_bounds, reference_zones = validate_manual_geometry(
        draft.get("reference_plot_bounds"), draft.get("reference_zones")
    )
    validate_reference_zone_pixel_geometry(
        reference_zones, draft.get("image_width"), draft.get("image_height")
    )

    anchors = draft.get("price_axis_anchors", [])
    if not isinstance(anchors, list):
        raise ValueError("price_axis_anchors must be a JSON array")
    axis_scale = draft.get("price_axis_scale")
    axis_plot_bounds = draft.get("price_axis_plot_bounds")
    if anchors:
        normalized_anchors, normalized_scale = validate_price_axis_annotations(
            plot_bounds, anchors, axis_scale
        )
        normalized_axis_plot, _ = validate_manual_geometry(axis_plot_bounds, [])
        if normalized_axis_plot != plot_bounds:
            raise ValueError("price_axis_plot_bounds must match reference_plot_bounds")
        axis_plot_bounds = normalized_axis_plot
        axis_scale = normalized_scale
    elif axis_scale is not None or axis_plot_bounds is not None:
        raise ValueError("price-axis scale and plot bounds require at least 3 anchors")
    else:
        normalized_anchors = []

    return {
        "fixture_id": fixture_id,
        "image_file": draft.get("image_file"),
        "image_sha256": draft.get("image_sha256"),
        "image_width": draft.get("image_width"),
        "image_height": draft.get("image_height"),
        "reference_zones": reference_zones,
        "reference_plot_bounds": plot_bounds,
        "price_axis_anchors": normalized_anchors,
        "price_axis_scale": axis_scale,
        "price_axis_plot_bounds": axis_plot_bounds,
        "annotation_metadata": annotation_metadata,
        "market_context": market_context,
    }


def _validate_sanitized_prediction(result, fixture_id):
    if not isinstance(result, dict):
        raise ValueError("chart prediction result must be a JSON object")
    try:
        sanitized = sanitize_chart_analysis(result)
    except (TypeError, ValueError) as exc:
        raise ValueError("chart prediction is not a valid sanitized ResearchAI result") from exc

    supplied = {
        "analysis": result.get("analysis"),
        "plot_bounds": result.get("plot_bounds"),
        "zones": result.get("zones"),
    }
    try:
        matches_sanitized = _canonical_json(sanitized) == _canonical_json(supplied)
    except (TypeError, ValueError) as exc:
        raise ValueError("chart prediction must contain finite JSON values") from exc
    if not matches_sanitized:
        raise ValueError(
            "chart prediction must be the sanitized ResearchAI result; "
            "malformed or altered overlay geometry is not accepted"
        )

    metadata, _, _ = _validate_prediction_metadata(
        result.get("prediction_metadata"), 1, fixture_id
    )
    return sanitized, metadata


def build_prediction_capture(reviewed_draft_path, chart_result):
    """Bind a sanitized AI result to the exact independently reviewed screenshot."""
    reviewed_draft_path = Path(reviewed_draft_path).expanduser().resolve()
    draft, _ = load_draft_and_image(reviewed_draft_path)
    reviewed = _validate_reviewed_draft(draft)

    source_hash = (
        chart_result.get("source_image_sha256")
        if isinstance(chart_result, dict) else None
    )
    if not isinstance(source_hash, str) or not _SHA256_RE.fullmatch(source_hash):
        raise ValueError(
            "chart result is missing source_image_sha256; use the current "
            "ResearchAI.analyze_chart_image result"
        )
    if source_hash.lower() != reviewed["image_sha256"].lower():
        raise ValueError("AI prediction screenshot SHA-256 does not match the reviewed annotation")

    sanitized, metadata = _validate_sanitized_prediction(
        chart_result, reviewed["fixture_id"]
    )
    return {
        "capture_schema": _CAPTURE_SCHEMA,
        "fixture_id": reviewed["fixture_id"],
        "image_sha256": reviewed["image_sha256"].lower(),
        "image_width": reviewed["image_width"],
        "image_height": reviewed["image_height"],
        "prediction": {
            **sanitized,
            "prediction_metadata": metadata,
        },
    }


def _write_new_json(path, value):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            created = True
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except Exception:
        if created:
            path.unlink(missing_ok=True)
        raise
    return path.resolve()


def write_prediction_capture(reviewed_draft_path, chart_result, output_path=None):
    """Write a hash-bound capture without storing the screenshot or user identity."""
    reviewed_draft_path = Path(reviewed_draft_path).expanduser().resolve()
    capture = build_prediction_capture(reviewed_draft_path, chart_result)
    if output_path is None:
        output_path = reviewed_draft_path.with_name(
            f"{capture['fixture_id']}.prediction.json"
        )
    output_path = Path(output_path).expanduser()
    if output_path.resolve(strict=False) == reviewed_draft_path:
        raise ValueError("prediction capture must not overwrite the reviewed annotation")
    saved = _write_new_json(output_path, capture)
    return {
        "fixture_id": capture["fixture_id"],
        "capture_file": str(saved),
        "image_sha256": capture["image_sha256"],
        "benchmark_ready": False,
    }


def _load_capture(capture_path, reviewed):
    capture = _read_json(capture_path)
    expected_fields = {
        "capture_schema", "fixture_id", "image_sha256",
        "image_width", "image_height", "prediction",
    }
    if set(capture) != expected_fields:
        raise ValueError("prediction capture fields do not match the supported schema")
    if capture.get("capture_schema") != _CAPTURE_SCHEMA:
        raise ValueError("unsupported prediction capture schema")
    if capture.get("fixture_id") != reviewed["fixture_id"]:
        raise ValueError("prediction capture fixture_id does not match the reviewed annotation")
    image_hash = capture.get("image_sha256")
    if not isinstance(image_hash, str) or not _SHA256_RE.fullmatch(image_hash):
        raise ValueError("prediction capture image_sha256 must be a SHA-256 hex digest")
    if image_hash.lower() != reviewed["image_sha256"].lower():
        raise ValueError("prediction capture screenshot SHA-256 does not match the reviewed annotation")
    for field in ("image_width", "image_height"):
        value = capture.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"prediction capture {field} must be a positive integer")
        if value != reviewed[field]:
            raise ValueError(f"prediction capture {field} does not match the reviewed screenshot")

    prediction = capture.get("prediction")
    if not isinstance(prediction, dict) or set(prediction) != {
        "analysis", "plot_bounds", "zones", "prediction_metadata"
    }:
        raise ValueError("prediction payload fields do not match the supported schema")
    sanitized, metadata = _validate_sanitized_prediction(
        prediction, reviewed["fixture_id"]
    )
    return {
        **sanitized,
        "prediction_metadata": metadata,
    }


def _write_new_jsonl(path, record):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            created = True
            stream.write(_canonical_json(record) + "\n")
    except Exception:
        if created:
            path.unlink(missing_ok=True)
        raise
    return path.resolve()


def assemble_benchmark_fixture(
    reviewed_draft_path,
    capture_path,
    output_path=None,
    *,
    max_image_bytes=DEFAULT_MAX_IMAGE_BYTES,
):
    """Create and preflight one benchmark JSONL record; never overwrite files.

    The manifest is kept beside the reviewed draft so its relative image_file
    continues to resolve to the exact screenshot used for annotation.
    """
    if (
        isinstance(max_image_bytes, bool)
        or not isinstance(max_image_bytes, int)
        or max_image_bytes <= 0
    ):
        raise ValueError("max_image_bytes must be a positive integer")
    reviewed_draft_path = Path(reviewed_draft_path).expanduser().resolve()
    capture_path = Path(capture_path).expanduser().resolve()
    draft, _ = load_draft_and_image(reviewed_draft_path)
    reviewed = _validate_reviewed_draft(draft)
    prediction = _load_capture(capture_path, reviewed)

    output = (
        reviewed_draft_path.with_name(f"{reviewed['fixture_id']}.fixture.jsonl")
        if output_path is None else Path(output_path).expanduser()
    )
    output_resolved = output.resolve(strict=False)
    if output_resolved.parent != reviewed_draft_path.parent:
        raise ValueError(
            "fixture JSONL must be written beside the reviewed annotation so image_file remains relative"
        )
    if output_resolved in {reviewed_draft_path, capture_path}:
        raise ValueError("fixture JSONL must not overwrite an input file")
    if output_resolved.exists():
        raise FileExistsError(
            f"fixture output already exists; refusing to overwrite: {output_resolved}"
        )

    record = {
        "fixture_id": reviewed["fixture_id"],
        "image_file": reviewed["image_file"],
        "image_sha256": reviewed["image_sha256"],
        "image_width": reviewed["image_width"],
        "image_height": reviewed["image_height"],
        "reference_zones": reviewed["reference_zones"],
        "predicted_zones": prediction["zones"],
        "prediction_metadata": prediction["prediction_metadata"],
        "annotation_metadata": reviewed["annotation_metadata"],
        "market_context": reviewed["market_context"],
        "reference_plot_bounds": reviewed["reference_plot_bounds"],
        "predicted_plot_bounds": prediction["plot_bounds"],
    }
    if reviewed["price_axis_anchors"]:
        record.update({
            "price_axis_anchors": reviewed["price_axis_anchors"],
            "price_axis_scale": reviewed["price_axis_scale"],
            "price_axis_plot_bounds": reviewed["price_axis_plot_bounds"],
        })

    # Preflight with the exact benchmark validator before creating the output.
    # A valid fixture is scoreable; it is not evidence of model quality.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="x", encoding="utf-8", newline="\n", suffix=".jsonl",
            prefix=".stockmind-fixture-check-", dir=reviewed_draft_path.parent,
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(_canonical_json(record) + "\n")
        validation = evaluate_benchmark_manifest(
            temp_path, max_image_bytes=max_image_bytes
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    saved = _write_new_jsonl(output_resolved, record)
    return {
        "fixture_id": reviewed["fixture_id"],
        "manifest_file": str(saved),
        "image_sha256": reviewed["image_sha256"],
        "validation_fixture_count": validation["fixture_count"],
        "benchmark_ready": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Capture and assemble reviewed StockMind DNS benchmark fixtures."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    capture_parser = commands.add_parser(
        "capture", help="bind a saved ResearchAI chart result to a reviewed screenshot"
    )
    capture_parser.add_argument("reviewed_draft", type=Path)
    capture_parser.add_argument("chart_result", type=Path)
    capture_parser.add_argument("--output", type=Path)

    assemble_parser = commands.add_parser(
        "assemble", help="create one benchmark JSONL record and preflight it"
    )
    assemble_parser.add_argument("reviewed_draft", type=Path)
    assemble_parser.add_argument("prediction_capture", type=Path)
    assemble_parser.add_argument("--output", type=Path)
    assemble_parser.add_argument(
        "--max-image-bytes", type=int, default=DEFAULT_MAX_IMAGE_BYTES
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            chart_result = _read_json(args.chart_result)
            output = write_prediction_capture(
                args.reviewed_draft, chart_result, args.output
            )
        else:
            output = assemble_benchmark_fixture(
                args.reviewed_draft,
                args.prediction_capture,
                args.output,
                max_image_bytes=args.max_image_bytes,
            )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
