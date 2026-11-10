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
from datetime import datetime
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
    _validate_adjudication_metadata,
    _validate_annotation_metadata,
    _validate_market_context,
    _validate_prediction_metadata,
    evaluate_benchmark_manifest,
)

_CAPTURE_SCHEMA = "stockmind.chart-prediction-capture.v1"
_ADJUDICATION_SCHEMA = "stockmind.annotation-adjudication.v1"
_ADJUDICATOR_ID_RE = re.compile(r"^adj-[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
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


def _file_sha256(path):
    path = Path(path).expanduser()
    with path.open("rb") as stream:
        raw = stream.read(_MAX_JSON_BYTES + 1)
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError(f"annotation JSON exceeds the {_MAX_JSON_BYTES}-byte limit")
    return hashlib.sha256(raw).hexdigest()


def _load_reviewed_annotation(path):
    draft, image = load_draft_and_image(Path(path).expanduser())
    try:
        return _validate_reviewed_draft(draft)
    finally:
        image.close()


def _load_reviewed_pair(annotation_a_path, annotation_b_path):
    from .annotation_agreement import compare_reviewed_annotations

    reviewed_a = _load_reviewed_annotation(annotation_a_path)
    reviewed_b = _load_reviewed_annotation(annotation_b_path)
    agreement_report = compare_reviewed_annotations(
        annotation_a_path, annotation_b_path
    )
    return reviewed_a, reviewed_b, agreement_report


def build_adjudication_record(
    annotation_a_path,
    annotation_b_path,
    *,
    adjudicator_id,
    adjudicated_at,
    decision,
    rationale,
):
    """Bind a human selection to two exact, independently approved annotations."""
    annotation_a_path = Path(annotation_a_path).expanduser().resolve()
    annotation_b_path = Path(annotation_b_path).expanduser().resolve()
    if annotation_a_path == annotation_b_path:
        raise ValueError("adjudication requires two different reviewed annotation files")
    reviewed_a, reviewed_b, agreement_report = _load_reviewed_pair(
        annotation_a_path, annotation_b_path
    )
    if not isinstance(decision, str) or decision not in {"accept_a", "accept_b"}:
        raise ValueError("decision must be accept_a or accept_b")
    if not isinstance(adjudicator_id, str) or not _ADJUDICATOR_ID_RE.fullmatch(adjudicator_id):
        raise ValueError("adjudicator_id must be an anonymized ID like 'adj-003'")
    if not isinstance(rationale, str):
        raise ValueError("rationale must be a string")
    rationale = rationale.strip()
    if (
        not 20 <= len(rationale) <= 1000
        or any(not character.isprintable() for character in rationale)
    ):
        raise ValueError("rationale must be 20..1000 printable characters")
    if not isinstance(adjudicated_at, str) or not adjudicated_at.strip():
        raise ValueError("adjudicated_at must be an ISO-8601 timestamp with a timezone")
    try:
        parsed_adjudicated_at = datetime.fromisoformat(
            adjudicated_at.strip().replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(
            "adjudicated_at must be an ISO-8601 timestamp with a timezone"
        ) from exc
    if parsed_adjudicated_at.tzinfo is None or parsed_adjudicated_at.utcoffset() is None:
        raise ValueError("adjudicated_at must include a timezone")

    def provenance(path, reviewed):
        return {
            "file_sha256": _file_sha256(path),
            "annotator_id": reviewed["annotation_metadata"]["annotator_id"],
            "reviewer_id": reviewed["annotation_metadata"]["reviewer_id"],
            "reviewed_at": reviewed["annotation_metadata"]["reviewed_at"],
        }

    metadata = {
        "schema": _ADJUDICATION_SCHEMA,
        "decision": decision,
        "selected_annotation": "a" if decision == "accept_a" else "b",
        "adjudicator_id": adjudicator_id,
        "adjudicated_at": adjudicated_at.strip(),
        "rationale": rationale,
        "annotation_a": provenance(annotation_a_path, reviewed_a),
        "annotation_b": provenance(annotation_b_path, reviewed_b),
        "agreement_report": agreement_report,
        "agreement_report_sha256": hashlib.sha256(
            json.dumps(
                agreement_report, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest(),
    }
    selected = reviewed_a if decision == "accept_a" else reviewed_b
    _validate_adjudication_metadata(
        metadata,
        1,
        reviewed_a["fixture_id"],
        image_hash=reviewed_a["image_sha256"].lower(),
        image_width=reviewed_a["image_width"],
        image_height=reviewed_a["image_height"],
        market_context=reviewed_a["market_context"],
        annotation_metadata=selected["annotation_metadata"],
    )
    return {
        "adjudication_schema": _ADJUDICATION_SCHEMA,
        "fixture_id": reviewed_a["fixture_id"],
        "image_sha256": reviewed_a["image_sha256"].lower(),
        "image_width": reviewed_a["image_width"],
        "image_height": reviewed_a["image_height"],
        "market_context": reviewed_a["market_context"],
        "guideline_version": reviewed_a["annotation_metadata"]["guideline_version"],
        "adjudication_metadata": metadata,
    }


def write_adjudication_record(
    annotation_a_path,
    annotation_b_path,
    *,
    adjudicator_id,
    adjudicated_at,
    decision,
    rationale,
    output_path=None,
):
    record = build_adjudication_record(
        annotation_a_path,
        annotation_b_path,
        adjudicator_id=adjudicator_id,
        adjudicated_at=adjudicated_at,
        decision=decision,
        rationale=rationale,
    )
    annotation_a_path = Path(annotation_a_path).expanduser().resolve()
    if output_path is None:
        output_path = annotation_a_path.with_name(
            f"{record['fixture_id']}.adjudication.json"
        )
    output_path = Path(output_path).expanduser()
    if output_path.resolve(strict=False) in {annotation_a_path, Path(annotation_b_path).expanduser().resolve()}:
        raise ValueError("adjudication output must not overwrite an input annotation")
    saved = _write_new_json(output_path, record)
    return {
        "fixture_id": record["fixture_id"],
        "adjudication_file": str(saved),
        "decision": record["adjudication_metadata"]["decision"],
        "agreement_report_sha256": record["adjudication_metadata"]["agreement_report_sha256"],
        "benchmark_ready": False,
    }


def _validate_adjudication_sidecar(
    adjudication_path,
    annotation_a_path,
    annotation_b_path,
    reviewed_a,
    reviewed_b,
    agreement_report,
):
    sidecar = _read_json(adjudication_path)
    expected_fields = {
        "adjudication_schema", "fixture_id", "image_sha256",
        "image_width", "image_height", "market_context",
        "guideline_version", "adjudication_metadata",
    }
    if set(sidecar) != expected_fields:
        raise ValueError("adjudication sidecar fields do not match the supported schema")
    if sidecar.get("adjudication_schema") != _ADJUDICATION_SCHEMA:
        raise ValueError("unsupported adjudication sidecar schema")
    expected_top_level = {
        "fixture_id": reviewed_a["fixture_id"],
        "image_sha256": reviewed_a["image_sha256"].lower(),
        "image_width": reviewed_a["image_width"],
        "image_height": reviewed_a["image_height"],
        "market_context": reviewed_a["market_context"],
        "guideline_version": reviewed_a["annotation_metadata"]["guideline_version"],
    }
    for field, expected in expected_top_level.items():
        if sidecar.get(field) != expected:
            raise ValueError(f"adjudication sidecar {field} does not match reviewed annotations")

    metadata = sidecar.get("adjudication_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("adjudication_metadata must be an object")
    expected_provenance = (
        (annotation_a_path, reviewed_a, "annotation_a"),
        (annotation_b_path, reviewed_b, "annotation_b"),
    )
    for path, reviewed, label in expected_provenance:
        expected = {
            "file_sha256": _file_sha256(path),
            "annotator_id": reviewed["annotation_metadata"]["annotator_id"],
            "reviewer_id": reviewed["annotation_metadata"]["reviewer_id"],
            "reviewed_at": reviewed["annotation_metadata"]["reviewed_at"],
        }
        if metadata.get(label) != expected:
            raise ValueError(
                f"adjudication {label} provenance does not match the exact reviewed file"
            )
    if _canonical_json(metadata.get("agreement_report")) != _canonical_json(agreement_report):
        raise ValueError(
            "adjudication agreement report does not match independently recomputed annotations"
        )
    selected = reviewed_a if metadata.get("decision") == "accept_a" else reviewed_b
    _validate_adjudication_metadata(
        metadata,
        1,
        reviewed_a["fixture_id"],
        image_hash=reviewed_a["image_sha256"].lower(),
        image_width=reviewed_a["image_width"],
        image_height=reviewed_a["image_height"],
        market_context=reviewed_a["market_context"],
        annotation_metadata=selected["annotation_metadata"],
    )
    return metadata


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
    second_reviewed_draft_path=None,
    adjudication_path=None,
    max_image_bytes=DEFAULT_MAX_IMAGE_BYTES,
):
    """Create a benchmark fixture only after independent review and adjudication."""
    if (
        isinstance(max_image_bytes, bool)
        or not isinstance(max_image_bytes, int)
        or max_image_bytes <= 0
    ):
        raise ValueError("max_image_bytes must be a positive integer")
    if second_reviewed_draft_path is None or adjudication_path is None:
        raise ValueError(
            "benchmark assembly requires second_reviewed_draft_path and adjudication_path"
        )

    annotation_a_path = Path(reviewed_draft_path).expanduser().resolve()
    annotation_b_path = Path(second_reviewed_draft_path).expanduser().resolve()
    capture_path = Path(capture_path).expanduser().resolve()
    adjudication_path = Path(adjudication_path).expanduser().resolve()
    if annotation_a_path == annotation_b_path:
        raise ValueError("benchmark assembly requires two different reviewed annotation files")

    reviewed_a, reviewed_b, agreement_report = _load_reviewed_pair(
        annotation_a_path, annotation_b_path
    )
    adjudication = _validate_adjudication_sidecar(
        adjudication_path,
        annotation_a_path,
        annotation_b_path,
        reviewed_a,
        reviewed_b,
        agreement_report,
    )
    if adjudication["decision"] == "accept_a":
        selected_path, reviewed = annotation_a_path, reviewed_a
    else:
        selected_path, reviewed = annotation_b_path, reviewed_b

    prediction = _load_capture(capture_path, reviewed)
    output = (
        selected_path.with_name(f"{reviewed['fixture_id']}.fixture.jsonl")
        if output_path is None else Path(output_path).expanduser()
    )
    output_resolved = output.resolve(strict=False)
    if output_resolved.parent != selected_path.parent:
        raise ValueError(
            "fixture JSONL must be written beside the adjudicated annotation so image_file remains relative"
        )
    input_paths = {
        selected_path,
        annotation_a_path,
        annotation_b_path,
        capture_path,
        adjudication_path,
    }
    if output_resolved in input_paths:
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
        "adjudication_metadata": adjudication,
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

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="x", encoding="utf-8", newline="\n", suffix=".jsonl",
            prefix=".stockmind-fixture-check-", dir=selected_path.parent,
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
        "decision": adjudication["decision"],
        "adjudication_file": str(adjudication_path),
        "adjudication_report_sha256": adjudication["agreement_report_sha256"],
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

    adjudicate_parser = commands.add_parser(
        "adjudicate", help="compare two approved annotations and record a human selection"
    )
    adjudicate_parser.add_argument("annotation_a", type=Path)
    adjudicate_parser.add_argument("annotation_b", type=Path)
    adjudicate_parser.add_argument("--adjudicator-id", required=True)
    adjudicate_parser.add_argument("--adjudicated-at", required=True)
    adjudicate_parser.add_argument("--decision", choices=("accept_a", "accept_b"), required=True)
    adjudicate_parser.add_argument("--rationale", required=True)
    adjudicate_parser.add_argument("--output", type=Path)

    assemble_parser = commands.add_parser(
        "assemble", help="create a benchmark fixture after independent review and adjudication"
    )
    assemble_parser.add_argument("reviewed_draft", type=Path, help="annotation A")
    assemble_parser.add_argument("prediction_capture", type=Path)
    assemble_parser.add_argument("--annotation-b", type=Path, required=True)
    assemble_parser.add_argument("--adjudication", type=Path, required=True)
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
        elif args.command == "adjudicate":
            output = write_adjudication_record(
                args.annotation_a,
                args.annotation_b,
                adjudicator_id=args.adjudicator_id,
                adjudicated_at=args.adjudicated_at,
                decision=args.decision,
                rationale=args.rationale,
                output_path=args.output,
            )
        else:
            output = assemble_benchmark_fixture(
                args.reviewed_draft,
                args.prediction_capture,
                args.output,
                second_reviewed_draft_path=args.annotation_b,
                adjudication_path=args.adjudication,
                max_image_bytes=args.max_image_bytes,
            )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
