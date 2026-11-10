"""Offline pairwise agreement checks for independently marked chart screenshots.

The utility does not call a model, market-data provider, Telegram or network
service. It verifies that two approved annotation drafts refer to the same
canonical screenshot and annotation guideline before comparing their geometry.
"""

import argparse
import json
import math
from pathlib import Path

from .annotation_editor import load_draft_and_image
from .annotation_fixture import _validate_reviewed_draft
from .overlay_engine import normalized_plot_bounds_to_pixel_box
from .price_axis import calibrate_price_axis
from .zone_evaluation import evaluate_manual_annotation_agreement


def _plot_agreement(first_bounds, second_bounds, image_width, image_height):
    first = normalized_plot_bounds_to_pixel_box(first_bounds, image_width, image_height)
    second = normalized_plot_bounds_to_pixel_box(second_bounds, image_width, image_height)
    if first is None or second is None:
        raise ValueError("both annotations must contain valid reference_plot_bounds")

    intersection_left = max(first[0], second[0])
    intersection_top = max(first[1], second[1])
    intersection_right = min(first[2], second[2])
    intersection_bottom = min(first[3], second[3])
    intersection = (
        max(0, intersection_right - intersection_left)
        * max(0, intersection_bottom - intersection_top)
    )
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    edge_displacement = {
        "left": abs(first[0] - second[0]),
        "right": abs(first[2] - second[2]),
        "top": abs(first[1] - second[1]),
        "bottom": abs(first[3] - second[3]),
    }
    return {
        "iou": intersection / union if union else None,
        "edge_displacement_px": edge_displacement,
        "edge_displacement_normalized": {
            "left": edge_displacement["left"] / image_width,
            "right": edge_displacement["right"] / image_width,
            "top": edge_displacement["top"] / image_height,
            "bottom": edge_displacement["bottom"] / image_height,
        },
        "annotator_a_bounds_px": list(first),
        "annotator_b_bounds_px": list(second),
    }



def _price_axis_agreement(first, second, *, sample_count=11):
    """Compare two manual price-axis fits only across their shared anchor range.

    Prices are sampled from each annotator's independently validated
    calibration. No extrapolation is performed. This is calibration agreement,
    not proof that either set of manually read labels is correct.
    """
    anchors_a = first["price_axis_anchors"]
    anchors_b = second["price_axis_anchors"]
    base = {
        "annotator_a_anchor_count": len(anchors_a),
        "annotator_b_anchor_count": len(anchors_b),
        "annotator_a_scale": first["price_axis_scale"],
        "annotator_b_scale": second["price_axis_scale"],
    }
    if not anchors_a and not anchors_b:
        return {"status": "not_provided", **base}
    if not anchors_a or not anchors_b:
        return {"status": "missing_on_one_side", **base}

    calibration_a = calibrate_price_axis(
        anchors_a, scale=first["price_axis_scale"]
    )
    calibration_b = calibrate_price_axis(
        anchors_b, scale=second["price_axis_scale"]
    )
    overlap_start = max(calibration_a.min_y, calibration_b.min_y)
    overlap_end = min(calibration_a.max_y, calibration_b.max_y)
    overlap_span = overlap_end - overlap_start
    result = {
        **base,
        "scale_match": calibration_a.scale == calibration_b.scale,
        "annotator_a_max_anchor_relative_error": calibration_a.max_anchor_relative_error,
        "annotator_b_max_anchor_relative_error": calibration_b.max_anchor_relative_error,
        "overlap_y": [overlap_start, overlap_end] if overlap_span > 1e-9 else None,
        "overlap_span": max(0.0, overlap_span),
    }
    if overlap_span <= 1e-9:
        return {"status": "no_overlap", **result}

    relative_errors = []
    absolute_errors = []
    for index in range(sample_count):
        y = overlap_start + overlap_span * index / (sample_count - 1)
        price_a = calibration_a.price_at_y(y)
        price_b = calibration_b.price_at_y(y)
        high = max(price_a, price_b)
        low = min(price_a, price_b)
        midpoint = high / 2.0 + low / 2.0
        absolute_error = high - low
        relative_error = absolute_error / midpoint
        if not math.isfinite(relative_error) or not math.isfinite(absolute_error):
            raise ValueError("price-axis agreement exceeds the supported numeric range")
        absolute_errors.append(absolute_error)
        relative_errors.append(relative_error)

    return {
        "status": "compared",
        **result,
        "sample_count": sample_count,
        "mean_absolute_price_disagreement": math.fsum(value / sample_count for value in absolute_errors),
        "max_absolute_price_disagreement": max(absolute_errors),
        "mean_relative_price_disagreement": math.fsum(relative_errors) / sample_count,
        "max_relative_price_disagreement": max(relative_errors),
        "relative_disagreement_by_y": [
            {
                "y": overlap_start + overlap_span * index / (sample_count - 1),
                "relative_disagreement": relative_errors[index],
            }
            for index in range(sample_count)
        ],
    }


def _load_reviewed(path):
    draft, image = load_draft_and_image(Path(path).expanduser())
    try:
        return _validate_reviewed_draft(draft)
    finally:
        image.close()


def compare_reviewed_annotations(annotation_a_path, annotation_b_path, *, iou_threshold=0.5):
    """Compare two separately approved annotations of the exact same screenshot."""
    first = _load_reviewed(annotation_a_path)
    second = _load_reviewed(annotation_b_path)

    for field in ("fixture_id", "image_width", "image_height", "market_context"):
        if first[field] != second[field]:
            raise ValueError(f"annotations must have identical {field}")
    if first["image_sha256"].lower() != second["image_sha256"].lower():
        raise ValueError("annotations must reference the exact same screenshot SHA-256")

    metadata_a = first["annotation_metadata"]
    metadata_b = second["annotation_metadata"]
    if metadata_a["guideline_version"] != metadata_b["guideline_version"]:
        raise ValueError("annotations must use the same guideline_version")
    def identity_token(value):
        return value.split("-", 1)[1].casefold()

    if identity_token(metadata_a["annotator_id"]) == identity_token(metadata_b["annotator_id"]):
        raise ValueError("annotations must have different annotator_id values")
    if (
        identity_token(metadata_a["reviewer_id"]) == identity_token(metadata_b["annotator_id"])
        or identity_token(metadata_b["reviewer_id"]) == identity_token(metadata_a["annotator_id"])
    ):
        raise ValueError("reviewer identity tokens must not overlap the other annotation's annotator ID")

    dimensions = {
        "image_width": first["image_width"],
        "image_height": first["image_height"],
    }
    zone_agreement = evaluate_manual_annotation_agreement(
        first["reference_zones"],
        second["reference_zones"],
        iou_threshold=iou_threshold,
        **dimensions,
    )
    return {
        "report_schema": "stockmind.annotation-agreement.v1",
        "fixture_id": first["fixture_id"],
        "image_sha256": first["image_sha256"].lower(),
        **dimensions,
        "market_context": first["market_context"],
        "guideline_version": metadata_a["guideline_version"],
        "annotator_ids": [
            metadata_a["annotator_id"],
            metadata_b["annotator_id"],
        ],
        "reviewer_ids": [
            metadata_a["reviewer_id"],
            metadata_b["reviewer_id"],
        ],
        "plot_agreement": _plot_agreement(
            first["reference_plot_bounds"],
            second["reference_plot_bounds"],
            **dimensions,
        ),
        "price_axis_agreement": _price_axis_agreement(first, second),
        "zone_agreement": zone_agreement,
        "interpretation": (
            "Pairwise manual agreement only; this report is not AI detection accuracy "
            "and does not select or adjudicate benchmark ground truth."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare two independently approved annotations of one exact chart screenshot."
    )
    parser.add_argument("annotation_a", type=Path)
    parser.add_argument("annotation_b", type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    try:
        report = compare_reviewed_annotations(
            args.annotation_a,
            args.annotation_b,
            iou_threshold=args.iou_threshold,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
