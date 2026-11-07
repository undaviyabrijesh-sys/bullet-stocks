"""Offline evaluation of AI-drawn chart zones against manual annotations.

Coordinates are normalized to the full screenshot (0..1). This module does not
infer or repair geometry; it scores only boxes that would actually be rendered.
Optional price-axis scoring requires manually verified axis anchors.
"""
from functools import lru_cache
import math

from .overlay_engine import normalized_plot_bounds_to_pixel_box, normalized_zone_to_pixel_box
from .price_axis import calibrate_price_axis, zone_price_bounds
from .plot_evaluation import build_plot_boundary_accuracy

MAX_EVALUATED_ZONES = 8
ZONE_TYPES = {"supply", "demand"}


def _valid_box(zone, *, prediction, image_width, image_height):
    """Return pixel-rounded geometry normalized back to the screenshot frame."""
    pixel_box = normalized_zone_to_pixel_box(
        zone,
        image_width,
        image_height,
        require_draw_overlay=prediction,
        min_width=3 if prediction else 1,
        min_height=2 if prediction else 1,
    )
    if pixel_box is None:
        return None
    left, top, right, bottom = pixel_box
    normalized_box = (
        left / image_width,
        top / image_height,
        right / image_width,
        bottom / image_height,
    )
    # Price-axis scoring must use the same quantized boundaries as the image
    # renderer, not the model's higher-precision coordinates.
    pixel_aligned_zone = dict(zone)
    pixel_aligned_zone.update({
        "x_start": normalized_box[0],
        "y_top": normalized_box[1],
        "x_end": normalized_box[2],
        "y_bottom": normalized_box[3],
    })
    return normalized_box, pixel_aligned_zone


def _iou(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _match(predictions, references, threshold):
    """Find maximum-cardinality, then maximum-IoU one-to-one matches."""
    scores = tuple(tuple(_iou(p["box"], r["box"]) for r in references) for p in predictions)

    @lru_cache(maxsize=None)
    def solve(pred_index, used_reference_mask):
        if pred_index >= len(predictions):
            return (0, 0.0, ())

        best = solve(pred_index + 1, used_reference_mask)
        for ref_index, score in enumerate(scores[pred_index]):
            if used_reference_mask & (1 << ref_index) or score < threshold or score <= 0:
                continue
            sub_count, sub_iou, sub_pairs = solve(pred_index + 1, used_reference_mask | (1 << ref_index))
            candidate = (
                sub_count + 1,
                sub_iou + score,
                ((pred_index, ref_index, score),) + sub_pairs,
            )
            if candidate[0] > best[0] or (
                candidate[0] == best[0] and candidate[1] > best[1] + 1e-12
            ):
                best = candidate
        return best

    return solve(0, 0)[2]


def _metrics(predictions, references, matches):
    true_positive = len(matches)
    false_positive = len(predictions) - true_positive
    false_negative = len(references) - true_positive
    precision = true_positive / len(predictions) if predictions else None
    recall = true_positive / len(references) if references else None
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = (2 * true_positive / f1_denominator) if f1_denominator else None
    matched_ious = [match["iou"] for match in matches]
    matched_center_errors = [match["center_error_normalized"] for match in matches]
    return {
        "predicted": len(predictions),
        "reference": len(references),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": sum(matched_ious) / len(matched_ious) if matched_ious else None,
        "mean_center_error_normalized": (
            sum(matched_center_errors) / len(matched_center_errors)
            if matched_center_errors else None
        ),
    }


def _price_comparison(prediction, reference, calibration):
    try:
        predicted_bounds = zone_price_bounds(prediction["zone"], calibration)
        reference_bounds = zone_price_bounds(reference["zone"], calibration)
    except ValueError as exc:
        return {"status": "unscored", "reason": str(exc)}

    high_error = abs(predicted_bounds["price_high"] - reference_bounds["price_high"])
    low_error = abs(predicted_bounds["price_low"] - reference_bounds["price_low"])
    high_pct = 100 * high_error / reference_bounds["price_high"]
    low_pct = 100 * low_error / reference_bounds["price_low"]
    return {
        "status": "scored",
        "predicted_price_bounds": predicted_bounds,
        "reference_price_bounds": reference_bounds,
        "price_high_abs_error": high_error,
        "price_low_abs_error": low_error,
        "price_high_error_pct": high_pct,
        "price_low_error_pct": low_pct,
    }


def _price_accuracy(matches, calibration):
    if calibration is None:
        return None
    comparisons = [match["price_comparison"] for match in matches]
    scored = [item for item in comparisons if item and item["status"] == "scored"]
    boundary_errors = [
        item[key]
        for item in scored
        for key in ("price_high_abs_error", "price_low_abs_error")
    ]
    boundary_percent_errors = [
        item[key]
        for item in scored
        for key in ("price_high_error_pct", "price_low_error_pct")
    ]
    return {
        "scale": calibration.scale,
        "anchor_count": calibration.anchor_count,
        "max_anchor_relative_error_pct": 100 * calibration.max_anchor_relative_error,
        "scored_match_count": len(scored),
        "unscored_match_count": len(comparisons) - len(scored),
        "mean_boundary_abs_error": (
            sum(boundary_errors) / len(boundary_errors) if boundary_errors else None
        ),
        "max_boundary_abs_error": max(boundary_errors) if boundary_errors else None,
        "mean_boundary_error_pct": (
            sum(boundary_percent_errors) / len(boundary_percent_errors)
            if boundary_percent_errors else None
        ),
    }


def evaluate_zone_predictions(
    predicted_zones,
    reference_zones,
    *,
    iou_threshold=0.5,
    price_axis_anchors=None,
    price_axis_plot_bounds=None,
    price_axis_scale="linear",
    max_axis_relative_error=0.02,
    predicted_plot_bounds=None,
    reference_plot_bounds=None,
    image_width,
    image_height,
):
    """Compare AI rectangles with manual annotations, optionally in price units.

    Predictions without draw_overlay=True, without a valid predicted_plot_bounds
    frame, or with invalid geometry are counted as ignored (not as drawn false
    positives). The model plot frame is mandatory for every drawable prediction,
    even when no manual reference plot was supplied. Invalid manual reference
    annotations, including zones outside a supplied manual plot boundary, raise
    ValueError so a broken fixture cannot silently skew scores.
    At most eight drawable predictions are considered. Invalid or out-of-plot
    candidates are skipped without consuming that cap, matching the renderer.
    image_width and image_height must be the original screenshot dimensions;
    predictions too small to pass the renderer's pixel threshold are ignored
    and their reference zones remain false negatives.

    When price_axis_anchors are supplied, a manually reviewed
    price_axis_plot_bounds rectangle is also required. Every anchor y must fall
    within that main price-plot's vertical span (the labels may sit in the
    adjacent price scale). Anchors must be 3..20 manually verified
    {"y": normalized_full_image_y, "price": positive_price} points from the
    same screenshot. Price errors are reported only for matched zones wholly
    inside the calibrated y range. No OCR or price-axis extrapolation is done.
    """
    for dimension, value in (("image_width", image_width), ("image_height", image_height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{dimension} must be a positive integer")
    if isinstance(iou_threshold, bool) or not isinstance(iou_threshold, (int, float)):
        raise ValueError("iou_threshold must be a number greater than 0 and at most 1")
    iou_threshold = float(iou_threshold)
    if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be a number greater than 0 and at most 1")
    if not isinstance(predicted_zones, (list, tuple)):
        raise ValueError("predicted_zones must be a list")
    if not isinstance(reference_zones, (list, tuple)):
        raise ValueError("reference_zones must be a list")
    if len(reference_zones) > MAX_EVALUATED_ZONES:
        raise ValueError(f"reference_zones cannot exceed {MAX_EVALUATED_ZONES} entries")

    calibration = None
    if price_axis_anchors is None:
        if price_axis_plot_bounds is not None:
            raise ValueError("price_axis_plot_bounds requires price_axis_anchors")
    else:
        if price_axis_plot_bounds is None:
            raise ValueError(
                "price_axis_plot_bounds is required with price_axis_anchors"
            )
        plot_box = normalized_plot_bounds_to_pixel_box(
            price_axis_plot_bounds, image_width, image_height
        )
        if plot_box is None:
            raise ValueError("price_axis_plot_bounds must be a valid normalized plot rectangle")
        _, plot_top, _, plot_bottom = plot_box
        plot_top_normalized = plot_top / image_height
        plot_bottom_normalized = plot_bottom / image_height
        calibration = calibrate_price_axis(
            price_axis_anchors,
            scale=price_axis_scale,
            max_relative_error=max_axis_relative_error,
        )
        for index, anchor in enumerate(price_axis_anchors):
            anchor_y = float(anchor["y"])  # validated by calibrate_price_axis
            if not plot_top_normalized <= anchor_y < plot_bottom_normalized:
                raise ValueError(
                    f"price-axis anchor at index {index} falls outside the manually marked main price plot"
                )

    valid_predictions = []
    ignored_predictions = 0
    for index, zone in enumerate(predicted_zones):
        validated = _valid_box(zone, prediction=True, image_width=image_width, image_height=image_height)
        if validated is None:
            ignored_predictions += 1
            continue
        box, pixel_aligned_zone = validated
        valid_predictions.append({
            "index": index,
            "zone_type": zone["zone_type"],
            "box": box,
            "box_px": [
                round(box[0] * image_width),
                round(box[1] * image_height),
                round(box[2] * image_width),
                round(box[3] * image_height),
            ],
            "zone": pixel_aligned_zone,
        })

    valid_references = []
    for index, zone in enumerate(reference_zones):
        validated = _valid_box(zone, prediction=False, image_width=image_width, image_height=image_height)
        if validated is None:
            raise ValueError(f"Invalid manual reference zone at index {index}")
        box, pixel_aligned_zone = validated
        valid_references.append({
            "index": index,
            "zone_type": zone["zone_type"],
            "box": box,
            "box_px": [
                round(box[0] * image_width),
                round(box[1] * image_height),
                round(box[2] * image_width),
                round(box[3] * image_height),
            ],
            "zone": pixel_aligned_zone,
        })

    # Resolve the independently reviewed main plot before scoring overlays.
    if reference_plot_bounds is None and price_axis_plot_bounds is not None:
        reference_plot_bounds = price_axis_plot_bounds
    if predicted_plot_bounds is not None and reference_plot_bounds is None:
        raise ValueError("reference_plot_bounds is required with predicted_plot_bounds")
    reference_plot_box = None
    if reference_plot_bounds is not None:
        reference_plot_box = normalized_plot_bounds_to_pixel_box(
            reference_plot_bounds, image_width, image_height
        )
        if reference_plot_box is None:
            raise ValueError(
                "reference_plot_bounds must be a valid normalized plot rectangle"
            )

        # Manual reference zones are ground truth for the main price chart.
        # Reject annotations outside that independently reviewed plot instead
        # of allowing off-panel rectangles to distort precision/recall.
        for zone in valid_references:
            left, top, right, bottom = zone["box_px"]
            if (
                left < reference_plot_box[0]
                or top < reference_plot_box[1]
                or right > reference_plot_box[2]
                or bottom > reference_plot_box[3]
            ):
                raise ValueError(
                    "manual reference zone at index "
                    f"{zone['index']} falls outside the manually marked main price plot"
                )

    if reference_plot_bounds is not None and price_axis_plot_bounds is not None:
        axis_box = normalized_plot_bounds_to_pixel_box(
            price_axis_plot_bounds, image_width, image_height
        )
        if reference_plot_box != axis_box:
            raise ValueError(
                "reference_plot_bounds must match price_axis_plot_bounds when both are supplied"
            )

    # Production rendering always requires a valid model plot frame. Apply
    # that rule even when a fixture has no manual plot annotation; otherwise a
    # zone with no renderable frame could be counted as a true positive.
    predicted_plot_box = normalized_plot_bounds_to_pixel_box(
        predicted_plot_bounds, image_width, image_height
    )
    drawable_predictions = []
    for zone in valid_predictions:
        left, top, right, bottom = zone["box_px"]
        if (
            predicted_plot_box is None
            or left < predicted_plot_box[0]
            or top < predicted_plot_box[1]
            or right > predicted_plot_box[2]
            or bottom > predicted_plot_box[3]
        ):
            ignored_predictions += 1
            continue
        if len(drawable_predictions) >= MAX_EVALUATED_ZONES:
            ignored_predictions += 1
            continue
        drawable_predictions.append(zone)
    valid_predictions = drawable_predictions

    plot_boundary_accuracy = None
    if reference_plot_bounds is not None:
        plot_boundary_accuracy = build_plot_boundary_accuracy(
            reference_plot_bounds,
            predicted_plot_bounds,
            [
                {"prediction_index": zone["index"], "box_px": zone["box_px"]}
                for zone in valid_predictions
            ],
            image_width,
            image_height,
        )

    by_type = {}
    all_matches = []
    for zone_type in ("supply", "demand"):
        predictions = [zone for zone in valid_predictions if zone["zone_type"] == zone_type]
        references = [zone for zone in valid_references if zone["zone_type"] == zone_type]
        local_matches = _match(predictions, references, iou_threshold)
        matches = []
        for pred_index, ref_index, score in local_matches:
            pred = predictions[pred_index]
            ref = references[ref_index]
            pred_center = ((pred["box"][0] + pred["box"][2]) / 2, (pred["box"][1] + pred["box"][3]) / 2)
            ref_center = ((ref["box"][0] + ref["box"][2]) / 2, (ref["box"][1] + ref["box"][3]) / 2)
            center_error = math.hypot(pred_center[0] - ref_center[0], pred_center[1] - ref_center[1])
            match = {
                "prediction_index": pred["index"],
                "reference_index": ref["index"],
                "zone_type": zone_type,
                "iou": score,
                "center_error_normalized": center_error,
                "price_comparison": (
                    _price_comparison(pred, ref, calibration) if calibration else None
                ),
            }
            matches.append(match)
            all_matches.append(match)
        by_type[zone_type] = _metrics(predictions, references, matches)

    overall = _metrics(valid_predictions, valid_references, all_matches)

    return {
        "iou_threshold": iou_threshold,
        "plot_boundary_accuracy": plot_boundary_accuracy,
        "overall": overall,
        "by_zone_type": by_type,
        "ignored_prediction_count": ignored_predictions,
        "matches": sorted(all_matches, key=lambda item: item["prediction_index"]),
        "price_accuracy": _price_accuracy(all_matches, calibration),
    }
