"""Offline evaluation of AI-drawn chart zones against manual annotations.

Coordinates are normalized to the full screenshot (0..1). This module does not
infer or repair geometry; it scores only boxes that would actually be rendered.
"""
from functools import lru_cache
import math

MAX_EVALUATED_ZONES = 8
ZONE_TYPES = {"supply", "demand"}


def _valid_box(zone, *, prediction):
    if not isinstance(zone, dict):
        return None
    zone_type = zone.get("zone_type")
    if not isinstance(zone_type, str) or zone_type not in ZONE_TYPES:
        return None
    if prediction and zone.get("draw_overlay") is not True:
        return None

    raw = [zone.get("x_start"), zone.get("x_end"), zone.get("y_top"), zone.get("y_bottom")]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw):
        return None
    try:
        x0, x1, y0, y1 = (float(value) for value in raw)
    except (OverflowError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (x0, x1, y0, y1)):
        return None
    if x1 <= x0 or y1 <= y0 or y1 - y0 > 0.45:
        return None
    return (x0, y0, x1, y1)


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


def evaluate_zone_predictions(predicted_zones, reference_zones, *, iou_threshold=0.5):
    """Compare rendered AI rectangles with manually annotated rectangles.

    Predictions without draw_overlay=True or with invalid geometry are counted
    as ignored (not as drawn false positives). Invalid manual reference
    annotations raise ValueError so a broken fixture cannot silently skew scores.
    Only the first eight predictions are considered because the overlay renderer
    draws at most eight; extra predictions are reported as ignored.
    """
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

    valid_predictions = []
    ignored_predictions = max(0, len(predicted_zones) - MAX_EVALUATED_ZONES)
    for index, zone in enumerate(predicted_zones[:MAX_EVALUATED_ZONES]):
        box = _valid_box(zone, prediction=True)
        if box is None:
            ignored_predictions += 1
            continue
        valid_predictions.append({"index": index, "zone_type": zone["zone_type"], "box": box})

    valid_references = []
    for index, zone in enumerate(reference_zones):
        box = _valid_box(zone, prediction=False)
        if box is None:
            raise ValueError(f"Invalid manual reference zone at index {index}")
        valid_references.append({"index": index, "zone_type": zone["zone_type"], "box": box})

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
            }
            matches.append(match)
            all_matches.append(match)
        by_type[zone_type] = _metrics(predictions, references, matches)

    overall = _metrics(valid_predictions, valid_references, all_matches)
    return {
        "iou_threshold": iou_threshold,
        "overall": overall,
        "by_zone_type": by_type,
        "ignored_prediction_count": ignored_predictions,
        "matches": sorted(all_matches, key=lambda item: item["prediction_index"]),
    }
