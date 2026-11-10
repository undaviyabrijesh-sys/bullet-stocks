"""Plot-boundary and zone-containment metrics for chart screenshot evaluation.

These metrics compare the model's estimated plot rectangle with a separately
reviewed manual rectangle. Right-margin and bottom-panel overlap are geometric
proxies only; they do not identify the semantic contents of excluded regions.
"""
from .overlay_engine import normalized_plot_bounds_to_pixel_box


_EDGE_NAMES = ("left", "right", "top", "bottom")


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection_area(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return max(0, right - left) * max(0, bottom - top)


def _right_margin_area(box, reference):
    left = max(box[0], reference[2])
    right = box[2]
    top = max(box[1], reference[1])
    bottom = min(box[3], reference[3])
    return max(0, right - left) * max(0, bottom - top)


def _bottom_panel_area(box, reference):
    left = max(box[0], reference[0])
    right = min(box[2], reference[2])
    top = max(box[1], reference[3])
    bottom = box[3]
    return max(0, right - left) * max(0, bottom - top)


def _left_margin_area(box, reference):
    """Area extending left of the reference plot, within its vertical span."""
    left = box[0]
    right = min(box[2], reference[0])
    top = max(box[1], reference[1])
    bottom = min(box[3], reference[3])
    return max(0, right - left) * max(0, bottom - top)


def _top_controls_area(box, reference):
    """Area extending above the reference plot, within its horizontal span."""
    left = max(box[0], reference[0])
    right = min(box[2], reference[2])
    top = box[1]
    bottom = min(box[3], reference[1])
    return max(0, right - left) * max(0, bottom - top)


def _percent(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else None


def _validate_pixel_box(box, width, height, label):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"{label} must contain four pixel coordinates")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in box):
        raise ValueError(f"{label} coordinates must be integers")
    left, top, right, bottom = box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"{label} must be a non-empty rectangle inside the screenshot")
    return [left, top, right, bottom]


def build_plot_boundary_accuracy(
    reference_plot_bounds,
    predicted_plot_bounds,
    zone_boxes_px,
    image_width,
    image_height,
):
    """Measure estimated plot geometry and predicted-zone containment in pixels."""
    for name, value in (("image_width", image_width), ("image_height", image_height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    reference_box = normalized_plot_bounds_to_pixel_box(
        reference_plot_bounds, image_width, image_height
    )
    if reference_box is None:
        raise ValueError("reference_plot_bounds must be a valid normalized plot rectangle")

    if predicted_plot_bounds is None:
        predicted_box = None
        status = "missing_prediction"
    else:
        converted = normalized_plot_bounds_to_pixel_box(
            predicted_plot_bounds, image_width, image_height
        )
        predicted_box = list(converted) if converted is not None else None
        status = "scored" if predicted_box is not None else "invalid_prediction"

    if not isinstance(zone_boxes_px, (list, tuple)):
        raise ValueError("zone_boxes_px must be a list")
    normalized_zone_boxes = []
    seen_indices = set()
    for item in zone_boxes_px:
        if not isinstance(item, dict):
            raise ValueError("each zone_boxes_px entry must be an object")
        index = item.get("prediction_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("zone_boxes_px prediction_index must be a non-negative integer")
        if index in seen_indices:
            raise ValueError("zone_boxes_px prediction_index values must be unique")
        seen_indices.add(index)
        box = _validate_pixel_box(
            item.get("box_px"), image_width, image_height,
            f"zone_boxes_px[{index}].box_px",
        )
        normalized_zone_boxes.append({"prediction_index": index, "box_px": box})
    normalized_zone_boxes.sort(key=lambda item: item["prediction_index"])

    # This evidence represents boxes the production renderer actually draws:
    # every zone must fit within the model's valid plot frame.
    if normalized_zone_boxes:
        if predicted_box is None:
            raise ValueError("drawable zone boxes require valid predicted plot bounds")
        for item in normalized_zone_boxes:
            left, top, right, bottom = item["box_px"]
            if (
                left < predicted_box[0]
                or top < predicted_box[1]
                or right > predicted_box[2]
                or bottom > predicted_box[3]
            ):
                raise ValueError(
                    "zone_boxes_px contains a zone outside predicted plot bounds"
                )

    reference_area = _area(reference_box)
    plot_iou = None
    edge_px = None
    edge_normalized = None
    plot_outside_pct = None
    right_plot_pct = None
    bottom_plot_pct = None
    left_plot_pct = None
    top_plot_pct = None
    if predicted_box is not None:
        predicted_area = _area(predicted_box)
        intersection = _intersection_area(predicted_box, reference_box)
        union = predicted_area + reference_area - intersection
        plot_iou = intersection / union if union else None
        outside = predicted_area - intersection
        plot_outside_pct = _percent(outside, predicted_area)
        right_plot_pct = _percent(
            _right_margin_area(predicted_box, reference_box), predicted_area
        )
        bottom_plot_pct = _percent(
            _bottom_panel_area(predicted_box, reference_box), predicted_area
        )
        left_plot_pct = _percent(
            _left_margin_area(predicted_box, reference_box), predicted_area
        )
        top_plot_pct = _percent(
            _top_controls_area(predicted_box, reference_box), predicted_area
        )
        edge_px = {
            "left": abs(predicted_box[0] - reference_box[0]),
            "right": abs(predicted_box[2] - reference_box[2]),
            "top": abs(predicted_box[1] - reference_box[1]),
            "bottom": abs(predicted_box[3] - reference_box[3]),
        }
        edge_normalized = {
            "left": edge_px["left"] / image_width,
            "right": edge_px["right"] / image_width,
            "top": edge_px["top"] / image_height,
            "bottom": edge_px["bottom"] / image_height,
        }

    zone_areas = []
    zone_outside_count = 0
    outside_area = 0
    right_zone_area = 0
    bottom_zone_area = 0
    left_zone_area = 0
    top_zone_area = 0
    for item in normalized_zone_boxes:
        box = item["box_px"]
        area = _area(box)
        intersection = _intersection_area(box, reference_box)
        outside = area - intersection
        if outside:
            zone_outside_count += 1
        zone_areas.append(area)
        outside_area += outside
        right_zone_area += _right_margin_area(box, reference_box)
        bottom_zone_area += _bottom_panel_area(box, reference_box)
        left_zone_area += _left_margin_area(box, reference_box)
        top_zone_area += _top_controls_area(box, reference_box)

    total_zone_area = sum(zone_areas)
    zone_count = len(normalized_zone_boxes)
    return {
        "status": status,
        "image_width": image_width,
        "image_height": image_height,
        "reference_bounds_px": list(reference_box),
        "predicted_bounds_px": predicted_box,
        "plot_iou": plot_iou,
        "edge_displacement_px": edge_px,
        "edge_displacement_normalized": edge_normalized,
        "predicted_plot_outside_reference_pct": plot_outside_pct,
        "right_margin_overlap_pct": right_plot_pct,
        "bottom_panel_overlap_pct": bottom_plot_pct,
        "left_toolbar_overlap_pct": left_plot_pct,
        "top_controls_overlap_pct": top_plot_pct,
        "predicted_zone_count": zone_count,
        "zones_outside_reference_plot_count": zone_outside_count,
        "zones_outside_reference_plot_pct": _percent(zone_outside_count, zone_count),
        "predicted_zone_area_outside_reference_pct": _percent(outside_area, total_zone_area),
        "right_margin_zone_area_pct": _percent(right_zone_area, total_zone_area),
        "bottom_panel_zone_area_pct": _percent(bottom_zone_area, total_zone_area),
        "left_toolbar_zone_area_pct": _percent(left_zone_area, total_zone_area),
        "top_controls_zone_area_pct": _percent(top_zone_area, total_zone_area),
        "zone_boxes_px": normalized_zone_boxes,
    }


def _strict_json_equal(actual, expected):
    """Compare report evidence without Python's bool/int/float equality loophole."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (
            actual.keys() == expected.keys()
            and all(_strict_json_equal(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            len(actual) == len(expected)
            and all(_strict_json_equal(left, right) for left, right in zip(actual, expected))
        )
    return actual == expected


def validate_plot_boundary_accuracy(
    metrics,
    reference_plot_bounds,
    predicted_plot_bounds,
    image_width,
    image_height,
    label,
):
    """Recompute report metrics from the included pixel evidence before comparison."""
    if metrics is None:
        if reference_plot_bounds is not None or predicted_plot_bounds is not None:
            raise ValueError(f"{label} omits plot-boundary metrics for supplied bounds")
        return
    if not isinstance(metrics, dict) or "zone_boxes_px" not in metrics:
        raise ValueError(f"{label} must include zone_boxes_px evidence")
    expected = build_plot_boundary_accuracy(
        reference_plot_bounds,
        predicted_plot_bounds,
        metrics["zone_boxes_px"],
        image_width,
        image_height,
    )
    if not _strict_json_equal(metrics, expected):
        raise ValueError(f"{label} does not agree with its plot bounds and zone-box evidence")


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def aggregate_plot_boundary_accuracy(fixtures):
    """Aggregate plot-boundary and zone-containment results across fixtures."""
    rows = [
        fixture["plot_boundary_accuracy"]
        for fixture in fixtures
        if fixture.get("plot_boundary_accuracy") is not None
    ]
    if not rows:
        return None
    scored = [row for row in rows if row["status"] == "scored"]
    zone_rows = [row for row in rows if row["predicted_zone_count"] > 0]
    predicted_count = sum(row["predicted_zone_count"] for row in rows)
    outside_count = sum(row["zones_outside_reference_plot_count"] for row in rows)
    edge_means = {
        edge: _mean([
            row["edge_displacement_normalized"][edge]
            for row in scored
        ])
        for edge in _EDGE_NAMES
    }
    return {
        "fixture_count": len(rows),
        "scored_fixture_count": len(scored),
        "missing_prediction_count": sum(row["status"] == "missing_prediction" for row in rows),
        "invalid_prediction_count": sum(row["status"] == "invalid_prediction" for row in rows),
        "mean_plot_iou": _mean([row["plot_iou"] for row in scored]),
        "mean_edge_displacement_normalized": edge_means,
        "mean_predicted_plot_outside_reference_pct": _mean([
            row["predicted_plot_outside_reference_pct"] for row in scored
        ]),
        "mean_right_margin_overlap_pct": _mean([
            row["right_margin_overlap_pct"] for row in scored
        ]),
        "mean_bottom_panel_overlap_pct": _mean([
            row["bottom_panel_overlap_pct"] for row in scored
        ]),
        "mean_left_toolbar_overlap_pct": _mean([
            row["left_toolbar_overlap_pct"] for row in scored
        ]),
        "mean_top_controls_overlap_pct": _mean([
            row["top_controls_overlap_pct"] for row in scored
        ]),
        "predicted_zone_count": predicted_count,
        "zones_outside_reference_plot_count": outside_count,
        "zones_outside_reference_plot_pct": _percent(outside_count, predicted_count),
        "mean_predicted_zone_area_outside_reference_pct": _mean([
            row["predicted_zone_area_outside_reference_pct"] for row in zone_rows
        ]),
        "mean_right_margin_zone_area_pct": _mean([
            row["right_margin_zone_area_pct"] for row in zone_rows
        ]),
        "mean_bottom_panel_zone_area_pct": _mean([
            row["bottom_panel_zone_area_pct"] for row in zone_rows
        ]),
        "mean_left_toolbar_zone_area_pct": _mean([
            row["left_toolbar_zone_area_pct"] for row in zone_rows
        ]),
        "mean_top_controls_zone_area_pct": _mean([
            row["top_controls_zone_area_pct"] for row in zone_rows
        ]),
    }


def compare_plot_boundary_accuracy(baseline, candidate):
    """Return paired fixture or aggregate plot metrics and candidate-minus-baseline deltas."""
    if baseline is None and candidate is None:
        return None
    is_fixture = any(
        row is not None and "plot_iou" in row
        for row in (baseline, candidate)
    )
    if is_fixture:
        scalar_fields = (
            "plot_iou", "predicted_plot_outside_reference_pct",
            "right_margin_overlap_pct", "bottom_panel_overlap_pct",
            "left_toolbar_overlap_pct", "top_controls_overlap_pct",
            "predicted_zone_count", "zones_outside_reference_plot_count",
            "zones_outside_reference_plot_pct",
            "predicted_zone_area_outside_reference_pct",
            "right_margin_zone_area_pct", "bottom_panel_zone_area_pct",
            "left_toolbar_zone_area_pct", "top_controls_zone_area_pct",
        )
        edge_field = "edge_displacement_normalized"
    else:
        scalar_fields = (
            "fixture_count", "scored_fixture_count", "missing_prediction_count",
            "invalid_prediction_count", "mean_plot_iou",
            "mean_predicted_plot_outside_reference_pct", "mean_right_margin_overlap_pct",
            "mean_bottom_panel_overlap_pct", "mean_left_toolbar_overlap_pct",
            "mean_top_controls_overlap_pct", "predicted_zone_count",
            "zones_outside_reference_plot_count", "zones_outside_reference_plot_pct",
            "mean_predicted_zone_area_outside_reference_pct",
            "mean_right_margin_zone_area_pct", "mean_bottom_panel_zone_area_pct",
            "mean_left_toolbar_zone_area_pct", "mean_top_controls_zone_area_pct",
        )
        edge_field = "mean_edge_displacement_normalized"

    delta = {}
    for field in scalar_fields:
        first = baseline.get(field) if baseline else None
        second = candidate.get(field) if candidate else None
        delta[field] = (
            second - first
            if isinstance(first, (int, float)) and not isinstance(first, bool)
            and isinstance(second, (int, float)) and not isinstance(second, bool)
            else None
        )
    delta[edge_field] = {
        edge: (
            candidate[edge_field][edge] - baseline[edge_field][edge]
            if baseline and candidate
            and baseline[edge_field][edge] is not None
            and candidate[edge_field][edge] is not None
            else None
        )
        for edge in _EDGE_NAMES
    }
    return {"baseline": baseline, "candidate": candidate, "delta": delta}
