# StockMind chart-zone benchmark

This benchmark compares AI overlay rectangles with zones manually marked on the exact same chart screenshot. It is an offline evaluation aid; it does not alter bot responses or make a live model call.

## Annotation rules

- Keep the original screenshot unchanged and record its SHA-256, width and height.
- Mark each visible reference zone as supply or demand.
- Express each rectangle relative to the full screenshot: x_start, x_end, y_top and y_bottom are normalized to 0..1.
- Use the same full-image coordinate frame for the AI prediction and manual annotation. Do not crop one side or score against a different screenshot.
- Keep no more than eight reference zones, matching the current overlay cap.
- Do not infer unreadable price values. Geometry-only fixtures need no price-axis data.
- For price-level scoring, manually record at least three legible price-axis anchors from the same screenshot. Each anchor is `{"y": 0..1, "price": positive_number}`, where y is normalized to the full screenshot. Record anchors at distinct vertical positions and in top-to-bottom order with descending prices.
- Choose `linear` for a standard arithmetic price axis or `log` for a logarithmic axis. Do not use the logarithmic option merely because price values are large.

A fixture record should contain a stable fixture_id, image_sha256, image_width, image_height, reference_zones and predicted_zones. Each reference zone needs zone_type and the four normalized coordinates. Predicted zones should be the structured AI zones, including draw_overlay. Optional price_axis_anchors and price_axis_scale fields enable price-unit scoring.

## Geometry scoring

Call `evaluate_zone_predictions(predicted_zones, reference_zones, iou_threshold=0.5)` from `app.zone_evaluation`.

The evaluator:
- matches only zones with the same type (supply-to-supply, demand-to-demand);
- performs one-to-one matching, maximizing the number of qualifying matches before maximizing total IoU;
- reports precision, recall, F1, false positives, false negatives, mean matched intersection-over-union (IoU), and normalized center displacement;
- reports invalid or disabled AI rectangles separately as ignored predictions;
- treats a disabled/invalid AI rectangle as no drawn prediction, so a missed reference zone remains a false negative;
- rejects malformed manual annotations rather than silently corrupting the benchmark;
- scores only the first eight AI zones because that is the renderer's current limit.

Use the same IoU threshold across a benchmark release. Report aggregate and per-fixture results, and retain the fixture/image hash so future model changes can be compared on identical inputs.

## Optional price-axis scoring

Pass the manually reviewed anchors as `price_axis_anchors` and set `price_axis_scale="linear"` or `"log"`:

```python
result = evaluate_zone_predictions(
    predicted_zones,
    reference_zones,
    iou_threshold=0.5,
    price_axis_anchors=[
        {"y": 0.18, "price": 125.0},
        {"y": 0.43, "price": 100.0},
        {"y": 0.68, "price": 75.0},
    ],
    price_axis_scale="linear",
)
```

Price-axis calibration:
- requires 3..20 manually verified anchors; two points are deliberately insufficient because they cannot expose a distorted/non-uniform scale;
- fits either a linear price axis or a linear fit in log-price space;
- rejects reversed/non-monotonic anchors and fits whose maximum anchor residual exceeds 2% by default (the caller may set a stricter tolerance, up to 10%);
- never extrapolates beyond the first/last anchor y-coordinate;
- reports matched-zone high/low boundary error in price units and as a percentage, separately from IoU;
- marks a matched zone as unscored if either boundary falls outside the calibrated y range rather than guessing a price.

This is a manual calibration and scoring tool, not automatic axis OCR. Anchor y-coordinates must use the same full-image coordinate frame as the AI and reference rectangles. Keep the screenshot, anchors and annotations together in the fixture record. Do not mix anchors from another panel, crop, symbol or timeframe.

## Important limitations

There are currently no manually annotated real chart screenshots in this repository. Unit tests use synthetic rectangles and synthetic axis anchors only to verify the scoring logic; they are not evidence of model accuracy. Do not publish a zone-detection accuracy or claim improvement until a real, manually reviewed fixture set has been scored.

IoU and normalized center displacement measure screenshot geometry only. Price-boundary error is only as trustworthy as the manually read axis anchors and chosen scale. Neither metric verifies that a zone is structurally meaningful, fresh/unmitigated, or useful for a trading decision.
