# StockMind chart-zone benchmark

This benchmark compares AI overlay rectangles with zones manually marked on the exact same chart screenshot. It is an offline evaluation aid; it does not alter bot responses or make a live model call.

## Annotation rules

- Keep the original screenshot unchanged and record its SHA-256, width and height.
- Mark each visible reference zone as supply or demand.
- Express each rectangle relative to the full screenshot: x_start, x_end, y_top and y_bottom are normalized to 0..1.
- Use the same full-image coordinate frame for the AI prediction and manual annotation. Do not crop one side or score against a different screenshot.
- Keep no more than eight reference zones, matching the current overlay cap.
- Do not infer unreadable price values. Geometry-only fixtures need no price-axis data.
- For price-level scoring, manually record at least three legible price-axis anchors from the same screenshot. Each anchor is an object with y normalized to the full screenshot and a positive numeric price. Record anchors at distinct vertical positions and in top-to-bottom order with descending prices.
- Choose linear for a standard arithmetic price axis or log for a logarithmic axis. Do not use the logarithmic option merely because price values are large.

A fixture record should contain a stable fixture_id, image_file, image_sha256, image_width, image_height, reference_zones and predicted_zones. The image_file must be a relative path contained within the manifest directory. Each reference zone needs zone_type and the four normalized coordinates. Predicted zones should be the structured AI zones, including draw_overlay. Optional price_axis_anchors and price_axis_scale fields enable price-unit scoring. Additional descriptive metadata fields may be included.

## Geometry scoring

Call evaluate_zone_predictions(predicted_zones, reference_zones, iou_threshold=0.5) from app.zone_evaluation.

The evaluator:
- matches only zones with the same type (supply-to-supply, demand-to-demand);
- performs one-to-one matching, maximizing the number of qualifying matches before maximizing total IoU;
- reports precision, recall, F1, false positives, false negatives, mean matched intersection-over-union (IoU), and normalized center displacement;
- reports invalid or disabled AI rectangles separately as ignored predictions;
- treats a disabled/invalid AI rectangle as no drawn prediction, so a missed reference zone remains a false negative;
- rejects malformed manual annotations rather than silently corrupting the benchmark;
- scores only the first eight AI zones because that is the renderer's current limit.

Use the same IoU threshold across a benchmark release. Report aggregate and per-fixture results, and retain the fixture/image hash so future model changes can be compared on identical inputs.

## Batch manifest runner

The batch runner verifies that each fixture still points to the exact annotated screenshot before scoring it. It reads at most the configured byte limit plus one byte, checks the SHA-256, actual image format, decoded image limit and recorded dimensions; rejects duplicate fixture IDs and paths that escape the manifest directory; then reports per-fixture results and micro-aggregated precision, recall, F1, IoU, center displacement and optional price-boundary errors. The default byte limit is 5 MiB, matching the Telegram bot's default upload cap. Use `--max-image-bytes` when the bot's `MAX_CHART_IMAGE_BYTES` setting has been overridden.

Create a UTF-8 JSONL file with one fixture object per line. The following is the schema shape only; values are placeholders, not a real chart annotation:

~~~json
{
  "fixture_id": "replace-with-reviewed-id",
  "image_file": "images/reviewed-chart.png",
  "image_sha256": "replace-with-64-character-image-sha256",
  "image_width": 1600,
  "image_height": 900,
  "reference_zones": [
    {
      "zone_type": "demand",
      "x_start": 0.2,
      "x_end": 0.8,
      "y_top": 0.55,
      "y_bottom": 0.68
    }
  ],
  "predicted_zones": [
    {
      "zone_type": "demand",
      "label": "AI DEMAND",
      "x_start": 0.2,
      "x_end": 0.8,
      "y_top": 0.55,
      "y_bottom": 0.68,
      "draw_overlay": true
    }
  ]
}
~~~

Run from the repository root:

~~~bash
python -m stockmind_telegram_mvp.app.zone_benchmark stockmind_telegram_mvp/benchmark/fixtures.jsonl
# If MAX_CHART_IMAGE_BYTES is customized in the bot environment, pass the same limit:
python -m stockmind_telegram_mvp.app.zone_benchmark stockmind_telegram_mvp/benchmark/fixtures.jsonl --max-image-bytes 10485760
~~~

The JSON report is written to standard output and can be redirected to a versioned report file. A missing/empty manifest, malformed record, hash mismatch, changed dimensions, unsupported image or invalid manual annotation fails the run instead of emitting misleading aggregate metrics. The runner performs no AI request; predictions must be captured separately and paired with the exact source screenshot.

Aggregate precision/recall/F1 are micro-averaged from total TP/FP/FN counts. Mean IoU and center displacement are averaged across matched zones, not across screenshots. Price errors are averaged across scored zone boundaries; zones outside the calibrated y-range are counted as unscored rather than extrapolated.

## Optional price-axis scoring

Pass manually reviewed anchors as price_axis_anchors and set price_axis_scale to linear or log:

~~~python
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
~~~

Price-axis calibration:
- requires 3..20 manually verified anchors; two points are deliberately insufficient because they cannot expose a distorted/non-uniform scale;
- fits either a linear price axis or a linear fit in log-price space;
- rejects reversed/non-monotonic anchors and fits whose maximum anchor residual exceeds 2% by default (the caller may set a stricter tolerance, up to 10%);
- never extrapolates beyond the first/last anchor y-coordinate;
- reports matched-zone high/low boundary error in price units and as a percentage, separately from IoU;
- marks a matched zone as unscored if either boundary falls outside the calibrated y range rather than guessing a price.

This is a manual calibration and scoring tool, not automatic axis OCR. Anchor y-coordinates must use the same full-image coordinate frame as the AI and reference rectangles. Keep the screenshot, anchors and annotations together in the fixture record. Do not mix anchors from another panel, crop, symbol or timeframe.

## Important limitations

There are currently no manually annotated real chart screenshots in this repository. Unit tests use synthetic rectangles, synthetic axis anchors and temporary synthetic PNGs only to verify scoring and file-integrity logic; they are not evidence of model accuracy. Do not publish a zone-detection accuracy or claim improvement until a real, manually reviewed fixture set has been scored.

IoU and normalized center displacement measure screenshot geometry only. Price-boundary error is only as trustworthy as the manually read axis anchors and chosen scale. Neither metric verifies that a zone is structurally meaningful, fresh/unmitigated, or useful for a trading decision.
