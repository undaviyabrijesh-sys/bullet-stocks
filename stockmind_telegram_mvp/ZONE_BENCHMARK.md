# StockMind chart-zone benchmark

This benchmark compares AI overlay rectangles with zones manually marked on the exact same chart screenshot. It is an offline evaluation aid; it does not alter bot responses or make a live model call.

## Prediction provenance

Every fixture must include a prediction_metadata object containing provider, the model identifier returned by the provider, requested_model, prompt_version, prompt_sha256, generated_at, settings and optional system_fingerprint. generated_at must be an ISO-8601 timestamp with an explicit timezone. Missing provider-returned model IDs or timestamps are not guessed; such incomplete metadata cannot be benchmarked.

The chart analyzer returns this metadata with its structured result. prompt_sha256 hashes the static system/chart instructions and structured response schema (not the screenshot, user text or prior conversation). Bump CHART_PROMPT_VERSION when the intended prompt behavior changes; the hash also detects accidental prompt edits that were not accompanied by a version bump. settings records the chart request's temperature, image detail, response format and any reasoning setting.

One manifest may contain only one prediction profile: provider, returned/requested model, prompt version/hash, system fingerprint and settings must match across all fixtures. Per-fixture generation timestamps may differ. A mixed-profile manifest fails rather than silently averaging unlike model configurations. To compare model configurations, create separate manifests using the same screenshots and manual reference zones, then compare their reports.

## Annotation rules

Every fixture must include an `annotation_metadata` object. Only annotations that have been independently reviewed and approved are eligible for scoring:

~~~json
{
  "source": "manual",
  "annotator_id": "ann-001",
  "reviewer_id": "rev-002",
  "guideline_version": "dns-manual-v1",
  "review_status": "approved",
  "annotated_at": "2026-11-05T19:00:00Z",
  "reviewed_at": "2026-11-05T19:15:00Z"
}
~~~

- Use pseudonymous IDs with the `ann-` and `rev-` prefixes. Never put names, email addresses, chat handles or other personal information in benchmark metadata; the validator rejects extra fields.
- Use a distinct reviewer ID (`rev-...`); project controls must ensure it maps to a different human from the annotator. A pending, rejected or self-reviewed annotation cannot be scored.
- The validator checks declared metadata and formats; it cannot authenticate annotators or prove that a human review actually occurred. Keep reviewer-to-ID mappings and audit evidence in an access-controlled system outside this repository.
- Both timestamps must be ISO-8601 with an explicit timezone, and review time cannot precede annotation time.
- Use one guideline version throughout a manifest. When annotation rules change, increment the version and benchmark that release separately.
- Preserve the provenance metadata with each report so a later model comparison can be traced to its approved ground truth.

- Keep the original screenshot unchanged and record its SHA-256, width and height.
- Mark each visible reference zone as supply or demand.
- Express each rectangle relative to the full screenshot: x_start, x_end, y_top and y_bottom are normalized to 0..1.
- Use the same full-image coordinate frame for the AI prediction and manual annotation. Do not crop one side or score against a different screenshot.
- Keep no more than eight reference zones, matching the current overlay cap.
- Do not infer unreadable price values. Geometry-only fixtures need no price-axis data.
- For price-level scoring, manually record at least three legible price-axis anchors from the same screenshot. Each anchor is an object with y normalized to the full screenshot and a positive numeric price. Record anchors at distinct vertical positions and in top-to-bottom order with descending prices.
- Choose linear for a standard arithmetic price axis or log for a logarithmic axis. Do not use the logarithmic option merely because price values are large.

A fixture record should contain a stable fixture_id, image_file, image_sha256, image_width, image_height, reference_zones, predicted_zones, prediction_metadata and approved annotation_metadata. The image_file must be a relative path contained within the manifest directory. Optional `market_context` can record `asset_class`, `symbol`, `exchange` and `timeframe`. Values are trimmed, limited to 32/64/64/16 characters respectively, and control characters are rejected. Asset class, exchange and timeframe are case-folded; symbols are uppercased so case-only variants share one group. Omitted dimensions are reported in a `null` group rather than silently excluded. Timeframe aliases such as `60m` and `1h` are intentionally not assumed equivalent. Each reference zone needs zone_type and the four normalized coordinates. Predicted zones should be the structured AI zones, including draw_overlay. Optional price_axis_anchors and price_axis_scale fields enable price-unit scoring. Additional descriptive metadata fields may be included.

## Geometry scoring

Call evaluate_zone_predictions(predicted_zones, reference_zones, image_width=1600, image_height=900, iou_threshold=0.5) from app.zone_evaluation. Always pass the original screenshot dimensions.

The evaluator:
- matches only zones with the same type (supply-to-supply, demand-to-demand);
- performs one-to-one matching, maximizing the number of qualifying matches before maximizing total IoU;
- reports precision, recall, F1, false positives, false negatives, mean matched intersection-over-union (IoU), and normalized center displacement;
- uses the same pixel rounding as the Telegram renderer and ignores predictions narrower than 3 px or shorter than 2 px, exactly as the renderer does;
- treats normalized pixel boxes as half-open bounds and converts them to Pillow's inclusive `right - 1`, `bottom - 1` drawing endpoints, so the rendered fill extent matches the evaluator's pixel geometry;
- converts pixel-rounded boxes back to normalized screenshot coordinates before IoU and center-error scoring, and uses those same quantized y-boundaries for price-axis errors;
- reports invalid, disabled or sub-minimum AI rectangles separately as ignored predictions; a sub-minimum prediction leaves its manual reference as a false negative;
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
  ],
  "market_context": {
    "asset_class": "crypto",
    "symbol": "BTCUSDT",
    "exchange": "binance",
    "timeframe": "1h"
  },
  "annotation_metadata": {
    "source": "manual",
    "annotator_id": "ann-001",
    "reviewer_id": "rev-002",
    "guideline_version": "dns-manual-v1",
    "review_status": "approved",
    "annotated_at": "2026-11-05T19:00:00Z",
    "reviewed_at": "2026-11-05T19:15:00Z"
  },
  "prediction_metadata": {
    "provider": "openai",
    "model": "provider-returned-model-id",
    "requested_model": "configured-model-alias",
    "prompt_version": "dns-chart-v1",
    "prompt_sha256": "replace-with-64-character-prompt-sha256",
    "generated_at": "2026-11-05T20:24:00Z",
    "system_fingerprint": null,
    "settings": {
      "temperature": 0.1,
      "image_detail": "high",
      "response_format": "json_schema:chart_zone_map"
    }
  }
}
~~~

Run from the repository root:

~~~bash
python -m stockmind_telegram_mvp.app.zone_benchmark stockmind_telegram_mvp/benchmark/fixtures.jsonl
# If MAX_CHART_IMAGE_BYTES is customized in the bot environment, pass the same limit:
python -m stockmind_telegram_mvp.app.zone_benchmark stockmind_telegram_mvp/benchmark/fixtures.jsonl --max-image-bytes 10485760
~~~

The JSON report is written to standard output and can be redirected to a versioned report file. Each fixture report includes a SHA-256 fingerprint of the exact manual reference-zone JSON, the price-axis calibration settings (when present), and the IoU threshold. These fingerprints allow paired model comparisons to prove that both reports used the same ground truth and scoring rules. In addition to overall and supply/demand metrics, it includes micro-aggregated breakdowns by asset class, symbol, exchange and timeframe. Missing context is retained as a `null` group so incomplete metadata is visible. Price-boundary errors are also aggregated within each context group when calibrated fixtures are available. A missing/empty manifest, malformed record, missing/invalid provenance, invalid market context, mixed prediction profiles, mixed annotation guideline versions, missing or unapproved annotation provenance, hash mismatch, changed dimensions, unsupported image or invalid manual annotation fails the run instead of emitting misleading aggregate metrics. The runner performs no AI request; predictions must be captured separately and paired with the exact source screenshot.

Aggregate precision/recall/F1 are micro-averaged from total TP/FP/FN counts. Mean IoU and center displacement are averaged across matched zones, not across screenshots. Price errors are averaged across scored zone boundaries; zones outside the calibrated y-range are counted as unscored rather than extrapolated.

## Paired model comparison

Generate one benchmark JSON report per model using the same reviewed screenshot fixtures and reference annotations. Save each command's standard output to a separate report file, then run:

~~~bash
python -m stockmind_telegram_mvp.app.zone_benchmark_compare baseline-report.json candidate-report.json
~~~

The comparison is paired by `fixture_id`. It refuses to compare reports unless they contain exactly the same fixture IDs and, for every fixture, the same screenshot SHA-256, reference-zone fingerprint, annotation provenance, market context, IoU threshold and price-axis calibration fingerprint. Both reports must use the same annotation guideline version. Missing, extra or altered fixtures fail closed rather than creating a misleading comparison.

Before comparing, the comparator revalidates report provenance, annotation metadata, normalized market context, confusion-matrix arithmetic, precision/recall/F1, match uniqueness, IoU and center-error averages, supply/demand reconciliation, per-zone price calculations, fixture-level price summaries and report-level aggregates. A report whose summary fields do not agree with its included fixture evidence is rejected rather than trusted.

The JSON output includes both prediction profiles, candidate-minus-baseline deltas for overall and supply/demand metrics, per-fixture changes, market-context breakdowns, ignored prediction counts and price-boundary summaries. Positive precision/recall/F1/IoU deltas indicate higher values; negative center-error and price-error deltas indicate lower errors. A `null` delta means one side has no defined metric; it is not treated as zero. The comparator does not declare a trading winner or infer structural quality.

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
