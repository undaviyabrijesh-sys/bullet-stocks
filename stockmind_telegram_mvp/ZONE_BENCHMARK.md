# StockMind chart-zone benchmark

This benchmark compares AI overlay rectangles with zones manually marked on the exact same chart screenshot. It is an offline evaluation aid; it does not alter bot responses or make a live model call.

## Strict JSON inputs

The chart-response parser, annotation editor, prediction-capture loader, JSONL benchmark runner and paired-report loader reject duplicate object keys and non-standard numeric constants (`NaN`, `Infinity` and `-Infinity`). Python's default JSON decoder otherwise accepts these ambiguous/non-standard inputs (and silently keeps the last duplicate key). Rejecting them keeps model output, screenshot provenance and benchmark evidence unambiguous. Ordinary JSON strings containing words such as `NaN` remain valid. The model tool-argument parser uses the same strict decoder, so duplicate keys and non-standard numeric constants are rejected before any market-data tool is dispatched.

Plot-boundary report evidence is also compared with exact JSON value types after its pixel metrics are recomputed. In particular, JSON booleans cannot stand in for numeric values (true is not an acceptable substitute for 1 or 1.0, and false is not an acceptable substitute for 0 or 0.0). This matters because Python's ordinary equality treats booleans as integers. The validator checks nested edge metrics and pixel evidence as well as top-level scores.

## Prediction provenance

Every fixture must include a prediction_metadata object containing provider, the model identifier returned by the provider, requested_model, prompt_version, prompt_sha256, generated_at, settings and optional system_fingerprint. The chart prompt is supplied as a separate system message from the user's caption. Chart-image requests are stateless: prior conversation turns are not forwarded to the vision model, preventing stale levels and earlier user instructions from contaminating a new screenshot analysis while reducing input-token cost. Text embedded in a screenshot is treated as chart content, not executable instructions; the current caption can describe the requested analysis but cannot override the chart evidence rules or structured output contract. generated_at must be an ISO-8601 timestamp with an explicit timezone. Missing provider-returned model IDs or timestamps are not guessed; such incomplete metadata cannot be benchmarked.

The chart analyzer returns this metadata with its structured result. prompt_sha256 hashes the static system/chart instructions, structured response schema and chart context policy (not the screenshot, user caption or prior conversation). Bump CHART_PROMPT_VERSION when the intended prompt behavior changes; the hash also detects accidental prompt edits that were not accompanied by a version bump. settings records the chart request's temperature, image detail, response format, reasoning setting and context policy.

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
- When `reference_plot_bounds` or `price_axis_plot_bounds` is supplied, every manual reference zone must fit completely inside that independently reviewed main price-plot rectangle after pixel rounding. If a zone is clipped by the screenshot edge, annotate only its visible portion; do not extrapolate it beyond the plot. The evaluator rejects manual zones outside the supplied plot so off-panel annotations cannot silently distort precision/recall.
- Use the same full-image coordinate frame for the AI prediction and manual annotation. Do not crop one side or score against a different screenshot.
- Keep no more than eight reference zones, matching the current overlay cap.
- Do not infer unreadable price values. Geometry-only fixtures need no price-axis data.
- For price-level scoring, manually record at least three legible price-axis anchors from the same screenshot and manually mark the main price-plot rectangle as `price_axis_plot_bounds`. Each anchor is an object with y normalized to the full screenshot and a positive numeric price. Anchor y-coordinates must fall inside the marked plot's vertical span (the labels themselves may sit in the adjacent price scale). Record anchors at distinct vertical positions and in top-to-bottom order with descending prices. Anchors in toolbars or indicator panels are rejected.
- Choose linear for a standard arithmetic price axis or log for a logarithmic axis. Do not use the logarithmic option merely because price values are large.

A fixture record should contain a stable fixture_id, image_file, image_sha256, image_width, image_height, reference_zones, predicted_zones, prediction_metadata and approved annotation_metadata. The image_file must be a relative path contained within the manifest directory. Optional `market_context` can record `asset_class`, `symbol`, `exchange` and `timeframe`. Values are trimmed, limited to 32/64/64/16 characters respectively, and control characters are rejected. Asset class, exchange and timeframe are case-folded; symbols are uppercased so case-only variants share one group. Omitted dimensions are reported in a `null` group rather than silently excluded. Timeframe aliases such as `60m` and `1h` are intentionally not assumed equivalent. Each reference zone needs zone_type and the four normalized coordinates. Predicted zones should be the structured AI zones, including draw_overlay. Because the Telegram renderer requires a valid model plot frame before drawing any zone, include predicted_plot_bounds and an independently reviewed reference_plot_bounds for scored fixtures. If predicted_plot_bounds is missing or invalid, all AI zones are treated as ignored predictions (and matching manual zones remain false negatives), even when no manual plot boundary is supplied; they are never credited as drawable detections. If a model plot frame is supplied without a manual reference plot, the manifest is rejected rather than reporting an unpaired plot score. When price_axis_anchors are present, price_axis_plot_bounds is required and must describe the manually reviewed main price-plot rectangle; price_axis_scale selects linear or log price-unit scoring. Additional descriptive metadata fields may be included.

## Structured price-label validation

The live chart-response sanitizer accepts a numeric price range only when both
fields are plain positive decimal strings and price_low < price_high. The
higher value belongs at y_top; the lower value belongs at y_bottom. If
either field is incomplete, malformed, non-positive or reversed, both structured
price fields are withheld and the analysis receives an explicit warning. Valid
pixel geometry remains independently drawable: a bad text range does not move
or silently repair the overlay.

This ordering check does not prove that the stated prices correspond to the
zone's normalized y-coordinates. The live Telegram flow has no independently
verified axis anchors, so it must not claim that mapping is calibrated. The
offline evaluator's existing price_accuracy measures coordinate-derived
prices against manual reference geometry using manually entered linear/log
anchors. Comparing the model's stated price_low/price_high labels directly
with those calibrated coordinate prices remains a separate benchmark task.
## Geometry scoring

Call evaluate_zone_predictions(predicted_zones, reference_zones, image_width=1600, image_height=900, iou_threshold=0.5) from app.zone_evaluation. Always pass the original screenshot dimensions.

The evaluator:
- requires valid predicted_plot_bounds before any prediction can count as a drawable overlay, matching the Telegram renderer even for fixtures that omit manual plot annotations;
- matches only zones with the same type (supply-to-supply, demand-to-demand);
- performs one-to-one matching, maximizing the number of qualifying matches before maximizing total IoU;
- reports precision, recall, F1, false positives, false negatives, mean matched intersection-over-union (IoU), and normalized center displacement;
- uses the same pixel rounding as the Telegram renderer and ignores predictions narrower than 3 px or shorter than 2 px, exactly as the renderer does;
- treats normalized pixel boxes as half-open bounds and converts them to Pillow's inclusive `right - 1`, `bottom - 1` drawing endpoints, so the rendered fill extent matches the evaluator's pixel geometry;
- converts pixel-rounded boxes back to normalized screenshot coordinates before IoU and center-error scoring, and uses those same quantized y-boundaries for price-axis errors;
- reports invalid, disabled or sub-minimum AI rectangles separately as ignored predictions; a sub-minimum prediction leaves its manual reference as a false negative;
- treats a disabled/invalid AI rectangle as no drawn prediction, so a missed reference zone remains a false negative;
- rejects malformed manual annotations rather than silently corrupting the benchmark;
- scores up to eight drawable AI zones, matching the renderer; malformed or out-of-plot candidates do not consume that limit.

Use the same IoU threshold across a benchmark release. Report aggregate and per-fixture results, and retain the fixture/image hash so future model changes can be compared on identical inputs.

## Batch manifest runner

The batch runner verifies that each fixture still points to the exact annotated screenshot before scoring it. It reads at most the configured byte limit plus one byte, checks the SHA-256, actual image format, decoded image limit and recorded dimensions; rejects duplicate fixture IDs, duplicate screenshot SHA-256 hashes (so identical image bytes cannot silently overweight the sample), and paths that escape the manifest directory; then reports per-fixture results and micro-aggregated precision, recall, F1, IoU, center displacement and optional price-boundary errors. The default byte limit is 5 MiB, matching the Telegram bot's default upload cap. Use `--max-image-bytes` when the bot's `MAX_CHART_IMAGE_BYTES` setting has been overridden.

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

Screenshot uniqueness is byte-level: the runner rejects two fixtures with the same SHA-256, but does not attempt perceptual matching of visually identical screenshots that were re-encoded or cropped differently.

The report includes both micro- and macro-averaged DNS metrics. Micro precision/recall/F1 are calculated from total TP/FP/FN counts, so charts with many annotated zones contribute more. Macro metrics first calculate each fixture's precision, recall, F1, mean IoU and mean center displacement, then take the arithmetic mean across fixtures so every screenshot has equal weight. Macro F1 is the mean of fixture-level F1 values; it is not recomputed from macro precision and macro recall. Undefined per-fixture values (for example, precision when a chart has no drawable predictions) remain excluded rather than being treated as zero or one. A chart with no reference zones and no drawable predictions therefore remains in the total fixture count but contributes to no defined macro metric; this avoids inventing a score for an empty chart. Each macro metric reports its own defined-fixture count and coverage ratio (defined fixtures divided by total fixtures), and the total fixture count remains visible. A zero coverage ratio means the metric is undefined for every chart, not that the score is zero. The same macro metrics are reported overall and separately for supply and demand. Context breakdowns remain micro-aggregated. Mean IoU and center displacement in the micro report are averaged across matched zones, not across screenshots. Price errors are averaged across scored zone boundaries; zones outside the calibrated y-range are counted as unscored rather than extrapolated.


## Plot-boundary and overlay containment scoring

A fixture may optionally include both `reference_plot_bounds` (manually reviewed) and
`predicted_plot_bounds` (returned by the AI), each using normalized full-screenshot
coordinates. If only the manual reference is present, the report records a missing
model boundary rather than a perfect score. A predicted boundary without an independent
manual reference is rejected. When `price_axis_plot_bounds` is already supplied for
price-axis calibration, it can serve as the manual plot reference; if both manual
fields are supplied, they must resolve to the same pixel rectangle.

When a manual plot reference is supplied, DNS scoring mirrors the production
renderer: a prediction is counted as drawable only when its model-estimated plot
bounds are valid and the pixel-rounded zone rectangle fits completely inside that
model plot. If the model plot boundary is missing or invalid, all candidate
overlays are withheld; if an individual zone crosses the model plot boundary,
that zone is counted as ignored and its matching manual reference remains a false
negative. This prevents benchmark precision/recall from crediting rectangles
that Telegram would never display. The report's pixel evidence is also checked
against the model plot bounds during paired comparison.

For every fixture with a manual plot reference, the evaluator reports:

- plot-boundary IoU and left/right/top/bottom edge displacement in pixels and normalized units;
- the percentage of the predicted plot rectangle outside the manual plot;
- right-margin and bottom-panel overlap percentages. These are geometric side-region
  proxies only: the evaluator does not prove that the right margin is a price scale or
  that the lower region is an indicator panel;
- the number and percentage of drawable predicted zones that are not fully contained
  in the manual plot, plus the percentage of predicted zone area outside it;
- right-margin and bottom-panel overlap of predicted zone area, separately from the
  total outside-area measure.

Invalid or missing model plot bounds have no IoU or edge score. Zone-containment
measurements remain available when the manual plot boundary is valid. Paired comparison
requires the exact same manual plot-boundary fingerprint for each screenshot and
revalidates the per-fixture pixel evidence before calculating deltas.

Example optional manifest fields:

~~~json
{
  "reference_plot_bounds": {
    "x_start": 0.08, "x_end": 0.92, "y_top": 0.12, "y_bottom": 0.90
  },
  "predicted_plot_bounds": {
    "x_start": 0.08, "x_end": 0.91, "y_top": 0.12, "y_bottom": 0.90
  }
}
~~~


## Paired model comparison

Generate one benchmark JSON report per model using the same reviewed screenshot fixtures and reference annotations. Save each command's standard output to a separate report file, then run:

~~~bash
python -m stockmind_telegram_mvp.app.zone_benchmark_compare baseline-report.json candidate-report.json
~~~

The comparison is paired by `fixture_id`. It refuses to compare reports unless they contain exactly the same fixture IDs and, for every fixture, the same screenshot SHA-256, reference-zone fingerprint, annotation provenance, market context, IoU threshold and price-axis calibration fingerprint. Both reports must use the same annotation guideline version. Missing, extra or altered fixtures fail closed rather than creating a misleading comparison.

Before comparing, the comparator revalidates report provenance, annotation metadata, normalized market context, confusion-matrix arithmetic, precision/recall/F1, match uniqueness, IoU and center-error averages, supply/demand reconciliation, per-zone price calculations, fixture-level price summaries and report-level aggregates. A report whose summary fields do not agree with its included fixture evidence is rejected rather than trusted. All report-level aggregates use type-aware JSON comparison: booleans cannot masquerade as numeric scores or counts, integer counts cannot be floating-point values, and prediction-profile settings must match their canonical JSON fingerprint (so `true` is not `1`). Macro score and coverage values must be finite JSON numbers, and fixture counts must be JSON integers.

The JSON output includes both prediction profiles, candidate-minus-baseline deltas for micro-aggregated overall and supply/demand metrics, equal-weight macro metrics for overall and supply/demand results (including each metric's defined-fixture counts and coverage ratios), per-fixture changes, market-context breakdowns, ignored prediction counts and price-boundary summaries. Positive precision/recall/F1/IoU deltas indicate higher values; negative center-error and price-error deltas indicate lower errors. A `null` delta means one side has no defined metric; it is not treated as zero. The comparator does not declare a trading winner or infer structural quality.

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
    price_axis_plot_bounds={
        "x_start": 0.08, "x_end": 0.92,
        "y_top": 0.12, "y_bottom": 0.9,
    },
    price_axis_scale="linear",
)
~~~

Price-axis calibration:
- requires 3..20 manually verified anchors and a manually marked main price-plot rectangle; two points are deliberately insufficient because they cannot expose a distorted/non-uniform scale;
- rejects anchor y-coordinates outside the marked main price plot, preventing toolbar or lower indicator-panel labels from calibrating the candle chart;
- fits either a linear price axis or a linear fit in log-price space;
- rejects reversed/non-monotonic anchors and fits whose maximum anchor residual exceeds 2% by default (the caller may set a stricter tolerance, up to 10%);
- never extrapolates beyond the first/last anchor y-coordinate;
- reports matched-zone high/low boundary error from normalized, pixel-rounded geometry in price units and as a percentage, separately from IoU;
- separately compares the model's stated `price_low`/`price_high` with the prices implied by those same pixel-rounded overlay coordinates, reporting absolute and percentage disagreement plus scored/unscored counts;
- marks a matched zone as unscored if either boundary falls outside the calibrated y range rather than guessing a price;
- marks a price comparison unscored if an absolute or percentage error cannot be represented as a finite number, and uses overflow-safe arithmetic for aggregate means.

This is a manual calibration and scoring tool, not automatic axis OCR. Anchor y-coordinates must use the same full-image coordinate frame as the AI and reference rectangles. Keep the screenshot, anchors and annotations together in the fixture record. Do not mix anchors from another panel, crop, symbol or timeframe.

For each matched zone, `price_accuracy` keeps two measurements separate: (1) geometry price error, which compares the model's pixel-rounded rectangle with the manually annotated rectangle after axis calibration; and (2) `reported_price_consistency`, which compares the model's stated numeric bounds with prices derived from its own pixel-rounded rectangle. The second metric detects a model response whose drawn rectangle and written price labels contradict one another. It is unscored when either reported price is missing/invalid or a boundary falls outside the manually calibrated y range; no extrapolation or guessed values are used. Relative disagreements within 1e-12 are normalized to zero to avoid treating binary floating-point interpolation noise as a price-label mismatch; this tolerance is computational only and does not imply market-price precision. These reported-label consistency counts apply to matched zones only, not unmatched false-positive predictions. Paired reports show baseline, candidate and candidate-minus-baseline deltas for this metric. It does not establish that the manually read axis labels or the underlying zone interpretation are correct.

## Important limitations

There are currently no manually annotated real chart screenshots in this repository. Unit tests use synthetic rectangles, synthetic axis anchors and temporary synthetic PNGs only to verify scoring and file-integrity logic; they are not evidence of model accuracy. Do not publish a zone-detection accuracy or claim improvement until a real, manually reviewed fixture set has been scored.

IoU and normalized center displacement measure screenshot geometry only. Price-boundary error is only as trustworthy as the manually read axis anchors and chosen scale. Neither metric verifies that a zone is structurally meaningful, fresh/unmitigated, or useful for a trading decision.


## Screenshot intake drafts

Use the intake helper to preserve an original screenshot and create a **pending annotation draft** before manually marking it. Run from the repository root:

```bash
python -m stockmind_telegram_mvp.app.annotation_intake path/to/chart.png \
  --output-dir stockmind_telegram_mvp/benchmark/inbox \
  --fixture-id btc-1h-001 \
  --asset-class crypto --symbol BTCUSDT --exchange binance --timeframe 1h
```

The command accepts static JPEG, PNG or WEBP images only. It applies the same decoded-image validation used by the Telegram upload path, reads at most the configured byte limit plus one byte, copies the original encoded bytes unchanged, and records SHA-256 plus actual pixel dimensions. It refuses path-like fixture IDs, overwrites, output image-directory symlinks that escape the inbox, oversized or malformed files, animated images, and unsupported market-context fields. Context values are normalized using the benchmark's grouping conventions. The draft does not record the source machine's path or any user identity.

The output contains `images/<fixture-id>.<ext>` and `<fixture-id>.draft.json`. The draft deliberately has `benchmark_ready: false`, empty reference zones, null plot bounds, pending annotation metadata, and no AI prediction/provenance. **It is not a benchmark JSONL record and must not be passed to the benchmark runner.** A human annotator must add the visible supply/demand rectangles and reviewed main-plot bounds, then a different human must review them and fill pseudonymous annotator/reviewer IDs plus timezone-aware timestamps. Capture the actual AI prediction and provider-returned prediction metadata for the exact same screenshot separately. Only after those steps should a benchmark JSONL fixture be assembled using the schema above.

The intake helper never invents chart levels, zone coordinates, model output, annotator identities or approval times. Its tests use synthetic images only to verify file handling; they are not chart-accuracy fixtures. The inbox path is ignored by Git so private chart screenshots and unfinished drafts are not accidentally committed. Keep the original screenshot and reviewer audit evidence in access-controlled storage; the manifest's pseudonymous IDs alone do not authenticate the humans who approved an annotation.


## Offline visual annotation editor

The intake draft can now be annotated and independently reviewed with a local Tk desktop editor. It uses only the screenshot and draft on the selected machine; it does not call a model, market-data provider, Telegram, or a network service. The original screenshot and intake draft are never modified.

Run from the repository root on a desktop Python installation with Tk support:

```bash
# First pass: draw the main price-plot boundary, then supply/demand rectangles.
python -m stockmind_telegram_mvp.app.annotation_editor path/to/btc-1h-001.draft.json --mode annotate

# Independent second pass: inspect the saved rectangles and approve or reject.
python -m stockmind_telegram_mvp.app.annotation_editor path/to/btc-1h-001.draft.annotated.json --mode review
```

The editor resolves the screenshot path from the draft's relative `image_file` by default. Use `--image path/to/screenshot.png` only when the screenshot is stored elsewhere. Before opening the editor it uses the same chart-image preparation routine as inference, enforces the 5 MiB editor limit, and verifies the exact SHA-256 and pixel dimensions recorded in the draft. It rejects screenshots whose EXIF orientation would change the encoded bytes; re-run `annotation_intake` and use its canonical image rather than annotating a frame that the benchmark will later reject. A mismatch is rejected rather than allowing annotations against a different image.

In annotation mode, select `plot` and drag the main price-plot rectangle first. Then select `demand` or `supply` and draw each visible zone. Coordinates are normalized to the full original screenshot, matching the benchmark schema. Zones must fit completely inside the marked plot, and no more than eight reference zones are accepted. A chart with no visible supply/demand zone may be saved with an empty zone list, provided the plot boundary is marked. Use an anonymized annotator ID such as `ann-001`; do not enter a person's name or contact information.

For optional price-boundary scoring, select `axis_anchor`, choose `linear` or `log`, click inside both the horizontal and vertical bounds of the marked main price plot at the vertical level of a legible price-axis tick (do not click in the right-side price-label margin), and enter that exact visible price label. Clicks in unused canvas space beyond the screenshot are rejected even when the marked plot reaches an image edge; they are not clamped into a valid calibration point. Add 3..20 anchors at distinct y positions, from top to bottom with descending prices. This is manual entry, not OCR: do not estimate unreadable labels or use values from another screenshot. Anchor y-coordinates are normalized against the full screenshot and must lie inside the marked plot's vertical span. On save, the editor fits the selected scale using the existing 2% default maximum anchor residual; inconsistent anchors are rejected. The saved `price_axis_plot_bounds` is copied from the manually marked plot rectangle. If price labels are unreadable, leave the anchor tool unused and save a geometry-only annotation.

The annotation pass writes a new `.annotated.json` file with `status: pending_review`, a timezone-aware timestamp and pending review metadata. Review mode is read-only: a separate reviewer enters an ID such as `rev-002` and explicitly approves or rejects the rectangles. The review pass writes a new `.reviewed.json` file. Existing files are never overwritten. Both approved and rejected drafts remain `benchmark_ready: false`; approval alone does not create a benchmark fixture.

The editor records the manually marked main plot rectangle, supply/demand geometry, and optional manually read price-axis anchors plus the chosen linear/log scale. Before annotation can be saved or reviewed, each reference zone is checked at the original screenshot dimensions using the evaluator's pixel rounding and reference-zone rules: it must remain at least one pixel wide and high and cannot exceed 45% of screenshot height. This prevents approval of boxes that the benchmark would later reject as unscorable. When anchors are present, the saved `price_axis_plot_bounds` must match the reviewed main plot and the 3..20 anchors must pass the calibration fit. The reviewer sees the anchor markers and values in read-only mode and revalidates them before approval. After approval, capture the actual AI prediction and provider-returned prediction metadata for the exact same screenshot, then assemble the benchmark JSONL record using the schema above. The benchmark runner remains the final validator. The editor's geometry/state-transition tests run headlessly; an interactive Tk window still needs a manual desktop smoke test on the target operating system. The CLI reports success only when the current editor session completes its own save operation; an output file appearing concurrently from another process does not count as a successful save. Closing the window without saving exits with status 1; successfully saving an annotation or review in the current session exits with status 0.


## Prediction capture and fixture assembly

After a reviewer has approved a .reviewed.json annotation, pair it with the
actual result returned by ResearchAI.analyze_chart_image. That result now
includes source_image_sha256, computed from the exact image bytes submitted
to the vision request. The hash prevents a prediction from another screenshot
being attached to the reviewed annotation.

The current Telegram bot does not automatically persist user screenshots or
chart results. For a local development harness, serialize the actual returned
chart-result dictionary as UTF-8 JSON without editing its geometry or
prediction_metadata. It must contain the provider-returned model and generation
timestamp; if either is missing, capture fails rather than substituting a
configured alias or local clock.

Create a hash-bound prediction capture:

~~~bash
python -m stockmind_telegram_mvp.app.annotation_fixture capture \
  stockmind_telegram_mvp/benchmark/inbox/btc-1h-001.reviewed.json \
  path/to/actual-chart-result.json
~~~

Then assemble and preflight one JSONL fixture:

~~~bash
python -m stockmind_telegram_mvp.app.annotation_fixture assemble \
  stockmind_telegram_mvp/benchmark/inbox/btc-1h-001.reviewed.json \
  stockmind_telegram_mvp/benchmark/inbox/btc-1h-001.prediction.json

python -m stockmind_telegram_mvp.app.zone_benchmark \
  stockmind_telegram_mvp/benchmark/inbox/btc-1h-001.fixture.jsonl
~~~

The capture contains only the screenshot hash/dimensions, fixture ID and
sanitized model result/provenance; it does not copy the screenshot or store a
Telegram user/chat ID. The assembler requires an independently approved
annotation, validates the manual geometry and optional price-axis calibration,
checks fixture ID/hash/dimensions, rejects altered or non-sanitized model
geometry, and requires real provider-returned model/time metadata. It writes
the manifest beside the reviewed draft so the relative image_file remains
valid. Existing outputs are never overwritten.

Before writing the JSONL file, the assembler runs the exact benchmark manifest
validator against a temporary one-record manifest. A successful preflight
means the record is structurally scoreable; it does not establish chart
accuracy. Each command creates one fixture record. Combine records from the
same annotation inbox into a separate JSONL manifest to score a dataset.
There are still no real manually reviewed chart fixtures in this repository;
the tests use synthetic images and rectangles only.


## Screenshot orientation and exact-byte matching

EXIF orientation values include mirrored, transposed and transverse images as well as rotations. The normalization contract covers values 2 through 8; value 1 remains byte-for-byte unchanged. Tests use an asymmetric color grid to verify exact pixel placement, not only output dimensions, so a mirrored or transposed frame cannot pass a rotation-only check.

Telegram chart uploads and local benchmark screenshots are passed through the same
`prepare_chart_image` preprocessing step. It validates the encoded image and
normalizes EXIF rotation/flip metadata before the bytes are sent to the vision model
or used by the Pillow overlay renderer. This prevents a provider that honors EXIF
orientation from producing coordinates in a different frame than the overlay.

The annotation intake utility applies the same normalization before saving the
screenshot, dimensions and SHA-256. The hash therefore identifies the canonical
bytes stored in the inbox, not the original EXIF-bearing upload. For a benchmark,
use the exact normalized image saved under `images/` as the screenshot submitted
to the bot; do not independently re-export or recompress it. The model provenance
hash is computed from the exact bytes sent to vision. If orientation normalization
would exceed the configured byte limit, processing stops with an explicit error
rather than sending mismatched or oversized data.

The benchmark runner independently enforces this contract: it calls the same
preparation routine and rejects a fixture image if EXIF normalization would change
its bytes. This prevents a hand-built manifest from scoring model coordinates
against a raw, rotated screenshot. Re-run `annotation_intake` and use the
canonical image it writes; do not simply update the manifest hash around an
EXIF-bearing source file.


## Direct overlay renderer orientation contract

The public `render_overlay` entry point always validates and normalizes the
supplied screenshot before converting normalized coordinates to pixels. There
is no public boolean flag that disables image validation or EXIF normalization.

The Telegram handler has already run `prepare_chart_image` on the exact bytes
sent to the vision provider. To avoid decoding a large screenshot a second time,
that internal handler uses `_render_overlay_prepared` with those same canonical
bytes. This underscore-prefixed helper is an internal fast path; direct callers
and offline tools should use `render_overlay`.
