# StockMind chart-zone benchmark

This benchmark compares AI overlay rectangles with zones manually marked on the exact same chart screenshot. It is an offline evaluation aid; it does not alter bot responses or make a live model call.

## Strict JSON inputs

The chart-response parser, annotation editor, prediction-capture loader, JSONL benchmark runner and paired-report loader reject duplicate object keys and non-standard numeric constants (`NaN`, `Infinity` and `-Infinity`). Python's default JSON decoder otherwise accepts these ambiguous/non-standard inputs (and silently keeps the last duplicate key). Rejecting them keeps model output, screenshot provenance and benchmark evidence unambiguous. Ordinary JSON strings containing words such as `NaN` remain valid. The model tool-argument parser uses the same strict decoder, so duplicate keys and non-standard numeric constants are rejected before any market-data tool is dispatched.

Plot-boundary report evidence is also compared with exact JSON value types after its pixel metrics are recomputed. In particular, JSON booleans cannot stand in for numeric values (true is not an acceptable substitute for 1 or 1.0, and false is not an acceptable substitute for 0 or 0.0). This matters because Python's ordinary equality treats booleans as integers. The validator checks nested edge metrics and pixel evidence as well as top-level scores.

## Prediction provenance

Every fixture must include a prediction_metadata object containing provider, the model identifier returned by the provider, requested_model, prompt_version, prompt_sha256, generated_at, settings and optional system_fingerprint. The current `dns-chart-v9` chart-analysis contract requires a qualitative per-zone confidence category (`high`, `medium` or `low`) and records `confidence_policy=high_medium_only`. It also requires an explicit zone `origin` (`user_marked` or `ai_inferred`) and records `zone_origin_policy=user_marked_only_when_visibly_marked`. Version 9 keeps the version-8 evidence gate and fixes a contradictory prompt sequence by requiring main price-plot bounds to be identified before any zone mapping. Version 8 introduced `zone_evidence_policy=compact_base_plus_directional_displacement_v1`: a candidate must be anchored to a compact base or last opposing candle/cluster immediately before a clear directional departure (up for demand, down for supply). An isolated wick, a single rejection without follow-through, a horizontal level alone, or a pause without displacement is insufficient. The prompt prefers fewer supported zones, avoids duplicate boxes for one base, and withholds overlays when base boundaries are ambiguous. The benchmark runner rejects v7/v8/v9 manifests that omit their required policies, omit/alter a zone confidence or origin value, use a label that contradicts its origin/type, attempt to draw a low-confidence zone, or retain non-zero geometry for any non-drawable zone. Earlier prompt versions remain readable for historical comparisons. Confidence is a model-reported evidence category, not a calibrated probability; origin is a model-reported visual classification, not independent proof of authorship.

The chart prompt is supplied as a separate system message from the user's caption. Chart-image requests are stateless: prior conversation turns are not forwarded to the vision model, preventing stale levels and earlier user instructions from contaminating a new screenshot analysis while reducing input-token cost. Text embedded in a screenshot is treated as chart content, not executable instructions; the current caption can describe the requested analysis but cannot override the chart evidence rules or structured output contract. generated_at must be an ISO-8601 timestamp with an explicit timezone. Missing provider-returned model IDs or timestamps are not guessed; such incomplete metadata cannot be benchmarked.

The chart analyzer returns this metadata with its structured result. prompt_sha256 hashes the static system/chart instructions, structured response schema and chart context policy (not the screenshot, user caption or prior conversation). Bump CHART_PROMPT_VERSION when the intended prompt behavior changes; the hash also detects accidental prompt edits that were not accompanied by a version bump. settings records the chart request's temperature, image detail, response format, reasoning setting, context policy and zone-evidence policy. Optional `usage` records only internally consistent provider-returned prompt/completion/total token counts and available image, cached-prompt and reasoning-token subtotals. Optional `request_latency_ms` measures the local round-trip around the chart completion request; it excludes image download/preparation and later overlay rendering. These per-fixture telemetry fields are not part of the prediction-profile fingerprint, because they naturally vary by screenshot. They are raw observability data, not a currency cost estimate: calculate cost only with the applicable model and date-specific pricing, and do not substitute missing usage with estimates. The v8 evidence gate is a prompt-level candidate-admission guardrail intended to reduce unsupported zones; it may also reduce recall. Version 9 changes analysis ordering only: it moves plot-boundary identification to step one so the model cannot be instructed to map zones before defining the coordinate frame. This is a prompt-consistency correction, not measured evidence of better DNS accuracy. It is not measured evidence of improved accuracy. Only a reviewed screenshot benchmark can establish the precision/recall trade-off.

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
- For each annotation, use a distinct reviewer ID (`rev-...`). Across the pair, the two reviewer suffix tokens must differ case-insensitively, and neither reviewer may share an identity token with either annotator. This rejects obvious pseudonym reuse across roles. Project controls must still ensure that the IDs map to different humans. A pending, rejected or self-reviewed annotation cannot be scored.
- The validator checks declared metadata, role prefixes and suffix-token separation; it cannot authenticate annotators or prove that a human review actually occurred. Keep reviewer-to-ID mappings and audit evidence in an access-controlled system outside this repository.
- Both timestamps must be ISO-8601 with an explicit timezone, and review time cannot precede annotation time.
- Use one guideline version throughout a manifest. When annotation rules change, increment the version and benchmark that release separately.
- Preserve the provenance metadata with each report so a later model comparison can be traced to its approved ground truth.

- Keep the canonical intake screenshot unchanged after draft creation and record its SHA-256, width and height.
- The intake tool also validates raw container boundaries before preserving an image: PNG chunk framing must terminate exactly at IEND; JPEG application/comment markers and scan framing are checked (including byte-stuffed data, restart markers, progressive multi-scan images and in-scan DNL markers); WEBP RIFF length and chunk framing must cover the exact byte stream. Unknown WEBP chunks, EXIF/XMP or animation flags, malformed chunk boundaries, and trailing bytes trigger canonical re-encoding. ICC color profiles are retained to avoid changing chart colors. PNG overlays preserve an RGBA source's alpha channel and carry a structurally recognizable RGB profile when compatible; when a PNG exceeds the lossless-output budget, or a JPEG is required for another source format, transparent pixels are explicitly composited onto white before JPEG encoding rather than having alpha silently discarded. No ICC color transform is applied. Grayscale, CMYK and malformed profiles are not attached to RGB/RGBA output; this means ICC profile headers/tags may still contain device or manufacturer descriptors, so this intake process is privacy-conscious but is not a guarantee of complete image-metadata anonymization. Animated uploads are rejected by the chart-image validator.
- Mark each visible reference zone as supply or demand.
- Express each rectangle relative to the full screenshot: x_start, x_end, y_top and y_bottom are normalized to 0..1.
- When `reference_plot_bounds` or `price_axis_plot_bounds` is supplied, every manual reference zone must fit completely inside that independently reviewed main price-plot rectangle after pixel rounding. If a zone is clipped by the screenshot edge, annotate only its visible portion; do not extrapolate it beyond the plot. The evaluator rejects manual zones outside the supplied plot so off-panel annotations cannot silently distort precision/recall.
- Use the same full-image coordinate frame for the AI prediction and manual annotation. Do not crop one side or score against a different screenshot.
- Keep no more than eight reference zones, matching the current overlay cap.
- Do not infer unreadable price values. Geometry-only fixtures need no price-axis data.
- For price-level scoring, manually record at least three legible price-axis anchors from the same screenshot and manually mark the main price-plot rectangle as `price_axis_plot_bounds`. Each anchor is an object with y normalized to the full screenshot and a positive numeric price. Anchor y-coordinates must fall inside the marked plot's vertical span (the labels themselves may sit in the adjacent price scale). The evaluator checks the original normalized screenshot coordinates, including both plot edges; it does not compare manual anchors with pixel-rounded plot edges. Record anchors at distinct vertical positions and in top-to-bottom order with descending prices. Anchors in toolbars or indicator panels are rejected.
- Choose linear for a standard arithmetic price axis or log for a logarithmic axis. Do not use the logarithmic option merely because price values are large.
- The calibration object revalidates its fit metadata on every price lookup: `anchor_count` must be an integer from 3 through 20, and `max_anchor_relative_error` must be finite and within 0..10%. Direct construction of the calibration dataclass cannot bypass these fit-quality bounds.

A fixture record should contain a stable fixture_id, image_file, image_sha256, image_width, image_height, reference_zones, predicted_zones, prediction_metadata and approved annotation_metadata. The image_file must be a relative path contained within the manifest directory. Optional `market_context` can record `asset_class`, `symbol`, `exchange` and `timeframe`. Values are trimmed, limited to 32/64/64/16 characters respectively, and control characters are rejected. Asset class, exchange and timeframe are case-folded; symbols are uppercased so case-only variants share one group. Omitted dimensions are reported in a `null` group rather than silently excluded. Timeframe aliases such as `60m` and `1h` are intentionally not assumed equivalent. Each reference zone needs zone_type and the four normalized coordinates. Predicted zones should be the structured AI zones, including `origin` (`user_marked` or `ai_inferred`), `confidence` (`high`, `medium` or `low`) and `draw_overlay`. Origin is a model-reported visual classification, not independent proof of authorship; confidence is a qualitative evidence category, not a calibrated probability. Low-confidence zones are not drawable and are excluded from prediction counts; their corresponding manual references remain false negatives. Because the Telegram renderer requires a valid model plot frame before drawing any zone, include predicted_plot_bounds and an independently reviewed reference_plot_bounds for scored fixtures. If predicted_plot_bounds is missing or invalid, all AI zones are treated as ignored predictions (and matching manual zones remain false negatives), even when no manual plot boundary is supplied; they are never credited as drawable detections. If a model plot frame is supplied without a manual reference plot, the manifest is rejected rather than reporting an unpaired plot score. When price_axis_anchors are present, price_axis_plot_bounds is required and must describe the manually reviewed main price-plot rectangle. If reference_plot_bounds is also supplied, its four normalized coordinates must match price_axis_plot_bounds so zone, plot-boundary and price-axis scores use one adjudicated manual frame; price_axis_scale selects linear or log price-unit scoring. Additional descriptive metadata fields may be included.

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

For live Telegram analysis, the sanitizer also receives the exact prepared
screenshot dimensions. It uses the same rounded half-open pixel-box helpers as
the renderer before retaining `draw_overlay=true`. Zones that collapse below
the renderer's 3-pixel width or 2-pixel height, or fall outside the quantized
plot rectangle, are zeroed before delivery. This keeps the structured response
and the final overlay's drawable-zone decision aligned; it does not change the
model's visual confidence or claim that a detected zone is market-valid.
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

## Confidence-stratified evidence

The evaluator and batch report separate drawable predictions by the model-reported
qualitative confidence category (high and medium). For each category they
report candidate count, drawable count, matched count, false positives,
suppressed/invalid count, the number of fixtures containing candidates,
drawable predictions and matches, precision, mean matched IoU and mean matched
center error. The fixture counts disclose when pooled zone metrics come from
only a small number of screenshots; they are counts of distinct reviewed
images, not a claim that market observations are statistically independent.
These are descriptive results on the manually reviewed fixture set; they do
not turn model confidence into a calibrated probability.

Per-confidence recall is intentionally omitted: a missed manual reference has
no model confidence label and cannot honestly be assigned to a confidence
bucket. The report also counts low-confidence candidates, but marks their
geometry as unscored. The v6 sanitizer intentionally zeroes low-confidence
coordinates before rendering and benchmark intake, so the benchmark cannot
measure whether those hidden rectangles would have matched. Low-confidence
candidate counts must not be interpreted as correct or incorrect detections.

Legacy predictions with missing or unsupported confidence are grouped under
not_reported_or_invalid. They remain visible in the candidate count, but they
are not considered drawable: the production renderer requires an explicit
high- or medium-confidence value. Such candidates increment
suppressed_or_invalid_count, cannot contribute a true or false positive, and
leave any otherwise matching manual reference as a false negative. The manual
reference geometry path opts out of the confidence gate because reference
annotations are not model overlays. For current dns-chart-v9 manifests, the
schema validator requires an explicit valid confidence category and policy,
so this group should be empty.

Paired comparison validates the per-fixture confidence counts against the
matched-zone evidence and overall drawable/ignored counts, then reports
baseline-to-candidate deltas for each confidence group. This is still a
descriptive benchmark, not statistical calibration; use enough independently
reviewed screenshots across symbols and timeframes before drawing conclusions.

## Batch manifest runner

The batch runner verifies that each fixture still points to the exact annotated screenshot before scoring it. It reads at most the configured byte limit plus one byte, checks the SHA-256, actual image format, decoded image limit, recorded dimensions and the annotation-intake metadata/container policy; rejects screenshots that would change under metadata sanitization (including descriptive EXIF/XMP/IPTC fields, non-allowlisted PNG chunks, unrecognized JPEG APP/COM segments, non-structural WEBP chunks or bytes after the image container), so fixtures must use the exact canonical image emitted by `annotation_intake`; rejects duplicate fixture IDs, duplicate screenshot SHA-256 hashes (so identical image bytes cannot silently overweight the sample), and paths that escape the manifest directory; then reports per-fixture results and micro-aggregated precision, recall, F1, IoU, center displacement and optional price-boundary errors. The default byte limit is 5 MiB, matching the Telegram bot's default upload cap; the benchmark hard maximum is 32 MiB. Use `--max-image-bytes` when the bot's `MAX_CHART_IMAGE_BYTES` setting has been overridden, provided it does not exceed that hard maximum.

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
      "origin": "ai_inferred",
      "confidence": "high",
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
    "prompt_version": "dns-chart-v9",
    "prompt_sha256": "replace-with-64-character-prompt-sha256",
    "generated_at": "2026-11-05T20:24:00Z",
    "system_fingerprint": null,
    "settings": {
      "temperature": 0.1,
      "image_detail": "high",
      "response_format": "json_schema:chart_zone_map",
      "conversation_context": "caption_only",
      "plot_bounds_policy": "required_for_overlay",
      "confidence_policy": "high_medium_only",
      "zone_origin_policy": "user_marked_only_when_visibly_marked",
      "zone_evidence_policy": "compact_base_plus_directional_displacement_v1"
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
- normalizes linear anchor prices before least-squares fitting and evaluates the fitted mapping around the anchor centroid instead of extrapolating to y=0; an unrepresentable legacy intercept is retained as null without rejecting finite prices inside the calibrated region, while an unrepresentable slope or anchor residual is rejected; logarithmic calibration is tested across the finite floating-point price range;
- requires centered calibration references (`reference_y` and `reference_value`) to be supplied together; a partial pair is rejected instead of silently falling back to a potentially stale legacy intercept;
- validates the calibration object again at lookup (supported scale, finite descending slope, ordered finite y-range and complete in-range centered references); coordinates within 1e-9 of an anchor boundary are clamped to that boundary to absorb binary-rounding drift without extrapolating;
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

The command accepts static JPEG, PNG or WEBP images only. It applies the same decoded-image validation used by the Telegram upload path and reads at most the configured byte limit plus one byte. Before storage it normalizes EXIF orientation and removes detected descriptive metadata. For JPEG, it inspects Pillow's raw APP/COM segment list and parses the marker stream through entropy-coded scans, requiring a terminal EOI marker exactly at end-of-file. Unknown application segments, comments, malformed or truncated marker streams, and any bytes after EOI trigger canonical re-encoding; a standard JFIF header without an embedded thumbnail, ICC profile segments and the Adobe color-transform segment are retained. The marker parser skips length-delimited APP payloads, byte-stuffed FF 00 data and restart markers so marker-like bytes inside image data are not mistaken for the end of the file. PNG intake also parses the raw chunk stream: tIME, unknown chunks, malformed or incomplete chunk layouts, and any bytes after IEND trigger a lossless re-encode instead of being trusted merely because Pillow ignores them. Required image structure and common color/transparency chunks are allowlisted; unknown PNG extensions are not assumed safe. WEBP cleanup is lossless; metadata-bearing JPEGs are re-encoded at quality 100 with 4:4:4 chroma and remain lossy. Inputs needing neither orientation normalization nor metadata removal remain byte-for-byte unchanged. The raw upload is not retained by intake. SHA-256 and dimensions describe the exact canonical stored bytes and normalized image; if the raw original is needed for audit, retain it separately in access-controlled storage. The command refuses path-like fixture IDs, overwrites, output image-directory symlinks that escape the inbox, oversized or malformed files, animated images, and unsupported market-context fields. Context values are normalized using the benchmark's grouping conventions. The draft does not record the source machine's path or any user identity.

The output contains `images/<fixture-id>.<ext>` and `<fixture-id>.draft.json`. The draft deliberately has `benchmark_ready: false`, empty reference zones, null plot bounds, pending annotation metadata, and no AI prediction/provenance. **It is not a benchmark JSONL record and must not be passed to the benchmark runner.** Two different annotators must independently mark the same screenshot, and each annotation must be separately reviewed by a different reviewer. A third, distinct adjudicator must inspect the pairwise agreement report and explicitly select one annotation with a rationale. Capture the actual AI prediction and provider-returned prediction metadata for the exact same screenshot separately. Only after all of those steps should a benchmark JSONL fixture be assembled using the schema above.

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

The editor resolves the screenshot path from the draft's relative `image_file` by default. This field must be a non-empty portable POSIX relative path: backslashes, empty path components, `.` and `..` components, absolute paths, and symlinks resolving outside the draft directory are rejected before the image is opened. Use `--image path/to/screenshot.png` only when the screenshot is stored elsewhere. Before opening the editor it uses the same chart-image preparation routine as inference, enforces the 5 MiB editor limit, and verifies the exact SHA-256 and pixel dimensions recorded in the draft. It rejects screenshots whose EXIF orientation would change the encoded bytes; re-run `annotation_intake` and use its canonical image rather than annotating a frame that the benchmark will later reject. A mismatch is rejected rather than allowing annotations against a different image.

In annotation mode, select `plot` and drag the main price-plot rectangle first. Then select `demand` or `supply` and draw each visible zone. Coordinates are normalized to the full original screenshot, matching the benchmark schema. Each drag endpoint is quantized with the same Python `round()` rule used by the overlay/evaluator to the nearest source-image pixel boundary before it is saved; a later zoom or reload therefore reproduces the same scored source-pixel edges. The preview rectangles are drawn from those quantized source-pixel boundaries and then scaled into the current view, rather than drawing unquantized normalized coordinates. Manual price-axis anchor rows are likewise quantized to a source-image row before they are saved. This is pixel alignment, not sub-pixel OCR precision. Zones must fit completely inside the marked plot, and no more than eight reference zones are accepted. A chart with no visible supply/demand zone may be saved with an empty zone list, provided the plot boundary is marked. Use an anonymized annotator ID such as `ann-001`; do not enter a person's name or contact information.

For optional price-boundary scoring, select `axis_anchor`, choose `linear` or `log`, click inside both the horizontal and vertical bounds of the marked main price plot at the vertical level of a legible price-axis tick (do not click in the right-side price-label margin), and enter that exact visible price label. Clicks in unused canvas space beyond the screenshot are rejected even when the marked plot reaches an image edge; they are not clamped into a valid calibration point. Add 3..20 anchors at distinct y positions, from top to bottom with descending prices. This is manual entry, not OCR: do not estimate unreadable labels or use values from another screenshot. Anchor y-coordinates are normalized against the full screenshot and must lie inside the marked plot's vertical span. On save, the editor fits the selected scale using the existing 2% default maximum anchor residual; inconsistent anchors are rejected. The saved `price_axis_plot_bounds` is copied from the manually marked plot rectangle. If price labels are unreadable, leave the anchor tool unused and save a geometry-only annotation.

The annotation pass writes a new `.annotated.json` file with `status: pending_review`, a timezone-aware timestamp and pending review metadata. Review mode is read-only: a separate reviewer enters an ID such as `rev-002` and explicitly approves or rejects the rectangles. The review pass writes a new `.reviewed.json` file. Existing files are never overwritten. Both approved and rejected drafts remain `benchmark_ready: false`; approval alone does not create a benchmark fixture.

The editor records the manually marked main plot rectangle, supply/demand geometry, and optional manually read price-axis anchors plus the chosen linear/log scale. Before annotation can be saved or reviewed, each reference zone is checked at the original screenshot dimensions using the evaluator's pixel rounding and reference-zone rules: it must remain at least one pixel wide and high and cannot exceed 45% of screenshot height. This prevents approval of boxes that the benchmark would later reject as unscorable. When anchors are present, the saved `price_axis_plot_bounds` must match the reviewed main plot and the 3..20 anchors must pass the calibration fit. The reviewer sees the anchor markers and values in read-only mode and revalidates them before approval. Repeat the full annotation and review process with a second annotator, then use the adjudication command to record a separate human selection and rationale. Only then capture the actual AI prediction and provider-returned metadata for the exact same screenshot and assemble the benchmark JSONL record. The benchmark runner remains the final validator. The editor's geometry/state-transition tests run headlessly; an interactive Tk window still needs a manual desktop smoke test on the target operating system. The CLI reports success only when the current editor session completes its own save operation; an output file appearing concurrently from another process does not count as a successful save. Closing the window without saving exits with status 1; successfully saving an annotation or review in the current session exits with status 0.


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
python -m stockmind_telegram_mvp.app.annotation_fixture adjudicate \
  stockmind_telegram_mvp/benchmark/inbox/btc-1h-001.reviewed.json \
  stockmind_telegram_mvp/benchmark/inbox-b/btc-1h-001.reviewed.json \
  --adjudicator-id adj-003 \
  --adjudicated-at 2026-11-10T12:30:00Z \
  --decision accept_a \
  --rationale "Annotation A follows the agreed boundary rule after reviewing the measured disagreements."

python -m stockmind_telegram_mvp.app.annotation_fixture assemble \
  stockmind_telegram_mvp/benchmark/inbox/btc-1h-001.reviewed.json \
  stockmind_telegram_mvp/benchmark/inbox/btc-1h-001.prediction.json \
  --annotation-b stockmind_telegram_mvp/benchmark/inbox-b/btc-1h-001.reviewed.json \
  --adjudication stockmind_telegram_mvp/benchmark/inbox/btc-1h-001.adjudication.json

python -m stockmind_telegram_mvp.app.zone_benchmark \
  stockmind_telegram_mvp/benchmark/inbox/btc-1h-001.fixture.jsonl
~~~

The capture contains only the screenshot hash/dimensions, fixture ID and
sanitized model result/provenance; it does not copy the screenshot or store a
Telegram user/chat ID. The assembler requires two independently approved annotations plus
a valid adjudication sidecar, validates the manual geometry and optional
price-axis calibration, checks fixture ID/hash/dimensions, rejects altered or
non-sanitized model geometry, and requires real provider-returned model/time
metadata. It writes the manifest beside the adjudicated annotation so the
relative image_file remains valid. Existing outputs are never overwritten.

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

For an EXIF-oriented source, `prepare_chart_image` fully decodes and integrity-
checks the untrusted input before transposing and re-encoding it. A successful
Pillow encode preserves the source format and pixel count, so the normalized
output is not reopened for a second full decode/verify cycle. Callers that need
an explicit output postcondition can validate the returned bytes separately.
The orientation regression tests do so.

The Telegram handler has already run `prepare_chart_image` on the exact bytes
sent to the vision provider. To avoid decoding a large screenshot a second time,
that internal handler uses `_render_overlay_prepared` with those same canonical
bytes. This underscore-prefixed helper is an internal fast path; direct callers
and offline tools should use `render_overlay`.


## Multi-resolution pixel geometry contract

Zone and plot coordinates are quantized to half-open source-image pixel bounds
`[left, right) × [top, bottom)` by the shared overlay geometry helpers. The
renderer adapts those bounds to Pillow's inclusive rectangle endpoints by
subtracting one from the right and bottom edges. The evaluator scores the same
quantized bounds, rather than the model's unrounded normalized coordinates.

Regression coverage exercises even and odd screenshot dimensions (640×360,
641×359, 1280×720 and 1920×1080), including zones touching the outer image
edges. It checks the actual Pillow raster footprint against the half-open box
and checks that sub-pixel coordinate differences which quantize to the same
source pixels receive identical geometry scores. If a caller supplies
dimensions whose coordinate scaling cannot be represented by the numeric
runtime, the geometry helper rejects the box instead of leaking an overflow
exception. These synthetic tests verify coordinate semantics; they are not
evidence of market-zone detection accuracy.

## Reviewed screenshot immutability

Prediction capture and benchmark assembly reload the canonical screenshot referenced
by the reviewed annotation and verify its SHA-256 and dimensions. Replacing or
editing that image after annotation/review invalidates the handoff, even when the
saved annotation and prediction-capture JSON still agree with each other. Re-intake
the exact intended screenshot and repeat annotation and independent review; do not
repair a mismatch by editing a stored hash. A regression test changes the source
image after a valid capture and verifies that both capture and assembly fail closed.

## Mandatory independent annotation adjudication

A scoreable JSONL record must include adjudication_metadata in addition
to the approved manual annotation metadata. The field contains two distinct
reviewed annotation provenance records, a hash-bound pairwise agreement report,
and an explicit decision (accept_a or accept_b) made by a separate adjudicator.
The adjudicator must be distinct from both annotators and both reviewers,
provide a timezone-aware timestamp after both reviews, and supply a
20..1000-character rationale.

Use python -m stockmind_telegram_mvp.app.annotation_fixture adjudicate to
create the sidecar and assemble --annotation-b ... --adjudication ... to
create the manifest. Assembly recomputes agreement from the current reviewed
files, checks their exact file hashes, validates the report hash and selection,
then runs the normal manifest preflight. The benchmark runner rejects records
without valid adjudication metadata, including hand-authored JSONL records. The validated adjudication metadata is retained in each fixture result, and paired model comparisons require the same adjudication evidence on both sides.

The adjudicator must inspect the measured disagreements; the software does not
select whichever annotation has the higher IoU. If neither annotation is
acceptable, do not admit that screenshot as ground truth. Pseudonymous IDs and
rationale are an audit trail, not cryptographic identity proof. Keep both
reviewed files and the adjudication sidecar with the benchmark source manifest.

## Directional plot-boundary spill diagnostics

Plot-boundary reports separately measure predicted plot and drawable-zone area that extends beyond the manually marked main price plot on all four sides. The right price-scale margin and bottom indicator panel retain their existing metrics; left-toolbar spill and top-control spill are reported as `left_toolbar_overlap_pct` / `top_controls_overlap_pct` for the plot and `left_toolbar_zone_area_pct` / `top_controls_zone_area_pct` for drawn zones. These are geometric overlap proxies based on the manually reviewed reference rectangle, not semantic detection of toolbar or indicator content. The directional values are included in aggregate reports and paired baseline-versus-candidate deltas. Each side-strip metric excludes the corners: the residual between the overall outside-reference-area percentage and the sum of the four side-strip percentages is corner spill. A regression test covers all four corners and checks this decomposition for both the predicted plot and drawn zones. Zone-area percentages use the sum of predicted box areas as their denominator, so overlapping predicted boxes contribute separately rather than representing unique screenshot coverage; do not interpret them as a pixel-union measurement.

## Overlay output fidelity and upload budget

When the source screenshot is PNG, the overlay renderer now prefers a lossless
PNG result so untouched candle bodies, wicks, grid lines and chart labels are
not recompressed as JPEG. Compatible RGB ICC profiles are retained on that
output. To stay below Telegram's photo-upload size ceiling, PNG output is
limited to 9,000,000 bytes; a larger encoded result falls back to the existing
optimized JPEG output at quality 88. JPEG and WEBP source screenshots continue
to produce JPEG overlays. Telegram upload filenames follow the actual encoded
container, so PNG bytes are not mislabeled with a .jpg extension.

EXIF normalization now checks an embedded ICC profile against the final pixel
mode after any Pillow mode conversion. RGB/RGBA/palette, grayscale and CMYK
profiles are retained only for matching channel data; malformed, truncated or
mode-incompatible profiles are omitted rather than attached to different pixel
channels. The overlay renderer similarly attaches only a structurally valid
RGB profile to its RGB output. This is a conservative ICC-header compatibility
check, not a color-managed conversion: CMYK and grayscale source pixels are
still converted by Pillow's normal mode conversion when the overlay is rendered.
No colorimetric accuracy improvement is claimed.

The regression tests verify pixel equality outside a drawn zone, a changed
pixel inside the zone, PNG-to-JPEG fallback when the PNG budget is exceeded,
and unchanged JPEG output for JPEG inputs. These tests validate image encoding
and pixel preservation only; they do not measure whether the model selected
the correct supply/demand zones on real market screenshots.


## Bounded JSONL manifest records

The offline benchmark reader limits each UTF-8 JSONL record to 1 MiB by
default, including its newline, and limits the complete manifest to 32 MiB.
Both limits are enforced while reading in binary mode with a one-byte-over-limit
sentinel before UTF-8 decoding or JSON parsing. An oversized record or manifest
is rejected without reading the remainder of the file. Screenshot bytes retain
their separate 5 MiB default limit. Blank lines count toward both manifest
limits.

The CLI exposes `--max-manifest-bytes` and `--max-manifest-line-bytes` for
controlled local datasets. The total manifest limit has a hard maximum of
128 MiB; the per-record limit has a hard maximum of 16 MiB. These caps keep
configuration errors from silently disabling resource bounds. The limits are
independent: increasing one does not increase the other. Typical model metadata
and zone geometry should fit comfortably below the defaults. The byte ceilings
bound input size, but the evaluator still retains parsed fixture metrics in
memory; use separate manifests for very large benchmark collections.


## Paired inference usage and latency

The paired comparison report includes an `inference_telemetry` section. Its
`baseline` and `candidate` summaries report the fixture coverage, observed
totals, means, minima and maxima for provider-reported prompt, completion,
total, image, cached-prompt and reasoning token counts. Request latency is
summarized separately in milliseconds. Optional token subtotals have their
own coverage counts because providers may omit them even when total usage is
available.

`paired_deltas` uses candidate minus baseline for the same fixture ID and
only includes a delta when both reports contain that metric for that fixture.
It reports the paired fixture count, missing-pair count, total delta, mean
delta, minimum and maximum delta. Negative token or latency deltas indicate
lower observed consumption or shorter model-call time for the candidate;
they do not by themselves establish a lower currency cost or a faster
end-to-end Telegram response.

Missing telemetry is never replaced with zero. If one side has no provider
usage or request timing for a fixture, that fixture is excluded from that
metric's paired delta and the coverage fields disclose the omission. The
per-fixture comparison rows retain the raw baseline and candidate telemetry
for auditability.

Each metric summary now also reports its median and nearest-rank p95. Paired
candidate-minus-baseline deltas include the median and nearest-rank p95 of the
per-fixture differences. The nearest-rank method selects the sorted value at
rank ceil(0.95 * n); for small samples p95 can equal the maximum, so always
read these statistics alongside the reported fixture coverage rather than as
a stable tail-latency guarantee.

Token totals are provider-reported counts, not estimated image-token counts.
No currency cost is inferred: pricing, cached-input treatment and model
rates can vary by model and effective date. Request latency covers the
vision completion round trip only; it excludes image download/preparation,
queueing before the request and overlay rendering/upload. The synthetic
comparison tests validate arithmetic and missing-data handling, not actual
provider savings or real-market DNS accuracy.


## Optional date-bound currency cost estimates

The paired comparison CLI can estimate USD cost when you provide an explicit
rate card. No model prices are built into StockMind and the tool does not fetch
or guess current provider rates. The rate card must use exact provider-returned
model IDs and a date window covering the fixture's provider-returned
`generated_at` date.

Example rate-card shape (the values below are placeholders, not actual prices):

~~~json
{
  "schema": "stockmind.usage-pricing.v1",
  "version": "replace-with-your-rate-card-version",
  "currency": "USD",
  "effective_from": "2026-11-01",
  "effective_until": null,
  "source": "Record the official pricing source and retrieval date here",
  "models": {
    "exact-provider-model-id": {
      "input_usd_per_million_tokens": "0",
      "cached_input_usd_per_million_tokens": "0",
      "output_usd_per_million_tokens": "0"
    }
  }
}
~~~

Run the comparison with:

~~~bash
python -m stockmind_telegram_mvp.app.zone_benchmark_compare \
  baseline-report.json candidate-report.json \
  --pricing-file pricing-card.json
~~~

Rates must be decimal strings (not JSON floats), non-negative, and expressed
per one million tokens. The card is versioned and date-bounded; a fixture is
left unpriced when its actual model is absent, its provider timestamp falls
outside the card's effective dates, usage is missing, or a discounted cached
input rate is supplied without the cached-token count needed to calculate it.
Missing values are never treated as zero. The cost-comparison helper also
rejects duplicate, missing or extra fixture IDs before aggregation; it never
silently overwrites a duplicate fixture or reports a partial baseline/candidate
pair as if it were complete.

Provider-reported `image_tokens` are a subset of prompt/input tokens and are
not charged a second time. `reasoning_tokens` are a subset of completion/output
tokens and are not charged a second time. If no cached-input rate is supplied,
all prompt tokens use the standard input rate. Cost outputs are rounded to 18 decimal places in USD. The
estimator uses an expanded local Decimal context for token-rate multiplication,
aggregation and paired differences, so large provider token counts do not lose
precision or fail during fixed-point formatting. This preserves the full
per-token precision allowed by rate cards (12 fractional places per million
tokens, divided by 1,000,000) and prevents very small per-fixture costs from
rounding to zero before aggregation. Estimates are based only on the supplied
card; they exclude taxes, discounts, credits, retries not represented in the
fixture, and any non-token charges. Keep the original rate-card source with
exported reports. These estimates are not invoices and are not evidence of
lower production spend.


## Request-stage latency logs

The Telegram bot emits privacy-minimized timing records through its application
logger. Each Telegram update receives a random UUID4-derived, opaque
32-character hexadecimal request ID. The ID is scoped to the update task and
is shared by its upload-download, research and handler-total timing records.
It is not derived from a Telegram user/chat ID and is not reused as a persistent
user identifier. A successful or failed research pipeline records a
`StockMind request timing` event with `request_id`, `mode`, `outcome`,
total pipeline milliseconds, and whichever stage measurements were reached:

- `image_prepare`: screenshot orientation normalization and metadata cleanup.
- `analysis_execution`: the chart-vision call for screenshots, or the complete
  tool-enabled research loop for text requests.
- `overlay_render`: local overlay generation and validation.
- `telegram_text_delivery`: final analysis/error text delivery.
- `telegram_overlay_delivery`: optional overlay photo/document delivery.

Photo and image-document handlers log Telegram file retrieval/download
duration when a download is attempted, plus a handler total on every handler
exit. This includes unsupported-file, oversized-upload, quota/preflight and
download-error paths. The handler total starts at handler entry and includes
preflight checks, Telegram file retrieval, bounded download, lock and
chart-capacity waits, analysis, and final delivery. It does not include
Telegram update ingress delay before the handler begins.

These measurements are process logs, not persisted per-user analytics. They
intentionally omit user IDs, chat IDs, prompts, symbols, image hashes, and
image bytes. The random request ID only joins events belonging to one Telegram
update and should still be treated as operational metadata under normal log
retention controls. Aggregate them offline before drawing latency conclusions;
a single request does not establish p50/p95 performance. The model-provider
request latency already captured by the AI client remains a separate, narrower
measurement.


## Offline pipeline latency report

The `pipeline_telemetry` module aggregates the existing application log events
without requiring a database or changing bot runtime behavior. From the
repository root, run:

```bash
python -m stockmind_telegram_mvp.app.pipeline_telemetry /path/to/stockmind.log > latency-report.json
```

It also accepts `-` as the input path to read standard input. The report groups
request totals and stage timings by request `mode` and `outcome`, and reports
sample count, median (p50), nearest-rank p95, minimum and maximum. Each stage
includes a `missing_count`; stages not reached on an error path are not treated
as zero. Photo/document download and handler-total events are summarized as
separate distributions.

The report now joins upload download, request-stage and handler-total events
by the opaque request ID, so interleaved concurrent uploads are not paired by
line proximity. It retains partial rows when an upload is rejected before a
research event, counts legacy records without IDs separately, and excludes
IDs with duplicate or conflicting events from the correlated rows. These
correlations identify the same update; they do not make stage durations
additive. The handler total begins before Telegram file retrieval, while the
research total begins after the per-user lock and chart-capacity admission.
Likewise, stage percentiles are independent distributions, not additive
components of a synthetic end-to-end percentile. `analysis_execution` is a complete tool-enabled loop for text
requests and a vision call for chart requests; it is not a provider-only timing
for every mode. The provider request timer remains a narrower separate metric.

The parser exports only recognized timing fields and never copies raw log
lines or unrelated logger context into the JSON report. Keep source log files
under the same access and retention controls as other application logs. A
report with zero parsed events or non-zero `rejected_timing_lines` should be
investigated before using its percentiles. Small samples, especially p95, are
descriptive only and are not latency guarantees.


## Telegram upload outcome telemetry

Photo and document handler completion events include a bounded `outcome` field alongside the opaque request ID and total handler duration. Outcome values are intentionally low-cardinality: completed, delivery_failed, timeout, incomplete_response, invalid_image, research_failed, rejected, download_failed, request_busy, capacity_timeout, storage_unavailable, handler_error, or unknown. They contain no user, symbol, prompt, or provider identifiers.

The offline telemetry report counts these outcomes and attaches the final handler outcome to the correlated request row. Download timing events remain outcome-free because they are emitted before analysis and delivery finish. Legacy handler logs without an outcome remain readable and are counted separately.

A request is counted in `complete_upload_request_count` only when it has a chart-mode research timing event plus both download and total-handler timing measurements. Text-mode request events cannot be misclassified as completed chart uploads. Conflicting or duplicate events remain excluded from per-request correlation.

## Numerically stable metric aggregation

Zone-evaluation means scale samples by the largest absolute value before summing,
then restore the scale after averaging. This keeps the intermediate sum bounded
and avoids rounding every uniformly subnormal sample to zero before the mean is
computed. Empty metric collections remain undefined (null in reports); the
benchmark does not manufacture zero-valued observations for missing data.
