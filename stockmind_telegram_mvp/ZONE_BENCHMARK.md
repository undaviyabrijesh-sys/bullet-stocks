# StockMind chart-zone benchmark

This benchmark compares AI overlay rectangles with zones manually marked on the exact same chart screenshot. It is an offline evaluation aid; it does not alter bot responses or make a live model call.

## Annotation rules

- Keep the original screenshot unchanged and record its SHA-256, width and height.
- Mark each visible reference zone as supply or demand.
- Express each rectangle relative to the full screenshot: x_start, x_end, y_top and y_bottom are normalized to 0..1.
- Use the same full-image coordinate frame for the AI prediction and manual annotation. Do not crop one side or score against a different screenshot.
- Keep no more than eight reference zones, matching the current overlay cap.
- Do not infer unreadable price values. This benchmark measures pixel geometry, not the correctness of a price-axis conversion.

A fixture record should contain a stable fixture_id, image_sha256, image_width, image_height, reference_zones and predicted_zones. Each reference zone needs zone_type and the four normalized coordinates. Predicted zones should be the structured AI zones, including draw_overlay.

## Scoring

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

## Important limitations

There are currently no manually annotated real chart screenshots in this repository. Unit tests use synthetic rectangles only to verify the scoring logic; they are not evidence of model accuracy. Do not publish a zone-detection accuracy or claim improvement until a real, manually reviewed fixture set has been scored.

IoU and normalized center displacement measure screenshot geometry only. They do not verify that a zone is structurally meaningful, that its price bounds agree with the visible scale, or that it is fresh/unmitigated. Price-axis accuracy needs a separate annotation/calibration field and should be evaluated independently.
