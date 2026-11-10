# Independent manual annotation agreement

StockMind can compare two independently marked versions of the same chart
before either is treated as benchmark ground truth. This is a quality-control
step for human labels; it is not an AI accuracy score and it does not choose an
adjudicated reference automatically.

## Prepare two annotations

1. Run annotation intake twice in separate output folders using the exact
   same screenshot and the same fixture ID.
2. Have two different annotators independently mark the plot boundary and
   supply/demand zones using the same guideline version.
3. Have each annotation reviewed and approved through the existing review step.
4. Keep the source screenshot unchanged. The comparison verifies its SHA-256,
   dimensions, symbol/timeframe context and guideline version.
5. Use distinct pseudonymous annotator IDs and distinct reviewer IDs. Each
   reviewer must differ from both annotators, and the two reviewers must also
   be different from each other.

Do not copy one annotator's boxes into the second draft. The comparison is
meaningful only when the drawings were made independently.

## Run the comparison

From the repository root:

    python -m stockmind_telegram_mvp.app.annotation_agreement \
      path/to/annotator-a.reviewed.json \
      path/to/annotator-b.reviewed.json \
      --iou-threshold 0.5

The JSON report includes:

- plot-boundary IoU and edge displacement in screenshot pixels;
- manual price-axis calibration agreement over the overlapping, manually anchored y range;
- one-to-one zone matches, separately for supply and demand;
- unmatched zones from each annotator;
- each annotator's matched-zone coverage;
- symmetric agreement F1, mean matched IoU and normalized center displacement;
- screenshot hash, market context, guideline version and pseudonymous provenance.

Matching uses the same screenshot pixel quantization and maximum-cardinality,
then maximum-IoU assignment rule as the AI zone evaluator. A supply zone never
matches a demand zone.

## Manual price-axis agreement

When both annotators provide at least three validated manual price-axis anchors,
the report fits each annotator's declared linear or logarithmic calibration and
compares the two fitted prices at 11 evenly spaced y positions within their
shared anchor range. It reports mean and maximum absolute and relative price
disagreement, each fit's maximum anchor residual, and whether the annotators
selected the same scale.

No values are extrapolated beyond either annotator's manually anchored range.
If neither annotator supplied anchors, the status is `not_provided`; if only
one did, it is `missing_on_one_side`; and if their anchor ranges do not
overlap, it is `no_overlap`. Only a shared range produces numeric comparison
metrics. A different selected scale is reported explicitly rather than
silently normalized away.

The report also includes the shared y-range as a fraction of annotator A's
calibrated range, annotator B's calibrated range, and the union of both ranges.
These coverage values matter because two fits can agree closely over a narrow
overlap while covering only a small part of the chart. A non-overlapping range,
or an overlap of at most 1e-9 in normalized screenshot coordinates, is reported
as `no_overlap` with zero coverage fractions; no price samples are generated.
The comparison uses 2..101 evenly spaced samples, defaults to 11, and never
extrapolates beyond either annotator's manually anchored range.

This measures agreement between two fitted manual calibrations. It does not
verify the visible labels against an external exchange price or prove that
either annotator read the screenshot correctly.

## Interpretation and limitations

- Agreement F1 is a symmetric summary of the two sets; annotator coverage is
  the fraction of one annotator's zones that found a qualifying counterpart.
- Mean IoU is calculated only over matched pairs. Always read it alongside
  unmatched counts and the configured IoU threshold.
- Empty or unmatched sets are not converted into a perfect agreement score.
- The report does not authenticate pseudonymous identities or prove that the
  two humans worked independently; project owners must manage that process.
- High agreement does not prove that either annotation is objectively correct.
  Resolve disagreements through a documented adjudication step before using
  a reference as benchmark ground truth.
- Synthetic unit-test images validate the software path only. They are not
  chart fixtures and provide no evidence of market-chart detection accuracy.

## Human adjudication before benchmark admission

Pairwise agreement is a review aid, not an automatic winner-selection rule. A
screenshot is not admitted as benchmark ground truth until a third, distinct
adjudicator has inspected the two approved annotations and explicitly selected
one of them. The supported decisions are **accept_a** and **accept_b**.
If neither annotation is acceptable, do not assemble the fixture; repeat the
manual annotation/review process or exclude the screenshot. Merging annotations
is intentionally not automated.

Create a hash-bound adjudication sidecar from the two reviewed JSON files:

~~~bash
python -m stockmind_telegram_mvp.app.annotation_fixture adjudicate \
  path/to/annotator-a.reviewed.json \
  path/to/annotator-b.reviewed.json \
  --adjudicator-id adj-003 \
  --adjudicated-at 2026-11-10T12:30:00Z \
  --decision accept_a \
  --rationale "Annotation A follows the agreed boundary rule after reviewing the measured disagreements."
~~~

The adjudicator ID must be distinct from both annotators and both reviewers.
The timestamp must include a timezone and cannot precede either review. The
rationale must contain 20..1000 printable characters. The sidecar embeds the
pairwise agreement report and its canonical SHA-256, plus SHA-256 hashes of the
exact two reviewed JSON files. Editing either reviewed file after adjudication
invalidates the sidecar.

Capture the actual model response for the same screenshot, then assemble with
both annotations and the adjudication sidecar:

~~~bash
python -m stockmind_telegram_mvp.app.annotation_fixture assemble \
  path/to/annotator-a.reviewed.json \
  path/to/btc-1h-001.prediction.json \
  --annotation-b path/to/annotator-b.reviewed.json \
  --adjudication path/to/btc-1h-001.adjudication.json
~~~

The prediction capture may be bound to either reviewed annotation because both
must identify the exact same screenshot hash, dimensions and fixture ID. The
selected annotation supplies the final reference geometry and price-axis
anchors; the JSONL manifest is written beside that selected annotation so its
relative screenshot path remains valid.

The assembler recomputes pairwise agreement from the two current reviewed
files, verifies both raw-file hashes, verifies the embedded report hash and
contents, checks the adjudicator's role separation and timestamp, and runs the
normal benchmark preflight before writing. The benchmark runner also requires
adjudication_metadata in every JSONL record and validates its schema,
identities, timestamp, selected annotation, fixture context and report hash.
A hand-written record without this evidence is rejected. The full adjudication metadata is retained in the benchmark result, and paired model comparisons require identical adjudication evidence for each screenshot.

This is an auditable pseudonymous attestation, not a digital signature or
identity-verification service. It cannot prove that humans are who their IDs
claim to be or that they worked independently. Preserve the reviewed files and
adjudication sidecar with the manifest. Synthetic test annotations only test
the software contract; they are not real chart ground truth.
