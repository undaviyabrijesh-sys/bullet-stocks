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
5. Use distinct pseudonymous annotator IDs. Reviewer IDs must not overlap the
   other annotation's annotator ID.

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
- one-to-one zone matches, separately for supply and demand;
- unmatched zones from each annotator;
- each annotator's matched-zone coverage;
- symmetric agreement F1, mean matched IoU and normalized center displacement;
- screenshot hash, market context, guideline version and pseudonymous provenance.

Matching uses the same screenshot pixel quantization and maximum-cardinality,
then maximum-IoU assignment rule as the AI zone evaluator. A supply zone never
matches a demand zone.

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
