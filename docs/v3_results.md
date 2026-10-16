# v3.0 time-patch transformer results

Experiment version: **v3.0**. Code folder:
`experiments/v3_0_time_patch_transformer/`.

Completed run: `artifacts/runs/20261012T210954Z_v3.0_743b9c95f323/`.

This experiment does not replace v2.0. It uses the frozen v2.0 speaker folds and
the same 64-band log-mel features. The new model is different: each 16-frame
patch contains all 64 mel bands, and a 32-wide, one-layer transformer reads the
25-patch sequence. It has 47,105 parameters.

Two inputs were declared before test scores were viewed:

- `raw_logmel`: original log-mel windows, 64 bands by 400 frames.
- `time_centered_logmel`: the same windows after subtracting each recording's
  average value in every mel band. The average spectrum is removed; changes
  over time remain.

The static baseline averages the 64 bands over time and uses logistic regression.
Threshold is fixed at 0.5. No person overlap was found in the saved splits.

## Main result: 46 speakers

Percentages.

| Method | F1 | Recall | Specificity | Precision | Accuracy | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Static-spectrum baseline | 98.04 | 100.00 | 95.24 | 96.15 | 97.83 | 0.9886 |
| New transformer, raw log-mel | 88.89 | 80.00 | 100.00 | 100.00 | 89.13 | 0.9371 |
| New transformer, average spectrum removed | 96.00 | 96.00 | 95.24 | 96.00 | 95.65 | 0.9943 |

Confusion counts are 25/0/20/1 for the baseline, 20/5/21/0 for the raw
transformer, and 24/1/20/1 for the centered transformer, in TP/FN/TN/FP order.

The raw transformer is unstable across folds: its F1 scores are 100, 100, 100,
100, and 0. One fold missed all five held-out patients. Its pooled score is the
number to quote, not any individual fold.

Removing the three flagged controls from evaluation does not materially change
either transformer: raw F1 remains 88.89, and centered F1 remains 96.00.

## Native-16-kHz retraining: 39 speakers

| Method | F1 | Recall | Specificity | Precision | Accuracy | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Static-spectrum baseline | 84.85 | 77.78 | 95.24 | 93.33 | 87.18 | 0.9577 |
| New transformer, raw log-mel | 91.43 | 88.89 | 95.24 | 94.12 | 92.31 | 0.9762 |
| New transformer, average spectrum removed | 97.14 | 94.44 | 100.00 | 100.00 | 97.44 | 1.0000 |

## What this experiment shows

On the main 46-person comparison, the new transformer does **not** beat the
simple static-spectrum baseline. Adding a different transformer is therefore
not automatically an improvement.

Removing the average spectrum does not make the new model fail. On the smaller
native-16-kHz subset, the time-centered transformer scores higher than the
static baseline. Changes over time carry class information beyond the average
spectrum in this internal split.

That still does not establish Parkinson's-specific detection. Recording dates
remain separated by class, and time-varying microphone or room characteristics
could survive average-spectrum removal. These are internal results on 46 and 39
people, not an independent external test.
