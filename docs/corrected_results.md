# Corrected results for all three members

Experiment version: **v2.0**. See [version definitions](experiment_versions.md).

Completed run: `artifacts/runs/20261012T183328Z_f-all_cead14cc32f1/`.

**Interpretation update:** The [subsequent high-score audit](high_score_audit.md)
reproduced the results but found that they survive removal of within-window
time ordering. Static and low-energy acoustic controls also score highly, and
recording dates separate the classes. The reported metrics remain valid for
this internal experiment; they do not establish disease-specific generalization
or elimination of all acquisition-related confounding.

All three pipelines were rerun. Independent prediction replay passed for all
30 saved outer-fold models: three methods, two training scenarios, five folds.
The checks also verified canonical speaker identities, disjoint inner and outer
splits, training-only scalers, native-rate filtering, and pooled confusion counts.

## Evaluation

- Main cohort: 46 common speakers, 25 PD and 21 elderly controls.
- Vitantonio D has no B1/B2 passage recording and is absent from the common cohort.
- Same five stratified outer speaker folds for all methods; seed 42.
- Fixed classification threshold 0.5, declared before this rerun.
- Each speaker contributes one held-out prediction. Repeated sessions remain together.
- Methods retain different speech tasks, so this is a comparison of complete pipelines.
- These are revised internal OOF results after earlier dataset inspection, not an independent external test.

## Main pooled speaker metrics

All five core metrics below are percentages. AUROC, AP, and MCC in later tables
use the 0 to 1 scale, except that MCC can also be negative.

| Member and method | F1 | Recall | Specificity | Precision | Accuracy |
|---|---:|---:|---:|---:|---:|
| Arannamoy: corrected PatchTST | 97.96 | 96.00 | 100.00 | 100.00 | 97.83 |
| Mohibur: frozen Whisper + logistic regression | 93.88 | 92.00 | 95.24 | 95.83 | 93.48 |
| Osman: acoustic features + logistic regression | 97.96 | 96.00 | 100.00 | 100.00 | 97.83 |

| Member | TP | FN | TN | FP |
|---|---:|---:|---:|---:|
| Arannamoy | 24 | 1 | 21 | 0 |
| Mohibur | 23 | 2 | 20 | 1 |
| Osman | 24 | 1 | 21 | 0 |

| Member | AUROC | Average precision | MCC | Balanced accuracy (%) | NPV (%) |
|---|---:|---:|---:|---:|---:|
| Arannamoy | 1.0000 | 1.0000 | 0.9573 | 98.00 | 95.45 |
| Mohibur | 0.9905 | 0.9929 | 0.8699 | 93.62 | 90.91 |
| Osman | 0.9924 | 0.9945 | 0.9573 | 98.00 | 95.45 |

Average precision is not the same numerical definition as trapezoidal AUPRC.
Arannamoy's AUROC of 1.0 coexists with one missed patient: the probabilities
rank patients above controls, but one patient remains below the fixed 0.5 threshold.

## Main fold mean and sample standard deviation

Percentages; five folds; standard deviation uses ddof=1. These describe fold
variation and are not confidence intervals for performance in a new population.

| Member | F1 | Recall | Specificity | Precision | Accuracy |
|---|---:|---:|---:|---:|---:|
| Arannamoy | 97.78 +/- 4.97 | 96.00 +/- 8.94 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 97.78 +/- 4.97 |
| Mohibur | 93.18 +/- 10.90 | 92.00 +/- 17.89 | 95.00 +/- 11.18 | 96.67 +/- 7.45 | 93.33 +/- 9.94 |
| Osman | 97.78 +/- 4.97 | 96.00 +/- 8.94 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 97.78 +/- 4.97 |

## Main OOF evaluation with flagged controls removed

43 speakers: 25 PD and 18 controls. The three flagged controls present in the
common cohort are excluded from evaluation. This does not retrain the models.

| Member | F1 | Recall | Specificity | Precision | Accuracy |
|---|---:|---:|---:|---:|---:|
| Arannamoy | 97.96 | 96.00 | 100.00 | 100.00 | 97.67 |
| Mohibur | 95.83 | 92.00 | 100.00 | 100.00 | 95.35 |
| Osman | 97.96 | 96.00 | 100.00 | 100.00 | 97.67 |

| Member | TP | FN | TN | FP | AUROC | Average precision | MCC | Balanced accuracy (%) | NPV (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Arannamoy | 24 | 1 | 18 | 0 | 1.0000 | 1.0000 | 0.9537 | 98.00 | 94.74 |
| Mohibur | 23 | 2 | 18 | 0 | 0.9933 | 0.9957 | 0.9099 | 96.00 | 90.00 |
| Osman | 24 | 1 | 18 | 0 | 0.9933 | 0.9957 | 0.9537 | 98.00 | 94.74 |

## Native-16-kHz-only retraining and evaluation

39 speakers: 18 PD and 21 controls. Only recordings originally sampled at
16 kHz are used in either training or testing. The main outer-fold assignments
are retained. This is a smaller speaker cohort than the main experiment.

| Member | F1 | Recall | Specificity | Precision | Accuracy |
|---|---:|---:|---:|---:|---:|
| Arannamoy | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Mohibur | 94.44 | 94.44 | 95.24 | 94.44 | 94.87 |
| Osman | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |

| Member | TP | FN | TN | FP | AUROC | Average precision | MCC | Balanced accuracy (%) | NPV (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Arannamoy | 18 | 0 | 21 | 0 | 1.0000 | 1.0000 | 1.0000 | 100.00 | 100.00 |
| Mohibur | 17 | 1 | 20 | 1 | 0.9921 | 0.9907 | 0.8968 | 94.84 | 95.24 |
| Osman | 18 | 0 | 21 | 0 | 1.0000 | 1.0000 | 1.0000 | 100.00 | 100.00 |

### Native-rate fold mean and sample standard deviation

| Member | F1 | Recall | Specificity | Precision | Accuracy |
|---|---:|---:|---:|---:|---:|
| Arannamoy | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 |
| Mohibur | 93.33 +/- 14.91 | 93.33 +/- 14.91 | 95.00 +/- 11.18 | 93.33 +/- 14.91 | 94.29 +/- 12.78 |
| Osman | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 |

The perfect scores on this 39-speaker subset do not establish performance on
new populations. Native-rate restriction does not make microphones, recruitment,
recording rooms, or other unmeasured cohort differences identical.

## Acoustic inputs and checks

| Member | Recordings in main / native runs | Inputs |
|---|---:|---|
| Arannamoy | 773 / 613 | All tasks; fixed-reference 64-band log-mel; positional, channel-independent PatchTST; 29,553 parameters |
| Mohibur | 94 / 74 | B1/B2 passages; frozen Whisper-large-v3; 1,280-dimensional speaker embeddings; logistic regression |
| Osman | 485 / 385 | Five vowels; pitch median/deviation, jitter, shimmer, 13 MFCC means/deviations, vowel identity; logistic regression |

Twenty speaker-label shuffles were also checked for Osman's native-rate model,
repeating inner model selection. Shuffled-label pooled AUROC averaged 0.4213,
with range 0.1852 to 0.6931. This is a sanity check against automatic high scores,
not proof that all acquisition-related confounding has been eliminated.

## Saved evidence

Under the completed run directory:

- `summary.json`: every scenario, pooled metrics, individual folds, mean and SD.
- `pooled_metrics.csv`: main, flag-excluded, and native-rate pooled metrics.
- `fold_mean_sd.csv`: main and native-rate fold mean and SD for all ten metrics.
- `main/<member>/speaker_predictions.csv`: one OOF prediction per common speaker.
- `native16k/<member>/speaker_predictions.csv`: native-rate OOF predictions.
- `<scenario>/<member>/fold_<number>/`: split IDs, selected settings, models, and predictions.
- `diagnostics/all_members_prediction_audit.json`: all 30 independent model replay checks.
- `diagnostics/osman_shuffled_label_control.json`: shuffled-label control scores.
- `protocol.json`, `config.yaml`, `folds.csv`, and `source/`: frozen provenance.

The implementation and reproduction commands are in `docs/corrected_comparison.md`.
