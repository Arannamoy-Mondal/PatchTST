# v4.0 WavLM proposal experiment results

Completed run: `artifacts/runs/20261014T191654Z_v4.0_eb438e63b942/`.

This separate, revised internal experiment does not replace the official v2.0 PatchTST result. All classifiers use the same frozen WavLM-large recording embeddings and the same speech tasks and speaker folds.

## Pooled speaker metrics

All values are percentages. AUPRC is trapezoidal; AP is average precision.

| Scenario | Classifier | People | F1 | Recall | Specificity | Precision | Accuracy | AUROC | AUPRC | AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main | Logistic regression | 46 | 97.96 | 96.00 | 100.00 | 100.00 | 97.83 | 99.43 | 99.56 | 99.57 |
| main | Random Forest | 46 | 97.96 | 96.00 | 100.00 | 100.00 | 97.83 | 99.43 | 99.56 | 99.57 |
| main | XGBoost | 46 | 96.00 | 96.00 | 95.24 | 96.00 | 95.65 | 98.48 | 99.02 | 99.03 |
| without_sheet_flag | Logistic regression | 43 | 97.96 | 96.00 | 100.00 | 100.00 | 97.67 | 99.78 | 99.84 | 99.85 |
| without_sheet_flag | Random Forest | 43 | 97.96 | 96.00 | 100.00 | 100.00 | 97.67 | 99.56 | 99.70 | 99.70 |
| without_sheet_flag | XGBoost | 43 | 97.96 | 96.00 | 100.00 | 100.00 | 97.67 | 98.67 | 99.21 | 99.23 |
| native16k | Logistic regression | 39 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| native16k | Random Forest | 39 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| native16k | XGBoost | 39 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |

## main fold results

| Classifier | Fold | People | F1 | Recall | Specificity | Precision | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0 | 10 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Logistic regression | 1 | 9 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Logistic regression | 2 | 9 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Logistic regression | 3 | 9 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Logistic regression | 4 | 9 | 88.89 | 80.00 | 100.00 | 100.00 | 88.89 |
| Logistic regression | Mean/sample SD | — | 97.78 +/- 4.97 | 96.00 +/- 8.94 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 97.78 +/- 4.97 |
| Random Forest | 0 | 10 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Random Forest | 1 | 9 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Random Forest | 2 | 9 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Random Forest | 3 | 9 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Random Forest | 4 | 9 | 88.89 | 80.00 | 100.00 | 100.00 | 88.89 |
| Random Forest | Mean/sample SD | — | 97.78 +/- 4.97 | 96.00 +/- 8.94 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 97.78 +/- 4.97 |
| XGBoost | 0 | 10 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| XGBoost | 1 | 9 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| XGBoost | 2 | 9 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| XGBoost | 3 | 9 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| XGBoost | 4 | 9 | 80.00 | 80.00 | 75.00 | 80.00 | 77.78 |
| XGBoost | Mean/sample SD | — | 96.00 +/- 8.94 | 96.00 +/- 8.94 | 95.00 +/- 11.18 | 96.00 +/- 8.94 | 95.56 +/- 9.94 |

## native16k fold results

| Classifier | Fold | People | F1 | Recall | Specificity | Precision | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0 | 10 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Logistic regression | 1 | 6 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Logistic regression | 2 | 9 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Logistic regression | 3 | 7 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Logistic regression | 4 | 7 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Logistic regression | Mean/sample SD | — | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 |
| Random Forest | 0 | 10 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Random Forest | 1 | 6 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Random Forest | 2 | 9 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Random Forest | 3 | 7 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Random Forest | 4 | 7 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Random Forest | Mean/sample SD | — | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 |
| XGBoost | 0 | 10 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| XGBoost | 1 | 6 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| XGBoost | 2 | 9 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| XGBoost | 3 | 7 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| XGBoost | 4 | 7 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| XGBoost | Mean/sample SD | — | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 100.00 +/- 0.00 |

## Answer to the proposal question

- Random Forest: F1 ties the logistic baseline; AUROC ties the logistic baseline; AUPRC ties the logistic baseline.
- XGBoost: F1 is below the logistic baseline; AUROC is below the logistic baseline; AUPRC is below the logistic baseline.

The sole missed patient in the main experiment, `PD_CFAERLEI54F`, is excluded from the native-rate cohort. The smaller-cohort perfect results are not independent confirmation on the same people. No classifier superiority is established by these descriptive comparisons.

## Conditional fine-tuning

Training-side baseline AUROC below 0.80 triggered 0 of ten outer folds. Test scores never choose this branch.

If no folds trigger, the conditional pipeline equals the frozen logistic baseline; it is not an independently trained fourth model.

## Evidence and limitations

All 30 saved classifier models, their inner selection and actual conditional branches passed replay. Every main raw WAV fingerprint matched. Fresh embedding replay covers the first path at each native sample rate, not all recordings.

The original checker falsely flagged nine weighted scaler means because it ignored sklearn's float32 input-weight cast. Separately versioned v4.0-audit.1 reproduces that cast with unchanged tolerances; all checks pass and all models/predictions remain unchanged. The initial failure and its resolution are preserved separately.

The main cohort contains 46 people and 773 recordings. Native-only retraining contains 39 people and 613 recordings. Flag exclusion evaluates 43 people without retraining. Young controls stay excluded. Recording-condition confounding and adaptation to previously inspected data remain unresolved; this is not external clinical validation.

[Technical report](wavlm_v4_0_technical_report.pdf) · [Editable source](wavlm_v4_0_report/README.md) · [Interactive dashboard](wavlm_v4_0_report/generated/statistical_dashboard.html) · [All 80 partition CSVs](wavlm_v4_0_report/generated/fold_file_lists.zip)
