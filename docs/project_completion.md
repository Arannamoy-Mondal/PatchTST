# Internal project completion

The external benchmark is intentionally deferred. It is the next phase, not part
of the completed internal project.

## Official result

Use experiment **v2.0**, Arannamoy's corrected PatchTST, from:

`artifacts/runs/20261012T183328Z_f-all_cead14cc32f1/`

The model input is a 64-band log-mel spectrogram. Each window has 64 frequency
bands and 400 time frames. The decision threshold is fixed at 0.5.

| Required result | Speakers | F1 | Recall | Specificity | Precision | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Main pooled out-of-fold result | 46 | 97.96% | 96.00% | 100.00% | 100.00% | 97.83% |
| Main result with flagged controls removed | 43 | 97.96% | 96.00% | 100.00% | 100.00% | 97.67% |
| Retrained on native 16 kHz recordings only | 39 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

The native-16-kHz F1 did not collapse. High performance is therefore achievable
on this retained cohort without any 44.1-kHz recordings. However, the cohort
also changes: the sole patient missed in the main run is among those excluded.
The original main predictions are already perfect on the retained 39 people.
Native-only retraining is not independent confirmation, and recording dates
remain class-separated. See the [publication-readiness audit](publication_readiness_audit.md)
for the evidence and the limits of the disease-specific interpretation.

Version v3.0 was an additional experiment. Its transformer did not beat the
static-spectrum baseline, so it does not replace this official result.

## What is finished

- Eligible corpus index, repeat-session identities, and speaker folds.
- Corrected PatchTST training and the two group-member baselines.
- Speaker-level recall, specificity, precision, F1, and accuracy.
- The three required results above, plus fold variation.
- Saved predictions, checkpoints, configuration, and source snapshot.
- A results plot that reads only the saved speaker-prediction tables.

## Explicitly deferred

An independent external benchmark, including any ICLR or ICML submission, comes
after this internal project. No external dataset is included in the current
result.
