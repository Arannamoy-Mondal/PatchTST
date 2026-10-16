# Why the corrected scores are unusually high

Audit version: **v2.0-audit.1**, derived from pipeline **v2.0**.
See [version definitions and features](experiment_versions.md).

## Conclusion

The saved metrics reproduce, and the new audit did not find person overlap or
direct label input. However, high scores survive removal of time ordering and
can be obtained with simple static acoustic summaries. The corpus also has
recording dates and low-energy audio characteristics that differ by class.

The evidence supports describing the result as strong internal classification
on this corpus. It does not establish that PatchTST learned disease-specific
temporal patterns, or that all acquisition-related confounding has been removed.
Earlier assurances about fixing all issues should be read more narrowly:
identified implementation and speaker-splitting problems were corrected.

## What was checked again

- All 30 canonical outer train/test splits are disjoint.
- Each scenario has exactly one pooled held-out prediction per eligible person.
- Implementation hashes still match the frozen corrected run.
- All ten PatchTST selected epoch counts reproduce from inner validation loss
  alone, and each outer-training refit uses that count.
- Decoded PCM hashes reveal zero identical audio signals assigned to different
  people among the 773 main recordings. This check is not a proof against every
  possible near-duplicate or unrecognized identity.
- The original PatchTST predictions were independently recomputed from the
  checkpoints again during the perturbation experiment below.

## Removing temporal information from the saved PatchTST

The same five saved outer models were evaluated on their original held-out
people. Each transformation was applied only as a diagnostic, with no fitting
or threshold changes. All core metrics below are percentages.

| Input to the saved model | F1 | Recall | Specificity | Precision | Accuracy |
|---|---:|---:|---:|---:|---:|
| Original log-mel windows | 97.96 | 96.00 | 100.00 | 100.00 | 97.83 |
| Randomly shuffled frames within each window | 97.96 | 96.00 | 100.00 | 100.00 | 97.83 |
| Each window replaced by its constant time-averaged spectrum | 97.96 | 96.00 | 100.00 | 100.00 | 97.83 |

All three versions have pooled AUROC 1.0. Predictions are not numerically
identical: shuffling changes speaker probabilities by 0.0329 on average, and
constant windows change them by 0.0421 on average. None of the 46 classifications
changes. Thus, the original ordering and temporal variation inside four-second
windows are not necessary for these particular classification decisions.

This is a test-time perturbation result, not a controlled retraining comparison
or proof that temporal information never matters on new data.

## Simple diagnostic classifiers

These controls use the same outer person folds. Each person receives a vector
formed by averaging recording-level measurements. Imputation and scaling are
fitted only on training people. Logistic regression uses fixed C=1 and threshold
0.5. There is no test-based selection of a best control model.

### Main common cohort: 46 speakers

| Diagnostic input | F1 | Recall | Specificity | Precision | Accuracy | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Average 64-band spectrum, no time ordering | 98.04 | 100.00 | 95.24 | 96.15 | 97.83 | 0.9886 |
| Average spectrum of lowest-energy 10% of frames | 95.83 | 92.00 | 100.00 | 100.00 | 95.65 | 0.9524 |
| Raw level, DC, zero, clipping, and sample-parity statistics | 89.80 | 88.00 | 90.48 | 91.67 | 89.13 | 0.9371 |
| Age and sex only, diagnostic metadata baseline | 66.67 | 68.00 | 57.14 | 65.38 | 63.04 | 0.5143 |

The simple average-spectrum classifier also correctly classifies 45 of 46
people. It makes a different type of error from PatchTST, explaining its slightly
different F1 despite equal accuracy. Strong classification does not require a
transformer on this split.

For B1/B2 passages alone, the low-energy-spectrum control also classifies 45 of
46 people correctly: F1 97.96%, recall 96.00%, specificity 100%, precision 100%,
accuracy 97.83%, AUROC 0.9638. On vowels alone the same diagnostic is weaker:
F1 83.33%, recall 80.00%, specificity 85.71%, precision 86.96%, accuracy 82.61%,
AUROC 0.8857.

Low-energy frames have not been verified as speech-free. They may contain soft
speech, breaths, pauses, and noise. These results are evidence that simple cues
are sufficient, not proof that the signal is entirely background noise.

## Acquisition-related concerns remain at native 16 kHz

The recording dates parsed from the filenames are completely separated by
class. Among native-16-kHz recordings:

- PD recordings: January 24 and February 10, 2017.
- Control recordings: March 17, 21, 23, 30 and April 2, 2017.

Dates are not fed to any model. The concern is that a recording session's
microphone, gain, room, or background signature could be present in the waveform
and correlate with the class. Speaker-disjoint folds can still share such
recording conditions between training and testing.

At native 16 kHz, the median across speakers of the average recording-level
quietest-frame RMS is approximately -66.77 dBFS for controls and -50.34 dBFS for
patients: a difference of 16.43 dB. These are digital signal levels, not
calibrated physical voice loudness or certified background-noise measurements.

On the 39-person native-rate subset, a one-feature classifier using only this
low-energy level gives F1 91.89%, recall 94.44%, specificity 90.48%, precision
89.47%, accuracy 92.31%, and AUROC 0.9471. The low-energy-spectrum control gives
F1 94.44%, recall 94.44%, specificity 95.24%, precision 94.44%, accuracy 94.87%,
and AUROC 0.9762.

Therefore, restricting the sample rate does not remove the broader concern
about cohort-specific recording characteristics. Static spectral features can
also carry genuine disease cues; these tests cannot assign a causal share to
disease versus acquisition conditions.

## Full PatchTST shuffled-label sanity check

One fixed random permutation of labels was applied at person level, using seed
10084. All of a person's recordings inherited the same shuffled label. The
original five outer folds, inner epoch selection, outer-training refit, and
threshold were retained. The original results were not overwritten.

| F1 | Recall | Specificity | Precision | Accuracy | AUROC |
|---:|---:|---:|---:|---:|---:|
| 53.57% | 60.00% | 23.81% | 48.39% | 43.48% | 0.4400 |

This makes an implementation that automatically produces high scores less
likely. One shuffled-label run is a sanity check, not a permutation significance
test, and it cannot distinguish true disease cues from class-correlated recording
conditions in the real labels.

## Interpretation

The high score is consistent with strong static acoustic separation, preserved
by the corrected model, plus averaging many recordings per person. The main
evaluation uses 10 to 32 recordings per speaker, with a median of 16. It remains
only 46 independent people; these recordings do not increase the independent
sample size to 773.

The architecture and threshold were also revised after earlier results on this
corpus had been inspected. Inner validation protects per-run model selection,
but it cannot turn iterative development on the same corpus into an untouched
external evaluation.

The next convincing test is an independent cohort or a prospectively collected
test set with patients and controls recorded under matched conditions. A
recording-session-held-out analysis may help assess transfer, but this corpus's
class-separated dates do not by themselves resolve the disease-versus-session
ambiguity. No diagnostic score here replaces the original headline result.

## Artifacts

All new diagnostics are under:

`artifacts/runs/20261012T183328Z_f-all_cead14cc32f1/diagnostics/high_score_audit/`

Important files:

- `control_metrics.csv`: all 22 main/native-rate control results, including weak controls.
- `diagnostic_recording_features.csv`: signal statistics and static spectral measurements.
- `patchtst_inference_ablations.json`: original, shuffled-frame, and constant-spectrum metrics and fold variation.
- `patchtst_label_permutation/metrics.json`: fully retrained label-shuffle result.
- `patchtst_label_permutation/label_mapping.csv`: the fixed shuffled labels.
- `passages_low_energy_spectrum_metrics.json` and `vowels_low_energy_spectrum_metrics.json`: task-restricted checks.
- `audit_high_scores.py`, `audit_patchtst_ablations.py`, and `audit_patchtst_permutation.py`: archived diagnostic code.
