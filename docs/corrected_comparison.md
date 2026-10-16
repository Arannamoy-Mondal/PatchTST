# Corrected comparison of the three member pipelines

Experiment version: **v2.0**. See [versions and features](experiment_versions.md).

## Evaluation fixed before the rerun

The corpus index has 786 eligible recordings and 47 working speaker identities:
25 people with Parkinson's disease and 22 elderly controls. Repeat sessions for
Vito S (46M), Nicola S (44M), and Roberto R (49M) share one identity each.

The comparison uses the 46 people available to every method: 25 patients and
21 controls. Vitantonio D has no B1/B2 passage recording and therefore cannot
enter the three-way comparison. The raw corpus is unchanged.

All methods share one five-fold, stratified speaker split, seed 42. Their speech
tasks differ, so this compares complete pipelines rather than isolating the
effect of the classifier alone. Each threshold is fixed at 0.5. Neither the
outer test labels nor their probabilities select a threshold or model setting.

This is a revised internal evaluation after earlier inspection of the corpus
and its results. It is not an untouched external validation. Speaker grouping
prevents identity leakage; it cannot establish that recording conditions or
other unmeasured differences between cohorts have been eliminated.

## The three corrected implementations

### Arannamoy: compact channel-independent PatchTST

- All eligible tasks for the common speakers.
- Audio resampled to 16 kHz; 64 log-mel bands, FFT 512, window 400 samples,
  frame hop 160 samples. Fixed power reference 1.0, absolute floor 1e-10,
  no recording-relative decibel clipping, and no peak normalization.
- Four-second windows with a two-second step and final-tail coverage.
  Training uses at most eight evenly spaced windows per file; evaluation uses all.
- A shared encoder processes each mel band independently. Learned positional
  embeddings retain patch order. A flattened classification head retains band
  identity. Fixed input scaling replaces per-window normalization.
- Patch length and stride 16, encoder width 16, one encoder layer, four heads,
  dropout 0.2. This compact size was selected using CPU timing, before the rerun
  produced test scores. Training keeps the channel-independent architecture.
- Inner speaker validation selects the number of epochs using speaker loss.
  A fresh model then trains on all outer-training speakers for that many epochs.
  The maximum is 20 epochs, with validation patience five.
- Loss weights give equal total weight to each class, then each speaker within
  a class, then each file within a speaker. All random seeds are recorded.

### Mohibur: frozen Whisper-large-v3 and logistic regression

- B1/B2 passage recordings only; 30-second windows with final-tail coverage.
- The original frozen pretrained encoder, pinned to a recorded model revision.
  CPU float32 is used instead of the notebook's GPU float16.
- Mean encoder output gives 1,280 features per window. Window vectors are
  averaged per recording, and recording vectors per speaker.
- The standard pretrained Whisper frontend is retained; no encoder fitting
  takes place on the Italian corpus. Extracting these fixed features before
  cross-validation does not fit a transform to the held-out cohort.
- Standard scaling and logistic regression are fitted inside the training
  partitions. Three inner speaker folds choose C from 0.01, 0.1, 1, and 10.

### Osman: handcrafted acoustics and logistic regression

- Five vowels: VA1/VA2 through VU1/VU2. The common cohort has 485 recordings.
- Pitch median and deviation, local jitter, local shimmer, and means and
  deviations of 13 MFCCs: 30 acoustic measurements, plus vowel identity.
- Canonical person identities replace folder-based IDs. Repeat sessions cannot
  cross an outer or inner training/test boundary.
- MFCCs use fixed-reference log power without a recording-relative clipping
  floor. Audio trimming and the pitch settings follow the original method.
- Missing-value imputation, scaling, and vowel encoding are fitted only on
  training data. Inner speaker folds choose logistic-regression C. Each person
  has equal total recording weight during fitting.

## Metrics and sensitivity checks

The main outputs are one OOF probability and decision per person, pooled F1,
recall, specificity, precision, and accuracy. Also saved are AUROC, average
precision, balanced accuracy, MCC, NPV, and all four confusion counts.
Average precision is named explicitly rather than treated as trapezoidal AUPRC.

For recording/window classifiers, probabilities are averaged by file first,
then equally across a person's files. The five fold scores are reported with
their mean and sample standard deviation (ddof=1), not the best fold.

Two additional analyses are required:

1. Exclude the three flagged controls present in the common cohort from the
   main OOF evaluation. This leaves 43 speakers; models are not retrained.
2. Retrain and evaluate using only recordings whose native rate is 16 kHz.
   This leaves 39 common speakers, 18 patients and 21 controls. Main outer-fold
   assignments are retained. Every resulting test fold still has both classes.

The second analysis reduces sample-rate confounding but does not establish
that microphones, room acoustics, or recruitment differences are identical.

## Reproduce and inspect a run

```bash
uv sync --locked
uv run python scripts/compare_members.py --prepare
```

The preparation command prints a versioned run directory. Substitute that path
for RUN below:

```bash
uv run python scripts/compare_members.py --run RUN --member arannamoy
uv run python scripts/compare_members.py --run RUN --member mohibur
uv run python scripts/compare_members.py --run RUN --member osman
uv run python scripts/compare_members.py --run RUN --summarize
uv run python scripts/evaluate.py --run RUN
```

The acoustic model's saved prediction provenance and an optional shuffled-label
sanity check can be checked with:

```bash
uv run python scripts/check_corrected_results.py --run RUN --permutations 20
```

Preparation freezes the configuration, person folds, raw-file hashes,
implementation hashes, source snapshot, and dependency versions. Changing the
implementation requires preparing a new run. Completed folds can be resumed
within an unchanged run. Feature caches include the audio hash and extraction
settings, preventing an old array from being silently attached to changed audio
or changed preprocessing.

Each scenario/member directory stores its recording manifest, inner and outer
split IDs, selected settings, fitted models, fold predictions, pooled speaker
predictions, and metrics. `summary.json`, `pooled_metrics.csv`,
`fold_mean_sd.csv`, and `REPORT.md` collect all three members after completion.

The old notebook outputs and the earlier `artifacts/runs/main` results are
historical. They are not the corrected results.
