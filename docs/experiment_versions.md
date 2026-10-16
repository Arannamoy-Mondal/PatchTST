# Experiment versions and acoustic features

The version registry is `config/experiment_versions.yaml`. Versions v1.0--v3.0
were labeled retrospectively; v4.0 was declared before its training run.
These labels identify saved evidence, not the Python package release.

| Experiment version | Meaning | Primary artifacts |
|---|---|---|
| `v1.0` | Original Arannamoy pipeline; historical | `artifacts/runs/main/` |
| `v2.0` | Corrected three-member evaluation; current internal result | `artifacts/runs/20261012T183328Z_f-all_cead14cc32f1/` |
| `v2.0-audit.1` | Follow-up diagnostic investigation of v2.0 | The v2.0 run's `diagnostics/high_score_audit/` |
| `v2.0-audit.2` | Raw-audio replay, cohort explanation and publication assessment | `artifacts/runs/20261012T215648Z_v2.0-audit.2_2167119cc167/` |
| `v4.0` | Completed WavLM proposal classifier comparison; separate from the official result | `artifacts/runs/20261014T191654Z_v4.0_eb438e63b942/` |
| `v4.0-audit.1` | Scaler-check dtype resolution and unchanged-model replay | `artifacts/runs/20261014T194536Z_v4.0-audit.1_9a910f079ae0/` |

These experiment labels are separate from the Python package version `0.1.0`.
The registry records parent versions, source snapshots, feature definitions,
artifact locations, and SHA-256 fingerprints of key configurations and results.
The v2.0 protocol also records implementation hashes and dependency versions.

## v1.0: Original pipeline

Arannamoy's model used log-mel power spectrograms:

- Audio resampled to 16 kHz.
- 64 mel-frequency bands, FFT size 512, 400-sample analysis window, 160-sample hop.
- Fixed power reference 1.0, with the library's default recording-relative
  80-dB clipping range still active.
- Four-second model windows: 64 bands by 400 frames.

The model then normalized every band within each window and averaged band
embeddings before its transformer. It had no patch positional encoding. Those
operations removed useful level and frequency-band information. The main score
was evaluated on 47 people, with validation-selected thresholds.

The previous source is archived at
`artifacts/runs/20261012T182626Z_f-all_legacy-source/`.

## v2.0: Corrected pipeline

### Arannamoy: the exact model input

**The only acoustic input is a 64-band log-mel power spectrogram.** Each band
describes energy in a frequency region over time; the logarithm expresses that
energy on a decibel scale.

| Setting | Value |
|---|---|
| Target sample rate | 16,000 Hz |
| Mel bands | 64 |
| FFT size | 512 samples |
| Analysis window | 400 samples, 25 ms |
| Frame hop | 160 samples, 10 ms |
| Power-to-dB reference | Fixed at 1.0 |
| Absolute power floor | 1e-10 |
| Recording-relative dB clipping | Disabled |
| Model window | 4 seconds, 400 frames |
| Model-window step | 2 seconds |
| Input shape | 64 bands by 400 frames |
| Fixed model input scaling | (log-mel value + 40) / 40 |

There are 64 feature channels and 25,600 values in one input window. These
values are not 25,600 different named acoustic measurements.

Peak normalization and per-window normalization are absent. The shared
channel-independent encoder, positional embeddings, and flattened head retain
frequency-band identity and allow the model to use patch order. Whether the
trained model actually needs that order is investigated in `v2.0-audit.1`.

Jitter, shimmer, HNR, MFCCs, pitch summaries, CPS, age, sex, dates, and native
sample rate are not inputs to Arannamoy's v2.0 model. Native rate is used for
resampling and sensitivity filtering; it is not passed into the classifier.

### Other members in the same comparison

| Member | Acoustic representation | Tasks |
|---|---|---|
| Mohibur | Frozen Whisper-large-v3 encoder; 1,280-dimensional vectors averaged by recording and speaker | B1/B2 passages |
| Osman | Pitch median and standard deviation, local jitter, local shimmer, mean and standard deviation of 13 MFCCs; 30 numeric features plus vowel identity | Five vowels |

The common main cohort has 46 people. The classifier threshold is fixed at
0.5 and outer person folds are shared. Full results are in
[`corrected_results.md`](corrected_results.md).

## v2.0-audit.1: High-score investigation

This version labels the analysis of v2.0, rather than a new replacement model.
Its additional representations are diagnostic controls, kept distinct from the
features used for the primary PatchTST result.

| Diagnostic | Features or transformation |
|---|---|
| Static spectrum | Mean of each of the 64 log-mel bands over a recording, then equal-weight average across a speaker's recordings |
| Centered static spectrum | The same 64 values with their across-band mean removed per recording |
| Low-energy spectrum | Mean of each mel band over the lowest-energy 10% of frames, then recording-to-speaker averaging |
| Raw recording statistics | Eight values: RMS level, low-energy RMS level, DC offset, absolute DC offset, low-energy DC offset, zero-sample fraction, even-valued PCM-sample fraction, and clipping fraction |
| Individual controls | Duration, native-rate indicator, signal level, DC offset, absolute DC offset, or low-energy level |
| Demographic control | Age and sex, explicitly restricted to a diagnostic baseline |
| Frame-order perturbation | Randomly shuffle time frames within each existing 64-by-400 model window |
| Constant-spectrum perturbation | Replace each model window by its time-averaged spectrum, repeated across all 400 frames |
| Label permutation | Keep v2.0 log-mel features; retrain after shuffling labels at person level |

Low-energy frames are not certified silence: they may contain soft speech,
breaths, and noise. Recording dates were examined as metadata but were not
classifier inputs.

The audit found that v2.0 classifications survive the two temporal perturbations
and that static features already discriminate the groups strongly. Acquisition
effects remain unresolved. See [`high_score_audit.md`](high_score_audit.md).

## v2.0-audit.2: Publication-readiness recheck

This revision replays all 30 saved v2.0 outer models, recomputes all 773 main
log-mel arrays from raw audio, and replays all ten PatchTST checkpoints using
those fresh arrays. Six existing static/low-energy diagnostic controls are
refitted with their original fixed settings. Feature definitions remain those
of v2.0 and v2.0-audit.1; this audit introduces no new primary model.

The initial `1e-5` dB feature-comparison failure is preserved in the evidence.
Changing only BLAS thread count explains it; all arrays match bitwise with
one BLAS thread. The follow-up resolution is saved separately.

The sole patient missed in the main run is excluded from the native-rate cohort.
The original main OOF predictions are already perfect on the retained 39 people.
This is a cohort-composition observation, not another headline score.
See [the assessment](publication_readiness_audit.md) for publication precedents,
remaining confounding and prioritized improvements.

## v3.0: Time-patch transformer

`v3.0` is a new experiment in `experiments/v3_0_time_patch_transformer/`. It does
not replace the v2.0 result. It reuses the v2.0 folds and 64-band log-mel
features.

The model input is still 64 bands by 400 frames. In the centered variant, each
band's average over the recording is subtracted before windowing. The
transformer then sees 25 patches; every patch contains all 64 bands across 16
frames. Results are in [`v3_results.md`](v3_results.md).

## v2.1: Readable Python rewrite

`v2.1` is a readability release, not a new trained experiment. It preserves the
v2.0 feature definitions, person grouping, folds, model mathematics, threshold,
and training procedure. The active files use descriptive names and shorter steps.

The exact completed v2.0 implementation remains separately importable at
`src/patchtst/v2_0_frozen/`. Its file hashes are listed in
`src/patchtst/v2_0_frozen/source_hashes.csv`.

The readable model can load completed v2.0 checkpoints through
`load_v2_checkpoint`, because descriptive layer names differ from the names
stored in those historical weight files. A fresh v2.1 training run must be
prepared as its own timestamped run and must not overwrite v2.0 results.

## v4.0: WavLM proposal experiment

The user requested the experiment described in
`Detecting Parkinson's Disease from Speech PP.pdf`, following the latest
technical report's evaluation and evidence pattern. The new code is under
`experiments/v4_0_wavlm_classifiers/`; it does not replace v2.0.

The input is a frozen, pinned WavLM-large representation: 1,024 final-layer
values averaged within at-most-ten-second waveform windows and then within
recordings. WavLM's standard frontend performs per-window mean/variance
normalization; no additional peak normalization is applied.

Logistic regression, Random Forest and XGBoost use exactly the same recordings,
features, outer folds, three-fold inner speaker splits and fixed threshold 0.5.
The cohort remains the report's 46 people; the native-rate scenario has 39.
Young controls remain excluded. A below-0.80 training-side baseline AUROC
can trigger a training-only comparison of two/four-layer partial fine-tuning.
Held-out test scores never choose that branch. The exact deviations from the
proposal, search grids and reproduction commands are in the experiment README.

Completed run: `artifacts/runs/20261014T191654Z_v4.0_eb438e63b942/`.
Full five-metric results and fold variation are in [the results](v4_results.md).
Random Forest ties logistic regression on main pooled F1, AUROC and AUPRC;
XGBoost scores lower. No outer fold triggers partial fine-tuning.
All three native-rate classifiers classify the smaller 39-person cohort correctly;
that cohort excludes the main missed patient and is not independent validation.

The [40-page report](wavlm_v4_0_technical_report.pdf) follows the latest report's
plain-English structure, with six statistical figures, all 773 exact filenames,
80 partition CSVs and an offline dashboard. Source and review records live under
`docs/wavlm_v4_0_report/`. v2.0 remains the official result.
The initial attempt, `artifacts/runs/20261014T184657Z_v4.0_39701f53f615/`,
is preserved as interrupted: synthetic nested checks found a legacy XGBoost
wrapper serialization incompatibility. Native booster export fixes it; all
15 synthetic model replays pass the predeclared numerical tolerance.

## v4.0-audit.1: Scaler-check resolution

The original check used uncast float64 weights when independently calculating
weighted scaler means. The installed sklearn version first casts weights to the
float32 input dtype, then accumulates in float64. Nine expected-mean checks failed,
although all predicted probabilities reproduced.

The separately saved audit reproduces that cast within the original absolute and
relative `1e-10` tolerances. All 30 classifier replays, ten conditional branches,
773 raw-file fingerprints and two predeclared fresh-embedding examples pass.
Every model, prediction and selected setting is unchanged. The initial failure
remains in the primary run's `replay_checks.json`; its `status.json` points to the
resolved verification. This is a checker correction, not another trained model.
It does not establish acquisition-condition or external clinical generalization.

## Versioning rules for subsequent work

1. Refer to results as an experiment version, member, and scenario, for example
   `v2.0 / Arannamoy / main` or `v2.0-audit.1 / static-spectrum control / main`.
2. Record changed features, model settings, cohorts, or split protocols under a
   new pipeline version, with its own timestamped run and parent version.
3. Give additional diagnostics their own audit revision, such as
   `v2.0-audit.2`, while preserving the original primary predictions.
4. Add the exact run path, configuration/source provenance, and result hashes
   to the registry when registering a completed version.
5. Keep all five core metrics and the evaluation unit attached to every result;
   an audit control's score must not silently become the primary model score.
