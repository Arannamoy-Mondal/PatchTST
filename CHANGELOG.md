# Experiment changelog

Versions v1.0--v3.0 were labeled retrospectively; v4.0 was declared before training. See
`config/experiment_versions.yaml` for the exact artifacts and integrity hashes,
and `docs/experiment_versions.md` for feature specifications.

## v4.0 - Completed WavLM proposal experiment and report

- Added `experiments/v4_0_wavlm_classifiers/` at the user's request, following the
  latest technical report's evaluation and provenance pattern.
- Implemented frozen, pinned WavLM-large embeddings and matched logistic,
  Random Forest and XGBoost classifier comparisons.
- Reused the corrected main/native cohorts and outer folds; kept young controls out.
- Added training-only nested selection, exact partition exports, refitted model
  saves, speaker OOF aggregation, AUPRC/AP distinction and saved-model replay.
- Implemented conditional upper-layer fine-tuning using a training-side AUROC gate.
- Window, class-weight, classifier and real upper-layer gradient smoke checks passed.
- Preserved the initial feature-extraction attempt after a synthetic check caught
  an XGBoost serialization incompatibility; corrected native booster export and
  verified all 15 synthetic saved-model replays.
- Completed `artifacts/runs/20261014T191654Z_v4.0_eb438e63b942/`; full results are
  in `docs/v4_results.md`. Random Forest did not improve on logistic regression;
  XGBoost scored lower on the main cohort. No training-side fold triggered fine-tuning.
- Added the 40-page `docs/wavlm_v4_0_technical_report.pdf`, editable source,
  six statistical figures, exact roles for all 773 files, 80 partition CSVs
  and an offline Plotly score/file browser.
- All printed filenames, zipped partitions and 80 offline filters passed review;
  all 403 report/source/artifact fingerprints matched. The original v2.0 report
  and all 98 of its registered fingerprints remain unchanged.
- Retained the official v2.0 result and deferred external benchmark.

## v4.0-audit.1 - Scaler-check dtype resolution

- Preserved the original nine weighted-scaler checker failures separately.
- Reproduced sklearn's float32 input-weight cast and float64 accumulation without
  changing tolerances, trained models, predictions, settings or audio features.
- All 30 saved classifiers, ten conditional branches, 773 raw-file hashes and
  two predeclared fresh-embedding examples passed the resolved audit.
- Evidence: `artifacts/runs/20261014T194536Z_v4.0-audit.1_9a910f079ae0/`.

## Report 1.2 - Exact fold files and statistical plots

- Extended the verified report to 41 pages and 21 black-and-white figures:
  `docs/patchtst_technical_report_v1_2.pdf`.
- Added all 773 exact filenames and their roles in every fold, a directory key,
  and 20 CSV lists for inner training, validation, final refit and testing.
- Added six Matplotlib figures for split sizes, fold metrics, score distributions,
  ranking curves, epoch selection and recording availability.
- Added an offline Plotly dashboard with interactive plots and exact-path search;
  bundled the file-list CSVs into a ZIP download.
- Verified every printed file row, all CSV partitions, all 20 interactive filters,
  the PDF layout and 98 artifact/source/generated-file fingerprints.
- Saved editable sources under `docs/whitepaper_v1_2/`. The trained result remains
  v2.0 / Arannamoy / main; Report 1.2 is a document extension dated 13 October 2026.

## Report 1.1 - Current verified PatchTST result only

- Added `docs/patchtst_technical_report_v1_1.pdf`, focused on the current official
  v2.0 / Arannamoy / main result and its v2.0-audit.2 verification.
- Kept one input/model, all five held-out fold results, fold spread and pooled
  metrics, with a dedicated verified-OOF evidence section.
- Added report-build checks for speaker folds, inner/outer separation,
  held-out prediction sources, two-stage averaging, threshold and epoch records.
- Used the supported wording "speaker-disjoint and leakage-checked" and retained
  the evidence of unresolved recording-condition confounding.
- Updated the main report link and saved source under `docs/whitepaper_v1_1/`.
  Document version 1.1 does not introduce a new trained experiment.

## Report 1.0 - Single-model technical whitepaper

- Added `docs/patchtst_technical_report_v1_0.pdf`: an 18-page black-and-white,
  single-column report focused on v2.0 Arannamoy PatchTST.
- Included 15 figures explaining the full pipeline, OOF folds, preprocessing,
  model structure, training, aggregation, leakage controls and recording dates.
- Built metric tables and the speaker-score plot from saved predictions.
- Saved editable LaTeX, the report builder and artifact fingerprints under
  `docs/whitepaper_v1_0/`. Report 1.0 is a document version, not a trained model.

## v2.0-audit.2 - Raw-audio recheck and publication assessment

- Replayed all 30 v2.0 outer models and refitted six existing diagnostic controls.
- Verified all 773 main WAV hashes, decoded hashes and fresh log-mel arrays.
- Resolved an initial tiny feature discrepancy as BLAS-thread rounding; retained
  the original check and its follow-up evidence separately.
- Replayed all ten PatchTST models directly from freshly extracted raw-audio features.
- Found that the only main PatchTST error is a patient excluded by native-rate filtering.
- Added a publication assessment with current literature and corrected the
  interpretation of the perfect native-rate result in the completion record.

## v3.0 - Time-patch transformer experiment

- Added a separate experiment folder, `experiments/v3_0_time_patch_transformer/`.
- Reused frozen v2.0 speaker folds and log-mel features without overwriting v2.0.
- Trained a 47,105-parameter transformer on raw and average-spectrum-removed inputs.
- Repeated the static-spectrum logistic baseline in the same run.
- Saved results under `artifacts/runs/20261012T210954Z_v3.0_743b9c95f323/`.
- The new transformer did not beat the baseline on the main 46-person comparison.

## v2.1 - Readable Python implementation

- Rewrote the active pipeline with descriptive names, shorter functions, and
  plain-English comments while preserving v2.0 experiment behavior.
- Preserved the completed implementation separately in `src/patchtst/v2_0_frozen/`.
- Added explicit loading support for historical v2.0 PatchTST checkpoints.
- Checked the readable index, folds, logistic replay, and PatchTST replay against
  the completed v2.0 artifacts.
- Did not replace v2.0 results with a new training run.

## v2.0-audit.1 - High-score audit

- Rechecked person splits, frozen source, decoded audio hashes, and epoch selection.
- Tested the saved PatchTST with shuffled frames and constant temporal means.
- Evaluated static-spectrum, low-energy, recording-statistic, and metadata controls.
- Completed one full person-label-shuffled PatchTST retraining.
- Documented recording-date separation and remaining acquisition-related concerns.

## v2.0 - Corrected three-member comparison

- Shared 46-person outer folds and fixed threshold 0.5.
- Corrected repeat-session grouping in the acoustic baseline evaluation.
- Preserved sound-level and frequency-band information in channel-independent
  PatchTST, with explicit patch positions.
- Corrected file-to-speaker aggregation, training weights, and seed control.
- Used training-side selection and saved refitted outer models.
- Published main, flag-excluded, and native-rate results with prediction replay.

## v1.0 - Original Arannamoy pipeline

- Historical log-mel transformer run on 47 people.
- Per-window normalization, early mel-band averaging, and no positional encoding.
- Validation threshold selection and direct window-to-speaker averaging.
- Source and results retained for comparison with v2.0.
