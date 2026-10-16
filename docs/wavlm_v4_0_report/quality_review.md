# WavLM v4.0 report — completed review

Reviewed 14 October 2026. Document 1.0 describes experiment v4.0, verified by
the separately recorded v4.0-audit.1. It does not replace the official PatchTST report.

## Deliverables

- `docs/wavlm_v4_0_technical_report.pdf`: 40 pages, one pipeline diagram and six
  black-and-white statistical figures.
- `docs/v4_results.md`: all three classifiers, main/flag-excluded/native results,
  every main/native fold and mean/sample SD.
- `generated/statistical_dashboard.html`: offline speaker scores and exact-path browser.
- `generated/fold_file_lists.zip`: all 80 main/native outer/inner partition CSVs.
- `patchtst_style_wavlm.tex`, `build_report.py`, `check_report.py`: editable source,
  saved-record builder and review tools. No training takes place during report generation.

## Data and printed rows

- All 773 printed filenames, directory keys, five three-character fold-role entries
  and native-rate eligibility marks match the catalogue. Original spaces are preserved.
- All 80 partition CSVs match the saved speaker splits and the unmodified raw paths.
- The ZIP copies match the original partition CSVs byte for byte.
- Every person receives one held-out prediction; final refit and inner training are
  clearly distinguished. Three inner speaker folds rotate validation within development.
- Report metrics are recomputed from saved speaker predictions, including flag exclusion,
  fixed decisions, pooled confusion counts and fold mean/sample SD.
- All 403 registered source/artifact/generated fingerprints and the PDF fingerprint match.

## Visual and browser checks

- All 40 pages were rendered and inspected as contact sheets. Main result tables,
  sensitivity tables, audit explanation and file appendix pages were examined in detail.
- The final LaTeX compilation has no overfull/underfull boxes, unresolved references
  or ignored table errors. Exact-file tables are explicitly divided into balanced pages.
- No text crosses the checked 15-point page-edge boundary.
- Poppler RGB renders are achromatic on every page, using exact channel equality.
  PyMuPDF vector recoloring retains text and vector plots; no page is rasterized.
- All 80 offline browser filter combinations return the expected exact paths, without
  JavaScript errors. The interactive score plot renders with networking disabled.
- Narrow-screen review finds no whole-page horizontal overflow.

## Numerical evidence and interpretation

- v4.0-audit.1 replays all 30 classifiers, all ten conditional branches and two
  predeclared raw-to-embedding examples. It rehashes all 773 main raw recordings.
- The original scaler-check failure is preserved. Reconstructing sklearn's actual
  float32 weight cast resolves the expected-mean discrepancy without changing
  tolerances, models, features or predictions. The resolved maximum expected-mean
  difference is approximately `4.44e-16`.
- No fold triggers partial fine-tuning. Those branch outputs are the logistic baseline,
  not an independent fourth trained model.
- Random Forest ties the main logistic baseline; XGBoost does not improve it.
- The native-rate cohort excludes the main missed patient. Its perfect results are
  not independent confirmation on the same participants.
- Recording-condition confounding remains unresolved. The report is speaker-disjoint
  and leakage-checked, not clinical external validation.

The v2.0 technical report and all 98 of its registered fingerprints were also
checked and remain unchanged. Raw audio, old trained results and `PLAN.md` were
not modified.
