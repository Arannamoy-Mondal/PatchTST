# Report 1.1 — completed review

Reviewed 12 October 2026.

- PDF: `docs/patchtst_technical_report_v1_1.pdf`.
- Scope: current official **v2.0 / Arannamoy PatchTST / main**, verified by
  **v2.0-audit.2**.
- Length: 18 pages and 15 black-and-white figures.
- All pages were rendered and visually reviewed; revised diagram and artifact
  table pages were checked again after final layout adjustments.
- The LaTeX log contains no overfull/underfull boxes or unresolved references.
- Full-page RGB checks found achromatic output throughout. Text checks found
  no unresolved reference markers, sparse overflow pages or page-edge problems.
- All 52 source/artifact/generated-file fingerprints and the PDF fingerprint
  match `report_manifest.json`.

## Main OOF records

The report builder verified:

- 46 unique speaker rows matching the frozen speaker labels and five folds.
- 773 unique recordings assigned to their speakers' folds.
- Disjoint inner-training, validation and held-out test people in each fold,
  with both classes in each group.
- Pooled predictions matching the appropriate held-out fold outputs exactly.
- Window means followed by equally weighted recording means reproducing speaker
  scores within `1e-6`; largest observed difference was about `6.82e-8`.
- Identical reconstructed decisions at fixed threshold 0.5.
- Selected epochs and refit lengths matching the inner-validation records.
- All five fold metrics, pooled metrics, fold means and sample SD matching the
  saved metrics.

`generated/oof_checks.json` records the per-fold findings. It also references
the prior audit's 773 bitwise-matching fresh feature arrays and five passed main
checkpoint replays. The report build reads that prior evidence; it does not
repeat audio extraction, model inference or training.

The supported wording is **speaker-disjoint and leakage-checked**.
Recording-condition confounding and reuse of this corpus during development
remain explicit limits on interpretation.
