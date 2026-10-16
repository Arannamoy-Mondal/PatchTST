# Report 1.2 — completed review

Reviewed 13 October 2026.

## Deliverables

- PDF: `docs/patchtst_technical_report_v1_2.pdf` — 41 pages, 21 figures.
- Offline interactive companion: `generated/statistical_dashboard.html`.
- Exact file-list package: `generated/fold_file_lists.zip`.
- Focus: **v2.0 / Arannamoy PatchTST / main**, verified by **v2.0-audit.2**.

## File lists and OOF records

- All 20 exported partition CSVs match the saved `split.json` files joined to
  the frozen 773-recording manifest.
- Every recording has one role in each fold and is test data exactly once.
- Each speaker's recordings share a role within each fold.
- Inner training, validation and test file sets are disjoint; final refit
  training is the union of inner training and validation.
- The directory key and filenames reconstruct all original project-relative paths.
- All **773 printed PDF file rows**, including their five roles, match the CSV
  catalogue. Literal spaces in 16 original filenames are preserved. The initial
  review parser assumed filenames had no spaces; correcting that parser resolved
  the check without changing the filenames or report data.
- The ZIP's 20 partition CSVs and three catalogue/count CSVs match the loose files.
- The existing speaker-separation, held-out prediction-source, averaging,
  fixed-threshold and epoch-record checks pass.
- All fold, pooled, mean and sample-SD metrics match the saved results.

## Layout and plots

- All 41 pages were rendered and visually reviewed; statistical plots, directory
  paths and appendix continuation pages were inspected.
- The final LaTeX log has no overfull/underfull boxes or unresolved references.
- Full-page RGB checks found achromatic PDF output throughout.
- Text checks found no unresolved reference markers, sparse overflow pages or
  text outside the checked page-edge bounds.
- All 98 source/artifact/generated-file fingerprints and the PDF hash match
  `report_manifest.json`.

## Interactive companion

- All six Plotly figures rendered in headless Chrome with networking disabled.
- All 20 fold/role filter combinations returned exactly the expected paths.
- Speaker-ID search returned the expected recordings.
- No browser JavaScript errors were reported.
- Narrow-screen checks confirmed charts scroll inside their containers without
  whole-page horizontal overflow.

The plots are descriptive summaries of saved results. These document checks
do not establish external clinical performance or remove recording-condition
confounding. Prior raw-audio replay evidence is referenced, not rerun during
the document build.
