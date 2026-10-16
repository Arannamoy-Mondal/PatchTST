# Report 1.0 — completed document review

- Output: `docs/patchtst_technical_report_v1_0.pdf`.
- Length: 18 pages, with 15 figures.
- Focus: v2.0 / Arannamoy PatchTST / main.
- Main fold, pooled and supporting-check metrics were recalculated from the
  saved speaker probabilities and compared with the saved metric records.
- Every page was rendered for visual inspection. The final pages were checked
  again after correcting short overflow pages and command formatting.
- A full-page RGB rendering check found no colored content: all output is
  achromatic. Symbols and written labels distinguish the plotted groups.
- The final LaTeX log has no overfull/underfull boxes or unresolved references.
- The PDF text check found no unresolved reference markers or text outside
  the checked page-edge bounds. All 18 pages contain substantive content.
- Source-data and PDF fingerprints match `report_manifest.json`.

The PDF's diagrams are vector graphics. The probability plot uses the actual
46 saved OOF speaker scores. The averaging example and input/window sketches
are labeled as illustrations. This review verifies the document and its
connection to saved results; it is not new model training or external validation.
