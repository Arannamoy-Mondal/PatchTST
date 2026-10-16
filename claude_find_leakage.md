# OOF leakage review

Date: 13 October 2026
Scope: every code path in this repository that produces, replays, or reports out-of-fold (OOF) results.
Changes to the project: none. This file is the only new project file. No model was trained, no pipeline was run, and git was not run.

## Reviewer and cost

- Model: Claude Haiku 5.5, model ID `claude-haiku-5-5`. Assistant knowledge cutoff: June 2026.
- Reasoning effort: max (set with `/effort max` for this session).
- Tokens used for this review: see Section 9.

## 1. Verdict

- **Official result (v2.0, Arannamoy PatchTST, main cohort, plus the two logistic comparison members): no OOF leakage found.** Folds are speaker-disjoint. Hyperparameters and epochs are chosen only inside training speakers. Preprocessing is fitted inside training folds. The threshold is fixed at 0.5. Every speaker is scored once. I replayed the fold rules and the selections from the saved files: 457 checks pass and 1 heuristic check fails. The failing check is the Vito S identity question in Section 4.
- **Historical Osman notebook: real OOF leakage.** It treated repeat recording folders as separate people, so one person's two sessions landed on opposite sides of its holdout split, and in both of its cross-validation loops. Its numbers are superseded and should not be quoted.
- **Latent hazards and weaker checks:** an unused threshold-search helper in the active code, two verification checks that can miss a missing speaker, one audit script that cannot load v2.0 checkpoints, and historical v1.0 method issues. None of these changes the official numbers.

## 2. What was read, and how

- **Executed v2.0 code** (`artifacts/runs/20261012T183328Z_f-all_cead14cc32f1/source/src/patchtst/`), read line by line. Every module is byte-identical to `src/patchtst/v2_0_frozen/`, which I confirmed with `cmp`. The modules are `comparison.py`, `comparison_features.py`, `train.py`, `metrics.py`, `model/patchtst.py`, `data/index.py`, `data/splits.py`, `data/audio.py`, `data/demographics.py`, `config.py`.
- **Active v2.1 code**, read line by line: the same modules under `src/patchtst/`. Their leakage-relevant logic matches the executed code.
- **Entry points and verification:** `scripts/compare_members.py`, `train.py`, `evaluate.py`, `build_manifest.py`, `cache_features.py`, `check_config.py`, `check_data.py`, `verify_comparison.py`, `check_corrected_results.py`.
- **Diagnostics and audits:** the three high-score diagnostics, plus `recheck_publication.py` and `resolve_feature_rounding.py` in the v2.0-audit.2 run.
- **Experiments:** v3.0 (`run.py`, `model.py`) read in full. The v1.0 legacy `train.py` and `scripts/train.py` read in full. The legacy model, splits, and metrics were checked at the specific lines cited.
- **Reporting:** `docs/whitepaper_v1_2/build_report.py` and `notebooks/plot_speaker_results.py` were read in full. `docs/whitepaper_v1_2/report_assets.py` lines 1–120 were read. The rest of that file (plotting and dashboard HTML) and the v1.0 and v1.1 report builders were scanned for fitting, thresholds, and selection only.
- **Member notebooks** (historical originals): their code cells were extracted to the scratchpad. Mohibur cells 3, 6, and 7 were read at the leakage-relevant lines. Osman cells 1, 3, 17, and 18 were read at the leakage-relevant lines. The other cells were checked by pattern and by their printed outputs.
- **Read-only checks** (scratchpad only): `leak_checks.py` (457 pass, 1 fail, 8 sections), `sheet_detail.py` (spreadsheet rows), `osman_repeat_check.py` (replays of the Osman splits). Two earlier replay drafts grouped all three "Vito S" folders together, which is wrong for the 46M pair. `osman_repeat_check.py` is the corrected version.

## 3. Official v2.0 pipeline: each leakage route checked

| # | Route to leakage | Where it is handled | Verdict and evidence |
|---|---|---|---|
| 1 | Same person in train and test | Folds are assigned per person (`src/patchtst/data/splits.py:17-22`, `comparison.py:69`) and copied to each file (`comparison.py:70`) | Pass. Each speaker has one fold. For all 30 model runs (3 members, 2 scenarios, 5 folds), outer train and outer test are disjoint and together cover every speaker (`leak_checks.py` sections 1–2). |
| 2 | Inner selection sees outer test people | C is chosen by inner stratified CV on training people only (`comparison.py:176-192`). PatchTST epochs are chosen by a speaker-level validation split of training people only (`train.py:113-120`, `90-101`). | Pass. Replayed from saved files: all 20 logistic folds reproduce their C from inner AUROC alone, and all 10 PatchTST folds reproduce the epoch from validation loss and refit that many epochs (`leak_checks.py` sections 4–5). |
| 3 | Preprocessing fitted on all data | Logistic imputer, scaler, and one-hot encoder sit inside the pipeline, fitted on training rows (`comparison.py:145-167`). PatchTST scaling is fixed (`model/patchtst.py:22-23, 41`). Log-mel uses fixed reference 1.0 (`data/audio.py:25-42`). MFCC uses fixed reference 1.0 (`comparison_features.py:62`). | Pass. No data-dependent statistics anywhere in the pipeline. |
| 4 | Sample or class weights from test people | PatchTST weights come from training windows only (`train.py:40-46`). Logistic weights come from training rows only (`comparison.py:159-161`). | Pass. |
| 5 | Threshold chosen on test or OOF data | Fixed at 0.5 in config, protocol, and scoring (`comparison.py:108`, `194`; `train.py:137-139`). `choose_threshold` exists (`metrics.py:63`), but nothing in the official path calls it. | Pass. Latent hazard F2. |
| 6 | Duplicate audio across folds | Identical bytes across different people are checked (`comparison.py:61-62`). Unique paths are checked (`comparison.py:63`). | Pass. One PD speaker's B1 and B2 files are byte-identical, but both are in fold 0 (`leak_checks.py` section 6). See F8. |
| 7 | Same person under two IDs | Folders merge by cohort and token, then by cohort, normalized name, and birth-year/sex (`data/index.py:192-206`). IDs are built at `index.py:208-223`. | Pass for every token. The Vito S pair is the one open question (Section 4). |
| 8 | Test set changed after seeing results | The native cohort is defined by file sample rate before any results (`comparison.py:74`). Native analyses keep the main folds (`leak_checks.py` section 3). Sheet-flag removal is evaluation-only (`comparison.py:295`). The protocol and source hashes are frozen before the run (`comparison.py:98-116`), and `run_member` refuses to run if code changed (`comparison.py:207-210`). | Pass. |
| 9 | Pooling mixes folds or repeats people | One row per person and exact coverage are required (`comparison.py:288-289`). Two-stage averaging (`metrics.py:9-19`). | Pass. |
| 10 | State shared across folds | The spectrogram cache is keyed by file (`train.py:17`, `30-32`). It holds features only, and arrays are sliced, not modified. | Pass. |
| 11 | Inference depends on other samples | No batch normalization in the official model. Each window is scored on its own. | Pass. |
| 12 | Label or metadata used as input | The model's forward pass takes windows only. Logistic inputs are filtered by prefix (`comparison.py:175`), which excludes label, age, sex, and native rate. The audits check `no_label_or_metadata_input`. | Pass. |
| 13 | Folds stratified with all labels | `StratifiedKFold` uses every person's label to balance folds. | Standard practice. Test labels shape the fold composition, not the training data. Not counted as leakage. |

## 4. The one open identity question: Vito S

Two PD speakers share a name and sex: `PD_VSIPTIOZ46M` (fold 0, born 1946) and `PD_VSIOTLOP47M` (fold 2, born 1947). The code keeps them apart. If they were one person, folds 0 and 2 would both leak, because each one's test person would also appear in training.

Evidence that they are two people:

- The spreadsheet (`Italian Parkinson's Voice and speech/28 People with Parkinson's disease/TAB 5.xlsx`) has 28 person rows. That is 25 people plus 3 extra rows for the repeat-session people (Nicola S, Roberto R, and the 46M Vito S). The extra rows carry different timing and CPS values, so they are separate sessions.
- Three Vito S rows exist. Two are aged 71, which fits the two 46M sessions (born 1946). One is aged 70, which fits the 47M person (born 1947) in 2017.
- The token letter codes differ (VSIPTIOZ against VSIOTLOP). The folders also sit in different batches: 46M in 6-10 and 11-16, and 47M in 17-28.

Verdict: the source structure supports two people. The code does not test for this, and only the original study records could prove it. Recommendation: add a printed same-name collision check for human review. I did not add it, because AGENTS.md asks for no code changes under `src/` or `scripts/` without a request.

## 5. Historical code

### 5.1 Osman original notebook: leakage (historical)

`osman_goni_codes/osman_goni_codes.ipynb` uses folder-level IDs `S001`–`S050` (code cell 1, lines 39–42; assert at line 50). So each repeat-session folder is a separate "speaker". I replayed its splits with sklearn, using the notebook's seeds and ordering. The replayed holdout reproduces the printed test IDs exactly: S007, S011, S015, S020, S023, S024, S034, S043, S044, S047.

| Split (notebook location) | Seed | Nicola S (S030, S047) | Vito S 46M (S033, S050) | Roberto R (S027, S032) |
|---|---|---|---|---|
| Single 80/20 holdout (cell 3, line 15) | 42 | S030 in train, S047 in **test** | both in train | both in train |
| Outer 5-fold, five-vowel CV (cell 17, line 51) | 2026 | split: folds 1 and 2 | split: folds 3 and 4 | same fold (4) |
| C-selection 5-fold (cell 18, line 15) | 42 | split: folds 2 and 4 | split: folds 0 and 2 | same fold (3) |

Impact: in the holdout, one test speaker's other session was in the training set. In both nested CV loops, two people's sessions are split across folds. The historical numbers are inflated by an amount I have not measured. v2.0 merges repeat sessions and fixes this. Action: do not quote the Osman notebook numbers, and mark the notebook as superseded.

### 5.2 Mohibur original notebook: no leakage found

- Identity: the speaker ID is a hash of cohort, folder name (casefolded), birth year, and sex (cell 3, lines 67–73). Repeat sessions therefore merge. Its printed output shows 25 PD and 21 elderly controls, and the merged Vito S entry with sessions 6-10 and 11-16.
- OOF: one row per person (cell 6). A 5-fold stratified split is applied over people (cell 7, lines 59–70). The scaler sits inside the pipeline, and C is chosen by inner CV.
- External MDVR-KCL test (cells 16–20): the model is fitted on Italian speakers only, and no MDVR labels are used for any choice in the code. This is not in the project completion record. If it is quoted, label it as external.

### 5.3 v1.0 legacy code: no test leakage, method issues

- The threshold is chosen by maximizing F1 on the inner validation people (`artifacts/runs/20261012T182626Z_f-all_legacy-source/src/patchtst/train.py:117-118`). That is about nine people, so the threshold is unstable. It is not test leakage.
- The 16 kHz-only check re-splits folds (`.../scripts/train.py:48`), so its folds differ from the main run. v2.0 keeps the main folds.
- Per-window, per-band standardization (`.../src/patchtst/model/patchtst.py:48-49`) removes loudness cues. v2.0 changed this.

### 5.4 v3.0 experiment: no leakage

- The folds are inherited from v2.0. The static baseline is fitted per fold (`experiments/v3_0_time_patch_transformer/run.py:250-277`). Epoch selection uses only the speaker-level validation split (`run.py:180-200`).
- Two variants are scored on the same OOF predictions. Do not pick the better one from those predictions. Report both, as the current results do.

## 6. Diagnostics, verification, and reporting code

- `diagnostics/high_score_audit/audit_high_scores.py:33-43`: each control is fitted on training folds with fixed C=1.0. Nothing is tuned on test data. Pass.
- `audit_patchtst_ablations.py:26-41`: applies inference-time changes to held-out inputs only. Pass. Line 26 loads checkpoints with strict key matching into the renamed class, which fails (F5).
- `audit_patchtst_permutation.py:23-26`: shuffles labels once across all people and applies the same shuffle to training and test. Folds are unchanged. Pass. It is a null sanity check, not a significance test.
- `v2.0-audit.2/recheck_publication.py:204-208`: control refits use training folds only. Pass. Line 212 uses the NaN-skipping difference (F4).
- `scripts/verify_comparison.py:117-121`: inner and outer separation. Line 161 is NaN-safe. Pass.
- `scripts/check_corrected_results.py:52`: NaN-skipping maximum (F4). Its permutation control reuses the official `logistic_fold`. Otherwise pass.
- `docs/whitepaper_v1_2/build_report.py:54-160`: re-derives speaker and fold rules, two-stage averaging, epoch selection, and refit length from the saved files. No model is fitted. Pass. Its phrase "Speaker-disjoint and leakage-checked" (line 187) is broader than the checks it runs (F9).
- `notebooks/plot_speaker_results.py`: reads saved predictions and plots them. Pass.
- `docs/whitepaper_v1_2/report_assets.py`: `write_file_lists` uses saved splits and checks disjointness. The remaining plotting code was scanned, and its only thresholds are fixed 0.5 markers.
- `scripts/build_manifest.py`: writes `data/folds.csv` for 47 people. It disagrees with the official folds for 15 of the 46 shared people (F10). The official run never reads `data/`.

## 7. Findings

| ID | Severity | Where | Finding | Recommended action (not applied) |
|---|---|---|---|---|
| F1 | High, historical | `osman_goni_codes/osman_goni_codes.ipynb`, cells 1, 3, 17, 18 | Folder-level IDs place repeat-session people on both sides of the holdout and of both CV loops (Section 5.1). | Mark superseded. Do not quote. |
| F2 | Medium, latent | `src/patchtst/metrics.py:63-81` (frozen copy: `metrics.py:22-33`) | `choose_threshold` picks the F1-maximizing threshold. It is unused now. Any call on OOF or test data would leak. | Delete it, or restrict it to training data and rename it. |
| F3 | Medium, needs source records | `src/patchtst/data/index.py:192-206` | No automatic check for same-name people with close birth years in different folds. The Vito S pair is the case (Section 4). | Add a printed collision check for review. |
| F4 | Low | `scripts/check_corrected_results.py:52`; `artifacts/.../v2.0-audit.2/resolve_feature_rounding.py:63`; `artifacts/.../v2.0-audit.2/recheck_publication.py:212` | `.abs().max()` ignores NaN, so a missing speaker could pass. Coverage is enforced elsewhere, and `verify_comparison.py:161` is NaN-safe. | Use an explicit coverage check, or a comparison that fails on NaN. |
| F5 | Low, reproducibility | `artifacts/.../diagnostics/high_score_audit/audit_patchtst_ablations.py:26` | Loads v2.0 checkpoints with strict key matching into the renamed v2.1 class, so it fails. `PatchTST.load_v2_checkpoint` exists for this. | Use `load_v2_checkpoint`, as `verify_comparison.py` does. |
| F6 | Low, historical | v1.0 `train.py:117-118`; v1.0 `scripts/train.py:48` | Threshold chosen by F1 on about nine validation people. The 16 kHz check re-splits folds. Neither is test leakage. | Note in v1.0 documentation. |
| F7 | Low, historical | v1.0 `model/patchtst.py:48-49` | Per-window, per-band standardization removes absolute-level cues. The project's notes say hypophonia (low loudness) is a cue. | Note. v2.0 already changed this. |
| F8 | Info (correction) | `train.py:40-46`; `comparison.py:159-161` | The B1/B2 duplicate does not double-weight its speaker, because weights are normalized per speaker. My earlier statement that it did was wrong. | Keep as a data-hygiene note. |
| F9 | Info | `docs/whitepaper_v1_2/build_report.py:187` | "Speaker-disjoint and leakage-checked" is broader than the checks listed. | Name the checks that were run. |
| F10 | Low | `scripts/build_manifest.py`; `data/folds.csv` | `data/` holds a 47-person fold table that disagrees with the official folds for 15 of 46 shared people. The official run does not read it, but mixing the two would be wrong. | Remove or rename it, or tag it as v1.0. |
| F11 | Info | `mohibur_rahman_codes/mohibur_rahman_codes.ipynb`, cells 16–20 | External MDVR-KCL test (37 speakers) is not in the completion record. No MDVR labels are used for choices in the code. | Label it as external if quoted. |
| F12 | Info, confound | `data/manifest.csv`; v2.0 results | Recording-session shortcut. PD at 44.1 kHz on 26 May and 16 June 2016 (160 files). PD at 16 kHz on 24 January and 10 February 2017 (277 files). Controls at 16 kHz on five dates from 17 March to 2 April 2017 (349 files). Speaker-disjoint folds cannot remove this. The audits show the score tracks recording conditions. | Run a decodability check: can sample rate or session month be predicted from the 16 kHz windows? Not run. |
| F13 | Info | `comparison.py:70` | Writes NaN folds for 13 non-common recordings into `manifest.csv`. Harmless, because `run_member` filters first (`comparison.py:212-213`). | None needed. |
| F14 | Info | Diagnostic scripts | Hard-coded `/root/Projects/PatchTST/...` paths. | Use project-relative paths. |

## 8. Not checked

- I did not run the pipeline, the audits, or the replays. I relied on the saved outputs, the replays that the audit scripts recorded, and my own read-only checks.
- I did not read the rest of `report_assets.py` (plotting and HTML) or the v1.0 and v1.1 report builders line by line.
- I did not check whether the GitHub repository is public, or DataPort's terms for the corpus.
- The Osman CV replays assume the notebook's sample ordering and labels. The holdout replay matches the printed test IDs, which supports that assumption.

## 9. Tokens used

- First counter reading in this review, taken after the first inventory step: 14,974,721.
- Last counter reading before this file was written: 14,669,321.
- Measured change between those two readings: about 305,000 tokens.
- Estimate for the whole review, including the work before the first reading, the report, and the edits: about 330,000 tokens. Of that, roughly 125,000 is the file contents I read and wrote; the rest is reasoning and text (estimate).
- Caveat: the counter does not separate input, output, or reasoning tokens, and it may not match billing. It also rose between turns, so treat the total as an estimate.
