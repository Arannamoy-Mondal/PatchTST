# Project memory

Parkinson's speech classification with PatchTST on the Italian Parkinson's Voice and Speech corpus (Dimauro and Girardi, IEEE DataPort DOI 10.21227/aw6b-tg17).

The user writes the pipeline by hand. Do not add implementation code under `src/` or `scripts/` unless they explicitly ask for it. The checklist they tick is `PLAN.md`. Do not renumber or rewrite those steps unless they ask. When they say which step they are on, help them understand that step only.

When giving code, write plain Python that a beginner can read from top to bottom. Use `if`, `else`, `print`, and direct comparisons. Do not use `raise`, `assert`, `assertTrue`, `assertEqual`, `assertRaises`, or other test-framework checks. If something is wrong, print the error and what was expected. Do not hide a failure inside an exception.

Use English only in explanations, progress updates, reasoning summaries, code comments, and reports. The user explicitly requested this language preference be saved.

Owner: Arannamoy Mondal (`arannamoymondal13@gmail.com`). Remote: `git@github.com:Arannamoy-Mondal/PatchTST.git`, branch `main`. Last known commit message: `chore:add literature_review`. This environment has no `git` binary. Do not assume a clean status.

## Decision

- Question: PD versus elderly healthy control, from speech.
- Drop the 15 young healthy controls from training and from the reported score. Keep the folder on disk. Do not add a young-versus-PD accuracy to the result.
- The number that counts is speaker-level precision, recall, specificity, accuracy, and F1, pooled across 5 stratified speaker folds. Window accuracy is a debug number only.
- Also report the mean and spread across folds. Do not quote the best fold.
- Three numbers must exist before a result is trusted: main pooled F1, F1 with `sheet_flag` speakers removed, and F1 trained only on native 16 kHz. Trust the third if the first collapses.

## Metrics

Quote five speaker-level metrics together. Do not lead with accuracy, and do not score windows.

- Recall (sensitivity) = TP / (TP + FN). Of the people with Parkinson's, how many were caught?
- Specificity = TN / (TN + FP). Of the elderly controls, how many were correctly left alone?
- Precision = TP / (TP + FP). Of the people called Parkinson's, how many were? This is precision in a balanced speaker sample, not clinical positive predictive value. Community prevalence over 60 is near 1%.
- F1 = 2 · precision · recall / (precision + recall). The comparison number, because it falls if either precision or recall is bought by destroying the other. It ignores true negatives, so it cannot stand alone.
- Accuracy = (TP + TN) / all speakers. Interpretable here because the speaker split is about 25 versus 22, but still not sufficient.

The clinical pair is recall and specificity. A high accuracy with weak recall is missing patients. A high F1 with weak specificity is over-calling Parkinson's. Either pattern is a failure.

Compute the four cells only after averaging window probabilities to one score per speaker and applying one threshold. The threshold is chosen on training folds, never on the test fold. Publish the pooled speaker table, plus mean and spread across the five folds. Do not publish the best fold. A fold has about 9 or 10 test speakers.

Window accuracy and file-level accuracy are debug numbers. Several recordings from one person are not independent trials. `l_17_file.pdf` reports 81.53±2.17% leave-one-individual-out accuracy versus 92.75±1.21% leave-one-recording-out on the Oxford voice set, and treats published 98–100% voice accuracies as suspect when the split is not subject-disjoint. Naranjo et al., cited there, got 75.2% with a subject-aware model on 40 PD and 40 controls. `l_19_...Zhao...pdf` reports about 99% and 90% on this Italian corpus from a random 80/20 file split; do not copy that protocol. `l_02_...pdf` scores the person by the median of recording probabilities.

`l_06_2111.10207v1.pdf`, already cited in `README.md`, reports accuracy, specificity, precision, sensitivity, and F1 together on this corpus and uses F1 as the comparison column. `l_03fnagi-14-1036588.pdf` defines that same set plus false-alarm rate, Matthews correlation, and ROC, because each confusion-matrix cell is a different decision. AUC may be plotted beside the operating point (`l_02`, `l_03`, `l_16`); it does not replace sensitivity and specificity at a chosen threshold. Matthews correlation uses all four cells and may be an extra column later. It is not required to match the README or `l_06`.

## Data rules

- Raw audio stays in `Italian Parkinson's Voice and speech/`. Never write into it, rename a WAV, or move the corpus.
- Config groups are only `28 People with Parkinson's disease` and `22 Elderly Healthy Control`.
- Do not label a file PD because the path contains `Parkinson`. The dataset root contains that word.
- Resample everything to 16 kHz before features. About 160 PD files from 2016 are 44.1 kHz. Every elderly file is 16 kHz. Native rate is a label leak.
- Do not peak-normalize. Do not use a per-file max as the dB reference. Hypophonia is a cue.
- One person, one `speaker_id`. Collapse `VSIPTIOZ46M` (Vito S, both sessions), Nicola S (`44M`, two token spellings), and Roberto R (`49M`, two token spellings). The other Vito S (`47M`) is a different person.
- Antonia G has one bad export: `VI2` says `63F` instead of `52F`. Use the folder's modal token and flag that file.
- Expected index after dropping young controls: 786 files, 47 speakers (25 PD, 22 elderly). The tree on disk has 831 WAVs because it still includes 45 young files. Label 1 is PD, 0 is elderly control.
- Split speakers, then copy the fold onto files. Stratify. Every test fold must contain both classes. No speaker in two folds.
- CPS, age, sex, and native sample rate are metadata. They are not model inputs.
- `sheet_flag` marks Agnese P, Gilda C, Giovanna G, and Vitantonio D. Keep them in the main run.
- `config/base.yaml` now uses stratified speaker folds. The user explicitly requested full implementation and then correction and rerunning of all three member pipelines.

## Current tree

The pipeline has been implemented at the user's explicit request. `PLAN.md` remains unchecked; do not tick it automatically. `data/`, `artifacts/`, and `src/patchtst/data/` now exist. The older compact model and its results are historical; use the corrected comparison entry point for new work.

`scripts/compare_members.py` runs the corrected three-member comparison, documented in `docs/corrected_comparison.md`. Its common cohort is 46 speakers (25 PD, 21 elderly); Vitantonio D has no passage recordings. The native-16-kHz comparison has 39 people. Keep each member's original tasks and identify results as pipeline comparisons. Threshold 0.5 is fixed; logistic-regression C and PatchTST epoch count are selected only within training speakers. Frozen run metadata and source snapshots live under versioned run directories.

Completed corrected run: `artifacts/runs/20261012T183328Z_f-all_cead14cc32f1/`. All 30 saved outer-model prediction replays passed. Full results are in `docs/corrected_results.md` and the run's `summary.json`. Main pooled F1: Arannamoy 0.979592, Mohibur 0.938776, Osman 0.979592. Native-rate pooled F1: 1.0, 0.944444, 1.0 respectively. These are revised internal OOF results on a common task-available cohort, not an untouched external validation. Do not silently reuse the old `artifacts/runs/summary.json` as the corrected result.

Later high-score audit: `docs/high_score_audit.md`; artifacts under that run's `diagnostics/high_score_audit/`. PatchTST retains the same 46 decisions after shuffling frames or replacing each window with its time-averaged spectrum (probabilities do change). A fixed-C static-spectrum logistic control gets F1 0.980392 on the main cohort. Recording dates are class-separated even at native 16 kHz: PD in January/February 2017, controls in March/April. Quietest-frame digital levels differ markedly. One fully retrained person-label permutation gives PatchTST F1 0.535714, AUROC 0.44. Do not claim all leakage/confounding is eliminated or that the high score proves disease-specific temporal learning. Original headline metrics are retained as internal results, with this interpretation update.

The user decided to finish the internal project before handling the external benchmark or an ICLR/ICML submission. Official completion record: `docs/project_completion.md`. Official result remains v2.0 Arannamoy PatchTST. Do not start another model for publication until that deferred benchmark phase begins. The results notebook reads only saved speaker predictions.

Publication-readiness recheck `v2.0-audit.2`: `docs/publication_readiness_audit.md`, artifacts `artifacts/runs/20261012T215648Z_v2.0-audit.2_2167119cc167/`. All 30 saved model replays passed again; all 773 fresh raw-audio log-mel arrays matched the cache bitwise with one BLAS thread, and all ten fresh-feature PatchTST replays passed. The initial 1.5e-5 dB feature discrepancy was BLAS-thread rounding; original failure and follow-up resolution are preserved separately. Main PatchTST's sole missed patient `PD_CFAERLEI54F` is excluded from the native-rate cohort; original main OOF predictions already classify the retained 39 people perfectly. Native-only retraining is not independent confirmation. Every native speaker fold shares recording dates across train/test; each date is class-specific. Publication can be possible as a rigorous bounded audit; prior categorical claims that no second dataset means no possible publication were too strong. A new architecture is not an ICLR/ICML requirement. External clinical generalization remains unverified.

The user requested a new transformer experiment in a different folder with versioning. Completed experiment `v3.0` is in `experiments/v3_0_time_patch_transformer/`; results are in `artifacts/runs/20261012T210954Z_v3.0_743b9c95f323/` and `docs/v3_results.md`. It reuses v2.0 folds and does not replace v2.0. Main 46-person F1: static baseline 0.980392, raw time-patch transformer 0.888889, time-centered transformer 0.960000. Native-rate F1: 0.848485, 0.914286, 0.971429. The new transformer did not beat the baseline on the main comparison. Do not describe v3.0 as external validation.

The user requested a human-readable implementation and suggested C++ or Java. Python remains the project language because the audio and model libraries are Python-based. Active code is now the readable `v2.1` rewrite. The exact completed implementation is preserved in `src/patchtst/v2_0_frozen/` and must not be treated as a new trained result. Use `PatchTST.load_v2_checkpoint` for historical checkpoints. Keep beginner-readable Python style in future edits.

The user requested explicit experiment versioning. Registry: `config/experiment_versions.yaml`; explanation and features: `docs/experiment_versions.md`; changes: `CHANGELOG.md`. Retrospective labels: `v1.0` = original Arannamoy pipeline, `v2.0` = corrected comparison, `v2.0-audit.1` = high-score diagnostics. Use version/member/scenario when discussing results. The audit is not a new primary model. Keep new configurations and diagnostics separately versioned with parent references and exact artifact paths. These labels are distinct from the Python package version.

The user wants presentations focused on the current official model/input only: v2.0 / Arannamoy PatchTST / main and its 64-by-400 log-mel windows, verified by v2.0-audit.2. Latest plain-English technical report: `docs/patchtst_technical_report_v1_2.pdf`, completed 13 October 2026, 41 pages and 21 black-and-white figures. Editable source, saved-record OOF checks, builders, build instructions and fingerprints are under `docs/whitepaper_v1_2/`. Report 1.2 extends the accepted report with all 773 exact filenames and five-fold role assignments, 20 partition CSVs, six Matplotlib statistical figures and an offline Plotly dashboard with a searchable file browser. Downloads: `generated/statistical_dashboard.html` and `generated/fold_file_lists.zip` under that source folder. Distinguish inner training from final refit training (inner train plus validation). All printed file rows, CSV partitions, interactive filters and 98 source/artifact/generated-file hashes passed review. The report shows only the main result, every fold and the pooled metrics; its builder checks speaker assignments, inner/outer separation, held-out prediction sources, two-stage averaging, fixed threshold and epoch records. Say "speaker-disjoint and leakage-checked", not unconditionally "leakage-free": recording-condition confounding remains unresolved. Document version 1.2 is not a new experiment. Reports 1.0/1.1 and the old `docs/pipeline.pdf` are historical documents.

The user requested the WavLM experiment from `Detecting Parkinson's Disease from Speech PP.pdf`, following the latest technical report's pattern. Completed `v4.0` is under `experiments/v4_0_wavlm_classifiers/`, with run `artifacts/runs/20261014T191654Z_v4.0_eb438e63b942/` and results `docs/v4_results.md`. It compares logistic regression, Random Forest and XGBoost on identical frozen WavLM-large 1,024-value recording embeddings, all tasks, corrected 46/39-person cohorts and v2.0 outer folds. Main logistic and Random Forest: F1 97.9592%, recall 96%, specificity 100%, precision 100%, accuracy 97.8261%; XGBoost: F1 96%, recall 96%, specificity 95.2381%, precision 96%, accuracy 95.6522%. Random Forest ties the logistic baseline in main F1/AUROC/AUPRC; XGBoost scores lower. All three native-rate retrainings give all five metrics 100% on the smaller cohort, which excludes their common missed patient `PD_CFAERLEI54F`. These are not independent external confirmations. None of ten training-side AUROC values (0.992063--1.0) triggers partial fine-tuning, so no adapted upper-layer model was trained. The proposal deviations (elderly-only corrected cohort, training-only fine-tuning gate, explicit grid loops rather than GridSearchCV/Optuna wrappers) are documented. Pinned model revision: `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`; the standard frontend uses per-window mean/variance normalization, not peak normalization. XGBoost CPU is isolated under `/tmp/opencode/wavlm-v4-deps`; the project's pinned CPU PyTorch is unchanged. v2.0 remains official, the external benchmark remains deferred, and raw audio, old results and `PLAN.md` remain untouched.

Verification `v4.0-audit.1` is under `artifacts/runs/20261014T194536Z_v4.0-audit.1_9a910f079ae0/`. The original v4.0 `replay_checks.json` preserves nine checker-only weighted-scaler failures: sklearn casts supplied float64 weights to the float32 input dtype before float64 accumulation, which the initial expected-mean check ignored. The corrected audit reproduces that cast within unchanged `1e-10` tolerances, replays all 30 models and ten conditional branches, rehashes all 773 main WAVs, and reproduces two predeclared fresh-embedding examples exactly. Models, predictions, selected settings and features are unchanged; this is not a full all-feature/fresh-training reproduction. Use `audit_scaler_dtype.py` for replaying this frozen result rather than bypassing its source guard. The earlier extraction attempt `artifacts/runs/20261014T184657Z_v4.0_39701f53f615/` is preserved as interrupted after synthetic checks caught an XGBoost wrapper serialization incompatibility; native booster export fixes it. All 15 synthetic nested model replays and upper-layer gradient smoke checks passed. The actual `1e-6` model-replay tolerance was never relaxed.

New proposal-experiment report: `docs/wavlm_v4_0_technical_report.pdf`, document 1.0, 40 pages, one pipeline diagram and six black-and-white statistical figures. Source/review: `docs/wavlm_v4_0_report/`; companions: `generated/statistical_dashboard.html` and `generated/fold_file_lists.zip`. All 773 printed file/role/native-mark rows, 80 exact partition CSVs, all 80 offline filter combinations, achromatic Poppler page renders and 403 source/artifact/generated fingerprints passed review. All 40 pages were rendered for visual review. Original v2.0 Report 1.2 and its 98 fingerprints remain unchanged. This new report describes v4.0 only and does not promote it to the official model. Layout fixtures under `/tmp/opencode/` are explicitly not v4.0 results.

The following tree describes the original starting files; generated directories and implemented modules have since been added.

```text
PatchTST/
├── AGENTS.md                  # this memory
├── PLAN.md                    # 100-step checklist, user ticks boxes
├── README.md                  # stub: one literature link and the five metric names
├── pyproject.toml             # package patchtst 0.1.0, Python >=3.11, uv_build
├── uv.lock
├── .python-version            # 3.11
├── .venv/                     # local uv env, Python 3.11.17; not the project
├── .env                       # empty file, gitignored
├── .gitignore                 # only ".env" — data/ and artifacts/ are not ignored yet
├── .gitattributes             # Git LFS rules for corpus, zip, PDFs, and also src, README, pyproject.toml
├── config/base.yaml           # root, two groups, target_sr 16000, tasks all, group_kfold, 5, seed 42
├── src/patchtst/__init__.py   # prints "Hello from patchtst!"
├── scripts/check_data.py      # loads yaml and prints the two group paths; unused rows list
├── scripts/exp.ipynb          # 7 cells: pandas, yaml, FILE CODES.xlsx, Tab 3.xlsx head
├── literature_review/         # 20 PDFs, l_01 through l_20
├── Detecting Parkinson's Disease from Speech PP.pdf   # 680K, project paper, not read into code
├── Italian Parkinson's Voice and speech.zip           # 566 MB original archive
└── Italian Parkinson's Voice and speech/              # extracted corpus, do not modify
    ├── 15 Young Healthy Control/          # 15 speakers, 45 wavs, not a training class
    ├── 22 Elderly Healthy Control/        # 22 speakers, 349 wavs, negative class
    └── 28 People with Parkinson's disease/  # 28 folders, 437 wavs, positive class
```

Working-tree files are real content, not LFS pointer stubs, even though `.gitattributes` marks the corpus, literature PDFs, `src`, `README.md`, and `pyproject.toml` as LFS. Do not rewrite those attributes unless asked. Do not commit secrets. `.env` is empty.

Dependencies now also include pinned CPU PyTorch, praat-parselmouth, transformers, and accelerate. PyTorch uses an explicit CPU package index; do not replace it with an unconfigured CUDA dependency.

`config/base.yaml` points at the two correct groups and target sample rate 16000. It also holds corrected feature settings, model settings, output paths, and member-specific task selections.

## Corpus facts already verified

- 831 wav, 6 xlsx, 1 stray txt. All wavs are mono PCM 16-bit.
- Young: only `B1`, `B2`, `PR1`. Ages 19–29, 13 M / 2 F. One file is prefixed `PR11`. One filename has a double dot.
- Elderly and PD: passage `B1`/`B2`, syllables `D1`/`D2` (`/pa/`, `/ta/`), vowels `VA`–`VU` twice, words `PR1`, phrases `FB1`.
- `FILE CODES.xlsx` is a legend, not a table to join. Demographics are `Tab 3.xlsx` and `TAB 5.xlsx`.
- CPS is reference letters divided by reading time. Metadata only.
- Folder names do not always match sheet names (`VITO L` is Laforgia, `LISCO G` is Giuseppe L, `SUMMO L` is Luigia S, `Mario B` is Mario M in the sheet).

## Literature that bears on this corpus

Only these files in `literature_review/` actually discuss this dataset:

- `l_06_2111.10207v1.pdf` describes all three age groups, then classifies 495 vowel files. Young speakers have no vowels, so that experiment is PD plus elderly.
- `l_19_...Zhao...pdf` trains on this corpus: 28 PD and 20 healthy controls, vowels only. Those controls are a subset of the elderly group. Its 80/20 file split is not the split to copy.
- `l_07_s41531-025-01025-9.pdf` catalogs the dataset as 28 PD and 22 controls, ages 67.2 versus 67.1.
- `l_12_...pdf` lists 28 PD and 37 controls. That 37 is the zip-file headcount, not a training protocol.

`l_16` and `l_02` are not this corpus, but they are why age matching matters. The other sixteen PDFs are background. `README.md` currently cites only `l_06` and the DataPort page.

## Layout the code should grow into

Do not create these until the matching `PLAN.md` step.

- `config/base.yaml` remains the only place for paths, sample rate, tasks, seed, and fold count.
- `src/patchtst/` holds library code. `scripts/` only load config and call one function.
- `data/` and `artifacts/` are generated and must be gitignored when step 6 is done.
- `notebooks/` reads artifacts. It does not train. `scripts/exp.ipynb` has not been moved yet.
- A run writes only under `artifacts/runs/<date>_f<fold>_<config-hash>/`, including a frozen copy of the config.
- Do not start the model until the manifest, speaker, split, and audio checks in `PLAN.md` pass.

## Citations to keep in the README

- Dimauro et al., VoxTester, IEEE MeMeA 2016, doi:10.1109/MeMeA.2016.7533761.
- Dimauro et al., IEEE Access 2017, doi:10.1109/ACCESS.2017.2762475.
- Dimauro and Girardi, Italian Parkinson's Voice and Speech, IEEE DataPort, 2019, doi:10.21227/aw6b-tg17.
