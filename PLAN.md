# Pipeline checklist

PD versus elderly healthy control. Speaker-level score. Young controls stay out of the label.

Do the steps in order. Do not start a later phase until that phase's check passes. Tick a box only when the "done when" line is true.

Progress: 0 / 100

## A. Freeze the decision

- [ ] **1.** Write the question in one sentence: PD versus elderly healthy control, from speech. Done when it does not mention the young group.
- [ ] **2.** Write the score you will report: speaker-level precision, recall, specificity, accuracy, F1. Done when window accuracy is listed as a debug number only.
- [ ] **3.** Record the three citations: VoxTester 2016, IEEE Access 2017, DataPort DOI `10.21227/aw6b-tg17`. Done when they are in the README, not only in a notebook.
- [ ] **4.** Point `config/base.yaml` at the two existing group folders. Done when the young folder is absent from `groups`.
- [ ] **5.** Set `target_sr: 16000`, `tasks: all`, `n_splits: 5`, `seed: 42`, `group_by: speaker`. Done when no other file contains these numbers.
- [ ] **6.** Add `.gitignore` for `data/`, `artifacts/`, `.venv/`, `__pycache__/`, the corpus, and the zip. Done when `git status` does not show a WAV.
- [ ] **7.** Leave the WAV tree where it is. Done when no script writes inside it.
- [ ] **8.** Move `scripts/exp.ipynb` to `notebooks/`. Done when the notebook is marked "plots only".

**Check:** the README states the cohort and the metric, and the config names exactly two groups.

## B. Package skeleton

- [ ] **9.** `src/patchtst/config.py` loads the yaml. Done when a missing key raises and does not default silently.
- [ ] **10.** Add `src/patchtst/data/__init__.py`. Done when the package imports.
- [ ] **11.** Add `src/patchtst/model/__init__.py`. Done when it is empty. Do not put a network here yet.
- [ ] **12.** Add `scripts/build_manifest.py`. Done when it loads config and exits. No parsing yet.
- [ ] **13.** Add `scripts/cache_features.py`. Done when it loads config and exits.
- [ ] **14.** Add `scripts/train.py`. Done when it loads config and exits.
- [ ] **15.** Add `scripts/evaluate.py`. Done when it loads config and exits.
- [ ] **16.** Add `tests/` and a runner. Done when one test runs and fails on purpose, so you know the runner works.

**Check:** `uv run` can import `patchtst` and the four scripts exit 0.

## C. File index

Assign to `src/patchtst/data/index.py`.

- [ ] **17.** `cohort_of` matches `Elderly Healthy Control` and `People with Parkinson`. Done when it does not search for the substring `Parkinson` alone.
- [ ] **18.** Test an elderly path. Done when the label is 0.
- [ ] **19.** Test a PD path. Done when the label is 1.
- [ ] **20.** Test a path whose only `Parkinson` is the dataset root. Done when the function refuses it or returns elderly, never PD.
- [ ] **21.** Parse a 2017 name: `DDMMYYYY` plus `HHMM`. Done when date and token are separate.
- [ ] **22.** Parse a 2016 name: `DDMMYY` plus `HHMM`. Done when the year is 2016, not 16.
- [ ] **23.** Map `PR11` to `PR1`. Done when that file is not a new task.
- [ ] **24.** Strip a trailing double dot before parsing. Done when Alberto R's word-list file parses.
- [ ] **25.** Remove spaces inside the token. Done when `VDIET AS47M` becomes one token.
- [ ] **26.** Match the longest task prefix. Done when `FB1` is not read as `B1`.
- [ ] **27.** Walk only `cfg["data"]["groups"]`. Done when young files never enter the table.
- [ ] **28.** Read rate, channels, subtype, and duration with `soundfile.info`. Done when you do not load the waveform to index it.

**Check:** 786 rows. 437 PD, 349 elderly. Every file is mono PCM 16-bit. About 160 PD files are 44100 Hz. Every elderly file is 16000 Hz.

## D. Speaker identity

Still `index.py`.

- [ ] **29.** Replace a folder's odd token with the modal token. Done when Antonia G's `VI2` `63F` becomes `52F` and the file is flagged `token_typo`.
- [ ] **30.** Test that flag. Done when exactly one file is flagged.
- [ ] **31.** Union folders with the same cohort and the same token. Done when `VSIPTIOZ46M` is one person.
- [ ] **32.** Union folders with the same cohort, display name, and birth-year/sex. Done when Nicola S and Roberto R each collapse.
- [ ] **33.** Do not union across cohort. Done when an elderly token cannot merge with a PD token.
- [ ] **34.** Build `speaker_id` as cohort plus the surviving token. Done when it does not contain the folder path.
- [ ] **35.** Test the count. Done when there are 47 speakers: 25 PD, 22 elderly.
- [ ] **36.** Test the three repeat people. Done when each has two folders and both sample rates.
- [ ] **37.** Test the other Vito S. Done when the `47M` token is a different `speaker_id`.
- [ ] **38.** Store `native_sr`, `duration_s`, `task`, and `session_date` on every row. Done when a later stage does not re-parse the filename.
- [ ] **39.** Write `data/manifest.csv` and `data/speakers.csv` from `build_manifest.py`. Done when re-running the script does not change the row count.
- [ ] **40.** Print the three linked speakers at the end of that script. Done when you can see the merge without opening the CSV.

**Check:** `speaker_id.nunique() == 47`, and no young name appears.

## E. Spreadsheet join

Assign to `src/patchtst/data/demographics.py`, called by the manifest script.

- [ ] **41.** Read `TAB 5.xlsx` and `Tab 3.xlsx`, not `FILE CODES.xlsx`. Done when sex and age columns are found by name.
- [ ] **42.** Require spreadsheet sex to equal the token sex. Done when a mismatch fails the build.
- [ ] **43.** Match `VITO L`, `LISCO G`, and `SUMMO L` to the swapped sheet rows. Done when all 22 elderly folders match.
- [ ] **44.** Match `Mario B` to the only Mario row, age within one year. Done when he is not left unmatched.
- [ ] **45.** Store sheet age, and also `session_year - birth_year`. Done when the two ages differ by at most 1 on every speaker.
- [ ] **46.** Flag the four starred elderly names. Done when `sheet_flag` is true only for Agnese P, Gilda C, Giovanna G, and Vitantonio D.
- [ ] **47.** Keep CPS columns, named as metadata. Done when no training function accepts them as input.
- [ ] **48.** Fail the build if any speaker is unmatched. Done when you do not fill age with a silent null.

**Check:** 47 of 47 speakers matched. Zero sex mismatches.

## F. Splits

Assign to `src/patchtst/data/splits.py`.

- [ ] **49.** Build the fold table from one row per `speaker_id`, not per file. Done when a person with 16 files is one unit.
- [ ] **50.** Use stratified 5-fold, shuffle, seed 42. Done when you do not use plain `GroupKFold` on a PD-then-control sort.
- [ ] **51.** Copy `fold` onto every file of that speaker. Done when the column is the test fold.
- [ ] **52.** Test that no speaker appears in two folds. Done when the test fails if a merge was forgotten.
- [ ] **53.** Test that every fold contains both labels. Done when each test fold has about 5 PD and 4 or 5 elderly speakers.
- [ ] **54.** Write `data/folds.csv`. Done when it has 47 rows.
- [ ] **55.** Add a filter flag for `sheet_flag`, default off. Done when the main split still includes those four.
- [ ] **56.** Add a filter flag for `native_sr == 16000`, default off. Done when this is the microphone sensitivity run, not the main run.

**Check:** both filter tables can be built without refitting the speaker ids.

## G. Audio

Assign to `src/patchtst/data/audio.py` and `scripts/cache_features.py`.

- [ ] **57.** Read PCM as float32. Done when you do not divide by `max(abs(y))`.
- [ ] **58.** Resample to `target_sr` only when the file rate differs. Done when 16 kHz files are not resampled twice.
- [ ] **59.** Test one 44.1 kHz file. Done when mel frames are about `duration_s * 100`, not `duration_s * 441`.
- [ ] **60.** Compute log-mel with `n_mels=64`, `n_fft=512`, window 400, hop 160. Done when the shape is `(64, time)`.
- [ ] **61.** Use a fixed dB reference of 1.0. Done when a quiet file stays quieter than a loud file.
- [ ] **62.** Name each cache file by a hash of the WAV path. Done when changing the task filter cannot attach the wrong array.
- [ ] **63.** Skip a cache file only if it exists and its first dimension is 64. Done when a changed `n_mels` recomputes it.
- [ ] **64.** Write `feature_path` back into the manifest. Done when 786 paths exist.
- [ ] **65.** Record feature settings in the config, not in `audio.py`. Done when changing hop means editing yaml only.
- [ ] **66.** Time the cache once and write the elapsed seconds into the README. Done when you know whether training can recompute or must read the cache.

**Check:** a 73-second 44.1 kHz passage has about 7300 frames. A 6-second `/pa/` has about 600.

## H. Windows

Assign to `src/patchtst/data/dataset.py`. Do not start the network before this check.

- [ ] **67.** Define one window as 4.0 s, 400 frames, hop 2.0 s. Done when both numbers come from config.
- [ ] **68.** Edge-pad a clip shorter than 400 frames. Done when the shortest vowel still yields one window.
- [ ] **69.** Training takes at most 8 windows, evenly spaced. Done when a 250-second reading does not yield 100 windows.
- [ ] **70.** Testing takes every window. Done when the cap is not applied in `evaluate`.
- [ ] **71.** Each window inherits the file label and the `speaker_id`. Done when the dataset never receives a folder name.
- [ ] **72.** Test the batch shape. Done when it is `(n, 64, 400)`, which is `(batch, n_vars, seq_len)`.
- [ ] **73.** Refuse a batch that mixes two folds. Done when the dataset is constructed from one split.
- [ ] **74.** Add a task filter that reads `tasks` from config. Done when `all` and a vowel-only list both work without reindexing speakers.

**Check:** one long PD file contributes 8 training windows and more than 8 test windows.

## I. Model

Assign to `src/patchtst/model/patchtst.py`. Add the training library to `pyproject.toml` here, not before.

- [ ] **75.** Add the training dependency. Done when the data package still imports without it.
- [ ] **76.** Implement patching along time, patch length and stride from config. Done when channels stay independent.
- [ ] **77.** Put a classification head on the patch encoder. Done when the output is one logit, not a forecast.
- [ ] **78.** Fix the input contract in a docstring. Done when it says `(batch, 64, 400)` in, one probability out.
- [ ] **79.** Test a forward pass on a dummy batch. Done when the shape is `(batch, 1)` and the graph runs on CPU.
- [ ] **80.** Save the config values that define the network next to the checkpoint later. Done when patch length is not inferred from a weight file.
- [ ] **81.** Do not feed sex, age, CPS, or `native_sr` into the network. Done when the forward signature accepts only the spectrogram window.
- [ ] **82.** Count parameters and write the number down. Done when you know the model is not larger than 47 speakers can support.

**Check:** a random batch from step 72 passes through the network.

## J. Training

Assign to `src/patchtst/train.py` and `scripts/train.py`.

- [ ] **83.** Train one fold. Done when test speakers are excluded before the first batch.
- [ ] **84.** Write the run to `artifacts/runs/<date>_f<fold>_<config-hash>/`. Done when a second run does not overwrite the first.
- [ ] **85.** Copy the yaml into that folder as `config.yaml`. Done when the copy is the config that was used, not a later edit.
- [ ] **86.** Save the checkpoint there. Done when the path is in a small `run.json`.
- [ ] **87.** Log train loss per epoch in that folder. Done when the notebook is not the log.
- [ ] **88.** Use class weights or a sampler. Done when the loss is not left raw despite 437 PD files and 349 elderly files.
- [ ] **89.** Stop on the validation speaker loss, not on window accuracy. Done when early stopping cannot see the test fold.
- [ ] **90.** Repeat for folds 0–4 with the same seed. Done when five run directories exist.

**Check:** opening any run directory tells you the fold, the seed, and the config hash without reading code.

## K. Evaluation

Assign to `src/patchtst/evaluate.py` and `src/patchtst/metrics.py`. Training without this phase is not a result.

- [ ] **91.** Score every test window. Done when `window_scores.csv` has `speaker_id`, `path`, `task`, and probability.
- [ ] **92.** Average probabilities by file, then by `speaker_id`. Done when `speaker_scores.csv` exists.
- [ ] **93.** Compute precision, recall, specificity, accuracy, and F1 on speaker scores. Done when `metrics.json` exists.
- [ ] **94.** Pool the five test folds and compute the same five numbers once. Done when this pooled number is the one you quote.
- [ ] **95.** Also report the mean and spread across the five folds. Done when you do not quote the best fold.
- [ ] **96.** Assert that the speaker count in `speaker_scores.csv` equals the test speakers for that fold. Done when a dropped speaker fails the run.

**Check:** the quoted F1 is computed from about 9 or 10 speakers per fold, not from thousands of windows.

## L. The two runs that tell you if it is real

- [ ] **97.** Rerun evaluation with `sheet_flag` speakers removed. Done when you record whether the pooled F1 moves.
- [ ] **98.** Rerun training and evaluation on `native_sr == 16000` only. Done when the seven PD speakers who exist only at 44.1 kHz are gone.
- [ ] **99.** If the 16 kHz-only F1 collapses, write that in the run note. Done when you do not publish the mixed-rate number as the disease result.
- [ ] **100.** Plot from `notebooks/` by reading `speaker_scores.csv` only. Done when the notebook cannot see a WAV path except through that CSV.

**Check:** three numbers exist side by side: main pooled F1, F1 without the four flagged controls, and F1 on 16 kHz only. The third number is the one you trust.
