"""Build the current official PatchTST report from its main OOF records.

Run from any directory with the project's Python environment. This builder
reads saved tables; it never trains a model or opens a recording.
"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from report_assets import build_assets


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RUN = PROJECT / "artifacts/runs/20261012T183328Z_f-all_cead14cc32f1"
AUDIT = PROJECT / "artifacts/runs/20261012T215648Z_v2.0-audit.2_2167119cc167"
METRIC_NAMES = ["f1", "recall", "specificity", "precision", "accuracy"]


def calculate_metrics(people):
    decisions = people.probability >= people.threshold
    patients = people.label == 1
    true_positive = int((decisions & patients).sum())
    false_negative = int((~decisions & patients).sum())
    true_negative = int((~decisions & ~patients).sum())
    false_positive = int((decisions & ~patients).sum())
    precision = 0.0
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    return {
        "speakers": len(people),
        "tp": true_positive,
        "fn": false_negative,
        "tn": true_negative,
        "fp": false_positive,
        "f1": 2 * true_positive / (2 * true_positive + false_positive + false_negative),
        "recall": true_positive / (true_positive + false_negative),
        "specificity": true_negative / (true_negative + false_positive),
        "precision": precision,
        "accuracy": (true_positive + true_negative) / len(people),
    }


def percentages(scores):
    return " & ".join("%.2f" % (100 * scores[name]) for name in METRIC_NAMES)


def check_oof_records(predictions):
    """Follow each speaker's score back to the correct held-out fold."""
    folds_path = RUN / "folds.csv"
    manifest_path = RUN / "main/arannamoy/manifest.csv"
    folds = pd.read_csv(folds_path)
    manifest = pd.read_csv(manifest_path)
    speaker_columns = ["speaker_id", "label", "fold"]
    expected = folds[speaker_columns].sort_values("speaker_id").reset_index(drop=True)
    observed = predictions[speaker_columns].sort_values("speaker_id").reset_index(drop=True)
    if len(predictions) != 46 or not predictions.speaker_id.is_unique:
        print("ERROR: expected exactly one OOF row for each of 46 speakers.")
        return
    if not observed.equals(expected) or set(folds.fold) != set(range(5)):
        print("ERROR: expected OOF identities, labels and folds to match frozen folds.csv.")
        return
    manifest_people = manifest[speaker_columns].drop_duplicates()
    manifest_people = manifest_people.sort_values("speaker_id").reset_index(drop=True)
    if len(manifest) != 773 or not manifest.path.is_unique or not manifest_people.equals(expected):
        print("ERROR: expected 773 unique recordings with the frozen speaker labels and folds.")
        return
    if not predictions.probability.between(0, 1).all() or not (predictions.threshold == 0.5).all():
        print("ERROR: expected finite probabilities between 0 and 1 and fixed threshold 0.5.")
        return
    if not (predictions.prediction == (predictions.probability >= 0.5).astype(int)).all():
        print("ERROR: expected saved decisions to follow the fixed 0.5 threshold.")
        return

    input_files = [folds_path, manifest_path]
    all_people = set(folds.speaker_id)
    checks = []
    for fold in range(5):
        directory = RUN / "main/arannamoy" / ("fold_" + str(fold))
        split = json.loads((directory / "split.json").read_text())
        for name in ("outer_train", "outer_test", "inner_train", "inner_validation"):
            if len(split[name]) != len(set(split[name])):
                print("ERROR: duplicate speaker in fold", fold, name)
                return
        train = set(split["outer_train"])
        test = set(split["outer_test"])
        inner_train = set(split["inner_train"])
        validation = set(split["inner_validation"])
        people = predictions[predictions.fold == fold]
        if test != set(people.speaker_id) or train != all_people - test:
            print("ERROR: fold", fold, "must test its assigned people and train on their complement.")
            return
        if (inner_train & validation) or (inner_train | validation) != train:
            print("ERROR: fold", fold, "inner groups must be disjoint and cover only outer-training people.")
            return
        for group in (train, test, inner_train, validation):
            if set(folds.loc[folds.speaker_id.isin(group), "label"]) != {0, 1}:
                print("ERROR: expected both classes in every training, validation and test group.")
                return

        fold_predictions = pd.read_csv(directory / "speaker_predictions.csv")
        fold_predictions = fold_predictions.sort_values("speaker_id").reset_index(drop=True)
        # The pooled table adds sheet_flag metadata after joining fold outputs.
        prediction_columns = ["speaker_id", "label", "probability", "threshold", "prediction", "fold"]
        pooled_rows = people[prediction_columns].sort_values("speaker_id").reset_index(drop=True)
        if not fold_predictions.equals(pooled_rows):
            print("ERROR: pooled OOF rows must equal their own held-out fold's saved speaker rows.")
            return
        windows = pd.read_csv(directory / "window_predictions.csv")
        file_columns = ["path", "speaker_id", "label", "task"]
        observed_files = windows[file_columns].drop_duplicates().sort_values("path").reset_index(drop=True)
        expected_files = manifest.loc[manifest.fold == fold, file_columns].sort_values("path").reset_index(drop=True)
        if not observed_files.equals(expected_files) or not windows.probability.between(0, 1).all():
            print("ERROR: fold", fold, "window scores must cover exactly its held-out manifest recordings.")
            return
        file_means = windows.groupby(["speaker_id", "path"])["probability"].mean()
        speaker_means = file_means.groupby("speaker_id").mean()
        speaker_means = speaker_means.reindex(fold_predictions.speaker_id)
        difference = np.abs(speaker_means.to_numpy() - fold_predictions.probability.to_numpy())
        # Saved predictions used float32; reading CSV values uses float64 here.
        if difference.max() > 1e-6:
            print("ERROR: fold", fold, "two-stage averaging differs by", difference.max(), "expected at most 1e-6.")
            return
        if not np.array_equal(speaker_means.to_numpy() >= 0.5, fold_predictions.prediction.to_numpy() == 1):
            print("ERROR: reconstructed speaker decisions must match the saved OOF decisions.")
            return

        selection = json.loads((directory / "selection.json").read_text())
        history = json.loads((directory / "selection_history.json").read_text())
        refit = json.loads((directory / "refit_history.json").read_text())
        best_loss = float("inf")
        selected_epoch = 0
        for row in history:
            if row["val_speaker_loss"] < best_loss - 0.0001:
                best_loss = row["val_speaker_loss"]
                selected_epoch = row["epoch"]
        if selected_epoch != selection["selected_epochs"] or len(refit) != selected_epoch:
            print("ERROR: fold", fold, "epoch selection and refit length must match validation history.")
            return
        checks.append({
            "fold": fold,
            "outer_train_speakers": len(train),
            "inner_train_speakers": len(inner_train),
            "inner_validation_speakers": len(validation),
            "test_speakers": len(test),
            "test_recordings": len(expected_files),
            "speaker_overlap": len(train & test),
            "pooled_rows_match_held_out_fold": True,
            "maximum_aggregation_probability_difference": float(difference.max()),
            "selected_epochs": selected_epoch,
            "passed": True,
        })
        for name in ("split.json", "selection.json", "selection_history.json", "refit_history.json",
                     "speaker_predictions.csv", "window_predictions.csv", "checkpoint.pt"):
            input_files.append(directory / name)

    resolution_path = AUDIT / "feature_rounding_resolution.json"
    resolution = json.loads(resolution_path.read_text())
    main_replays = []
    for replay in resolution["raw_audio_model_replays"]:
        if replay["scenario"] == "main":
            main_replays.append(replay)
    if len(main_replays) != 5 or {row["fold"] for row in main_replays} != set(range(5)):
        print("ERROR: expected completed audit evidence for all five main fresh-feature replays.")
        return
    if not all(row["passed"] and row["decisions_match"] for row in main_replays):
        print("ERROR: expected every main fresh-feature replay to pass.")
        return
    if resolution["raw_recordings_recomputed"] != 773 or not resolution["all_773_features_bitwise_match_with_one_blas_thread"]:
        print("ERROR: expected completed audit evidence for 773 bitwise-matching feature arrays.")
        return
    input_files.append(resolution_path)
    record = {
        "scope": "v2.0 / arannamoy / main",
        "unique_oof_speakers": len(predictions),
        "recordings": len(manifest),
        "threshold": 0.5,
        "fold_checks_recomputed_by_report_builder": checks,
        "prior_raw_audio_evidence": str(resolution_path.relative_to(PROJECT)),
        "prior_main_raw_audio_replays": main_replays,
        "interpretation": "Speaker-disjoint and leakage-checked; recording-condition confounding remains unresolved.",
    }
    return record, input_files


def main():
    main_path = RUN / "main/arannamoy/speaker_predictions.csv"
    metrics_path = RUN / "main/arannamoy/metrics.json"
    main_predictions = pd.read_csv(main_path)
    recorded = json.loads(metrics_path.read_text())
    verification = check_oof_records(main_predictions)
    if verification is None:
        return
    oof_checks, input_files = verification
    pooled = calculate_metrics(main_predictions)
    valid = True
    for name, value in pooled.items():
        if abs(value - recorded["pooled"][name]) > 1e-10:
            print("ERROR: recomputed", name, "differs from the saved report.")
            valid = False
    if not valid:
        print("ERROR: expected 46 unique held-out speakers, five folds and matching metrics.")
        return
    generated = HERE / "generated"
    generated.mkdir(exist_ok=True)
    fold_lines = []
    count_lines = []
    training_lines = []
    fold_scores = []
    input_files.extend([main_path, metrics_path, RUN / "config.yaml", RUN / "protocol.json"])
    for fold, people in main_predictions.groupby("fold"):
        scores = calculate_metrics(people)
        saved = recorded["folds"][int(fold)]
        if any(abs(scores[name] - saved[name]) > 1e-10 for name in scores):
            print("ERROR: fold", fold, "does not match the saved metrics.")
            return
        fold_scores.append(scores)
        fold_lines.append(str(fold) + " & " + str(len(people)) + " & " + percentages(scores) + r" \\")
        count_lines.append(" & ".join(str(value) for value in (fold, scores["tp"], scores["fn"], scores["tn"], scores["fp"])) + r" \\")
        directory = RUN / "main/arannamoy" / ("fold_" + str(fold))
        split_path = directory / "split.json"
        selection_path = directory / "selection.json"
        split = json.loads(split_path.read_text())
        selection = json.loads(selection_path.read_text())
        training_lines.append(" & ".join(str(value) for value in (
            fold, len(split["inner_train"]), len(split["inner_validation"]),
            len(split["outer_train"]), len(split["outer_test"]), selection["selected_epochs"],
        )) + r" \\")
    spread = []
    for name in METRIC_NAMES:
        values = np.array([scores[name] for scores in fold_scores]) * 100
        if abs(values.mean() / 100 - recorded["mean_sd"][name]["mean"]) > 1e-10:
            print("ERROR: recomputed fold mean differs for", name)
            return
        if abs(values.std(ddof=1) / 100 - recorded["mean_sd"][name]["sd"]) > 1e-10:
            print("ERROR: recomputed sample SD differs for", name)
            return
        spread.append("%.2f $\\pm$ %.2f" % (values.mean(), values.std(ddof=1)))
    fold_lines.append(r"\midrule Mean $\pm$ SD & -- & " + " & ".join(spread) + r" \\")
    (generated / "fold_rows.tex").write_text("\n".join(fold_lines) + "\n")
    (generated / "confusion_rows.tex").write_text("\n".join(count_lines) + "\n")
    (generated / "training_rows.tex").write_text("\n".join(training_lines) + "\n")
    (generated / "pooled_row.tex").write_text("46 & " + percentages(pooled) + r" \\" + "\n")
    (generated / "oof_checks.json").write_text(json.dumps(oof_checks, indent=2) + "\n")

    displayed = main_predictions.sort_values(["label", "probability"]).reset_index(drop=True)
    displayed["speaker_order"] = np.arange(1, len(displayed) + 1)
    for label, name in ((0, "controls"), (1, "patients")):
        displayed.loc[displayed.label == label, ["speaker_order", "probability"]].to_csv(generated / (name + ".csv"), index=False)

    if not build_assets(RUN, generated, main_predictions, recorded):
        return

    latex = shutil.which("pdflatex")
    if latex is None:
        print("ERROR: pdflatex is required to build the PDF.")
        return
    build_directory = Path("/tmp/opencode/patchtst-whitepaper-v1_2-build")
    build_directory.mkdir(parents=True, exist_ok=True)
    for pass_number in range(1, 4):
        command = [latex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "-output-directory=" + str(build_directory), "patchtst_whitepaper.tex"]
        completed = subprocess.run(command, cwd=HERE, capture_output=True, text=True)
        (build_directory / ("pass_" + str(pass_number) + ".txt")).write_text(completed.stdout + completed.stderr)
        if completed.returncode != 0:
            print("ERROR: PDF compilation failed on pass", pass_number)
            print(completed.stdout[-6000:])
            print("Full output:", build_directory)
            return

    pdf = PROJECT / "docs/patchtst_technical_report_v1_2.pdf"
    shutil.copy2(build_directory / "patchtst_whitepaper.pdf", pdf)
    input_files.append(AUDIT / "findings.json")
    hashes = {}
    for path in input_files + [HERE / "patchtst_whitepaper.tex", Path(__file__), HERE / "report_assets.py", HERE / "requirements.txt"]:
        hashes[str(path.relative_to(PROJECT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(generated.rglob("*")):
        if path.is_file():
            hashes[str(path.relative_to(PROJECT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "document_version": "1.2",
        "parent_document_version": "1.1",
        "document_date": "2026-10-13",
        "experiment_version": "v2.0",
        "member": "arannamoy",
        "primary_scenario": "main",
        "evidence_audit": "v2.0-audit.2",
        "primary_run": str(RUN.relative_to(PROJECT)),
        "report_pdf": str(pdf.relative_to(PROJECT)),
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "inputs_sha256": hashes,
        "metric_check": "Main pooled metrics, all five folds, fold means and sample SD recomputed from saved speaker probabilities",
        "oof_check": "Frozen speaker assignments, inner/outer separation, fold-to-pooled identity, two-stage aggregation, fixed threshold and epoch records verified",
        "scope": "Current official PatchTST result, main cohort only",
        "visual_style": "Black ink, white background, one column, serif text and vector diagrams",
        "extension": "Exact file membership for every fold, 20 partition CSVs, six Matplotlib figures and offline Plotly dashboard",
    }
    (HERE / "report_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("PASS: 46 unique OOF speakers; all five folds have disjoint inner-training, validation and test groups.")
    print("PASS: held-out fold rows, two-stage averaging, threshold and epoch records agree.")
    print("PASS: main fold, pooled, mean and sample-SD metrics match the saved results.")
    print("PDF:", pdf)
    print("Build log:", build_directory / "patchtst_whitepaper.log")


if __name__ == "__main__":
    main()
