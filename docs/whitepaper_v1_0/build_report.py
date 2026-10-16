"""Build the single-model report from saved speaker predictions and settings.

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


def main():
    main_path = RUN / "main/arannamoy/speaker_predictions.csv"
    native_path = RUN / "native16k/arannamoy/speaker_predictions.csv"
    metrics_path = RUN / "main/arannamoy/metrics.json"
    main_predictions = pd.read_csv(main_path)
    native_predictions = pd.read_csv(native_path)
    recorded = json.loads(metrics_path.read_text())
    native_metrics_path = RUN / "native16k/arannamoy/metrics.json"
    native_recorded = json.loads(native_metrics_path.read_text())
    pooled = calculate_metrics(main_predictions)
    valid = main_predictions.speaker_id.is_unique and len(main_predictions) == 46
    valid = valid and set(main_predictions.fold) == set(range(5))
    valid = valid and bool((main_predictions.threshold == 0.5).all())
    for name, value in pooled.items():
        if abs(value - recorded["pooled"][name]) > 1e-10:
            print("ERROR: recomputed", name, "differs from the saved report.")
            valid = False
    if not valid:
        print("ERROR: expected 46 unique held-out speakers, five folds and matching metrics.")
        return
    native_valid = len(native_predictions) == 39 and native_predictions.speaker_id.is_unique
    native_valid = native_valid and bool((native_predictions.threshold == 0.5).all())
    sensitivity_sources = [
        (native_predictions, native_recorded["pooled"]),
        (main_predictions[~main_predictions.sheet_flag], recorded["without_sheet_flag"]["pooled"]),
    ]
    for predictions, expected in sensitivity_sources:
        for name, value in calculate_metrics(predictions).items():
            if abs(value - expected[name]) > 1e-10:
                native_valid = False
                print("ERROR: supporting-check metric differs:", name)
    if not native_valid:
        print("ERROR: expected 39 unique native-rate people and matching supporting metrics.")
        return

    generated = HERE / "generated"
    generated.mkdir(exist_ok=True)
    fold_lines = []
    count_lines = []
    training_lines = []
    fold_scores = []
    input_files = [main_path, native_path, metrics_path, native_metrics_path, RUN / "config.yaml", RUN / "protocol.json", RUN / "folds.csv"]
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
        input_files.extend([split_path, selection_path])
    spread = []
    for name in METRIC_NAMES:
        values = np.array([scores[name] for scores in fold_scores]) * 100
        spread.append("%.2f $\\pm$ %.2f" % (values.mean(), values.std(ddof=1)))
    fold_lines.append(r"\midrule Mean $\pm$ SD & -- & " + " & ".join(spread) + r" \\")
    (generated / "fold_rows.tex").write_text("\n".join(fold_lines) + "\n")
    (generated / "confusion_rows.tex").write_text("\n".join(count_lines) + "\n")
    (generated / "training_rows.tex").write_text("\n".join(training_lines) + "\n")
    (generated / "pooled_row.tex").write_text("46 & " + percentages(pooled) + r" \\" + "\n")
    sensitivity_lines = []
    for title, people in (
        ("Main evaluation", main_predictions),
        ("Flagged controls excluded", main_predictions[~main_predictions.sheet_flag]),
        ("Native 16 kHz retraining", native_predictions),
    ):
        scores = calculate_metrics(people)
        sensitivity_lines.append(title + " & " + str(len(people)) + " & " + percentages(scores) + r" \\")
    (generated / "sensitivity_rows.tex").write_text("\n".join(sensitivity_lines) + "\n")

    displayed = main_predictions.sort_values(["label", "probability"]).reset_index(drop=True)
    displayed["speaker_order"] = np.arange(1, len(displayed) + 1)
    for label, name in ((0, "controls"), (1, "patients")):
        displayed.loc[displayed.label == label, ["speaker_order", "probability"]].to_csv(generated / (name + ".csv"), index=False)

    latex = shutil.which("pdflatex")
    if latex is None:
        print("ERROR: pdflatex is required to build the PDF.")
        return
    build_directory = Path("/tmp/opencode/patchtst-whitepaper-v1_0-build")
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

    pdf = PROJECT / "docs/patchtst_technical_report_v1_0.pdf"
    shutil.copy2(build_directory / "patchtst_whitepaper.pdf", pdf)
    input_files.extend([AUDIT / "findings.json", AUDIT / "feature_rounding_resolution.json"])
    hashes = {}
    for path in input_files + [HERE / "patchtst_whitepaper.tex", Path(__file__)]:
        hashes[str(path.relative_to(PROJECT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(generated.iterdir()):
        hashes[str(path.relative_to(PROJECT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "document_version": "1.0",
        "document_date": "2026-10-12",
        "experiment_version": "v2.0",
        "member": "arannamoy",
        "primary_scenario": "main",
        "evidence_audit": "v2.0-audit.2",
        "primary_run": str(RUN.relative_to(PROJECT)),
        "report_pdf": str(pdf.relative_to(PROJECT)),
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "inputs_sha256": hashes,
        "metric_check": "All main pooled, fold and supporting-check metrics recomputed from saved speaker probabilities",
        "visual_style": "Black ink, white background, one column, serif text and vector diagrams",
    }
    (HERE / "report_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("PASS: saved predictions match all main fold and pooled metrics.")
    print("PDF:", pdf)
    print("Build log:", build_directory / "patchtst_whitepaper.log")


if __name__ == "__main__":
    main()
