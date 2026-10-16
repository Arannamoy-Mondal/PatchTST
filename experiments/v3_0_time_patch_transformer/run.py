"""Version v3.0: compare a new time-patch transformer with the static baseline.

The speaker folds and log-mel files come from completed experiment v2.0.
This script writes only to a new run directory.
"""

import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from patchtst.data.audio import slice_windows
from patchtst.metrics import classification_scores, speaker_table
from model import TimePatchTransformer

PROJECT = Path(__file__).resolve().parents[2]
SOURCE_RUN = PROJECT / "artifacts/runs/20261012T183328Z_f-all_cead14cc32f1"
EXPERIMENT_NAME = "v3.0"
VARIANTS = ("raw_logmel", "time_centered_logmel")
THRESHOLD = 0.5
SETTINGS = {
    "width": 32,
    "layers": 1,
    "heads": 4,
    "dropout": 0.2,
    "epochs": 12,
    "patience": 4,
    "batch_size": 16,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "validation_fraction": 0.2,
    "threads": 4,
    "seed": 42,
    "threshold": THRESHOLD,
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def prepare_spectrogram(spectrogram, variant):
    """Optionally remove the recording's average spectrum before windowing."""
    values = spectrogram.astype(np.float32)
    if variant == "time_centered_logmel":
        values = values - values.mean(axis=1, keepdims=True)
    return values


def windows_for(recordings, variant, limit):
    """Cut cached recordings into model windows without using another person's audio."""
    groups, labels, speakers, paths = [], [], [], []
    for recording in recordings.itertuples(index=False):
        spectrogram = np.load(recording.feature_path)
        spectrogram = prepare_spectrogram(spectrogram, variant)
        windows = slice_windows(spectrogram, {
            "features": {
                "hop_length": 160,
                "window_sec": 4.0,
                "hop_sec": 2.0,
                "max_windows_per_file": 8,
            },
            "data": {"target_sr": 16000},
        }, limit=limit)
        groups.append(windows)
        labels.extend([recording.label] * len(windows))
        speakers.extend([recording.speaker_id] * len(windows))
        paths.extend([recording.path] * len(windows))
    return (
        np.concatenate(groups),
        np.asarray(labels, dtype=np.float32),
        np.asarray(speakers),
        np.asarray(paths),
    )


def balanced_weights(labels, speakers, paths):
    rows = pd.DataFrame({"label": labels, "speaker_id": speakers, "path": paths})
    windows_per_file = rows.groupby("path").size()
    files_per_person = rows.groupby("speaker_id").path.nunique()
    people_per_class = rows.drop_duplicates("speaker_id").groupby("label").size()
    weights = 1.0 / (
        rows.path.map(windows_per_file)
        * rows.speaker_id.map(files_per_person)
        * rows.label.map(people_per_class)
    )
    return (weights / weights.mean()).to_numpy(dtype=np.float32)


def predict(model, windows):
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(windows), SETTINGS["batch_size"]):
            batch = torch.from_numpy(windows[start:start + SETTINGS["batch_size"]])
            logits = model(batch)
            probabilities.append(torch.sigmoid(logits).squeeze(1).numpy())
    return np.concatenate(probabilities)


def fit(training, validation, seed, epoch_count):
    seed_everything(seed)
    windows, labels, speakers, paths = training
    weights = balanced_weights(labels, speakers, paths)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(windows), torch.from_numpy(labels), torch.from_numpy(weights)),
        batch_size=SETTINGS["batch_size"],
        shuffle=True,
        generator=generator,
    )
    model = TimePatchTransformer(
        SETTINGS["width"], SETTINGS["layers"], SETTINGS["heads"], SETTINGS["dropout"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"]
    )
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    history = []
    best_loss = float("inf")
    best_epoch = 1
    waited = 0
    started = time.monotonic()
    for epoch in range(1, epoch_count + 1):
        model.train()
        total = 0.0
        for batch_windows, batch_labels, batch_weights in loader:
            optimizer.zero_grad()
            loss = (loss_function(model(batch_windows).squeeze(1), batch_labels) * batch_weights).mean()
            if not torch.isfinite(loss):
                print("ERROR: training loss is not finite.")
                return None
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch_windows)
        record = {"epoch": epoch, "train_loss": total / len(windows)}
        if validation is not None:
            features, true_labels, ids, file_paths = validation
            people = speaker_table(pd.DataFrame({
                "speaker_id": ids, "path": file_paths, "label": true_labels,
                "probability": predict(model, features),
            }))
            probability = np.clip(people.probability.to_numpy(), 1e-7, 1 - 1e-7)
            target = people.label.to_numpy()
            value = float(-(target * np.log(probability) + (1 - target) * np.log(1 - probability)).mean())
            record["validation_speaker_loss"] = value
            if value < best_loss - 0.0001:
                best_loss, best_epoch, waited = value, epoch, 0
            else:
                waited += 1
        history.append(record)
        print("Epoch", epoch, record, "elapsed", round(time.monotonic() - started, 1), flush=True)
        if validation is not None and waited >= SETTINGS["patience"]:
            break
    return model, history, best_epoch


def train_fold(training_files, test_files, variant, fold):
    seed = SETTINGS["seed"] + fold
    people = training_files.groupby("speaker_id", as_index=False).agg(label=("label", "first"))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=SETTINGS["validation_fraction"], random_state=seed)
    train_positions, validation_positions = next(splitter.split(people, people.label))
    train_ids = set(people.iloc[train_positions].speaker_id)
    validation_ids = set(people.iloc[validation_positions].speaker_id)
    test_ids = set(test_files.speaker_id)
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        print("ERROR: a person appears in more than one split.")
        return None
    selection = fit(
        windows_for(training_files[training_files.speaker_id.isin(train_ids)], variant, True),
        windows_for(training_files[training_files.speaker_id.isin(validation_ids)], variant, False),
        seed,
        SETTINGS["epochs"],
    )
    if selection is None:
        return None
    unused_model, history, selected_epochs = selection
    del unused_model
    print("Refit epochs:", selected_epochs, flush=True)
    refit = fit(windows_for(training_files, variant, True), None, seed, selected_epochs)
    if refit is None:
        return None
    model, refit_history, unused_epoch = refit
    features, labels, ids, paths = windows_for(test_files, variant, False)
    people_scores = speaker_table(pd.DataFrame({
        "speaker_id": ids, "path": paths, "label": labels.astype(int),
        "probability": predict(model, features),
    }))
    people_scores["threshold"] = THRESHOLD
    people_scores["prediction"] = (people_scores.probability >= THRESHOLD).astype(int)
    people_scores["fold"] = fold
    return {
        "model": model,
        "history": history,
        "refit_history": refit_history,
        "selected_epochs": selected_epochs,
        "scores": people_scores,
        "split": {
            "inner_train": sorted(train_ids),
            "inner_validation": sorted(validation_ids),
            "outer_train": sorted(train_ids | validation_ids),
            "outer_test": sorted(test_ids),
        },
    }


def add_extra_scores(scores):
    predicted = (scores.probability >= scores.threshold).astype(int)
    result = classification_scores(scores, THRESHOLD)
    result["auroc"] = float(roc_auc_score(scores.label, scores.probability))
    result["average_precision"] = float(average_precision_score(scores.label, scores.probability))
    result["mcc"] = float(matthews_corrcoef(scores.label, predicted))
    result["balanced_accuracy"] = (result["recall"] + result["specificity"]) / 2
    return result


def summarize(scores):
    folds = []
    for fold, rows in scores.groupby("fold"):
        folds.append({"fold": int(fold), **add_extra_scores(rows)})
    variation = {}
    for name in ("f1", "recall", "specificity", "precision", "accuracy", "auroc"):
        values = np.array([row[name] for row in folds])
        variation[name] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1))}
    return {"pooled": add_extra_scores(scores), "folds": folds, "mean_sd": variation}


def static_baseline(recordings):
    """Repeat the fast average-spectrum control on these exact folds."""
    rows = []
    for recording in recordings.itertuples(index=False):
        spectrum = np.load(recording.feature_path).mean(axis=1)
        row = {"speaker_id": recording.speaker_id, "label": recording.label, "fold": recording.fold}
        for index, value in enumerate(spectrum):
            row["band_%02d" % index] = float(value)
        rows.append(row)
    files = pd.DataFrame(rows)
    columns = ["band_%02d" % index for index in range(64)]
    people = files.groupby("speaker_id", as_index=False)[columns].mean()
    meta = files.groupby("speaker_id", as_index=False).agg(label=("label", "first"), fold=("fold", "first"))
    people = people.merge(meta, on="speaker_id")
    probability = np.zeros(len(people))
    for fold in range(5):
        train = people.fold != fold
        test = people.fold == fold
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=1.0, solver="liblinear", max_iter=5000, random_state=42),
        )
        model.fit(people.loc[train, columns], people.loc[train, "label"])
        probability[test.to_numpy()] = model.predict_proba(people.loc[test, columns])[:, 1]
    people["probability"] = probability
    people["threshold"] = THRESHOLD
    people["prediction"] = (people.probability >= THRESHOLD).astype(int)
    return people


def run_variant(recordings, variant, destination):
    destination.mkdir(parents=True, exist_ok=True)
    tables = []
    for fold in range(5):
        print("RUN", variant, "fold", fold, flush=True)
        result = train_fold(recordings[recordings.fold != fold], recordings[recordings.fold == fold], variant, fold)
        if result is None:
            return None
        fold_directory = destination / ("fold_" + str(fold))
        fold_directory.mkdir(exist_ok=True)
        torch.save(result["model"].state_dict(), fold_directory / "checkpoint.pt")
        write_json(fold_directory / "history.json", result["history"])
        write_json(fold_directory / "refit_history.json", result["refit_history"])
        write_json(fold_directory / "split.json", result["split"])
        result["scores"].to_csv(fold_directory / "speaker_predictions.csv", index=False)
        tables.append(result["scores"])
    scores = pd.concat(tables, ignore_index=True)
    expected = set(recordings.speaker_id)
    if not scores.speaker_id.is_unique or set(scores.speaker_id) != expected:
        print("ERROR: predictions do not cover each person once.")
        return None
    flags = recordings.groupby("speaker_id").sheet_flag.first()
    scores["sheet_flag"] = scores.speaker_id.map(flags)
    scores.to_csv(destination / "speaker_predictions.csv", index=False)
    report = summarize(scores)
    report["without_sheet_flag"] = summarize(scores[~scores.sheet_flag.astype(bool)])
    report["recordings"] = len(recordings)
    write_json(destination / "metrics.json", report)
    print("COMPLETE", variant, report["pooled"], flush=True)
    return report


def create_run_directory():
    code = Path(__file__).read_bytes() + Path(__file__).with_name("model.py").read_bytes()
    folds = (SOURCE_RUN / "folds.csv").read_bytes()
    digest = hashlib.sha256(code + folds + json.dumps(SETTINGS, sort_keys=True).encode()).hexdigest()[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = PROJECT / "artifacts/runs" / (timestamp + "_v3.0_" + digest)
    destination.mkdir(parents=True)
    return destination


def main():
    torch.set_num_threads(SETTINGS["threads"])
    recordings = pd.read_csv(SOURCE_RUN / "arannamoy_feature_manifest.csv")
    recordings = recordings[recordings.fold.notna()].copy()
    recordings["fold"] = recordings.fold.astype(int)
    if recordings.speaker_id.nunique() != 46:
        print("ERROR: expected the frozen 46-person v2.0 cohort.")
        return
    run_directory = create_run_directory()
    write_json(run_directory / "settings.json", {
        "experiment": EXPERIMENT_NAME,
        "parent_run": str(SOURCE_RUN.relative_to(PROJECT)),
        "question": "Does a multivariate time-patch transformer beat the static-spectrum baseline, and does it still work after average spectra are removed?",
        "settings": SETTINGS,
        "variants": {
            "raw_logmel": "Original fixed-reference 64-band log-mel windows",
            "time_centered_logmel": "Same windows after subtracting each recording's per-band time average",
        },
        "not_a_replacement_for": "v2.0",
    })
    reports = {}
    for scenario, subset in (
        ("main", recordings),
        ("native16k", recordings[recordings.native_sr == 16000]),
    ):
        reports[scenario] = {}
        if scenario == "native16k" and subset.speaker_id.nunique() != 39:
            print("ERROR: native-rate subset does not have 39 people.")
            return
        scenario_directory = run_directory / scenario
        scenario_directory.mkdir(parents=True, exist_ok=True)
        baseline = static_baseline(subset)
        baseline.to_csv(scenario_directory / "static_baseline_predictions.csv", index=False)
        reports[scenario]["static_baseline"] = summarize(baseline)
        write_json(scenario_directory / "static_baseline_metrics.json", reports[scenario]["static_baseline"])
        for variant in VARIANTS:
            report = run_variant(subset, variant, run_directory / scenario / variant)
            if report is None:
                print("ERROR: experiment stopped.")
                return
            reports[scenario][variant] = report
    write_json(run_directory / "summary.json", reports)
    print("EXPERIMENT_DIRECTORY=" + str(run_directory), flush=True)


if __name__ == "__main__":
    main()
