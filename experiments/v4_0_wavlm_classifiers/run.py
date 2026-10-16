"""Prepare, extract, train and verify a separately versioned proposal experiment."""

import argparse
import importlib.metadata
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download
from sklearn.model_selection import StratifiedKFold

from classifiers import run_method
from common import METHODS, PROJECT, check_recordings, fingerprint, summarize, write_json
from features import extract_recordings, feature_table

HERE = Path(__file__).resolve().parent


def source_files():
    paths = list(HERE.glob("*.py")) + [HERE / "config.yaml", HERE / "requirements.txt"]
    paths += [PROJECT / "src/patchtst/data/audio.py", PROJECT / "src/patchtst/metrics.py"]
    return sorted(paths)


def export_splits(recordings, settings, directory):
    people = recordings[["speaker_id", "label", "fold"]].drop_duplicates().sort_values("speaker_id").reset_index(drop=True)
    for fold in range(settings["splits"]["n_splits"]):
        destination = directory / ("fold_" + str(fold))
        destination.mkdir(parents=True, exist_ok=True)
        training = people[people.fold != fold].reset_index(drop=True)
        testing = people[people.fold == fold]
        splitter = StratifiedKFold(n_splits=settings["inner_folds"], shuffle=True, random_state=settings["splits"]["seed"] + 100 + fold)
        inner = []
        for inner_fold, (fit_positions, validation_positions) in enumerate(splitter.split(training, training.label)):
            fit_ids = set(training.iloc[fit_positions].speaker_id)
            validation_ids = set(training.iloc[validation_positions].speaker_id)
            inner.append({"inner_fold": inner_fold, "train": sorted(fit_ids), "validation": sorted(validation_ids)})
            for role, ids in [("train", fit_ids), ("validation", validation_ids)]:
                recordings[recordings.speaker_id.isin(ids)].to_csv(destination / ("inner_" + str(inner_fold) + "_" + role + ".csv"), index=False)
        for role, ids in [("train", set(training.speaker_id)), ("test", set(testing.speaker_id))]:
            recordings[recordings.speaker_id.isin(ids)].to_csv(destination / ("outer_" + role + ".csv"), index=False)
        write_json(destination / "split.json", {"outer_train": sorted(training.speaker_id), "outer_test": sorted(testing.speaker_id), "inner": inner})


def prepare():
    settings = yaml.safe_load((HERE / "config.yaml").read_text())
    base_path = PROJECT / settings["base_config"]
    base = yaml.safe_load(base_path.read_text())
    settings["data"] = base["data"]
    settings["splits"] = base["splits"]
    settings["train"] = {"threshold": base["train"]["threshold"]}
    settings["inner_folds"] = base["comparison"]["inner_folds"]
    versions = {}
    for package in ["numpy", "pandas", "scikit-learn", "librosa", "soundfile", "torch", "transformers", "huggingface-hub", "xgboost-cpu"]:
        versions[package] = importlib.metadata.version(package)
    settings["dependency_versions"] = versions
    pretrained = {}
    for name in ["config.json", "preprocessor_config.json", "pytorch_model.bin"]:
        path = Path(hf_hub_download(settings["features"]["model"], name, revision=settings["features"]["revision"]))
        pretrained[name] = {"cache_path": str(path), "sha256": fingerprint(path)}
    settings["pretrained_files"] = pretrained
    parent = PROJECT / settings["parent_run"]
    eligible = pd.read_csv(parent / "manifest.csv")
    recordings = eligible[eligible.fold.notna()].copy().sort_values("path").reset_index(drop=True)
    recordings["fold"] = recordings.fold.astype(int)
    if not check_recordings(recordings, 46, 773):
        return None
    source_hashes = {path.relative_to(PROJECT).as_posix(): fingerprint(path) for path in source_files()}
    import hashlib
    digest = hashlib.sha256((json.dumps(settings, sort_keys=True) + json.dumps(source_hashes, sort_keys=True)).encode()).hexdigest()[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = PROJECT / settings["outputs"]["run_root"] / (stamp + "_v4.0_" + digest)
    if destination.exists():
        print("ERROR: refusing to overwrite an existing experiment directory.")
        return None
    destination.mkdir(parents=True)
    for relative_path in source_hashes:
        target = destination / "source" / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / relative_path, target)
    recordings.to_csv(destination / "manifest.csv", index=False)
    shutil.copy2(parent / "folds.csv", destination / "folds.csv")
    shutil.copy2(base_path, destination / "base_config.yaml")
    (destination / "config.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    write_json(destination / "settings.json", settings)
    inputs = [base_path, parent / "manifest.csv", parent / "folds.csv", PROJECT / settings["proposal"], PROJECT / settings["report_template"]]
    write_json(destination / "protocol.json", {
        "experiment": "v4.0", "parent": "v2.0", "source_sha256": source_hashes,
        "input_sha256": {path.relative_to(PROJECT).as_posix(): fingerprint(path) for path in inputs},
        "selection_metric": "mean inner speaker AUROC", "headline_metric": "pooled speaker F1 with all five core metrics",
        "threshold": settings["train"]["threshold"], "threshold_policy": "fixed before this experiment",
        "main_cohort": "same 46 speakers and 773 files as the latest technical report",
        "differences_from_proposal": ["Young controls excluded, following the existing project decision",
            "Canonical 46-person shared cohort, not 65 nominal folder identities",
            "Fine-tuning gate uses training-side inner AUROC in each outer fold, never pooled test AUROC",
            "Explicit ParameterGrid loops replace wrapper searches so speaker averaging and selection remain visible"],
        "external_validation": False, "replaces_official_v2_result": False,
        "interpretation": "Revised internal experiment on previously studied data; acquisition confounding remains unresolved.",
    })
    for scenario, subset, expected_people, expected_files in [
        ("main", recordings, 46, 773),
        ("native16k", recordings[recordings.native_sr == 16000], 39, 613),
    ]:
        if not check_recordings(subset, expected_people, expected_files):
            return None
        scenario_directory = destination / scenario
        scenario_directory.mkdir()
        subset.to_csv(scenario_directory / "manifest.csv", index=False)
        export_splits(subset, settings, scenario_directory / "splits")
    write_json(destination / "status.json", {"stage": "prepared", "complete": False})
    print("PREPARED_RUN=" + destination.relative_to(PROJECT).as_posix(), flush=True)
    return destination


def verify_frozen_sources(directory):
    protocol = json.loads((directory / "protocol.json").read_text())
    if protocol["experiment"] != "v4.0":
        print("ERROR: this entry point may only write to its own v4.0 run.")
        return False
    for relative_path, expected in protocol["source_sha256"].items():
        if fingerprint(PROJECT / relative_path) != expected:
            print("ERROR: implementation changed after preparation:", relative_path, "Prepare a new run.")
            return False
    settings = json.loads((directory / "settings.json").read_text())
    for package, expected in settings["dependency_versions"].items():
        if importlib.metadata.version(package) != expected:
            print("ERROR: dependency version changed:", package, "expected", expected)
            return False
    for name, record in settings["pretrained_files"].items():
        if fingerprint(record["cache_path"]) != record["sha256"]:
            print("ERROR: pretrained model file changed:", name)
            return False
    return True


def classify(directory, settings):
    manifest = pd.read_csv(directory / "embedding_manifest.csv")
    features = feature_table(manifest)
    if features is None:
        return False
    reports = {}
    pooled_rows = []
    spread_rows = []
    decisions = []
    for scenario in ["main", "native16k"]:
        rows = features if scenario == "main" else features[features.native_sr == 16000]
        recordings = manifest if scenario == "main" else manifest[manifest.native_sr == 16000]
        reports[scenario] = {}
        for method in METHODS:
            result = run_method(method, rows, settings, directory / scenario / method, directory / scenario / "splits")
            if result is None:
                return False
            reports[scenario][method] = result
            pooled_rows.append({"scenario": scenario, "method": method, **result["pooled"]})
            if scenario == "main":
                pooled_rows.append({"scenario": "without_sheet_flag", "method": method, **result["without_sheet_flag"]["pooled"]})
            for name, variation in result["mean_sd"].items():
                spread_rows.append({"scenario": scenario, "method": method, "metric": name, **variation})
        from fine_tune import run_conditional_fold
        adaptive_tables = []
        for fold in range(settings["splits"]["n_splits"]):
            baseline = directory / scenario / "logistic_regression" / ("fold_" + str(fold))
            selection = json.loads((baseline / "selection.json").read_text())
            target = directory / scenario / "conditional_fine_tuning" / ("fold_" + str(fold))
            decision = run_conditional_fold(recordings, settings, fold, selection, target)
            if decision is None:
                return False
            decisions.append({"scenario": scenario, **decision})
            if decision["triggered"]:
                prediction_path = target / "speaker_predictions.csv"
            else:
                prediction_path = baseline / "speaker_predictions.csv"
            scores = pd.read_csv(prediction_path)
            scores["branch"] = "upper_layer_fine_tuning" if decision["triggered"] else "frozen_logistic_baseline"
            adaptive_tables.append(scores)
        adaptive = pd.concat(adaptive_tables, ignore_index=True)
        adaptive.to_csv(directory / scenario / "conditional_fine_tuning" / "speaker_predictions.csv", index=False)
        conditional = summarize(adaptive, settings["train"]["threshold"])
        conditional["triggered_folds"] = sum(row["triggered"] for row in decisions if row["scenario"] == scenario)
        conditional["not_an_independent_fourth_model_when_no_folds_trigger"] = True
        conditional["without_sheet_flag"] = summarize(adaptive[~adaptive.sheet_flag], settings["train"]["threshold"])
        reports[scenario]["conditional_fine_tuning"] = conditional
        write_json(directory / scenario / "conditional_fine_tuning" / "metrics.json", conditional)
    write_json(directory / "fine_tuning_decisions.json", decisions)
    write_json(directory / "summary.json", reports)
    pd.DataFrame(pooled_rows).to_csv(directory / "pooled_metrics.csv", index=False)
    pd.DataFrame(spread_rows).to_csv(directory / "fold_mean_sd.csv", index=False)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--classify", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--all", action="store_true")
    arguments = parser.parse_args()
    directory = prepare() if arguments.prepare or arguments.all else arguments.run
    if directory is None:
        print("Use --prepare or --run RUN with the requested stage.")
        return False
    directory = directory.resolve()
    if not directory.is_relative_to(PROJECT / "artifacts/runs") or not (directory / "protocol.json").is_file():
        print("ERROR: expected a prepared run inside artifacts/runs.")
        return False
    if not verify_frozen_sources(directory):
        return False
    settings = json.loads((directory / "settings.json").read_text())
    if arguments.extract or arguments.all:
        write_json(directory / "status.json", {"stage": "extracting", "complete": False})
        manifest = pd.read_csv(directory / "manifest.csv")
        if extract_recordings(manifest, settings, directory) is None:
            return False
        write_json(directory / "status.json", {"stage": "features_ready", "complete": False})
    if arguments.classify or arguments.all:
        if not (directory / "embedding_manifest.csv").is_file():
            print("ERROR: run --extract before --classify.")
            return False
        write_json(directory / "status.json", {"stage": "training", "complete": False})
        if not classify(directory, settings):
            return False
        write_json(directory / "status.json", {"stage": "trained_pending_replay", "complete": False})
    if arguments.check or arguments.all:
        from checks import replay_check
        checks = replay_check(directory, settings)
        if checks is None or not checks["all_passed"]:
            write_json(directory / "status.json", {"stage": "replay_failed", "complete": False})
            return False
        write_json(directory / "status.json", {"stage": "completed_and_replayed", "complete": True})
    print("EXPERIMENT_DIRECTORY=" + directory.relative_to(PROJECT).as_posix(), flush=True)
    return True


if __name__ == "__main__":
    if not main():
        sys.exit(1)
