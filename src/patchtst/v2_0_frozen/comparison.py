"""Reproducible comparison of three corrected, speaker-disjoint pipelines."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from patchtst.config import load_config
from patchtst.data.index import build_manifest
from patchtst.data.splits import assign_folds
from patchtst.metrics import classification_scores, speaker_table

MEMBERS = ["arannamoy", "mohibur", "osman"]
METRICS = ["f1", "recall", "specificity", "precision", "accuracy", "balanced_accuracy", "auroc", "average_precision", "mcc", "npv"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def selected_files(files, cfg, member):
    key = {"arannamoy": "patchtst_tasks", "mohibur": "whisper_tasks", "osman": "acoustic_tasks"}[member]
    tasks = cfg["comparison"][key]
    if tasks == "all":
        return files.copy()
    return files[files.task.isin(tasks)].copy()


def prepare():
    cfg = load_config()
    if cfg is None:
        return None
    files, speakers = build_manifest(cfg)
    if files is None:
        return None
    checks = {
        "786 files": len(files) == 786,
        "47 canonical people": speakers.speaker_id.nunique() == 47,
        "25 PD and 22 controls": speakers.label.value_counts().to_dict() == {1: 25, 0: 22},
        "one corrected token": int(files.token_typo.sum()) == 1,
        "four flagged controls": int(speakers.sheet_flag.sum()) == 4,
        "one label per person": bool((files.groupby("speaker_id").label.nunique() == 1).all()),
        "demographics matched": bool(files.sheet_matched.all()),
        "age matches within one year": bool(((files.age - files.age_from_token).abs() <= 1).all()),
    }
    files["audio_sha256"] = [hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in files.path]
    checks["no identical WAV assigned to different people"] = bool((files.groupby("audio_sha256").speaker_id.nunique() == 1).all())
    checks["no repeated recording"] = files.path.is_unique
    common = set(speakers.speaker_id)
    for member in MEMBERS:
        common &= set(selected_files(files, cfg, member).speaker_id)
    checks["46 common people"] = len(common) == 46
    common_files = files[files.speaker_id.isin(common)]
    folds = assign_folds(common_files, cfg)
    files = files.merge(folds[["speaker_id", "fold"]], on="speaker_id", how="left")
    native_common = set(common)
    native = common_files[common_files.native_sr == cfg["data"]["target_sr"]]
    for member in MEMBERS:
        native_common &= set(selected_files(native, cfg, member).speaker_id)
    checks["39 common native-rate people"] = len(native_common) == 39
    for fold in range(cfg["splits"]["n_splits"]):
        test = folds[folds.fold == fold]
        checks[f"fold {fold} has both labels"] = set(test.label) == {0, 1}
        native_test = test[test.speaker_id.isin(native_common)]
        checks[f"native fold {fold} has both labels"] = set(native_test.label) == {0, 1}
    for name, passed in checks.items():
        print("PASS" if passed else "ERROR", name, flush=True)
    if not all(checks.values()):
        print("ERROR: preparation stopped; all manifest checks must pass.")
        return None
    config_text = yaml.safe_dump(cfg, sort_keys=True)
    source_paths = sorted(Path("src/patchtst").rglob("*.py")) + [Path("scripts/compare_members.py")]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    identity = config_text + files.to_csv(index=False) + json.dumps(source_hashes, sort_keys=True)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(cfg["outputs"]["run_root"]) / f"{timestamp}_f-all_{digest}"
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.yaml").write_text(config_text)
    files.to_csv(root / "manifest.csv", index=False)
    speakers.to_csv(root / "all_speakers.csv", index=False)
    folds.to_csv(root / "folds.csv", index=False)
    for path in source_paths + [Path("pyproject.toml"), Path("uv.lock")]:
        destination = root / "source" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    versions = {}
    for package in ["torch", "transformers", "numpy", "scipy", "scikit-learn", "librosa", "praat-parselmouth", "soundfile"]:
        versions[package] = importlib.metadata.version(package)
    protocol = {
        "created_utc": timestamp, "checks": checks, "source_hashes": source_hashes,
        "versions": versions, "common_speakers": sorted(common), "native_speakers": sorted(native_common),
        "threshold": cfg["train"]["threshold"], "threshold_policy": "Fixed before rerunning; no test-based threshold search",
        "aggregation": "Window probabilities to file means, then equal-weight speaker means",
        "spread": "Sample standard deviation across five outer folds, ddof=1",
        "without_sheet_flag": "Evaluation-only exclusion from the main OOF table; no retraining",
        "comparison_scope": "Same speakers and outer folds; each member retains the original speech-task selection",
        "development_status": "Revised internal evaluation after earlier dataset inspection; not an untouched external test",
        "native_rate": "Retrained and evaluated using only native-rate recordings; retain main outer-fold assignments",
    }
    write_json(root / "protocol.json", protocol)
    print("PREPARED_RUN=" + str(root), flush=True)
    return root


def score_table(table):
    scores = classification_scores(table, float(table.threshold.iloc[0]))
    predictions = (table.probability >= table.threshold).astype(int)
    scores["balanced_accuracy"] = (scores["recall"] + scores["specificity"]) / 2
    scores["auroc"] = float(roc_auc_score(table.label, table.probability))
    scores["average_precision"] = float(average_precision_score(table.label, table.probability))
    scores["mcc"] = float(matthews_corrcoef(table.label, predictions))
    denominator = scores["tn"] + scores["fn"]
    scores["npv"] = scores["tn"] / denominator if denominator > 0 else 0.0
    return scores


def summarize_predictions(table):
    pooled = score_table(table)
    folds = []
    for fold, rows in table.groupby("fold"):
        folds.append({"fold": int(fold), **score_table(rows)})
    statistics = {}
    for name in METRICS:
        values = np.array([row[name] for row in folds])
        statistics[name] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1))}
    return {"pooled": pooled, "folds": folds, "mean_sd": statistics}


def make_logistic(member, columns, c_value, seed):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())])
    if member == "osman":
        preprocessing = ColumnTransformer([("acoustic", numeric, columns), ("vowel", OneHotEncoder(handle_unknown="ignore"), ["vowel"])])
        solver = "lbfgs"
    else:
        preprocessing = numeric
        solver = "liblinear"
    return Pipeline([("preprocessing", preprocessing), ("classifier", LogisticRegression(C=c_value, solver=solver, max_iter=5000, random_state=seed))])


def fit_predict_logistic(training, testing, member, columns, c_value, seed):
    inputs = columns + ["vowel"] if member == "osman" else columns
    model = make_logistic(member, columns, c_value, seed)
    counts = training.groupby("speaker_id").size()
    weights = 1.0 / training.speaker_id.map(counts)
    weights = weights / weights.mean()
    model.fit(training[inputs], training.label, classifier__sample_weight=weights)
    predicted = testing[["speaker_id", "label"]].copy()
    if "path" in testing:
        predicted["path"] = testing.path
    predicted["probability"] = model.predict_proba(testing[inputs])[:, 1]
    return model, speaker_table(predicted), predicted


def logistic_fold(features, files, cfg, member, fold, folder):
    speakers = files.groupby("speaker_id", as_index=False).agg(label=("label", "first"), fold=("fold", "first"))
    training = speakers[speakers.fold != fold].reset_index(drop=True)
    test_ids = set(speakers.loc[speakers.fold == fold, "speaker_id"])
    train_ids = set(training.speaker_id)
    columns = [column for column in features if column.startswith(("pitch_", "jitter_", "shimmer_", "mfcc_", "embedding_"))]
    inner = StratifiedKFold(n_splits=cfg["comparison"]["inner_folds"], shuffle=True, random_state=cfg["splits"]["seed"] + 100 + fold)
    split_rows = []
    candidates = []
    for c_value in cfg["comparison"]["c_grid"]:
        aucs = []
        for inner_fold, (fit_idx, val_idx) in enumerate(inner.split(training, training.label)):
            fit_ids = set(training.iloc[fit_idx].speaker_id)
            val_ids = set(training.iloc[val_idx].speaker_id)
            if fit_ids & val_ids or fit_ids & test_ids or val_ids & test_ids:
                print("ERROR: speaker overlap in nested logistic regression.")
                return None
            if c_value == cfg["comparison"]["c_grid"][0]:
                split_rows.append({"inner_fold": inner_fold, "train": sorted(fit_ids), "validation": sorted(val_ids)})
            fitted, table, unused = fit_predict_logistic(features[features.speaker_id.isin(fit_ids)], features[features.speaker_id.isin(val_ids)], member, columns, c_value, cfg["splits"]["seed"])
            aucs.append(float(roc_auc_score(table.label, table.probability)))
        candidates.append({"C": c_value, "mean_inner_auroc": float(np.mean(aucs))})
    best = sorted(candidates, key=lambda row: (-row["mean_inner_auroc"], row["C"]))[0]
    model, table, recording_predictions = fit_predict_logistic(features[features.speaker_id.isin(train_ids)], features[features.speaker_id.isin(test_ids)], member, columns, best["C"], cfg["splits"]["seed"])
    table["threshold"] = cfg["train"]["threshold"]
    table["prediction"] = (table.probability >= table.threshold).astype(int)
    write_json(folder / "selection.json", {"selected_C": best["C"], "candidates": candidates})
    write_json(folder / "split.json", {"outer_train": sorted(train_ids), "outer_test": sorted(test_ids), "inner": split_rows})
    recording_predictions.to_csv(folder / "recording_predictions.csv", index=False)
    joblib.dump(model, folder / "model.joblib")
    return table


def run_member(root, member):
    root = Path(root)
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    for source, expected in protocol["source_hashes"].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
            print("ERROR: implementation changed after protocol freeze:", source, "Prepare a new run.")
            return False
    all_files = pd.read_csv(root / "manifest.csv")
    chosen = all_files[all_files.speaker_id.isin(protocol["common_speakers"])].copy()
    chosen["fold"] = chosen.fold.astype(int)
    chosen = selected_files(chosen, cfg, member)
    if member == "osman":
        from patchtst.comparison_features import acoustic_features
        features = acoustic_features(chosen, cfg)
        if features is None:
            return False
        features.to_csv(root / "osman_acoustic_features.csv", index=False)
    elif member == "mohibur":
        from patchtst.comparison_features import whisper_features
        chosen = whisper_features(chosen, cfg)
        if chosen is None:
            return False
        chosen.to_csv(root / "mohibur_embedding_manifest.csv", index=False)
    else:
        from patchtst.data.audio import cache_features
        chosen = cache_features(chosen, cfg, Path(cfg["outputs"]["cache_root"]) / "logmel")
        chosen.to_csv(root / "arannamoy_feature_manifest.csv", index=False)
    for scenario in ["main", "native16k"]:
        files = chosen.copy()
        expected_ids = set(protocol["common_speakers"])
        if scenario == "native16k":
            expected_ids = set(protocol["native_speakers"])
            files = files[(files.native_sr == cfg["data"]["target_sr"]) & files.speaker_id.isin(expected_ids)]
        destination = root / scenario / member
        destination.mkdir(parents=True, exist_ok=True)
        files.to_csv(destination / "manifest.csv", index=False)
        if set(files.speaker_id) != expected_ids:
            print("ERROR: missing or extra people for", member, scenario)
            return False
        if member == "mohibur":
            feature_rows = []
            for speaker_id, group in files.groupby("speaker_id"):
                vector = np.stack([np.load(path) for path in group.embedding_path]).mean(axis=0)
                row = {"speaker_id": speaker_id, "label": int(group.label.iloc[0])}
                row.update({f"embedding_{index:04d}": float(value) for index, value in enumerate(vector)})
                feature_rows.append(row)
            scenario_features = pd.DataFrame(feature_rows)
        elif member == "osman":
            scenario_features = features[features.path.isin(files.path)].copy()
        tables = []
        for fold in range(cfg["splits"]["n_splits"]):
            folder = destination / f"fold_{fold}"
            folder.mkdir(exist_ok=True)
            print("RUN", member, scenario, "fold", fold, flush=True)
            if (folder / "complete.json").exists():
                table = pd.read_csv(folder / "speaker_predictions.csv")
            elif member == "arannamoy":
                import torch
                from patchtst.train import train_fold
                result = train_fold(files[files.fold != fold], files[files.fold == fold], cfg, fold=fold)
                if result is None:
                    return False
                table = result["speaker_scores"]
                torch.save(result["model"].state_dict(), folder / "checkpoint.pt")
                write_json(folder / "selection_history.json", result["history"])
                write_json(folder / "refit_history.json", result["refit_history"])
                write_json(folder / "selection.json", {"selected_epochs": result["selected_epochs"], "parameters": result["n_parameters"]})
                write_json(folder / "split.json", result["split"])
                result["window_scores"].to_csv(folder / "window_predictions.csv", index=False)
            else:
                table = logistic_fold(scenario_features, files, cfg, member, fold, folder)
                if table is None:
                    return False
            if set(table.speaker_id) != set(files.loc[files.fold == fold, "speaker_id"]) or not table.speaker_id.is_unique:
                print("ERROR: outer-fold prediction coverage failed.")
                return False
            if not np.isfinite(table.probability).all():
                print("ERROR: invalid outer-fold probabilities.")
                return False
            table["fold"] = fold
            table.to_csv(folder / "speaker_predictions.csv", index=False)
            write_json(folder / "complete.json", {"fold": fold, "speakers": len(table)})
            tables.append(table)
        oof = pd.concat(tables, ignore_index=True)
        if not oof.speaker_id.is_unique or set(oof.speaker_id) != expected_ids:
            print("ERROR: pooled OOF coverage failed.")
            return False
        flags = files.groupby("speaker_id").sheet_flag.first()
        oof["sheet_flag"] = oof.speaker_id.map(flags)
        oof.to_csv(destination / "speaker_predictions.csv", index=False)
        report = summarize_predictions(oof)
        report["without_sheet_flag"] = summarize_predictions(oof[~oof.sheet_flag])
        report["recordings"] = len(files)
        write_json(destination / "metrics.json", report)
        pd.DataFrame(report["folds"]).to_csv(destination / "fold_metrics.csv", index=False)
        print("COMPLETE", member, scenario, report["pooled"], flush=True)
    return True


def summarize_run(root):
    root = Path(root)
    reports = {}
    rows, spread_rows = [], []
    for scenario in ["main", "native16k"]:
        reports[scenario] = {}
        for member in MEMBERS:
            path = root / scenario / member / "metrics.json"
            if not path.exists():
                print("ERROR: results are incomplete:", path)
                return False
            report = json.loads(path.read_text())
            reports[scenario][member] = report
            rows.append({"scenario": scenario, "member": member, **report["pooled"]})
            for metric in METRICS:
                spread_rows.append({"scenario": scenario, "member": member, "metric": metric, **report["mean_sd"][metric]})
            if scenario == "main":
                rows.append({"scenario": "without_sheet_flag", "member": member, **report["without_sheet_flag"]["pooled"]})
    write_json(root / "summary.json", reports)
    pd.DataFrame(rows).to_csv(root / "pooled_metrics.csv", index=False)
    pd.DataFrame(spread_rows).to_csv(root / "fold_mean_sd.csv", index=False)
    lines = ["# Corrected three-member evaluation", "", "Same canonical speakers and outer folds; task-specific pipelines. Threshold fixed at 0.5.",
             "", "These are revised internal cross-validation results, not an independent external validation.", ""]
    for scenario in ["main", "native16k", "without_sheet_flag"]:
        lines += ["## " + scenario, "", "| Member | Speakers | F1 | Recall | Specificity | Precision | Accuracy | AUROC |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            if row["scenario"] == scenario:
                values = [f"{row[name]:.4f}" for name in ["f1", "recall", "specificity", "precision", "accuracy", "auroc"]]
                lines.append("| " + row["member"] + " | " + str(row["speakers"]) + " | " + " | ".join(values) + " |")
        lines.append("")
    lines += ["## Fold variation", "", "Five-fold mean and sample standard deviation are in `fold_mean_sd.csv`. Individual folds, confusion counts, average precision, MCC, and NPV are saved alongside the pooled results.",
              "", "The no-sheet-flag result removes flagged speakers from evaluation only. The native16k result retrains each model on native-16-kHz recordings only.",
              "", "Arannamoy: fixed-reference 64-band log-mel, positional channel-independent PatchTST; all tasks. Mohibur: frozen Whisper-large-v3 passage embeddings and nested logistic regression. Osman: pitch, jitter, shimmer, 13 MFCC means and deviations, vowel identity, and nested logistic regression."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print("Report:", root / "REPORT.md")
    return True


def main():
    parser = argparse.ArgumentParser(description="Corrected three-member speaker-level evaluation")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run")
    parser.add_argument("--member", choices=MEMBERS)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run and args.member:
        run_member(args.run, args.member)
    elif args.run and args.summarize:
        summarize_run(args.run)
    else:
        print("Use --prepare, then --run PATH --member NAME, then --run PATH --summarize.")
