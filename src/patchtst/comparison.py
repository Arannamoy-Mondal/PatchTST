"""Prepare and run the three speaker-safe pipelines with shared person folds."""

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

MEMBER_NAMES = ["arannamoy", "mohibur", "osman"]
METRIC_NAMES = [
    "f1", "recall", "specificity", "precision", "accuracy",
    "balanced_accuracy", "auroc", "average_precision", "mcc", "npv",
]
MEMBER_TASK_SETTING = {
    "arannamoy": "patchtst_tasks",
    "mohibur": "whisper_tasks",
    "osman": "acoustic_tasks",
}


def write_json(path, value):
    """Save readable JSON without silently permitting invalid numbers."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def recordings_for_member(recordings, settings, member_name):
    """Keep only the speech tasks used by one member's original method."""
    task_setting_name = MEMBER_TASK_SETTING[member_name]
    selected_tasks = settings["comparison"][task_setting_name]
    if selected_tasks == "all":
        return recordings.copy()
    return recordings[recordings.task.isin(selected_tasks)].copy()


def add_audio_hashes(recordings):
    """Attach a hash of every recording's exact file bytes."""
    recordings = recordings.copy()
    recordings["audio_sha256"] = [
        hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in recordings.path
    ]
    return recordings


def manifest_checks(recordings, speakers, settings):
    """Check the known corpus counts and build the shared person cohorts."""
    checks = {
        "786 files": len(recordings) == 786,
        "47 canonical people": speakers.speaker_id.nunique() == 47,
        "25 PD and 22 controls": speakers.label.value_counts().to_dict() == {1: 25, 0: 22},
        "one corrected token": int(recordings.token_typo.sum()) == 1,
        "four flagged controls": int(speakers.sheet_flag.sum()) == 4,
        "one label per person": bool((recordings.groupby("speaker_id").label.nunique() == 1).all()),
        "demographics matched": bool(recordings.sheet_matched.all()),
        "age matches within one year": bool(((recordings.age - recordings.age_from_token).abs() <= 1).all()),
    }
    recordings = add_audio_hashes(recordings)
    checks["no identical WAV assigned to different people"] = bool(
        (recordings.groupby("audio_sha256").speaker_id.nunique() == 1).all()
    )
    checks["no repeated recording"] = recordings.path.is_unique

    common_speaker_ids = set(speakers.speaker_id)
    for member_name in MEMBER_NAMES:
        common_speaker_ids &= set(recordings_for_member(recordings, settings, member_name).speaker_id)
    checks["46 common people"] = len(common_speaker_ids) == 46
    common_recordings = recordings[recordings.speaker_id.isin(common_speaker_ids)]
    person_folds = assign_folds(common_recordings, settings)
    recordings = recordings.merge(person_folds[["speaker_id", "fold"]], on="speaker_id", how="left")

    native_recordings = common_recordings[common_recordings.native_sr == settings["data"]["target_sr"]]
    native_speaker_ids = set(common_speaker_ids)
    for member_name in MEMBER_NAMES:
        native_speaker_ids &= set(recordings_for_member(native_recordings, settings, member_name).speaker_id)
    checks["39 common native-rate people"] = len(native_speaker_ids) == 39

    for fold_number in range(settings["splits"]["n_splits"]):
        test_people = person_folds[person_folds.fold == fold_number]
        checks["fold %s has both labels" % fold_number] = set(test_people.label) == {0, 1}
        native_test_people = test_people[test_people.speaker_id.isin(native_speaker_ids)]
        checks["native fold %s has both labels" % fold_number] = set(native_test_people.label) == {0, 1}
    return recordings, person_folds, checks, common_speaker_ids, native_speaker_ids


def copy_frozen_sources(run_directory):
    """Copy the implementation used by a prepared run into that run."""
    source_paths = sorted(Path("src/patchtst").rglob("*.py"))
    source_paths.append(Path("scripts/compare_members.py"))
    source_hashes = {}
    for source_path in source_paths:
        source_hashes[str(source_path)] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        destination = run_directory / "source" / source_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
    for extra_path in (Path("pyproject.toml"), Path("uv.lock")):
        destination = run_directory / "source" / extra_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(extra_path, destination)
    return source_hashes


def prepare():
    """Create a new versioned run directory after every manifest check passes."""
    settings = load_config()
    if settings is None:
        return None
    recordings, speakers = build_manifest(settings)
    if recordings is None:
        return None
    recordings, person_folds, checks, common_ids, native_ids = manifest_checks(
        recordings,
        speakers,
        settings,
    )
    for check_name, check_passed in checks.items():
        print("PASS" if check_passed else "ERROR", check_name, flush=True)
    if not all(checks.values()):
        print("ERROR: preparation stopped because a manifest check failed.")
        return None

    settings_text = yaml.safe_dump(settings, sort_keys=True)
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path("src/patchtst").rglob("*.py"))
    }
    source_hashes["scripts/compare_members.py"] = hashlib.sha256(
        Path("scripts/compare_members.py").read_bytes()
    ).hexdigest()
    identity_text = settings_text + recordings.to_csv(index=False) + json.dumps(source_hashes, sort_keys=True)
    short_hash = hashlib.sha256(identity_text.encode()).hexdigest()[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = Path(settings["outputs"]["run_root"]) / (timestamp + "_f-all_" + short_hash)
    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / "config.yaml").write_text(settings_text)
    recordings.to_csv(run_directory / "manifest.csv", index=False)
    speakers.to_csv(run_directory / "all_speakers.csv", index=False)
    person_folds.to_csv(run_directory / "folds.csv", index=False)
    copy_frozen_sources(run_directory)

    package_versions = {}
    package_names = [
        "torch", "transformers", "numpy", "scipy", "scikit-learn",
        "librosa", "praat-parselmouth", "soundfile",
    ]
    for package_name in package_names:
        package_versions[package_name] = importlib.metadata.version(package_name)
    protocol = {
        "created_utc": timestamp,
        "checks": checks,
        "source_hashes": source_hashes,
        "versions": package_versions,
        "common_speakers": sorted(common_ids),
        "native_speakers": sorted(native_ids),
        "threshold": settings["train"]["threshold"],
        "threshold_policy": "Fixed before rerunning; no test-based threshold search",
        "aggregation": "Window probabilities to file means, then equal-weight speaker means",
        "spread": "Sample standard deviation across five outer folds, ddof=1",
        "without_sheet_flag": "Evaluation-only exclusion from the main OOF table; no retraining",
        "comparison_scope": "Same speakers and outer folds; each member retains the original speech-task selection",
        "development_status": "Revised internal evaluation after earlier dataset inspection; not an untouched external test",
        "native_rate": "Retrained and evaluated using only native-rate recordings; retain main outer-fold assignments",
    }
    write_json(run_directory / "protocol.json", protocol)
    print("PREPARED_RUN=" + str(run_directory), flush=True)
    return run_directory


def score_people(people):
    """Add ranking and correlation scores to the five core classification scores."""
    threshold = float(people.threshold.iloc[0])
    scores = classification_scores(people, threshold)
    predicted_labels = (people.probability >= people.threshold).astype(int)
    scores["balanced_accuracy"] = (scores["recall"] + scores["specificity"]) / 2
    scores["auroc"] = float(roc_auc_score(people.label, people.probability))
    scores["average_precision"] = float(average_precision_score(people.label, people.probability))
    scores["mcc"] = float(matthews_corrcoef(people.label, predicted_labels))
    negative_count = scores["tn"] + scores["fn"]
    if negative_count:
        scores["npv"] = scores["tn"] / negative_count
    else:
        scores["npv"] = 0.0
    return scores


def summarize_predictions(people):
    """Summarize pooled predictions and the variation across five folds."""
    fold_summaries = []
    for fold_number, fold_people in people.groupby("fold"):
        fold_summaries.append({"fold": int(fold_number), **score_people(fold_people)})
    variation = {}
    for metric_name in METRIC_NAMES:
        values = np.array([summary[metric_name] for summary in fold_summaries])
        variation[metric_name] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1))}
    return {"pooled": score_people(people), "folds": fold_summaries, "mean_sd": variation}


def make_logistic_model(member_name, numeric_columns, regularization_strength, seed):
    """Build preprocessing and logistic regression fitted only during training."""
    numeric_preparation = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    if member_name == "osman":
        preparation = ColumnTransformer([
            ("acoustic", numeric_preparation, numeric_columns),
            ("vowel", OneHotEncoder(handle_unknown="ignore"), ["vowel"]),
        ])
        solver_name = "lbfgs"
    else:
        preparation = numeric_preparation
        solver_name = "liblinear"
    classifier = LogisticRegression(
        C=regularization_strength,
        solver=solver_name,
        max_iter=5000,
        random_state=seed,
    )
    return Pipeline([("preprocessing", preparation), ("classifier", classifier)])


def fit_and_predict_logistic(training_rows, test_rows, member_name, numeric_columns, strength, seed):
    """Fit one logistic model and return person-level test probabilities."""
    if member_name == "osman":
        input_columns = numeric_columns + ["vowel"]
    else:
        input_columns = numeric_columns
    model = make_logistic_model(member_name, numeric_columns, strength, seed)
    recordings_per_person = training_rows.groupby("speaker_id").size()
    sample_weights = 1.0 / training_rows.speaker_id.map(recordings_per_person)
    sample_weights = sample_weights / sample_weights.mean()
    model.fit(
        training_rows[input_columns],
        training_rows.label,
        classifier__sample_weight=sample_weights,
    )

    predictions = test_rows[["speaker_id", "label"]].copy()
    if "path" in test_rows:
        predictions["path"] = test_rows.path
    predictions["probability"] = model.predict_proba(test_rows[input_columns])[:, 1]
    return model, speaker_table(predictions), predictions


def logistic_fold(features, recordings, settings, member_name, fold_number, fold_directory):
    """Choose regularization inside training people, then score the held-out people."""
    people = recordings.groupby("speaker_id", as_index=False).agg(
        label=("label", "first"),
        fold=("fold", "first"),
    )
    training_people = people[people.fold != fold_number].reset_index(drop=True)
    test_ids = set(people.loc[people.fold == fold_number, "speaker_id"])
    training_ids = set(training_people.speaker_id)
    numeric_columns = [
        column_name for column_name in features.columns
        if column_name.startswith(("pitch_", "jitter_", "shimmer_", "mfcc_", "embedding_"))
    ]
    inner_splitter = StratifiedKFold(
        n_splits=settings["comparison"]["inner_folds"],
        shuffle=True,
        random_state=settings["splits"]["seed"] + 100 + fold_number,
    )
    inner_splits = []
    candidate_results = []
    for regularization_strength in settings["comparison"]["c_grid"]:
        inner_scores = []
        for inner_number, (fit_positions, validation_positions) in enumerate(
            inner_splitter.split(training_people, training_people.label)
        ):
            fit_ids = set(training_people.iloc[fit_positions].speaker_id)
            validation_ids = set(training_people.iloc[validation_positions].speaker_id)
            ids_overlap = fit_ids & validation_ids or fit_ids & test_ids or validation_ids & test_ids
            if ids_overlap:
                print("ERROR: the same person crosses a logistic-regression split.")
                return None
            if regularization_strength == settings["comparison"]["c_grid"][0]:
                inner_splits.append({
                    "inner_fold": inner_number,
                    "train": sorted(fit_ids),
                    "validation": sorted(validation_ids),
                })
            fitted_model, validation_people, unused_rows = fit_and_predict_logistic(
                features[features.speaker_id.isin(fit_ids)],
                features[features.speaker_id.isin(validation_ids)],
                member_name,
                numeric_columns,
                regularization_strength,
                settings["splits"]["seed"],
            )
            inner_scores.append(float(roc_auc_score(
                validation_people.label,
                validation_people.probability,
            )))
        candidate_results.append({
            "C": regularization_strength,
            "mean_inner_auroc": float(np.mean(inner_scores)),
        })

    # Equal scores choose the smaller C because candidates are sorted by strength too.
    best_candidate = sorted(
        candidate_results,
        key=lambda result: (-result["mean_inner_auroc"], result["C"]),
    )[0]
    final_model, person_scores, recording_scores = fit_and_predict_logistic(
        features[features.speaker_id.isin(training_ids)],
        features[features.speaker_id.isin(test_ids)],
        member_name,
        numeric_columns,
        best_candidate["C"],
        settings["splits"]["seed"],
    )
    person_scores["threshold"] = settings["train"]["threshold"]
    person_scores["prediction"] = (
        person_scores.probability >= person_scores.threshold
    ).astype(int)
    write_json(fold_directory / "selection.json", {
        "selected_C": best_candidate["C"],
        "candidates": candidate_results,
    })
    write_json(fold_directory / "split.json", {
        "outer_train": sorted(training_ids),
        "outer_test": sorted(test_ids),
        "inner": inner_splits,
    })
    recording_scores.to_csv(fold_directory / "recording_predictions.csv", index=False)
    joblib.dump(final_model, fold_directory / "model.joblib")
    return person_scores


def extract_member_features(recordings, settings, member_name, run_directory):
    """Create or reuse the feature cache selected for one member."""
    if member_name == "osman":
        from patchtst.comparison_features import acoustic_features

        features = acoustic_features(recordings, settings)
        if features is not None:
            features.to_csv(run_directory / "osman_acoustic_features.csv", index=False)
        return recordings, features
    if member_name == "mohibur":
        from patchtst.comparison_features import whisper_features

        recordings_with_embeddings = whisper_features(recordings, settings)
        if recordings_with_embeddings is not None:
            recordings_with_embeddings.to_csv(run_directory / "mohibur_embedding_manifest.csv", index=False)
        return recordings_with_embeddings, None

    from patchtst.data.audio import cache_features

    recordings_with_features = cache_features(
        recordings,
        settings,
        Path(settings["outputs"]["cache_root"]) / "logmel",
    )
    recordings_with_features.to_csv(run_directory / "arannamoy_feature_manifest.csv", index=False)
    return recordings_with_features, None


def whisper_person_table(recordings):
    """Average recording embeddings into one 1,280-value row per person."""
    person_rows = []
    for speaker_id, person_recordings in recordings.groupby("speaker_id"):
        embedding_paths = person_recordings.embedding_path
        person_embedding = np.stack([np.load(path) for path in embedding_paths]).mean(axis=0)
        person_row = {
            "speaker_id": speaker_id,
            "label": int(person_recordings.label.iloc[0]),
        }
        for feature_number, feature_value in enumerate(person_embedding):
            person_row["embedding_%04d" % feature_number] = float(feature_value)
        person_rows.append(person_row)
    return pd.DataFrame(person_rows)


def train_one_fold(member_name, recordings, features, settings, fold_number, fold_directory):
    """Train whichever member was requested and return person-level test scores."""
    if member_name == "arannamoy":
        import torch
        from patchtst.train import train_fold

        result = train_fold(
            recordings[recordings.fold != fold_number],
            recordings[recordings.fold == fold_number],
            settings,
            fold=fold_number,
        )
        if result is None:
            return None
        torch.save(result["model"].state_dict(), fold_directory / "checkpoint.pt")
        write_json(fold_directory / "selection_history.json", result["history"])
        write_json(fold_directory / "refit_history.json", result["refit_history"])
        write_json(fold_directory / "selection.json", {
            "selected_epochs": result["selected_epochs"],
            "parameters": result["n_parameters"],
        })
        write_json(fold_directory / "split.json", result["split"])
        result["window_scores"].to_csv(fold_directory / "window_predictions.csv", index=False)
        return result["speaker_scores"]
    return logistic_fold(features, recordings, settings, member_name, fold_number, fold_directory)


def run_scenario(member_name, recordings, features, settings, protocol, scenario_name, run_directory):
    """Train five folds for one cohort scenario and save its pooled predictions."""
    expected_speaker_ids = set(protocol["common_speakers"])
    if scenario_name == "native16k":
        expected_speaker_ids = set(protocol["native_speakers"])
        native_rate = settings["data"]["target_sr"]
        recordings = recordings[
            (recordings.native_sr == native_rate)
            & recordings.speaker_id.isin(expected_speaker_ids)
        ]
    scenario_directory = run_directory / scenario_name / member_name
    scenario_directory.mkdir(parents=True, exist_ok=True)
    recordings.to_csv(scenario_directory / "manifest.csv", index=False)
    if set(recordings.speaker_id) != expected_speaker_ids:
        print("ERROR: scenario people do not match the prepared cohort:", member_name, scenario_name)
        return False

    if member_name == "mohibur":
        scenario_features = whisper_person_table(recordings)
    elif member_name == "osman":
        scenario_features = features[features.path.isin(recordings.path)].copy()
    else:
        scenario_features = None

    fold_tables = []
    for fold_number in range(settings["splits"]["n_splits"]):
        fold_directory = scenario_directory / ("fold_" + str(fold_number))
        fold_directory.mkdir(exist_ok=True)
        print("RUN", member_name, scenario_name, "fold", fold_number, flush=True)
        if (fold_directory / "complete.json").exists():
            person_scores = pd.read_csv(fold_directory / "speaker_predictions.csv")
        else:
            person_scores = train_one_fold(
                member_name,
                recordings,
                scenario_features,
                settings,
                fold_number,
                fold_directory,
            )
            if person_scores is None:
                return False

        expected_test_ids = set(recordings.loc[recordings.fold == fold_number, "speaker_id"])
        valid_coverage = person_scores.speaker_id.is_unique and set(person_scores.speaker_id) == expected_test_ids
        if not valid_coverage or not np.isfinite(person_scores.probability).all():
            print("ERROR: held-out predictions are incomplete or invalid.")
            return False
        person_scores["fold"] = fold_number
        person_scores.to_csv(fold_directory / "speaker_predictions.csv", index=False)
        write_json(fold_directory / "complete.json", {
            "fold": fold_number,
            "speakers": len(person_scores),
        })
        fold_tables.append(person_scores)

    pooled_predictions = pd.concat(fold_tables, ignore_index=True)
    valid_pool = pooled_predictions.speaker_id.is_unique and set(pooled_predictions.speaker_id) == expected_speaker_ids
    if not valid_pool:
        print("ERROR: pooled predictions do not cover every person exactly once.")
        return False
    sheet_flag_by_person = recordings.groupby("speaker_id").sheet_flag.first()
    pooled_predictions["sheet_flag"] = pooled_predictions.speaker_id.map(sheet_flag_by_person)
    pooled_predictions.to_csv(scenario_directory / "speaker_predictions.csv", index=False)

    report = summarize_predictions(pooled_predictions)
    report["without_sheet_flag"] = summarize_predictions(
        pooled_predictions[~pooled_predictions.sheet_flag]
    )
    report["recordings"] = len(recordings)
    write_json(scenario_directory / "metrics.json", report)
    pd.DataFrame(report["folds"]).to_csv(scenario_directory / "fold_metrics.csv", index=False)
    print("COMPLETE", member_name, scenario_name, report["pooled"], flush=True)
    return True


def implementation_matches_protocol(protocol):
    """Refuse to continue a prepared run after its implementation changes."""
    for source_path, expected_hash in protocol["source_hashes"].items():
        current_hash = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
        if current_hash != expected_hash:
            print("ERROR: implementation changed after this run was prepared:", source_path)
            print("Prepare a new run instead of mixing two implementations.")
            return False
    return True


def run_member(run_directory, member_name):
    """Run one member on the main cohort and the native-rate cohort."""
    run_directory = Path(run_directory)
    settings = yaml.safe_load((run_directory / "config.yaml").read_text())
    protocol = json.loads((run_directory / "protocol.json").read_text())
    if not implementation_matches_protocol(protocol):
        return False

    recordings = pd.read_csv(run_directory / "manifest.csv")
    recordings = recordings[recordings.speaker_id.isin(protocol["common_speakers"])].copy()
    recordings["fold"] = recordings.fold.astype(int)
    recordings = recordings_for_member(recordings, settings, member_name)
    recordings, features = extract_member_features(recordings, settings, member_name, run_directory)
    if recordings is None or (member_name == "osman" and features is None):
        return False

    for scenario_name in ("main", "native16k"):
        scenario_completed = run_scenario(
            member_name,
            recordings,
            features,
            settings,
            protocol,
            scenario_name,
            run_directory,
        )
        if not scenario_completed:
            return False
    return True


def summarize_run(run_directory):
    """Collect saved member results into CSV and Markdown summaries."""
    run_directory = Path(run_directory)
    reports = {}
    pooled_rows = []
    variation_rows = []
    for scenario_name in ("main", "native16k"):
        reports[scenario_name] = {}
        for member_name in MEMBER_NAMES:
            metrics_path = run_directory / scenario_name / member_name / "metrics.json"
            if not metrics_path.exists():
                print("ERROR: results are incomplete:", metrics_path)
                return False
            report = json.loads(metrics_path.read_text())
            reports[scenario_name][member_name] = report
            pooled_rows.append({"scenario": scenario_name, "member": member_name, **report["pooled"]})
            for metric_name in METRIC_NAMES:
                variation_rows.append({
                    "scenario": scenario_name,
                    "member": member_name,
                    "metric": metric_name,
                    **report["mean_sd"][metric_name],
                })
            if scenario_name == "main":
                pooled_rows.append({
                    "scenario": "without_sheet_flag",
                    "member": member_name,
                    **report["without_sheet_flag"]["pooled"],
                })

    write_json(run_directory / "summary.json", reports)
    pd.DataFrame(pooled_rows).to_csv(run_directory / "pooled_metrics.csv", index=False)
    pd.DataFrame(variation_rows).to_csv(run_directory / "fold_mean_sd.csv", index=False)
    report_lines = [
        "# Corrected three-member evaluation",
        "",
        "Same canonical speakers and outer folds; task-specific pipelines. Threshold fixed at 0.5.",
        "",
        "These are revised internal cross-validation results, not an independent external validation.",
        "",
    ]
    for scenario_name in ("main", "native16k", "without_sheet_flag"):
        report_lines.extend([
            "## " + scenario_name,
            "",
            "| Member | Speakers | F1 | Recall | Specificity | Precision | Accuracy | AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in pooled_rows:
            if row["scenario"] == scenario_name:
                displayed_values = [
                    "%.4f" % row[metric_name]
                    for metric_name in ("f1", "recall", "specificity", "precision", "accuracy", "auroc")
                ]
                report_lines.append(
                    "| " + row["member"] + " | " + str(row["speakers"]) + " | "
                    + " | ".join(displayed_values) + " |"
                )
        report_lines.append("")
    report_lines.extend([
        "## Fold variation",
        "",
        "Five-fold mean and sample standard deviation are in `fold_mean_sd.csv`.",
        "",
        "The no-sheet-flag result removes flagged speakers from evaluation only. The native16k result retrains each model on native-16-kHz recordings only.",
        "",
        "Arannamoy: fixed-reference 64-band log-mel, positional channel-independent PatchTST; all tasks. Mohibur: frozen Whisper-large-v3 passage embeddings and nested logistic regression. Osman: pitch, jitter, shimmer, 13 MFCC means and deviations, vowel identity, and nested logistic regression.",
    ])
    (run_directory / "REPORT.md").write_text("\n".join(report_lines) + "\n")
    print(pd.DataFrame(pooled_rows).to_string(index=False))
    print("Report:", run_directory / "REPORT.md")
    return True


def main():
    """Dispatch preparation, one member run, or final result collection."""
    argument_parser = argparse.ArgumentParser(
        description="Corrected three-member speaker-level evaluation"
    )
    argument_parser.add_argument("--prepare", action="store_true")
    argument_parser.add_argument("--run")
    argument_parser.add_argument("--member", choices=MEMBER_NAMES)
    argument_parser.add_argument("--summarize", action="store_true")
    arguments = argument_parser.parse_args()

    if arguments.prepare:
        prepare()
    elif arguments.run and arguments.member:
        run_member(arguments.run, arguments.member)
    elif arguments.run and arguments.summarize:
        summarize_run(arguments.run)
    else:
        print("Use --prepare, then --run PATH --member NAME, then --run PATH --summarize.")
