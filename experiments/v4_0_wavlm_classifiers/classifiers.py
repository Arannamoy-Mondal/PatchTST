"""Three classifiers on exactly the same WavLM recording embeddings and folds."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import METHODS, people_from_recordings, score_people, summarize, training_weights, write_json


def make_model(method, parameters, seed, threads):
    if method == "logistic_regression":
        classifier = LogisticRegression(solver="liblinear", max_iter=5000, random_state=seed, **parameters)
        return Pipeline([("scale", StandardScaler()), ("classifier", classifier)])
    if method == "random_forest":
        classifier = RandomForestClassifier(random_state=seed, n_jobs=threads, **parameters)
    else:
        from xgboost import XGBClassifier
        classifier = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            random_state=seed, n_jobs=threads, **parameters,
        )
    return Pipeline([("classifier", classifier)])


def fit_model(method, parameters, training, columns, settings, seed):
    model = make_model(method, parameters, seed, settings["features"]["threads"])
    weights = training_weights(training)
    arguments = {"classifier__sample_weight": weights}
    if method == "logistic_regression":
        arguments["scale__sample_weight"] = weights
    model.fit(training[columns], training.label, **arguments)
    return model


def predict_recordings(model, rows, columns):
    scores = rows[["path", "speaker_id", "label", "task"]].copy()
    scores["probability"] = model.predict_proba(rows[columns])[:, 1]
    return scores


def run_method(method, features, settings, destination, splits_directory):
    destination.mkdir(parents=True, exist_ok=True)
    threshold = settings["train"]["threshold"]
    columns = [name for name in features.columns if name.startswith("embedding_")]
    if len(columns) != settings["features"]["hidden_size"]:
        print("ERROR: the classifier must see exactly 1024 embedding columns.")
        return None
    pooled = []
    for fold in range(settings["splits"]["n_splits"]):
        fold_directory = destination / ("fold_" + str(fold))
        fold_directory.mkdir(parents=True, exist_ok=True)
        split = json.loads((splits_directory / ("fold_" + str(fold)) / "split.json").read_text())
        training = features[features.speaker_id.isin(split["outer_train"])]
        testing = features[features.speaker_id.isin(split["outer_test"])]
        results = []
        validation_tables = []
        candidates = list(ParameterGrid(settings["classifiers"][method]))
        for candidate_number, parameters in enumerate(candidates):
            inner_results = []
            for inner in split["inner"]:
                inner_training = training[training.speaker_id.isin(inner["train"])]
                validation = training[training.speaker_id.isin(inner["validation"])]
                model = fit_model(method, parameters, inner_training, columns, settings, settings["splits"]["seed"] + fold)
                scores = predict_recordings(model, validation, columns)
                people = people_from_recordings(scores, validation, threshold, fold)
                inner_results.append(score_people(people, threshold))
                people["candidate"] = candidate_number
                people["inner_fold"] = inner["inner_fold"]
                validation_tables.append(people)
            mean_auroc = float(np.mean([row["auroc"] for row in inner_results]))
            results.append({"candidate": candidate_number, "parameters": parameters,
                            "mean_inner_speaker_auroc": mean_auroc, "inner_metrics": inner_results})
        # Candidate order is declared before any held-out predictions are made.
        best = results[0]
        for candidate in results[1:]:
            if candidate["mean_inner_speaker_auroc"] > best["mean_inner_speaker_auroc"]:
                best = candidate
        write_json(fold_directory / "selection.json", {
            "selection_metric": "mean_inner_speaker_auroc", "tie_rule": "first candidate in frozen ParameterGrid order",
            "selected": best, "candidates": results,
        })
        pd.concat(validation_tables, ignore_index=True).to_csv(fold_directory / "inner_predictions.csv", index=False)
        final_model = fit_model(method, best["parameters"], training, columns, settings, settings["splits"]["seed"] + fold)
        joblib.dump(final_model, fold_directory / "model.joblib")
        if method == "xgboost":
            # Save the native booster directly; newer sklearn removes a legacy mixin attribute.
            final_model.named_steps["classifier"].get_booster().save_model(fold_directory / "model.ubj")
        training_record = training[["path", "speaker_id", "label"]].copy()
        training_record["weight"] = training_weights(training)
        training_record.to_csv(fold_directory / "fit_rows.csv", index=False)
        write_json(fold_directory / "fit.json", {
            "training_speakers": sorted(training.speaker_id.unique()), "training_recordings": len(training),
            "feature_columns": columns, "refit_from_scratch": True,
            "parameters": best["parameters"], "threshold": threshold,
        })
        recordings = predict_recordings(final_model, testing, columns)
        people = people_from_recordings(recordings, testing, threshold, fold)
        recordings.to_csv(fold_directory / "recording_predictions.csv", index=False)
        people.to_csv(fold_directory / "speaker_predictions.csv", index=False)
        write_json(fold_directory / "metrics.json", score_people(people, threshold))
        pooled.append(people)
        print("Completed", method, "fold", fold, "selected", best["parameters"], flush=True)
    people = pd.concat(pooled, ignore_index=True)
    if not people.speaker_id.is_unique or set(people.speaker_id) != set(features.speaker_id):
        print("ERROR: expected exactly one held-out prediction per speaker.")
        return None
    people.to_csv(destination / "speaker_predictions.csv", index=False)
    result = summarize(people, threshold)
    result["without_sheet_flag"] = summarize(people[~people.sheet_flag], threshold)
    write_json(destination / "metrics.json", result)
    return result
