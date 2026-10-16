"""Small, readable helpers shared by this separate proposal experiment."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

from patchtst.metrics import classification_scores, speaker_table

PROJECT = Path(__file__).resolve().parents[2]
METHODS = ("logistic_regression", "random_forest", "xgboost")
METRICS = ("f1", "recall", "specificity", "precision", "accuracy", "auroc", "auprc", "average_precision")


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial." + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def score_people(people, threshold):
    result = classification_scores(people, threshold)
    if people.label.nunique() == 2:
        precision, recall, unused_thresholds = precision_recall_curve(people.label, people.probability)
        result["auroc"] = float(roc_auc_score(people.label, people.probability))
        result["auprc"] = float(auc(recall, precision))
        result["average_precision"] = float(average_precision_score(people.label, people.probability))
    else:
        result["auroc"] = None
        result["auprc"] = None
        result["average_precision"] = None
    return result


def summarize(people, threshold):
    folds = []
    for fold, subset in people.groupby("fold"):
        folds.append({"fold": int(fold), **score_people(subset, threshold)})
    spread = {}
    for name in METRICS:
        values = [row[name] for row in folds if row[name] is not None]
        spread[name] = {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "defined_folds": len(values),
        }
    return {"pooled": score_people(people, threshold), "folds": folds, "mean_sd": spread}


def people_from_recordings(recording_scores, metadata, threshold, fold):
    people = speaker_table(recording_scores)
    flags = metadata.groupby("speaker_id").sheet_flag.first()
    people["sheet_flag"] = people.speaker_id.map(flags).astype(bool)
    people["threshold"] = threshold
    people["prediction"] = (people.probability >= threshold).astype(int)
    people["fold"] = fold
    return people


def check_recordings(recordings, expected_people, expected_files):
    problems = []
    if recordings.speaker_id.nunique() != expected_people or len(recordings) != expected_files:
        problems.append("expected " + str(expected_people) + " people and " + str(expected_files) + " recordings")
    if recordings.path.duplicated().any():
        problems.append("a recording path is repeated")
    if recordings.groupby("speaker_id").label.nunique().max() != 1:
        problems.append("one speaker has conflicting labels")
    if recordings.groupby("speaker_id").fold.nunique().max() != 1:
        problems.append("one speaker has more than one outer fold")
    if set(recordings.label) != {0, 1} or recordings.cohort.eq("YHC").any():
        problems.append("only PD and elderly controls may enter")
    for fold in range(5):
        if recordings.loc[recordings.fold == fold, "label"].nunique() != 2:
            problems.append("outer fold " + str(fold) + " must contain both classes")
    for problem in problems:
        print("ERROR:", problem, flush=True)
    return len(problems) == 0


def training_weights(rows):
    """Give classes equal influence, then people and their files equal shares."""
    files_per_person = rows.groupby("speaker_id").size()
    people_per_class = rows.drop_duplicates("speaker_id").groupby("label").size()
    weights = 1.0 / (rows.speaker_id.map(files_per_person) * rows.label.map(people_per_class))
    return (weights / weights.mean()).to_numpy(dtype=float)
