"""Turn window probabilities into one decision and score per person."""

import numpy as np
import pandas as pd


def speaker_table(window_scores):
    """Average windows by recording first, then recordings by person.

    Equal recording weight prevents one long recording from dominating a person.
    """
    scores = window_scores
    if "path" in scores.columns:
        scores = scores.groupby(["speaker_id", "path"], as_index=False).agg(
            label=("label", "first"),
            probability=("probability", "mean"),
        )
    return (
        scores.groupby("speaker_id", as_index=False)
        .agg(label=("label", "first"), probability=("probability", "mean"))
        .sort_values("speaker_id")
        .reset_index(drop=True)
    )


def classification_scores(people, threshold):
    """Calculate the five core speaker-level scores and confusion counts."""
    predicted_label = (people["probability"] >= threshold).astype(int)
    true_label = people["label"].astype(int)
    true_positive = int(((predicted_label == 1) & (true_label == 1)).sum())
    false_negative = int(((predicted_label == 0) & (true_label == 1)).sum())
    true_negative = int(((predicted_label == 0) & (true_label == 0)).sum())
    false_positive = int(((predicted_label == 1) & (true_label == 0)).sum())

    patient_count = true_positive + false_negative
    control_count = true_negative + false_positive
    called_patient_count = true_positive + false_positive
    speaker_count = patient_count + control_count
    recall = true_positive / patient_count if patient_count else 0.0
    specificity = true_negative / control_count if control_count else 0.0
    precision = true_positive / called_patient_count if called_patient_count else 0.0
    if precision + recall:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0
    accuracy = (true_positive + true_negative) / speaker_count if speaker_count else 0.0

    return {
        "threshold": float(threshold),
        "speakers": int(speaker_count),
        "tp": true_positive,
        "fn": false_negative,
        "tn": true_negative,
        "fp": false_positive,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1_score,
        "accuracy": accuracy,
    }


def choose_threshold(people):
    """Choose the training-side threshold with the best F1 score.

    When several thresholds tie, the one closest to 0.5 is kept. This function
    is retained for compatibility; the current comparison uses a fixed threshold.
    """
    best_threshold = 0.5
    best_f1_score = -1.0
    for step_number in range(1, 20):
        candidate_threshold = step_number / 20
        candidate_scores = classification_scores(people, candidate_threshold)
        candidate_f1_score = candidate_scores["f1"]
        better_score = candidate_f1_score > best_f1_score
        same_score = candidate_f1_score == best_f1_score
        closer_to_half = abs(candidate_threshold - 0.5) < abs(best_threshold - 0.5)
        if better_score or (same_score and closer_to_half):
            best_f1_score = candidate_f1_score
            best_threshold = candidate_threshold
    return best_threshold


def summarize_folds(fold_scores):
    """Return the mean and sample standard deviation across folds."""
    summary = {}
    metric_names = ("recall", "specificity", "precision", "f1", "accuracy")
    for metric_name in metric_names:
        values = np.array([fold[metric_name] for fold in fold_scores], dtype=float)
        summary[metric_name + "_mean"] = float(values.mean())
        summary[metric_name + "_std"] = float(values.std(ddof=1))
    return summary
