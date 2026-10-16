"""Speaker-level scores. Windows are averaged before any metric is computed."""

from __future__ import annotations

import numpy as np
import pandas as pd


def speaker_table(windows: pd.DataFrame) -> pd.DataFrame:
    if "path" in windows.columns:
        windows = windows.groupby(["speaker_id", "path"], as_index=False).agg(
            label=("label", "first"), probability=("probability", "mean")
        )
    return (
        windows.groupby("speaker_id", as_index=False)
        .agg(label=("label", "first"), probability=("probability", "mean"))
        .sort_values("speaker_id")
        .reset_index(drop=True)
    )


def choose_threshold(speakers: pd.DataFrame) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for step in range(1, 20):
        threshold = step / 20
        scores = classification_scores(speakers, threshold)
        if scores["f1"] > best_f1 or (
            scores["f1"] == best_f1 and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_f1 = scores["f1"]
            best_threshold = threshold
    return best_threshold


def classification_scores(speakers: pd.DataFrame, threshold: float) -> dict[str, float]:
    predicted = (speakers["probability"] >= threshold).astype(int)
    label = speakers["label"].astype(int)
    true_positive = int(((predicted == 1) & (label == 1)).sum())
    false_negative = int(((predicted == 0) & (label == 1)).sum())
    true_negative = int(((predicted == 0) & (label == 0)).sum())
    false_positive = int(((predicted == 1) & (label == 0)).sum())
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) else 0.0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total = true_positive + false_negative + true_negative + false_positive
    accuracy = (true_positive + true_negative) / total if total else 0.0
    return {
        "threshold": float(threshold),
        "speakers": int(total),
        "tp": true_positive,
        "fn": false_negative,
        "tn": true_negative,
        "fp": false_positive,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
    }


def summarize_folds(fold_scores: list[dict]) -> dict[str, float]:
    summary = {}
    for name in ("recall", "specificity", "precision", "f1", "accuracy"):
        values = np.array([item[name] for item in fold_scores], dtype=float)
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_std"] = float(values.std(ddof=1))
    return summary
