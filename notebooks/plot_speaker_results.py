"""Plot the frozen v2.0 speaker results. This file never opens a WAV."""

from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT / "artifacts/runs/20261012T183328Z_f-all_cead14cc32f1"
OUTPUT = Path(__file__).resolve().parent / "official_speaker_results.svg"


def scores_from_predictions(predictions):
    """Calculate the five speaker-level scores from one saved prediction table."""
    true_positive = int(((predictions.prediction == 1) & (predictions.label == 1)).sum())
    false_negative = int(((predictions.prediction == 0) & (predictions.label == 1)).sum())
    true_negative = int(((predictions.prediction == 0) & (predictions.label == 0)).sum())
    false_positive = int(((predictions.prediction == 1) & (predictions.label == 0)).sum())
    recall = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    precision = true_positive / (true_positive + false_positive)
    f1_score = 2 * precision * recall / (precision + recall)
    accuracy = (true_positive + true_negative) / len(predictions)
    return {
        "speakers": len(predictions),
        "f1": f1_score,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "accuracy": accuracy,
    }


def bar_chart(rows):
    """Draw one grouped bar chart from the three required result tables."""
    metric_names = ["f1", "recall", "specificity", "precision", "accuracy"]
    colors = ["#1f4e79", "#5b8e7d", "#b08968"]
    width = 760
    height = 420
    left = 70
    bottom = 70
    plot_width = 620
    plot_height = 280
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="70" y="32" font-size="18" font-family="sans-serif">Official internal speaker-level results</text>',
    ]
    for tick in range(6):
        value = tick * 20
        y = bottom + plot_height - plot_height * value / 100
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_width}" y2="{y}" stroke="#dddddd"/>')
        parts.append(f'<text x="8" y="{y + 4}" font-size="12" font-family="sans-serif">{value}</text>')
    group_width = plot_width / len(metric_names)
    for metric_index, metric_name in enumerate(metric_names):
        for row_index, row in enumerate(rows):
            bar_height = plot_height * row[metric_name] * 100 / 100
            x = left + metric_index * group_width + 18 + row_index * 28
            y = bottom + plot_height - bar_height
            parts.append(
                f'<rect x="{x}" y="{y}" width="24" height="{bar_height}" fill="{colors[row_index]}"/>'
            )
        label_x = left + metric_index * group_width + 42
        parts.append(
            f'<text x="{label_x}" y="{bottom + plot_height + 24}" text-anchor="middle" '
            f'font-size="12" font-family="sans-serif">{metric_name}</text>'
        )
    for row_index, row in enumerate(rows):
        parts.append(
            f'<rect x="{70 + row_index * 220}" y="390" width="14" height="14" fill="{colors[row_index]}"/>'
        )
        parts.append(
            f'<text x="{90 + row_index * 220}" y="402" font-size="12" font-family="sans-serif">{row["name"]}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    main_predictions = pd.read_csv(RUN / "main/arannamoy/speaker_predictions.csv")
    native_predictions = pd.read_csv(RUN / "native16k/arannamoy/speaker_predictions.csv")
    without_flagged = main_predictions[~main_predictions.sheet_flag.astype(bool)]
    rows = [
        {"name": "Main", **scores_from_predictions(main_predictions)},
        {"name": "No flagged controls", **scores_from_predictions(without_flagged)},
        {"name": "Native 16 kHz", **scores_from_predictions(native_predictions)},
    ]
    for row in rows:
        print(
            row["name"],
            "speakers", row["speakers"],
            "F1", round(row["f1"], 4),
            "recall", round(row["recall"], 4),
            "specificity", round(row["specificity"], 4),
            "precision", round(row["precision"], 4),
            "accuracy", round(row["accuracy"], 4),
        )
    OUTPUT.write_text(bar_chart(rows), encoding="utf-8")
    print("Wrote", OUTPUT)


if __name__ == "__main__":
    main()
