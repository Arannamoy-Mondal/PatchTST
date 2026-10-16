"""Replay Osman's saved models and optionally repeat a label-shuffle sanity check."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from patchtst.comparison import logistic_fold, write_json
from patchtst.metrics import speaker_table


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--run", required=True)
    argument_parser.add_argument("--permutations", type=int, default=0)
    arguments = argument_parser.parse_args()

    run_directory = Path(arguments.run)
    settings = yaml.safe_load((run_directory / "config.yaml").read_text())
    acoustic_features = pd.read_csv(run_directory / "osman_acoustic_features.csv")
    diagnostics_directory = run_directory / "diagnostics"
    diagnostics_directory.mkdir(exist_ok=True)
    audit_rows = []

    for scenario_name in ("main", "native16k"):
        scenario_directory = run_directory / scenario_name / "osman"
        recordings = pd.read_csv(scenario_directory / "manifest.csv")
        scenario_features = acoustic_features[acoustic_features.path.isin(recordings.path)].copy()
        for fold_number in range(settings["splits"]["n_splits"]):
            fold_directory = scenario_directory / ("fold_" + str(fold_number))
            split = json.loads((fold_directory / "split.json").read_text())
            model = joblib.load(fold_directory / "model.joblib")
            saved_scores = pd.read_csv(fold_directory / "speaker_predictions.csv")
            test_features = scenario_features[
                scenario_features.speaker_id.isin(split["outer_test"])
            ].copy()
            training_features = scenario_features[
                scenario_features.speaker_id.isin(split["outer_train"])
            ]
            input_names = list(model.feature_names_in_)
            test_features["probability"] = model.predict_proba(test_features[input_names])[:, 1]
            replay_scores = speaker_table(
                test_features[["speaker_id", "path", "label", "probability"]]
            )
            replay_by_person = replay_scores.set_index("speaker_id").probability
            saved_by_person = saved_scores.set_index("speaker_id").probability
            maximum_difference = float((replay_by_person - saved_by_person).abs().max())
            fitted_row_count = int(
                model.named_steps["preprocessing"]
                .named_transformers_["acoustic"]
                .named_steps["scale"]
                .n_samples_seen_
            )
            checks = {
                "no_outer_person_overlap": not bool(set(split["outer_train"]) & set(split["outer_test"])),
                "no_label_or_metadata_input": not bool(set(input_names) & {
                    "label", "speaker_id", "native_sr", "age", "sex", "path", "fold",
                }),
                "training_only_scaler": fitted_row_count == len(training_features),
                "saved_predictions_match": maximum_difference < 1e-6,
            }
            for inner_split in split["inner"]:
                training_ids = set(inner_split["train"])
                validation_ids = set(inner_split["validation"])
                split_number = inner_split["inner_fold"]
                checks["inner_%s_disjoint" % split_number] = not bool(
                    training_ids & validation_ids
                    or (training_ids | validation_ids) & set(split["outer_test"])
                )
            if scenario_name == "native16k":
                checks["native_16khz_only"] = bool(
                    (recordings.native_sr == settings["data"]["target_sr"]).all()
                )
            print("PASS" if all(checks.values()) else "ERROR", scenario_name, "fold", fold_number)
            audit_rows.append({"scenario": scenario_name, "fold": fold_number, "checks": checks})

    write_json(diagnostics_directory / "osman_prediction_audit.json", audit_rows)
    if not all(all(row["checks"].values()) for row in audit_rows):
        print("ERROR: prediction audit failed.")
        return
    if arguments.permutations <= 0:
        return

    recordings = pd.read_csv(run_directory / "native16k/osman/manifest.csv")
    original_features = acoustic_features[acoustic_features.path.isin(recordings.path)].copy()
    people = recordings.groupby("speaker_id", as_index=False).agg(label=("label", "first"))
    random_generator = np.random.default_rng(settings["splits"]["seed"] + 10000)
    shuffled_scores = []
    for permutation_number in range(arguments.permutations):
        shuffled_labels = dict(zip(
            people.speaker_id,
            random_generator.permutation(people.label.to_numpy()),
        ))
        shuffled_recordings = recordings.copy()
        shuffled_features = original_features.copy()
        shuffled_recordings["label"] = shuffled_recordings.speaker_id.map(shuffled_labels)
        shuffled_features["label"] = shuffled_features.speaker_id.map(shuffled_labels)
        fold_tables = []
        for fold_number in range(settings["splits"]["n_splits"]):
            fold_directory = (
                diagnostics_directory / "label_shuffle"
                / ("permutation_%02d" % permutation_number)
                / ("fold_" + str(fold_number))
            )
            fold_directory.mkdir(parents=True, exist_ok=True)
            person_scores = logistic_fold(
                shuffled_features,
                shuffled_recordings,
                settings,
                "osman",
                fold_number,
                fold_directory,
            )
            if person_scores is None:
                print("ERROR: label-shuffle training did not complete.")
                return
            fold_tables.append(person_scores)
        pooled_predictions = pd.concat(fold_tables, ignore_index=True)
        pooled_auroc = float(roc_auc_score(
            pooled_predictions.label,
            pooled_predictions.probability,
        ))
        shuffled_scores.append(pooled_auroc)
        print("Label shuffle", permutation_number + 1, "pooled AUROC", round(pooled_auroc, 4), flush=True)

    write_json(diagnostics_directory / "osman_shuffled_label_control.json", {
        "description": "Labels shuffled by person; original folds and model selection repeated",
        "auroc_values": shuffled_scores,
        "mean": float(np.mean(shuffled_scores)),
        "minimum": float(np.min(shuffled_scores)),
        "maximum": float(np.max(shuffled_scores)),
    })


if __name__ == "__main__":
    main()
