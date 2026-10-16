"""Replay saved models and check that every held-out person was scored once."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix

from patchtst.comparison import MEMBER_NAMES, write_json
from patchtst.metrics import speaker_table
from patchtst.model.patchtst import PatchTST
from patchtst.train import predict_windows, windows_for


def person_ids(recordings):
    return set(recordings.speaker_id)


def confusion_counts(people):
    decisions = (people.probability >= people.threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        people.label,
        decisions,
        labels=[0, 1],
    ).ravel()
    return [int(true_negative), int(false_positive), int(false_negative), int(true_positive)]


def logistic_replay(model, recordings, member_name, acoustic_features):
    """Recreate one saved logistic model's person scores from its inputs."""
    input_names = list(model.feature_names_in_)
    if member_name == "osman":
        feature_rows = acoustic_features[acoustic_features.path.isin(recordings.path)].copy()
        feature_rows["probability"] = model.predict_proba(feature_rows[input_names])[:, 1]
        person_scores = speaker_table(feature_rows[["speaker_id", "path", "label", "probability"]])
        fitted_row_count = int(
            model.named_steps["preprocessing"]
            .named_transformers_["acoustic"]
            .named_steps["scale"]
            .n_samples_seen_
        )
    else:
        person_rows = []
        for speaker_id, person_recordings in recordings.groupby("speaker_id"):
            embedding_paths = person_recordings.embedding_path
            embedding = np.stack([np.load(path) for path in embedding_paths]).mean(axis=0)
            person_row = {"speaker_id": speaker_id, "label": int(person_recordings.label.iloc[0])}
            for feature_number, feature_value in enumerate(embedding):
                person_row["embedding_%04d" % feature_number] = float(feature_value)
            person_rows.append(person_row)
        feature_rows = pd.DataFrame(person_rows)
        feature_rows["probability"] = model.predict_proba(feature_rows[input_names])[:, 1]
        person_scores = feature_rows[["speaker_id", "label", "probability"]]
        fitted_row_count = int(model.named_steps["preprocessing"].named_steps["scale"].n_samples_seen_)
    return person_scores, input_names, fitted_row_count


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--run", required=True)
    arguments = argument_parser.parse_args()

    run_directory = Path(arguments.run)
    settings = yaml.safe_load((run_directory / "config.yaml").read_text())
    protocol = json.loads((run_directory / "protocol.json").read_text())
    torch.set_num_threads(settings["train"]["threads"])
    acoustic_features = pd.read_csv(run_directory / "osman_acoustic_features.csv")
    audit_rows = []

    for scenario_name in ("main", "native16k"):
        if scenario_name == "main":
            expected_ids = set(protocol["common_speakers"])
        else:
            expected_ids = set(protocol["native_speakers"])
        for member_name in MEMBER_NAMES:
            member_directory = run_directory / scenario_name / member_name
            recordings = pd.read_csv(member_directory / "manifest.csv")
            pooled_predictions = pd.read_csv(member_directory / "speaker_predictions.csv")
            metrics = json.loads((member_directory / "metrics.json").read_text())
            label_by_person = recordings.groupby("speaker_id").label.first()
            coverage_is_valid = pooled_predictions.speaker_id.is_unique and person_ids(pooled_predictions) == expected_ids
            labels_match = np.array_equal(
                pooled_predictions.speaker_id.map(label_by_person).to_numpy(),
                pooled_predictions.label.to_numpy(),
            )
            counts_match = confusion_counts(pooled_predictions) == [
                metrics["pooled"][count_name] for count_name in ("tn", "fp", "fn", "tp")
            ]
            if not coverage_is_valid or not labels_match or not counts_match:
                print("ERROR: pooled predictions do not match the saved manifest or metrics.")
                return

            for fold_number in range(settings["splits"]["n_splits"]):
                fold_directory = member_directory / ("fold_" + str(fold_number))
                split = json.loads((fold_directory / "split.json").read_text())
                training_recordings = recordings[recordings.fold != fold_number]
                test_recordings = recordings[recordings.fold == fold_number]
                training_ids = person_ids(training_recordings)
                test_ids = person_ids(test_recordings)
                checks = {
                    "outer_training_ids_match": set(split["outer_train"]) == training_ids,
                    "outer_test_ids_match": set(split["outer_test"]) == test_ids,
                    "no_person_overlap": not bool(training_ids & test_ids),
                    "no_identical_audio_overlap": not bool(
                        set(training_recordings.audio_sha256) & set(test_recordings.audio_sha256)
                    ),
                    "fixed_threshold": bool((
                        pooled_predictions.loc[pooled_predictions.fold == fold_number, "threshold"]
                        == settings["train"]["threshold"]
                    ).all()),
                }
                if member_name == "arannamoy":
                    inner_training_ids = set(split["inner_train"])
                    validation_ids = set(split["inner_validation"])
                    checks["inner_split_disjoint"] = not bool(inner_training_ids & validation_ids)
                    checks["inner_split_uses_training_only"] = inner_training_ids | validation_ids == training_ids
                    model = PatchTST(settings).eval()
                    checkpoint_loaded = model.load_v2_checkpoint(fold_directory / "checkpoint.pt")
                    if not checkpoint_loaded:
                        print("ERROR: could not load the saved PatchTST checkpoint.")
                        return
                    windows, labels, speaker_ids, paths = windows_for(test_recordings, settings, False)
                    replay_scores = speaker_table(pd.DataFrame({
                        "speaker_id": speaker_ids,
                        "path": paths,
                        "label": labels,
                        "probability": predict_windows(model, windows, settings),
                    }))
                else:
                    for inner_split in split["inner"]:
                        inner_training_ids = set(inner_split["train"])
                        validation_ids = set(inner_split["validation"])
                        split_number = inner_split["inner_fold"]
                        checks["inner_%s_disjoint" % split_number] = not bool(inner_training_ids & validation_ids)
                        checks["inner_%s_training_only" % split_number] = (
                            inner_training_ids | validation_ids == training_ids
                        )
                    model = joblib.load(fold_directory / "model.joblib")
                    replay_scores, input_names, fitted_count = logistic_replay(
                        model,
                        test_recordings,
                        member_name,
                        acoustic_features,
                    )
                    forbidden_inputs = {"label", "age", "sex", "native_sr", "speaker_id", "path", "fold"}
                    checks["no_label_or_metadata_input"] = not bool(set(input_names) & forbidden_inputs)
                    if member_name == "osman":
                        checks["training_only_scaler"] = fitted_count == len(training_recordings)
                    else:
                        checks["training_only_scaler"] = fitted_count == len(training_ids)

                saved_scores = pd.read_csv(fold_directory / "speaker_predictions.csv")
                replay_by_person = replay_scores.set_index("speaker_id").probability
                saved_by_person = saved_scores.set_index("speaker_id").probability
                probability_difference = (replay_by_person - saved_by_person).abs()
                checks["saved_predictions_match"] = bool((probability_difference < 1e-5).all())
                if scenario_name == "native16k":
                    checks["native_16khz_only"] = bool((recordings.native_sr == settings["data"]["target_sr"]).all())
                all_checks_passed = all(checks.values())
                print(
                    "PASS" if all_checks_passed else "ERROR",
                    member_name,
                    scenario_name,
                    "fold",
                    fold_number,
                    "| maximum probability difference:",
                    float(probability_difference.max()),
                    flush=True,
                )
                audit_rows.append({
                    "member": member_name,
                    "scenario": scenario_name,
                    "fold": fold_number,
                    "checks": checks,
                })

    diagnostics_directory = run_directory / "diagnostics"
    diagnostics_directory.mkdir(exist_ok=True)
    write_json(diagnostics_directory / "all_members_prediction_audit.json", audit_rows)
    if all(all(row["checks"].values()) for row in audit_rows):
        print("PASS: all 30 saved outer models and their person splits were verified.")
    else:
        print("ERROR: at least one verification check failed.")


if __name__ == "__main__":
    main()
