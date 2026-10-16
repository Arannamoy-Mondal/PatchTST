"""Replay saved models and verify identities, selection, aggregation and inputs."""

import json

import joblib
import numpy as np
import pandas as pd

from classifiers import predict_recordings
from common import METHODS, PROJECT, fingerprint, people_from_recordings, score_people, training_weights, write_json
from features import extract_one, feature_table, load_encoder


def replay_check(run_directory, settings, fresh_features=True, output_path=None):
    manifest = pd.read_csv(run_directory / "embedding_manifest.csv")
    features = feature_table(manifest)
    if features is None:
        return None
    columns = [name for name in features.columns if name.startswith("embedding_")]
    checks = []
    for scenario in ["main", "native16k"]:
        rows = features if scenario == "main" else features[features.native_sr == 16000]
        for method in METHODS:
            pooled = pd.read_csv(run_directory / scenario / method / "speaker_predictions.csv")
            for fold in range(settings["splits"]["n_splits"]):
                directory = run_directory / scenario / method / ("fold_" + str(fold))
                split = json.loads((run_directory / scenario / "splits" / ("fold_" + str(fold)) / "split.json").read_text())
                train_ids, test_ids = set(split["outer_train"]), set(split["outer_test"])
                training = rows[rows.speaker_id.isin(train_ids)]
                testing = rows[rows.speaker_id.isin(test_ids)]
                separated = not bool(train_ids & test_ids)
                for inner in split["inner"]:
                    inner_train, validation = set(inner["train"]), set(inner["validation"])
                    separated = separated and not bool(inner_train & validation) and inner_train | validation == train_ids
                model = joblib.load(directory / "model.joblib")
                recording_scores = predict_recordings(model, testing, columns)
                replay = people_from_recordings(recording_scores, testing, settings["train"]["threshold"], fold)
                saved = pd.read_csv(directory / "speaker_predictions.csv")
                joined = replay.merge(saved, on="speaker_id", suffixes=("_replay", "_saved"), validate="one_to_one")
                difference = float(np.max(np.abs(joined.probability_replay - joined.probability_saved)))
                pooled_fold = pooled[pooled.fold == fold].sort_values("speaker_id")
                saved_sorted = saved.sort_values("speaker_id")
                pooled_matches = (
                    pooled.speaker_id.is_unique and set(saved.speaker_id) == test_ids
                    and pooled_fold.speaker_id.tolist() == saved_sorted.speaker_id.tolist()
                    and np.allclose(pooled_fold.probability, saved_sorted.probability, atol=1e-12, rtol=0)
                )
                fit = json.loads((directory / "fit.json").read_text())
                fit_rows = pd.read_csv(directory / "fit_rows.csv")
                fit_matches = (
                    set(fit["training_speakers"]) == train_ids and fit["training_recordings"] == len(training)
                    and fit["feature_columns"] == columns and fit_rows.path.tolist() == training.path.tolist()
                    and np.allclose(fit_rows.weight, training_weights(training), atol=1e-12, rtol=0)
                )
                if method == "logistic_regression":
                    values = training[columns].to_numpy()
                    # sklearn first casts weights to the input dtype, then accumulates in float64.
                    scaler_weights = training_weights(training).astype(values.dtype).astype(np.float64)
                    expected_mean = np.average(values.astype(np.float64), axis=0, weights=scaler_weights)
                    fit_matches = fit_matches and np.allclose(model.named_steps["scale"].mean_, expected_mean, atol=1e-10, rtol=1e-10)
                selection = json.loads((directory / "selection.json").read_text())
                inner_predictions = pd.read_csv(directory / "inner_predictions.csv")
                selection_matches = True
                for candidate in selection["candidates"]:
                    values = []
                    for inner in split["inner"]:
                        validation = inner_predictions[(inner_predictions.candidate == candidate["candidate"]) & (inner_predictions.inner_fold == inner["inner_fold"])]
                        if set(validation.speaker_id) != set(inner["validation"]) or not validation.speaker_id.is_unique:
                            selection_matches = False
                        values.append(score_people(validation, settings["train"]["threshold"])["auroc"])
                    if abs(float(np.mean(values)) - candidate["mean_inner_speaker_auroc"]) > 1e-12:
                        selection_matches = False
                chosen = selection["selected"]
                expected_best = selection["candidates"][0]
                for candidate in selection["candidates"][1:]:
                    if candidate["mean_inner_speaker_auroc"] > expected_best["mean_inner_speaker_auroc"]:
                        expected_best = candidate
                selection_matches = selection_matches and chosen == expected_best
                passed = separated and pooled_matches and fit_matches and selection_matches and difference < 1e-6
                checks.append({"scenario": scenario, "method": method, "fold": fold, "speaker_separation": separated,
                    "training_only_fit": fit_matches, "selection_recomputed": selection_matches,
                    "pooled_source_matches": pooled_matches, "maximum_probability_difference": difference, "passed": bool(passed)})
                if not passed:
                    print("ERROR: saved-model replay or provenance check failed:", scenario, method, fold)
    raw_match = True
    for row in manifest.itertuples(index=False):
        if fingerprint(PROJECT / row.path) != row.audio_sha256:
            raw_match = False
            print("ERROR: raw recording fingerprint changed:", row.path)
    fresh_checks = []
    if fresh_features:
        loaded = load_encoder(settings)
        if loaded is None:
            return None
        # Predeclared deterministic examples: first path at each original sample rate.
        for rate in sorted(manifest.native_sr.unique()):
            row = manifest[manifest.native_sr == rate].sort_values("path").iloc[0]
            fresh = extract_one(PROJECT / row.path, settings, *loaded)
            if fresh is None:
                return None
            cached = np.load(PROJECT / row.embedding_path, allow_pickle=False)
            difference = float(np.max(np.abs(fresh[0] - cached)))
            passed = difference < 1e-6
            fresh_checks.append({"path": row.path, "native_sr": int(rate), "maximum_embedding_difference": difference, "passed": passed})
            if not passed:
                print("ERROR: fresh WavLM feature replay exceeded 1e-6:", row.path)
    conditional_checks = []
    from fine_tune import UpperLayerClassifier, audio_windows, predict_windows
    import torch
    for scenario in ["main", "native16k"]:
        recordings = manifest if scenario == "main" else manifest[manifest.native_sr == 16000]
        adaptive = pd.read_csv(run_directory / scenario / "conditional_fine_tuning" / "speaker_predictions.csv")
        for fold in range(settings["splits"]["n_splits"]):
            directory = run_directory / scenario / "conditional_fine_tuning" / ("fold_" + str(fold))
            decision = json.loads((directory / "decision.json").read_text())
            baseline = run_directory / scenario / "logistic_regression" / ("fold_" + str(fold))
            selection = json.loads((baseline / "selection.json").read_text())
            expected_gate = selection["selected"]["mean_inner_speaker_auroc"] < settings["fine_tuning"]["trigger_auroc"]
            gate_passes = decision["triggered"] == expected_gate and not decision["test_labels_used_for_trigger"]
            if decision["triggered"]:
                loaded = load_encoder(settings)
                if loaded is None:
                    return None
                extractor, backbone = loaded
                fit = json.loads((directory / "fit.json").read_text())
                model = UpperLayerClassifier(backbone, fit["parameters"]["upper_layers"], settings["fine_tuning"]["dropout"])
                state = torch.load(directory / "upper_layers_and_head.pt", map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=False)
                testing = recordings[recordings.fold == fold]
                window_scores = predict_windows(model, audio_windows(testing, settings, extractor, False))
                replay = people_from_recordings(window_scores, testing, settings["train"]["threshold"], fold)
                split = json.loads((directory / "split.json").read_text())
                train_ids, test_ids = set(split["outer_train"]), set(split["outer_test"])
                gate_passes = gate_passes and not bool(train_ids & test_ids) and set(fit["training_speakers"]) == train_ids
                del model, backbone, extractor
            else:
                replay = pd.read_csv(baseline / "speaker_predictions.csv")
            saved = adaptive[adaptive.fold == fold]
            joined = replay.merge(saved, on="speaker_id", suffixes=("_replay", "_saved"), validate="one_to_one")
            difference = float(np.max(np.abs(joined.probability_replay - joined.probability_saved)))
            passed = gate_passes and len(joined) == len(saved) and difference < 1e-6
            conditional_checks.append({"scenario": scenario, "fold": fold, "triggered": decision["triggered"],
                "training_only_gate": gate_passes, "maximum_probability_difference": difference, "passed": bool(passed)})
            if not passed:
                print("ERROR: conditional branch or gate replay failed:", scenario, fold)
    result = {"saved_model_replays": checks, "raw_recordings_rehashed": len(manifest), "all_raw_hashes_match": raw_match,
        "fresh_feature_replays": fresh_checks, "fresh_feature_scope": "first path per native rate, not all recordings",
        "conditional_replays": conditional_checks,
        "all_passed": bool(raw_match and all(row["passed"] for row in checks) and all(row["passed"] for row in fresh_checks)
                           and all(row["passed"] for row in conditional_checks)),
        "interpretation": "Speaker-disjoint and leakage-checked; recording-condition confounding remains unresolved."}
    if output_path is None:
        output_path = run_directory / "replay_checks.json"
    write_json(output_path, result)
    return result
