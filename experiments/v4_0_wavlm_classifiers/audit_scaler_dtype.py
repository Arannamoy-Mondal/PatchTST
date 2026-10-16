"""v4.0-audit.1: resolve a checker-only weight dtype mismatch without retraining."""

import argparse
import hashlib
import inspect
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from checks import replay_check
from common import PROJECT, fingerprint, training_weights, write_json
from features import feature_table


def protected_hashes(parent):
    paths = [parent / "summary.json", parent / "pooled_metrics.csv", parent / "fold_mean_sd.csv", parent / "replay_checks.json"]
    for scenario in ["main", "native16k"]:
        paths += list((parent / scenario).rglob("model.joblib")) + list((parent / scenario).rglob("model.ubj"))
        paths += list((parent / scenario).rglob("speaker_predictions.csv"))
        paths += list((parent / scenario).rglob("recording_predictions.csv"))
        paths += list((parent / scenario).rglob("selection.json"))
    return {path.relative_to(PROJECT).as_posix(): fingerprint(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    arguments = parser.parse_args()
    parent = arguments.run.resolve()
    protocol = json.loads((parent / "protocol.json").read_text())
    settings = json.loads((parent / "settings.json").read_text())
    if protocol["experiment"] != "v4.0":
        print("ERROR: expected the frozen v4.0 parent run.")
        return False
    allowed_change = "experiments/v4_0_wavlm_classifiers/checks.py"
    for name, expected in protocol["source_sha256"].items():
        if name != allowed_change and fingerprint(PROJECT / name) != expected:
            print("ERROR: trained implementation changed outside the checker:", name)
            return False
    initial = json.loads((parent / "replay_checks.json").read_text())
    before = protected_hashes(parent)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    code_hash = hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name("checks.py").read_bytes()).hexdigest()[:12]
    directory = PROJECT / "artifacts/runs" / (stamp + "_v4.0-audit.1_" + code_hash)
    directory.mkdir(parents=True)
    for path in Path(__file__).parent.glob("*.py"):
        target = directory / "source" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    scaler_source = inspect.getsource(StandardScaler.partial_fit)
    (directory / "source" / "sklearn_scaler_partial_fit.txt").write_text(scaler_source)
    if "_check_sample_weight(sample_weight, X, dtype=X.dtype)" not in scaler_source:
        print("ERROR: the installed sklearn scaler does not use the expected input-dtype weight cast.")
        return False
    manifest = pd.read_csv(parent / "embedding_manifest.csv")
    features = feature_table(manifest)
    if features is None:
        return False
    columns = [name for name in features.columns if name.startswith("embedding_")]
    comparisons = []
    for scenario in ["main", "native16k"]:
        rows = features if scenario == "main" else features[features.native_sr == 16000]
        for fold in range(5):
            training = rows[rows.fold != fold]
            model = joblib.load(parent / scenario / "logistic_regression" / ("fold_" + str(fold)) / "model.joblib")
            values = training[columns].to_numpy()
            original_weights = training_weights(training)
            original_expectation = np.average(values, axis=0, weights=original_weights)
            actual_weights = original_weights.astype(values.dtype).astype(np.float64)
            dtype_expectation = np.average(values.astype(np.float64), axis=0, weights=actual_weights)
            fitted_mean = model.named_steps["scale"].mean_
            passed = bool(np.allclose(fitted_mean, dtype_expectation, atol=1e-10, rtol=1e-10))
            comparisons.append({"scenario": scenario, "fold": fold, "feature_dtype": str(values.dtype),
                "provided_weight_dtype": str(original_weights.dtype),
                "maximum_original_expectation_error": float(np.max(np.abs(fitted_mean - original_expectation))),
                "maximum_dtype_matched_error": float(np.max(np.abs(fitted_mean - dtype_expectation))),
                "tolerance_unchanged": {"absolute": 1e-10, "relative": 1e-10}, "passed": passed})
    resolved = replay_check(parent, settings, output_path=directory / "resolved_replay_checks.json")
    after = protected_hashes(parent)
    passed = resolved is not None and resolved["all_passed"] and before == after and all(row["passed"] for row in comparisons)
    record = {"audit_version": "v4.0-audit.1", "parent": parent.relative_to(PROJECT).as_posix(),
        "initial_check": "Nine scaler-provenance checks used uncast float64 weights although sklearn fits float32 input using float32-cast weights.",
        "initial_failed_fit_checks": sum(not row["training_only_fit"] for row in initial["saved_model_replays"]),
        "corrected_check": "Reproduce sklearn's dtype cast, then its float64 accumulation; do not relax either tolerance.",
        "scaler_comparisons": comparisons, "all_models_and_predictions_unchanged": before == after,
        "protected_sha256": before, "source_sha256": {path.relative_to(directory).as_posix(): fingerprint(path) for path in (directory / "source").iterdir()},
        "no_retraining": True, "no_changed_feature_extraction": True, "all_passed": bool(passed)}
    write_json(directory / "resolution.json", record)
    if not passed:
        print("ERROR: dtype resolution or full replay did not pass; the original failure remains preserved.")
        return False
    write_json(parent / "status.json", {"stage": "completed_with_documented_scaler_check_resolution", "complete": True,
        "original_check": "replay_checks.json", "verification_audit": "v4.0-audit.1",
        "verification_file": (directory / "resolved_replay_checks.json").relative_to(PROJECT).as_posix(),
        "resolution_file": (directory / "resolution.json").relative_to(PROJECT).as_posix()})
    print("RESOLVED_AUDIT=" + directory.relative_to(PROJECT).as_posix())
    print("All 30 model replays, ten conditional branches and both fresh-feature examples passed. Models and predictions were not changed.")
    return True


if __name__ == "__main__":
    if not main():
        sys.exit(1)
