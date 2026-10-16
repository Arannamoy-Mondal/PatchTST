"""Beginner-readable checks. Failures print their cause and return a failing exit code."""

import sys

import numpy as np
import pandas as pd
import torch
import yaml

from classifiers import fit_model, predict_recordings
from common import PROJECT, people_from_recordings, score_people, training_weights
from features import window_positions, load_encoder
from fine_tune import UpperLayerClassifier


def main():
    settings = yaml.safe_load((PROJECT / "experiments/v4_0_wavlm_classifiers/config.yaml").read_text())
    base = yaml.safe_load((PROJECT / settings["base_config"]).read_text())
    settings["data"] = base["data"]
    settings["splits"] = base["splits"]
    settings["train"] = {"threshold": base["train"]["threshold"]}
    failures = 0
    for sample_count, expected in [(5, [0]), (10, [0]), (20, [0, 10]), (25, [0, 10, 15])]:
        actual = window_positions(sample_count, 10)
        if actual != expected:
            failures += 1
            print("ERROR: window starts", actual, "expected", expected)
    print("Window coverage checks finished.")
    rows = pd.DataFrame({"speaker_id": ["a", "a", "b", "c", "c", "c"], "label": [0, 0, 0, 1, 1, 1],
                         "path": ["a1", "a2", "b1", "c1", "c2", "c3"]})
    rows["weight"] = training_weights(rows)
    sums = rows.groupby("label").weight.sum()
    if abs(sums.iloc[0] - sums.iloc[1]) > 1e-12:
        failures += 1
        print("ERROR: training weights must give both classes equal total influence.")
    print("Training class weights:", sums.to_dict())
    generator = np.random.default_rng(42)
    values = generator.normal(size=(24, 1024))
    columns = ["embedding_%04d" % number for number in range(1024)]
    table = pd.DataFrame(values, columns=columns)
    table["speaker_id"] = ["person_" + str(number // 2) for number in range(24)]
    table["path"] = ["file_" + str(number) for number in range(24)]
    table["label"] = [(number // 2) % 2 for number in range(24)]
    table["sheet_flag"] = False
    table["task"] = "VA1"
    training, testing = table.iloc[:16], table.iloc[16:]
    candidates = [("logistic_regression", {"C": 0.1}),
                  ("random_forest", {"n_estimators": 4, "max_depth": 2}),
                  ("xgboost", {"n_estimators": 4, "max_depth": 1})]
    for method, parameters in candidates:
        model = fit_model(method, parameters, training, columns, settings, 42)
        recordings = predict_recordings(model, testing, columns)
        people = people_from_recordings(recordings, testing, 0.5, 0)
        if len(people) != 4 or not np.isfinite(people.probability).all():
            failures += 1
            print("ERROR:", method, "must produce four finite speaker scores.")
        print("Synthetic classifier check:", method, score_people(people, 0.5))
    loaded = load_encoder(settings)
    if loaded is None:
        return False
    extractor, encoder = loaded
    model = UpperLayerClassifier(encoder, 2, settings["fine_tuning"]["dropout"])
    model.training_mode()
    inputs = dict(extractor(generator.normal(size=16000).astype(np.float32), sampling_rate=16000, return_tensors="pt"))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(inputs), torch.tensor([1.0]))
    loss.backward()
    frozen_layers_have_no_grad = all(parameter.grad is None for layer in encoder.encoder.layers[:-2] for parameter in layer.parameters())
    upper_layers_have_grad = all(parameter.grad is not None for layer in encoder.encoder.layers[-2:] for parameter in layer.parameters())
    if not frozen_layers_have_no_grad or not upper_layers_have_grad or not torch.isfinite(loss):
        failures += 1
        print("ERROR: only the two selected upper layers and head should receive finite training gradients.")
    print("Upper-layer gradient smoke check:", frozen_layers_have_no_grad and upper_layers_have_grad)
    print("SELF_CHECK_FAILURES=" + str(failures))
    return failures == 0


if __name__ == "__main__":
    if not main():
        sys.exit(1)
