"""Conditional upper-layer WavLM adaptation; decisions use training people only."""

import gc
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid, StratifiedShuffleSplit
from torch import nn

from patchtst.data.audio import load_audio
from common import PROJECT, people_from_recordings, score_people, training_weights, write_json
from features import load_encoder, window_positions


class UpperLayerClassifier(nn.Module):
    def __init__(self, backbone, layer_count, dropout):
        super().__init__()
        self.backbone = backbone
        self.layer_count = layer_count
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.config.hidden_size, 1))
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        for layer in backbone.encoder.layers[-layer_count:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        backbone.config.apply_spec_augment = False
        backbone.config.layerdrop = 0.0

    def training_mode(self):
        # Lower layers remain deterministic; only selected upper layers and head train.
        self.backbone.eval()
        for layer in self.backbone.encoder.layers[-self.layer_count:]:
            layer.train()
        self.head.train()

    def forward(self, inputs):
        hidden = self.backbone(**inputs).last_hidden_state
        return self.head(hidden.mean(dim=1)).squeeze(1)


def audio_windows(recordings, settings, extractor, limit):
    items = []
    rate = settings["data"]["target_sr"]
    samples = round(settings["features"]["seconds"] * rate)
    for row in recordings.itertuples(index=False):
        audio = load_audio(PROJECT / row.path, rate)
        starts = window_positions(len(audio), samples)
        maximum = settings["fine_tuning"]["max_windows_per_file"]
        if limit and len(starts) > maximum:
            positions = np.linspace(0, len(starts) - 1, maximum).round().astype(int)
            starts = [starts[position] for position in positions]
        for start in starts:
            window = audio[start:start + samples]
            if len(window) < settings["features"]["minimum_samples"]:
                window = np.pad(window, (0, settings["features"]["minimum_samples"] - len(window)), mode="edge")
            inputs = dict(extractor(window, sampling_rate=rate, return_tensors="pt", return_attention_mask=True))
            items.append({"inputs": inputs, "speaker_id": row.speaker_id, "path": row.path, "label": int(row.label)})
    return items


def predict_windows(model, items):
    model.eval()
    rows = []
    with torch.inference_mode():
        for item in items:
            probability = float(torch.sigmoid(model(item["inputs"])).item())
            rows.append({"speaker_id": item["speaker_id"], "path": item["path"], "label": item["label"], "probability": probability})
    return pd.DataFrame(rows)


def fit_upper_layers(training, validation, settings, parameters, seed, epochs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    loaded = load_encoder(settings)
    if loaded is None:
        return None
    extractor, backbone = loaded
    model = UpperLayerClassifier(backbone, parameters["upper_layers"], settings["fine_tuning"]["dropout"])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=parameters["learning_rate"], weight_decay=settings["fine_tuning"]["weight_decay"],
    )
    items = audio_windows(training, settings, extractor, True)
    window_rows = pd.DataFrame([{key: item[key] for key in ["speaker_id", "path", "label"]} for item in items])
    file_rows = window_rows.drop_duplicates("path").reset_index(drop=True)
    file_weights = dict(zip(file_rows.path, training_weights(file_rows)))
    windows_per_file = window_rows.groupby("path").size()
    weights = np.array([file_weights[item["path"]] / windows_per_file[item["path"]] for item in items])
    weights = weights / weights.mean()
    validation_items = audio_windows(validation, settings, extractor, False) if validation is not None else None
    best_loss = float("inf")
    best_epoch = 1
    wait = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.training_mode()
        order = np.random.default_rng(seed + epoch).permutation(len(items))
        total_loss = 0.0
        for position in order:
            item = items[position]
            optimizer.zero_grad()
            logits = model(item["inputs"])
            target = torch.tensor([float(item["label"])])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target) * float(weights[position])
            if not torch.isfinite(loss):
                print("ERROR: upper-layer training loss is not finite.")
                return None
            loss.backward()
            nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
            optimizer.step()
            total_loss += float(loss.item())
        record = {"epoch": epoch, "training_loss": total_loss / len(items)}
        if validation_items is not None:
            window_scores = predict_windows(model, validation_items)
            people = people_from_recordings(window_scores, validation, settings["train"]["threshold"], 0)
            probabilities = np.clip(people.probability.to_numpy(), 1e-7, 1 - 1e-7)
            labels = people.label.to_numpy()
            value = float(-(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities)).mean())
            record["validation_speaker_loss"] = value
            record["validation_speaker_auroc"] = score_people(people, settings["train"]["threshold"])["auroc"]
            if value < best_loss - 0.0001:
                best_loss, best_epoch, wait = value, epoch, 0
            else:
                wait += 1
        history.append(record)
        print("Upper-layer epoch", epoch, record, flush=True)
        if validation is not None and wait >= settings["fine_tuning"]["patience"]:
            break
    return model, extractor, history, best_epoch


def run_conditional_fold(recordings, settings, fold, baseline_selection, destination):
    destination.mkdir(parents=True, exist_ok=True)
    mean_auroc = baseline_selection["selected"]["mean_inner_speaker_auroc"]
    triggered = mean_auroc < settings["fine_tuning"]["trigger_auroc"]
    decision = {"fold": fold, "trigger_source": settings["fine_tuning"]["trigger_source"],
                "baseline_inner_auroc": mean_auroc, "trigger_threshold": settings["fine_tuning"]["trigger_auroc"],
                "triggered": triggered, "test_labels_used_for_trigger": False}
    write_json(destination / "decision.json", decision)
    if not triggered:
        print("Fine-tuning not triggered in fold", fold, "training-side AUROC:", mean_auroc, flush=True)
        return decision
    training = recordings[recordings.fold != fold]
    testing = recordings[recordings.fold == fold]
    people = training[["speaker_id", "label"]].drop_duplicates().sort_values("speaker_id").reset_index(drop=True)
    seed = settings["splits"]["seed"] + 200 + fold
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=settings["fine_tuning"]["validation_fraction"], random_state=seed)
    fit_positions, validation_positions = next(splitter.split(people, people.label))
    fit_ids = set(people.iloc[fit_positions].speaker_id)
    validation_ids = set(people.iloc[validation_positions].speaker_id)
    candidates = list(ParameterGrid({name: settings["fine_tuning"][name] for name in ["upper_layers", "learning_rate"]}))
    records = []
    for parameters in candidates:
        selection = fit_upper_layers(training[training.speaker_id.isin(fit_ids)], training[training.speaker_id.isin(validation_ids)],
                                     settings, parameters, seed, settings["fine_tuning"]["epochs"])
        if selection is None:
            return None
        model, extractor, history, epoch = selection
        selected_history = history[epoch - 1]
        records.append({"parameters": parameters, "selected_epochs": epoch,
                        "validation_auroc": selected_history["validation_speaker_auroc"], "history": history})
        del model, extractor
        gc.collect()
    best = records[0]
    for candidate in records[1:]:
        if candidate["validation_auroc"] > best["validation_auroc"]:
            best = candidate
    write_json(destination / "selection.json", {"selected": best, "candidates": records})
    write_json(destination / "split.json", {"inner_train": sorted(fit_ids), "inner_validation": sorted(validation_ids),
               "outer_train": sorted(training.speaker_id.unique()), "outer_test": sorted(testing.speaker_id.unique())})
    refit = fit_upper_layers(training, None, settings, best["parameters"], seed, best["selected_epochs"])
    if refit is None:
        return None
    model, extractor, history, unused_epoch = refit
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = {name: value for name, value in model.state_dict().items() if name in trainable_names}
    torch.save(state, destination / "upper_layers_and_head.pt")
    write_json(destination / "refit_history.json", history)
    write_json(destination / "fit.json", {"parameters": best["parameters"], "epochs": best["selected_epochs"],
               "training_speakers": sorted(training.speaker_id.unique()), "trainable_names": sorted(trainable_names)})
    window_scores = predict_windows(model, audio_windows(testing, settings, extractor, False))
    window_scores.to_csv(destination / "window_predictions.csv", index=False)
    scores = people_from_recordings(window_scores, testing, settings["train"]["threshold"], fold)
    scores.to_csv(destination / "speaker_predictions.csv", index=False)
    del model, extractor
    gc.collect()
    return decision
