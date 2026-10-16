"""Speaker-safe training with inner epoch selection and outer-training refit."""

import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit

from patchtst.data.audio import slice_windows
from patchtst.metrics import classification_scores, speaker_table
from patchtst.model.patchtst import PatchTST

_SPEC_CACHE = {}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _windows_for(frame, cfg, limit):
    arrays, labels, speakers, paths = [], [], [], []
    for row in frame.itertuples(index=False):
        if row.feature_path not in _SPEC_CACHE:
            _SPEC_CACHE[row.feature_path] = np.load(row.feature_path)
        windows = slice_windows(_SPEC_CACHE[row.feature_path], cfg, limit=limit)
        arrays.append(windows)
        labels.extend([row.label] * len(windows))
        speakers.extend([row.speaker_id] * len(windows))
        paths.extend([row.path] * len(windows))
    return np.concatenate(arrays), np.asarray(labels, dtype=np.float32), np.asarray(speakers), np.asarray(paths)


def training_weights(labels, speakers, paths):
    rows = pd.DataFrame({"label": labels, "speaker_id": speakers, "path": paths})
    windows_per_file = rows.groupby("path").size()
    files_per_speaker = rows.groupby("speaker_id").path.nunique()
    speakers_per_class = rows.drop_duplicates("speaker_id").groupby("label").size()
    weights = 1.0 / (rows.path.map(windows_per_file) * rows.speaker_id.map(files_per_speaker) * rows.label.map(speakers_per_class))
    return (weights / weights.mean()).to_numpy(dtype=np.float32)


def predict_windows(model, arrays, cfg):
    model.eval()
    probabilities = []
    batch_size = cfg["train"]["batch_size"]
    with torch.inference_mode():
        for start in range(0, len(arrays), batch_size):
            logits = model(torch.from_numpy(arrays[start:start + batch_size]))
            probabilities.append(torch.sigmoid(logits).squeeze(1).numpy())
    return np.concatenate(probabilities)


def fit_model(training, validation, cfg, seed, epochs):
    seed_everything(seed)
    x, y, speakers, paths = training
    weights = training_weights(y, speakers, paths)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(weights)),
                        batch_size=cfg["train"]["batch_size"], shuffle=True, generator=generator)
    model = PatchTST(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    history = []
    best_loss = float("inf")
    best_epoch = 1
    waited = 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        for batch_x, batch_y, batch_weight in loader:
            optimizer.zero_grad()
            logits = model(batch_x).squeeze(1)
            loss = (loss_function(logits, batch_y) * batch_weight).mean()
            if not torch.isfinite(loss):
                print("ERROR: non-finite training loss; no metrics will be published.")
                return None
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += loss.item() * len(batch_x)
        entry = {"epoch": epoch, "train_loss": loss_total / len(x)}
        if validation is not None:
            vx, vy, vs, vp = validation
            probability = predict_windows(model, vx, cfg)
            table = speaker_table(pd.DataFrame({"speaker_id": vs, "path": vp, "label": vy, "probability": probability}))
            p = np.clip(table.probability.to_numpy(), 1e-7, 1 - 1e-7)
            target = table.label.to_numpy()
            value = float(-(target * np.log(p) + (1 - target) * np.log(1 - p)).mean())
            entry["val_speaker_loss"] = value
            if value < best_loss - 0.0001:
                best_loss, best_epoch, waited = value, epoch, 0
            else:
                waited += 1
        history.append(entry)
        print("Epoch", epoch, entry, "elapsed seconds", round(time.monotonic() - started, 1), flush=True)
        if validation is not None and waited >= cfg["train"]["patience"]:
            break
    return model, history, best_epoch


def train_fold(train_files, test_files, cfg, device=None, fold=0):
    torch.set_num_threads(cfg["train"]["threads"])
    speakers = train_files.groupby("speaker_id", as_index=False).agg(label=("label", "first"))
    seed = cfg["splits"]["seed"] + fold
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=cfg["train"]["validation_fraction"], random_state=seed)
    fit_index, val_index = next(splitter.split(speakers, speakers.label))
    fit_ids = set(speakers.iloc[fit_index].speaker_id)
    val_ids = set(speakers.iloc[val_index].speaker_id)
    test_ids = set(test_files.speaker_id)
    if fit_ids & val_ids or fit_ids & test_ids or val_ids & test_ids:
        print("ERROR: speaker overlap in PatchTST split.")
        return None
    fit_files = train_files[train_files.speaker_id.isin(fit_ids)]
    val_files = train_files[train_files.speaker_id.isin(val_ids)]
    fitted = fit_model(_windows_for(fit_files, cfg, True), _windows_for(val_files, cfg, False), cfg, seed, cfg["train"]["epochs"])
    if fitted is None:
        return None
    unused_model, selection_history, selected_epochs = fitted
    del unused_model
    print("Refitting on all outer-training speakers for", selected_epochs, "epochs.", flush=True)
    fitted = fit_model(_windows_for(train_files, cfg, True), None, cfg, seed, selected_epochs)
    if fitted is None:
        return None
    model, refit_history, unused_epoch = fitted
    x, y, ids, paths = _windows_for(test_files, cfg, False)
    windows = pd.DataFrame({"speaker_id": ids, "path": paths, "label": y.astype(int), "probability": predict_windows(model, x, cfg)})
    windows["task"] = windows.path.map(test_files.set_index("path").task)
    table = speaker_table(windows)
    threshold = cfg["train"]["threshold"]
    table["threshold"] = threshold
    table["prediction"] = (table.probability >= threshold).astype(int)
    return {"model": model, "history": selection_history, "refit_history": refit_history,
            "selected_epochs": selected_epochs, "threshold": threshold, "window_scores": windows,
            "speaker_scores": table, "scores": classification_scores(table, threshold),
            "n_parameters": sum(item.numel() for item in model.parameters()),
            "split": {"inner_train": sorted(fit_ids), "inner_validation": sorted(val_ids),
                      "outer_train": sorted(fit_ids | val_ids), "outer_test": sorted(test_ids)}}
