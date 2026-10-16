"""Train one held-out fold without letting its people enter training."""

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

SAVED_SPECTROGRAMS = {}


def seed_everything(seed):
    """Make model initialization and batch order repeatable for one fold."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def windows_for(recordings, settings, limit_training_windows):
    """Load cached spectrograms and return model windows plus their owners."""
    window_groups = []
    labels = []
    speaker_ids = []
    recording_paths = []
    for recording in recordings.itertuples(index=False):
        if recording.feature_path not in SAVED_SPECTROGRAMS:
            SAVED_SPECTROGRAMS[recording.feature_path] = np.load(recording.feature_path)
        windows = slice_windows(
            SAVED_SPECTROGRAMS[recording.feature_path],
            settings,
            limit=limit_training_windows,
        )
        window_groups.append(windows)
        labels.extend([recording.label] * len(windows))
        speaker_ids.extend([recording.speaker_id] * len(windows))
        recording_paths.extend([recording.path] * len(windows))
    return (
        np.concatenate(window_groups),
        np.asarray(labels, dtype=np.float32),
        np.asarray(speaker_ids),
        np.asarray(recording_paths),
    )


def training_weights(labels, speaker_ids, recording_paths):
    """Give each class, person, and recording equal total training weight."""
    windows = pd.DataFrame({
        "label": labels,
        "speaker_id": speaker_ids,
        "path": recording_paths,
    })
    windows_per_recording = windows.groupby("path").size()
    recordings_per_person = windows.groupby("speaker_id").path.nunique()
    people_per_class = windows.drop_duplicates("speaker_id").groupby("label").size()
    raw_weights = 1.0 / (
        windows.path.map(windows_per_recording)
        * windows.speaker_id.map(recordings_per_person)
        * windows.label.map(people_per_class)
    )
    balanced_weights = raw_weights / raw_weights.mean()
    return balanced_weights.to_numpy(dtype=np.float32)


def predict_windows(model, windows, settings):
    """Predict probabilities for windows in small batches."""
    model.eval()
    probability_groups = []
    batch_size = settings["train"]["batch_size"]
    with torch.inference_mode():
        for start_position in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start_position:start_position + batch_size])
            logits = model(batch)
            probability_groups.append(torch.sigmoid(logits).squeeze(1).numpy())
    return np.concatenate(probability_groups)


def fit_model(training_windows, validation_windows, settings, seed, epoch_count):
    """Train until the requested epoch or until validation stops improving."""
    seed_everything(seed)
    windows, labels, speaker_ids, recording_paths = training_windows
    weights = training_weights(labels, speaker_ids, recording_paths)
    random_generator = torch.Generator().manual_seed(seed)
    training_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(windows),
            torch.from_numpy(labels),
            torch.from_numpy(weights),
        ),
        batch_size=settings["train"]["batch_size"],
        shuffle=True,
        generator=random_generator,
    )
    model = PatchTST(settings)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["train"]["lr"],
        weight_decay=settings["train"]["weight_decay"],
    )
    loss_function = nn.BCEWithLogitsLoss(reduction="none")

    history = []
    best_validation_loss = float("inf")
    best_epoch_number = 1
    epochs_without_improvement = 0
    started_at = time.monotonic()
    for epoch_number in range(1, epoch_count + 1):
        model.train()
        weighted_loss_total = 0.0
        for batch_windows, batch_labels, batch_weights in training_loader:
            optimizer.zero_grad()
            logits = model(batch_windows).squeeze(1)
            weighted_loss = (loss_function(logits, batch_labels) * batch_weights).mean()
            if not torch.isfinite(weighted_loss):
                print("ERROR: training loss is not finite.")
                return None
            weighted_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            weighted_loss_total += weighted_loss.item() * len(batch_windows)

        epoch_record = {
            "epoch": epoch_number,
            "train_loss": weighted_loss_total / len(windows),
        }
        if validation_windows is not None:
            validation_features, validation_labels, validation_ids, validation_paths = validation_windows
            probabilities = predict_windows(model, validation_features, settings)
            people = speaker_table(pd.DataFrame({
                "speaker_id": validation_ids,
                "path": validation_paths,
                "label": validation_labels,
                "probability": probabilities,
            }))
            clipped_probabilities = np.clip(people.probability.to_numpy(), 1e-7, 1 - 1e-7)
            true_labels = people.label.to_numpy()
            person_losses = -(
                true_labels * np.log(clipped_probabilities)
                + (1 - true_labels) * np.log(1 - clipped_probabilities)
            )
            validation_loss = float(person_losses.mean())
            epoch_record["val_speaker_loss"] = validation_loss
            if validation_loss < best_validation_loss - 0.0001:
                best_validation_loss = validation_loss
                best_epoch_number = epoch_number
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        history.append(epoch_record)
        elapsed_seconds = round(time.monotonic() - started_at, 1)
        print("Epoch", epoch_number, epoch_record, "elapsed seconds", elapsed_seconds, flush=True)
        if validation_windows is not None and epochs_without_improvement >= settings["train"]["patience"]:
            break
    return model, history, best_epoch_number


def train_fold(training_recordings, test_recordings, settings, device=None, fold=0):
    """Select epochs on inner validation, then refit and score the held-out fold."""
    torch.set_num_threads(settings["train"]["threads"])
    people = training_recordings.groupby("speaker_id", as_index=False).agg(label=("label", "first"))
    seed = settings["splits"]["seed"] + fold
    person_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=settings["train"]["validation_fraction"],
        random_state=seed,
    )
    training_positions, validation_positions = next(person_splitter.split(people, people.label))
    inner_training_ids = set(people.iloc[training_positions].speaker_id)
    validation_ids = set(people.iloc[validation_positions].speaker_id)
    test_ids = set(test_recordings.speaker_id)
    person_sets_overlap = (
        inner_training_ids & validation_ids
        or inner_training_ids & test_ids
        or validation_ids & test_ids
    )
    if person_sets_overlap:
        print("ERROR: the same person appears in more than one PatchTST split.")
        return None

    inner_training_recordings = training_recordings[
        training_recordings.speaker_id.isin(inner_training_ids)
    ]
    validation_recordings = training_recordings[
        training_recordings.speaker_id.isin(validation_ids)
    ]
    selection_result = fit_model(
        windows_for(inner_training_recordings, settings, True),
        windows_for(validation_recordings, settings, False),
        settings,
        seed,
        settings["train"]["epochs"],
    )
    if selection_result is None:
        return None
    unused_selection_model, selection_history, selected_epoch_count = selection_result
    del unused_selection_model

    print("Refitting on all outer-training people for", selected_epoch_count, "epochs.", flush=True)
    refit_result = fit_model(
        windows_for(training_recordings, settings, True),
        None,
        settings,
        seed,
        selected_epoch_count,
    )
    if refit_result is None:
        return None
    model, refit_history, unused_epoch_count = refit_result

    test_windows, test_labels, test_speaker_ids, test_paths = windows_for(
        test_recordings,
        settings,
        False,
    )
    window_scores = pd.DataFrame({
        "speaker_id": test_speaker_ids,
        "path": test_paths,
        "label": test_labels.astype(int),
        "probability": predict_windows(model, test_windows, settings),
    })
    task_by_recording = test_recordings.set_index("path")["task"]
    window_scores["task"] = window_scores.path.map(task_by_recording)
    person_scores = speaker_table(window_scores)
    threshold = settings["train"]["threshold"]
    person_scores["threshold"] = threshold
    person_scores["prediction"] = (person_scores.probability >= threshold).astype(int)
    return {
        "model": model,
        "history": selection_history,
        "refit_history": refit_history,
        "selected_epochs": selected_epoch_count,
        "threshold": threshold,
        "window_scores": window_scores,
        "speaker_scores": person_scores,
        "scores": classification_scores(person_scores, threshold),
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "split": {
            "inner_train": sorted(inner_training_ids),
            "inner_validation": sorted(validation_ids),
            "outer_train": sorted(inner_training_ids | validation_ids),
            "outer_test": sorted(test_ids),
        },
    }


# The verification script uses the previous function name.
_windows_for = windows_for
