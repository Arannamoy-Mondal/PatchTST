"""Frozen WavLM-large extraction: no labels, training or cohort-fitted transform."""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMModel

from patchtst.data.audio import load_audio
from common import PROJECT, fingerprint, write_json


def window_positions(sample_count, window_samples):
    if sample_count <= window_samples:
        return [0]
    positions = list(range(0, sample_count - window_samples + 1, window_samples))
    final_position = sample_count - window_samples
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def load_encoder(settings):
    features = settings["features"]
    torch.set_num_threads(features["threads"])
    extractor = AutoFeatureExtractor.from_pretrained(features["model"], revision=features["revision"])
    if not extractor.do_normalize or extractor.sampling_rate != settings["data"]["target_sr"]:
        print("ERROR: expected the pinned 16-kHz mean/variance-normalizing WavLM frontend.")
        return None
    encoder = WavLMModel.from_pretrained(
        features["model"], revision=features["revision"], dtype=torch.float32,
    ).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if encoder.config.hidden_size != features["hidden_size"]:
        print("ERROR: expected", features["hidden_size"], "WavLM features.")
        return None
    return extractor, encoder


def extract_one(path, settings, extractor, encoder):
    sample_rate = settings["data"]["target_sr"]
    features = settings["features"]
    audio = load_audio(path, sample_rate)
    if len(audio) == 0 or not np.isfinite(audio).all():
        print("ERROR: expected nonempty finite audio:", path)
        return None
    window_samples = round(features["seconds"] * sample_rate)
    starts = window_positions(len(audio), window_samples)
    vectors = []
    frame_counts = []
    for start in starts:
        window = audio[start:start + window_samples]
        if len(window) < features["minimum_samples"]:
            window = np.pad(window, (0, features["minimum_samples"] - len(window)), mode="edge")
        inputs = extractor(window, sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True)
        with torch.inference_mode():
            hidden = encoder(**inputs).last_hidden_state
            vector = hidden.mean(dim=1).squeeze(0).numpy().astype(np.float32)
        if vector.shape != (features["hidden_size"],) or not np.isfinite(vector).all():
            print("ERROR: expected a finite 1024-value window embedding:", path)
            return None
        vectors.append(vector)
        frame_counts.append(int(hidden.shape[1]))
    windows = np.stack(vectors)
    return windows.mean(axis=0).astype(np.float32), windows, starts, frame_counts


def extract_partition(number, recordings, settings, run_directory):
    destination = run_directory / "feature_shards" / ("worker_" + str(number))
    destination.mkdir(parents=True, exist_ok=True)
    return extract_recordings(recordings, settings, destination, parallel=False)


def extract_recordings(recordings, settings, run_directory, parallel=True):
    if parallel and settings["features"]["workers"] > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import get_context
        import pandas as pd
        tables = []
        with ProcessPoolExecutor(max_workers=settings["features"]["workers"], mp_context=get_context("spawn")) as pool:
            jobs = []
            for number in range(settings["features"]["workers"]):
                subset = recordings.iloc[number::settings["features"]["workers"]].copy()
                jobs.append(pool.submit(extract_partition, number, subset, settings, run_directory))
            for job in as_completed(jobs):
                table = job.result()
                if table is None:
                    return None
                tables.append(table)
        result = pd.concat(tables, ignore_index=True).sort_values("path").reset_index(drop=True)
        if len(result) != len(recordings) or result.path.duplicated().any():
            print("ERROR: feature workers must cover every recording exactly once.")
            return None
        result.to_csv(run_directory / "embedding_manifest.csv", index=False)
        return result
    cache_root = PROJECT / settings["outputs"]["cache_root"]
    cache_root.mkdir(parents=True, exist_ok=True)
    feature_settings = {
        "features": settings["features"], "sample_rate": settings["data"]["target_sr"],
        "resampler": "librosa-soxr_hq", "dependency_versions": settings["dependency_versions"],
        "pretrained_files": settings["pretrained_files"],
    }
    cache_settings_text = json.dumps(feature_settings, sort_keys=True)
    loaded = None
    rows = []
    started = time.monotonic()
    for number, recording in enumerate(recordings.itertuples(index=False), start=1):
        audio_path = PROJECT / recording.path
        if fingerprint(audio_path) != recording.audio_sha256:
            print("ERROR: raw audio differs from the frozen manifest:", recording.path)
            return None
        key = hashlib.sha256((recording.audio_sha256 + cache_settings_text).encode()).hexdigest()
        embedding_path = cache_root / (key + ".npy")
        windows_path = cache_root / (key + "_windows.npy")
        metadata_path = cache_root / (key + ".json")
        if embedding_path.is_file() and windows_path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            vector = np.load(embedding_path, allow_pickle=False)
            windows = np.load(windows_path, allow_pickle=False)
            valid = (
                vector.shape == (settings["features"]["hidden_size"],)
                and windows.ndim == 2 and windows.shape[1] == len(vector)
                and np.isfinite(vector).all() and np.isfinite(windows).all()
                and metadata["embedding_sha256"] == fingerprint(embedding_path)
                and metadata["windows_sha256"] == fingerprint(windows_path)
                and np.array_equal(vector, windows.mean(axis=0).astype(np.float32))
            )
            if not valid:
                print("ERROR: cached WavLM features failed their checks:", embedding_path)
                return None
        else:
            if loaded is None:
                print("Loading the pinned frozen WavLM-large encoder.", flush=True)
                loaded = load_encoder(settings)
                if loaded is None:
                    return None
                extractor, encoder = loaded
                write_json(run_directory / "encoder.json", {
                    "parameters": sum(parameter.numel() for parameter in encoder.parameters()),
                    "trainable_parameters": sum(parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad),
                    "config": encoder.config.to_dict(), "frontend": extractor.to_dict(),
                })
            output = extract_one(audio_path, settings, *loaded)
            if output is None:
                return None
            vector, windows, starts, frame_counts = output
            for path, values in [(embedding_path, vector), (windows_path, windows)]:
                temporary = path.with_suffix(".partial." + str(os.getpid()))
                with temporary.open("wb") as handle:
                    np.save(handle, values, allow_pickle=False)
                temporary.replace(path)
            metadata = {
                "audio_sha256": recording.audio_sha256, "extraction": feature_settings,
                "starts_samples": starts, "frame_counts": frame_counts,
                "embedding_sha256": fingerprint(embedding_path), "windows_sha256": fingerprint(windows_path),
            }
            write_json(metadata_path, metadata)
        row = {"path": recording.path, "embedding_path": embedding_path.relative_to(PROJECT).as_posix(),
               "windows_path": windows_path.relative_to(PROJECT).as_posix(),
               "metadata_path": metadata_path.relative_to(PROJECT).as_posix(),
               "embedding_sha256": metadata["embedding_sha256"], "window_count": len(windows)}
        rows.append(row)
        if number == 1 or number % 10 == 0 or number == len(recordings):
            print("WavLM", run_directory.name, "recordings:", number, "/", len(recordings),
                  "elapsed seconds:", round(time.monotonic() - started, 1), flush=True)
    import pandas as pd
    result = recordings.merge(pd.DataFrame(rows), on="path", validate="one_to_one")
    result.to_csv(run_directory / "embedding_manifest.csv", index=False)
    return result


def feature_table(manifest):
    columns = ["embedding_%04d" % index for index in range(1024)]
    vectors = []
    for row in manifest.itertuples(index=False):
        path = PROJECT / row.embedding_path
        if not path.is_file() or fingerprint(path) != row.embedding_sha256:
            print("ERROR: saved embedding fingerprint differs:", row.path)
            return None
        vector = np.load(path, allow_pickle=False)
        if vector.shape != (1024,) or not np.isfinite(vector).all():
            print("ERROR: expected a finite 1024-value recording embedding:", row.path)
            return None
        vectors.append(vector)
    import pandas as pd
    table = pd.DataFrame(np.stack(vectors), columns=columns)
    metadata = manifest[["path", "speaker_id", "label", "fold", "sheet_flag", "task", "native_sr"]].reset_index(drop=True)
    return pd.concat([metadata, table], axis=1)
