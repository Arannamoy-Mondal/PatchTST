"""Frozen speech embeddings and handcrafted acoustic features for the comparison."""

import hashlib
import json
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


def cache_key(row, settings):
    text = row.audio_sha256 + json.dumps(settings, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def acoustic_features(files, cfg):
    import parselmouth
    from parselmouth.praat import call
    from patchtst.data.audio import load_audio

    settings = dict(cfg["comparison"])
    settings["rate"] = cfg["data"]["target_sr"]
    settings["version"] = "fixed-reference-mfcc-v2"
    folder = Path(cfg["outputs"]["cache_root"]) / "acoustic"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    rate = settings["rate"]
    for number, row in enumerate(files.itertuples(index=False), 1):
        target = folder / (cache_key(row, settings) + ".json")
        if target.exists():
            values = json.loads(target.read_text())
        else:
            audio = load_audio(row.path, rate).astype(np.float64)
            audio, unused = librosa.effects.trim(audio, top_db=settings["acoustic_trim_db"])
            margin = round(settings["acoustic_margin_seconds"] * rate)
            if len(audio) > rate + 2 * margin:
                audio = audio[margin:-margin]
            if len(audio) < rate or np.max(np.abs(audio)) < 1e-5:
                print("ERROR: acoustic extraction needs at least one second of non-silent audio:", row.path)
                return None
            sound = parselmouth.Sound(audio, sampling_frequency=rate)
            pitch = sound.to_pitch(time_step=0.01, pitch_floor=settings["pitch_floor"], pitch_ceiling=settings["pitch_ceiling"])
            voiced = pitch.selected_array["frequency"]
            voiced = voiced[voiced > 0]
            values = {"pitch_median": None, "pitch_std": None, "jitter_local": None, "shimmer_local": None}
            if len(voiced) > 0:
                values["pitch_median"] = float(np.median(voiced))
                values["pitch_std"] = float(np.std(voiced))
            try:
                pulses = call(sound, "To PointProcess (periodic, cc)", settings["pitch_floor"], settings["pitch_ceiling"])
                values["jitter_local"] = float(call(pulses, "Get jitter (local)", 0, 0, 0.0016, 1.25 / settings["pitch_floor"], 1.3))
                values["shimmer_local"] = float(call([sound, pulses], "Get shimmer (local)", 0, 0, 0.0016, 1.25 / settings["pitch_floor"], 1.3, 1.6))
            except parselmouth.PraatError as error:
                print("Acoustic measurement unavailable; training-only median imputation will be used:", row.path, str(error))
            mel = librosa.feature.melspectrogram(
                y=audio, sr=rate, n_mels=settings["acoustic_n_mels"],
                n_fft=settings["acoustic_n_fft"], hop_length=settings["acoustic_hop_length"], power=2.0,
            )
            logmel = librosa.power_to_db(mel, ref=1.0, amin=cfg["features"]["amin"], top_db=None)
            mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=settings["acoustic_n_mfcc"])
            for index in range(settings["acoustic_n_mfcc"]):
                values[f"mfcc_{index + 1:02d}_mean"] = float(mfcc[index].mean())
                values[f"mfcc_{index + 1:02d}_std"] = float(mfcc[index].std())
            for key, value in values.items():
                if value is not None and not np.isfinite(value):
                    values[key] = None
            target.write_text(json.dumps(values, indent=2, allow_nan=False))
        rows.append({"path": row.path, "speaker_id": row.speaker_id, "label": row.label, "vowel": row.task[:2], **values})
        if number % 50 == 0 or number == len(files):
            print("Acoustic features:", number, "/", len(files), flush=True)
    return pd.DataFrame(rows)


def whisper_features(files, cfg):
    import gc
    import torch
    from math import gcd
    from scipy.signal import resample_poly
    from transformers import AutoFeatureExtractor, WhisperModel

    torch.set_num_threads(cfg["train"]["threads"])
    settings = {"model": cfg["comparison"]["whisper_model"], "revision": cfg["comparison"]["whisper_revision"],
                "seconds": cfg["comparison"]["whisper_seconds"], "rate": cfg["data"]["target_sr"],
                "dtype": cfg["comparison"]["whisper_dtype"], "resampler": "scipy-resample-poly", "version": 1}
    folder = Path(cfg["outputs"]["cache_root"]) / "whisper"
    folder.mkdir(parents=True, exist_ok=True)
    encoder = None
    extractor = None
    rate = settings["rate"]
    size = round(settings["seconds"] * rate)
    paths = []
    started = time.monotonic()
    for number, row in enumerate(files.itertuples(index=False), 1):
        target = folder / (cache_key(row, settings) + ".npy")
        if target.exists():
            vector = np.load(target, allow_pickle=False)
            if vector.shape != (1280,) or not np.isfinite(vector).all():
                print("ERROR: invalid cached Whisper vector:", target)
                return None
        else:
            if encoder is None:
                print("Loading frozen Whisper-large-v3 encoder on CPU in float32.", flush=True)
                extractor = AutoFeatureExtractor.from_pretrained(settings["model"], revision=settings["revision"])
                full_model = WhisperModel.from_pretrained(
                    settings["model"], revision=settings["revision"], dtype=torch.float32,
                    use_safetensors=True, attn_implementation="sdpa",
                )
                encoder = full_model.encoder.eval()
                for parameter in encoder.parameters():
                    parameter.requires_grad_(False)
                del full_model
                gc.collect()
            audio, original_rate = sf.read(row.path, dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if original_rate != rate:
                factor = gcd(original_rate, rate)
                audio = resample_poly(audio, rate // factor, original_rate // factor)
            if len(audio) < size:
                print("ERROR: passage is shorter than the frozen window length:", row.path)
                return None
            starts = list(range(0, len(audio) - size + 1, size))
            if starts[-1] != len(audio) - size:
                starts.append(len(audio) - size)
            vectors = []
            for start in starts:
                chunk = audio[start:start + size].astype(np.float32)
                inputs = extractor(chunk, sampling_rate=rate, return_tensors="pt")
                with torch.inference_mode():
                    hidden = encoder(input_features=inputs.input_features).last_hidden_state
                    vectors.append(hidden.mean(dim=1).squeeze(0).numpy())
            vector = np.stack(vectors).mean(axis=0).astype(np.float32)
            if vector.shape != (1280,) or not np.isfinite(vector).all():
                print("ERROR: invalid extracted Whisper vector:", row.path)
                return None
            partial = target.with_suffix(".partial")
            with partial.open("wb") as handle:
                np.save(handle, vector)
            partial.replace(target)
        paths.append(str(target))
        print("Whisper features:", number, "/", len(files), "elapsed seconds:", round(time.monotonic() - started, 1), flush=True)
    output = files.copy()
    output["embedding_path"] = paths
    return output
