"""Create cached features for the acoustic and frozen-Whisper pipelines."""

import hashlib
import json
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


def cache_key(recording, extraction_settings):
    """Name a cache file from the exact audio bytes and extraction settings."""
    settings_text = json.dumps(extraction_settings, sort_keys=True)
    return hashlib.sha256((recording.audio_sha256 + settings_text).encode()).hexdigest()


def acoustic_features(recordings, settings):
    """Extract pitch, jitter, shimmer, and MFCC summaries for vowel recordings."""
    import parselmouth
    from parselmouth.praat import call
    from patchtst.data.audio import load_audio

    extraction_settings = dict(settings["comparison"])
    extraction_settings["rate"] = settings["data"]["target_sr"]
    extraction_settings["version"] = "fixed-reference-mfcc-v2"
    cache_directory = Path(settings["outputs"]["cache_root"]) / "acoustic"
    cache_directory.mkdir(parents=True, exist_ok=True)
    feature_rows = []
    sample_rate = extraction_settings["rate"]

    for recording_number, recording in enumerate(recordings.itertuples(index=False), start=1):
        cache_path = cache_directory / (cache_key(recording, extraction_settings) + ".json")
        if cache_path.exists():
            feature_values = json.loads(cache_path.read_text())
        else:
            audio = load_audio(recording.path, sample_rate).astype(np.float64)
            audio, unused_trim_indexes = librosa.effects.trim(
                audio,
                top_db=extraction_settings["acoustic_trim_db"],
            )
            edge_sample_count = round(extraction_settings["acoustic_margin_seconds"] * sample_rate)
            long_enough_to_trim_edges = len(audio) > sample_rate + 2 * edge_sample_count
            if long_enough_to_trim_edges:
                audio = audio[edge_sample_count:-edge_sample_count]
            if len(audio) < sample_rate or np.max(np.abs(audio)) < 1e-5:
                print("ERROR: vowel recording is too short after trimming:", recording.path)
                return None

            sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
            pitch = sound.to_pitch(
                time_step=0.01,
                pitch_floor=extraction_settings["pitch_floor"],
                pitch_ceiling=extraction_settings["pitch_ceiling"],
            )
            voiced_frequencies = pitch.selected_array["frequency"]
            voiced_frequencies = voiced_frequencies[voiced_frequencies > 0]
            feature_values = {
                "pitch_median": None,
                "pitch_std": None,
                "jitter_local": None,
                "shimmer_local": None,
            }
            if len(voiced_frequencies) > 0:
                feature_values["pitch_median"] = float(np.median(voiced_frequencies))
                feature_values["pitch_std"] = float(np.std(voiced_frequencies))
            try:
                pulses = call(
                    sound,
                    "To PointProcess (periodic, cc)",
                    extraction_settings["pitch_floor"],
                    extraction_settings["pitch_ceiling"],
                )
                feature_values["jitter_local"] = float(call(
                    pulses, "Get jitter (local)", 0, 0, 0.0016,
                    1.25 / extraction_settings["pitch_floor"], 1.3,
                ))
                feature_values["shimmer_local"] = float(call(
                    [sound, pulses], "Get shimmer (local)", 0, 0, 0.0016,
                    1.25 / extraction_settings["pitch_floor"], 1.3, 1.6,
                ))
            except parselmouth.PraatError as error:
                print("Acoustic measurement unavailable; training-only imputation will handle it.")
                print("Recording:", recording.path, "| reason:", str(error))

            mel_power = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_mels=extraction_settings["acoustic_n_mels"],
                n_fft=extraction_settings["acoustic_n_fft"],
                hop_length=extraction_settings["acoustic_hop_length"],
                power=2.0,
            )
            log_mel = librosa.power_to_db(
                mel_power,
                ref=1.0,
                amin=settings["features"]["amin"],
                top_db=None,
            )
            mfcc_values = librosa.feature.mfcc(
                S=log_mel,
                n_mfcc=extraction_settings["acoustic_n_mfcc"],
            )
            for mfcc_number in range(extraction_settings["acoustic_n_mfcc"]):
                column_number = mfcc_number + 1
                feature_values["mfcc_%02d_mean" % column_number] = float(mfcc_values[mfcc_number].mean())
                feature_values["mfcc_%02d_std" % column_number] = float(mfcc_values[mfcc_number].std())
            for feature_name, feature_value in feature_values.items():
                if feature_value is not None and not np.isfinite(feature_value):
                    feature_values[feature_name] = None
            cache_path.write_text(json.dumps(feature_values, indent=2, allow_nan=False))

        feature_rows.append({
            "path": recording.path,
            "speaker_id": recording.speaker_id,
            "label": recording.label,
            "vowel": recording.task[:2],
            **feature_values,
        })
        if recording_number % 50 == 0 or recording_number == len(recordings):
            print("Acoustic features:", recording_number, "/", len(recordings), flush=True)
    return pd.DataFrame(feature_rows)


def whisper_features(recordings, settings):
    """Extract frozen Whisper-large-v3 vectors without fitting the encoder."""
    import gc
    import torch
    from math import gcd
    from scipy.signal import resample_poly
    from transformers import AutoFeatureExtractor, WhisperModel

    torch.set_num_threads(settings["train"]["threads"])
    comparison_settings = settings["comparison"]
    extraction_settings = {
        "model": comparison_settings["whisper_model"],
        "revision": comparison_settings["whisper_revision"],
        "seconds": comparison_settings["whisper_seconds"],
        "rate": settings["data"]["target_sr"],
        "dtype": comparison_settings["whisper_dtype"],
        "resampler": "scipy-resample-poly",
        "version": 1,
    }
    cache_directory = Path(settings["outputs"]["cache_root"]) / "whisper"
    cache_directory.mkdir(parents=True, exist_ok=True)
    encoder = None
    feature_extractor = None
    target_sample_rate = extraction_settings["rate"]
    window_sample_count = round(extraction_settings["seconds"] * target_sample_rate)
    embedding_paths = []
    started_at = time.monotonic()

    for recording_number, recording in enumerate(recordings.itertuples(index=False), start=1):
        cache_path = cache_directory / (cache_key(recording, extraction_settings) + ".npy")
        if cache_path.exists():
            embedding = np.load(cache_path, allow_pickle=False)
            valid_embedding = embedding.shape == (1280,) and np.isfinite(embedding).all()
            if not valid_embedding:
                print("ERROR: cached Whisper embedding is invalid:", cache_path)
                return None
        else:
            if encoder is None:
                print("Loading frozen Whisper-large-v3 encoder on CPU in float32.", flush=True)
                feature_extractor = AutoFeatureExtractor.from_pretrained(
                    extraction_settings["model"],
                    revision=extraction_settings["revision"],
                )
                full_model = WhisperModel.from_pretrained(
                    extraction_settings["model"],
                    revision=extraction_settings["revision"],
                    dtype=torch.float32,
                    use_safetensors=True,
                    attn_implementation="sdpa",
                )
                encoder = full_model.encoder.eval()
                for parameter in encoder.parameters():
                    parameter.requires_grad_(False)
                del full_model
                gc.collect()

            audio, native_sample_rate = sf.read(recording.path, dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if native_sample_rate != target_sample_rate:
                common_factor = gcd(native_sample_rate, target_sample_rate)
                audio = resample_poly(
                    audio,
                    target_sample_rate // common_factor,
                    native_sample_rate // common_factor,
                )
            if len(audio) < window_sample_count:
                print("ERROR: passage is shorter than 30 seconds:", recording.path)
                return None

            window_start_positions = list(range(0, len(audio) - window_sample_count + 1, window_sample_count))
            final_position = len(audio) - window_sample_count
            if window_start_positions[-1] != final_position:
                window_start_positions.append(final_position)
            window_embeddings = []
            for start_position in window_start_positions:
                speech_window = audio[start_position:start_position + window_sample_count].astype(np.float32)
                model_inputs = feature_extractor(
                    speech_window,
                    sampling_rate=target_sample_rate,
                    return_tensors="pt",
                )
                with torch.inference_mode():
                    hidden_states = encoder(input_features=model_inputs.input_features).last_hidden_state
                    window_embeddings.append(hidden_states.mean(dim=1).squeeze(0).numpy())
            embedding = np.stack(window_embeddings).mean(axis=0).astype(np.float32)
            valid_embedding = embedding.shape == (1280,) and np.isfinite(embedding).all()
            if not valid_embedding:
                print("ERROR: extracted Whisper embedding is invalid:", recording.path)
                return None
            temporary_path = cache_path.with_suffix(".partial")
            with temporary_path.open("wb") as cache_file:
                np.save(cache_file, embedding)
            temporary_path.replace(cache_path)

        embedding_paths.append(str(cache_path))
        elapsed_seconds = round(time.monotonic() - started_at, 1)
        print(
            "Whisper features:", recording_number, "/", len(recordings),
            "| elapsed seconds:", elapsed_seconds,
            flush=True,
        )

    recordings_with_embeddings = recordings.copy()
    recordings_with_embeddings["embedding_path"] = embedding_paths
    return recordings_with_embeddings
