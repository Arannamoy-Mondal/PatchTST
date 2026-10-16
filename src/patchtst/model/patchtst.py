"""Channel-independent PatchTST classifier.

Input:  (batch size, 64 mel bands, 400 time frames)
Output: one logit per window. Apply sigmoid outside the model for a probability.
"""

import torch
from torch import nn


class PatchTST(nn.Module):
    """Classify one speech window without mixing mel bands inside the encoder."""

    def __init__(self, settings):
        super().__init__()
        feature_settings = settings["features"]
        model_settings = settings["model"]
        sample_rate = settings["data"]["target_sr"]
        frame_hop = feature_settings["hop_length"]
        self.time_frame_count = round(
            feature_settings["window_sec"] * sample_rate / frame_hop
        )
        self.mel_band_count = feature_settings["n_mels"]
        self.patch_length = model_settings["patch_len"]
        self.patch_stride = model_settings["stride"]
        self.patch_count = (
            self.time_frame_count - self.patch_length
        ) // self.patch_stride + 1
        self.decibel_offset = model_settings["db_offset"]
        self.decibel_scale = model_settings["db_scale"]

        hidden_width = model_settings["d_model"]
        self.patch_projection = nn.Linear(self.patch_length, hidden_width)
        self.patch_positions = nn.Parameter(torch.empty(1, self.patch_count, hidden_width))
        nn.init.normal_(self.patch_positions, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_width,
            nhead=model_settings["n_heads"],
            dim_feedforward=hidden_width * 4,
            dropout=model_settings["dropout"],
            batch_first=True,
            activation="gelu",
        )
        self.time_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_settings["n_layers"],
        )
        self.dropout = nn.Dropout(model_settings["dropout"])
        flattened_width = self.mel_band_count * self.patch_count * hidden_width
        self.classification_head = nn.Linear(flattened_width, 1)

    def forward(self, speech_windows):
        """Return one classification logit for every speech window."""
        batch_size, mel_band_count, time_frame_count = speech_windows.shape
        expected_shape = (self.mel_band_count, self.time_frame_count)
        actual_shape = (mel_band_count, time_frame_count)
        if actual_shape != expected_shape:
            print("ERROR: expected", expected_shape, "features; got", actual_shape)
            return None

        # Fixed scaling keeps quiet and loud recordings numerically distinct.
        scaled_windows = (speech_windows + self.decibel_offset) / self.decibel_scale

        # Cut each mel band into time patches. Bands stay separate here.
        time_patches = scaled_windows.unfold(-1, self.patch_length, self.patch_stride)
        time_patches = time_patches.reshape(
            batch_size * mel_band_count,
            self.patch_count,
            self.patch_length,
        )
        encoded_patches = self.patch_projection(time_patches) + self.patch_positions
        encoded_patches = self.time_encoder(encoded_patches)

        # Flattening, rather than averaging, lets the final layer see each band.
        flattened_features = encoded_patches.reshape(batch_size, -1)
        return self.classification_head(self.dropout(flattened_features))

    def load_v2_checkpoint(self, checkpoint_path):
        """Load a completed v2.0 checkpoint saved with the earlier layer names."""
        saved_weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        old_name_to_new_name = {
            "projection.": "patch_projection.",
            "position": "patch_positions",
            "encoder.": "time_encoder.",
            "head.": "classification_head.",
        }
        renamed_weights = {}
        for old_name, weight in saved_weights.items():
            new_name = old_name
            for old_prefix, new_prefix in old_name_to_new_name.items():
                if old_name.startswith(old_prefix):
                    new_name = new_prefix + old_name[len(old_prefix):]
                    break
            renamed_weights[new_name] = weight
        incompatible_names = self.load_state_dict(renamed_weights, strict=True)
        if incompatible_names.missing_keys or incompatible_names.unexpected_keys:
            print("ERROR: v2.0 checkpoint did not match the readable model.")
            return False
        return True
