"""Channel-independent PatchTST classifier: (batch, mel, time) to one logit.

Each mel band passes through the same time-patch encoder independently.
Learned positions retain patch order. The classification head retains band
identity. Fixed input scaling preserves differences in absolute sound level.
"""

import torch
from torch import nn


class PatchTST(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        features = cfg["features"]
        model = cfg["model"]
        self.seq_len = round(features["window_sec"] * cfg["data"]["target_sr"] / features["hop_length"])
        self.n_vars = features["n_mels"]
        self.patch_len = model["patch_len"]
        self.stride = model["stride"]
        self.n_patches = (self.seq_len - self.patch_len) // self.stride + 1
        self.db_offset = model["db_offset"]
        self.db_scale = model["db_scale"]
        width = model["d_model"]
        self.projection = nn.Linear(self.patch_len, width)
        self.position = nn.Parameter(torch.empty(1, self.n_patches, width))
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=model["n_heads"], dim_feedforward=width * 4,
            dropout=model["dropout"], batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=model["n_layers"])
        self.dropout = nn.Dropout(model["dropout"])
        self.head = nn.Linear(self.n_vars * self.n_patches * width, 1)

    def forward(self, windows):
        batch, channels, time = windows.shape
        if channels != self.n_vars or time != self.seq_len:
            print("ERROR: expected", self.n_vars, "bands and", self.seq_len, "frames; got", tuple(windows.shape))
            return None
        scaled = (windows + self.db_offset) / self.db_scale
        patches = scaled.unfold(-1, self.patch_len, self.stride)
        patches = patches.reshape(batch * channels, self.n_patches, self.patch_len)
        hidden = self.projection(patches) + self.position
        hidden = self.encoder(hidden)
        hidden = hidden.reshape(batch, -1)
        return self.head(self.dropout(hidden))
