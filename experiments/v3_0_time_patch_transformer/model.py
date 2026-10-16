"""A small transformer that reads time patches made from all mel bands.

This is intentionally different from the v2.0 channel-independent PatchTST.
Each patch contains every mel band at one short span of time, so the model
can relate frequency and time together.
"""

import torch
from torch import nn


class TimePatchTransformer(nn.Module):
    """Classify one 64-by-400 log-mel window."""

    def __init__(self, width=32, layers=1, heads=4, dropout=0.2):
        super().__init__()
        self.mel_band_count = 64
        self.frame_count = 400
        self.patch_length = 16
        self.patch_stride = 16
        self.patch_count = (self.frame_count - self.patch_length) // self.patch_stride + 1
        self.width = width
        self.patch_projection = nn.Linear(self.mel_band_count * self.patch_length, width)
        self.positions = nn.Parameter(torch.empty(1, self.patch_count, width))
        nn.init.normal_(self.positions, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.patch_count * width, 1)

    def forward(self, windows):
        """Return one logit for each window shaped (batch, 64, 400)."""
        batch_size, band_count, frame_count = windows.shape
        if band_count != self.mel_band_count or frame_count != self.frame_count:
            print("ERROR: expected 64 bands and 400 frames; got", tuple(windows.shape))
            return None
        patches = windows.unfold(-1, self.patch_length, self.patch_stride)
        patches = patches.permute(0, 2, 1, 3)
        patches = patches.reshape(batch_size, self.patch_count, -1)
        hidden = self.patch_projection(patches) + self.positions
        hidden = self.encoder(hidden)
        return self.classifier(self.dropout(hidden.reshape(batch_size, -1)))
