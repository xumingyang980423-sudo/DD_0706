"""Frozen TrackNet inference for ir_track observation mode."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class TrackNet(nn.Module):
    """Must match training/scripts/train_tracknet.py spatial_v2 architecture."""

    def __init__(self, in_ch: int = 1, spatial: int = 4, channels: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, channels, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        flat_dim = channels * spatial * spatial
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class TorchTrackNetRunner:
    """Load checkpoint once and run batch inference on IR frames."""

    def __init__(self, checkpoint: str | Path, device: torch.device | str) -> None:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"TrackNet checkpoint not found: {path}")
        self.device = device
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.model = TrackNet(in_ch=int(ckpt.get("in_ch", 1))).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def predict(self, frame: torch.Tensor) -> torch.Tensor:
        """frame: [N, H, W] or [N, 1, H, W] in [0, 1]. Returns raw 6D logits/values."""
        if frame.ndim == 3:
            frame = frame.unsqueeze(1)
        return self.model(frame)
