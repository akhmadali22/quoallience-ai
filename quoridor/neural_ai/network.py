"""Residual policy/value network. Install optional dependencies from requirements-nn.txt."""
from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised on installations without optional torch
    raise ImportError(
        "Neural Quoridor requires PyTorch. Install it with: pip install -r requirements-nn.txt"
    ) from exc

from .encoder import ACTION_SIZE, INPUT_CHANNELS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.layers(x))


class QuoridorNet(nn.Module):
    """Shared residual trunk with policy logits and side-to-move value heads."""

    def __init__(self, channels: int = 64, residual_blocks: int = 4):
        super().__init__()
        if channels < 8 or residual_blocks < 1:
            raise ValueError("channels must be >= 8 and residual_blocks >= 1")
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*(ResidualBlock(channels) for _ in range(residual_blocks)))
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, ACTION_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * 9 * 9, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        features = self.trunk(self.stem(x))
        return self.policy_head(features), self.value_head(features).squeeze(-1)
