"""Checkpoint loading and inference helpers for the neural Quoridor agent."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .network import QuoridorNet


def load_model(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> QuoridorNet:
    import torch

    path = Path(checkpoint)
    payload: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    config = payload.get("model_config", {})
    model = QuoridorNet(
        channels=int(config.get("channels", 64)),
        residual_blocks=int(config.get("residual_blocks", 4)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def save_checkpoint(
    model: QuoridorNet,
    path: str | Path,
    *,
    optimizer=None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "channels": model.stem[0].out_channels,
            "residual_blocks": len(model.trunk),
        },
        "step": step,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
