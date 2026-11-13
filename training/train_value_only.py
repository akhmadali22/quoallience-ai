#!/usr/bin/env python3
"""Fine-tune only the value head of a QuoridorNet checkpoint.

EXP-001: Bootstrap value learning from heuristic decisive games.

The trunk and policy head are frozen. Only value head parameters receive
gradient updates. This isolates the value-learning intervention and prevents
disruption to the policy prior that the existing network has learned.

Input JSONL format (from generate_heuristic_games.py):
  {"state": <10x9x9 list>, "value": <float>, ...}

The "policy" field is NOT required (value-only training).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from quoridor.neural_ai.network import QuoridorNet
from quoridor.neural_ai.policy import load_model, save_checkpoint


def load_value_samples(path: Path) -> list[dict]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_num}: {e}") from e

            if "state" not in record or "value" not in record:
                continue  # Skip records without required fields

            v = float(record["value"])
            if not -1.0 <= v <= 1.0:
                raise ValueError(
                    f"Value out of [-1, 1] at {path}:{line_num}: {v}"
                )
            samples.append({"state": record["state"], "value": v})

    if not samples:
        raise ValueError(f"No usable value samples found in {path}")
    return samples


def freeze_parameters(model: QuoridorNet, freeze_trunk: bool = True) -> None:
    """Freeze policy head and optionally trunk/stem parameters."""
    for name, param in model.named_parameters():
        if name.startswith("policy_head"):
            param.requires_grad_(False)
        elif name.startswith("value_head"):
            param.requires_grad_(True)
        else:
            param.requires_grad_(not freeze_trunk)


def verify_frozen_unchanged(
    model: QuoridorNet,
    original_params: dict[str, "torch.Tensor"],
    tol: float = 1e-9,
) -> None:
    """Assert that all frozen parameters are identical to originals."""
    import torch

    for name, param in model.named_parameters():
        if param.requires_grad:
            continue
        orig = original_params[name]
        if not torch.allclose(param.data, orig, atol=tol):
            max_diff = (param.data - orig).abs().max().item()
            raise RuntimeError(
                f"Frozen parameter '{name}' changed during training! "
                f"max_abs_diff={max_diff:.2e}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "replay", type=Path,
        help="Path to JSONL file with value training samples (from generate_heuristic_games.py).",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Incumbent checkpoint to fine-tune (not modified).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("models/checkpoints/exp_001_vonly_v1.pt"),
        help="Output checkpoint path.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261113001)
    parser.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="Fraction of samples held out for validation MSE reporting.",
    )
    parser.add_argument(
        "--train-trunk", action="store_true",
        help="Also train the residual trunk (policy head remains frozen). Default is trunk frozen.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning-rate must be positive; weight-decay must be non-negative")
    if not 0.0 <= args.val_fraction < 1.0:
        parser.error("val-fraction must be in [0, 1)")

    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    # ---- Load model and freeze ----
    model = load_model(args.checkpoint, device=device)
    freeze_parameters(model, freeze_trunk=not args.train_trunk)

    # Save a snapshot of original frozen weights for verification.
    original_params = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if not param.requires_grad
    }

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total "
          f"({100 * trainable / total:.1f}% — {'value head only' if not args.train_trunk else 'trunk + value head'})")

    # ---- Load samples ----
    samples = load_value_samples(args.replay)
    print(f"Loaded {len(samples):,} value training samples from {args.replay}")

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_fraction)) if args.val_fraction > 0 else 0
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"Split: {len(train_samples):,} train / {len(val_samples):,} val")

    # ---- Build tensors ----
    train_states = torch.tensor(
        [s["state"] for s in train_samples], dtype=torch.float32
    ).to(device)
    train_values = torch.tensor(
        [s["value"] for s in train_samples], dtype=torch.float32
    ).to(device)

    if val_samples:
        val_states = torch.tensor(
            [s["state"] for s in val_samples], dtype=torch.float32
        ).to(device)
        val_values = torch.tensor(
            [s["value"] for s in val_samples], dtype=torch.float32
        ).to(device)
    else:
        val_states = val_values = None

    # ---- Optimizer (value head parameters only) ----
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    n_train = len(train_samples)
    order = list(range(n_train))

    best_val_mse = float("inf")
    best_epoch = -1

    for epoch in range(args.epochs):
        random.shuffle(order)
        total_mse = 0.0
        batches = 0

        for start in range(0, n_train, args.batch_size):
            indices = order[start : start + args.batch_size]
            x = train_states[indices]
            y = train_values[indices]

            optimizer.zero_grad(set_to_none=True)
            _, predicted_value = model(x)
            loss = F.mse_loss(predicted_value, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            total_mse += float(loss.item())
            batches += 1

        train_mse = total_mse / batches

        # Validation MSE
        if val_states is not None:
            model.eval()
            with torch.no_grad():
                _, val_pred = model(val_states)
                val_mse = float(F.mse_loss(val_pred, val_values).item())
            model.train()
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch + 1
        else:
            val_mse = float("nan")

        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_mse={train_mse:.4f} "
            f"val_mse={val_mse:.4f}",
            flush=True,
        )

    # ---- Verify frozen unchanged ----
    verify_frozen_unchanged(model, original_params)
    print("✓ Frozen parameters (including policy head) verified unchanged.")

    if val_samples:
        print(f"Best val_mse={best_val_mse:.4f} at epoch {best_epoch}")

    # ---- Save candidate checkpoint ----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model,
        args.output,
        step=args.epochs,
        extra={
            "experiment": "EXP-001",
            "description": "Value-head-only fine-tune on heuristic decisive games",
            "incumbent": str(args.checkpoint),
            "replay": str(args.replay),
            "value_samples": len(samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "best_val_mse": best_val_mse if val_samples else None,
            "best_epoch": best_epoch if val_samples else None,
            "trunk_frozen": True,
            "policy_head_frozen": True,
            "seed": args.seed,
        },
    )
    print(f"Candidate checkpoint saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
