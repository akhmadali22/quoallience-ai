#!/usr/bin/env python3
"""Train a Quoridor policy/value network from self-play JSONL replay data."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from quoridor.neural_ai.encoder import INPUT_CHANNELS, ACTION_SIZE
from quoridor.neural_ai.network import QuoridorNet
from quoridor.neural_ai.policy import save_checkpoint
from training.config import TrainConfig


def load_samples(path: Path) -> list[dict]:
    samples = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                if "state" not in sample or "policy" not in sample or "value" not in sample:
                    continue
                samples.append(sample)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    if not samples:
        raise ValueError(f"No usable training samples found in {path}")
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path, help="Path to replay.jsonl")
    parser.add_argument("--output", default="models/checkpoints/neural_latest.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=4)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs/batch-size must be positive and learning-rate > 0")

    import torch
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    samples = load_samples(args.replay)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    model = QuoridorNet(args.channels, args.residual_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    model.train()

    states = torch.tensor([s["state"] for s in samples], dtype=torch.float32)
    targets = torch.zeros((len(samples), ACTION_SIZE), dtype=torch.float32)
    values = torch.tensor([s["value"] for s in samples], dtype=torch.float32)
    for row, sample in enumerate(samples):
        for action, probability in sample["policy"].items():
            action_index = int(action)
            if 0 <= action_index < ACTION_SIZE:
                targets[row, action_index] = float(probability)

    order = list(range(len(samples)))
    for epoch in range(args.epochs):
        random.shuffle(order)
        total_loss = total_policy = total_value = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            x = states[indices].to(device)
            target_policy = targets[indices].to(device)
            target_value = values[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, predicted_value = model(x)
            policy_loss = -(target_policy * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = torch.nn.functional.mse_loss(predicted_value, target_value)
            loss = policy_loss + value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            total_policy += float(policy_loss.item())
            total_value += float(value_loss.item())
            batches += 1
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"loss={total_loss / batches:.4f} "
            f"policy={total_policy / batches:.4f} "
            f"value={total_value / batches:.4f}",
            flush=True,
        )

    save_checkpoint(
        model, args.output, optimizer=optimizer, step=args.epochs,
        extra={"training_samples": len(samples), "replay": str(args.replay)},
    )
    print(f"Saved checkpoint: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
