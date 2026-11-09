#!/usr/bin/env python3
"""Train a Quoridor policy/value network from self-play JSONL replay data."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from quoridor.neural_ai.encoder import ACTION_SIZE
from quoridor.neural_ai.network import QuoridorNet
from quoridor.neural_ai.policy import load_model, save_checkpoint
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
    parser.add_argument("--checkpoint", help="Resume from an existing checkpoint.")
    parser.add_argument("--epochs", type=int, default=5)
    defaults = TrainConfig()
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--value-loss-weight", type=float, default=defaults.value_loss_weight)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=4)
    args = parser.parse_args()

    if (args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0
            or args.value_loss_weight < 0 or args.weight_decay < 0):
        parser.error("epochs/batch-size must be positive; learning-rate > 0; loss weight and weight decay must be non-negative")

    import torch
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    samples = load_samples(args.replay)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    model = load_model(args.checkpoint, device=device) if args.checkpoint else QuoridorNet(args.channels, args.residual_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
    model.train()

    states = torch.tensor([s["state"] for s in samples], dtype=torch.float32)
    targets = torch.zeros((len(samples), ACTION_SIZE), dtype=torch.float32)
    policy_masks = torch.ones((len(samples), ACTION_SIZE), dtype=torch.bool)
    values = torch.tensor([s["value"] for s in samples], dtype=torch.float32)
    for row, sample in enumerate(samples):
        if "policy_mask" in sample:
            policy_masks[row].zero_()
            for action in sample["policy_mask"]:
                action_index = int(action)
                if 0 <= action_index < ACTION_SIZE:
                    policy_masks[row, action_index] = True
            if not policy_masks[row].any():
                raise ValueError(f"Sample {row} has an empty policy_mask")
        for action, probability in sample["policy"].items():
            action_index = int(action)
            if 0 <= action_index < ACTION_SIZE:
                targets[row, action_index] = float(probability)
        if torch.any(targets[row].masked_select(~policy_masks[row]) > 1e-8):
            raise ValueError(f"Sample {row} assigns policy mass outside policy_mask")
        target_mass = targets[row].sum()
        if not torch.isfinite(target_mass) or target_mass <= 0:
            raise ValueError(f"Sample {row} has an invalid policy target")
        targets[row] /= target_mass

    order = list(range(len(samples)))
    for epoch in range(args.epochs):
        random.shuffle(order)
        total_loss = total_policy = total_value = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            x = states[indices].to(device)
            target_policy = targets[indices].to(device)
            policy_mask = policy_masks[indices].to(device)
            target_value = values[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, predicted_value = model(x)
            masked_logits = logits.masked_fill(~policy_mask, -torch.inf)
            log_probs = torch.log_softmax(masked_logits, dim=1)
            # Avoid 0 * -inf for actions excluded from this search target.
            log_probs = log_probs.masked_fill(~policy_mask, 0.0)
            policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
            value_loss = torch.nn.functional.mse_loss(predicted_value, target_value)
            loss = policy_loss + args.value_loss_weight * value_loss
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
        extra={
            "training_samples": len(samples), "replay": str(args.replay),
            "value_loss_weight": args.value_loss_weight,
            "weight_decay": args.weight_decay,
            "policy_masked_samples": sum("policy_mask" in sample for sample in samples),
        },
    )
    print(f"Saved checkpoint: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
