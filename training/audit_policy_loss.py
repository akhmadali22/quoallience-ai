#!/usr/bin/env python3
"""Measure checkpoint policy/value losses on replay without updating weights."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from quoridor.neural_ai.encoder import ACTION_SIZE, MOVE_ACTIONS
from quoridor.neural_ai.policy import load_model
from training.train import build_training_tensors, load_samples


def policy_loss_components(logits, targets, policy_masks):
    """Return per-sample total, pawn, and wall cross-entropy contributions."""
    import torch

    masked_logits = logits.masked_fill(~policy_masks, -torch.inf)
    log_probs = torch.log_softmax(masked_logits, dim=1)
    log_probs = log_probs.masked_fill(~policy_masks, 0.0)
    pawn = -(targets[:, :MOVE_ACTIONS] * log_probs[:, :MOVE_ACTIONS]).sum(dim=1)
    wall = -(targets[:, MOVE_ACTIONS:] * log_probs[:, MOVE_ACTIONS:]).sum(dim=1)
    return pawn + wall, pawn, wall


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.value_loss_weight < 0:
        parser.error("--batch-size must be positive and --value-loss-weight non-negative")

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    samples = load_samples(args.replay)
    states, targets, masks, values = build_training_tensors(samples)
    model = load_model(args.checkpoint, device=args.device)
    model.eval()

    totals = defaultdict(list)
    phases = defaultdict(lambda: defaultdict(list))
    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            end = min(start + args.batch_size, len(samples))
            x = states[start:end].to(args.device)
            target = targets[start:end].to(args.device)
            mask = masks[start:end].to(args.device)
            target_value = values[start:end].to(args.device)
            logits, predicted_value = model(x)
            total, pawn, wall = policy_loss_components(logits, target, mask)
            value = torch.nn.functional.mse_loss(
                predicted_value.reshape(-1), target_value, reduction="none"
            )
            pawn_mass = target[:, :MOVE_ACTIONS].sum(dim=1)
            wall_mass = target[:, MOVE_ACTIONS:].sum(dim=1)
            for offset in range(end - start):
                index = start + offset
                ply = int(samples[index].get("ply", 0))
                phase = f"{(ply // 10) * 10}-{(ply // 10) * 10 + 9}"
                row = {
                    "policy_loss": float(total[offset].item()),
                    "pawn_policy_loss": float(pawn[offset].item()),
                    "wall_policy_loss": float(wall[offset].item()),
                    "value_loss": float(value[offset].item()),
                    "pawn_target_mass": float(pawn_mass[offset].item()),
                    "wall_target_mass": float(wall_mass[offset].item()),
                }
                for key, number in row.items():
                    totals[key].append(number)
                    phases[phase][key].append(number)

    def summarize(values):
        return mean(values) if values else 0.0

    report = {
        "checkpoint": str(args.checkpoint),
        "replay": str(args.replay),
        "samples": len(samples),
        "action_space": {"total": ACTION_SIZE, "pawn": MOVE_ACTIONS, "wall": ACTION_SIZE - MOVE_ACTIONS},
        "value_loss_weight": args.value_loss_weight,
        "loss": {
            "policy_cross_entropy": summarize(totals["policy_loss"]),
            "pawn_policy_contribution": summarize(totals["pawn_policy_loss"]),
            "wall_policy_contribution": summarize(totals["wall_policy_loss"]),
            "value_mse": summarize(totals["value_loss"]),
            "total_objective": (
                summarize(totals["policy_loss"])
                + args.value_loss_weight * summarize(totals["value_loss"])
            ),
        },
        "target_mass": {
            "pawn": summarize(totals["pawn_target_mass"]),
            "wall": summarize(totals["wall_target_mass"]),
        },
        "phase_buckets_by_ply": {
            phase: {
                "samples": len(values["policy_loss"]),
                **{key: summarize(numbers) for key, numbers in values.items()},
            }
            for phase, values in sorted(
                phases.items(), key=lambda item: int(item[0].split("-")[0])
            )
        },
        "note": "Evaluation only. No optimizer step or checkpoint modification is performed. Pawn and wall loss values are additive contributions to the existing policy cross-entropy; no category reweighting is applied.",
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Policy loss audit: {args.output}")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
