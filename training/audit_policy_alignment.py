#!/usr/bin/env python3
"""Evaluate raw neural policy alignment with replay targets, without running MCTS or training."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from quoridor.neural_ai.encoder import ACTION_SIZE, MOVE_ACTIONS
from quoridor.neural_ai.policy import load_model
from training.train import build_training_tensors, load_samples


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _summarize(rows: list[dict]) -> dict:
    metric_names = (
        "policy_cross_entropy",
        "policy_kl",
        "target_entropy",
        "uniform_legal_cross_entropy",
        "nll_gain_vs_uniform",
        "target_pawn_mass",
        "predicted_pawn_mass",
        "target_wall_mass",
        "predicted_wall_mass",
        "best_target_action_rank",
        "target_mass_at_top1",
        "target_mass_at_top5",
        "target_mass_at_top10",
    )
    result = {"samples": len(rows)}
    for name in metric_names:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        result[name] = {
            "mean": _mean(values),
            "median": median(values) if values else None,
        }
    result["top1_predicted_category"] = {
        category: sum(row["top1_predicted_category"] == category for row in rows)
        for category in ("pawn", "wall")
    }
    result["categories"] = {}
    for category in ("pawn", "wall"):
        result["categories"][category] = {}
        for name in (
            "target_mass",
            "predicted_mass",
            "conditional_cross_entropy",
            "conditional_kl",
            "mean_target_probability_per_legal_action",
            "mean_predicted_probability_per_legal_action",
        ):
            values = [
                float(row["categories"][category][name])
                for row in rows
                if row["categories"][category].get(name) is not None
            ]
            result["categories"][category][name] = {
                "mean": _mean(values),
                "median": median(values) if values else None,
            }
    return result


def _sample_metrics(
    sample: dict,
    target: list[float],
    predicted: list[float],
    legal: list[int],
) -> dict:
    legal_count = len(legal)
    if legal_count == 0:
        raise ValueError("Encountered a replay sample with no legal policy actions")

    target_mass = sum(target)
    if not math.isfinite(target_mass) or target_mass <= 0:
        raise ValueError("Replay sample has an invalid policy target")
    target = [value / target_mass for value in target]

    target_entropy = -sum(p * math.log(p) for p in target if p > 0)
    cross_entropy = -sum(
        target[action] * math.log(max(predicted[action], 1e-30))
        for action in legal
        if target[action] > 0
    )
    policy_kl = max(0.0, cross_entropy - target_entropy)
    uniform_ce = math.log(legal_count)

    ranked = sorted(legal, key=lambda action: (-predicted[action], action))
    ranks = {action: index + 1 for index, action in enumerate(ranked)}
    target_support = [action for action in legal if target[action] > 1e-12]
    best_target_rank = min((ranks[action] for action in target_support), default=None)
    weighted_target_rank = (
        sum(target[action] * ranks[action] for action in target_support)
        / sum(target[action] for action in target_support)
        if target_support else None
    )

    top1 = ranked[:1]
    top5 = ranked[:5]
    top10 = ranked[:10]
    target_mass_at_top1 = sum(target[action] for action in top1)
    target_mass_at_top5 = sum(target[action] for action in top5)
    target_mass_at_top10 = sum(target[action] for action in top10)
    top_action = ranked[0]

    categories = {}
    for category, actions in (
        ("pawn", [action for action in legal if action < MOVE_ACTIONS]),
        ("wall", [action for action in legal if action >= MOVE_ACTIONS]),
    ):
        target_category_mass = sum(target[action] for action in actions)
        predicted_category_mass = sum(predicted[action] for action in actions)
        if target_category_mass > 0 and predicted_category_mass > 0:
            conditional_target = {
                action: target[action] / target_category_mass for action in actions
            }
            conditional_predicted = {
                action: predicted[action] / predicted_category_mass for action in actions
            }
            conditional_entropy = -sum(
                p * math.log(p) for p in conditional_target.values() if p > 0
            )
            conditional_ce = -sum(
                p * math.log(max(conditional_predicted[action], 1e-30))
                for action, p in conditional_target.items() if p > 0
            )
            conditional_kl = max(0.0, conditional_ce - conditional_entropy)
        else:
            conditional_ce = None
            conditional_kl = None

        count = len(actions)
        categories[category] = {
            "legal_actions": count,
            "target_mass": target_category_mass,
            "predicted_mass": predicted_category_mass,
            "conditional_cross_entropy": conditional_ce,
            "conditional_kl": conditional_kl,
            "mean_target_probability_per_legal_action": (
                target_category_mass / count if count else None
            ),
            "mean_predicted_probability_per_legal_action": (
                predicted_category_mass / count if count else None
            ),
        }

    ply = int(sample.get("ply", 0))
    return {
        "game_id": sample.get("game_id"),
        "seed": sample.get("seed"),
        "ply": ply,
        "player": int(sample.get("player", 0)),
        "legal_actions": legal_count,
        "target_support_actions": len(target_support),
        "policy_cross_entropy": cross_entropy,
        "policy_kl": policy_kl,
        "target_entropy": target_entropy,
        "uniform_legal_cross_entropy": uniform_ce,
        "nll_gain_vs_uniform": uniform_ce - cross_entropy,
        "target_pawn_mass": categories["pawn"]["target_mass"],
        "predicted_pawn_mass": categories["pawn"]["predicted_mass"],
        "target_wall_mass": categories["wall"]["target_mass"],
        "predicted_wall_mass": categories["wall"]["predicted_mass"],
        "best_target_action_rank": best_target_rank,
        "target_mass_weighted_action_rank": weighted_target_rank,
        "target_mass_at_top1": target_mass_at_top1,
        "target_mass_at_top5": target_mass_at_top5,
        "target_mass_at_top10": target_mass_at_top10,
        "top1_predicted_action": top_action,
        "top1_predicted_category": "pawn" if top_action < MOVE_ACTIONS else "wall",
        "categories": categories,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-ply", type=int, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.batch_size < 1 or (args.max_ply is not None and args.max_ply < 0):
        parser.error("--batch-size must be positive and --max-ply non-negative")
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose another path")

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    samples = load_samples(args.replay)
    if args.max_ply is not None:
        samples = [sample for sample in samples if int(sample.get("ply", -1)) <= args.max_ply]
    if not samples:
        parser.error("No replay samples match the requested ply range")

    states, targets, masks, _ = build_training_tensors(samples)
    model = load_model(args.checkpoint, device=args.device)
    model.eval()

    rows = []
    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            end = min(start + args.batch_size, len(samples))
            logits, _ = model(states[start:end].to(args.device))
            mask = masks[start:end].to(args.device)
            masked_logits = logits.masked_fill(~mask, -torch.inf)
            probabilities = torch.softmax(masked_logits, dim=1).cpu().tolist()
            target_batch = targets[start:end].cpu().tolist()
            mask_batch = masks[start:end].cpu().tolist()

            for offset, sample in enumerate(samples[start:end]):
                legal = [
                    action for action, allowed in enumerate(mask_batch[offset]) if allowed
                ]
                rows.append(
                    _sample_metrics(
                        sample,
                        target_batch[offset],
                        probabilities[offset],
                        legal,
                    )
                )
                print(
                    f"[{len(rows)}/{len(samples)}] game={sample.get('game_id')} "
                    f"ply={sample.get('ply')} CE={rows[-1]['policy_cross_entropy']:.4f} "
                    f"pawn_target={rows[-1]['target_pawn_mass']:.4f} "
                    f"pawn_prior={rows[-1]['predicted_pawn_mass']:.4f}",
                    flush=True,
                )

    phases = defaultdict(list)
    for row in rows:
        phase_start = (row["ply"] // 10) * 10
        phases[f"{phase_start}-{phase_start + 9}"].append(row)

    report = {
        "checkpoint": str(args.checkpoint),
        "replay": str(args.replay),
        "samples": len(rows),
        "action_space": {
            "total": ACTION_SIZE,
            "pawn": MOVE_ACTIONS,
            "wall": ACTION_SIZE - MOVE_ACTIONS,
        },
        "evaluation": {
            "search_used": False,
            "training_performed": False,
            "checkpoint_modified": False,
            "policy_mask_applied": True,
            "description": (
                "Raw neural policy probabilities are evaluated directly against "
                "replay MCTS visit targets. No MCTS search, optimizer step, loss "
                "reweighting, or architecture change is performed."
            ),
        },
        "overall": _summarize(rows),
        "phase_buckets_by_ply": {
            phase: _summarize(phase_rows)
            for phase, phase_rows in sorted(
                phases.items(), key=lambda item: int(item[0].split("-")[0])
            )
        },
        "per_sample": rows,
    }

    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Policy alignment audit: {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
