#!/usr/bin/env python3
"""Audit pawn/wall visit-target balance in AlphaZero replay JSONL files."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from collections import defaultdict

from quoridor.neural_ai.encoder import ACTION_SIZE, MOVE_ACTIONS


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def audit_samples(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("No replay samples found")

    pawn_mass: list[float] = []
    wall_mass: list[float] = []
    entropy: list[float] = []
    network_pawn_prior_mass: list[float] = []
    root_pawn_visit_ratio: list[float] = []
    pawn_mask_count: list[float] = []
    wall_mask_count: list[float] = []
    pawn_target_nonzero: list[float] = []
    wall_target_nonzero: list[float] = []
    outside_mask_mass = 0.0
    legacy_mask_samples = 0
    phase_samples: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "pawn_mass": [], "wall_mass": [], "entropy": [],
            "network_pawn_prior_mass": [], "root_pawn_visit_ratio": [],
            "pawn_mask_count": [], "wall_mask_count": [],
        }
    )

    for row, sample in enumerate(samples):
        policy = {
            int(action): float(probability)
            for action, probability in sample.get("policy", {}).items()
        }
        if any(not math.isfinite(value) or value < 0 for value in policy.values()):
            raise ValueError(f"Sample {row} has invalid policy probabilities")
        mask = {
            int(action) for action in sample.get("policy_mask", range(ACTION_SIZE))
            if 0 <= int(action) < ACTION_SIZE
        }
        if "policy_mask" not in sample:
            legacy_mask_samples += 1
        if not mask:
            raise ValueError(f"Sample {row} has an empty policy mask")

        outside = sum(value for action, value in policy.items() if action not in mask)
        outside_mask_mass += outside
        total = sum(policy.values())
        if not math.isfinite(total) or total <= 0:
            raise ValueError(f"Sample {row} has an invalid/empty policy target")
        if outside > 1e-8:
            raise ValueError(f"Sample {row} assigns {outside:.6g} target mass outside policy_mask")

        normalized = {action: value / total for action, value in policy.items()}
        p_mass = sum(value for action, value in normalized.items() if action < MOVE_ACTIONS)
        w_mass = sum(value for action, value in normalized.items() if action >= MOVE_ACTIONS)
        pawn_mass.append(p_mass)
        wall_mass.append(w_mass)
        entropy.append(-sum(p * math.log(p) for p in normalized.values() if p > 0))
        pawn_mask_count.append(float(sum(action < MOVE_ACTIONS for action in mask)))
        wall_mask_count.append(float(sum(action >= MOVE_ACTIONS for action in mask)))
        pawn_target_nonzero.append(float(sum(
            value > 1e-12 for action, value in normalized.items() if action < MOVE_ACTIONS
        )))
        wall_target_nonzero.append(float(sum(
            value > 1e-12 for action, value in normalized.items() if action >= MOVE_ACTIONS
        )))
        ply = int(sample.get("ply", 0))
        phase = f"{(ply // 10) * 10}-{(ply // 10) * 10 + 9}"
        phase_samples[phase]["pawn_mass"].append(p_mass)
        phase_samples[phase]["wall_mass"].append(w_mass)
        phase_samples[phase]["entropy"].append(entropy[-1])
        diagnostics = sample.get("root_diagnostics") or {}
        network_mass = diagnostics.get("root_network_pawn_prior_mass")
        visit_ratio = diagnostics.get("root_pawn_visit_ratio")
        if network_mass is not None:
            network_pawn_prior_mass.append(float(network_mass))
            phase_samples[phase]["network_pawn_prior_mass"].append(float(network_mass))
        if visit_ratio is not None:
            root_pawn_visit_ratio.append(float(visit_ratio))
            phase_samples[phase]["root_pawn_visit_ratio"].append(float(visit_ratio))
        phase_samples[phase]["pawn_mask_count"].append(pawn_mask_count[-1])
        phase_samples[phase]["wall_mask_count"].append(wall_mask_count[-1])

    phase_buckets = {}
    for phase, values in sorted(phase_samples.items(), key=lambda item: int(item[0].split("-")[0])):
        phase_buckets[phase] = {
            "samples": len(values["pawn_mass"]),
            "mean_pawn_target_mass": mean(values["pawn_mass"]),
            "mean_wall_target_mass": mean(values["wall_mass"]),
            "mean_target_entropy_nats": mean(values["entropy"]),
            "mean_network_pawn_prior_mass": (
                mean(values["network_pawn_prior_mass"]) if values["network_pawn_prior_mass"] else None
            ),
            "mean_root_pawn_visit_ratio": (
                mean(values["root_pawn_visit_ratio"]) if values["root_pawn_visit_ratio"] else None
            ),
            "mean_searched_pawn_actions": mean(values["pawn_mask_count"]),
            "mean_searched_wall_actions": mean(values["wall_mask_count"]),
            "zero_pawn_target_samples": sum(value <= 1e-12 for value in values["pawn_mass"]),
            "zero_wall_target_samples": sum(value <= 1e-12 for value in values["wall_mass"]),
        }

    return {
        "samples": len(samples),
        "action_space": {
            "total": ACTION_SIZE,
            "pawn": MOVE_ACTIONS,
            "wall": ACTION_SIZE - MOVE_ACTIONS,
        },
        "target_mass": {
            "pawn": summarize(pawn_mass),
            "wall": summarize(wall_mass),
        },
        "target_entropy_nats": summarize(entropy),
        "network_pawn_prior_mass": summarize(network_pawn_prior_mass),
        "root_pawn_visit_ratio": summarize(root_pawn_visit_ratio),
        "phase_buckets_by_ply": phase_buckets,
        "policy_mask_actions": {
            "pawn": summarize(pawn_mask_count),
            "wall": summarize(wall_mask_count),
        },
        "nonzero_target_actions": {
            "pawn": summarize(pawn_target_nonzero),
            "wall": summarize(wall_target_nonzero),
        },
        "samples_with_zero_category_target": {
            "pawn": sum(value <= 1e-12 for value in pawn_mass),
            "wall": sum(value <= 1e-12 for value in wall_mass),
        },
        "legacy_samples_without_policy_mask": legacy_mask_samples,
        "total_target_mass_outside_mask": outside_mask_mass,
        "interpretation": (
            "Pawn/wall target masses are normalized MCTS visit-target mass per "
            "position, averaged across samples. They are not legal-action counts "
            "or model-predicted probabilities. Training CE is separately "
            "decomposed into pawn and wall contributions by training.train."
        ),
    }


def load_replay(path: Path) -> list[dict]:
    samples = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_samples(load_replay(args.replay))
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Policy target audit: {args.output}")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
