#!/usr/bin/env python3
"""Paired value-calibration audit for EXP-001R on identical held-out states.

Evaluates incumbent and candidate on every position in the game-level test split.
Reports paired metrics by ply bucket, winner, side to move, and game. Positions
are not treated as independent evidence: game-level deltas and a deterministic
bootstrap over held-out games are included.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from quoridor.neural_ai.policy import load_model


PLY_BUCKETS = (
    ("0-9", 0, 10),
    ("10-19", 10, 20),
    ("20-29", 20, 30),
    ("30-59", 30, 60),
    ("60+", 60, 10**9),
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                for key in ("state", "game_id", "ply", "player", "winner"):
                    if key not in row:
                        raise ValueError(f"{path}:{line_no}: missing {key}")
                yield row


def metrics(rows: list[dict], label: str) -> dict:
    if not rows:
        return {"count": 0}
    errors = [r[label] - r["target"] for r in rows]
    correct = [((r[label] >= 0) == (r["target"] > 0)) for r in rows]
    return {
        "count": len(rows),
        "mse": sum(e * e for e in errors) / len(errors),
        "mae": sum(abs(e) for e in errors) / len(errors),
        "sign_accuracy": sum(correct) / len(correct),
        "prediction_mean": sum(r[label] for r in rows) / len(rows),
        "target_mean": sum(r["target"] for r in rows) / len(rows),
        "mean_error": sum(errors) / len(errors),
    }


def paired_summary(rows: list[dict]) -> dict:
    before = metrics(rows, "incumbent")
    after = metrics(rows, "candidate")
    if not rows:
        return {"count": 0, "incumbent": before, "candidate": after}
    return {
        "count": len(rows),
        "incumbent": before,
        "candidate": after,
        "delta_candidate_minus_incumbent": {
            "mse": after["mse"] - before["mse"],
            "mae": after["mae"] - before["mae"],
            "sign_accuracy": after["sign_accuracy"] - before["sign_accuracy"],
            "mean_error": after["mean_error"] - before["mean_error"],
        },
    }


def calibration_bins(rows: list[dict], key: str) -> list[dict]:
    edges = (-1.000001, -0.6, -0.2, 0.2, 0.6, 1.000001)
    result = []
    for lo, hi in zip(edges, edges[1:]):
        group = [r for r in rows if lo <= r[key] < hi]
        if not group:
            continue
        result.append({
            "range": [round(lo, 3), round(hi, 3)],
            "count": len(group),
            "mean_prediction": sum(r[key] for r in group) / len(group),
            "observed_win_rate": sum(r["target"] > 0 for r in group) / len(group),
        })
    return result


def bootstrap_game_delta(game_rows: dict[int, list[dict]], *, seed: int, samples: int) -> dict:
    game_ids = sorted(game_rows)
    per_game = {
        gid: {
            "mse": paired_summary(values)["delta_candidate_minus_incumbent"]["mse"],
            "mae": paired_summary(values)["delta_candidate_minus_incumbent"]["mae"],
            "sign_accuracy": paired_summary(values)["delta_candidate_minus_incumbent"]["sign_accuracy"],
        }
        for gid, values in game_rows.items()
    }
    if len(game_ids) < 2:
        return {"games": len(game_ids), "per_game": per_game, "bootstrap_95_ci": None}
    rng = random.Random(seed)
    distributions = {key: [] for key in ("mse", "mae", "sign_accuracy")}
    for _ in range(samples):
        draw = [rng.choice(game_ids) for _ in game_ids]
        for key in distributions:
            distributions[key].append(sum(per_game[g][key] for g in draw) / len(draw))
    intervals = {}
    for key, values in distributions.items():
        values.sort()
        intervals[key] = {
            "low": values[int(0.025 * (len(values) - 1))],
            "high": values[int(0.975 * (len(values) - 1))],
        }
    return {"games": len(game_ids), "per_game": per_game, "bootstrap_95_ci": intervals}


@torch.inference_mode()
def evaluate(model, rows: list[dict], device: str, batch_size: int) -> list[float]:
    model.eval()
    predictions = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        states = torch.tensor(
            [row["state"] for row in batch], dtype=torch.float32, device=device
        )
        _, values = model(states)
        predictions.extend(float(v) for v in values.detach().cpu().reshape(-1).tolist())
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261113004)
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 100:
        parser.error("batch-size must be positive and bootstrap-samples >= 100")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    test_ids = set(int(x) for x in report["split_game_ids"]["test"])
    rows = []
    seen = set()
    for row in read_jsonl(args.dataset):
        gid, ply = int(row["game_id"]), int(row["ply"])
        if gid not in test_ids:
            continue
        key = (gid, ply)
        if key in seen:
            raise ValueError(f"duplicate test position game_id/ply={key}")
        seen.add(key)
        player, winner = int(row["player"]), int(row["winner"])
        if player not in (0, 1) or winner not in (0, 1):
            raise ValueError(f"invalid player/winner in {key}")
        rows.append({
            "game_id": gid, "ply": ply, "player": player, "winner": winner,
            "target": 1.0 if player == winner else -1.0,
            "state": row["state"],
        })
    rows.sort(key=lambda r: (r["game_id"], r["ply"]))
    if not rows:
        raise RuntimeError("No rows found for held-out test game IDs")

    incumbent = load_model(args.incumbent, device=args.device)
    candidate = load_model(args.candidate, device=args.device)
    for model in (incumbent, candidate):
        model.eval()
    incumbent_values = evaluate(incumbent, rows, args.device, args.batch_size)
    candidate_values = evaluate(candidate, rows, args.device, args.batch_size)
    for row, before, after in zip(rows, incumbent_values, candidate_values):
        row["incumbent"] = before
        row["candidate"] = after
        del row["state"]

    groups = {
        "overall": rows,
        "by_winner": {
            str(w): [r for r in rows if r["winner"] == w] for w in (0, 1)
        },
        "by_player_to_move": {
            str(p): [r for r in rows if r["player"] == p] for p in (0, 1)
        },
        "by_ply_bucket": {
            name: [r for r in rows if lo <= r["ply"] < hi]
            for name, lo, hi in PLY_BUCKETS
        },
    }
    grouped_metrics = {"overall": paired_summary(rows)}
    for dimension in ("by_winner", "by_player_to_move", "by_ply_bucket"):
        grouped_metrics[dimension] = {
            key: paired_summary(group) for key, group in groups[dimension].items()
        }

    by_game = defaultdict(list)
    for row in rows:
        by_game[row["game_id"]].append(row)

    output = {
        "experiment": "EXP-001R-value-calibration-audit",
        "models": {
            "incumbent": str(args.incumbent),
            "candidate": str(args.candidate),
        },
        "dataset": str(args.dataset),
        "test_game_ids": sorted(test_ids),
        "positions": len(rows),
        "games": len(by_game),
        "method": {
            "same_identical_states_for_both_models": True,
            "target": "undiscounted +/-1 from player-to-move perspective",
            "game_level_split_reused_from_training_report": True,
            "positions_are_not_independent_units_for_bootstrap": True,
            "bootstrap_unit": "game_id",
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
        },
        "metrics": grouped_metrics,
        "calibration_bins": {
            "incumbent": calibration_bins(rows, "incumbent"),
            "candidate": calibration_bins(rows, "candidate"),
        },
        "game_level_analysis": bootstrap_game_delta(
            by_game, seed=args.seed, samples=args.bootstrap_samples
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "positions": len(rows),
        "games": len(by_game),
        "overall": grouped_metrics["overall"],
        "game_level_bootstrap_95_ci": output["game_level_analysis"]["bootstrap_95_ci"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
