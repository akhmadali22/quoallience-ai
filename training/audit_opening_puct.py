#!/usr/bin/env python3
"""Isolate vanilla PUCT on identical replay opening positions at several budgets.

This is an evaluation-only diagnostic. It does not train or modify a checkpoint.
Progressive widening, root noise, category balancing/exploration, repetition
penalties, and pawn-progress bonuses are explicitly disabled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from quoridor.engine import GameState, Wall
from quoridor.neural_ai.encoder import (
    ACTION_SIZE,
    MOVE_ACTIONS,
    decode_action,
    legal_action_indices,
)
from quoridor.neural_ai.mcts import MCTS
from quoridor.neural_ai.policy import load_model
from training.train import load_samples


def _plane_position(plane: list[list[float]]) -> tuple[int, int]:
    matches = [
        (x, y)
        for y, row in enumerate(plane)
        for x, value in enumerate(row)
        if value > 0.5
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one pawn marker, found {len(matches)}")
    return matches[0]


def decode_replay_state(sample: dict) -> GameState:
    """Reconstruct the exact engine state from the perspective-relative planes."""
    planes = sample["state"]
    if len(planes) != 10 or any(len(plane) != 9 for plane in planes):
        raise ValueError("Replay state must have shape 10x9x9")
    player = int(sample["player"])
    if player not in (0, 1):
        raise ValueError(f"Invalid replay player: {player}")

    own = _plane_position(planes[0])
    opponent = _plane_position(planes[1])
    pawns = [None, None]
    pawns[player] = own
    pawns[1 - player] = opponent

    walls = set()
    for orientation, plane_index in (("H", 4), ("V", 5)):
        for y in range(8):
            for x in range(8):
                if planes[plane_index][y][x] > 0.5:
                    walls.add(Wall(orientation, x, y))

    own_walls = round(float(planes[6][0][0]) * 10)
    opponent_walls = round(float(planes[7][0][0]) * 10)
    remaining = [None, None]
    remaining[player] = own_walls
    remaining[1 - player] = opponent_walls
    return GameState(
        pawns=(pawns[0], pawns[1]),
        walls=frozenset(walls),
        walls_remaining=(remaining[0], remaining[1]),
        current_player=player,
        winner=None,
        ply=int(sample["ply"]),
    )


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def summarize_actions(action_stats: dict[str, dict], c_puct: float) -> dict:
    categories = {}
    for category in ("pawn", "wall"):
        rows = [
            (int(action), details)
            for action, details in action_stats.items()
            if details["category"] == category
        ]
        priors = [float(row["network_prior"]) for _, row in rows]
        visits = [int(row["visits"]) for _, row in rows]
        visited_q = [
            (float(row["q"]), int(row["visits"]))
            for _, row in rows if int(row["visits"]) > 0
        ]
        q_weight = sum(weight for _, weight in visited_q)
        categories[category] = {
            "legal_actions": len(rows),
            "prior_mass": sum(priors),
            "uniform_legal_action_mass": (
                len(rows) / len(action_stats) if action_stats else 0.0
            ),
            "initial_exploration_mass": c_puct * sum(priors),
            "visits": sum(visits),
            "visit_ratio": None,
            "visit_weighted_q": (
                sum(q * weight for q, weight in visited_q) / q_weight
                if q_weight else None
            ),
            "visited_actions": sum(value > 0 for value in visits),
            "max_prior": max(priors, default=0.0),
            "max_final_u": max(
                (float(row["u_score"]) for _, row in rows if row["u_score"] is not None),
                default=None,
            ),
            "max_final_puct": max(
                (float(row["puct_score"]) for _, row in rows if row["puct_score"] is not None),
                default=None,
            ),
        }

    total_visits = sum(item["visits"] for item in categories.values())
    for item in categories.values():
        item["visit_ratio"] = item["visits"] / total_visits if total_visits else 0.0
    return categories


def run_position(model, sample: dict, *, budget: int, c_puct: float, device: str) -> dict:
    state = decode_replay_state(sample)
    legal = legal_action_indices(state)
    if not legal:
        raise ValueError(f"Replay sample at ply {state.ply} has no legal actions")
    if len(legal) > ACTION_SIZE:
        raise ValueError("Legal action count exceeds the policy action space")

    search = MCTS(
        model,
        simulations=budget,
        c_puct=c_puct,
        device=device,
        randomize_selection_ties=False,
        progressive_widening=False,
        category_balanced_prior=False,
        category_exploration=0.0,
        repetition_penalty=0.0,
        pawn_progress_bonus=0.0,
    )
    search.search(state, add_root_noise=False)
    stats = search.last_search_stats
    action_stats = stats["root_action_stats"]

    if set(map(int, action_stats)) != set(legal):
        raise RuntimeError(
            f"Root diagnostics/legal action mismatch at ply {state.ply}: "
            f"{len(action_stats)} diagnostics vs {len(legal)} legal actions"
        )

    action_rows = []
    for action_text, details in action_stats.items():
        action = int(action_text)
        kind, payload = decode_action(action)
        row = {
            "action": action,
            "category": details["category"],
            "move": list(payload) if kind == "move" else None,
            "wall": (
                {"orientation": payload.orientation, "x": payload.x, "y": payload.y}
                if kind == "wall" else None
            ),
            "prior": details["network_prior"],
            "uniform_legal_action_prior": 1.0 / len(legal),
            "initial_q": 0.0,
            "initial_u": c_puct * float(details["network_prior"]),
            "initial_puct_score": c_puct * float(details["network_prior"]),
            "visits": details["visits"],
            "q": details["q"],
            "final_u": details["u_score"],
            "final_puct_score": details["puct_score"],
            "active": details["active"],
        }
        action_rows.append(row)

    action_rows.sort(key=lambda row: (-row["visits"], -row["prior"], row["action"]))
    categories = summarize_actions(action_stats, c_puct)
    return {
        "sample": {
            "game_id": sample.get("game_id"),
            "seed": sample.get("seed"),
            "ply": state.ply,
            "player": state.current_player,
            "pawns": [list(position) for position in state.pawns],
            "walls_remaining": list(state.walls_remaining),
            "wall_count": len(state.walls),
        },
        "budget": budget,
        "simulations": stats["simulations"],
        "root_value": stats["root_value"],
        "legal_actions": stats["root_legal_actions"],
        "expanded_nodes": stats["expanded_nodes"],
        "average_leaf_depth": stats["average_leaf_depth"],
        "max_leaf_depth": stats["max_leaf_depth"],
        "network_prior_entropy": stats["root_network_prior_entropy"],
        "network_pawn_prior_mass": stats["root_network_pawn_prior_mass"],
        "network_wall_prior_mass": stats["root_network_wall_prior_mass"],
        "pawn_wall": categories,
        "actions": action_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 32, 64])
    parser.add_argument("--max-ply", type=int, default=19)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("data/mcts_opening_puct_audit.json"))
    parser.add_argument("--max-positions", type=int, default=0,
                        help="Optional smoke-test limit; zero means every matching replay position.")
    args = parser.parse_args()

    if not args.budgets or any(value < 1 for value in args.budgets):
        parser.error("All simulation budgets must be positive")
    if args.max_ply < 0 or args.c_puct < 0 or args.threads < 1 or args.max_positions < 0:
        parser.error("max-ply/max-positions must be non-negative; c-puct non-negative; threads positive")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose another --output path")

    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)

    samples = [
        sample for sample in load_samples(args.replay)
        if 0 <= int(sample.get("ply", -1)) <= args.max_ply
    ]
    samples.sort(key=lambda sample: (
        int(sample.get("game_id", 0)),
        int(sample.get("ply", 0)),
        int(sample.get("player", 0)),
    ))
    if args.max_positions:
        samples = samples[:args.max_positions]
    if not samples:
        parser.error(f"No replay positions found with ply <= {args.max_ply}")

    model = load_model(args.checkpoint, device=args.device)
    model.eval()

    runs = []
    for index, sample in enumerate(samples, 1):
        print(
            f"[{index}/{len(samples)}] game={sample.get('game_id')} "
            f"ply={sample.get('ply')} player={sample.get('player')}",
            flush=True,
        )
        for budget in args.budgets:
            result = run_position(
                model, sample, budget=budget, c_puct=args.c_puct, device=args.device
            )
            runs.append(result)
            summary = result["pawn_wall"]
            print(
                f"  sims={budget} pawn_prior={result['network_pawn_prior_mass']:.4f} "
                f"pawn_visits={summary['pawn']['visit_ratio']:.4f} "
                f"pawn_q={summary['pawn']['visit_weighted_q']} "
                f"wall_q={summary['wall']['visit_weighted_q']}",
                flush=True,
            )

    report = {
        "checkpoint": str(args.checkpoint),
        "replay": str(args.replay),
        "positions": len(samples),
        "ply_range": [0, args.max_ply],
        "budgets": args.budgets,
        "c_puct": args.c_puct,
        "device": args.device,
        "settings": {
            "vanilla_puct": True,
            "progressive_widening": False,
            "root_dirichlet_noise": False,
            "category_balanced_prior": False,
            "category_exploration": 0.0,
            "repetition_penalty": 0.0,
            "pawn_progress_bonus": 0.0,
            "randomize_selection_ties": False,
        },
        "interpretation": {
            "initial_u": "c_puct * network prior, with root Q=0 and sqrt(max(1,N))=1 before the first simulation.",
            "final_u": "PUCT exploration term after the requested simulations.",
            "q": "Root edge mean value from the perspective of the root player.",
            "final_puct_score": "Final root Q + U. This is a post-search snapshot, not a historical trace of every selection decision.",
            "uniform_legal_action_prior": "1 / number of legal root actions; useful as a branching-factor reference, not a policy target.",
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PUCT opening audit: {args.output}")
    print(f"Completed {len(samples)} positions x {len(args.budgets)} budgets = {len(runs)} searches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
