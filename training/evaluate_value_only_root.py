#!/usr/bin/env python3
"""Compare incumbent/candidate vanilla-PUCT root behavior on identical held-out states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quoridor.engine import GameState, Wall
from quoridor.neural_ai.encoder import encode_state
from quoridor.neural_ai.mcts import MCTS, MOVE_ACTIONS
from quoridor.neural_ai.policy import load_model


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def decode_state(encoded) -> GameState:
    """Invert the project's 10-plane side-to-move-relative encoder."""
    current_player = 0 if sum(encoded[2][0]) > 4.5 else 1

    def point(plane):
        locations = [(x, y) for y, row in enumerate(plane)
                     for x, value in enumerate(row) if value > 0.5]
        if len(locations) != 1:
            raise ValueError(f"Expected exactly one pawn in plane, got {locations}")
        return locations[0]

    own = point(encoded[0])
    opponent = point(encoded[1])
    pawns = [None, None]
    pawns[current_player] = own
    pawns[1 - current_player] = opponent

    walls = set()
    for y in range(9):
        for x in range(9):
            if encoded[4][y][x] > 0.5:
                walls.add(Wall("H", x, y))
            if encoded[5][y][x] > 0.5:
                walls.add(Wall("V", x, y))

    remaining = [0, 0]
    remaining[current_player] = round(float(encoded[6][0][0]) * 10)
    remaining[1 - current_player] = round(float(encoded[7][0][0]) * 10)
    ply = round(float(encoded[9][0][0]) * 200)
    state = GameState(
        pawns=(pawns[0], pawns[1]),
        walls=frozenset(walls),
        walls_remaining=(remaining[0], remaining[1]),
        current_player=current_player,
        winner=None,
        ply=ply,
    )
    rebuilt = encode_state(state)
    for channel in range(10):
        for y in range(9):
            for x in range(9):
                if abs(float(rebuilt[channel][y][x]) - float(encoded[channel][y][x])) > 1e-6:
                    raise ValueError("Encoded state failed round-trip validation")
    return state


def choose_probe_rows(dataset: Path, report: Path, max_positions: int):
    training_report = json.loads(report.read_text(encoding="utf-8"))
    test_ids = set(training_report["split_game_ids"]["test"])
    grouped = {}
    for row in read_jsonl(dataset):
        gid = int(row["game_id"])
        if gid in test_ids:
            grouped.setdefault(gid, []).append(row)
    target_plies = (0, 10, 20, 40, 60, 80)
    selected = []
    for gid in sorted(grouped):
        rows = sorted(grouped[gid], key=lambda r: int(r["ply"]))
        used = set()
        for target in target_plies:
            candidate = min(rows, key=lambda r: (abs(int(r["ply"]) - target), int(r["ply"])))
            key = int(candidate["ply"])
            if key not in used:
                selected.append(candidate)
                used.add(key)
    selected.sort(key=lambda r: (int(r["game_id"]), int(r["ply"])))
    return selected[:max_positions]


def summarize_root(model, state, *, simulations: int, c_puct: float, device: str):
    # Explicitly disable every optional search intervention. This is vanilla PUCT.
    search = MCTS(
        model, simulations=simulations, c_puct=c_puct, device=device,
        progressive_widening=False, initial_wall_candidates=0,
        category_balanced_prior=False, root_category_prior_mix=0.0,
        opening_goal_directed_prior=False, avoid_recent_state_repetition=False,
        category_exploration=0.0, repetition_penalty=0.0,
        pawn_progress_bonus=0.0, wall_distance_weight=0.0, wall_route_weight=0.0,
    )
    visits = search.search(state, add_root_noise=False)
    stats = search.last_search_stats
    action_stats = stats.get("root_action_stats", {})
    candidates = []
    for action, details in action_stats.items():
        if not details.get("active", True):
            continue
        candidates.append({
            "action": int(action),
            "category": "pawn" if int(action) < MOVE_ACTIONS else "wall",
            "prior": details.get("prior"),
            "q": details.get("q"),
            "u": details.get("u"),
            "puct_score": details.get("puct_score"),
            "visits": details.get("visits", 0),
        })
    best = {}
    for category in ("pawn", "wall"):
        group = [x for x in candidates if x["category"] == category]
        if group:
            best[category] = max(
                group,
                key=lambda x: (float(x["puct_score"] or float("-inf")), int(x["visits"]), -x["action"]),
            )
    selected_action = min(
        (int(a) for a, p in visits.items() if p == max(visits.values())),
        default=None,
    )
    selected_category = None if selected_action is None else (
        "pawn" if selected_action < MOVE_ACTIONS else "wall"
    )
    return {
        "root_value": stats.get("root_value"),
        "network_pawn_prior_mass": stats.get("root_network_pawn_prior_mass"),
        "network_wall_prior_mass": stats.get("root_network_wall_prior_mass"),
        "pawn_visits": stats.get("root_pawn_visits"),
        "wall_visits": stats.get("root_wall_visits"),
        "pawn_visit_ratio": stats.get("root_pawn_visit_ratio"),
        "wall_visit_ratio": stats.get("root_wall_visit_ratio"),
        "selected_action": selected_action,
        "selected_category": selected_category,
        "best_pawn_by_final_puct": best.get("pawn"),
        "best_wall_by_final_puct": best.get("wall"),
        "top_actions_by_visits": sorted(candidates, key=lambda x: (-x["visits"], x["action"]))[:10],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--max-positions", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.simulations < 1 or args.max_positions < 1:
        parser.error("simulations and max-positions must be positive")

    rows = choose_probe_rows(args.dataset, args.training_report, args.max_positions)
    if not rows:
        raise RuntimeError("No held-out test positions found")
    incumbent = load_model(args.incumbent, device=args.device)
    candidate = load_model(args.candidate, device=args.device)
    results = []
    for index, row in enumerate(rows, 1):
        state = decode_state(row["state"])
        baseline = summarize_root(
            incumbent, state, simulations=args.simulations,
            c_puct=args.c_puct, device=args.device,
        )
        trained = summarize_root(
            candidate, state, simulations=args.simulations,
            c_puct=args.c_puct, device=args.device,
        )
        results.append({
            "position_index": index,
            "game_id": int(row["game_id"]),
            "ply": int(row["ply"]),
            "player_to_move": int(row["player"]),
            "heuristic_winner": int(row["winner"]),
            "state_round_trip_valid": True,
            "incumbent": baseline,
            "candidate": trained,
            "delta": {
                "root_value": (trained["root_value"] or 0.0) - (baseline["root_value"] or 0.0),
                "pawn_prior_mass": (trained["network_pawn_prior_mass"] or 0.0) - (baseline["network_pawn_prior_mass"] or 0.0),
                "pawn_visit_ratio": (trained["pawn_visit_ratio"] or 0.0) - (baseline["pawn_visit_ratio"] or 0.0),
            },
        })
        print(
            f"[{index}/{len(rows)}] game={row['game_id']} ply={row['ply']} "
            f"pawn_prior={baseline['network_pawn_prior_mass']:.4f}->{trained['network_pawn_prior_mass']:.4f} "
            f"pawn_visits={baseline['pawn_visit_ratio']:.3f}->{trained['pawn_visit_ratio']:.3f}",
            flush=True,
        )

    summary = {
        "experiment": "EXP-001R-root-puct-comparison",
        "config": {
            "simulations": args.simulations, "c_puct": args.c_puct,
            "progressive_widening": False, "category_balanced_prior": False,
            "category_exploration": 0.0, "repetition_penalty": 0.0,
            "pawn_progress_bonus": 0.0, "opening_goal_directed_prior": False,
            "device": args.device,
        },
        "positions": len(results),
        "candidate_selected_pawn": sum(r["candidate"]["selected_category"] == "pawn" for r in results),
        "incumbent_selected_pawn": sum(r["incumbent"]["selected_category"] == "pawn" for r in results),
        "mean_delta": {
            key: sum(r["delta"][key] for r in results) / len(results)
            for key in ("root_value", "pawn_prior_mass", "pawn_visit_ratio")
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Root comparison saved: {args.output}")


if __name__ == "__main__":
    main()
