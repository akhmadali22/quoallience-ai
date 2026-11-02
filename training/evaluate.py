#!/usr/bin/env python3
"""Paired-seat evaluation: neural MCTS agent versus the existing heuristic AI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quoridor.advanced_ai import AIConfig, QuoridorAI
from quoridor.engine import GameState
from quoridor.neural_ai.encoder import apply_action_index, encode_action
from quoridor.neural_ai.mcts import MCTS
from quoridor.neural_ai.policy import load_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--heuristic-time", type=float, default=0.2)
    parser.add_argument("--heuristic-depth", type=int, default=3)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("data/neural_evaluation.json"))
    args = parser.parse_args()
    if args.games < 2 or args.simulations < 1 or args.max_plies < 1:
        parser.error("--games must be >= 2; simulations and max-plies must be positive")

    model = load_model(args.checkpoint, device=args.device)
    model.eval()
    heuristic_config = AIConfig(time_limit=args.heuristic_time, max_depth=args.heuristic_depth)
    rows = []

    for game in range(args.games):
        neural_player = game % 2
        state = GameState.initial()
        search = MCTS(model, simulations=args.simulations, device=args.device)
        heuristic = QuoridorAI(heuristic_config)
        while state.winner is None and state.ply < args.max_plies:
            if state.current_player == neural_player:
                action_index, _ = search.choose_action(state, temperature=0.0)
            else:
                result = heuristic.choose_action(state)
                action = result.action
                payload = action.destination if action.kind == "move" else action.wall
                action_index = encode_action((action.kind, payload))
            state = apply_action_index(state, action_index)

        winner = state.winner
        rows.append({
            "game": game + 1,
            "neural_player": neural_player,
            "winner": winner,
            "neural_win": winner == neural_player if winner is not None else None,
            "plies": state.ply,
            "result": "win" if winner is not None else "draw_max_plies",
        })
        print(
            f"[{game + 1}/{args.games}] neural=P{neural_player} "
            f"winner={winner} plies={state.ply}",
            flush=True,
        )

    decided = [row for row in rows if row["neural_win"] is not None]
    wins = sum(row["neural_win"] for row in decided)
    report = {
        "checkpoint": str(args.checkpoint),
        "games": len(rows),
        "neural_wins": wins,
        "decided_games": len(decided),
        "neural_win_rate": wins / len(decided) if decided else None,
        "seat_split": {
            str(player): sum(row["neural_win"] is True for row in rows if row["neural_player"] == player)
            for player in (0, 1)
        },
        "results": rows,
        "note": "This is a small paired-seat smoke evaluation, not a statistically conclusive rating.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluation report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
