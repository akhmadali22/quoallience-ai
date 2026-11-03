#!/usr/bin/env python3
"""Seeded paired-seat evaluation: neural MCTS agent versus heuristic AI."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from quoridor.advanced_ai import AIConfig, QuoridorAI
from quoridor.engine import GameState
from quoridor.neural_ai.encoder import apply_action_index, encode_action, legal_action_indices
from quoridor.neural_ai.mcts import MOVE_ACTIONS, MCTS
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
    parser.add_argument("--seed", type=int, default=2026, help="Base seed; each game uses seed + game index.")
    parser.add_argument("--category-exploration", type=float, default=0.35)
    parser.add_argument("--selection-temperature", type=float, default=0.2)
    parser.add_argument("--opponent", choices=("heuristic", "random"), default="heuristic")
    parser.add_argument("--output", type=Path, default=Path("data/neural_evaluation.json"))
    args = parser.parse_args()
    if args.games < 2 or args.simulations < 1 or args.max_plies < 1:
        parser.error("--games must be >= 2; simulations and max-plies must be positive")
    if args.category_exploration < 0 or args.selection_temperature < 0:
        parser.error("category exploration and selection temperature cannot be negative")

    model = load_model(args.checkpoint, device=args.device)
    model.eval()
    heuristic_config = AIConfig(time_limit=args.heuristic_time, max_depth=args.heuristic_depth)
    rows = []

    for game in range(args.games):
        neural_player = game % 2
        game_seed = args.seed + game
        rng = random.Random(game_seed)
        state = GameState.initial()
        search = MCTS(
            model,
            simulations=args.simulations,
            device=args.device,
            category_exploration=args.category_exploration,
        )
        heuristic = QuoridorAI(heuristic_config)
        opponent_rng = random.Random(game_seed ^ 0x5EED5EED)
        action_trace = []
        neural_pawn_actions = 0
        neural_wall_actions = 0

        while state.winner is None and state.ply < args.max_plies:
            if state.current_player == neural_player:
                action_index, _ = search.choose_action(state, temperature=args.selection_temperature, rng=rng)
                selected = search.last_search_stats.get('root_action_stats', {}).get(str(action_index), {})
                action_trace.append({'ply': state.ply, 'player': state.current_player, 'action': action_index, 'category': selected.get('category'), 'prior': selected.get('prior'), 'q': selected.get('q'), 'visits': selected.get('visits'), 'root_pawn_visits': search.last_search_stats.get('root_pawn_visits'), 'root_wall_visits': search.last_search_stats.get('root_wall_visits'), 'root_pawn_visit_ratio': search.last_search_stats.get('root_pawn_visit_ratio')})
                if action_index < MOVE_ACTIONS:
                    neural_pawn_actions += 1
                else:
                    neural_wall_actions += 1
            else:
                if args.opponent == "random":
                    action_index = opponent_rng.choice(legal_action_indices(state))
                else:
                    result = heuristic.choose_action(state)
                    action = result.action
                    payload = action.destination if action.kind == "move" else action.wall
                    action_index = encode_action((action.kind, payload))
            state = apply_action_index(state, action_index)

        winner = state.winner
        neural_win = winner == neural_player if winner is not None else None
        rows.append({
            "game": game + 1,
            "seed": game_seed,
            "neural_player": neural_player,
            "opponent": args.opponent,
            "action_trace": action_trace,
            "winner": winner,
            "neural_win": neural_win,
            "plies": state.ply,
            "neural_pawn_actions": neural_pawn_actions,
            "neural_wall_actions": neural_wall_actions,
            "neural_pawn_action_ratio": (
                neural_pawn_actions / (neural_pawn_actions + neural_wall_actions)
                if neural_pawn_actions + neural_wall_actions else 0.0
            ),
            "result": "win" if neural_win else (
                "loss" if winner is not None else "draw_max_plies"
            ),
        })
        print(
            f"[{game + 1}/{args.games}] seed={game_seed} neural=P{neural_player} "
            f"winner={winner} plies={state.ply} "
            f"neural_moves={neural_pawn_actions}/{neural_pawn_actions + neural_wall_actions}",
            flush=True,
        )

    decided = [row for row in rows if row["neural_win"] is not None]
    wins = sum(row["neural_win"] for row in decided)
    seat_results = {}
    for player in (0, 1):
        seat_rows = [row for row in rows if row["neural_player"] == player]
        seat_results[str(player)] = {
            "wins": sum(row["result"] == "win" for row in seat_rows),
            "losses": sum(row["result"] == "loss" for row in seat_rows),
            "draws": sum(row["result"] == "draw_max_plies" for row in seat_rows),
            "games": len(seat_rows),
        }

    total_neural_actions = sum(
        row["neural_pawn_actions"] + row["neural_wall_actions"] for row in rows
    )
    total_neural_pawn_actions = sum(row["neural_pawn_actions"] for row in rows)
    report = {
        "checkpoint": str(args.checkpoint),
        "config": {
            "games": args.games,
            "simulations": args.simulations,
            "heuristic_time": args.heuristic_time,
            "heuristic_depth": args.heuristic_depth,
            "max_plies": args.max_plies,
            "device": args.device,
            "base_seed": args.seed,
            "game_seeds": [row["seed"] for row in rows],
            "category_exploration": args.category_exploration,
            "opponent": args.opponent,
            "progressive_widening": True,
            "selection_temperature": args.selection_temperature,
            "tie_breaking": "seeded_random",
        },
        "neural_wins": wins,
        "neural_losses": sum(row["result"] == "loss" for row in rows),
        "draws": sum(row["result"] == "draw_max_plies" for row in rows),
        "decided_games": len(decided),
        "neural_win_rate": wins / len(decided) if decided else None,
        "seat_results": seat_results,
        "neural_action_totals": {
            "pawn_actions": total_neural_pawn_actions,
            "wall_actions": total_neural_actions - total_neural_pawn_actions,
            "pawn_action_ratio": (
                total_neural_pawn_actions / total_neural_actions
                if total_neural_actions else 0.0
            ),
        },
        "results": rows,
        "note": "Seeded paired-seat evaluation with stochastic action sampling when temperature is nonzero. This smoke evaluation is not a statistically conclusive rating.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluation report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
