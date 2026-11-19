#!/usr/bin/env python3
"""Seeded paired-seat evaluation: neural MCTS agent versus heuristic AI."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from quoridor.advanced_ai import AIConfig, QuoridorAI
from quoridor.engine import GameState, shortest_distance, shortest_path_metrics
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
    parser.add_argument("--repetition-penalty", type=float, default=0.75)
    parser.add_argument("--pawn-progress-bonus", type=float, default=0.0)
    parser.add_argument("--wall-distance-weight", type=float, default=0.0)
    parser.add_argument("--wall-route-weight", type=float, default=0.0)
    parser.add_argument("--repetition-window", type=int, default=4)
    parser.add_argument("--selection-temperature", type=float, default=0.2)
    parser.add_argument("--opponent", choices=("heuristic", "random"), default="heuristic")
    parser.add_argument("--output", type=Path, default=Path("data/neural_evaluation.json"))
    args = parser.parse_args()
    if args.games < 2 or args.simulations < 1 or args.max_plies < 1:
        parser.error("--games must be >= 2; simulations and max-plies must be positive")
    if args.category_exploration < 0 or args.selection_temperature < 0 or args.repetition_penalty < 0 or args.pawn_progress_bonus < 0 or args.wall_distance_weight < 0 or args.wall_route_weight < 0 or args.repetition_window < 1:
        parser.error("exploration, temperature and penalty values must be non-negative; repetition-window must be positive")

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
            repetition_penalty=args.repetition_penalty,
            pawn_progress_bonus=args.pawn_progress_bonus,
            repetition_window=args.repetition_window,
            wall_distance_weight=args.wall_distance_weight,
            wall_route_weight=args.wall_route_weight,
        )
        heuristic = QuoridorAI(heuristic_config)
        opponent_rng = random.Random(game_seed ^ 0x5EED5EED)
        action_trace = []
        neural_pawn_actions = 0
        neural_wall_actions = 0
        pawn_progress_moves = 0
        pawn_regression_moves = 0
        pawn_reversal_moves = 0
        own_distance_improvements = 0
        own_distance_regressions = 0
        opponent_distance_increases = 0
        opponent_distance_decreases = 0
        opponent_distance_delta_sum = 0
        opponent_distance_changed_actions = 0
        wall_opponent_path_increases = 0
        wall_opponent_path_decreases = 0
        wall_opponent_distance_delta_sum = 0
        wall_own_path_increases = 0
        wall_own_path_decreases = 0
        wall_own_distance_delta_sum = 0
        wall_actions_with_distance_effect = 0
        wall_opponent_route_reduction_sum = 0.0
        wall_own_route_reduction_sum = 0.0
        wall_net_route_score_sum = 0.0
        wall_opponent_routes_reduced = 0
        wall_opponent_routes_increased = 0
        wall_own_routes_reduced = 0
        wall_own_routes_increased = 0
        wall_route_samples = 0
        pawn_position_history = [state.pawns[neural_player]]

        while state.winner is None and state.ply < args.max_plies:
            if state.current_player == neural_player:
                action_index, _ = search.choose_action(
                    state, temperature=args.selection_temperature, rng=rng,
                    recent_pawn_positions=pawn_position_history,
                )
                selected = search.last_search_stats.get('root_action_stats', {}).get(str(action_index), {})
                before_own = state.pawns[neural_player]
                before_opponent = state.pawns[1 - neural_player]
                own_distance_before = shortest_distance(state, neural_player)
                opponent_distance_before = shortest_distance(state, 1 - neural_player)
                next_state = apply_action_index(state, action_index)
                own_distance_after = shortest_distance(next_state, neural_player)
                opponent_distance_after = shortest_distance(next_state, 1 - neural_player)
                pawn_reversal = (
                    action_index < MOVE_ACTIONS
                    and len(pawn_position_history) >= 2
                    and next_state.pawns[neural_player] == pawn_position_history[-2]
                )
                if action_index < MOVE_ACTIONS:
                    neural_pawn_actions += 1
                    if own_distance_after is not None and own_distance_before is not None:
                        if own_distance_after < own_distance_before:
                            pawn_progress_moves += 1
                        elif own_distance_after > own_distance_before:
                            pawn_regression_moves += 1
                    if pawn_reversal:
                        pawn_reversal_moves += 1
                    pawn_position_history.append(next_state.pawns[neural_player])
                else:
                    neural_wall_actions += 1
                if own_distance_after is not None and own_distance_before is not None:
                    if own_distance_after < own_distance_before:
                        own_distance_improvements += 1
                    elif own_distance_after > own_distance_before:
                        own_distance_regressions += 1
                opponent_distance_delta = (
                    None if opponent_distance_before is None or opponent_distance_after is None
                    else opponent_distance_before - opponent_distance_after
                )
                if opponent_distance_delta is not None:
                    opponent_distance_delta_sum += opponent_distance_delta
                    if opponent_distance_delta > 0:
                        opponent_distance_increases += 1
                        opponent_distance_changed_actions += 1
                    elif opponent_distance_delta < 0:
                        opponent_distance_decreases += 1
                        opponent_distance_changed_actions += 1

                wall_opponent_delta = None
                wall_own_delta = None
                wall_opponent_route_reduction = None
                wall_own_route_reduction = None
                wall_net_route_score = None
                if action_index >= MOVE_ACTIONS:
                    if opponent_distance_delta is not None:
                        wall_opponent_delta = opponent_distance_delta
                        wall_actions_with_distance_effect += int(opponent_distance_delta != 0)
                        wall_opponent_distance_delta_sum += opponent_distance_delta
                        if opponent_distance_delta > 0:
                            wall_opponent_path_increases += 1
                        elif opponent_distance_delta < 0:
                            wall_opponent_path_decreases += 1
                    if own_distance_before is not None and own_distance_after is not None:
                        wall_own_delta = own_distance_after - own_distance_before
                        wall_own_distance_delta_sum += wall_own_delta
                        if wall_own_delta > 0:
                            wall_own_path_increases += 1
                        elif wall_own_delta < 0:
                            wall_own_path_decreases += 1
                if action_index >= MOVE_ACTIONS:
                    own_routes_before = shortest_path_metrics(state, neural_player)[1]
                    own_routes_after = shortest_path_metrics(next_state, neural_player)[1]
                    opponent_routes_before = shortest_path_metrics(state, 1 - neural_player)[1]
                    opponent_routes_after = shortest_path_metrics(next_state, 1 - neural_player)[1]
                    wall_opponent_route_reduction = math.log1p(opponent_routes_before) - math.log1p(opponent_routes_after)
                    wall_own_route_reduction = math.log1p(own_routes_before) - math.log1p(own_routes_after)
                    wall_net_route_score = wall_opponent_route_reduction - wall_own_route_reduction
                    wall_route_samples += 1
                    wall_opponent_route_reduction_sum += wall_opponent_route_reduction
                    wall_own_route_reduction_sum += wall_own_route_reduction
                    wall_net_route_score_sum += wall_net_route_score
                    if wall_opponent_route_reduction > 1e-9:
                        wall_opponent_routes_reduced += 1
                    elif wall_opponent_route_reduction < -1e-9:
                        wall_opponent_routes_increased += 1
                    if wall_own_route_reduction > 1e-9:
                        wall_own_routes_reduced += 1
                    elif wall_own_route_reduction < -1e-9:
                        wall_own_routes_increased += 1
                action_trace.append({
                    'ply': state.ply,
                    'player': state.current_player,
                    'action': action_index,
                    'category': selected.get('category'),
                    'prior': selected.get('prior'),
                    'q': selected.get('q'),
                    'visits': selected.get('visits'),
                    'root_value': search.last_search_stats.get('root_value'),
                    'root_policy_entropy': search.last_search_stats.get('root_policy_entropy'),
                    'root_repetition_adjusted_actions': search.last_search_stats.get('root_repetition_adjusted_actions'),
                    'selection_adjustment': selected.get('selection_adjustment'),
                    'root_pawn_visits': search.last_search_stats.get('root_pawn_visits'),
                    'root_wall_visits': search.last_search_stats.get('root_wall_visits'),
                    'root_pawn_visit_ratio': search.last_search_stats.get('root_pawn_visit_ratio'),
                    'pawn_before': before_own,
                    'pawn_after': next_state.pawns[neural_player],
                    'opponent_pawn': before_opponent,
                    'own_distance_before': own_distance_before,
                    'own_distance_after': own_distance_after,
                    'opponent_distance_before': opponent_distance_before,
                    'opponent_distance_after': opponent_distance_after,
                    'opponent_distance_delta': opponent_distance_delta,
                    'wall_opponent_distance_delta': wall_opponent_delta,
                    'wall_own_distance_delta': wall_own_delta,
                    'wall_opponent_route_reduction': wall_opponent_route_reduction,
                    'wall_own_route_reduction': wall_own_route_reduction,
                    'wall_net_route_score': wall_net_route_score,
                    'pawn_distance_delta': (
                        None if own_distance_before is None or own_distance_after is None
                        else own_distance_before - own_distance_after
                    ),
                    'pawn_reversal': pawn_reversal,
                })
                state = next_state
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
        value_target = 0.0 if winner is None else (1.0 if winner == neural_player else -1.0)
        for trace in action_trace:
            trace["value_target"] = value_target
            trace["root_value_error"] = (
                None if trace["root_value"] is None
                else abs(float(trace["root_value"]) - value_target)
            )
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
            "pawn_progress_moves": pawn_progress_moves,
            "pawn_regression_moves": pawn_regression_moves,
            "pawn_reversal_moves": pawn_reversal_moves,
            "own_distance_improvements": own_distance_improvements,
            "own_distance_regressions": own_distance_regressions,
            "opponent_distance_increases": opponent_distance_increases,
            "opponent_distance_decreases": opponent_distance_decreases,
            "opponent_distance_changed_actions": opponent_distance_changed_actions,
            "opponent_distance_delta_sum": opponent_distance_delta_sum,
            "wall_opponent_path_increases": wall_opponent_path_increases,
            "wall_opponent_path_decreases": wall_opponent_path_decreases,
            "wall_opponent_distance_delta_sum": wall_opponent_distance_delta_sum,
            "wall_own_path_increases": wall_own_path_increases,
            "wall_own_path_decreases": wall_own_path_decreases,
            "wall_own_distance_delta_sum": wall_own_distance_delta_sum,
            "wall_actions_with_distance_effect": wall_actions_with_distance_effect,
            "wall_opponent_route_reduction_sum": wall_opponent_route_reduction_sum,
            "wall_own_route_reduction_sum": wall_own_route_reduction_sum,
            "wall_net_route_score_sum": wall_net_route_score_sum,
            "wall_opponent_routes_reduced": wall_opponent_routes_reduced,
            "wall_opponent_routes_increased": wall_opponent_routes_increased,
            "wall_own_routes_reduced": wall_own_routes_reduced,
            "wall_own_routes_increased": wall_own_routes_increased,
            "wall_route_samples": wall_route_samples,
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
    all_traces = [trace for row in rows for trace in row["action_trace"]]
    value_errors = [
        trace["root_value_error"] for trace in all_traces
        if trace["root_value_error"] is not None
    ]
    root_values = [
        trace["root_value"] for trace in all_traces
        if trace["root_value"] is not None
    ]
    opponent_deltas = [
        trace["opponent_distance_delta"] for trace in all_traces
        if trace["opponent_distance_delta"] is not None
    ]
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
            "repetition_penalty": search.repetition_penalty,
            "pawn_progress_bonus": search.pawn_progress_bonus,
            "wall_distance_weight": search.wall_distance_weight,
            "wall_route_weight": search.wall_route_weight,
            "repetition_window": search.repetition_window,
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
        "strategic_diagnostics": {
            "pawn_progress_moves": sum(row["pawn_progress_moves"] for row in rows),
            "pawn_regression_moves": sum(row["pawn_regression_moves"] for row in rows),
            "pawn_reversal_moves": sum(row["pawn_reversal_moves"] for row in rows),
            "own_distance_improvements": sum(row["own_distance_improvements"] for row in rows),
            "own_distance_regressions": sum(row["own_distance_regressions"] for row in rows),
            "opponent_distance_increases": sum(row["opponent_distance_increases"] for row in rows),
            "opponent_distance_decreases": sum(row["opponent_distance_decreases"] for row in rows),
            "opponent_distance_changed_actions": sum(row["opponent_distance_changed_actions"] for row in rows),
            "opponent_distance_delta_sum": sum(row["opponent_distance_delta_sum"] for row in rows),
            "mean_opponent_distance_delta": (
                sum(opponent_deltas) / len(opponent_deltas) if opponent_deltas else None
            ),
            "wall_opponent_path_increases": sum(row["wall_opponent_path_increases"] for row in rows),
            "wall_opponent_path_decreases": sum(row["wall_opponent_path_decreases"] for row in rows),
            "wall_opponent_distance_delta_sum": sum(row["wall_opponent_distance_delta_sum"] for row in rows),
            "wall_own_path_increases": sum(row["wall_own_path_increases"] for row in rows),
            "wall_own_path_decreases": sum(row["wall_own_path_decreases"] for row in rows),
            "wall_own_distance_delta_sum": sum(row["wall_own_distance_delta_sum"] for row in rows),
            "wall_actions_with_distance_effect": sum(row["wall_actions_with_distance_effect"] for row in rows),
            "wall_opponent_route_reduction_sum": sum(row["wall_opponent_route_reduction_sum"] for row in rows),
            "wall_own_route_reduction_sum": sum(row["wall_own_route_reduction_sum"] for row in rows),
            "wall_net_route_score_sum": sum(row["wall_net_route_score_sum"] for row in rows),
            "wall_opponent_routes_reduced": sum(row["wall_opponent_routes_reduced"] for row in rows),
            "wall_opponent_routes_increased": sum(row["wall_opponent_routes_increased"] for row in rows),
            "wall_own_routes_reduced": sum(row["wall_own_routes_reduced"] for row in rows),
            "wall_own_routes_increased": sum(row["wall_own_routes_increased"] for row in rows),
            "wall_route_samples": sum(row["wall_route_samples"] for row in rows),
        },
        "value_calibration": {
            "samples": len(value_errors),
            "mean_absolute_error": (
                sum(value_errors) / len(value_errors) if value_errors else None
            ),
            "mean_squared_error": (
                sum(error * error for error in value_errors) / len(value_errors)
                if value_errors else None
            ),
            "mean_root_value": (
                sum(root_values) / len(root_values) if root_values else None
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
