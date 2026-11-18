#!/usr/bin/env python3
"""Paired-seat MCTS versus quoridor.advanced_ai baseline tournament."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean

from quoridor.engine import GameState, shortest_distance
from quoridor.neural_ai.encoder import apply_action_index, encode_action
from quoridor.neural_ai.mcts import MCTS
from quoridor.neural_ai.policy import load_model
from quoridor.advanced_ai import AIConfig, QuoridorAI


def action_index(action) -> int:
    if action.kind == "move":
        return encode_action(("move", tuple(action.destination)))
    return encode_action(("wall", action.wall))


def action_effect(state: GameState, index: int) -> dict:
    """Measure immediate route impact of the selected action (diagnostic only)."""
    player = state.current_player
    own_before = shortest_distance(state, player)
    opp_before = shortest_distance(state, 1 - player)
    next_state = apply_action_index(state, index)
    own_after = shortest_distance(next_state, player)
    opp_after = shortest_distance(next_state, 1 - player)
    own_delta = None if own_before is None or own_after is None else own_after - own_before
    opp_delta = None if opp_before is None or opp_after is None else opp_after - opp_before
    kind = "wall" if index >= 81 else "pawn"
    urgent = (
        own_before is not None and opp_before is not None
        and (opp_before <= own_before + 1 or opp_before <= 4)
    )
    if kind == "wall" and urgent:
        context = "urgent_wall_response"
    elif kind == "wall":
        context = "strategic_wall_opportunity"
    elif urgent:
        context = "urgent_pawn_response"
    else:
        context = "routine_pawn"
    return {
        "action_kind": kind,
        "decision_context": context,
        "own_distance_before": own_before,
        "own_distance_after": own_after,
        "opponent_distance_before": opp_before,
        "opponent_distance_after": opp_after,
        "own_distance_delta": own_delta,
        "opponent_distance_delta": opp_delta,
    }


def compact_root_actions(root_stats: dict, selected_index: int) -> dict:\n    """Keep selected action plus top visit-ranked alternatives, not every legal edge."""\n    rows = root_stats.get("root_action_stats", {})\n    ranked = sorted(rows.items(), key=lambda item: (item[1].get("visits", 0), item[1].get("q", 0.0)), reverse=True)\n    keep = {str(selected_index)}\n    keep.update(key for key, _ in ranked[:8])\n    keep.update(key for key, row in ranked if row.get("category") == "wall" and row.get("visits", 0) > 0)\n    return {key: rows[key] for key in keep if key in rows}\n\n\ndef play_game(model, *, checkpoint: Path, budget: int, seed: int,
              mcts_player: int, max_plies: int, device: str,
              heuristic_time_limit: float, heuristic_max_depth: int) -> dict:
    rng = random.Random(seed)
    mcts = MCTS(
        model, simulations=budget, c_puct=1.5, device=device,
        category_exploration=0.0, repetition_penalty=0.0,
        pawn_progress_bonus=0.0, progressive_widening=False,
        wall_distance_weight=0.0, wall_route_weight=0.0,
        randomize_selection_ties=False,
    )
    heuristic = QuoridorAI(AIConfig(
        time_limit=heuristic_time_limit, max_depth=heuristic_max_depth
    ))
    state = GameState.initial()
    pawn_history = {0: [state.pawns[0]], 1: [state.pawns[1]]}
    state_history = []
    turns = []
    while state.winner is None and state.ply < max_plies:
        player = state.current_player
        if player == mcts_player:
            action, _ = mcts.choose_action(
                state, temperature=0.0, add_root_noise=False, rng=rng,
                recent_pawn_positions=pawn_history[player],
                recent_state_keys=state_history,
            )
            stats = mcts.last_search_stats
            selected = stats.get("root_action_stats", {}).get(str(action), {})
            effect = action_effect(state, action)
            turn = {
                "ply": state.ply, "player": player, "agent": "mcts",
                "budget": budget, "action_index": action,
                "selected_prior": selected.get("prior"),
                "selected_network_prior": selected.get("network_prior"),
                "selected_q": selected.get("q"),
                "selected_visits": selected.get("visits"),
                "expanded_nodes": stats.get("expanded_nodes"),
                "nn_evaluations": stats.get("nn_evaluations"),
                "mean_leaf_depth": stats.get("average_leaf_depth"),
                "pawn_visit_ratio": stats.get("root_pawn_visit_ratio"),
                "wall_visit_ratio": stats.get("root_wall_visit_ratio"),
                "root_action_stats": compact_root_actions(stats, action),
                **effect,
            }
        else:
            result = heuristic.choose_action(state)
            action = action_index(result.action)
            # Verify conversion and legality by applying through the shared engine.
            effect = action_effect(state, action)
            turn = {
                "ply": state.ply, "player": player, "agent": "advanced_ai",
                "action_index": action, "heuristic_score": result.score,
                "heuristic_depth": result.depth,
                "heuristic_elapsed_ms": result.elapsed_ms,
                "heuristic_timed_out": result.timed_out,
                "heuristic_nodes": result.nodes,
                **effect,
            }
        next_state = apply_action_index(state, action)
        turns.append(turn)
        if next_state.pawns[player] != state.pawns[player]:
            pawn_history[player].append(next_state.pawns[player])
        from quoridor.engine import state_repetition_key
        state_history.append(state_repetition_key(state))
        state_history = state_history[-5:]
        state = next_state

    winner_agent = (
        None if state.winner is None
        else "mcts" if state.winner == mcts_player else "advanced_ai"
    )
    return {
        "seed": seed, "mcts_player": mcts_player,
        "winner": state.winner, "winner_agent": winner_agent,
        "result_for_mcts": (
            "draw" if winner_agent is None
            else "win" if winner_agent == "mcts" else "loss"
        ),
        "plies": state.ply, "turns": turns,
    }


def summarize(games: list[dict]) -> dict:
    wins = sum(g["result_for_mcts"] == "win" for g in games)
    losses = sum(g["result_for_mcts"] == "loss" for g in games)
    draws = sum(g["result_for_mcts"] == "draw" for g in games)
    return {
        "wins": wins, "losses": losses, "draws": draws,
        "score_rate": (wins + 0.5 * draws) / len(games) if games else None,
        "mean_plies": mean([g["plies"] for g in games]) if games else None,
        "draw_at_ply_limit": sum(
            g["result_for_mcts"] == "draw" and g["plies"] >= 200 for g in games
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--budget", type=int, choices=(32, 128, 256), required=True)\n    parser.add_argument("--label", default="candidate")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--heuristic-time-limit", type=float, default=0.5)
    parser.add_argument("--heuristic-max-depth", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 1 or args.max_plies < 1 or args.heuristic_time_limit <= 0:
        parser.error("pairs, max_plies and heuristic time limit must be positive")
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    model = load_model(args.checkpoint, device=args.device)
    model.eval()
    games, paired = [], []
    for pair in range(args.pairs):
        seed = args.seed + pair
        pair_games = []
        for mcts_player in (0, 1):
            game_no = pair * 2 + mcts_player + 1
            print(
                f"[{game_no}/{args.pairs * 2}] seed={seed} "
                f"MCTS=P{mcts_player} budget={args.budget}", flush=True
            )
            game = play_game(
                model, checkpoint=args.checkpoint, budget=args.budget,
                seed=seed, mcts_player=mcts_player, max_plies=args.max_plies,
                device=args.device,
                heuristic_time_limit=args.heuristic_time_limit,
                heuristic_max_depth=args.heuristic_max_depth,
            )
            games.append(game)
            pair_games.append(game)
            print(f"  {game['result_for_mcts']} plies={game['plies']}", flush=True)
        points = sum(
            1.0 if g["result_for_mcts"] == "win"
            else 0.5 if g["result_for_mcts"] == "draw" else 0.0
            for g in pair_games
        )
        paired.append({
            "pair": pair + 1, "seed": seed, "mcts_points": points,
            "heuristic_points": 2.0 - points,
            "pair_result_for_mcts": (
                "win" if points > 1 else "loss" if points < 1 else "tie"
            ),
        })
    report = {
        "experiment": "mcts_vs_advanced_heuristic",\n        "agent_label": args.label,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "config": {
            "budget": args.budget, "pairs": args.pairs,
            "games": len(games), "base_seed": args.seed,
            "max_plies": args.max_plies, "device": args.device,
            "c_puct": 1.5, "temperature": 0.0, "root_noise": False,
            "progressive_widening": False, "category_exploration": 0.0,
            "repetition_penalty": 0.0, "pawn_progress_bonus": 0.0,
            "wall_distance_weight": 0.0, "wall_route_weight": 0.0,
            "heuristic_time_limit_seconds": args.heuristic_time_limit,
            "heuristic_max_depth": args.heuristic_max_depth,
        },
        "overall": summarize(games),
        "paired_seed_results": paired,
        "games": games,
        "interpretation_guardrail": (
            "This measures performance against this specific heuristic, not "
            "general playing strength or compute-matched performance. "
            "Route deltas are immediate structural diagnostics, not a game-theoretic oracle."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}")
    print(json.dumps(report["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
