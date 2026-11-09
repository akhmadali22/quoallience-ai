#!/usr/bin/env python3
"""Paired-seat MCTS-vs-MCTS tournament for measuring search-budget scaling."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

from quoridor.engine import GameState
from quoridor.neural_ai.encoder import apply_action_index
from quoridor.neural_ai.mcts import MCTS
from quoridor.neural_ai.policy import load_model


def play_game(
    model,
    *,
    pair_index: int,
    seed: int,
    low_simulations: int,
    high_simulations: int,
    low_player: int,
    max_plies: int,
    device: str,
    category_exploration: float,
    repetition_penalty: float,
    pawn_progress_bonus: float,
) -> dict:
    """Play one game; both agents share weights and differ only in MCTS budget."""
    budgets = {
        "low": low_simulations,
        "high": high_simulations,
    }
    player_agents = {
        low_player: "low",
        1 - low_player: "high",
    }
    searches = {
        player: MCTS(
            model,
            simulations=budgets[player_agents[player]],
            device=device,
            category_exploration=category_exploration,
            repetition_penalty=repetition_penalty,
            pawn_progress_bonus=pawn_progress_bonus,
        )
        for player in (0, 1)
    }
    # Random tie-breaking streams follow the agent identity, not the seat.
    rngs = {
        "low": random.Random(seed ^ 0x10A51),
        "high": random.Random(seed ^ 0x21B62),
    }
    pawn_history = {0: [], 1: []}
    state = GameState.initial()
    pawn_history[0].append(state.pawns[0])
    pawn_history[1].append(state.pawns[1])
    trace = []
    metrics = {
        label: {
            "turns": 0,
            "expanded_nodes": [],
            "nn_evaluations": [],
            "average_leaf_depth": [],
            "root_policy_entropy": [],
            "root_pawn_visit_ratio": [],
            "root_wall_visit_ratio": [],
            "selected_q": [],
            "selected_visits": [],
        }
        for label in ("low", "high")
    }

    while state.winner is None and state.ply < max_plies:
        player = state.current_player
        label = player_agents[player]
        search = searches[player]
        action, _visit_policy = search.choose_action(
            state,
            temperature=0.0,
            rng=rngs[label],
            recent_pawn_positions=pawn_history[player],
        )
        stats = search.last_search_stats
        selected = stats.get("root_action_stats", {}).get(str(action), {})
        metric = metrics[label]
        metric["turns"] += 1
        for key in (
            "expanded_nodes",
            "nn_evaluations",
            "average_leaf_depth",
            "root_policy_entropy",
            "root_pawn_visit_ratio",
            "root_wall_visit_ratio",
        ):
            if stats.get(key) is not None:
                metric[key].append(float(stats[key]))
        if selected.get("q") is not None:
            metric["selected_q"].append(float(selected["q"]))
        metric["selected_visits"].append(int(selected.get("visits", 0)))

        next_state = apply_action_index(state, action)
        trace.append({
            "ply": state.ply,
            "player": player,
            "agent": label,
            "simulations": budgets[label],
            "action": action,
            "category": selected.get("category"),
            "prior": selected.get("prior"),
            "q": selected.get("q"),
            "visits": selected.get("visits"),
            "root_value": stats.get("root_value"),
            "root_policy_entropy": stats.get("root_policy_entropy"),
            "expanded_nodes": stats.get("expanded_nodes"),
            "nn_evaluations": stats.get("nn_evaluations"),
            "average_leaf_depth": stats.get("average_leaf_depth"),
            "root_pawn_visit_ratio": stats.get("root_pawn_visit_ratio"),
            "root_wall_visit_ratio": stats.get("root_wall_visit_ratio"),
        })
        if next_state.pawns[player] != state.pawns[player]:
            pawn_history[player].append(next_state.pawns[player])
        state = next_state

    winner_agent = None if state.winner is None else player_agents[state.winner]
    low_outcome = (
        "draw" if winner_agent is None
        else "win" if winner_agent == "low"
        else "loss"
    )

    def summarize(values: list[float]) -> float | None:
        return mean(values) if values else None

    agent_summary = {}
    for label, values in metrics.items():
        agent_summary[label] = {
            "turns": values["turns"],
            "mean_expanded_nodes": summarize(values["expanded_nodes"]),
            "mean_nn_evaluations": summarize(values["nn_evaluations"]),
            "mean_leaf_depth": summarize(values["average_leaf_depth"]),
            "mean_root_policy_entropy": summarize(values["root_policy_entropy"]),
            "mean_root_pawn_visit_ratio": summarize(values["root_pawn_visit_ratio"]),
            "mean_root_wall_visit_ratio": summarize(values["root_wall_visit_ratio"]),
            "mean_selected_q": summarize(values["selected_q"]),
            "mean_selected_visits": summarize(values["selected_visits"]),
        }

    return {
        "pair": pair_index,
        "seed": seed,
        "low_player": low_player,
        "winner": state.winner,
        "winner_agent": winner_agent,
        "result_for_low": low_outcome,
        "plies": state.ply,
        "agent_metrics": agent_summary,
        "action_trace": trace,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--low-simulations", type=int, default=64)
    parser.add_argument("--high-simulations", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=20,
                        help="Number of paired seeds; each pair plays two games with seats swapped.")
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--category-exploration", type=float, default=0.35)
    parser.add_argument("--repetition-penalty", type=float, default=0.75)
    parser.add_argument("--pawn-progress-bonus", type=float, default=0.0)
    parser.add_argument("--output", type=Path,
                        default=Path("data/mcts_scaling/eval_64_vs_128.json"))
    args = parser.parse_args()

    if args.low_simulations < 1 or args.high_simulations <= args.low_simulations:
        parser.error("simulation budgets must satisfy 1 <= low < high")
    if args.pairs < 1 or args.max_plies < 1:
        parser.error("--pairs and --max-plies must be positive")
    if (args.category_exploration < 0 or args.repetition_penalty < 0
            or args.pawn_progress_bonus < 0):
        parser.error("exploration and preference values must be non-negative")

    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    model = load_model(args.checkpoint, device=args.device)
    model.eval()
    games = []
    pair_results = []

    for pair_index in range(args.pairs):
        pair_seed = args.seed + pair_index
        pair_games = []
        for low_player in (0, 1):
            game_number = pair_index * 2 + low_player + 1
            print(
                f"[{game_number}/{args.pairs * 2}] pair={pair_index + 1} "
                f"seed={pair_seed} low_player=P{low_player} "
                f"budgets={args.low_simulations}/{args.high_simulations}",
                flush=True,
            )
            result = play_game(
                model,
                pair_index=pair_index + 1,
                seed=pair_seed,
                low_simulations=args.low_simulations,
                high_simulations=args.high_simulations,
                low_player=low_player,
                max_plies=args.max_plies,
                device=args.device,
                category_exploration=args.category_exploration,
                repetition_penalty=args.repetition_penalty,
                pawn_progress_bonus=args.pawn_progress_bonus,
            )
            games.append(result)
            pair_games.append(result)
            print(
                f"  low={result['result_for_low']} winner={result['winner_agent']} "
                f"plies={result['plies']}",
                flush=True,
            )

        low_points = sum(
            1.0 if game["result_for_low"] == "win"
            else 0.5 if game["result_for_low"] == "draw"
            else 0.0
            for game in pair_games
        )
        pair_results.append({
            "pair": pair_index + 1,
            "seed": pair_seed,
            "low_points": low_points,
            "high_points": 2.0 - low_points,
            "pair_result_for_low": (
                "win" if low_points > 1.0
                else "loss" if low_points < 1.0
                else "draw"
            ),
        })

    low_wins = sum(game["result_for_low"] == "win" for game in games)
    low_losses = sum(game["result_for_low"] == "loss" for game in games)
    draws = sum(game["result_for_low"] == "draw" for game in games)
    low_points = low_wins + 0.5 * draws
    report = {
        "checkpoint": str(args.checkpoint),
        "protocol": "same checkpoint, MCTS-vs-MCTS, paired seed, swapped seats, deterministic max-visit selection",
        "config": {
            "pairs": args.pairs,
            "games": len(games),
            "low_simulations": args.low_simulations,
            "high_simulations": args.high_simulations,
            "max_plies": args.max_plies,
            "device": args.device,
            "base_seed": args.seed,
            "category_exploration": args.category_exploration,
            "repetition_penalty": args.repetition_penalty,
            "pawn_progress_bonus": args.pawn_progress_bonus,
            "progressive_widening": True,
            "selection_temperature": 0.0,
            "wall_distance_weight": 0.0,
            "wall_route_weight": 0.0,
        },
        "low_agent": {
            "wins": low_wins,
            "losses": low_losses,
            "draws": draws,
            "score_rate": low_points / len(games) if games else None,
        },
        "high_agent": {
            "wins": low_losses,
            "losses": low_wins,
            "draws": draws,
            "score_rate": (len(games) - low_points) / len(games) if games else None,
        },
        "paired_seed_results": pair_results,
        "games": games,
        "interpretation": (
            "This isolates the effect of MCTS simulation budget while holding "
            "checkpoint and search preferences constant. It does not by itself "
            "prove that the policy/value network is well trained."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluation report: {args.output}")
    print(
        f"low={low_wins} high={low_losses} draws={draws} "
        f"low_score_rate={report['low_agent']['score_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
