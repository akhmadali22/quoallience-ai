#!/usr/bin/env python3
"""Paired-seat MCTS tournaments that isolate neural policy and value contributions."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

import torch

from quoridor.engine import GameState
from quoridor.neural_ai.encoder import apply_action_index
from quoridor.neural_ai.mcts import MCTS
from quoridor.neural_ai.policy import load_model


AGENTS = {
    "full": ("neural", "neural"),
    "uniform_policy": ("uniform", "neural"),
    "zero_value": ("neural", "zero"),
    "uniform_policy_zero_value": ("uniform", "zero"),
}
AGENT_SEED_SALTS = {
    "full": 0x10A51,
    "uniform_policy": 0x21B62,
    "zero_value": 0x32C73,
    "uniform_policy_zero_value": 0x43D84,
}
MATCHUPS = (
    ("full", "uniform_policy"),
    ("full", "zero_value"),
    ("full", "uniform_policy_zero_value"),
)


class AblatedModel:
    """Wrap a trained model while selectively removing policy/value outputs."""

    def __init__(self, model, *, policy_mode: str, value_mode: str):
        if policy_mode not in {"neural", "uniform"}:
            raise ValueError(f"Unsupported policy mode: {policy_mode}")
        if value_mode not in {"neural", "zero"}:
            raise ValueError(f"Unsupported value mode: {value_mode}")
        self.model = model
        self.policy_mode = policy_mode
        self.value_mode = value_mode

    def __call__(self, inputs):
        logits, value = self.model(inputs)
        if self.policy_mode == "uniform":
            logits = torch.zeros_like(logits)
        if self.value_mode == "zero":
            value = torch.zeros_like(value)
        return logits, value


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def play_game(
    model,
    *,
    agent_a: str,
    agent_b: str,
    seed: int,
    a_player: int,
    simulations: int,
    max_plies: int,
    device: str,
    c_puct: float,
    progressive_widening: bool,
    initial_wall_candidates: int,
) -> dict:
    """Play one MCTS-vs-MCTS game with identical search settings."""
    player_agents = {a_player: agent_a, 1 - a_player: agent_b}
    wrapped_models = {}
    searches = {}
    for name in (agent_a, agent_b):
        policy_mode, value_mode = AGENTS[name]
        wrapped_models[name] = AblatedModel(
            model, policy_mode=policy_mode, value_mode=value_mode
        )

    for player, name in player_agents.items():
        searches[player] = MCTS(
            wrapped_models[name],
            simulations=simulations,
            c_puct=c_puct,
            device=device,
            randomize_selection_ties=True,
            progressive_widening=progressive_widening,
            initial_wall_candidates=initial_wall_candidates,
            category_exploration=0.0,
            repetition_penalty=0.0,
            pawn_progress_bonus=0.0,
            wall_distance_weight=0.0,
            wall_route_weight=0.0,
        )

    rngs = {
        name: random.Random(seed ^ AGENT_SEED_SALTS[name])
        for name in (agent_a, agent_b)
    }
    state = GameState.initial()
    trace = []
    metrics = {
        name: {
            "turns": 0,
            "root_values": [],
            "root_policy_entropies": [],
            "expanded_nodes": [],
            "nn_evaluations": [],
            "selected_q": [],
            "selected_visits": [],
            "pawn_visit_ratios": [],
            "wall_visit_ratios": [],
        }
        for name in (agent_a, agent_b)
    }

    while state.winner is None and state.ply < max_plies:
        player = state.current_player
        name = player_agents[player]
        search = searches[player]
        action, _policy = search.choose_action(
            state, temperature=0.0, rng=rngs[name]
        )
        stats = search.last_search_stats
        selected = stats.get("root_action_stats", {}).get(str(action), {})
        agent_metrics = metrics[name]
        agent_metrics["turns"] += 1
        for key, source in (
            ("root_values", "root_value"),
            ("root_policy_entropies", "root_policy_entropy"),
            ("expanded_nodes", "expanded_nodes"),
            ("nn_evaluations", "nn_evaluations"),
            ("pawn_visit_ratios", "root_pawn_visit_ratio"),
            ("wall_visit_ratios", "root_wall_visit_ratio"),
        ):
            value = stats.get(source)
            if value is not None:
                agent_metrics[key].append(float(value))
        if selected.get("q") is not None:
            agent_metrics["selected_q"].append(float(selected["q"]))
        agent_metrics["selected_visits"].append(int(selected.get("visits", 0)))

        next_state = apply_action_index(state, action)
        trace.append({
            "ply": state.ply,
            "player": player,
            "agent": name,
            "action": action,
            "category": selected.get("category"),
            "prior": selected.get("prior"),
            "q": selected.get("q"),
            "visits": selected.get("visits"),
            "root_value": stats.get("root_value"),
            "root_policy_entropy": stats.get("root_policy_entropy"),
            "expanded_nodes": stats.get("expanded_nodes"),
            "nn_evaluations": stats.get("nn_evaluations"),
            "root_pawn_visit_ratio": stats.get("root_pawn_visit_ratio"),
            "root_wall_visit_ratio": stats.get("root_wall_visit_ratio"),
        })
        state = next_state

    winner_agent = None if state.winner is None else player_agents[state.winner]
    result_for_a = (
        "draw" if winner_agent is None
        else "win" if winner_agent == agent_a
        else "loss"
    )
    summaries = {
        name: {
            "turns": values["turns"],
            "mean_root_value": _mean(values["root_values"]),
            "mean_root_policy_entropy": _mean(values["root_policy_entropies"]),
            "mean_expanded_nodes": _mean(values["expanded_nodes"]),
            "mean_nn_evaluations": _mean(values["nn_evaluations"]),
            "mean_selected_q": _mean(values["selected_q"]),
            "mean_selected_visits": _mean(values["selected_visits"]),
            "mean_pawn_visit_ratio": _mean(values["pawn_visit_ratios"]),
            "mean_wall_visit_ratio": _mean(values["wall_visit_ratios"]),
        }
        for name, values in metrics.items()
    }
    return {
        "seed": seed,
        "agent_a": agent_a,
        "agent_b": agent_b,
        "a_player": a_player,
        "winner": state.winner,
        "winner_agent": winner_agent,
        "result_for_a": result_for_a,
        "plies": state.ply,
        "agent_metrics": summaries,
        "action_trace": trace,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument(
        "--pairs", type=int, default=10,
        help="Paired seeds per matchup; each pair plays two games with seats swapped.",
    )
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument(
        "--progressive-widening", action="store_true",
        help="Enable production progressive widening. Disabled by default to isolate policy priors from candidate admission.",
    )
    parser.add_argument("--initial-wall-candidates", type=int, default=8)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/mcts_ablation/policy_value_ablation.json"),
    )
    args = parser.parse_args()

    if args.simulations < 1 or args.pairs < 1 or args.max_plies < 1:
        parser.error("--simulations, --pairs, and --max-plies must be positive")
    if args.c_puct < 0 or args.initial_wall_candidates < 0:
        parser.error("--c-puct and --initial-wall-candidates must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    model = load_model(args.checkpoint, device=args.device)
    model.eval()
    games = []
    matchup_reports = []

    for matchup_index, (agent_a, agent_b) in enumerate(MATCHUPS, 1):
        pair_results = []
        matchup_games = []
        a_wins = a_losses = draws = 0
        print(f"\n=== Matchup {matchup_index}/{len(MATCHUPS)}: {agent_a} vs {agent_b} ===", flush=True)

        for pair_index in range(args.pairs):
            seed = args.seed + pair_index
            pair_games = []
            for a_player in (0, 1):
                game_number = pair_index * 2 + a_player + 1
                print(
                    f"[{game_number}/{args.pairs * 2}] seed={seed} "
                    f"{agent_a}=P{a_player} simulations={args.simulations}",
                    flush=True,
                )
                result = play_game(
                    model,
                    agent_a=agent_a,
                    agent_b=agent_b,
                    seed=seed,
                    a_player=a_player,
                    simulations=args.simulations,
                    max_plies=args.max_plies,
                    device=args.device,
                    c_puct=args.c_puct,
                    progressive_widening=args.progressive_widening,
                    initial_wall_candidates=args.initial_wall_candidates,
                )
                games.append(result)
                matchup_games.append(result)
                pair_games.append(result)
                print(
                    f"  result={result['result_for_a']} "
                    f"winner={result['winner_agent']} plies={result['plies']}",
                    flush=True,
                )

            pair_points = sum(
                1.0 if game["result_for_a"] == "win"
                else 0.5 if game["result_for_a"] == "draw"
                else 0.0
                for game in pair_games
            )
            pair_result = (
                "win" if pair_points > 1.0
                else "loss" if pair_points < 1.0
                else "draw"
            )
            pair_results.append({
                "pair": pair_index + 1,
                "seed": seed,
                "a_points": pair_points,
                "b_points": 2.0 - pair_points,
                "result_for_a": pair_result,
            })

        a_wins = sum(game["result_for_a"] == "win" for game in matchup_games)
        a_losses = sum(game["result_for_a"] == "loss" for game in matchup_games)
        draws = sum(game["result_for_a"] == "draw" for game in matchup_games)
        a_score = a_wins + 0.5 * draws
        matchup_reports.append({
            "agent_a": agent_a,
            "agent_b": agent_b,
            "games": len(matchup_games),
            "agent_a_wins": a_wins,
            "agent_a_losses": a_losses,
            "draws": draws,
            "agent_a_score_rate": a_score / len(matchup_games),
            "agent_b_score_rate": (len(matchup_games) - a_score) / len(matchup_games),
            "paired_results": pair_results,
        })

    report = {
        "checkpoint": str(args.checkpoint),
        "protocol": (
            "MCTS-vs-MCTS; one checkpoint; paired seeds; swapped seats; "
            "deterministic max-visit selection; no root noise"
        ),
        "agent_definitions": {
            "full": {"policy": "neural", "value": "neural"},
            "uniform_policy": {"policy": "uniform legal-action prior", "value": "neural"},
            "zero_value": {"policy": "neural", "value": "zero"},
            "uniform_policy_zero_value": {"policy": "uniform legal-action prior", "value": "zero"},
        },
        "config": {
            "simulations": args.simulations,
            "pairs_per_matchup": args.pairs,
            "max_plies": args.max_plies,
            "device": args.device,
            "seed": args.seed,
            "c_puct": args.c_puct,
            "progressive_widening": args.progressive_widening,
            "initial_wall_candidates": args.initial_wall_candidates,
            "category_exploration": 0.0,
            "repetition_penalty": 0.0,
            "pawn_progress_bonus": 0.0,
            "wall_distance_weight": 0.0,
            "wall_route_weight": 0.0,
        },
        "matchups": matchup_reports,
        "games": games,
        "interpretation": (
            "The full agent is compared separately with uniform-policy, zero-value, "
            "and combined ablations. Progressive widening is disabled by default "
            "so policy ablation does not also change which wall candidates are admitted. "
            "All agents still use PUCT MCTS. This measures component contribution "
            "under this search protocol, not absolute playing strength."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nEvaluation report: {args.output}")
    for item in matchup_reports:
        print(
            f"{item['agent_a']} vs {item['agent_b']}: "
            f"{item['agent_a_wins']}-{item['agent_a_losses']}-{item['draws']} "
            f"score_rate={item['agent_a_score_rate']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
