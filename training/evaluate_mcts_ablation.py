#!/usr/bin/env python3
"""Paired-seat MCTS tournaments that isolate neural policy and value contributions."""
from __future__ import annotations

import argparse
import json
import random
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import torch

from quoridor.engine import GameState, shortest_distance
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


def _record_pawn_position(
    history: dict[int, list[tuple[int, int]]],
    repeat_counts: dict[int, int],
    *,
    player: int,
    position: tuple[int, int],
    action_category: str | None,
) -> bool:
    """Record a pawn destination, ignoring wall turns and other actions."""
    if action_category != "pawn":
        return False
    history[player].append(position)
    repeated = history[player].count(position) > 1
    if repeated:
        repeat_counts[player] += 1
    return repeated


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
    state_seen = Counter()
    action_ngrams = {size: Counter() for size in (4, 6, 8)}
    # Count a position only after an actual pawn move. Wall placements do not
    # create another observation of a pawn position.
    pawn_history = {0: [state.pawns[0]], 1: [state.pawns[1]]}
    pawn_position_repeats = {0: 0, 1: 0}
    consecutive_no_progress = 0
    max_consecutive_no_progress = 0
    no_progress_plies = 0
    progress_plies = 0
    previous_distances = [shortest_distance(state, 0), shortest_distance(state, 1)]

    def state_fingerprint(current):
        return (
            current.pawns,
            tuple(sorted((w.orientation, w.x, w.y) for w in current.walls)),
            current.walls_remaining,
            current.current_player,
        )

    def action_pattern_key(items):
        return tuple((item["player"], item["action"]) for item in items)

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
        fingerprint = state_fingerprint(state)
        state_seen[fingerprint] += 1
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
        next_distances = [
            shortest_distance(next_state, 0),
            shortest_distance(next_state, 1),
        ]
        distance_deltas = [
            None if before is None or after is None else after - before
            for before, after in zip(previous_distances, next_distances)
        ]
        own_delta = distance_deltas[player]
        opponent_delta = distance_deltas[1 - player]
        made_progress = (
            (own_delta is not None and own_delta < 0)
            or (selected.get("category") == "wall" and opponent_delta is not None and opponent_delta > 0)
        )
        if made_progress:
            progress_plies += 1
            consecutive_no_progress = 0
        else:
            no_progress_plies += 1
            consecutive_no_progress += 1
            max_consecutive_no_progress = max(
                max_consecutive_no_progress, consecutive_no_progress
            )

        pawn_position_repeat = _record_pawn_position(
            pawn_history,
            pawn_position_repeats,
            player=player,
            position=next_state.pawns[player],
            action_category=selected.get("category"),
        )

        trace_entry = {
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
            "pawns_before": [list(pos) for pos in state.pawns],
            "pawns_after": [list(pos) for pos in next_state.pawns],
            "shortest_distance_before": previous_distances,
            "shortest_distance_after": next_distances,
            "shortest_distance_delta": distance_deltas,
            "made_progress": made_progress,
            "pawn_position_repeat": pawn_position_repeat,
            "consecutive_no_progress_plies": consecutive_no_progress,
        }
        trace.append(trace_entry)
        for size, counts in action_ngrams.items():
            if len(trace) >= size:
                key = action_pattern_key(trace[-size:])
                counts[key] += 1
        state = next_state
        previous_distances = next_distances

    winner_agent = None if state.winner is None else player_agents[state.winner]
    repeated_states = sum(count - 1 for count in state_seen.values() if count > 1)
    repeated_state_positions = sum(count > 1 for count in state_seen.values())
    repeated_action_patterns = {
        str(size): sum(count - 1 for count in counts.values() if count > 1)
        for size, counts in action_ngrams.items()
    }
    draw_reason = (
        "not_a_draw" if state.winner is not None
        else "max_plies" if state.ply >= max_plies
        else "terminal_without_winner"
    )
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
        "termination": {
            "draw": state.winner is None,
            "draw_reason": draw_reason,
            "max_plies_reached": state.ply >= max_plies,
        },
        "diagnostics": {
            "repeated_state_occurrences": repeated_states,
            "unique_repeated_states": repeated_state_positions,
            # Counts repeated destinations on pawn-move turns only; wall turns
            # are intentionally excluded from both the history and numerator.
            "pawn_move_position_repeats": {
                str(player): pawn_position_repeats[player] for player in (0, 1)
            },
            "pawn_position_repeats": {
                str(player): pawn_position_repeats[player] for player in (0, 1)
            },
            "repeated_action_patterns": repeated_action_patterns,
            "no_progress_plies": no_progress_plies,
            "progress_plies": progress_plies,
            "no_progress_ratio": (
                no_progress_plies / state.ply if state.ply else 0.0
            ),
            "max_consecutive_no_progress_plies": max_consecutive_no_progress,
            "final_shortest_distances": previous_distances,
        },
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
    parser.add_argument(
        "--progress-jsonl", type=Path, default=None,
        help="Append and fsync one JSON record after each game; defaults to output with .jsonl suffix.",
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
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    progress_path = args.progress_jsonl or args.output.with_suffix(".jsonl")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_file = progress_path.open("a", encoding="utf-8")

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
                result["run_id"] = run_id
                games.append(result)
                progress_file.write(json.dumps(result, separators=(",", ":")) + "\n")
                progress_file.flush()
                os.fsync(progress_file.fileno())
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

    progress_file.close()
    report = {
        "run_id": run_id,
        "progress_jsonl": str(progress_path),
        "checkpoint": str(args.checkpoint),
        "protocol": (
            "MCTS-vs-MCTS; one checkpoint; paired seeds; swapped seats; "
            "randomized seeded max-visit ties; no root noise; per-game durable JSONL"
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
        "draw_summary": {
            "max_plies_draws": sum(g["termination"]["draw_reason"] == "max_plies" for g in games),
            "terminal_draws": sum(g["termination"]["draw_reason"] == "terminal_without_winner" for g in games),
            "wins": sum(g["winner"] is not None for g in games),
        },
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
