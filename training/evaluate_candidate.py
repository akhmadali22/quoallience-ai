#!/usr/bin/env python3
"""Paired-seat candidate-vs-incumbent evaluation for Quoridor MCTS models."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from quoridor.engine import GameState, state_repetition_key
from quoridor.neural_ai.encoder import apply_action_index
from quoridor.neural_ai.mcts import MOVE_ACTIONS, MCTS
from quoridor.neural_ai.policy import load_model


def exact_one_sided_sign_test(wins: int, losses: int) -> float:
    """P(X >= wins) under p=0.5, ignoring tied paired matches."""
    decisive = wins + losses
    if decisive == 0:
        return 1.0
    return sum(math.comb(decisive, k) for k in range(wins, decisive + 1)) / (2 ** decisive)



def compact_root_diagnostics(stats: dict) -> dict:
    """Expose per-pawn PUCT terms without bloating evaluation traces with all walls."""
    actions = stats.get("root_action_stats", {})
    pawn_actions = {
        action: details for action, details in actions.items()
        if details.get("category") == "pawn"
    }
    wall_actions = [
        (action, details) for action, details in actions.items()
        if details.get("category") == "wall"
    ]
    top_walls = sorted(
        wall_actions, key=lambda item: item[1].get("prior", 0.0), reverse=True
    )[:5]
    keys = (
        "root_value", "root_prior_entropy", "root_active_prior_entropy",
        "root_visit_entropy", "root_pawn_visit_ratio", "root_wall_visit_ratio",
        "root_active_pawn_prior_mass", "root_active_wall_prior_mass",
        "root_pawn_visits", "root_wall_visits", "root_legal_actions",
        "root_active_actions", "root_repetition_history_length",
        "root_repetition_window", "root_current_state_repetition_count",
        "root_pawn_actions_with_repeated_next_state",
        "root_pawn_actions_with_progress", "root_pawn_actions_with_regression",
    )
    compact = {key: stats.get(key) for key in keys}
    compact["pawn_actions"] = pawn_actions
    compact["top_wall_actions_by_prior"] = {
        action: details for action, details in top_walls
    }
    return compact


def play_game(
    candidate_model,
    incumbent_model,
    *,
    seed: int,
    candidate_player: int,
    simulations: int,
    max_plies: int,
    device: str,
    c_puct: float,
    category_exploration: float,
    repetition_penalty: float,
    pawn_progress_bonus: float,
    repetition_window: int,
    progressive_widening: bool,
    initial_wall_candidates: int,
) -> dict:
    models = {
        "candidate": candidate_model,
        "incumbent": incumbent_model,
    }
    player_agents = {
        candidate_player: "candidate",
        1 - candidate_player: "incumbent",
    }
    searches = {
        label: MCTS(
            model,
            simulations=simulations,
            c_puct=c_puct,
            device=device,
            randomize_selection_ties=True,
            progressive_widening=progressive_widening,
            initial_wall_candidates=initial_wall_candidates,
            category_exploration=category_exploration,
            repetition_penalty=repetition_penalty,
            pawn_progress_bonus=pawn_progress_bonus,
            repetition_window=repetition_window,
        )
        for label, model in models.items()
    }
    rngs = {
        "candidate": random.Random(seed ^ 0xCADA71),
        "incumbent": random.Random(seed ^ 0x1AC0B1),
    }
    pawn_history = {0: [], 1: []}
    state = GameState.initial()
    state_history = [state_repetition_key(state)]
    pawn_history[0].append(state.pawns[0])
    pawn_history[1].append(state.pawns[1])
    trace = []

    while state.winner is None and state.ply < max_plies:
        player = state.current_player
        label = player_agents[player]
        action, _ = searches[label].choose_action(
            state,
            temperature=0.0,
            rng=rngs[label],
            recent_pawn_positions=pawn_history[player],
            recent_state_keys=state_history,
        )
        search_stats = searches[label].last_search_stats
        next_state = apply_action_index(state, action)
        if next_state.pawns[player] != state.pawns[player]:
            pawn_history[player].append(next_state.pawns[player])
        trace.append({
            "ply": state.ply,
            "player": player,
            "agent": label,
            "action": action,
            "category": "pawn" if action < MOVE_ACTIONS else "wall",
            "root_diagnostics": compact_root_diagnostics(search_stats),
        })
        state = next_state
        state_history.append(state_repetition_key(state))

    winner_agent = None if state.winner is None else player_agents[state.winner]
    result = (
        "draw" if winner_agent is None
        else "win" if winner_agent == "candidate"
        else "loss"
    )
    return {
        "seed": seed,
        "candidate_player": candidate_player,
        "winner": state.winner,
        "winner_agent": winner_agent,
        "result_for_candidate": result,
        "plies": state.ply,
        "draw_reason": "max_plies" if winner_agent is None else None,
        "action_trace": trace,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("incumbent", type=Path)
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--category-exploration", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=0.0)
    parser.add_argument("--pawn-progress-bonus", type=float, default=0.0)
    parser.add_argument("--repetition-window", type=int, default=4)
    parser.add_argument("--progressive-widening", action="store_true",
                        help="Enable optional wall-candidate progressive widening.")
    parser.add_argument("--initial-wall-candidates", type=int, default=8)
    parser.add_argument("--min-decisive-pairs", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.pairs < 1 or args.simulations < 1 or args.max_plies < 1:
        parser.error("--pairs, --simulations, and --max-plies must be positive")
    if args.min_decisive_pairs < 1 or not 0 < args.alpha < 1:
        parser.error("--min-decisive-pairs must be positive and --alpha must be in (0, 1)")
    if min(args.category_exploration, args.repetition_penalty, args.pawn_progress_bonus) < 0:
        parser.error("exploration and preference values must be non-negative")
    if args.repetition_window < 1 or args.initial_wall_candidates < 0 or args.c_puct < 0:
        parser.error("invalid repetition window, initial wall candidates, or c-puct")

    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    candidate_model = load_model(args.candidate, device=args.device)
    incumbent_model = load_model(args.incumbent, device=args.device)
    candidate_model.eval()
    incumbent_model.eval()

    games = []
    paired_results = []
    for pair_index in range(args.pairs):
        seed = args.seed + pair_index
        pair_games = []
        for candidate_player in (0, 1):
            game_number = pair_index * 2 + candidate_player + 1
            print(
                f"[{game_number}/{args.pairs * 2}] seed={seed} "
                f"candidate=P{candidate_player}",
                flush=True,
            )
            game = play_game(
                candidate_model,
                incumbent_model,
                seed=seed,
                candidate_player=candidate_player,
                simulations=args.simulations,
                max_plies=args.max_plies,
                device=args.device,
                c_puct=args.c_puct,
                category_exploration=args.category_exploration,
                repetition_penalty=args.repetition_penalty,
                pawn_progress_bonus=args.pawn_progress_bonus,
                repetition_window=args.repetition_window,
                progressive_widening=args.progressive_widening,
                initial_wall_candidates=args.initial_wall_candidates,
            )
            games.append(game)
            pair_games.append(game)
            print(
                f"  candidate={game['result_for_candidate']} "
                f"plies={game['plies']}",
                flush=True,
            )

        points = sum(
            1.0 if game["result_for_candidate"] == "win"
            else 0.5 if game["result_for_candidate"] == "draw"
            else 0.0
            for game in pair_games
        )
        pair_result = "win" if points > 1 else "loss" if points < 1 else "draw"
        paired_results.append({
            "pair": pair_index + 1,
            "seed": seed,
            "candidate_points": points,
            "result_for_candidate": pair_result,
        })

    wins = sum(game["result_for_candidate"] == "win" for game in games)
    losses = sum(game["result_for_candidate"] == "loss" for game in games)
    draws = sum(game["result_for_candidate"] == "draw" for game in games)
    points = wins + 0.5 * draws
    paired_wins = sum(pair["result_for_candidate"] == "win" for pair in paired_results)
    paired_losses = sum(pair["result_for_candidate"] == "loss" for pair in paired_results)
    paired_draws = args.pairs - paired_wins - paired_losses
    decisive_pairs = paired_wins + paired_losses
    p_value = exact_one_sided_sign_test(paired_wins, paired_losses)
    score_rate = points / len(games)
    promotion_eligible = (
        decisive_pairs >= args.min_decisive_pairs
        and paired_wins > paired_losses
        and score_rate > 0.5
        and p_value < args.alpha
    )

    report = {
        "candidate": str(args.candidate),
        "incumbent": str(args.incumbent),
        "config": {
            "pairs": args.pairs,
            "games": len(games),
            "simulations": args.simulations,
            "max_plies": args.max_plies,
            "seed": args.seed,
            "device": args.device,
            "c_puct": args.c_puct,
            "category_exploration": args.category_exploration,
            "repetition_penalty": args.repetition_penalty,
            "pawn_progress_bonus": args.pawn_progress_bonus,
            "repetition_window": args.repetition_window,
            "progressive_widening": args.progressive_widening,
            "initial_wall_candidates": args.initial_wall_candidates,
            "min_decisive_pairs": args.min_decisive_pairs,
            "alpha": args.alpha,
            "selection_temperature": 0.0,
            "root_dirichlet_noise": False,
            "paired_seats": True,
        },
        "candidate_result": {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "score_rate": score_rate,
        },
        "paired_result": {
            "wins": paired_wins,
            "losses": paired_losses,
            "draws": paired_draws,
            "decisive_pairs": decisive_pairs,
            "one_sided_sign_test_p_value": p_value,
        },
        "draw_summary": {
            "max_plies_draws": sum(game["draw_reason"] == "max_plies" for game in games),
        },
        "promotion": {
            "eligible": promotion_eligible,
            "reason": (
                "candidate passed paired-seat promotion gate"
                if promotion_eligible else
                "not promoted: requires enough decisive pairs, more paired wins than losses, "
                "score rate above 0.5, and one-sided sign-test p-value below alpha"
            ),
        },
        "paired_results": paired_results,
        "games": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        f"candidate W-L-D={wins}-{losses}-{draws} "
        f"score={score_rate:.3f}; paired W-L-D={paired_wins}-{paired_losses}-{paired_draws}; "
        f"p={p_value:.5f}; promote={promotion_eligible}"
    )
    print(f"Evaluation report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
