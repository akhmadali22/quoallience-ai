#!/usr/bin/env python3
"""Generate decisive Quoridor games using the minimax heuristic AI (QuoridorAI).

These games serve as bootstrap training data for EXP-001: value-head fine-tuning.
They provide real +1/-1 value targets from decisive outcomes, breaking the degenerate
self-play draw loop that prevents the neural network from learning.

Output JSONL format (one record per position, suitable for train_value_only.py):
  {
    "state": <10x9x9 encoded list>,
    "value": <float, +1 if player-to-move won, -1 if lost, discounted by ply-to-end>,
    "game_id": <int>,
    "ply": <int>,
    "player": <int, 0 or 1>,
    "winner": <int, 0 or 1>,
    "total_plies": <int>
  }
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from quoridor.advanced_ai import AIConfig, QuoridorAI
from quoridor.engine import GameState, apply_move, apply_wall, shortest_distance
from quoridor.neural_ai.encoder import encode_action, encode_state, legal_action_indices


def play_game(
    game_id: int,
    *,
    seed: int,
    config: AIConfig,
    max_plies: int,
) -> dict | None:
    """Play a single heuristic self-play game. Returns None if game drawn at max_plies."""
    random.seed(seed)

    # Fresh AI instances per game to avoid cross-game history leakage.
    ai = [QuoridorAI(config), QuoridorAI(config)]

    state = GameState.initial()
    trajectory: list[tuple[GameState, int, int]] = []  # (state, player_to_move, chosen_action_index)

    while state.winner is None and state.ply < max_plies:
        player = state.current_player
        result = ai[player].choose_action(state)
        action = result.action

        payload = action.destination if action.kind == "move" else action.wall
        action_idx = encode_action((action.kind, payload))
        trajectory.append((state, player, action_idx))

        if action.kind == "move":
            state = apply_move(state, action.destination)
        else:
            state = apply_wall(state, action.wall)

    if state.winner is None:
        return None  # Draw — discard this game

    winner = state.winner
    total_plies = state.ply

    return {
        "game_id": game_id,
        "winner": winner,
        "total_plies": total_plies,
        "trajectory": trajectory,
    }


def build_samples(
    game: dict,
    *,
    discount_per_ply: float,
) -> list[dict]:
    """Convert a finished game into per-position training samples."""
    winner = game["winner"]
    total_plies = game["total_plies"]
    trajectory = game["trajectory"]
    game_id = game["game_id"]
    samples = []

    for ply_index, (state, player, action_idx) in enumerate(trajectory):
        plies_to_end = total_plies - state.ply
        discount = discount_per_ply ** plies_to_end

        # Value is from the perspective of the player to move.
        # +1 discounted if they won, -1 discounted if they lost.
        if player == winner:
            value = discount * 1.0
        else:
            value = discount * -1.0

        legal_masks = legal_action_indices(state)

        samples.append({
            "state": encode_state(state),
            "policy": {str(action_idx): 1.0},
            "policy_mask": legal_masks,
            "value": round(value, 6),
            "game_id": game_id,
            "ply": state.ply,
            "player": player,
            "winner": winner,
            "total_plies": total_plies,
        })

    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", type=int, default=300,
        help="Number of decisive games to generate (draws discarded).",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=600,
        help="Maximum game attempts (stops early if reached before --games decisive games).",
    )
    parser.add_argument(
        "--time-limit", type=float, default=0.3,
        help="Heuristic AI time limit per move in seconds.",
    )
    parser.add_argument(
        "--max-depth", type=int, default=3,
        help="Heuristic AI minimax depth.",
    )
    parser.add_argument(
        "--max-plies", type=int, default=300,
        help="Maximum plies per game before declaring a draw (discarded).",
    )
    parser.add_argument(
        "--discount", type=float, default=0.99,
        help="Per-ply discount factor for value targets (0.99 means wins close to game-end are valued higher).",
    )
    parser.add_argument(
        "--seed", type=int, default=20261113,
        help="Base random seed. Each game uses seed + game_attempt_index.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/experiments/exp_001/heuristic_games.jsonl"),
        help="Output JSONL path for training samples.",
    )
    parser.add_argument(
        "--summary", type=Path, default=None,
        help="Optional JSON path for run summary statistics.",
    )
    args = parser.parse_args()

    if args.games < 1 or args.max_attempts < args.games:
        parser.error("--games must be positive and --max-attempts >= --games")
    if args.time_limit <= 0:
        parser.error("--time-limit must be positive")
    if args.max_depth < 1:
        parser.error("--max-depth must be positive")
    if not 0.0 < args.discount <= 1.0:
        parser.error("--discount must be in (0, 1]")
    if args.max_plies < 10:
        parser.error("--max-plies must be at least 10")

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        parser.error(f"Output file already exists: {output}. Use a different path.")

    summary_path = args.summary
    if summary_path is None:
        summary_path = output.parent / "heuristic_games_summary.json"

    config = AIConfig(
        time_limit=args.time_limit,
        max_depth=args.max_depth,
    )

    decisive_games = 0
    drawn_games = 0
    total_samples = 0
    game_lengths: list[int] = []
    winner_counts = {0: 0, 1: 0}
    t_start = time.monotonic()

    with output.open("w", encoding="utf-8") as f:
        for attempt in range(args.max_attempts):
            if decisive_games >= args.games:
                break

            seed = args.seed + attempt
            t_game = time.monotonic()

            game = play_game(
                attempt + 1,
                seed=seed,
                config=config,
                max_plies=args.max_plies,
            )

            elapsed_game = time.monotonic() - t_game

            if game is None:
                drawn_games += 1
                print(
                    f"[{attempt + 1}] DRAW (attempt={attempt + 1} "
                    f"decisive={decisive_games}/{args.games} "
                    f"elapsed={elapsed_game:.1f}s)",
                    flush=True,
                )
                continue

            decisive_games += 1
            samples = build_samples(game, discount_per_ply=args.discount)
            game_lengths.append(game["total_plies"])
            winner_counts[game["winner"]] += 1
            total_samples += len(samples)

            for sample in samples:
                f.write(json.dumps(sample, separators=(",", ":")) + "\n")
            f.flush()

            print(
                f"[{attempt + 1}] DECISIVE game={decisive_games}/{args.games} "
                f"winner={game['winner']} plies={game['total_plies']} "
                f"samples={len(samples)} elapsed={elapsed_game:.1f}s",
                flush=True,
            )

    elapsed_total = time.monotonic() - t_start
    avg_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0

    summary = {
        "config": {
            "games_requested": args.games,
            "max_attempts": args.max_attempts,
            "time_limit": args.time_limit,
            "max_depth": args.max_depth,
            "max_plies": args.max_plies,
            "discount": args.discount,
            "seed": args.seed,
        },
        "results": {
            "decisive_games": decisive_games,
            "drawn_games": drawn_games,
            "total_attempts": decisive_games + drawn_games,
            "draw_rate": drawn_games / (decisive_games + drawn_games) if (decisive_games + drawn_games) else 0,
            "total_samples": total_samples,
            "avg_game_length": round(avg_length, 1),
            "min_game_length": min(game_lengths) if game_lengths else None,
            "max_game_length": max(game_lengths) if game_lengths else None,
            "winner_counts": winner_counts,
            "elapsed_seconds": round(elapsed_total, 1),
        },
        "output": str(output),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nDone. Decisive games: {decisive_games}/{decisive_games + drawn_games}")
    print(f"Total training samples: {total_samples}")
    print(f"Avg game length: {avg_length:.1f} plies")
    print(f"Winner distribution: P0={winner_counts[0]}, P1={winner_counts[1]}")
    print(f"Elapsed: {elapsed_total:.1f}s")
    print(f"Samples written to: {output}")
    print(f"Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
