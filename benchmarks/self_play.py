#!/usr/bin/env python3
"""Run reproducible Quoridor self-play matches and collect decision telemetry.

Example:
    python -m benchmarks.self_play --games 100 --max-depth 3 --time-limit 0.2
    python -m benchmarks.self_play --games 10 --opening-plies 0 --output-dir data/self_play/smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quoridor.advanced_ai import AIConfig, QuoridorAI
from quoridor.engine import (
    GameState,
    Wall,
    apply_move,
    apply_wall,
    legal_pawn_moves,
    shortest_distance,
)

SUMMARY_FIELDS = [
    "game_id",
    "seed",
    "starting_player",
    "opening_plies",
    "winner",
    "result",
    "total_plies",
    "ai_p0_walls",
    "ai_p1_walls",
    "avg_p0_thinking_ms",
    "avg_p1_thinking_ms",
    "max_p0_thinking_ms",
    "max_p1_thinking_ms",
    "avg_opponent_delay",
    "zero_delay_wall_rate",
    "avg_own_detour",
    "ai_version",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _apply_action(state: GameState, result_action: Any) -> GameState:
    if result_action.kind == "move":
        if result_action.destination is None:
            raise ValueError("AI returned a move without a destination")
        return apply_move(state, result_action.destination)
    if result_action.kind == "wall":
        if result_action.wall is None:
            raise ValueError("AI returned a wall action without a wall")
        return apply_wall(state, result_action.wall)
    raise ValueError(f"Unsupported action type: {result_action.kind!r}")


def measure_wall_effect(
    before: GameState,
    after: GameState,
    player: int,
) -> dict[str, int | None]:
    """Measure shortest-path changes caused by a wall, from the acting player's view."""
    opponent = 1 - player
    own_before = shortest_distance(before, player)
    own_after = shortest_distance(after, player)
    opponent_before = shortest_distance(before, opponent)
    opponent_after = shortest_distance(after, opponent)
    return {
        "own_distance_before": own_before,
        "own_distance_after": own_after,
        "own_detour": (
            own_after - own_before
            if own_before is not None and own_after is not None
            else None
        ),
        "opponent_distance_before": opponent_before,
        "opponent_distance_after": opponent_after,
        "opponent_delay": (
            opponent_after - opponent_before
            if opponent_before is not None and opponent_after is not None
            else None
        ),
    }


def _distance_snapshot(state: GameState) -> dict[str, int | None]:
    return {
        "distance_p0": shortest_distance(state, 0),
        "distance_p1": shortest_distance(state, 1),
    }


def _random_opening(
    state: GameState,
    plies: int,
    rng: random.Random,
    game_id: str,
    game_log: Path,
) -> GameState:
    """Apply reproducible random legal pawn moves to diversify deterministic self-play."""
    for _ in range(plies):
        if state.winner is not None:
            break
        player = state.current_player
        moves = legal_pawn_moves(state, player)
        if not moves:
            break
        before = state
        destination = rng.choice(moves)
        state = apply_move(state, destination)
        _write_jsonl(
            game_log,
            {
                "event": "opening_action",
                "game_id": game_id,
                "turn": before.ply + 1,
                "player": f"P{player}",
                "action_type": "move",
                "action": {"destination": list(destination)},
                "state_before": before.to_dict(),
                "state_after": state.to_dict(),
                "distances_before": _distance_snapshot(before),
                "distances_after": _distance_snapshot(state),
                "note": "Random legal pawn move used only to diversify the starting position.",
            },
        )
    return state


def _mean(values: list[float | int]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def play_game(
    *,
    game_id: str,
    seed: int,
    starting_player: int,
    opening_plies: int,
    max_plies: int,
    config: AIConfig,
    game_log: Path,
    wall_log: Path,
    ai_version: str,
) -> dict[str, Any]:
    rng = random.Random(seed)
    state = replace(GameState.initial(), current_player=starting_player)
    ais = {0: QuoridorAI(config), 1: QuoridorAI(config)}
    thinking: dict[int, list[float]] = {0: [], 1: []}
    wall_counts = {0: 0, 1: 0}
    wall_delays: list[int] = []
    wall_detours: list[int] = []

    state = _random_opening(state, opening_plies, rng, game_id, game_log)

    while state.winner is None and state.ply < max_plies:
        player = state.current_player
        before = state
        distances_before = _distance_snapshot(before)
        search = ais[player].choose_action(before)
        action = search.action
        after = _apply_action(before, action)
        distances_after = _distance_snapshot(after)
        thinking[player].append(float(search.elapsed_ms))

        action_data = action.to_dict()
        event = {
            "event": "action",
            "game_id": game_id,
            "turn": before.ply + 1,
            "player": f"P{player}",
            "action_type": action.kind,
            "action": action_data,
            "state_before": before.to_dict(),
            "state_after": after.to_dict(),
            "distances_before": distances_before,
            "distances_after": distances_after,
            "ai": {
                "version": ai_version,
                "score": search.score,
                "depth": search.depth,
                "elapsed_ms": search.elapsed_ms,
                "timed_out": search.timed_out,
                "nodes": search.nodes,
                "stats": dict(ais[player].stats),
                "config": asdict(config),
            },
        }
        _write_jsonl(game_log, event)

        if action.kind == "wall":
            wall_counts[player] += 1
            metrics = measure_wall_effect(before, after, player)
            if metrics["opponent_delay"] is not None:
                wall_delays.append(int(metrics["opponent_delay"]))
            if metrics["own_detour"] is not None:
                wall_detours.append(int(metrics["own_detour"]))
            _write_jsonl(
                wall_log,
                {
                    "event": "wall_analysis",
                    "game_id": game_id,
                    "turn": before.ply + 1,
                    "player": f"P{player}",
                    "wall": action_data,
                    **metrics,
                    "walls_remaining_before": list(before.walls_remaining),
                    "walls_remaining_after": list(after.walls_remaining),
                    "ai_score": search.score,
                    "ai_depth": search.depth,
                    "game_result": "pending",
                    "interpretation": (
                        "directly_delayed" if metrics["opponent_delay"] and metrics["opponent_delay"] > 0
                        else "no_shortest_path_delay" if metrics["opponent_delay"] == 0
                        else "distance_unavailable"
                    ),
                },
            )

        state = after

    winner = state.winner
    result = "win" if winner is not None else "draw_max_plies"
    all_thinking = thinking[0] + thinking[1]
    delay_rate = (
        sum(1 for delay in wall_delays if delay == 0) / len(wall_delays)
        if wall_delays else None
    )

    return {
        "game_id": game_id,
        "seed": seed,
        "starting_player": starting_player,
        "opening_plies": opening_plies,
        "winner": f"P{winner}" if winner is not None else "",
        "result": result,
        "total_plies": state.ply,
        "ai_p0_walls": wall_counts[0],
        "ai_p1_walls": wall_counts[1],
        "avg_p0_thinking_ms": _mean(thinking[0]),
        "avg_p1_thinking_ms": _mean(thinking[1]),
        "max_p0_thinking_ms": round(max(thinking[0]), 3) if thinking[0] else None,
        "max_p1_thinking_ms": round(max(thinking[1]), 3) if thinking[1] else None,
        "avg_opponent_delay": _mean(wall_delays),
        "zero_delay_wall_rate": round(delay_rate, 4) if delay_rate is not None else None,
        "avg_own_detour": _mean(wall_detours),
        "ai_version": ai_version,
        "total_thinking_ms": round(sum(all_thinking), 3),
        "final_state": state.to_dict(),
    }


def run_benchmark(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else Path("data/self_play") / timestamp
    games_dir = output_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    wall_log = output_dir / "wall_analysis.jsonl"
    summary_path = output_dir / "game_summary.csv"
    metadata_path = output_dir / "run_metadata.json"

    # Do not append results to an old run by accident.
    if wall_log.exists() or summary_path.exists():
        raise FileExistsError(
            f"Output already contains benchmark results: {output_dir}. "
            "Choose a new --output-dir to preserve the previous run."
        )

    config = AIConfig(time_limit=args.time_limit, max_depth=args.max_depth)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "games_requested": args.games,
        "max_plies": args.max_plies,
        "opening_plies": args.opening_plies,
        "base_seed": args.seed,
        "ai_version": args.ai_version,
        "config": asdict(config),
        "starting_player_policy": "alternates P0/P1 each game",
        "opening_policy": "seeded random legal pawn moves; not AI decisions",
        "result_note": "Self-play measures consistency; compare versions and seat assignments for strength estimates.",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for index in range(args.games):
            game_id = f"game_{index + 1:04d}"
            seed = args.seed + index
            starting_player = index % 2
            game_log = games_dir / f"{game_id}.jsonl"
            print(
                f"[{index + 1}/{args.games}] {game_id}: start=P{starting_player}, seed={seed}",
                flush=True,
            )
            summary = play_game(
                game_id=game_id,
                seed=seed,
                starting_player=starting_player,
                opening_plies=args.opening_plies,
                max_plies=args.max_plies,
                config=config,
                game_log=game_log,
                wall_log=wall_log,
                ai_version=args.ai_version,
            )
            writer.writerow({key: summary.get(key) for key in SUMMARY_FIELDS})
            handle.flush()
            print(
                f"  result={summary['result']} winner={summary['winner'] or '-'} "
                f"plies={summary['total_plies']} walls={summary['ai_p0_walls'] + summary['ai_p1_walls']} "
                f"avg_delay={summary['avg_opponent_delay']}",
                flush=True,
            )

    print(f"Benchmark data written to: {output_dir}", flush=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible Quoridor AI self-play and record telemetry."
    )
    parser.add_argument("--games", type=int, default=100, help="Number of games (default: 100).")
    parser.add_argument("--max-depth", type=int, default=3, help="AI search depth (default: 3).")
    parser.add_argument("--time-limit", type=float, default=0.2, help="Seconds per AI move (default: 0.2).")
    parser.add_argument("--max-plies", type=int, default=200, help="Maximum plies per game (default: 200).")
    parser.add_argument(
        "--opening-plies",
        type=int,
        default=4,
        help="Number of seeded random legal pawn moves before AI play (default: 4; use 0 for standard start).",
    )
    parser.add_argument("--seed", type=int, default=20261029, help="Base random seed.")
    parser.add_argument("--ai-version", default="working-tree", help="Git commit/tag label for the AI under test.")
    parser.add_argument("--output-dir", help="Output directory; default is a timestamped directory under data/self_play.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be at least 1")
    if args.max_depth < 1:
        parser.error("--max-depth must be at least 1")
    if args.time_limit <= 0:
        parser.error("--time-limit must be greater than 0")
    if args.max_plies < 1:
        parser.error("--max-plies must be at least 1")
    if args.opening_plies < 0:
        parser.error("--opening-plies cannot be negative")
    if args.opening_plies >= args.max_plies:
        parser.error("--opening-plies must be smaller than --max-plies")
    try:
        run_benchmark(args)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted. Completed game logs and summary rows are preserved.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
