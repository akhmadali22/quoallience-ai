"""Analyze CSV/JSONL output produced by benchmarks.self_play."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def analyze_run(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "game_summary.csv"
    wall_path = run_dir / "wall_analysis.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing game summary: {summary_path}")

    import csv

    with summary_path.open(newline="", encoding="utf-8") as handle:
        games = list(csv.DictReader(handle))
    if not games:
        raise ValueError(f"No completed game rows found in {summary_path}")

    p0_wins = sum(row.get("winner") == "P0" for row in games)
    p1_wins = sum(row.get("winner") == "P1" for row in games)
    draws = sum(row.get("result") != "win" for row in games)
    p0_delays: list[float] = []
    own_detours: list[float] = []
    zero_delay_walls = 0
    wall_count = 0
    wall_delays_by_game: dict[str, list[float]] = {}

    if wall_path.exists():
        with wall_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                wall = json.loads(line)
                delay = _number(wall.get("opponent_delay"))
                detour = _number(wall.get("own_detour"))
                wall_count += 1
                if delay is not None:
                    p0_delays.append(delay)
                    wall_delays_by_game.setdefault(wall["game_id"], []).append(delay)
                    if delay == 0:
                        zero_delay_walls += 1
                if detour is not None:
                    own_detours.append(detour)

    avg_p0_thinking = [
        value for row in games
        if (value := _number(row.get("avg_p0_thinking_ms"))) is not None
    ]
    avg_p1_thinking = [
        value for row in games
        if (value := _number(row.get("avg_p1_thinking_ms"))) is not None
    ]
    total_games = len(games)

    return {
        "run_dir": str(run_dir),
        "games": total_games,
        "p0_wins": p0_wins,
        "p1_wins": p1_wins,
        "draws_or_max_plies": draws,
        "p0_win_rate": round(p0_wins / total_games, 4),
        "p1_win_rate": round(p1_wins / total_games, 4),
        "draw_rate": round(draws / total_games, 4),
        "total_walls_placed": wall_count,
        "walls_with_zero_opponent_delay": zero_delay_walls,
        "zero_delay_wall_rate": round(zero_delay_walls / wall_count, 4) if wall_count else None,
        "avg_opponent_delay": round(statistics.fmean(p0_delays), 3) if p0_delays else None,
        "avg_own_detour": round(statistics.fmean(own_detours), 3) if own_detours else None,
        "avg_p0_thinking_ms_per_ai_turn": round(statistics.fmean(avg_p0_thinking), 3) if avg_p0_thinking else None,
        "avg_p1_thinking_ms_per_ai_turn": round(statistics.fmean(avg_p1_thinking), 3) if avg_p1_thinking else None,
        "games_with_wall_data": len(wall_delays_by_game),
        "interpretation_note": (
            "P0/P1 win rates measure seat/first-turn asymmetry in identical self-play, "
            "not absolute AI strength. Compare against another AI version or opponent."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Quoridor self-play telemetry.")
    parser.add_argument("run_dir", type=Path, help="Directory containing game_summary.csv and wall_analysis.jsonl.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()
    result = analyze_run(args.run_dir)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Run: {result['run_dir']}")
        print(f"Games: {result['games']}")
        print(
            f"Wins: P0={result['p0_wins']} ({result['p0_win_rate']:.1%}), "
            f"P1={result['p1_wins']} ({result['p1_win_rate']:.1%}), "
            f"draw/max-plies={result['draws_or_max_plies']}"
        )
        print(f"Walls analyzed: {result['total_walls_placed']}")
        print(f"Zero-delay wall rate: {result['zero_delay_wall_rate']}")
        print(f"Average opponent delay: {result['avg_opponent_delay']}")
        print(f"Average own detour: {result['avg_own_detour']}")
        print(f"Average thinking time P0/P1: {result['avg_p0_thinking_ms_per_ai_turn']} / {result['avg_p1_thinking_ms_per_ai_turn']} ms")
        print(result["interpretation_note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
