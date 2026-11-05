#!/usr/bin/env python3
"""Sweep small pawn-progress bonuses while keeping repetition avoidance enabled."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3026)
    parser.add_argument("--heuristic-time", type=float, default=0.2)
    parser.add_argument("--heuristic-depth", type=int, default=3)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--category-exploration", type=float, default=0.35)
    parser.add_argument("--selection-temperature", type=float, default=0.2)
    parser.add_argument("--repetition-penalty", type=float, default=0.75)
    parser.add_argument("--repetition-window", type=int, default=4)
    parser.add_argument("--opponent", choices=("heuristic", "random"), default="heuristic")
    parser.add_argument("--progress-values", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05])
    parser.add_argument("--output-dir", type=Path, default=Path("data/progress_sweep"))
    args = parser.parse_args()

    if args.games < 2 or args.simulations < 1 or args.max_plies < 1:
        parser.error("--games must be >= 2; simulations and max-plies must be positive")
    if args.repetition_penalty < 0 or args.repetition_window < 1:
        parser.error("repetition penalty must be non-negative and window must be positive")
    if not args.progress_values or any(value < 0 for value in args.progress_values):
        parser.error("progress values must be a non-empty list of non-negative numbers")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = [
        "--games", str(args.games),
        "--simulations", str(args.simulations),
        "--seed", str(args.seed),
        "--heuristic-time", str(args.heuristic_time),
        "--heuristic-depth", str(args.heuristic_depth),
        "--max-plies", str(args.max_plies),
        "--device", args.device,
        "--category-exploration", str(args.category_exploration),
        "--selection-temperature", str(args.selection_temperature),
        "--repetition-penalty", str(args.repetition_penalty),
        "--repetition-window", str(args.repetition_window),
        "--opponent", args.opponent,
    ]

    for progress in args.progress_values:
        label = f"{progress:.3f}".replace(".", "_")
        output = args.output_dir / f"eval_progress_{label}.json"
        command = [
            sys.executable, "-m", "training.evaluate", str(args.checkpoint),
            *common,
            "--pawn-progress-bonus", str(progress),
            "--output", str(output),
        ]
        print(
            f"\n=== repetition={args.repetition_penalty}, progress={progress:.3f} ===",
            flush=True,
        )
        subprocess.run(command, check=True)

    print(f"\nAll sweep reports saved under: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
