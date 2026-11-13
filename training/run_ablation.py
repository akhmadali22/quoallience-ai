#!/usr/bin/env python3
"""Run the four-way repetition/progress ablation with identical settings."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--heuristic-time", type=float, default=0.2)
    parser.add_argument("--heuristic-depth", type=int, default=3)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--category-exploration", type=float, default=0.35)
    parser.add_argument("--selection-temperature", type=float, default=0.2)
    parser.add_argument("--repetition-penalty", type=float, default=0.75)
    parser.add_argument("--pawn-progress-bonus", type=float, default=0.10)
    parser.add_argument("--repetition-window", type=int, default=4)
    parser.add_argument("--opponent", choices=("heuristic", "random"), default="heuristic")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ablation"))
    args = parser.parse_args()

    if args.games < 2 or args.simulations < 1 or args.max_plies < 1:
        parser.error("--games must be >= 2; simulations and max-plies must be positive")

    experiments = (
        ("baseline", 0.0, 0.0),
        ("repetition_only", args.repetition_penalty, 0.0),
        ("progress_only", 0.0, args.pawn_progress_bonus),
        ("combined", args.repetition_penalty, args.pawn_progress_bonus),
    )
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
        "--repetition-window", str(args.repetition_window),
        "--opponent", args.opponent,
    ]

    for name, repetition_penalty, progress_bonus in experiments:
        output = args.output_dir / f"eval_{name}.json"
        command = [
            sys.executable, "-m", "training.evaluate", str(args.checkpoint),
            *common,
            "--repetition-penalty", str(repetition_penalty),
            "--pawn-progress-bonus", str(progress_bonus),
            "--output", str(output),
        ]
        print(
            f"\\n=== {name}: repetition={repetition_penalty}, "
            f"progress={progress_bonus} ===",
            flush=True,
        )
        subprocess.run(command, check=True)

    print(f"\\nAll four reports saved under: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
