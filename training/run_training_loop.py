#!/usr/bin/env python3
"""Run bounded AlphaZero-style self-play, training, evaluation, and promotion iterations."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_step(command: list[str], *, repo_root: Path) -> None:
    print("\n$", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--games", type=int, default=20, help="Self-play games per iteration.")
    parser.add_argument("--selfplay-simulations", type=int, default=32)
    parser.add_argument("--evaluation-simulations", type=int, default=64)
    parser.add_argument("--evaluation-pairs", type=int, default=50)
    parser.add_argument("--min-decisive-pairs", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--category-exploration", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=0.0)
    parser.add_argument("--pawn-progress-bonus", type=float, default=0.0)
    parser.add_argument("--repetition-window", type=int, default=4)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--initial-wall-candidates", type=int, default=8)
    parser.add_argument("--progressive-widening", action="store_true",
                        help="Enable optional wall-candidate progressive widening in self-play and evaluation.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-root", type=Path, default=Path("data/alphazero_runs"))
    args = parser.parse_args()

    if min(
        args.iterations, args.games, args.selfplay_simulations,
        args.evaluation_simulations, args.evaluation_pairs,
        args.min_decisive_pairs, args.max_plies, args.epochs, args.batch_size,
    ) < 1:
        parser.error("iteration, game, simulation, pair, ply, epoch, and batch counts must be positive")
    if args.learning_rate <= 0 or args.value_loss_weight < 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive; loss weight and weight decay must be non-negative")
    if min(args.category_exploration, args.repetition_penalty, args.pawn_progress_bonus, args.c_puct) < 0:
        parser.error("PUCT exploration and preference values must be non-negative")
    if args.repetition_window < 1 or args.initial_wall_candidates < 0 or not 0 < args.alpha < 1:
        parser.error("invalid repetition window, initial wall candidates, or alpha")

    repo_root = Path(__file__).resolve().parents[1]
    initial_checkpoint = args.initial_checkpoint.resolve()
    if not initial_checkpoint.is_file():
        parser.error(f"Initial checkpoint not found: {initial_checkpoint}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = (repo_root / args.output_root / f"run_{run_id}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    champions_dir = run_dir / "champions"
    champions_dir.mkdir()
    incumbent = initial_checkpoint
    history = []

    for iteration in range(1, args.iterations + 1):
        iteration_seed = args.seed + iteration * 100000
        iteration_dir = run_dir / f"iteration_{iteration:03d}"
        selfplay_dir = iteration_dir / "self_play"
        replay_path = selfplay_dir / "replay.jsonl"
        candidate_path = iteration_dir / "candidate.pt"
        evaluation_path = iteration_dir / "candidate_evaluation.json"
        iteration_dir.mkdir(parents=True, exist_ok=False)

        print(
            f"\n========== ITERATION {iteration}/{args.iterations} "
            f"incumbent={incumbent} seed={iteration_seed} ==========",
            flush=True,
        )
        run_step([
            sys.executable, "-m", "training.self_play",
            "--checkpoint", str(incumbent),
            "--games", str(args.games),
            "--simulations", str(args.selfplay_simulations),
            "--max-plies", str(args.max_plies),
            "--device", args.device,
            "--seed", str(iteration_seed),
            "--output-dir", str(selfplay_dir),
            "--c-puct", str(args.c_puct),
            "--initial-wall-candidates", str(args.initial_wall_candidates),
            "--category-exploration", str(args.category_exploration),
            "--repetition-penalty", str(args.repetition_penalty),
            "--pawn-progress-bonus", str(args.pawn_progress_bonus),
            "--repetition-window", str(args.repetition_window),
        ] + (["--progressive-widening"] if args.progressive_widening else []), repo_root=repo_root)

        run_step([
            sys.executable, "-m", "training.train",
            str(replay_path),
            "--checkpoint", str(incumbent),
            "--output", str(candidate_path),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--value-loss-weight", str(args.value_loss_weight),
            "--weight-decay", str(args.weight_decay),
            "--device", args.device,
            "--seed", str(iteration_seed + 1),
        ], repo_root=repo_root)

        evaluation_command = [
            sys.executable, "-m", "training.evaluate_candidate",
            str(candidate_path),
            str(incumbent),
            "--pairs", str(args.evaluation_pairs),
            "--simulations", str(args.evaluation_simulations),
            "--max-plies", str(args.max_plies),
            "--seed", str(iteration_seed + 2),
            "--device", args.device,
            "--c-puct", str(args.c_puct),
            "--category-exploration", str(args.category_exploration),
            "--repetition-penalty", str(args.repetition_penalty),
            "--pawn-progress-bonus", str(args.pawn_progress_bonus),
            "--repetition-window", str(args.repetition_window),
            "--initial-wall-candidates", str(args.initial_wall_candidates),
            "--min-decisive-pairs", str(args.min_decisive_pairs),
            "--alpha", str(args.alpha),
            "--output", str(evaluation_path),
        ]
        if args.progressive_widening:
            evaluation_command.append("--progressive-widening")
        run_step(evaluation_command, repo_root=repo_root)

        report = json.loads(evaluation_path.read_text(encoding="utf-8"))
        promoted = bool(report["promotion"]["eligible"])
        promoted_checkpoint = None
        if promoted:
            promoted_checkpoint = champions_dir / f"champion_iteration_{iteration:03d}.pt"
            temporary_checkpoint = promoted_checkpoint.with_suffix(".pt.tmp")
            shutil.copy2(candidate_path, temporary_checkpoint)
            temporary_checkpoint.replace(promoted_checkpoint)
            incumbent = promoted_checkpoint

        entry = {
            "iteration": iteration,
            "seed": iteration_seed,
            "candidate_checkpoint": str(candidate_path),
            "evaluation_report": str(evaluation_path),
            "promoted": promoted,
            "promoted_checkpoint": (
                str(promoted_checkpoint) if promoted_checkpoint else None
            ),
            "incumbent_after_iteration": str(incumbent),
            "candidate_result": report["candidate_result"],
            "paired_result": report["paired_result"],
        }
        history.append(entry)
        state = {
            "run_id": run_id,
            "completed_iterations": iteration,
            "initial_checkpoint": str(initial_checkpoint),
            "current_incumbent": str(incumbent),
            "history": history,
        }
        state_path = run_dir / "training_state.json"
        temporary_state = state_path.with_suffix(".json.tmp")
        temporary_state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary_state.replace(state_path)
        print(
            f"Iteration {iteration}: promoted={promoted}; "
            f"current incumbent={incumbent}",
            flush=True,
        )

    print(f"\nTraining loop completed. Run directory: {run_dir}")
    print(f"Final incumbent: {incumbent}")
    print(f"State/history: {run_dir / 'training_state.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
