#!/usr/bin/env python3
"""Read-only audit of history and time-budget dependence in EXP-001 teacher decisions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import struct
import time
from collections import defaultdict
from pathlib import Path

from training.audit_value_training_coverage import restore_encoded_state, label_position
from quoridor.advanced_ai import AIConfig, QuoridorAI
from quoridor.engine import shortest_distance


def state_fingerprint(encoded: list) -> str:
    digest = hashlib.sha256()
    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
        else:
            value = float(node)
            if not math.isfinite(value):
                raise ValueError("Non-finite encoded state")
            digest.update(struct.pack("<f", value))
    visit(encoded)
    return digest.hexdigest()


def action_payload(action) -> dict:
    if action.kind == "move":
        return {"kind": "move", "destination": list(action.destination)}
    return {
        "kind": "wall",
        "orientation": action.wall.orientation,
        "x": action.wall.x,
        "y": action.wall.y,
    }


def sample_evenly(rows: list[dict], limit: int, seed: int) -> list[dict]:
    ordered = sorted(rows, key=lambda r: (r["game_id"], r["ply"]))
    if limit <= 0 or len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[len(ordered) // 2]]
    # Evenly cover the ordered occurrences. The seed is recorded for provenance;
    # sampling itself is deliberately deterministic and does not hide edge cases.
    indexes = sorted({round(i * (len(ordered) - 1) / (limit - 1)) for i in range(limit)})
    return [ordered[i] for i in indexes]


def prepare_history(ai: QuoridorAI, prior_states, player: int) -> None:
    """Recreate the last 16 real turns seen by this player's persistent AI instance."""
    prior_states = prior_states[-16:]
    ai.actual_state_history = [
        (state.ply, ai._repetition_key(state)) for state in prior_states
    ]
    ai.actual_pawn_history = [
        (state.ply, state.pawns[player], shortest_distance(state, player))
        for state in prior_states
    ]


def run_one(state, prior_states, *, history_mode: str, time_limit: float,
            max_depth: int) -> dict:
    config = AIConfig(time_limit=time_limit, max_depth=max_depth)
    ai = QuoridorAI(config)
    if history_mode == "actual":
        prepare_history(ai, prior_states, state.current_player)
    elif history_mode != "empty":
        raise ValueError(f"Unknown history mode: {history_mode}")

    started = time.perf_counter()
    result = ai.choose_action(state)
    wall_elapsed = (time.perf_counter() - started) * 1000.0
    return {
        "action": action_payload(result.action),
        "score": float(result.score),
        "completed_depth": int(result.depth),
        "timed_out": bool(result.timed_out),
        "elapsed_ms_reported": float(result.elapsed_ms),
        "elapsed_ms_wall": wall_elapsed,
        "nodes": int(result.nodes),
        "history_entries_before_root": len(prior_states[-16:]) if history_mode == "actual" else 0,
        "history_mode": history_mode,
        "time_limit_seconds": time_limit,
        "max_depth": max_depth,
    }


def mean_or_none(values):
    return statistics.mean(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
        default=Path("data/experiments/exp_001/heuristic_decisive_games.jsonl"))
    parser.add_argument("--output-dir", type=Path,
        default=Path("data/diagnostics/teacher_state_dependence/run_001"))
    parser.add_argument("--short-time-limit", type=float, default=0.08)
    parser.add_argument("--long-time-limit", type=float, default=0.32)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-occurrences-per-state", type=int, default=4,
        help="Evenly sampled occurrences per conflicting state; 0 means all occurrences.")
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()

    if not args.dataset.is_file():
        parser.error(f"Dataset missing: {args.dataset}")
    if args.short_time_limit <= 0 or args.long_time_limit <= args.short_time_limit:
        parser.error("Require 0 < short-time-limit < long-time-limit")
    if args.max_depth < 1 or args.max_occurrences_per_state < 0:
        parser.error("max-depth must be positive and occurrence limit non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")

    dataset_bytes = args.dataset.read_bytes()
    dataset_sha = hashlib.sha256(dataset_bytes).hexdigest()
    rows = []
    game_meta = {}
    for line_no, line in enumerate(dataset_bytes.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        required = {"state", "game_id", "ply", "player", "winner"}
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"Line {line_no}: missing fields {sorted(missing)}")
        row = {
            "state_encoded": raw["state"],
            "fingerprint": state_fingerprint(raw["state"]),
            "game_id": int(raw["game_id"]),
            "ply": int(raw["ply"]),
            "player": int(raw["player"]),
            "winner": int(raw["winner"]),
            "target": 1 if int(raw["player"]) == int(raw["winner"]) else -1,
            "total_plies": int(raw.get("total_plies", -1)),
        }
        rows.append(row)
        meta = (row["winner"], row["total_plies"])
        if row["game_id"] in game_meta and game_meta[row["game_id"]] != meta:
            raise ValueError(f"Inconsistent game metadata: game_id={row['game_id']}")
        game_meta[row["game_id"]] = meta

    # Rebuild every position, then reconstruct each game's chronological turns.
    rows_by_game = defaultdict(list)
    groups = defaultdict(list)
    for row in rows:
        state = restore_encoded_state({
            "state": row["state_encoded"], "player": row["player"], "ply": row["ply"]
        })
        if state.current_player != row["player"] or state.ply != row["ply"]:
            raise ValueError("Restored state metadata mismatch")
        row["state"] = state
        row["cohort"] = label_position(state)["cohort"]
        rows_by_game[row["game_id"]].append(row)
        groups[row["fingerprint"]].append(row)
    # Dataset trajectories contain one pre-move state for every ply.
    for game_id, members in rows_by_game.items():
        ordered = sorted(members, key=lambda r: r["ply"])
        expected_plies = list(range(game_meta[game_id][1]))
        actual_plies = [r["ply"] for r in ordered]
        if actual_plies != expected_plies:
            raise ValueError(
                f"Game {game_id} has incomplete/non-contiguous trajectory: "
                f"expected {len(expected_plies)} plies, got {len(actual_plies)}"
            )
        if any(r["player"] != (r["ply"] % 2) for r in ordered):
            raise ValueError(f"Game {game_id} has unexpected player/ply alternation")

    conflict_groups = {
        fp: members for fp, members in groups.items()
        if len({r["target"] for r in members}) > 1
    }
    if not conflict_groups:
        raise ValueError("No target-conflict states found; expected the frozen 52-state cohort")

    # Build the historical context available to the persistent AI object for each
    # player. The generator creates one AI instance per player per game and calls
    # it only on that player's turns. The production method retains the last 16.
    prior_contexts = {}
    for game_id, members in rows_by_game.items():
        ordered = sorted(members, key=lambda r: r["ply"])
        by_player = {0: [], 1: []}
        for row in ordered:
            player = row["player"]
            prior_contexts[(game_id, row["ply"])] = list(by_player[player][-16:])
            by_player[player].append(row["state"])

    selected = []
    for fp, members in sorted(conflict_groups.items()):
        selected.extend(sample_evenly(
            members, args.max_occurrences_per_state, args.seed
        ))
    selected.sort(key=lambda r: (r["fingerprint"], r["game_id"], r["ply"]))

    occurrences = []
    started_all = time.perf_counter()
    for index, row in enumerate(selected, 1):
        state = row["state"]
        prior = prior_contexts[(row["game_id"], row["ply"])]
        runs = {}
        for history_mode in ("empty", "actual"):
            for budget_name, budget in (
                ("short", args.short_time_limit),
                ("long", args.long_time_limit),
            ):
                key = f"{history_mode}_{budget_name}"
                runs[key] = run_one(
                    state, prior, history_mode=history_mode,
                    time_limit=budget, max_depth=args.max_depth,
                )

        occurrences.append({
            "fingerprint": row["fingerprint"],
            "game_id": row["game_id"],
            "ply": row["ply"],
            "player": row["player"],
            "winner": row["winner"],
            "target": row["target"],
            "cohort": row["cohort"],
            "prior_same_player_turns_available": len(prior),
            "runs": runs,
            "comparisons": {
                "history_changes_action_short": (
                    runs["empty_short"]["action"] != runs["actual_short"]["action"]
                ),
                "history_changes_action_long": (
                    runs["empty_long"]["action"] != runs["actual_long"]["action"]
                ),
                "history_changes_score_short": abs(
                    runs["empty_short"]["score"] - runs["actual_short"]["score"]
                ) > 1e-6,
                "history_changes_score_long": abs(
                    runs["empty_long"]["score"] - runs["actual_long"]["score"]
                ) > 1e-6,
                "short_vs_long_action_empty_history": (
                    runs["empty_short"]["action"] != runs["empty_long"]["action"]
                ),
                "short_vs_long_action_actual_history": (
                    runs["actual_short"]["action"] != runs["actual_long"]["action"]
                ),
                "short_timeout_empty_history": runs["empty_short"]["timed_out"],
                "short_timeout_actual_history": runs["actual_short"]["timed_out"],
                "long_timeout_empty_history": runs["empty_long"]["timed_out"],
                "long_timeout_actual_history": runs["actual_long"]["timed_out"],
            },
        })
        if index % 20 == 0:
            print(f"audited {index}/{len(selected)} occurrences", flush=True)

    by_fp = defaultdict(list)
    for item in occurrences:
        by_fp[item["fingerprint"]].append(item)

    per_state = []
    for fp, members in sorted(by_fp.items()):
        target_counts = {
            "positive": sum(r["target"] == 1 for r in conflict_groups[fp]),
            "negative": sum(r["target"] == -1 for r in conflict_groups[fp]),
        }
        positive_actions = {
            json.dumps(x["runs"]["actual_short"]["action"], sort_keys=True)
            for x in members if x["target"] == 1
        }
        negative_actions = {
            json.dumps(x["runs"]["actual_short"]["action"], sort_keys=True)
            for x in members if x["target"] == -1
        }
        per_state.append({
            "fingerprint": fp,
            "dataset_occurrences": len(conflict_groups[fp]),
            "audited_occurrences": len(members),
            "target_counts": target_counts,
            "target_mean": statistics.mean(r["target"] for r in conflict_groups[fp]),
            "history_action_change_short_count": sum(
                x["comparisons"]["history_changes_action_short"] for x in members
            ),
            "history_action_change_long_count": sum(
                x["comparisons"]["history_changes_action_long"] for x in members
            ),
            "short_long_action_change_empty_count": sum(
                x["comparisons"]["short_vs_long_action_empty_history"] for x in members
            ),
            "short_long_action_change_actual_count": sum(
                x["comparisons"]["short_vs_long_action_actual_history"] for x in members
            ),
            "same_actual_short_action_seen_under_both_targets": bool(
                positive_actions & negative_actions
            ),
            "shared_actual_short_action_count_across_targets": len(
                positive_actions & negative_actions
            ),
        })

    def run_summary(key):
        selected_runs = [x["runs"][key] for x in occurrences]
        return {
            "count": len(selected_runs),
            "timeout_count": sum(r["timed_out"] for r in selected_runs),
            "timeout_pct": 100.0 * sum(r["timed_out"] for r in selected_runs) / len(selected_runs),
            "mean_completed_depth": mean_or_none([r["completed_depth"] for r in selected_runs]),
            "depth_histogram": {
                str(d): sum(r["completed_depth"] == d for r in selected_runs)
                for d in sorted({r["completed_depth"] for r in selected_runs})
            },
            "mean_elapsed_ms": mean_or_none([r["elapsed_ms_reported"] for r in selected_runs]),
            "mean_nodes": mean_or_none([r["nodes"] for r in selected_runs]),
        }

    comparisons = {
        "history_action_change_short_count": sum(
            x["comparisons"]["history_changes_action_short"] for x in occurrences
        ),
        "history_action_change_long_count": sum(
            x["comparisons"]["history_changes_action_long"] for x in occurrences
        ),
        "history_score_change_short_count": sum(
            x["comparisons"]["history_changes_score_short"] for x in occurrences
        ),
        "history_score_change_long_count": sum(
            x["comparisons"]["history_changes_score_long"] for x in occurrences
        ),
        "short_long_action_change_empty_count": sum(
            x["comparisons"]["short_vs_long_action_empty_history"] for x in occurrences
        ),
        "short_long_action_change_actual_count": sum(
            x["comparisons"]["short_vs_long_action_actual_history"] for x in occurrences
        ),
        "short_timeout_empty_count": sum(
            x["comparisons"]["short_timeout_empty_history"] for x in occurrences
        ),
        "short_timeout_actual_count": sum(
            x["comparisons"]["short_timeout_actual_history"] for x in occurrences
        ),
        "long_timeout_empty_count": sum(
            x["comparisons"]["long_timeout_empty_history"] for x in occurrences
        ),
        "long_timeout_actual_count": sum(
            x["comparisons"]["long_timeout_actual_history"] for x in occurrences
        ),
    }
    report = {
        "experiment": "teacher_state_dependence_audit",
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_sha,
        "config": {
            "original_teacher_time_limit_seconds": 0.08,
            "short_time_limit_seconds": args.short_time_limit,
            "long_time_limit_seconds": args.long_time_limit,
            "max_depth": args.max_depth,
            "original_teacher_max_depth": 2,
            "max_occurrences_per_conflict_state": args.max_occurrences_per_state,
            "seed": args.seed,
            "sampling": "deterministic evenly spaced by (game_id, ply) within each conflict state",
            "history_reconstruction": (
                "same game's prior states on the same player's turns, last 16; "
                "matches generator's persistent per-player AI instances"
            ),
            "history_empty_definition": (
                "fresh QuoridorAI; choose_action records current root state itself"
            ),
            "read_only": True,
        },
        "integrity": {
            "records": len(rows),
            "games": len(game_meta),
            "unique_states": len(groups),
            "conflicting_states": len(conflict_groups),
            "conflict_occurrences": sum(len(v) for v in conflict_groups.values()),
            "selected_occurrences": len(selected),
            "selected_occurrence_pct": (
                100.0 * len(selected) / sum(len(v) for v in conflict_groups.values())
            ),
            "all_states_reconstructed": True,
            "elapsed_seconds": time.perf_counter() - started_all,
        },
        "run_summary": {
            "empty_short": run_summary("empty_short"),
            "actual_short": run_summary("actual_short"),
            "empty_long": run_summary("empty_long"),
            "actual_long": run_summary("actual_long"),
        },
        "comparisons": comparisons,
        "interpretation_limits": [
            "An action difference proves history can affect the teacher decision on this board; it does not by itself prove that history caused the eventual game winner.",
            "A longer time budget with the same max_depth isolates deadline sensitivity, not deeper search strength.",
            "The audit replays the current checked-out Advanced AI implementation. If that implementation differs from the exact code revision used to generate EXP-001, this is a reproduction audit rather than bit-exact historical replay.",
            "Evenly sampled occurrences cover every conflict state but may not include every trajectory occurrence when the per-state limit is nonzero."
        ],
        "per_state": per_state,
        "occurrences": occurrences,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in report.items() if k != "occurrences"}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "teacher_state_dependence.json").write_text(
        json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(
        f"conflicting_states={len(conflict_groups)} "
        f"conflict_occurrences={report['integrity']['conflict_occurrences']} "
        f"audited={len(selected)} elapsed={report['integrity']['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
