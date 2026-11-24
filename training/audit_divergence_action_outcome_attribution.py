#!/usr/bin/env python3
"""Compare first-divergence actions with a deterministic history-free minimax evaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
import time
from collections import Counter, defaultdict
from pathlib import Path

from training.audit_first_divergence_actor_attribution import (
    EXPECTED_CONFLICT_OCCURRENCES,
    EXPECTED_CONFLICT_STATES,
    EXPECTED_DATASET_SHA256,
    EXPECTED_GAMES,
    EXPECTED_PAIRS,
    fingerprint,
    reconstruct_action,
    same_state,
)
from training.audit_value_training_coverage import restore_encoded_state
from quoridor.advanced_ai import AIConfig, MATE_SCORE, QuoridorAI, SearchTimeout
from quoridor.neural_ai.encoder import apply_action_index, encode_state


EVALUATOR_DEPTH_AFTER_ACTION = 2
MAX_WALL_CANDIDATES = 8
TRANSPOSITION_TABLE_LIMIT = 50000
EXPECTED_RECORDS = 3780


def action_payload(index: int) -> dict:
    from quoridor.neural_ai.encoder import decode_action
    kind, payload = decode_action(index)
    if kind == "move":
        return {"kind": "move", "destination": list(payload)}
    return {
        "kind": "wall", "orientation": payload.orientation,
        "x": payload.x, "y": payload.y,
    }


def load_trajectories(dataset: Path):
    raw = dataset.read_bytes()
    dataset_sha = hashlib.sha256(raw).hexdigest()
    if dataset_sha != EXPECTED_DATASET_SHA256:
        raise ValueError(
            f"Dataset SHA mismatch: expected {EXPECTED_DATASET_SHA256}, got {dataset_sha}"
        )

    games = defaultdict(list)
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        required = {"state", "game_id", "ply", "player", "winner", "total_plies"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Line {line_number}: missing fields {sorted(missing)}")
        state = restore_encoded_state({
            "state": record["state"], "player": record["player"], "ply": record["ply"],
        })
        games[int(record["game_id"])].append({
            "state": state,
            "fp": fingerprint(record["state"]),
            "game_id": int(record["game_id"]),
            "ply": int(record["ply"]),
            "player": int(record["player"]),
            "winner": int(record["winner"]),
            "total_plies": int(record["total_plies"]),
            "target": 1 if int(record["player"]) == int(record["winner"]) else -1,
        })

    by_state = defaultdict(list)
    transition_count = 0
    for game_id, rows in games.items():
        rows.sort(key=lambda row: row["ply"])
        total, winner = rows[0]["total_plies"], rows[0]["winner"]
        if [row["ply"] for row in rows] != list(range(total)):
            raise ValueError(f"Game {game_id}: non-contiguous/incomplete plies")
        if any(row["total_plies"] != total or row["winner"] != winner for row in rows):
            raise ValueError(f"Game {game_id}: inconsistent terminal metadata")
        for index, row in enumerate(rows):
            if index + 1 < len(rows):
                action, _ = reconstruct_action(row["state"], expected_next=rows[index + 1]["state"])
            else:
                action, child = reconstruct_action(
                    row["state"], expected_winner=winner, expected_terminal_ply=total
                )
                if child.winner != winner or child.ply != total:
                    raise ValueError(f"Game {game_id}: terminal mismatch")
            row["action"] = action
            transition_count += 1
        for index, row in enumerate(rows):
            suffix = rows[index:]
            row["trajectory_actions"] = [item["action"] for item in suffix]
            row["trajectory_states"] = [item["state"] for item in suffix]
            by_state[row["fp"]].append(row)

    conflicts = {
        fp: sorted(rows, key=lambda row: (row["game_id"], row["ply"]))
        for fp, rows in by_state.items()
        if len({row["target"] for row in rows}) > 1
    }
    occurrences = sum(len(rows) for rows in conflicts.values())
    if (len(games), transition_count, len(conflicts), occurrences) != (
        EXPECTED_GAMES, EXPECTED_RECORDS, EXPECTED_CONFLICT_STATES,
        EXPECTED_CONFLICT_OCCURRENCES,
    ):
        raise ValueError("Frozen EXP-001 integrity cohort mismatch")
    return dataset_sha, games, conflicts, transition_count


def first_divergence(positive: dict, negative: dict):
    pa, na = positive["trajectory_actions"], negative["trajectory_actions"]
    limit = min(len(pa), len(na))
    common = 0
    while common < limit and pa[common] == na[common]:
        common += 1
    if common == limit or not same_state(
        positive["trajectory_states"][common], negative["trajectory_states"][common]
    ):
        raise ValueError("Invalid or missing shared first-divergence state")
    state = positive["trajectory_states"][common]
    return common, state, pa[common], na[common]


def evaluate_actions(state, root_player: int, action_indices: set[int]) -> dict[int, dict]:
    """Fixed-depth minimax from the original root player's perspective."""
    config = AIConfig(
        time_limit=3600.0,
        max_depth=EVALUATOR_DEPTH_AFTER_ACTION,
        max_wall_candidates=MAX_WALL_CANDIDATES,
        transposition_table_limit=TRANSPOSITION_TABLE_LIMIT,
        # The evaluator is history-free: a fresh AI instance has empty actual history.
    )
    ai = QuoridorAI(config)
    actor_player = state.current_player
    ai.root_player = actor_player
    ai.root_ply = state.ply
    ai.deadline = time.monotonic() + config.time_limit
    results = {}
    for action_index in sorted(action_indices):
        child = apply_action_index(state, action_index)
        try:
            score = ai._minimax(child, EVALUATOR_DEPTH_AFTER_ACTION, -math.inf, math.inf)
        except SearchTimeout as exc:
            raise RuntimeError(
                f"Fixed-depth evaluator timed out at ply={state.ply}, action={action_index}; "
                "audit aborted rather than mixing search depths."
            ) from exc
        results[action_index] = {
            "score_root_perspective": float(score if actor_player == root_player else -score),
            "score_actor_perspective": float(score),
            "action": action_payload(action_index),
            "terminal_after_forced_action": child.winner is not None,
            "nodes_cumulative": ai.nodes,
            "transposition_entries": len(ai.transposition_table),
        }
    return results


def classify_delta(delta: float) -> str:
    if delta > 1e-9:
        return "positive_action_higher_for_root"
    if delta < -1e-9:
        return "negative_action_higher_for_root"
    return "tie_within_numeric_tolerance"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
        default=Path("data/experiments/exp_001/heuristic_decisive_games.jsonl"))
    parser.add_argument("--output-dir", type=Path,
        default=Path("data/diagnostics/divergence_action_outcome_attribution/run_001"))
    args = parser.parse_args()
    if not args.dataset.is_file():
        parser.error(f"Dataset missing: {args.dataset}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")

    started = time.perf_counter()
    dataset_sha, games, conflicts, transition_count = load_trajectories(args.dataset)

    state_records = {}
    pair_specs = []
    for root_fp, rows in sorted(conflicts.items()):
        positives = [r for r in rows if r["target"] == 1]
        negatives = [r for r in rows if r["target"] == -1]
        for positive in positives:
            for negative in negatives:
                common, state, positive_action, negative_action = first_divergence(positive, negative)
                encoded_fp = fingerprint(encode_state(state))
                record = state_records.setdefault(encoded_fp, {
                    "state": state,
                    "root_player": positive["state"].current_player,
                    "actions": set(),
                })
                if record["root_player"] != positive["state"].current_player:
                    raise ValueError("Same divergence fingerprint has inconsistent root perspective")
                record["actions"].update((positive_action, negative_action))
                pair_specs.append({
                    "root_state_fingerprint": root_fp,
                    "divergence_state_fingerprint": encoded_fp,
                    "positive_game_id": positive["game_id"],
                    "positive_start_ply": positive["ply"],
                    "negative_game_id": negative["game_id"],
                    "negative_start_ply": negative["ply"],
                    "common_prefix_actions": common,
                    "divergence_ply": state.ply,
                    "actor_player_id": state.current_player,
                    "root_player_id": positive["state"].current_player,
                    "actor_role": (
                        "root_player" if state.current_player == positive["state"].current_player
                        else "opponent"
                    ),
                    "positive_action_index": positive_action,
                    "negative_action_index": negative_action,
                })

    if len(pair_specs) != EXPECTED_PAIRS:
        raise ValueError(f"Expected {EXPECTED_PAIRS} pairs, got {len(pair_specs)}")

    evaluations = {}
    evaluator_stats = {}
    for fp, record in sorted(state_records.items()):
        values = evaluate_actions(record["state"], record["root_player"], record["actions"])
        evaluations[fp] = values
        evaluator_stats[fp] = {
            "root_player": record["root_player"],
            "divergence_ply": record["state"].ply,
            "unique_observed_actions_evaluated": len(values),
            "nodes_cumulative": max((v["nodes_cumulative"] for v in values.values()), default=0),
            "transposition_entries": max((v["transposition_entries"] for v in values.values()), default=0),
        }

    for row in pair_specs:
        values = evaluations[row["divergence_state_fingerprint"]]
        positive_score = values[row["positive_action_index"]]["score_root_perspective"]
        negative_score = values[row["negative_action_index"]]["score_root_perspective"]
        delta = positive_score - negative_score
        row.update({
            "positive_action": values[row["positive_action_index"]]["action"],
            "negative_action": values[row["negative_action_index"]]["action"],
            "positive_action_score_root_perspective": positive_score,
            "negative_action_score_root_perspective": negative_score,
            "positive_minus_negative_score": delta,
            "comparison": classify_delta(delta),
        })

    comparison_counts = Counter(row["comparison"] for row in pair_specs)
    by_actor = defaultdict(list)
    by_context = defaultdict(list)
    for row in pair_specs:
        by_actor[row["actor_role"]].append(row)
        by_context[row["actor_role"]].append(row)

    def aggregate(rows):
        if not rows:
            return {"count": 0}
        deltas = [r["positive_minus_negative_score"] for r in rows]
        counts = Counter(r["comparison"] for r in rows)
        return {
            "pair_count": len(rows),
            "positive_action_higher_count": counts["positive_action_higher_for_root"],
            "negative_action_higher_count": counts["negative_action_higher_for_root"],
            "tie_count": counts["tie_within_numeric_tolerance"],
            "positive_action_higher_pct": 100 * counts["positive_action_higher_for_root"] / len(rows),
            "negative_action_higher_pct": 100 * counts["negative_action_higher_for_root"] / len(rows),
            "mean_positive_minus_negative_score": statistics.mean(deltas),
            "median_positive_minus_negative_score": statistics.median(deltas),
            "mean_absolute_score_gap": statistics.mean(abs(x) for x in deltas),
        }

    unique_comparisons = {}
    for row in pair_specs:
        key = (
            row["divergence_state_fingerprint"],
            row["positive_action_index"],
            row["negative_action_index"],
        )
        unique_comparisons[key] = row
    unique_rows = list(unique_comparisons.values())

    summary = {
        "experiment": "divergence_action_outcome_attribution_audit",
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_sha,
        "integrity": {
            "games": len(games),
            "records": transition_count,
            "conflicting_states": len(conflicts),
            "conflict_occurrences": sum(len(rows) for rows in conflicts.values()),
            "cross_outcome_pairs": len(pair_specs),
            "unique_divergence_states": len(state_records),
            "unique_state_action_evaluations": sum(len(r["actions"]) for r in state_records.values()),
            "all_first_divergence_states_reconstructed": True,
            "all_actions_legal_by_transition_reconstruction": True,
            "ambiguous_or_unmatched_transitions": 0,
        },
        "evaluator": {
            "type": "deterministic fixed-depth alpha-beta minimax with static path/resource heuristic",
            "implementation": "quoridor.advanced_ai.QuoridorAI._minimax",
            "depth_after_forced_action": EVALUATOR_DEPTH_AFTER_ACTION,
            "max_wall_candidates_per_node": MAX_WALL_CANDIDATES,
            "transposition_table_limit": TRANSPOSITION_TABLE_LIMIT,
            "history": "empty; a fresh evaluator instance is used per divergence state",
            "perspective": "minimax searches from the divergence actor perspective, then scores are sign-normalized to the original root player perspective",
            "wall_candidate_policy": "existing deterministic path-based candidate pruning; not exhaustive over all legal walls",
            "timeout_policy": "one-hour safety deadline; any timeout aborts the audit, so no mixed-depth rows are emitted",
            "neural_checkpoint_used": False,
            "teacher_time_limit_used": False,
            "limitations": [
                "This is a heuristic reference evaluator, not an exact game-theoretic solver or ground truth.",
                "The evaluator shares the repository's advanced_ai static evaluation and candidate-wall generator, so it is not fully independent of the historical teacher implementation.",
                "Only observed actions are compared; unobserved legal alternatives are not ranked.",
                "Wall search is candidate-pruned and depth-limited.",
                "Pair rows reuse the same occurrences and action evaluations; they are not independent statistical samples.",
            ],
        },
        "overall_pair_comparison": aggregate(pair_specs),
        "comparison_counts": dict(sorted(comparison_counts.items())),
        "by_divergence_actor": {
            actor: aggregate(rows) for actor, rows in sorted(by_actor.items())
        },
        "unique_state_action_pair_comparison": aggregate(unique_rows),
        "unique_state_action_pair_count": len(unique_rows),
        "per_divergence_state_evaluator_stats": evaluator_stats,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "interpretation_limits": [
            "A higher score means the evaluator prefers that action for the original root player, not necessarily that the action guarantees the eventual dataset outcome.",
            "When the divergence actor is the opponent, an action scored higher for the root player is an opponent mistake from the root player's perspective.",
            "A positive trajectory action having a higher score is alignment with the observed terminal label, not causal proof.",
            "No model training, optimizer, loss, target, checkpoint, PUCT, or source dataset is modified.",
        ],
        "pairs": pair_specs,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "pairs"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "divergence_action_outcome_attribution.json").write_text(
        json.dumps(summary, separators=(",", ":")) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "integrity": summary["integrity"],
        "evaluator": summary["evaluator"],
        "overall_pair_comparison": summary["overall_pair_comparison"],
        "by_divergence_actor": summary["by_divergence_actor"],
        "unique_state_action_pair_comparison": summary["unique_state_action_pair_comparison"],
        "elapsed_seconds": summary["elapsed_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
