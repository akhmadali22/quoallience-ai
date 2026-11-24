#!/usr/bin/env python3
"""Attribute the first divergent EXP-001 action to its actor and tactical context."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
from collections import Counter, defaultdict
from pathlib import Path

from training.audit_value_training_coverage import label_position, restore_encoded_state
from quoridor.neural_ai.encoder import (
    apply_action_index, decode_action, encode_state, legal_action_indices,
)

EXPECTED_DATASET_SHA256 = "d9e027c3e4537146dcfa7368baee0ccc01be4fbc067bff01f36fc50f859437b9"
EXPECTED_GAMES = 50
EXPECTED_RECORDS = 3780
EXPECTED_CONFLICT_STATES = 52
EXPECTED_CONFLICT_OCCURRENCES = 1987
EXPECTED_PAIRS = 19516


def fingerprint(encoded: list) -> str:
    digest = hashlib.sha256()

    def walk(value):
        if isinstance(value, list):
            for item in value:
                walk(item)
        else:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Non-finite encoded state value")
            digest.update(struct.pack("<f", number))

    walk(encoded)
    return digest.hexdigest()


def action_payload(index: int) -> dict:
    kind, payload = decode_action(index)
    if kind == "move":
        return {"kind": "move", "destination": list(payload)}
    return {
        "kind": "wall", "orientation": payload.orientation,
        "x": payload.x, "y": payload.y,
    }


def same_state(a, b) -> bool:
    return (
        a.pawns == b.pawns
        and a.walls == b.walls
        and a.walls_remaining == b.walls_remaining
        and a.current_player == b.current_player
        and a.winner == b.winner
        and a.ply == b.ply
    )


def reconstruct_action(state, expected_next=None, expected_winner=None,
                       expected_terminal_ply=None):
    matches = []
    for index in legal_action_indices(state):
        child = apply_action_index(state, index)
        if expected_next is not None:
            matched = same_state(child, expected_next)
        else:
            matched = (
                child.winner is not None
                and child.winner == expected_winner
                and child.ply == expected_terminal_ply
            )
        if matched:
            matches.append((index, child))
    if len(matches) != 1:
        raise ValueError(
            f"Non-unique action reconstruction at ply {state.ply}: "
            f"{len(matches)} matches"
        )
    return matches[0]


def percentile(values: list[int], fraction: float):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def context_for_state(state, cache: dict[str, dict]) -> tuple[str, dict]:
    encoded = encode_state(state)
    fp = fingerprint(encoded)
    if fp not in cache:
        label = label_position(state)
        cache[fp] = {
            "fingerprint": fp,
            "cohort": label["cohort"],
            "opponent_immediate_win": label["opponent_immediate_win"],
            "legal_action_count": label["legal_action_count"],
            "root_immediate_win_count": label["root_immediate_win_count"],
            "defensive_action_count": label["defensive_action_count"],
            "allows_immediate_loss_action_count": label["allows_immediate_loss_action_count"],
            "forced_loss": label["forced_loss"],
        }
    return fp, cache[fp]


def compare_pair(fp: str, positive: dict, negative: dict,
                 context_cache: dict[str, dict]) -> dict:
    pos_actions = positive["trajectory_actions"]
    neg_actions = negative["trajectory_actions"]
    limit = min(len(pos_actions), len(neg_actions))
    common = 0
    while common < limit and pos_actions[common] == neg_actions[common]:
        common += 1

    if common == limit:
        if len(pos_actions) != len(neg_actions):
            raise ValueError(
                f"Trajectory suffix length mismatch without action divergence at {fp}"
            )
        raise ValueError(
            f"Conflicting terminal outcomes have identical complete action suffixes at {fp}"
        )

    pos_state = positive["trajectory_states"][common]
    neg_state = negative["trajectory_states"][common]
    if not same_state(pos_state, neg_state):
        raise ValueError(
            f"First-divergence states do not match after {common} shared actions at {fp}"
        )

    root_player = positive["state"].current_player
    actor = pos_state.current_player
    actor_role = "root_player" if actor == root_player else "opponent"
    pos_action = pos_actions[common]
    neg_action = neg_actions[common]
    pos_payload = action_payload(pos_action)
    neg_payload = action_payload(neg_action)
    context_fp, context = context_for_state(pos_state, context_cache)

    return {
        "root_state_fingerprint": fp,
        "positive_game_id": positive["game_id"],
        "positive_start_ply": positive["ply"],
        "negative_game_id": negative["game_id"],
        "negative_start_ply": negative["ply"],
        "common_prefix_actions": common,
        "divergence_ply": pos_state.ply,
        "divergence_actor_player_id": actor,
        "root_player_id": root_player,
        "divergence_actor_role": actor_role,
        "positive_action_index": pos_action,
        "positive_action": pos_payload,
        "negative_action_index": neg_action,
        "negative_action": neg_payload,
        "positive_action_kind": pos_payload["kind"],
        "negative_action_kind": neg_payload["kind"],
        "action_kinds_match": pos_payload["kind"] == neg_payload["kind"],
        "divergence_state_fingerprint": context_fp,
        "tactical_context": context,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("data/experiments/exp_001/heuristic_decisive_games.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/diagnostics/first_divergence_actor_attribution/run_001"),
    )
    args = parser.parse_args()

    if not args.dataset.is_file():
        parser.error(f"Dataset missing: {args.dataset}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")

    raw = args.dataset.read_bytes()
    dataset_sha = hashlib.sha256(raw).hexdigest()
    if dataset_sha != EXPECTED_DATASET_SHA256:
        raise ValueError("Dataset SHA-256 does not match pinned EXP-001 artifact.")

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
            "state": record["state"],
            "player": record["player"],
            "ply": record["ply"],
        })
        if state.current_player != int(record["player"]):
            raise ValueError(f"Line {line_number}: player-to-move mismatch")
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
        total = rows[0]["total_plies"]
        winner = rows[0]["winner"]
        if [row["ply"] for row in rows] != list(range(total)):
            raise ValueError(f"Game {game_id}: non-contiguous or incomplete plies")
        if any(row["total_plies"] != total or row["winner"] != winner for row in rows):
            raise ValueError(f"Game {game_id}: inconsistent terminal metadata")

        for index, row in enumerate(rows):
            if index + 1 < len(rows):
                action, _ = reconstruct_action(
                    row["state"], expected_next=rows[index + 1]["state"]
                )
            else:
                action, child = reconstruct_action(
                    row["state"],
                    expected_winner=winner,
                    expected_terminal_ply=total,
                )
                if child.winner != winner or child.ply != total:
                    raise ValueError(f"Game {game_id}: final terminal state mismatch")
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
    conflict_occurrences = sum(len(rows) for rows in conflicts.values())
    if (
        len(games) != EXPECTED_GAMES
        or transition_count != EXPECTED_RECORDS
        or len(conflicts) != EXPECTED_CONFLICT_STATES
        or conflict_occurrences != EXPECTED_CONFLICT_OCCURRENCES
    ):
        raise ValueError(
            "Frozen cohort mismatch: expected 50 games, 3,780 transitions, "
            "52 conflicting states, and 1,987 occurrences."
        )

    context_cache: dict[str, dict] = {}
    pair_rows = []
    per_state = []
    for fp, rows in sorted(conflicts.items()):
        positive = [row for row in rows if row["target"] == 1]
        negative = [row for row in rows if row["target"] == -1]
        state_pairs = [
            compare_pair(fp, pos, neg, context_cache)
            for pos in positive for neg in negative
        ]
        if len(state_pairs) != len(positive) * len(negative):
            raise ValueError(f"Pair count mismatch for state {fp}")
        pair_rows.extend(state_pairs)

        actor_counts = Counter(row["divergence_actor_role"] for row in state_pairs)
        actor_kind_counts = Counter(
            (row["divergence_actor_role"], row["positive_action_kind"], row["negative_action_kind"])
            for row in state_pairs
        )
        prefix_lengths = [row["common_prefix_actions"] for row in state_pairs]
        tactical_counts = Counter(
            (row["divergence_actor_role"], row["tactical_context"]["cohort"])
            for row in state_pairs
        )
        per_state.append({
            "fingerprint": fp,
            "occurrences": len(rows),
            "positive_occurrences": len(positive),
            "negative_occurrences": len(negative),
            "cross_outcome_pairs": len(state_pairs),
            "first_divergence_actor_counts": dict(sorted(actor_counts.items())),
            "common_prefix_actions": {
                "mean": statistics.mean(prefix_lengths) if prefix_lengths else None,
                "median": statistics.median(prefix_lengths) if prefix_lengths else None,
                "p10": percentile(prefix_lengths, 0.10),
                "p25": percentile(prefix_lengths, 0.25),
                "p75": percentile(prefix_lengths, 0.75),
                "p90": percentile(prefix_lengths, 0.90),
            },
            "actor_and_action_kind_pairs": [
                {
                    "actor_role": actor,
                    "positive_action_kind": pos_kind,
                    "negative_action_kind": neg_kind,
                    "count": count,
                }
                for (actor, pos_kind, neg_kind), count in sorted(actor_kind_counts.items())
            ],
            "tactical_context_by_actor": [
                {"actor_role": actor, "cohort": cohort, "count": count}
                for (actor, cohort), count in sorted(tactical_counts.items())
            ],
        })

    if len(pair_rows) != EXPECTED_PAIRS:
        raise ValueError(
            f"Expected {EXPECTED_PAIRS:,} cross-outcome pairs, got {len(pair_rows):,}"
        )

    actor_counts = Counter(row["divergence_actor_role"] for row in pair_rows)
    actor_action_counts = Counter(
        (row["divergence_actor_role"], row["positive_action_kind"], row["negative_action_kind"])
        for row in pair_rows
    )
    context_counts = Counter(
        (
            row["divergence_actor_role"],
            row["tactical_context"]["cohort"],
            bool(row["tactical_context"]["forced_loss"]),
        )
        for row in pair_rows
    )
    prefix_lengths = [row["common_prefix_actions"] for row in pair_rows]
    prefix_histogram = Counter(prefix_lengths)
    actor_prefix = defaultdict(list)
    for row in pair_rows:
        actor_prefix[row["divergence_actor_role"]].append(row["common_prefix_actions"])

    summary = {
        "experiment": "first_divergence_actor_attribution_audit",
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_sha,
        "integrity": {
            "games": len(games),
            "records": transition_count,
            "conflicting_states": len(conflicts),
            "conflict_occurrences": conflict_occurrences,
            "cross_outcome_pairs": len(pair_rows),
            "verified_state_transitions": transition_count,
            "reconstructed_actions": transition_count,
            "all_reconstructed_actions_legal": True,
            "all_transitions_match_next_state": True,
            "all_terminal_winners_verified": True,
            "all_first_divergence_states_match_after_common_prefix": True,
            "all_pairs_have_a_first_divergence": True,
            "ambiguous_or_unmatched_transitions": 0,
        },
        "method": {
            "pairing": "all positive/negative occurrence pairs within each exact encoded root state",
            "first_divergence": "first action index at which the two reconstructed suffixes differ",
            "actor": "side to move in the identical divergence state, relative to the repeated root state's side to move",
            "positive_action": "action observed in the occurrence whose terminal target is +1",
            "negative_action": "action observed in the occurrence whose terminal target is -1",
            "tactical_context": "recomputed with the existing engine-backed label_position classifier at the common divergence state",
            "action_source": "uniquely reconstructed from legal state transitions; not a stored policy label",
            "read_only": True,
        },
        "first_divergence_actor_summary": {
            "root_player": actor_counts["root_player"],
            "opponent": actor_counts["opponent"],
            "root_player_pct": 100.0 * actor_counts["root_player"] / len(pair_rows),
            "opponent_pct": 100.0 * actor_counts["opponent"] / len(pair_rows),
        },
        "common_prefix_actions": {
            "mean": statistics.mean(prefix_lengths),
            "median": statistics.median(prefix_lengths),
            "p10": percentile(prefix_lengths, 0.10),
            "p25": percentile(prefix_lengths, 0.25),
            "p75": percentile(prefix_lengths, 0.75),
            "p90": percentile(prefix_lengths, 0.90),
            "histogram": dict(sorted(prefix_histogram.items())),
            "by_actor": {
                actor: {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "p10": percentile(values, 0.10),
                    "p90": percentile(values, 0.90),
                }
                for actor, values in sorted(actor_prefix.items())
            },
        },
        "actor_and_action_kind_pairs": [
            {
                "actor_role": actor,
                "positive_action_kind": pos_kind,
                "negative_action_kind": neg_kind,
                "count": count,
            }
            for (actor, pos_kind, neg_kind), count in sorted(actor_action_counts.items())
        ],
        "tactical_context_by_actor": [
            {
                "actor_role": actor,
                "cohort": cohort,
                "forced_loss": forced_loss,
                "count": count,
            }
            for (actor, cohort, forced_loss), count in sorted(context_counts.items())
        ],
        "unique_divergence_states": len(context_cache),
        "per_state": per_state,
        "pairs": pair_rows,
        "interpretation_limits": [
            "This is observational attribution, not a counterfactual intervention or proof of causality.",
            "The actor is identified from the side-to-move in the shared state immediately before the first different observed action.",
            "Both trajectories are identical through the common prefix; deterministic transition semantics require their divergence states to match exactly.",
            "The tactical cohort is a one-ply engine classification, not a full strategic evaluation.",
            "Positive/negative action labels describe terminal trajectory outcome, not action quality.",
            "All actions are reconstructed from legal transitions; the source dataset contains no stored policy/action field.",
            "No training, optimizer, target, loss, checkpoint, replay, PUCT, or source dataset is modified.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_only = {key: value for key, value in summary.items() if key != "pairs"}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_only, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "first_divergence_actor_attribution.json").write_text(
        json.dumps(summary, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "integrity": summary["integrity"],
        "first_divergence_actor_summary": summary["first_divergence_actor_summary"],
        "common_prefix_actions": {
            key: value for key, value in summary["common_prefix_actions"].items()
            if key != "histogram"
        },
        "unique_divergence_states": summary["unique_divergence_states"],
        "tactical_context_by_actor": summary["tactical_context_by_actor"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
