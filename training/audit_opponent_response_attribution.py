#!/usr/bin/env python3
"""Attribute EXP-001 opposite outcomes to the opponent's first response (read-only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path

from training.audit_value_training_coverage import restore_encoded_state
from quoridor.neural_ai.encoder import apply_action_index, decode_action, legal_action_indices

EXPECTED_DATASET_SHA256 = "d9e027c3e4537146dcfa7368baee0ccc01be4fbc067bff01f36fc50f859437b9"


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
    return {"kind": "wall", "orientation": payload.orientation,
            "x": payload.x, "y": payload.y}


def same_state(a, b) -> bool:
    return (a.pawns == b.pawns and a.walls == b.walls
            and a.walls_remaining == b.walls_remaining
            and a.current_player == b.current_player
            and a.winner == b.winner and a.ply == b.ply)


def reconstruct_action(state, expected_next=None, expected_winner=None,
                       expected_terminal_ply=None):
    matches = []
    for index in legal_action_indices(state):
        child = apply_action_index(state, index)
        if expected_next is not None:
            matched = same_state(child, expected_next)
        else:
            matched = (child.winner is not None and child.winner == expected_winner
                       and child.ply == expected_terminal_ply)
        if matched:
            matches.append((index, child))
    if len(matches) != 1:
        raise ValueError(
            f"Non-unique action reconstruction at ply {state.ply}: "
            f"{len(matches)} matches"
        )
    return matches[0]


def summarize_group(fp: str, rows: list[dict]) -> dict:
    positive = [r for r in rows if r["target"] == 1]
    negative = [r for r in rows if r["target"] == -1]
    root_counts = {
        target: Counter(r["actions"][0] for r in cohort)
        for target, cohort in ((1, positive), (-1, negative))
    }
    shared_roots = set(root_counts[1]) & set(root_counts[-1])

    response_rows = []
    for root in sorted(set(root_counts[1]) | set(root_counts[-1])):
        cohort = [r for r in rows if r["actions"][0] == root]
        by_response = defaultdict(Counter)
        for row in cohort:
            if len(row["actions"]) < 2:
                raise ValueError(f"Occurrence has no opponent response: {fp}")
            by_response[row["actions"][1]][row["target"]] += 1
        for response, counts in sorted(by_response.items()):
            support = counts[1] + counts[-1]
            response_rows.append({
                "root_action_index": root,
                "root_action": action_payload(root),
                "opponent_response_index": response,
                "opponent_response": action_payload(response),
                "positive_count": counts[1],
                "negative_count": counts[-1],
                "positive_target_rate": counts[1] / support,
                "support": support,
            })

    pair_counts = Counter()
    for p in positive:
        for n in negative:
            if p["actions"][0] != n["actions"][0]:
                pair_counts["different_root_action"] += 1
                continue
            pair_counts["same_root_action"] += 1
            if len(p["actions"]) < 2 or len(n["actions"]) < 2:
                pair_counts["missing_opponent_response"] += 1
                continue
            if p["actions"][1] == n["actions"][1]:
                pair_counts["same_opponent_response"] += 1
                if len(p["actions"]) >= 3 and len(n["actions"]) >= 3:
                    if p["actions"][2] == n["actions"][2]:
                        pair_counts["same_next_root_player_action"] += 1
                    else:
                        pair_counts["different_next_root_player_action"] += 1
                else:
                    pair_counts["missing_next_root_player_action"] += 1
            else:
                pair_counts["different_opponent_response"] += 1

    # Response-type association, conditioned on a root action and outcome cohort.
    type_counts = defaultdict(Counter)
    for row in rows:
        response = action_payload(row["actions"][1])
        type_counts[(row["target"], response["kind"])]["count"] += 1
    response_type_summary = []
    for (target, kind), counts in sorted(type_counts.items()):
        response_type_summary.append({
            "outcome": "positive" if target == 1 else "negative",
            "opponent_response_kind": kind,
            "count": counts["count"],
        })

    total_pairs = len(positive) * len(negative)
    return {
        "fingerprint": fp,
        "occurrences": len(rows),
        "positive_occurrences": len(positive),
        "negative_occurrences": len(negative),
        "shared_root_action_count": len(shared_roots),
        "shared_root_actions": [action_payload(x) for x in sorted(shared_roots)],
        "cross_outcome_pairs": total_pairs,
        "pairwise_response_comparison": dict(sorted(pair_counts.items())),
        "opponent_response_outcome_counts": response_rows,
        "opponent_response_kind_by_outcome": response_type_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
        default=Path("data/experiments/exp_001/heuristic_decisive_games.jsonl"))
    parser.add_argument("--output-dir", type=Path,
        default=Path("data/diagnostics/opponent_response_attribution/run_001"))
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
            "state": record["state"], "player": record["player"], "ply": record["ply"]
        })
        if state.current_player != int(record["player"]):
            raise ValueError(f"Line {line_number}: player-to-move mismatch")
        games[int(record["game_id"])].append({
            "state": state, "fp": fingerprint(record["state"]),
            "game_id": int(record["game_id"]), "ply": int(record["ply"]),
            "player": int(record["player"]), "winner": int(record["winner"]),
            "total_plies": int(record["total_plies"]),
            "target": 1 if int(record["player"]) == int(record["winner"]) else -1,
        })

    by_state = defaultdict(list)
    transition_count = 0
    for game_id, rows in games.items():
        rows.sort(key=lambda row: row["ply"])
        total = rows[0]["total_plies"]
        winner = rows[0]["winner"]
        if [r["ply"] for r in rows] != list(range(total)):
            raise ValueError(f"Game {game_id}: non-contiguous or incomplete plies")
        if any(r["total_plies"] != total or r["winner"] != winner for r in rows):
            raise ValueError(f"Game {game_id}: inconsistent terminal metadata")
        for index, row in enumerate(rows):
            if index + 1 < len(rows):
                action, _ = reconstruct_action(row["state"], expected_next=rows[index + 1]["state"])
            else:
                action, child = reconstruct_action(
                    row["state"], expected_winner=winner, expected_terminal_ply=total
                )
                if child.winner != winner or child.ply != total:
                    raise ValueError(f"Game {game_id}: final terminal state mismatch")
            row.setdefault("actions", []).append(action)
            transition_count += 1
        for row in rows:
            by_state[row["fp"]].append(row)

    conflicts = {
        fp: sorted(rows, key=lambda r: (r["game_id"], r["ply"]))
        for fp, rows in by_state.items()
        if len({r["target"] for r in rows}) > 1
    }
    conflict_occurrences = sum(len(rows) for rows in conflicts.values())
    if len(games) != 50 or transition_count != 3780 or len(conflicts) != 52 or conflict_occurrences != 1987:
        raise ValueError(
            "Frozen cohort mismatch: expected 50 games, 3,780 transitions, "
            "52 conflicting states, and 1,987 occurrences."
        )

    per_state = [summarize_group(fp, rows) for fp, rows in sorted(conflicts.items())]
    aggregate = Counter()
    root_response_outcomes = []
    response_kind_aggregate = defaultdict(Counter)
    for state in per_state:
        aggregate["pairs"] += state["cross_outcome_pairs"]
        for key, value in state["pairwise_response_comparison"].items():
            aggregate[key] += value
        for item in state["opponent_response_outcome_counts"]:
            root_response_outcomes.append({"fingerprint": state["fingerprint"], **item})
        for item in state["opponent_response_kind_by_outcome"]:
            response_kind_aggregate[(item["outcome"], item["opponent_response_kind"])]["count"] += item["count"]

    summary = {
        "experiment": "opponent_response_attribution_audit",
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_sha,
        "integrity": {
            "games": len(games), "records": transition_count,
            "conflicting_states": len(conflicts),
            "conflict_occurrences": conflict_occurrences,
            "verified_state_transitions": transition_count,
            "reconstructed_actions": transition_count,
            "all_reconstructed_actions_legal": True,
            "all_transitions_match_next_state": True,
            "all_terminal_winners_verified": True,
            "ambiguous_or_unmatched_transitions": 0,
        },
        "method": {
            "root_action": "action at offset 0 from the repeated state",
            "opponent_response": "action at offset 1, after the common root action",
            "next_root_player_action": "action at offset 2, after the opponent response",
            "outcome": "terminal winner from the repeated state's side-to-move perspective",
            "pairing": "all positive/negative occurrence pairs within each exact encoded state",
            "read_only": True,
            "action_source": "uniquely reconstructed from consecutive recorded states; not a stored policy label",
        },
        "aggregate_pairwise_comparison": {
            "cross_outcome_pairs": aggregate["pairs"],
            "same_root_action_pairs": aggregate["same_root_action"],
            "different_root_action_pairs": aggregate["different_root_action"],
            "same_opponent_response_pairs_given_same_root": aggregate["same_opponent_response"],
            "different_opponent_response_pairs_given_same_root": aggregate["different_opponent_response"],
            "same_next_root_player_action_pairs_given_same_root_and_response": aggregate["same_next_root_player_action"],
            "different_next_root_player_action_pairs_given_same_root_and_response": aggregate["different_next_root_player_action"],
            "missing_next_action_pairs": aggregate["missing_next_root_player_action"],
        },
        "opponent_response_kind_by_outcome": [
            {"outcome": outcome, "response_kind": kind, "count": counts["count"]}
            for (outcome, kind), counts in sorted(response_kind_aggregate.items())
        ],
        "per_state": per_state,
        "root_response_outcome_counts": root_response_outcomes,
        "interpretation_limits": [
            "Observational attribution only; no action was intervened on and no game was re-simulated.",
            "Opponent response comparisons are conditioned on the same exact root state and same observed root action.",
            "An association between a response and terminal outcome does not establish causation or response quality.",
            "The dataset does not store policy/action fields; actions are uniquely reconstructed from state transitions.",
            "This audit does not modify training, targets, loss, checkpoints, replay, PUCT, or source dataset.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(
        {k: v for k, v in summary.items() if k not in ("per_state", "root_response_outcome_counts")}, indent=2
    ) + "\n", encoding="utf-8")
    (args.output_dir / "opponent_response_attribution.json").write_text(
        json.dumps(summary, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "integrity": summary["integrity"],
        "aggregate_pairwise_comparison": summary["aggregate_pairwise_comparison"],
        "opponent_response_kind_by_outcome": summary["opponent_response_kind_by_outcome"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
