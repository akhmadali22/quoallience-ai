#!/usr/bin/env python3
"""Attribute EXP-001 conflicting outcomes to observed action trajectories (read-only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
from collections import Counter, defaultdict
from pathlib import Path

from training.audit_value_training_coverage import restore_encoded_state
from quoridor.neural_ai.encoder import apply_action_index, decode_action, legal_action_indices


EXPECTED_DATASET_SHA256 = "d9e027c3e4537146dcfa7368baee0ccc01be4fbc067ff01f36fc50f859437b9"


def fingerprint(encoded: list) -> str:
    h = hashlib.sha256()
    def walk(x):
        if isinstance(x, list):
            for v in x:
                walk(v)
        else:
            v = float(x)
            if not math.isfinite(v):
                raise ValueError("Non-finite state input")
            h.update(struct.pack("<f", v))
    walk(encoded)
    return h.hexdigest()


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


def reconstruct_observed_action(
    state,
    *,
    expected_next=None,
    expected_winner=None,
    expected_terminal_ply=None,
) -> tuple[int, object]:
    """Infer the unique legal action proven by the next state or terminal metadata."""
    matches = []
    for action_index in legal_action_indices(state):
        child = apply_action_index(state, action_index)
        if expected_next is not None:
            matched = same_state(child, expected_next)
        else:
            matched = (
                child.winner == expected_winner
                and child.ply == expected_terminal_ply
                and child.winner is not None
            )
        if matched:
            matches.append((action_index, child))

    if len(matches) != 1:
        location = (
            f"next ply {expected_next.ply}"
            if expected_next is not None
            else f"terminal winner={expected_winner}, ply={expected_terminal_ply}"
        )
        candidate_ids = [action for action, _ in matches]
        raise ValueError(
            f"Could not uniquely reconstruct action at game {state.ply}: "
            f"expected {location}; matched {len(matches)} legal actions "
            f"{candidate_ids[:10]}"
        )
    return matches[0]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path,
                   default=Path("data/experiments/exp_001/heuristic_decisive_games.jsonl"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/diagnostics/trajectory_outcome_attribution/run_001"))
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = p.parse_args()
    if not args.dataset.is_file():
        p.error(f"Dataset missing: {args.dataset}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error(f"Output directory is not empty: {args.output_dir}")
    if sorted(set(args.horizons)) != sorted(args.horizons) or any(x < 1 for x in args.horizons):
        p.error("Horizons must be unique positive integers in ascending order.")

    raw_bytes = args.dataset.read_bytes()
    dataset_sha = hashlib.sha256(raw_bytes).hexdigest()
    if dataset_sha != EXPECTED_DATASET_SHA256:
        raise ValueError("Dataset SHA-256 does not match pinned EXP-001 artifact.")

    games = defaultdict(list)
    for line_no, line in enumerate(raw_bytes.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        required = {"state", "game_id", "ply", "player", "winner", "total_plies"}
        if required - r.keys():
            raise ValueError(f"Line {line_no}: missing {sorted(required-r.keys())}")
        state = restore_encoded_state({"state": r["state"], "player": r["player"], "ply": r["ply"]})
        if state.current_player != int(r["player"]):
            raise ValueError(f"Line {line_no}: player mismatch")
        games[int(r["game_id"])].append({
            "state": state, "encoded": r["state"], "fp": fingerprint(r["state"]),
            "game_id": int(r["game_id"]), "ply": int(r["ply"]),
            "player": int(r["player"]), "winner": int(r["winner"]),
            "total_plies": int(r["total_plies"]),
            "target": 1 if int(r["player"]) == int(r["winner"]) else -1,
            "action": None,
        })

    all_rows = []
    conflict_groups = defaultdict(list)
    transition_checks = 0
    for game_id, rows in games.items():
        rows.sort(key=lambda x: x["ply"])
        if [r["ply"] for r in rows] != list(range(rows[0]["total_plies"])):
            raise ValueError(f"Game {game_id}: incomplete/non-contiguous trajectory")
        winner = rows[0]["winner"]
        total = rows[0]["total_plies"]
        if any(r["winner"] != winner or r["total_plies"] != total for r in rows):
            raise ValueError(f"Game {game_id}: inconsistent game metadata")
        for i, row in enumerate(rows):
            if i + 1 < len(rows):
                action, child = reconstruct_observed_action(
                    row["state"], expected_next=rows[i + 1]["state"]
                )
            else:
                action, child = reconstruct_observed_action(
                    row["state"], expected_winner=winner,
                    expected_terminal_ply=total,
                )
            row["action"] = action
            transition_checks += 1
            if i + 1 == len(rows) and (
                child.ply != total or child.winner != winner
            ):
                raise ValueError(
                    f"Game {game_id}: reconstructed final action does not produce "
                    "recorded winner/terminal ply"
                )
        for i, row in enumerate(rows):
            row["trajectory_actions"] = [x["action"] for x in rows[i:]]
            all_rows.append(row)
    by_fp = defaultdict(list)
    for row in all_rows:
        by_fp[row["fp"]].append(row)
    conflict_groups = {fp: sorted(rs, key=lambda x: (x["game_id"], x["ply"]))
                       for fp, rs in by_fp.items()
                       if len({x["target"] for x in rs}) > 1}
    if len(conflict_groups) != 52 or sum(map(len, conflict_groups.values())) != 1987:
        raise ValueError("Frozen conflict cohort mismatch; expected 52 states / 1,987 occurrences.")

    per_state = []
    pairwise = Counter()
    action_rows = []
    for fp, rows in sorted(conflict_groups.items()):
        pos = [r for r in rows if r["target"] == 1]
        neg = [r for r in rows if r["target"] == -1]
        pos_actions = Counter(r["action"] for r in pos)
        neg_actions = Counter(r["action"] for r in neg)
        shared = sorted(set(pos_actions) & set(neg_actions))
        all_actions = sorted(set(pos_actions) | set(neg_actions))
        for a in all_actions:
            action_rows.append({
                "fingerprint": fp, "action_index": a,
                "action": action_payload(a),
                "positive_count": pos_actions[a], "negative_count": neg_actions[a],
                "positive_target_rate": pos_actions[a] / (pos_actions[a] + neg_actions[a]),
                "support": pos_actions[a] + neg_actions[a],
            })
        first_divergence = Counter()
        common_prefix_lengths = []
        total_pairs = 0
        same_first_action_pairs = 0
        same_first_action_different_outcome_pairs = 0
        for a in pos:
            for b in neg:
                total_pairs += 1
                n = min(len(a["trajectory_actions"]), len(b["trajectory_actions"]))
                common = 0
                while common < n and a["trajectory_actions"][common] == b["trajectory_actions"][common]:
                    common += 1
                common_prefix_lengths.append(common)
                if common == 0:
                    first_divergence["at_root_action"] += 1
                elif common < n:
                    first_divergence[f"after_{common}_shared_actions"] += 1
                else:
                    first_divergence["no_divergence_within_shorter_trajectory"] += 1
                if a["action"] == b["action"]:
                    same_first_action_pairs += 1
                    if common >= 1 and common < n:
                        same_first_action_different_outcome_pairs += 1
        horizon = {}
        for h in args.horizons:
            pos_seq = Counter(tuple(r["trajectory_actions"][:h]) for r in pos)
            neg_seq = Counter(tuple(r["trajectory_actions"][:h]) for r in neg)
            horizon[str(h)] = {
                "shared_sequence_count": len(set(pos_seq) & set(neg_seq)),
                "positive_unique_sequences": len(pos_seq),
                "negative_unique_sequences": len(neg_seq),
                "cross_outcome_shared_occurrence_pairs": sum(
                    pos_seq[s] * neg_seq[s] for s in set(pos_seq) & set(neg_seq)
                ),
            }
        per_state.append({
            "fingerprint": fp, "occurrences": len(rows),
            "positive_occurrences": len(pos), "negative_occurrences": len(neg),
            "distinct_root_actions": len(all_actions),
            "shared_root_action_count": len(shared),
            "shared_root_actions": [action_payload(a) for a in shared],
            "positive_root_action_distribution": [
                {"action_index": a, "action": action_payload(a), "count": c}
                for a, c in sorted(pos_actions.items())
            ],
            "negative_root_action_distribution": [
                {"action_index": a, "action": action_payload(a), "count": c}
                for a, c in sorted(neg_actions.items())
            ],
            "cross_outcome_occurrence_pairs": total_pairs,
            "same_root_action_pair_count": same_first_action_pairs,
            "same_root_action_then_divergent_trajectory_pair_count": same_first_action_different_outcome_pairs,
            "common_prefix_action_count": {
                "mean": statistics.mean(common_prefix_lengths) if common_prefix_lengths else None,
                "median": statistics.median(common_prefix_lengths) if common_prefix_lengths else None,
                "histogram": dict(sorted(Counter(common_prefix_lengths).items())),
            },
            "first_divergence_pairs": dict(sorted(first_divergence.items())),
            "trajectory_sequence_overlap_by_horizon": horizon,
        })
        pairwise["pairs"] += total_pairs
        pairwise["same_root_action_pairs"] += same_first_action_pairs
        pairwise["same_root_action_then_divergent_pairs"] += same_first_action_different_outcome_pairs

    summary = {
        "experiment": "trajectory_outcome_attribution_audit",
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_sha,
        "integrity": {
            "games": len(games), "records": len(all_rows),
            "conflicting_states": len(conflict_groups),
            "conflict_occurrences": sum(len(v) for v in conflict_groups.values()),
            "verified_state_transitions": transition_checks,
            "all_reconstructed_actions_legal": True,
            "all_transitions_match_next_state": True,
            "all_terminal_winners_verified": True,
            "unique_action_reconstruction_required": True,
            "ambiguous_or_unmatched_transitions": 0,
        },
        "method": {
            "observed_action_source": (
                "uniquely reconstructed from legal state transitions; the pinned JSONL "
                "does not contain policy or action fields"
            ),
            "trajectory": "chronological actions from each repeated state occurrence to game end",
            "target": " +1 when side-to-move eventually won, otherwise -1",
            "pairing": "all positive-target occurrences paired with all negative-target occurrences within each exact encoded state",
            "horizons": args.horizons,
            "read_only": True,
        },
        "cross_outcome_pair_summary": {
            "total_pairs": pairwise["pairs"],
            "same_root_action_pairs": pairwise["same_root_action_pairs"],
            "same_root_action_then_divergent_pairs": pairwise["same_root_action_then_divergent_pairs"],
        },
        "action_conditioned_outcomes": action_rows,
        "per_state": per_state,
        "interpretation_limits": [
            "This is observational attribution, not a counterfactual intervention: actions were not changed and games were not re-simulated.",
            "Actions are inferred from the unique legal transition to the next recorded state; the final action is inferred from terminal winner and ply metadata. This is not a direct read of a stored policy action.",
            "Every reconstructed transition must match exactly one legal action. Zero-match or multi-match transitions abort the audit.",
            "The root action is verified to produce the next recorded state; later action divergence identifies where observed trajectories separate, not which action caused the eventual winner.",
            "A root action associated with both outcomes is not necessarily equally strong; report support and outcome counts and avoid causal claims from small samples.",
            "The value target is terminal winner from the current player's perspective, so all occurrences of an exact state have a fixed side-to-move but may have different eventual winners.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(
        {k: v for k, v in summary.items() if k not in ("per_state", "action_conditioned_outcomes")}, indent=2
    ) + "\n", encoding="utf-8")
    (args.output_dir / "trajectory_outcome_attribution.json").write_text(
        json.dumps(summary, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"integrity": summary["integrity"],
                      "cross_outcome_pair_summary": summary["cross_outcome_pair_summary"],
                      "states_with_shared_root_action": sum(x["shared_root_action_count"] > 0 for x in per_state),
                      "states_without_shared_root_action": sum(x["shared_root_action_count"] == 0 for x in per_state)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
