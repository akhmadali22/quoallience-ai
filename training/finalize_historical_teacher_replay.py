#!/usr/bin/env python3
"""Finalize and classify the exhaustive historical EXP-001 teacher replay."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HISTORICAL_COMMIT = "f40a4ce3df337939cd52a16108b281d4b1e6f4d4"
EXPECTED_DATASET_SHA256 = "d9e027c3e4537146dcfa7368baee0ccc01be4fbc067bff01f36fc50f859437b9"
EXPECTED_SOURCE_SHA256 = {
    "quoridor/advanced_ai.py": "83a4ef15847d3f6307a65890e8be4263035a5871637682bff0595848d90d2db6",
    "quoridor/engine.py": "5478cdf66537556a5b32c445ea24ddda0aa4c8a52c7b4597af9311f626090e9a",
    "training/generate_heuristic_games.py": "8f503401dafc4c5c8a4db22449973032e043434a0e851887f2f3d8a53e798c96",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.raw_dir / "summary.json"
    detail_path = args.raw_dir / "teacher_state_dependence.json"
    if not summary_path.is_file() or not detail_path.is_file():
        parser.error("Raw teacher_state_dependence audit outputs are missing.")
    if args.output_dir.exists() and any((args.output_dir / name).exists() for name in ("summary.json", "historical_teacher_replay.json")):
        parser.error(f"Final output files already exist in: {args.output_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    integrity = summary["integrity"]
    source_hashes = summary["current_source_sha256"]

    provenance_checks = {
        "dataset_sha256_matches_exp001": summary["dataset_sha256"] == EXPECTED_DATASET_SHA256,
        "historical_source_hashes_match": all(
            source_hashes.get(path) == expected
            for path, expected in EXPECTED_SOURCE_SHA256.items()
        ),
        "historical_generation_commit": HISTORICAL_COMMIT,
        "dataset_generation_config": {
            "games": 50,
            "time_limit_seconds": 0.08,
            "max_depth": 2,
            "max_plies": 300,
            "discount": 0.99,
            "seed": 20261113,
            "decisive_games": 50,
            "total_samples": 3780,
        },
    }
    if not provenance_checks["dataset_sha256_matches_exp001"]:
        raise ValueError("Dataset SHA-256 does not match the EXP-001 historical artifact.")
    if not provenance_checks["historical_source_hashes_match"]:
        raise ValueError(
            "Current source hashes do not match the historical EXP-001 source hashes. "
            "Do not interpret this as an exact historical replay."
        )

    occurrences = detail["occurrences"]
    by_state = defaultdict(list)
    for row in occurrences:
        by_state[row["fingerprint"]].append(row)

    dataset_occurrence_counts = {
        row["fingerprint"]: row["dataset_occurrences"]
        for row in detail["per_state"]
    }
    per_state = []
    state_lookup = {}
    for fp, members in sorted(by_state.items()):
        positive_actions = {
            json.dumps(row["runs"]["actual_short"]["action"], sort_keys=True)
            for row in members if row["target"] == 1
        }
        negative_actions = {
            json.dumps(row["runs"]["actual_short"]["action"], sort_keys=True)
            for row in members if row["target"] == -1
        }
        history_short = sum(
            row["comparisons"]["history_changes_action_short"] for row in members
        )
        history_long = sum(
            row["comparisons"]["history_changes_action_long"] for row in members
        )
        budget_empty = sum(
            row["comparisons"]["short_vs_long_action_empty_history"] for row in members
        )
        budget_actual = sum(
            row["comparisons"]["short_vs_long_action_actual_history"] for row in members
        )
        timeout_short = sum(
            row["comparisons"]["short_timeout_empty_history"]
            or row["comparisons"]["short_timeout_actual_history"]
            for row in members
        )
        same_action_both_targets = bool(positive_actions & negative_actions)
        record = {
            "fingerprint": fp,
            "dataset_occurrences": dataset_occurrence_counts[fp],
            "audited_occurrences": len(members),
            "target_counts_in_audited_occurrences": {
                "positive": sum(row["target"] == 1 for row in members),
                "negative": sum(row["target"] == -1 for row in members),
            },
            "history_action_changes_80ms": history_short,
            "history_action_changes_320ms": history_long,
            "budget_action_changes_empty_history": budget_empty,
            "budget_action_changes_actual_history": budget_actual,
            "short_budget_timeout_occurrences": timeout_short,
            "same_actual_80ms_action_seen_under_both_targets": same_action_both_targets,
            "shared_action_count_across_targets": len(positive_actions & negative_actions),
            "classification": {
                "history_influenced": bool(history_short or history_long),
                "deadline_sensitive": bool(budget_empty or budget_actual),
                "same_action_opposite_outcomes": same_action_both_targets,
                "no_shared_action_across_outcomes": not same_action_both_targets,
            },
        }
        per_state.append(record)
        state_lookup[fp] = record

    for row in occurrences:
        state_class = state_lookup[row["fingerprint"]]["classification"]
        row["historical_replay_classification"] = {
            "history_changes_action_at_80ms": row["comparisons"]["history_changes_action_short"],
            "history_changes_action_at_320ms": row["comparisons"]["history_changes_action_long"],
            "deadline_changes_action_without_history": row["comparisons"]["short_vs_long_action_empty_history"],
            "deadline_changes_action_with_actual_history": row["comparisons"]["short_vs_long_action_actual_history"],
            "short_budget_timed_out": (
                row["comparisons"]["short_timeout_empty_history"]
                or row["comparisons"]["short_timeout_actual_history"]
            ),
            "same_actual_80ms_action_occurs_under_opposite_target": (
                state_class["same_action_opposite_outcomes"]
            ),
        }

    def count_state(predicate):
        return sum(bool(predicate(row)) for row in per_state)

    def count_occ(predicate):
        return sum(bool(predicate(row)) for row in occurrences)

    classification_summary = {
        "state_counts": {
            "total_conflicting_states": len(per_state),
            "history_influenced": count_state(lambda x: x["classification"]["history_influenced"]),
            "deadline_sensitive": count_state(lambda x: x["classification"]["deadline_sensitive"]),
            "same_action_opposite_outcomes": count_state(lambda x: x["classification"]["same_action_opposite_outcomes"]),
            "no_shared_action_across_outcomes": count_state(lambda x: x["classification"]["no_shared_action_across_outcomes"]),
        },
        "occurrence_counts": {
            "total_audited_occurrences": len(occurrences),
            "history_changes_action_at_80ms": count_occ(lambda x: x["comparisons"]["history_changes_action_short"]),
            "history_changes_action_at_320ms": count_occ(lambda x: x["comparisons"]["history_changes_action_long"]),
            "deadline_changes_action_without_history": count_occ(lambda x: x["comparisons"]["short_vs_long_action_empty_history"]),
            "deadline_changes_action_with_actual_history": count_occ(lambda x: x["comparisons"]["short_vs_long_action_actual_history"]),
            "empty_history_80ms_timeout": count_occ(lambda x: x["comparisons"]["short_timeout_empty_history"]),
            "actual_history_80ms_timeout": count_occ(lambda x: x["comparisons"]["short_timeout_actual_history"]),
        },
    }

    if integrity["conflicting_states"] != 52 or integrity["conflict_occurrences"] != 1987:
        raise ValueError(
            "Unexpected frozen cohort size: expected 52 states / 1987 occurrences, "
            f"got {integrity['conflicting_states']} / {integrity['conflict_occurrences']}."
        )
    if integrity["selected_occurrences"] != 1987:
        raise ValueError(
            f"Audit is not exhaustive: selected {integrity['selected_occurrences']} / 1987."
        )

    report = {
        **detail,
        "experiment": "historical_teacher_replay_audit",
        "historical_provenance": provenance_checks,
        "exhaustive_classification": classification_summary,
        "per_state": per_state,
        "interpretation_limits": [
            "The historical Advanced AI, engine, encoder, and generator files were compared byte-for-byte against commit " + HISTORICAL_COMMIT + "; the three source hashes recorded by the replay also match the pinned EXP-001 source hashes.",
            "History-sensitive means the same board can produce a different selected action when reconstructed same-player history is supplied. It does not prove that history alone caused the final winner.",
            "Deadline-sensitive means the selected action changes between the original 80 ms budget and 320 ms at the same max_depth=2. This is not a deeper-search comparison.",
            "Same action under opposite targets demonstrates that the observed action and current board alone do not uniquely determine the terminal outcome across these trajectories.",
            "Classifications are non-exclusive. A state can be history-sensitive, deadline-sensitive, and still have the same action under both target classes.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "occurrences"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "historical_teacher_replay.json").write_text(
        json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "conflicting_states": integrity["conflicting_states"],
        "conflict_occurrences": integrity["conflict_occurrences"],
        "audited_occurrences": integrity["selected_occurrences"],
        "classification": classification_summary,
        "historical_provenance": provenance_checks,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
