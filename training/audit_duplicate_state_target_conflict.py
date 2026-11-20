#!/usr/bin/env python3
"""Audit duplicate encoded states, contradictory value targets, and state-grouped splits.

Read-only: no training, checkpoint writes, or dataset mutation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from training.audit_value_training_coverage import (
    EXPECTED_CHECKPOINT_SHA256,
    label_position,
    restore_encoded_state,
)
from training.train_value_only_strict import allocate_counts, split_by_game
from quoridor.neural_ai.policy import load_model


def fingerprint(state: list) -> str:
    """Hash the exact float32 tensor representation consumed by the network."""
    digest = hashlib.sha256()

    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
        else:
            value = float(node)
            if not (-float("inf") < value < float("inf")):
                raise ValueError("State contains a non-finite value")
            digest.update(struct.pack("<f", value))

    visit(state)
    return digest.hexdigest()


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    errors = [r["prediction"] - r["target"] for r in rows]
    return {
        "count": len(rows),
        "unique_states": len({r["fingerprint"] for r in rows}),
        "games": len({r["game_id"] for r in rows}),
        "mse": mean(e * e for e in errors),
        "mae": mean(abs(e) for e in errors),
        "sign_accuracy": mean(
            (r["prediction"] >= 0) == (r["target"] > 0) for r in rows
        ),
        "target_positive": sum(r["target"] > 0 for r in rows),
        "target_negative": sum(r["target"] < 0 for r in rows),
        "prediction_mean": mean(r["prediction"] for r in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
        default=Path("data/experiments/exp_001/heuristic_decisive_games.jsonl"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
        default=Path("data/diagnostics/duplicate_state_target_conflict/run_001"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261113003)
    parser.add_argument("--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256)
    args = parser.parse_args()

    if not args.dataset.is_file() or not args.checkpoint.is_file():
        parser.error("Dataset and checkpoint must exist")
    if min(args.batch_size, args.threads) < 1:
        parser.error("batch-size and threads must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")

    import torch
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    dataset_bytes = args.dataset.read_bytes()
    dataset_sha = hashlib.sha256(dataset_bytes).hexdigest()
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.expected_checkpoint_sha256 and checkpoint_sha != args.expected_checkpoint_sha256:
        parser.error(f"Checkpoint SHA-256 mismatch: {checkpoint_sha}")

    rows = []
    game_meta = {}
    seen_game_ply = set()
    for line_no, line in enumerate(dataset_bytes.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        gid, ply = int(raw["game_id"]), int(raw["ply"])
        player, winner = int(raw["player"]), int(raw["winner"])
        if player not in (0, 1) or winner not in (0, 1):
            raise ValueError(f"Invalid player/winner at line {line_no}")
        if (gid, ply) in seen_game_ply:
            raise ValueError(f"Duplicate game_id/ply at line {line_no}: {(gid, ply)}")
        seen_game_ply.add((gid, ply))
        meta = (winner, int(raw.get("total_plies", -1)))
        if gid in game_meta and game_meta[gid] != meta:
            raise ValueError(f"Inconsistent metadata for game {gid}")
        game_meta[gid] = meta
        target = 1 if player == winner else -1
        fp = fingerprint(raw["state"])
        rows.append({
            "state": raw["state"], "fingerprint": fp, "game_id": gid,
            "ply": ply, "player": player, "winner": winner,
            "target": target, "cohort": None, "prediction": None,
        })

    groups = defaultdict(list)
    for row in rows:
        groups[row["fingerprint"]].append(row)

    # Validate the existing split and identify its exact-state leakage.
    old_game_split = split_by_game(game_meta, args.seed)
    old_gid_to_split = {gid: split for split, ids in old_game_split.items() for gid in ids}
    old_state_splits = defaultdict(set)
    for row in rows:
        old_state_splits[row["fingerprint"]].add(old_gid_to_split[row["game_id"]])

    conflict_groups = []
    group_summaries = []
    for fp, members in groups.items():
        targets = Counter(r["target"] for r in members)
        cohorts = Counter()
        games = sorted({r["game_id"] for r in members})
        splits = sorted({old_gid_to_split[g] for g in games})
        group = {
            "fingerprint": fp, "record_count": len(members),
            "game_count": len(games), "game_ids": games,
            "target_counts": dict(targets),
            "target_conflict": len(targets) > 1,
            "cohort_counts": dict(cohorts), "old_splits": splits,
            "cross_split_leakage": len(splits) > 1,
        }
        group_summaries.append(group)
        if group["target_conflict"]:
            conflict_groups.append(group)

    # Rebuild every state and compute its tactical cohort before inference.
    states = []
    for row in rows:
        state = restore_encoded_state({
            "state": row["state"], "player": row["player"], "ply": row["ply"]
        })
        label = label_position(state)
        row["cohort"] = label["cohort"]
        row["forced_loss"] = bool(label["forced_loss"])
        states.append(state)
    # Cohort labels are now populated for every row; aggregate conflicts only
    # after this point so the cohort breakdown cannot silently remain empty.
    group_by_fp = {item["fingerprint"]: item for item in group_summaries}
    conflict_by_cohort = defaultdict(Counter)
    for fp, members in groups.items():
        item = group_by_fp[fp]
        cohort_counts = Counter(r["cohort"] for r in members)
        item["cohort_counts"] = dict(cohort_counts)
        item["forced_loss_record_count"] = sum(r["forced_loss"] for r in members)
        if item["target_conflict"]:
            for cohort, count in cohort_counts.items():
                conflict_by_cohort[cohort]["conflict_state_groups"] += 1
                conflict_by_cohort[cohort]["records"] += count

    model = load_model(args.checkpoint, device=args.device)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(states), args.batch_size):
            batch_states = states[start:start + args.batch_size]
            from quoridor.neural_ai.encoder import encode_state
            batch = torch.tensor(
                [encode_state(s) for s in batch_states],
                dtype=torch.float32, device=args.device,
            )
            _, values = model(batch)
            predictions.extend(float(v) for v in values.detach().cpu().tolist())
    if len(predictions) != len(rows):
        raise RuntimeError("Prediction count mismatch")
    for row, prediction in zip(rows, predictions):
        row["prediction"] = prediction

    # Assign each unique fingerprint wholly to one partition. This removes
    # exact encoded-state leakage. Games may span partitions; report that fact.
    rng = random.Random(args.seed)
    unique_fps = sorted(groups)
    rng.shuffle(unique_fps)
    n_train, n_val, n_test = allocate_counts(len(unique_fps))
    fp_split = {}
    for split, selected in (
        ("train", unique_fps[:n_train]),
        ("validation", unique_fps[n_train:n_train + n_val]),
        ("test", unique_fps[n_train + n_val:n_train + n_val + n_test]),
    ):
        fp_split.update({fp: split for fp in selected})
    for row in rows:
        row["state_group_split"] = fp_split[row["fingerprint"]]

    split_rows = {}
    for split in ("train", "validation", "test"):
        selected = [r for r in rows if r["state_group_split"] == split]
        split_rows[split] = {
            "records": metrics(selected),
            "cohort_counts": dict(Counter(r["cohort"] for r in selected)),
            "unique_state_groups": len({r["fingerprint"] for r in selected}),
            "games": len({r["game_id"] for r in selected}),
        }
    game_partitions = defaultdict(set)
    for row in rows:
        game_partitions[row["game_id"]].add(row["state_group_split"])

    # The checkpoint was trained on the original train-game partition. A new
    # random state split is not a held-out evaluation if its validation/test
    # states already occurred in that original training partition.
    old_train_fingerprints = {
        row["fingerprint"] for row in rows
        if old_gid_to_split[row["game_id"]] == "train"
    }
    state_grouped_overlap_with_original_training = {}
    for split in ("train", "validation", "test"):
        selected_fps = {
            fp for fp, assigned_split in fp_split.items() if assigned_split == split
        }
        overlap = selected_fps & old_train_fingerprints
        state_grouped_overlap_with_original_training[split] = {
            "unique_state_groups": len(selected_fps),
            "overlap_with_original_training_state_groups": len(overlap),
            "overlap_pct": 100.0 * len(overlap) / len(selected_fps) if selected_fps else None,
        }

    cohort_summaries = {}
    for cohort in sorted({r["cohort"] for r in rows}):
        selected = [r for r in rows if r["cohort"] == cohort]
        cohort_summaries[cohort] = {
            "records": metrics(selected),
            "unique_states": len({r["fingerprint"] for r in selected}),
            "conflicting_state_groups": sum(
                1 for g in conflict_groups
                if any(r["fingerprint"] == g["fingerprint"] and r["cohort"] == cohort for r in selected)
            ),
        }

    state_level = []
    for fp, members in groups.items():
        target_mean = mean(r["target"] for r in members)
        prediction_mean = mean(r["prediction"] for r in members)
        majority_target = 1 if sum(r["target"] > 0 for r in members) >= sum(r["target"] < 0 for r in members) else -1
        state_level.append({
            "fingerprint": fp, "record_count": len(members),
            "target_mean": target_mean, "majority_target": majority_target,
            "prediction": prediction_mean, "target_conflict": len({r["target"] for r in members}) > 1,
            "cohort_counts": dict(Counter(r["cohort"] for r in members)),
            "split": fp_split[fp],
        })

    old_metrics = {}
    for split, gids in old_game_split.items():
        selected = [r for r in rows if r["game_id"] in set(gids)]
        old_metrics[split] = {
            "records": metrics(selected),
            "cohort_counts": dict(Counter(r["cohort"] for r in selected)),
            "games": len(gids),
            "unique_states": len({r["fingerprint"] for r in selected}),
        }

    state_level_by_split = {}
    for split in ("train", "validation", "test"):
        selected_states = [r for r in state_level if r["split"] == split]
        if selected_states:
            state_level_by_split[split] = {
                "unique_states": len(selected_states),
                "mse_vs_mean_target": mean((r["prediction"] - r["target_mean"]) ** 2 for r in selected_states),
                "mae_vs_mean_target": mean(abs(r["prediction"] - r["target_mean"]) for r in selected_states),
                "sign_accuracy_vs_majority_target": mean(
                    (r["prediction"] >= 0) == (r["majority_target"] > 0)
                    for r in selected_states
                ),
                "conflicting_state_count": sum(r["target_conflict"] for r in selected_states),
                "cohort_counts": dict(Counter(
                    cohort for r in selected_states for cohort, count in r["cohort_counts"].items()
                    for _ in range(count)
                )),
            }

    report = {
        "experiment": "duplicate_state_target_conflict_audit",
        "dataset": str(args.dataset), "dataset_sha256": dataset_sha,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha,
        "config": {"seed": args.seed, "state_group_split": "80/10/10 by unique encoded-state fingerprint"},
        "integrity": {
            "records": len(rows), "games": len(game_meta),
            "unique_state_fingerprints": len(groups),
            "duplicate_records": len(rows) - len(groups),
            "duplicate_record_pct": 100.0 * (len(rows) - len(groups)) / len(rows),
            "duplicate_game_ply_count": 0,
        },
        "target_conflicts": {
            "conflicting_state_group_count": len(conflict_groups),
            "conflicting_record_count": sum(g["record_count"] for g in conflict_groups),
            "conflict_groups_by_cohort": dict(conflict_by_cohort),
            "examples": sorted(conflict_groups, key=lambda g: (-g["record_count"], g["fingerprint"]))[:100],
        },
        "original_game_split": {
            "metrics": old_metrics,
            "state_groups_spanning_splits": sum(len(v) > 1 for v in old_state_splits.values()),
            "records_in_state_groups_spanning_splits": sum(
                len(groups[fp]) for fp, splits in old_state_splits.items() if len(splits) > 1
            ),
            "unique_state_groups_spanning_splits": sum(len(v) > 1 for v in old_state_splits.values()),
        },
        "state_grouped_split": {
            "metrics": split_rows,
            "overlap_with_original_training": state_grouped_overlap_with_original_training,
            "games_spanning_splits": sum(len(v) > 1 for v in game_partitions.values()),
            "game_split_overlap_note": "Games may span partitions; exact encoded states never do. A fully game-disjoint split is impossible when a shared state such as the initial position connects all games.",
            "unique_state_group_count_by_split": {
                split: len({r["fingerprint"] for r in rows if r["state_group_split"] == split})
                for split in ("train", "validation", "test")
            },
        },
        "cohort_summary": cohort_summaries,
        "state_level_metrics": {
            "by_split": state_level_by_split,
            "count": len(state_level),
            "mse_vs_mean_target": mean((r["prediction"] - r["target_mean"]) ** 2 for r in state_level),
            "mae_vs_mean_target": mean(abs(r["prediction"] - r["target_mean"]) for r in state_level),
            "sign_accuracy_vs_majority_target": mean(
                (r["prediction"] >= 0) == (r["majority_target"] > 0) for r in state_level
            ),
            "conflict_state_count": sum(r["target_conflict"] for r in state_level),
        },
        "state_groups": group_summaries,
        "state_level_rows": state_level,
        "records": [{
            k: v for k, v in row.items() if k != "state"
        } for row in rows],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(
        {k: v for k, v in report.items() if k not in ("records", "state_groups", "state_level_rows")},
        indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "duplicate_state_target_conflict.json").write_text(
        json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"records={len(rows)} games={len(game_meta)} unique_states={len(groups)}")
    print(f"conflicting_state_groups={len(conflict_groups)}")
    print(f"old_split_state_leaks={report['original_game_split']['state_groups_spanning_splits']}")
    print(f"new_split={json.dumps(split_rows, sort_keys=True)}")
    print(f"outputs={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
