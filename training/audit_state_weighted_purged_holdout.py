#!/usr/bin/env python3
"""State-weighted dataset and purged game-holdout audit.

Read-only diagnostic. No training, optimizer, checkpoint writes, or dataset mutation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import struct
from collections import Counter, defaultdict
from pathlib import Path

from training.audit_value_training_coverage import (
    EXPECTED_CHECKPOINT_SHA256,
    label_position,
    restore_encoded_state,
)
from training.train_value_only_strict import split_by_game
from quoridor.neural_ai.encoder import encode_state
from quoridor.neural_ai.policy import load_model


def fingerprint(state: list) -> str:
    """Hash the exact float32 values consumed by the network."""
    digest = hashlib.sha256()

    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
        else:
            value = float(node)
            if not math.isfinite(value):
                raise ValueError("State contains a non-finite value")
            digest.update(struct.pack("<f", value))

    visit(state)
    return digest.hexdigest()


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def frequency_summary(groups: dict[str, list[dict]], total_records: int) -> dict:
    frequencies = [len(rows) for rows in groups.values()]
    ranked = sorted(frequencies, reverse=True)

    def top_share(fraction: float) -> float:
        count = max(1, math.ceil(len(ranked) * fraction))
        return 100.0 * sum(ranked[:count]) / total_records

    # Standard Gini coefficient over state frequencies.
    ordered = sorted(frequencies)
    n = len(ordered)
    total = sum(ordered)
    weighted = sum((i + 1) * value for i, value in enumerate(ordered))
    gini = (2.0 * weighted) / (n * total) - (n + 1.0) / n if n and total else 0.0

    return {
        "unique_states": len(frequencies),
        "records": total_records,
        "records_per_state": {
            "min": min(frequencies),
            "p25": quantile(frequencies, 0.25),
            "median": quantile(frequencies, 0.50),
            "mean": statistics.mean(frequencies),
            "p75": quantile(frequencies, 0.75),
            "p90": quantile(frequencies, 0.90),
            "p95": quantile(frequencies, 0.95),
            "p99": quantile(frequencies, 0.99),
            "max": max(frequencies),
        },
        "states_with_frequency": {
            str(k): sum(value == k for value in frequencies)
            for k in sorted(set(frequencies))
        },
        "top_frequency_state_record_share_pct": {
            "top_1pct_states": top_share(0.01),
            "top_5pct_states": top_share(0.05),
            "top_10pct_states": top_share(0.10),
        },
        "frequency_gini": gini,
        "effective_sample_size_if_each_record_weight_is_inverse_frequency": float(len(groups)),
        "max_state_record_weight_multiple_vs_singleton": max(frequencies),
    }


def loss_metrics(rows: list[dict]) -> dict:
    """Compare record-weighted, state-weighted, and aggregated-target objectives."""
    if not rows:
        return {"record_count": 0, "unique_states": 0}

    by_fp = defaultdict(list)
    for row in rows:
        by_fp[row["fingerprint"]].append(row)

    record_sq = [(r["prediction"] - r["target"]) ** 2 for r in rows]
    record_abs = [abs(r["prediction"] - r["target"]) for r in rows]
    record_sign = [
        (r["prediction"] >= 0) == (r["target"] > 0) for r in rows
    ]

    per_state = []
    for fp, members in by_fp.items():
        target_mean = statistics.mean(r["target"] for r in members)
        prediction_mean = statistics.mean(r["prediction"] for r in members)
        target_var = statistics.pvariance(r["target"] for r in members)
        per_state.append({
            "record_mse": statistics.mean((r["prediction"] - r["target"]) ** 2 for r in members),
            "record_mae": statistics.mean(abs(r["prediction"] - r["target"]) for r in members),
            "record_sign": statistics.mean(
                (r["prediction"] >= 0) == (r["target"] > 0) for r in members
            ),
            "aggregated_mse": (prediction_mean - target_mean) ** 2,
            "aggregated_mae": abs(prediction_mean - target_mean),
            "aggregated_sign": (prediction_mean >= 0) == (target_mean > 0),
            "target_mean": target_mean,
            "target_variance": target_var,
        })

    return {
        "record_count": len(rows),
        "unique_states": len(by_fp),
        "record_weighted": {
            "mse": statistics.mean(record_sq),
            "mae": statistics.mean(record_abs),
            "sign_accuracy": statistics.mean(record_sign),
        },
        "state_weighted_preserve_each_record_target": {
            "mse": statistics.mean(x["record_mse"] for x in per_state),
            "mae": statistics.mean(x["record_mae"] for x in per_state),
            "sign_accuracy": statistics.mean(x["record_sign"] for x in per_state),
        },
        "state_weighted_aggregated_target": {
            "mse_vs_mean_target": statistics.mean(x["aggregated_mse"] for x in per_state),
            "mae_vs_mean_target": statistics.mean(x["aggregated_mae"] for x in per_state),
            "sign_accuracy_vs_majority_target": statistics.mean(x["aggregated_sign"] for x in per_state),
        },
        "within_state_target_variance": {
            "state_weighted_mean": statistics.mean(x["target_variance"] for x in per_state),
            "record_weighted_mean": sum(
                len(members) * statistics.pvariance(r["target"] for r in members)
                for members in by_fp.values()
            ) / len(rows),
            "mse_decomposition_check": (
                statistics.mean(x["record_mse"] for x in per_state)
                - statistics.mean(x["aggregated_mse"] for x in per_state)
            ),
        },
    }


def target_summary(members: list[dict]) -> dict:
    targets = [r["target"] for r in members]
    positives = sum(y > 0 for y in targets)
    negatives = sum(y < 0 for y in targets)
    mean_target = statistics.mean(targets)
    variance = statistics.pvariance(targets)
    return {
        "record_count": len(members),
        "game_count": len({r["game_id"] for r in members}),
        "positive_count": positives,
        "negative_count": negatives,
        "positive_pct": 100.0 * positives / len(members),
        "mean_target": mean_target,
        "target_variance": variance,
        "target_stddev": math.sqrt(variance),
        "outcome_entropy_bits": (
            -(positives / len(members)) * math.log2(positives / len(members))
            if positives else 0.0
        ) + (
            -(negatives / len(members)) * math.log2(negatives / len(members))
            if negatives else 0.0
        ),
    }


def summarize_groups(groups: dict[str, list[dict]]) -> dict:
    rows = list(groups.values())
    conflict = [members for members in rows if len({r["target"] for r in members}) > 1]
    summaries = []
    for members in rows:
        summary = target_summary(members)
        players = {r["player"] for r in members}
        cohorts = {r["cohort"] for r in members}
        game_ids = {r["game_id"] for r in members}
        target_conflict = len({r["target"] for r in members}) > 1
        # Conflicting terminal outcomes across distinct games are expected
        # trajectory-dependent labels unless metadata/encoding disagree.
        anomaly_reasons = []
        if len(players) > 1:
            anomaly_reasons.append("same_encoded_input_has_different_player_identity")
        if len(cohorts) > 1:
            anomaly_reasons.append("same_encoded_input_has_different_tactical_cohort")
        if target_conflict and len(game_ids) < 2:
            anomaly_reasons.append("target_conflict_within_single_game")
        classification = (
            "metadata_or_encoding_inconsistency_suspected" if anomaly_reasons
            else "trajectory_outcome_variation" if target_conflict
            else "consistent_target"
        )
        summaries.append({
            "fingerprint": members[0]["fingerprint"],
            **summary,
            "players": sorted(players),
            "cohorts": dict(Counter(r["cohort"] for r in members)),
            "game_ids": sorted(game_ids),
            "target_conflict": target_conflict,
            "conflict_classification": classification,
            "anomaly_reasons": anomaly_reasons,
        })

    return {
        "state_count": len(rows),
        "conflicting_state_count": len(conflict),
        "records_in_conflicting_states": sum(len(x) for x in conflict),
        "conflict_classification_counts": dict(Counter(
            s["conflict_classification"] for s in summaries if s["target_conflict"]
        )),
        "target_variance_distribution": {
            "mean": statistics.mean(s["target_variance"] for s in summaries),
            "median": quantile([s["target_variance"] for s in summaries], 0.50),
            "p90": quantile([s["target_variance"] for s in summaries], 0.90),
            "max": max(s["target_variance"] for s in summaries),
        },
        "high_uncertainty_states": sorted(
            [s for s in summaries if s["target_conflict"]],
            key=lambda s: (-s["target_variance"], -s["record_count"], s["fingerprint"]),
        )[:100],
        "all_state_summaries": summaries,
    }


def cohort_coverage(rows: list[dict]) -> dict:
    by_cohort = defaultdict(list)
    for row in rows:
        by_cohort[row["cohort"]].append(row)
    result = {}
    for cohort, members in sorted(by_cohort.items()):
        grouped = defaultdict(list)
        for row in members:
            grouped[row["fingerprint"]].append(row)
        targets = [r["target"] for r in members]
        result[cohort] = {
            "records": len(members),
            "unique_states": len(grouped),
            "games": len({r["game_id"] for r in members}),
            "positive_targets": sum(y > 0 for y in targets),
            "negative_targets": sum(y < 0 for y in targets),
            "unique_state_target_conflicts": sum(
                len({r["target"] for r in group}) > 1 for group in grouped.values()
            ),
            "state_weighted_mean_target": statistics.mean(
                statistics.mean(r["target"] for r in group)
                for group in grouped.values()
            ),
        }
    return result


def partition_summary(rows: list[dict]) -> dict:
    if not rows:
        return {
            "records": 0, "unique_states": 0, "games": 0,
            "cohort_coverage": {}, "loss_metrics": {},
        }
    return {
        "records": len(rows),
        "unique_states": len({r["fingerprint"] for r in rows}),
        "games": len({r["game_id"] for r in rows}),
        "cohort_coverage": cohort_coverage(rows),
        "loss_metrics": loss_metrics(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
        default=Path("data/experiments/exp_001/heuristic_decisive_games.jsonl"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
        default=Path("data/diagnostics/state_weighted_purged_holdout/run_001"))
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
        parser.error(
            f"Checkpoint SHA-256 mismatch: expected {args.expected_checkpoint_sha256}, got {checkpoint_sha}"
        )

    rows = []
    game_meta = {}
    seen_game_ply = set()
    for line_no, line in enumerate(dataset_bytes.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        required = {"state", "game_id", "ply", "player", "winner"}
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"{args.dataset}:{line_no}: missing fields {sorted(missing)}")
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
        rows.append({
            "state": raw["state"], "fingerprint": fingerprint(raw["state"]),
            "game_id": gid, "ply": ply, "player": player, "winner": winner,
            "target": 1 if player == winner else -1,
            "cohort": None, "prediction": None,
        })

    if not rows:
        raise ValueError("Dataset contains no records")

    groups = defaultdict(list)
    for row in rows:
        groups[row["fingerprint"]].append(row)

    # Reconstruct and validate each sample, then compute deterministic tactical labels.
    states = []
    for row in rows:
        state = restore_encoded_state({
            "state": row["state"], "player": row["player"], "ply": row["ply"]
        })
        row["cohort"] = label_position(state)["cohort"]
        states.append(state)

    model = load_model(args.checkpoint, device=args.device)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(states), args.batch_size):
            batch_states = states[start:start + args.batch_size]
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

    # Existing training split: preserve the exact original winner-stratified game split.
    splits = split_by_game(game_meta, args.seed)
    gid_to_split = {gid: name for name, gids in splits.items() for gid in gids}
    train_fps = {r["fingerprint"] for r in rows if gid_to_split[r["game_id"]] == "train"}

    validation_raw = [r for r in rows if gid_to_split[r["game_id"]] == "validation"]
    test_raw = [r for r in rows if gid_to_split[r["game_id"]] == "test"]
    validation_purged = [r for r in validation_raw if r["fingerprint"] not in train_fps]
    validation_fps = {r["fingerprint"] for r in validation_purged}
    test_after_train_purge = [r for r in test_raw if r["fingerprint"] not in train_fps]
    test_purged = [r for r in test_after_train_purge if r["fingerprint"] not in validation_fps]

    purged_holdout = {
        "split_method": "original 80/10/10 winner-stratified game split, followed by exact-state purge",
        "seed": args.seed,
        "game_counts": {name: len(gids) for name, gids in splits.items()},
        "training_state_count": len(train_fps),
        "validation": {
            "before_purge": partition_summary(validation_raw),
            "removed_training_overlap_records": len(validation_raw) - len(validation_purged),
            "removed_training_overlap_unique_states": len({
                r["fingerprint"] for r in validation_raw
                if r["fingerprint"] in train_fps
            }),
            "after_training_purge": partition_summary(validation_purged),
        },
        "test": {
            "before_purge": partition_summary(test_raw),
            "removed_training_overlap_records": len(test_raw) - len(test_after_train_purge),
            "removed_training_overlap_unique_states": len({
                r["fingerprint"] for r in test_raw
                if r["fingerprint"] in train_fps
            }),
            "removed_validation_overlap_records": len(test_after_train_purge) - len(test_purged),
            "removed_validation_overlap_unique_states": len({
                r["fingerprint"] for r in test_after_train_purge
                if r["fingerprint"] in validation_fps
            }),
            "after_training_and_validation_purge": partition_summary(test_purged),
        },
        "post_purge_state_overlap": {
            "train_validation": len(train_fps & {r["fingerprint"] for r in validation_purged}),
            "train_test": len(train_fps & {r["fingerprint"] for r in test_purged}),
            "validation_test": len(validation_fps & {r["fingerprint"] for r in test_purged}),
        },
        "warning": (
            "Purged metrics describe the existing checkpoint, which was trained on the original training games. "
            "The purged validation/test subsets are useful diagnostics of novel encoded states, but their small "
            "sample sizes and game-level selection must be considered. No retraining was performed."
        ),
    }

    target_report = summarize_groups(groups)
    frequency_report = frequency_summary(groups, len(rows))
    overall_cohorts = cohort_coverage(rows)
    full_report = {
        "experiment": "state_weighted_dataset_and_purged_holdout_audit",
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_sha,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "config": {
            "seed": args.seed,
            "split": "same 80/10/10 winner-stratified game split as EXP-001R",
            "purge": "remove validation states seen in train; remove test states seen in train or retained validation",
            "target": "undiscounted +/-1 from player-to-move perspective",
            "read_only": True,
        },
        "integrity": {
            "records": len(rows),
            "games": len(game_meta),
            "unique_states": len(groups),
            "duplicate_records": len(rows) - len(groups),
            "duplicate_record_pct": 100.0 * (len(rows) - len(groups)) / len(rows),
            "all_states_round_trip_valid": True,
            "duplicate_game_ply_count": 0,
            "player_identity_conflict_state_count": sum(
                len({r["player"] for r in members}) > 1 for members in groups.values()
            ),
            "cohort_identity_conflict_state_count": sum(
                len({r["cohort"] for r in members}) > 1 for members in groups.values()
            ),
        },
        "state_frequency": frequency_report,
        "loss_baseline_comparison": {
            "definition": {
                "record_weighted": "mean loss over all records",
                "state_weighted_preserve_each_record_target": "mean within-state record loss, then equal weight per unique state",
                "state_weighted_aggregated_target": "one mean target and one prediction per unique state, equal weight per state",
            },
            "all_dataset": loss_metrics(rows),
            "original_game_split": {
                name: loss_metrics([r for r in rows if gid_to_split[r["game_id"]] == name])
                for name in ("train", "validation", "test")
            },
            "purged_validation": loss_metrics(validation_purged),
            "purged_test": loss_metrics(test_purged),
        },
        "target_aggregation": {
            "state_count": target_report["state_count"],
            "conflicting_state_count": target_report["conflicting_state_count"],
            "records_in_conflicting_states": target_report["records_in_conflicting_states"],
            "conflict_classification_counts": target_report["conflict_classification_counts"],
            "target_variance_distribution": target_report["target_variance_distribution"],
            "high_uncertainty_states": target_report["high_uncertainty_states"],
        },
        "tactical_coverage": {
            "all_dataset": overall_cohorts,
            "purged_validation": cohort_coverage(validation_purged),
            "purged_test": cohort_coverage(test_purged),
        },
        "purged_holdout": purged_holdout,
        "state_summaries": target_report["all_state_summaries"],
        "records": [{
            key: value for key, value in row.items()
            if key not in ("state",)
        } for row in rows],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in full_report.items()
               if key not in ("records", "state_summaries")}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "state_weighted_purged_holdout.json").write_text(
        json.dumps(full_report, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    print(f"records={len(rows)} games={len(game_meta)} unique_states={len(groups)}")
    print(f"duplicates={len(rows) - len(groups)}")
    print(f"purged_validation_records={len(validation_purged)} purged_test_records={len(test_purged)}")
    print(f"outputs={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
