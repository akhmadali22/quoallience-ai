#!/usr/bin/env python3
"""Audit tactical coverage and value targets in the exact EXP-001R training dataset.

Inference/data audit only. Reconstructs every encoded training state, labels
one-ply threats/defenses with the Quoridor engine, verifies the strict training
target and game-level split, then evaluates a frozen checkpoint by cohort.
No training, optimizer, replay mutation, or checkpoint writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from quoridor.engine import GameState, Wall, apply_move, validate_state, legal_pawn_moves
from quoridor.neural_ai.encoder import (
    apply_action_index, encode_state, legal_action_indices,
)
from quoridor.neural_ai.policy import load_model
from training.train_value_only_strict import split_by_game


EXPECTED_CHECKPOINT_SHA256 = "d58d9e47090e7ca0cbf3b4cfa02b7ab6a2a56ded16ce7c4090dcac202b72f098"


def _one_hot_position(plane: list[list[float]], name: str) -> tuple[int, int]:
    found = [(x, y) for y, row in enumerate(plane) for x, v in enumerate(row) if v > 0.5]
    if len(found) != 1:
        raise ValueError(f"{name} must contain exactly one pawn marker; found {len(found)}")
    return found[0]


def restore_encoded_state(row: dict) -> GameState:
    """Invert the repository's 10-plane perspective-relative state encoding."""
    planes = row["state"]
    if len(planes) != 10 or any(len(p) != 9 or any(len(r) != 9 for r in p) for p in planes):
        raise ValueError("Encoded state must have shape 10x9x9")

    own = _one_hot_position(planes[0], "own pawn plane")
    opponent = _one_hot_position(planes[1], "opponent pawn plane")
    own_goal_rows = [y for y, values in enumerate(planes[2]) if all(v > 0.5 for v in values)]
    opponent_goal_rows = [y for y, values in enumerate(planes[3]) if all(v > 0.5 for v in values)]
    if own_goal_rows == [0] and opponent_goal_rows == [8]:
        current_player = 0
    elif own_goal_rows == [8] and opponent_goal_rows == [0]:
        current_player = 1
    else:
        raise ValueError(f"Cannot infer player identity from goal planes: {own_goal_rows}/{opponent_goal_rows}")

    if current_player != int(row["player"]):
        raise ValueError(
            f"Encoded side-to-move {current_player} disagrees with row player {row['player']}"
        )

    pawns = (own, opponent) if current_player == 0 else (opponent, own)
    walls = set()
    for y in range(9):
        for x in range(9):
            if planes[4][y][x] > 0.5:
                if x >= 8 or y >= 8:
                    raise ValueError("Horizontal wall marker outside 8x8 anchor grid")
                walls.add(Wall("H", x, y))
            if planes[5][y][x] > 0.5:
                if x >= 8 or y >= 8:
                    raise ValueError("Vertical wall marker outside 8x8 anchor grid")
                walls.add(Wall("V", x, y))

    own_walls = int(round(float(planes[6][0][0]) * 10))
    opp_walls = int(round(float(planes[7][0][0]) * 10))
    if not (0 <= own_walls <= 10 and 0 <= opp_walls <= 10):
        raise ValueError(f"Invalid wall resources {own_walls}/{opp_walls}")
    walls_remaining = (own_walls, opp_walls) if current_player == 0 else (opp_walls, own_walls)

    ply = int(row["ply"])
    encoded_ply = float(planes[9][0][0])
    expected_ply = min(ply, 200) / 200.0
    if abs(encoded_ply - expected_ply) > 1e-5:
        raise ValueError(f"Encoded ply {encoded_ply} disagrees with metadata ply={ply}")

    state = GameState(
        pawns=pawns, walls=frozenset(walls), walls_remaining=walls_remaining,
        current_player=current_player, winner=None, ply=ply,
    )
    validate_state(state)
    # Round-trip catches plane ordering, wall orientation, and resource mistakes.
    rebuilt = encode_state(state)
    max_diff = max(
        abs(float(planes[c][y][x]) - float(rebuilt[c][y][x]))
        for c in range(10) for y in range(9) for x in range(9)
    )
    if max_diff > 1e-5:
        raise ValueError(f"Encoded-state round-trip mismatch: max_abs_diff={max_diff}")
    return state


def independent_immediate_win(state: GameState, player: int) -> bool:
    if state.winner is not None:
        return state.winner == player
    probe = GameState(
        pawns=state.pawns, walls=state.walls,
        walls_remaining=state.walls_remaining,
        current_player=player, winner=None, ply=state.ply,
    )
    return any(
        apply_move(probe, destination, player).winner == player
        for destination in legal_pawn_moves(probe, player)
    )


def label_position(state: GameState) -> dict:
    """Classify immediate threat, defensive actions, and strict forced losses."""
    root = state.current_player
    opponent = 1 - root
    threatened = independent_immediate_win(state, opponent)
    if not threatened:
        return {
            "cohort": "no_threat", "opponent_immediate_win": False,
            "legal_action_count": None, "root_immediate_win_count": 0,
            "defensive_action_count": 0, "allows_immediate_loss_action_count": 0,
            "forced_loss": False,
        }

    legal = legal_action_indices(state)
    root_wins = 0
    defenses = 0
    allows_loss = 0
    nonterminal_actions = 0
    for action in legal:
        child = apply_action_index(state, action)
        if child.winner == root:
            root_wins += 1
            continue
        if child.winner is not None:
            continue
        nonterminal_actions += 1
        if independent_immediate_win(child, opponent):
            allows_loss += 1
        else:
            defenses += 1

    forced_loss = bool(
        legal and root_wins == 0 and nonterminal_actions == len(legal)
        and allows_loss == len(legal)
    )
    if root_wins:
        cohort = "threat_with_immediate_win"
    elif defenses:
        cohort = "threat_with_defense"
    else:
        cohort = "threat_no_defense"

    return {
        "cohort": cohort, "opponent_immediate_win": True,
        "legal_action_count": len(legal), "root_immediate_win_count": root_wins,
        "defensive_action_count": defenses,
        "allows_immediate_loss_action_count": allows_loss,
        "forced_loss": forced_loss,
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    predictions = [float(r["prediction"]) for r in rows]
    targets = [int(r["training_target"]) for r in rows]
    errors = [p - y for p, y in zip(predictions, targets)]
    exact_forced = [r for r in rows if r["forced_loss"]]
    return {
        "count": len(rows),
        "game_count": len({r["game_id"] for r in rows}),
        "target_positive_count": sum(y > 0 for y in targets),
        "target_negative_count": sum(y < 0 for y in targets),
        "target_positive_pct": 100.0 * sum(y > 0 for y in targets) / len(targets),
        "prediction_mean": mean(predictions),
        "prediction_median": median(predictions),
        "prediction_min": min(predictions),
        "prediction_max": max(predictions),
        "mse_vs_training_target": mean(e * e for e in errors),
        "mae_vs_training_target": mean(abs(e) for e in errors),
        "sign_accuracy_vs_training_target": mean(
            (p >= 0) == (y > 0) for p, y in zip(predictions, targets)
        ),
        "forced_loss_count": len(exact_forced),
        "forced_loss_target_positive_count": sum(r["training_target"] > 0 for r in exact_forced),
        "forced_loss_prediction_positive_count": sum(r["prediction"] > 0 for r in exact_forced),
        "forced_loss_prediction_mean": mean([r["prediction"] for r in exact_forced]) if exact_forced else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
        default=Path("data/experiments/exp_001/heuristic_decisive_games.jsonl"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
        default=Path("data/diagnostics/value_training_coverage/run_001"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256)
    args = parser.parse_args()

    if not args.dataset.is_file() or not args.checkpoint.is_file():
        parser.error("Dataset and checkpoint must exist")
    if min(args.batch_size, args.threads) < 1:
        parser.error("--batch-size and --threads must be positive")
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

    samples = []
    game_meta: dict[int, tuple[int, int]] = {}
    seen = set()
    for line_no, line in enumerate(dataset_bytes.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"state", "game_id", "ply", "player", "winner"}
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{args.dataset}:{line_no}: missing fields {sorted(missing)}")
        game_id, ply = int(row["game_id"]), int(row["ply"])
        player, winner = int(row["player"]), int(row["winner"])
        if player not in (0, 1) or winner not in (0, 1):
            raise ValueError(f"{args.dataset}:{line_no}: invalid player/winner")
        if (game_id, ply) in seen:
            raise ValueError(f"{args.dataset}:{line_no}: duplicate game_id/ply {(game_id, ply)}")
        seen.add((game_id, ply))
        meta = (winner, int(row.get("total_plies", -1)))
        if game_id in game_meta and game_meta[game_id] != meta:
            raise ValueError(f"Inconsistent game metadata for game_id={game_id}")
        game_meta[game_id] = meta
        training_target = 1 if player == winner else -1
        samples.append({
            "state": row["state"], "game_id": game_id, "ply": ply,
            "player": player, "winner": winner, "total_plies": meta[1],
            "training_target": training_target,
            "stored_value": row.get("value"),
        })
    if not samples:
        raise ValueError("Dataset contains no samples")

    splits = split_by_game(game_meta, 20261113003)
    game_split = {gid: split for split, ids in splits.items() for gid in ids}
    model = load_model(args.checkpoint, device=args.device)
    model.eval()

    states = []
    labels = []
    for index, row in enumerate(samples, 1):
        state = restore_encoded_state(row)
        label = label_position(state)
        row["cohort"] = label["cohort"]
        row.update({k: v for k, v in label.items() if k != "cohort"})
        row["split"] = game_split[row["game_id"]]
        row["state_index"] = index
        states.append(state)
        labels.append(row)

    predictions = []
    with torch.inference_mode():
        for start in range(0, len(states), args.batch_size):
            chunk = states[start:start + args.batch_size]
            batch = torch.tensor(
                [encode_state(s) for s in chunk], dtype=torch.float32, device=args.device
            )
            _, values = model(batch)
            predictions.extend(float(v) for v in values.detach().cpu().tolist())
    if len(predictions) != len(samples):
        raise RuntimeError("Prediction count does not match dataset record count")

    records = []
    for row, prediction in zip(labels, predictions):
        record = {k: v for k, v in row.items() if k != "state"}
        record["prediction"] = prediction
        record["prediction_sign_matches_exact_forced_loss"] = (
            (prediction < 0) if row["forced_loss"] else None
        )
        records.append(record)

    cohort_summary = {}
    for cohort in (
        "no_threat", "threat_with_defense", "threat_no_defense",
        "threat_with_immediate_win",
    ):
        cohort_rows = [r for r in records if r["cohort"] == cohort]
        cohort_summary[cohort] = {
            "all": summarize(cohort_rows),
            "train": summarize([r for r in cohort_rows if r["split"] == "train"]),
            "validation": summarize([r for r in cohort_rows if r["split"] == "validation"]),
            "test": summarize([r for r in cohort_rows if r["split"] == "test"]),
        }

    split_summary = {}
    for split in ("train", "validation", "test"):
        split_rows = [r for r in records if r["split"] == split]
        split_summary[split] = {
            "records": len(split_rows),
            "games": len({r["game_id"] for r in split_rows}),
            "cohort_counts": dict(Counter(r["cohort"] for r in split_rows)),
            "forced_loss_count": sum(r["forced_loss"] for r in split_rows),
        }

    target_conflicts = [
        r for r in records if r["forced_loss"] and r["training_target"] > 0
    ]
    forced_rows = [r for r in records if r["forced_loss"]]
    stored_value_checks = []
    for row in records:
        stored = row["stored_value"]
        if stored is not None:
            # The source generator's stored value is discounted; strict EXP-001R
            # deliberately recomputes an undiscounted +/-1 target.
            expected_discount = 0.99 ** max(0, row["total_plies"] - row["ply"])
            expected_stored = expected_discount * row["training_target"]
            stored_value_checks.append(abs(float(stored) - expected_stored) <= 1.1e-6)

    report = {
        "experiment": "value_training_coverage_audit",
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_sha,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "provenance": {
            "training_script": "training/train_value_only_strict.py",
            "generator": "training/generate_heuristic_games.py",
            "target_used_by_strict_training": "undiscounted +1 if player_to_move == winner else -1",
            "stored_dataset_value": "discounted final game outcome; deliberately ignored by strict training",
            "split_seed": 20261113003,
            "split_method": "80/10/10 game-level split stratified by winner",
        },
        "dataset_integrity": {
            "record_count": len(records),
            "game_count": len(game_meta),
            "duplicate_game_id_ply_count": 0,
            "winner_counts": dict(Counter(meta[0] for meta in game_meta.values())),
            "all_encoded_states_round_trip_valid": True,
            "stored_discounted_value_check_count": len(stored_value_checks),
            "stored_discounted_value_match_count": sum(stored_value_checks),
            "stored_discounted_value_mismatch_count": len(stored_value_checks) - sum(stored_value_checks),
        },
        "split_summary": split_summary,
        "cohort_summary": cohort_summary,
        "forced_loss_target_conflicts": {
            "count": len(target_conflicts),
            "pct_of_forced_loss_positions": (
                100.0 * len(target_conflicts) / len(forced_rows) if forced_rows else None
            ),
            "examples": [
                {k: r[k] for k in (
                    "game_id", "ply", "player", "winner", "training_target",
                    "prediction", "split", "legal_action_count",
                    "root_immediate_win_count", "defensive_action_count",
                )}
                for r in target_conflicts[:50]
            ],
        },
        "interpretation_notes": [
            "threat_no_defense may include positions where a root immediate win exists; forced_loss excludes those.",
            "forced_loss is a one-ply engine proof: every legal root action leaves the opponent an immediate winning reply.",
            "A positive final-game training target on a forced-loss position indicates the played heuristic trajectory failed to realize an already available opponent win.",
            "The checkpoint prediction is evaluated from the encoded player-to-move perspective.",
        ],
        "positions": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps({
        k: v for k, v in report.items() if k != "positions"
    }, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "value_training_coverage.json").write_text(
        json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(f"Dataset records={len(records)} games={len(game_meta)}")
    print(f"Dataset SHA-256={dataset_sha}")
    print(f"Checkpoint SHA-256={checkpoint_sha}")
    print(f"Splits={json.dumps(split_summary, sort_keys=True)}")
    print(f"Cohorts={json.dumps({k: v['all']['count'] for k, v in cohort_summary.items()}, sort_keys=True)}")
    print(f"Forced-loss positions={len(forced_rows)} target conflicts={len(target_conflicts)}")
    print(f"Summary: {args.output_dir / 'summary.json'}")
    print(f"Full report: {args.output_dir / 'value_training_coverage.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
