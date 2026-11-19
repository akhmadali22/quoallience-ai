#!/usr/bin/env python3
"""Inference-only forced-outcome/value-perspective consistency audit.

Reads the frozen tactical_action_value_separation report. No training, MCTS/PUCT,
checkpoint writes, or model changes are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from quoridor.engine import (
    GameState, Wall, apply_move, validate_state, legal_pawn_moves,
)
from quoridor.neural_ai.encoder import (
    apply_action_index, decode_action, encode_state,
    encode_action, legal_action_indices,
)
from quoridor.neural_ai.policy import load_model
from training.audit_tactical_label_head_calibration import (
    independent_immediate_win, restore_state,
)


def mirror_state(state: GameState) -> GameState:
    """Reflect board horizontally; preserve player identities and side-to-move."""
    pawns = tuple((8 - x, y) for x, y in state.pawns)
    walls = frozenset(Wall(w.orientation, 7 - w.x, w.y) for w in state.walls)
    mirrored = GameState(
        pawns=pawns, walls=walls, walls_remaining=state.walls_remaining,
        current_player=state.current_player, winner=state.winner, ply=state.ply,
    )
    validate_state(mirrored)
    return mirrored


def mirror_action_index(index: int) -> int:
    kind, payload = decode_action(index)
    if kind == "move":
        x, y = payload
        return encode_action(("move", (8 - x, y)))
    return encode_action(("wall", Wall(payload.orientation, 7 - payload.x, payload.y)))


def model_values(model, tasks: list[tuple[str, GameState]], device: str, batch_size: int) -> dict[str, float]:
    import torch
    result: dict[str, float] = {}
    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start:start + batch_size]
        batch = torch.tensor(
            [encode_state(state) for _, state in chunk],
            dtype=torch.float32, device=device,
        )
        with torch.inference_mode():
            _, values = model(batch)
        for (key, _), value in zip(chunk, values.detach().cpu().tolist()):
            result[key] = float(value)
    return result


def terminal_fixture_check() -> dict:
    """Verify engine terminal semantics independently of the neural network."""
    state = GameState(
        pawns=((4, 1), (4, 7)), walls=frozenset(),
        walls_remaining=(10, 10), current_player=0, winner=None, ply=0,
    )
    validate_state(state)
    terminal = apply_move(state, (4, 0))
    return {
        "fixture": state.to_dict(),
        "winning_move": [4, 0],
        "terminal_state": terminal.to_dict(),
        "winner": terminal.winner,
        "current_player_after_terminal_move": terminal.current_player,
        "legal_moves_after_terminal": legal_pawn_moves(terminal),
        "expected_winner": 0,
        "expected_current_player_after_terminal_move": 1,
        "passed": (
            terminal.winner == 0
            and terminal.current_player == 1
            and legal_pawn_moves(terminal) == []
        ),
        "terminal_root_perspective_value_for_player_0": 1.0,
        "terminal_root_perspective_value_for_player_1": -1.0,
        "note": "Terminal outcome is exact engine adjudication; neural output is not queried on terminal states.",
    }


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values), "mean": mean(values), "median": median(values),
        "min": min(values), "max": max(values),
        "positive_count": sum(v > 0 for v in values),
        "negative_count": sum(v < 0 for v in values),
        "zero_count": sum(v == 0 for v in values),
        "positive_pct": 100.0 * sum(v > 0 for v in values) / len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--source-json", type=Path,
        default=Path("data/diagnostics/tactical_action_value_separation/run_001/tactical_action_value_separation.json"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--expected-checkpoint-sha256", default="d58d9e47090e7ca0cbf3b4cfa02b7ab6a2a56ded16ce7c4090dcac202b72f098")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/diagnostics/forced_outcome_value_consistency/run_001"),
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.source_json.is_file():
        parser.error("Checkpoint and --source-json must exist")
    if min(args.batch_size, args.threads) < 1:
        parser.error("--batch-size and --threads must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")

    import torch
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    source_bytes = args.source_json.read_bytes()
    source = json.loads(source_bytes)
    source_positions = source.get("positions", [])
    if not source_positions:
        parser.error("Source JSON has no positions[]")

    # Use the frozen controls from the preceding experiment. Do not resample
    # positions, so every result is directly comparable with that audit.
    selected = [
        item for item in source_positions
        if item.get("cohort") in {"threat_no_defense", "no_threat"}
    ]
    if not selected:
        parser.error("Source JSON contains no threat_no_defense/no_threat cohorts")

    model = load_model(args.checkpoint, device=args.device)
    model.eval()
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.expected_checkpoint_sha256 and checkpoint_sha != args.expected_checkpoint_sha256:
        parser.error(f"Checkpoint SHA-256 mismatch: expected {args.expected_checkpoint_sha256}, got {checkpoint_sha}")

    records = []
    tasks: list[tuple[str, GameState]] = []
    task_meta: dict[str, tuple[str, str]] = {}
    state_by_id: dict[str, GameState] = {}
    transition_checks = {"actions_checked": 0, "mirror_transition_mismatches": 0,
                         "mirror_legal_set_mismatches": 0, "invalid_mirrored_states": 0,
                         "reply_terminal_semantics_mismatches": 0}

    for pos_index, item in enumerate(selected):
        position_id = item.get("position_id", f"source_{pos_index:03d}")
        state = restore_state(item["state"])
        root = state.current_player
        validate_state(state)
        mirrored = mirror_state(state)
        state_by_id[position_id] = state

        root_key = f"{position_id}:root"
        root_mirror_key = f"{position_id}:root_mirror"
        tasks.extend([(root_key, state), (root_mirror_key, mirrored)])
        task_meta[root_key] = (position_id, "root")
        task_meta[root_mirror_key] = (position_id, "root_mirror")

        legal = legal_action_indices(state)
        mirrored_legal = set(legal_action_indices(mirrored))
        reflected_legal = {mirror_action_index(a) for a in legal}
        if reflected_legal != mirrored_legal:
            transition_checks["mirror_legal_set_mismatches"] += 1

        pre_threat = independent_immediate_win(state, 1 - root)
        root_winning_actions = []
        action_rows = []
        for action in legal:
            child = apply_action_index(state, action)
            reflected_action = mirror_action_index(action)
            mirrored_child = apply_action_index(mirrored, reflected_action)
            expected_mirrored_child = mirror_state(child)
            transition_checks["actions_checked"] += 1
            if mirrored_child != expected_mirrored_child:
                transition_checks["mirror_transition_mismatches"] += 1

            if child.winner == root:
                root_winning_actions.append(action)
                action_rows.append({
                    "action_index": action, "action": item_action(action),
                    "child_terminal": True, "winner": child.winner,
                    "exact_child_value_root_perspective": 1.0,
                    "opponent_immediate_winning_replies": [],
                })
                continue
            if child.winner is not None:
                action_rows.append({
                    "action_index": action, "action": item_action(action),
                    "child_terminal": True, "winner": child.winner,
                    "exact_child_value_root_perspective": -1.0,
                    "opponent_immediate_winning_replies": [],
                })
                continue

            child_key = f"{position_id}:action:{action}:child"
            child_mirror_key = f"{position_id}:action:{action}:child_mirror"
            tasks.extend([(child_key, child), (child_mirror_key, mirrored_child)])
            task_meta[child_key] = (position_id, f"action_{action}_child")
            task_meta[child_mirror_key] = (position_id, f"action_{action}_child_mirror")

            opponent = child.current_player
            winning_replies = []
            for destination in legal_pawn_moves(child, opponent):
                terminal = apply_move(child, destination, opponent)
                if terminal.winner == opponent:
                    terminal_ok = (terminal.current_player == root and legal_pawn_moves(terminal) == [])
                    if not terminal_ok:
                        transition_checks["reply_terminal_semantics_mismatches"] += 1
                    winning_replies.append({
                        "move": list(destination),
                        "winner": terminal.winner,
                        "terminal_current_player": terminal.current_player,
                        "terminal_has_no_legal_pawn_moves": legal_pawn_moves(terminal) == [],
                        "terminal_semantics_passed": terminal_ok,
                        "exact_terminal_value_root_perspective": -1.0,
                        "exact_terminal_value_winner_perspective": 1.0,
                    })

            action_rows.append({
                "action_index": action, "action": item_action(action),
                "child_terminal": False,
                "child_current_player": opponent,
                "opponent_has_immediate_winning_reply": bool(winning_replies),
                "opponent_immediate_winning_replies": winning_replies,
                "child_value_task": child_key,
                "child_mirror_value_task": child_mirror_key,
            })

        # A forced-loss label requires every legal root action to be nonterminal
        # and leave at least one immediate winning reply for the opponent.
        is_forced_loss = bool(
            pre_threat and not root_winning_actions and action_rows
            and all(not r.get("child_terminal", False)
                    and r.get("opponent_has_immediate_winning_reply", False)
                    for r in action_rows)
        )
        records.append({
            "position_id": position_id, "cohort": item.get("cohort"),
            "state": state.to_dict(), "current_player_root": root,
            "legal_action_count": len(legal),
            "opponent_immediate_win_before_root_action": pre_threat,
            "root_immediate_winning_action_count": len(root_winning_actions),
            "root_immediate_winning_action_indices": root_winning_actions,
            "forced_loss_by_one_ply_terminal_proof": is_forced_loss,
            "root_value_task": root_key, "root_mirror_value_task": root_mirror_key,
            "actions": action_rows,
        })

    values = model_values(model, tasks, args.device, args.batch_size)

    forced_records = []
    all_forced_root_values = []
    all_forced_child_side_values = []
    all_forced_child_root_values = []
    all_forced_mirror_root_abs_diffs = []
    all_forced_mirror_child_abs_diffs = []
    child_tasks_by_cohort = defaultdict(list)

    for record in records:
        root_value = values[record["root_value_task"]]
        root_mirror_value = values[record["root_mirror_value_task"]]
        record["root_value_side_to_move"] = root_value
        record["root_value_root_perspective"] = root_value
        record["root_mirror_value_side_to_move"] = root_mirror_value
        record["root_mirror_abs_difference"] = abs(root_value - root_mirror_value)
        record["actions_with_values"] = []
        for action in record["actions"]:
            row = dict(action)
            if not row["child_terminal"]:
                child_value = values[row["child_value_task"]]
                child_mirror_value = values[row["child_mirror_value_task"]]
                row["child_value_side_to_move_opponent"] = child_value
                row["child_value_root_perspective"] = -child_value
                row["child_mirror_value_side_to_move_opponent"] = child_mirror_value
                row["child_mirror_abs_difference"] = abs(child_value - child_mirror_value)
                row["expected_sign_if_opponent_has_immediate_win"] = "positive_for_child_side_to_move"
                child_tasks_by_cohort[record["cohort"]].append(child_value)
                if record["forced_loss_by_one_ply_terminal_proof"]:
                    all_forced_child_side_values.append(child_value)
                    all_forced_child_root_values.append(-child_value)
                    all_forced_mirror_child_abs_diffs.append(abs(child_value - child_mirror_value))
            record["actions_with_values"].append(row)
        if record["forced_loss_by_one_ply_terminal_proof"]:
            all_forced_root_values.append(root_value)
            all_forced_mirror_root_abs_diffs.append(abs(root_value - root_mirror_value))
            forced_records.append(record)

    terminal_check = terminal_fixture_check()
    if not terminal_check["passed"]:
        raise RuntimeError("Engine terminal fixture check failed")

    forced_summary = {
        "forced_loss_position_count": len(forced_records),
        "root_value_side_to_move": summarize(all_forced_root_values),
        "child_value_side_to_move_opponent": summarize(all_forced_child_side_values),
        "child_value_converted_to_root_perspective": summarize(all_forced_child_root_values),
        "root_horizontal_mirror_abs_difference": summarize(all_forced_mirror_root_abs_diffs),
        "child_horizontal_mirror_abs_difference": summarize(all_forced_mirror_child_abs_diffs),
        "interpretation_sign_expectations": {
            "forced_loss_root_value": "negative",
            "after_root_action_child_value_for_opponent_to_move": "positive when opponent has an immediate winning reply",
            "same_child_value_from_root_perspective": "negative",
            "terminal_after_opponent_winning_reply": "exactly -1 from original root perspective",
        },
    }
    cohort_summary = {}
    for cohort in sorted({r["cohort"] for r in records}):
        cohort_records = [r for r in records if r["cohort"] == cohort]
        cohort_summary[cohort] = {
            "position_count": len(cohort_records),
            "forced_loss_count": sum(r["forced_loss_by_one_ply_terminal_proof"] for r in cohort_records),
            "root_value_side_to_move": summarize([r["root_value_side_to_move"] for r in cohort_records]),
            "root_horizontal_mirror_abs_difference": summarize([r["root_mirror_abs_difference"] for r in cohort_records]),
            "child_value_side_to_move": summarize(child_tasks_by_cohort[cohort]),
        }

    report = {
        "experiment": "forced_outcome_value_consistency_audit",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "source_json": str(args.source_json),
        "source_json_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_experiment": source.get("experiment"),
        "source_position_count": len(source_positions),
        "selected_position_count": len(records),
        "evaluation_contract": {
            "network_value_semantics": "QuoridorNet value head is tanh scalar; encoder input is side-to-move perspective",
            "root_value": "raw network output is current root player's perspective",
            "child_value": "after root action, opponent is side-to-move; raw network output is opponent perspective; root perspective is its negation",
            "terminal_states": "not sent through network; exact engine winner determines +/-1",
            "mcts_or_puct_used": False, "training_performed": False,
            "checkpoint_modified": False, "architecture_modified": False,
        },
        "engine_semantics_check": terminal_check,
        "horizontal_symmetry_checks": transition_checks,
        "forced_loss_summary": forced_summary,
        "cohort_summary": cohort_summary,
        "forced_loss_positions": forced_records,
        "all_positions": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "forced_outcome_value_consistency.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary = {k: v for k, v in report.items() if k not in {"forced_loss_positions", "all_positions"}}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Selected positions: {len(records)}")
    print(f"One-ply forced-loss positions: {len(forced_records)}")
    print(f"Neural value tasks: {len(tasks)}")
    print(f"Checkpoint SHA-256: {checkpoint_sha}")
    print(f"Mirror legal-set mismatches: {transition_checks['mirror_legal_set_mismatches']}")
    print(f"Mirror transition mismatches: {transition_checks['mirror_transition_mismatches']}")
    print(f"Summary: {args.output_dir / 'summary.json'}")
    return 0


def item_action(index: int) -> dict:
    kind, payload = decode_action(index)
    if kind == "move":
        return {"type": "pawn", "to": list(payload)}
    return {"type": "wall", "orientation": payload.orientation, "x": payload.x, "y": payload.y}


if __name__ == "__main__":
    raise SystemExit(main())
