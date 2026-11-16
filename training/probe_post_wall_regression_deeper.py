#!/usr/bin/env python3
"""Run paired multi-ply tactical continuations for saved post-wall regression roots.

The exact 100 roots are taken from the prior representative-probe report. Root
searches and checkpoints are reused unchanged. Each regression/progress branch
then receives a fixed number of alternating MCTS moves, using the matching
vanilla or repetition-avoidance condition at every continuation ply.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import torch
import ijson

from quoridor.engine import state_repetition_key
from quoridor.neural_ai.encoder import apply_action_index, encode_state
from quoridor.neural_ai.mcts import MCTS
from training.audit_post_wall_pawn_regression import (
    load_model,
    reconstruct_positions,
    restore_state,
)

AGENTS = ("candidate", "incumbent")
VARIANTS = ("vanilla_puct", "with_repetition_avoidance")


def tuple_key(value: list) -> tuple:
    pawns, walls, remaining, player, winner = value
    return (
        tuple(tuple(p) for p in pawns),
        tuple(tuple(w) for w in walls),
        tuple(remaining),
        player,
        winner,
    )


def action_by_category(variant: dict) -> dict:
    """Recover the best-visit regression and progress actions from saved traces."""
    latest = {}
    for snapshot in variant.get("selection_trace", []):
        if snapshot.get("selection_depth") != 0:
            continue
        for row in snapshot.get("candidate_pawn_actions", []):
            action = row.get("action")
            delta = row.get("distance_delta")
            if action is None or delta is None:
                continue
            category = "regression" if float(delta) < 0 else "progress" if float(delta) > 0 else None
            if category is None:
                continue
            visits = row.get("visits_after_backup")
            if visits is None:
                visits = row.get("visits", 0)
            key = int(action)
            previous = latest.get(key)
            candidate = {
                "action": key,
                "category": category,
                "visits": int(visits or 0),
                "prior": float(row.get("prior", 0.0) or 0.0),
                "distance_delta": float(delta),
            }
            if previous is None or candidate["visits"] >= previous["visits"]:
                latest[key] = candidate
    best = {}
    for row in latest.values():
        category = row["category"]
        current = best.get(category)
        if current is None or (row["visits"], row["prior"]) > (current["visits"], current["prior"]):
            best[category] = row
    return best


def selected_root_keys(reference: dict) -> set[tuple[str, int, int]]:
    rows = reference.get("rows") or []
    keys = {
        (
            str(row["source_agent"]),
            int(row["source_game_index"]),
            int(row["ply"]),
        )
        for row in rows
    }
    if len(keys) != 100:
        raise ValueError(f"Reference probe must identify exactly 100 unique roots; found {len(keys)}.")
    for agent in AGENTS:
        count = sum(key[0] == agent for key in keys)
        if count != 50:
            raise ValueError(f"Reference sample must contain 50 {agent} roots; found {count}.")
    for variant in VARIANTS:
        count = sum(row.get("variant") == variant for row in rows)
        if count != 100:
            raise ValueError(f"Reference report must contain 100 rows for {variant}; found {count}.")
    return keys


def build_history_map(source_evaluation: dict, expected_keys: set) -> dict:
    """Reconstruct exact chronological repetition histories for the fixed sample."""
    histories = {}
    for agent in AGENTS:
        for position in reconstruct_positions(source_evaluation, agent):
            key = (agent, int(position["source_game_index"]), int(position["ply"]))
            if key in expected_keys:
                histories[key] = position
    missing = expected_keys - set(histories)
    if missing:
        preview = sorted(missing)[:5]
        raise ValueError(f"Could not reconstruct history for {len(missing)} sampled roots; examples={preview}")
    return histories


def evaluate_original_player(model, state, original_player: int, device: str) -> float:
    if state.winner is not None:
        return 1.0 if state.winner == original_player else -1.0
    tensor = torch.tensor(
        encode_state(state), dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.inference_mode():
        _policy, value = model(tensor)
    current_player_value = float(value[0].item())
    return current_player_value if state.current_player == original_player else -current_player_value


def continue_branch(
    model,
    root_state,
    initial_action: int,
    recent_state_keys: list[tuple],
    *,
    original_player: int,
    budget: int,
    c_puct: float,
    continuation_plies: int,
    avoid_repetition: bool,
    device: str,
    seed: int,
) -> dict:
    """Apply the saved root action, then search alternating moves for N plies."""
    torch.manual_seed(seed)
    random.seed(seed)
    state = apply_action_index(root_state, initial_action)
    history = list(recent_state_keys)
    current_key = state_repetition_key(state)
    if not history or history[-1] != state_repetition_key(root_state):
        history.append(state_repetition_key(root_state))
    history.append(current_key)
    moves = []
    for offset in range(continuation_plies):
        if state.winner is not None:
            break
        player = int(state.current_player)
        search = MCTS(
            model,
            simulations=budget,
            c_puct=c_puct,
            device=device,
            randomize_selection_ties=False,
            progressive_widening=False,
            category_balanced_prior=False,
            root_category_prior_mix=0.0,
            opening_goal_directed_prior=False,
            avoid_recent_state_repetition=avoid_repetition,
            category_exploration=0.0,
            repetition_penalty=0.0,
            pawn_progress_bonus=0.0,
        )
        policy = search.search(
            state,
            add_root_noise=False,
            recent_state_keys=history,
            record_selection_trace=False,
        )
        if not policy:
            raise RuntimeError(
                f"No continuation action at ply={state.ply}, player={player}, "
                f"avoid_repetition={avoid_repetition}"
            )

        # MCTS.search() returns normalized visit probabilities, not raw visit
        # counts. Read the backed-up edge counts from last_search_stats instead.
        search_stats = search.last_search_stats
        root_action_stats = search_stats.get("root_action_stats", {})
        root_action_visits = {
            int(action): int(action_stats["visits"])
            for action, action_stats in root_action_stats.items()
        }
        root_visit_total = sum(root_action_visits.values())
        if not root_action_visits or root_visit_total != budget:
            raise RuntimeError(
                "Continuation root visit accounting mismatch: "
                f"ply={state.ply}, expected={budget}, actual={root_visit_total}, "
                f"stats_simulations={search_stats.get('simulations')}"
            )
        if set(root_action_visits) != set(policy):
            raise RuntimeError(
                "Continuation policy/action visit keys differ: "
                f"ply={state.ply}, policy={len(policy)}, "
                f"visit_counts={len(root_action_visits)}"
            )

        action = max(root_action_visits, key=root_action_visits.get)
        selected_visits = root_action_visits[action]
        state = apply_action_index(state, int(action))
        history.append(state_repetition_key(state))
        moves.append({
            "continuation_index": offset + 1,
            "player": player,
            "action": int(action),
            "selected_action_visits": selected_visits,
            "root_visit_total": root_visit_total,
            "root_simulations_requested": int(search_stats["simulations"]),
            "root_action_visit_counts": {
                str(action): count
                for action, count in sorted(root_action_visits.items())
            },
            "state_ply_after_move": int(state.ply),
            "terminal_after_move": state.winner is not None,
        })
    final_value = evaluate_original_player(model, state, original_player, device)
    return {
        "initial_action": int(initial_action),
        "continuation_moves": moves,
        "continuation_plies_completed": len(moves),
        "final_ply": int(state.ply),
        "final_current_player": int(state.current_player),
        "final_winner": state.winner,
        "terminal": state.winner is not None,
        "final_value_for_original_player": final_value,
    }


def summarize(rows: list[dict]) -> dict:
    deltas = [row["regression_minus_progress"] for row in rows]
    return {
        "paired_positions": len(rows),
        "regression_better_count": sum(value > 1e-12 for value in deltas),
        "progress_better_count": sum(value < -1e-12 for value in deltas),
        "equal_count": sum(abs(value) <= 1e-12 for value in deltas),
        "regression_better_percent": (
            100.0 * sum(value > 1e-12 for value in deltas) / len(deltas) if deltas else None
        ),
        "mean_regression_final_value": mean(
            row["regression_final_value"] for row in rows
        ) if rows else None,
        "mean_progress_final_value": mean(
            row["progress_final_value"] for row in rows
        ) if rows else None,
        "mean_regression_minus_progress": mean(deltas) if deltas else None,
        "median_regression_minus_progress": median(deltas) if deltas else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_checkpoint", type=Path)
    parser.add_argument("incumbent_checkpoint", type=Path)
    parser.add_argument("audit_json", type=Path)
    parser.add_argument("source_evaluation", type=Path)
    parser.add_argument("reference_probes_json", type=Path)
    parser.add_argument("--continuation-plies", type=int, default=4,
                        help="Additional alternating MCTS moves after the saved root action.")
    parser.add_argument("--simulations", type=int, default=None,
                        help="MCTS simulations per continuation ply; defaults to audit budget.")
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    if args.continuation_plies < 1 or args.threads < 1:
        parser.error("--continuation-plies and --threads must be positive")
    for output in (args.output, args.csv):
        if output.exists():
            parser.error(f"Output already exists: {output}; choose a fresh path.")
    for source in (args.audit_json, args.source_evaluation, args.reference_probes_json):
        if not source.is_file():
            parser.error(f"Input file not found: {source}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    torch.set_num_threads(args.threads)
    reference = json.loads(args.reference_probes_json.read_text(encoding="utf-8"))
    expected_keys = selected_root_keys(reference)
    source_evaluation = json.loads(args.source_evaluation.read_text(encoding="utf-8"))
    history_map = build_history_map(source_evaluation, expected_keys)

    audit_runs = {}
    with args.audit_json.open("rb") as audit_handle:
        audit_runs_iter = ijson.items(audit_handle, "runs.item")
        for run in audit_runs_iter:
            source = run.get("source") or {}
            key = (
                str(source.get("source_agent")),
                int(source.get("source_game_index", -1)),
                int(source.get("ply", -1)),
            )
            if key in expected_keys:
                if key in audit_runs:
                    raise ValueError(f"Duplicate audit root for {key}")
                audit_runs[key] = run
    missing_audit = expected_keys - set(audit_runs)
    if missing_audit:
        raise ValueError(f"Audit is missing {len(missing_audit)} sampled roots.")

    budget = args.simulations
    if budget is None:
        budget = int(reference.get("root_search_budget_reused", 32))
    if budget < 1:
        parser.error("--simulations must be positive")
    c_puct = float(reference.get("c_puct", 1.5))

    models = {
        "candidate": load_model(args.candidate_checkpoint, device=args.device),
        "incumbent": load_model(args.incumbent_checkpoint, device=args.device),
    }
    for model in models.values():
        model.eval()

    output_rows = []
    branch_count = 0
    total_branches = len(expected_keys) * len(VARIANTS) * 2
    for key in sorted(expected_keys):
        agent, game_index, ply = key
        run = audit_runs[key]
        source = run.get("source") or {}
        state = restore_state(run["full_state"])
        position = history_map[key]
        history = [tuple_key(item) for item in position["recent_state_keys"]]
        if not history or history[-1] != state_repetition_key(state):
            raise ValueError(f"Reconstructed repetition history does not end at audit root: {key}")
        model = models[agent]
        original_player = int(state.current_player)
        for variant_name in VARIANTS:
            variant = (run.get("variants") or {}).get(variant_name) or {}
            actions = action_by_category(variant)
            if "regression" not in actions or "progress" not in actions:
                raise ValueError(f"Missing regression/progress root action for {key}, {variant_name}")
            avoid = variant_name == "with_repetition_avoidance"
            results = {}
            pair_seed = (
                args.seed
                + (0 if agent == "candidate" else 1_000_000)
                + game_index * 10_000
                + ply * 10
                + (0 if variant_name == "vanilla_puct" else 1)
            )
            for category in ("regression", "progress"):
                branch_count += 1
                root_action = actions[category]["action"]
                branch_seed = pair_seed
                print(
                    f"[{branch_count}/{total_branches}] agent={agent} game={game_index} "
                    f"ply={ply} variant={variant_name} branch={category}",
                    flush=True,
                )
                results[category] = continue_branch(
                    model,
                    state,
                    root_action,
                    history,
                    original_player=original_player,
                    budget=budget,
                    c_puct=c_puct,
                    continuation_plies=args.continuation_plies,
                    avoid_repetition=avoid,
                    device=args.device,
                    seed=branch_seed,
                )
            reg_value = results["regression"]["final_value_for_original_player"]
            prog_value = results["progress"]["final_value_for_original_player"]
            output_rows.append({
                "source_agent": agent,
                "checkpoint_role": agent,
                "source_game_index": game_index,
                "source_seed": source.get("source_seed"),
                "ply": ply,
                "variant": variant_name,
                "regression_action": actions["regression"]["action"],
                "regression_root_visits": actions["regression"]["visits"],
                "progress_action": actions["progress"]["action"],
                "progress_root_visits": actions["progress"]["visits"],
                "continuation_plies_requested": args.continuation_plies,
                "regression_final_value": reg_value,
                "progress_final_value": prog_value,
                "regression_minus_progress": reg_value - prog_value,
                "regression_terminal": results["regression"]["terminal"],
                "progress_terminal": results["progress"]["terminal"],
                "regression_continuation_moves": results["regression"]["continuation_moves"],
                "progress_continuation_moves": results["progress"]["continuation_moves"],
            })

    grouped = {
        agent: {
            variant: summarize([
                row for row in output_rows
                if row["checkpoint_role"] == agent and row["variant"] == variant
            ])
            for variant in VARIANTS
        }
        for agent in AGENTS
    }
    report = {
        "experiment": "post_wall_pawn_regression_deeper_tactical_validation",
        "source_audit": str(args.audit_json),
        "source_evaluation": str(args.source_evaluation),
        "reference_probe_report": str(args.reference_probes_json),
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "incumbent_checkpoint": str(args.incumbent_checkpoint),
        "sampled_roots": len(expected_keys),
        "sample_identity": "Exactly the 100 unique roots listed in the prior representative response-probe report.",
        "checkpoint_split": "Candidate-origin roots use candidate checkpoint; incumbent-origin roots use incumbent checkpoint.",
        "root_actions": "Best-visit regression and progress actions recovered from the saved root traces; no root search rerun.",
        "visit_instrumentation": {\n            "version": 2,\n            "source": "MCTS.last_search_stats.root_action_stats[*].visits",\n            "root_visit_total_validation": "Every completed continuation search must sum to the requested simulation budget.",\n            "historical_report_warning": "Visit fields in post_wall_pawn_regression_deeper_4ply.json and its CSV are invalid because they summed normalized policy probabilities; final values and move choices remain usable.",\n        },\n        "continuation": {
            "additional_plies_after_root_action": args.continuation_plies,
            "simulations_per_continuation_ply": budget,
            "c_puct": c_puct,
            "search": "Fresh vanilla PUCT tree at each continuation ply; root noise, progressive widening, category balancing, category exploration, repetition penalty, and pawn progress bonus disabled.",
            "repetition_condition": "Vanilla condition disables full-state repetition filtering at every continuation ply; repetition-avoidance condition enables it at every continuation ply using reconstructed chronological state history.",
            "move_selection": "Highest root visit count after the fixed MCTS budget.",
            "evaluation": "Unchanged matching checkpoint, converted to the original root player's perspective; terminal outcomes use exact win/loss values.",
            "paired_branches": "Regression and progress start from the exact same root state, checkpoint, search condition, and continuation budget.",
        },
        "training_or_checkpoint_changes": False,
        "policy_prior_or_heuristic_changes": False,
        "by_checkpoint_role_and_variant": grouped,
        "interpretation": "Multi-ply fixed-budget MCTS continuation diagnostic, not exhaustive minimax or proof of optimal play.",
        "rows": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "source_agent", "checkpoint_role", "source_game_index", "source_seed", "ply",
        "variant", "regression_action", "regression_root_visits", "progress_action",
        "progress_root_visits", "continuation_plies_requested", "regression_final_value",
        "progress_final_value", "regression_minus_progress", "regression_terminal",
        "progress_terminal", "regression_continuation_moves", "progress_continuation_moves",
    ]
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in output_rows:
            flattened = dict(row)
            flattened["regression_continuation_moves"] = json.dumps(row["regression_continuation_moves"])
            flattened["progress_continuation_moves"] = json.dumps(row["progress_continuation_moves"])
            writer.writerow(flattened)
    print(f"Deeper tactical report: {args.output}")
    print(f"Deeper tactical CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
