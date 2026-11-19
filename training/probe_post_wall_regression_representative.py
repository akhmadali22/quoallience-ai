#!/usr/bin/env python3
"""Run expanded one-ply response probes on a stratified sample of saved audit roots.

This reuses the expensive root-search traces already stored in the audit JSON.
It does NOT rerun the 32-simulation root audits, retrain, or modify checkpoints.
For each sampled root and search variant, it probes the highest-visit regression
and progress pawn actions, then gives the opponent the same vanilla-PUCT reply
budget and evaluates the resulting state with the unchanged network.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
import ijson

from training.audit_post_wall_pawn_regression import (
    load_model,
    restore_state,
    response_probe,
)

VARIANTS = ("vanilla_puct", "with_repetition_avoidance")
AGENTS = ("candidate", "incumbent")


def candidate_rows(snapshot):
    rows = snapshot.get("candidate_pawn_actions", []) if isinstance(snapshot, dict) else []
    return rows if isinstance(rows, list) else []


def final_action_by_category(variant):
    """Recover final edge visit counts from saved root snapshots, without rerunning root search."""
    latest = {}
    for snapshot in variant.get("selection_trace", []):
        if snapshot.get("selection_depth") != 0:
            continue
        for row in candidate_rows(snapshot):
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
            prior = float(row.get("prior", 0.0) or 0.0)
            key = str(int(action))
            old = latest.get(key)
            if old is None or int(visits or 0) >= old["visits"]:
                latest[key] = {
                    "action": int(action),
                    "category": category,
                    "visits": int(visits or 0),
                    "prior": prior,
                    "distance_delta": float(delta),
                }
    best = {}
    for row in latest.values():
        category = row["category"]
        current = best.get(category)
        if current is None or (row["visits"], row["prior"]) > (current["visits"], current["prior"]):
            best[category] = row
    return best


def stratified_sample(runs, limit, seed):
    """Select evenly across both checkpoint agents and, within each, source games."""
    by_agent_game = defaultdict(list)
    for run in runs:
        agent = (run.get("source") or {}).get("source_agent")
        game = (run.get("source") or {}).get("source_game_index")
        if agent in AGENTS and game is not None:
            by_agent_game[(agent, int(game))].append(run)

    rng = random.Random(seed)
    selected = []
    per_agent = max(1, limit // len(AGENTS))
    for agent in AGENTS:
        game_queues = {}
        for (group_agent, game), group in by_agent_game.items():
            if group_agent != agent:
                continue
            ordered = sorted(group, key=lambda r: int((r.get("source") or {}).get("ply", 0)))
            rng.shuffle(ordered)
            game_queues[game] = ordered
        games = sorted(game_queues)
        cursor = 0
        agent_selected = 0
        while games and agent_selected < per_agent:
            game = games[cursor % len(games)]
            queue = game_queues[game]
            if queue:
                selected.append(queue.pop(0))
                agent_selected += 1
            games = [g for g in games if game_queues[g]]
            cursor += 1
        if agent_selected < per_agent:
            raise RuntimeError(
                f"Only {agent_selected} roots available for {agent}; "
                f"need {per_agent}. Check that the audit includes both source agents."
            )
    if len(selected) < limit:
        raise RuntimeError(f"Selected only {len(selected)} roots, requested {limit}.")
    return selected[:limit]


def summarize(rows):
    values = [row["regression_minus_progress"] for row in rows]
    return {
        "comparable_positions": len(rows),
        "regression_better_count": sum(v > 0 for v in values),
        "progress_better_count": sum(v < 0 for v in values),
        "equal_count": sum(abs(v) <= 1e-12 for v in values),
        "regression_better_percent": 100 * sum(v > 0 for v in values) / len(values) if values else None,
        "mean_regression_value": mean(r["regression_response_value"] for r in rows) if rows else None,
        "mean_progress_value": mean(r["progress_response_value"] for r in rows) if rows else None,
        "mean_regression_minus_progress": mean(values) if values else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_checkpoint", type=Path)
    parser.add_argument("incumbent_checkpoint", type=Path)
    parser.add_argument("audit_json", type=Path)
    parser.add_argument("--positions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    if args.positions < 2 or args.positions % 2:
        parser.error("--positions must be an even number so candidate/incumbent are balanced.")
    if args.threads < 1:
        parser.error("--threads must be positive.")
    for output in (args.output, args.csv):
        if output.exists():
            parser.error(f"Output already exists: {output}; choose a fresh path.")
    if not args.audit_json.is_file():
        parser.error(f"Audit JSON not found: {args.audit_json}")

    torch.set_num_threads(args.threads)
    # First pass reads only compact root metadata; second pass materializes
    # the 100 selected roots. The 450 MB trace is never loaded as one object.
    metadata = []
    for index, run in enumerate(ijson.items(args.audit_json.open("rb"), "runs.item")):
        source = run.get("source") or {}
        metadata.append({
            "_index": index,
            "source": {
                "source_agent": source.get("source_agent"),
                "source_game_index": source.get("source_game_index"),
                "ply": source.get("ply"),
            },
        })
    if not metadata:
        parser.error("Audit JSON contains no runs.")
    selected_meta = stratified_sample(metadata, args.positions, args.seed)
    selected_indices = {row["_index"] for row in selected_meta}
    selected = []
    for index, run in enumerate(ijson.items(args.audit_json.open("rb"), "runs.item")):
        if index in selected_indices:
            selected.append(run)
    selected.sort(key=lambda run: (
        (run.get("source") or {}).get("source_agent", ""),
        int((run.get("source") or {}).get("source_game_index", 0)),
        int((run.get("source") or {}).get("ply", 0)),
    ))

    def top_level_scalar(key, default):
        scalar_events = {"string", "number", "boolean", "null"}
        with args.audit_json.open("rb") as handle:
            for prefix, event, value in ijson.parse(handle):
                if prefix == key and event in scalar_events:
                    return value
        return default

    models = {
        "candidate": load_model(args.candidate_checkpoint, device=args.device),
        "incumbent": load_model(args.incumbent_checkpoint, device=args.device),
    }
    for model in models.values():
        model.eval()

    c_puct = float(top_level_scalar("c_puct", 1.5))
    budget = int(top_level_scalar("simulations", 32))
    output_rows = []
    total_jobs = len(selected) * len(VARIANTS)
    job = 0

    for run in selected:
        source = run.get("source") or {}
        agent = source.get("source_agent")
        if agent not in models:
            raise RuntimeError(f"Unexpected source_agent={agent!r}")
        state = restore_state(run["full_state"])
        model = models[agent]
        for variant_name in VARIANTS:
            job += 1
            variant = (run.get("variants") or {}).get(variant_name) or {}
            actions = final_action_by_category(variant)
            if "regression" not in actions or "progress" not in actions:
                continue
            reg = actions["regression"]
            prog = actions["progress"]
            print(
                f"[{job}/{total_jobs}] agent={agent} "
                f"game={source.get('source_game_index')} ply={source.get('ply')} "
                f"variant={variant_name}",
                flush=True,
            )
            reg_result = response_probe(model, state, reg["action"], budget, c_puct, args.device)
            prog_result = response_probe(model, state, prog["action"], budget, c_puct, args.device)
            if reg_result.get("value_for_original_player") is None or prog_result.get("value_for_original_player") is None:
                continue
            reg_value = float(reg_result["value_for_original_player"])
            prog_value = float(prog_result["value_for_original_player"])
            output_rows.append({
                "source_agent": agent,
                "checkpoint_role": agent,
                "source_game_index": source.get("source_game_index"),
                "source_seed": source.get("source_seed"),
                "ply": source.get("ply"),
                "variant": variant_name,
                "regression_action": reg["action"],
                "regression_visits": reg["visits"],
                "progress_action": prog["action"],
                "progress_visits": prog["visits"],
                "regression_response_value": reg_value,
                "progress_response_value": prog_value,
                "regression_minus_progress": reg_value - prog_value,
                "regression_reply": reg_result.get("opponent_reply"),
                "progress_reply": prog_result.get("opponent_reply"),
                "regression_terminal": reg_result.get("terminal"),
                "progress_terminal": prog_result.get("terminal"),
            })

    grouped = {}
    for agent in AGENTS:
        grouped[agent] = {}
        for variant in VARIANTS:
            subset = [r for r in output_rows if r["source_agent"] == agent and r["variant"] == variant]
            grouped[agent][variant] = summarize(subset)

    report_out = {
        "experiment": "post_wall_pawn_regression_representative_response_probes",
        "source_audit": str(args.audit_json),
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "incumbent_checkpoint": str(args.incumbent_checkpoint),
        "requested_positions": args.positions,
        "sampled_roots": len(selected),
        "sample_seed": args.seed,
        "sampling": "Balanced candidate/incumbent; round-robin across source_game_index within each agent; earliest ply ordering within each game, shuffled deterministically.",
        "root_search_replayed": False,
        "root_search_budget_reused": budget,
        "response_search_budget": budget,
        "c_puct": c_puct,
        "settings": {
            "root_search": "Reused saved final selection_trace; no root MCTS rerun.",
            "opponent_response": "Vanilla PUCT, same response budget for regression and progress.",
            "checkpoint_split": "Candidate roots use candidate checkpoint; incumbent roots use incumbent checkpoint.",
            "training_or_checkpoint_changes": False,
            "policy_prior_or_heuristic_changes": False,
        },
        "by_checkpoint_role_and_variant": grouped,
        "interpretation": "One-ply opponent MCTS response plus unchanged neural value; diagnostic proxy, not minimax proof.",
        "rows": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report_out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    columns = list(output_rows[0]) if output_rows else [
        "source_agent", "checkpoint_role", "source_game_index", "source_seed", "ply",
        "variant", "regression_action", "regression_visits", "progress_action",
        "progress_visits", "regression_response_value", "progress_response_value",
        "regression_minus_progress", "regression_reply", "progress_reply",
        "regression_terminal", "progress_terminal",
    ]
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Probe report: {args.output}")
    print(f"Probe CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
