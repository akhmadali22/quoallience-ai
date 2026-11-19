#!/usr/bin/env python3
"""EXP: Prior-Q Conflict Validation.

Reconstructs continuation positions from the saved post-wall audit and performs
paired 32-vs-128 root-search replays. No training or checkpoint writes occur.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from quoridor.engine import state_repetition_key
from quoridor.neural_ai.encoder import apply_action_index, legal_action_indices
from quoridor.neural_ai.mcts import MCTS
from training.audit_post_wall_pawn_regression import (
    load_model,
    reconstruct_positions,
    restore_state,
)

AGENTS = ("candidate", "incumbent")
SOURCE_VARIANTS = ("vanilla_puct", "with_repetition_avoidance")
BRANCHES = ("regression", "progress")
BUDGETS = (32, 128)
REPLAY_VARIANTS = ("vanilla_puct", "with_repetition_avoidance")


def tuple_key(value):
    pawns, walls, remaining, player, winner = value
    return (tuple(tuple(p) for p in pawns), tuple(tuple(w) for w in walls),
            tuple(remaining), player, winner)


def list_key(key):
    pawns, walls, remaining, player, winner = key
    return [[list(p) for p in pawns], [list(w) for w in walls],
            list(remaining), player, winner]


def build_root_map(source, expected):
    found = {}
    for agent in AGENTS:
        for pos in reconstruct_positions(source, agent):
            key = (agent, int(pos["source_game_index"]), int(pos["ply"]))
            if key in expected:
                if key in found:
                    raise ValueError(f"Duplicate reconstructed root: {key}")
                found[key] = pos
    missing = expected - set(found)
    if missing:
        raise ValueError(f"Missing {len(missing)} roots; examples={sorted(missing)[:5]}")
    return found


def conflict_pairs(diagnostics):
    """Only compare Q values actually observed on visited edges."""
    pairs = []
    for high in diagnostics:
        if int(high.get("raw_visits", 0)) <= 0:
            continue
        hp = high.get("network_prior")
        hq = high.get("final_q")
        if hp is None or hq is None:
            continue
        for low in diagnostics:
            if int(low.get("raw_visits", 0)) <= 0 or low["action"] == high["action"]:
                continue
            lp, lq = low.get("network_prior"), low.get("final_q")
            if lp is None or lq is None:
                continue
            if float(hp) > float(lp) and float(hq) < float(lq):
                pairs.append({
                    "high_prior_action": int(high["action"]),
                    "high_prior": float(hp),
                    "high_q": float(hq),
                    "high_visits": int(high["raw_visits"]),
                    "low_prior_action": int(low["action"]),
                    "low_prior": float(lp),
                    "low_q": float(lq),
                    "low_visits": int(low["raw_visits"]),
                    "prior_gap": float(hp) - float(lp),
                    "q_gap": float(lq) - float(hq),
                    "severity": (float(hp) - float(lp)) * (float(lq) - float(hq)),
                })
    return sorted(pairs, key=lambda x: (-x["severity"], x["high_prior_action"], x["low_prior_action"]))


def reconstruct_conflict_positions(deeper, source_eval, max_per_stratum):
    rows = deeper.get("rows") or []
    expected = {
        (str(r["checkpoint_role"]), int(r["source_game_index"]), int(r["ply"]))
        for r in rows
    }
    roots = build_root_map(source_eval, expected)
    candidates = defaultdict(dict)

    for row in rows:
        agent = str(row["checkpoint_role"])
        root_key = (agent, int(row["source_game_index"]), int(row["ply"]))
        root = roots[root_key]
        root_state = restore_state(root["state"])
        root_history = [tuple_key(k) for k in root["recent_state_keys"]]

        for branch in BRANCHES:
            state = root_state
            history = list(root_history)
            initial_action = int(row[f"{branch}_action"])
            state = apply_action_index(state, initial_action)
            history.append(state_repetition_key(state))

            for move in row.get(f"{branch}_continuation_moves") or []:
                if state.winner is not None:
                    raise ValueError(f"Unexpected terminal before recorded continuation: {root_key}")
                if int(move["player"]) != int(state.current_player):
                    raise ValueError(f"Player mismatch at reconstructed continuation: {root_key}")
                if int(move["state_ply_after_move"]) != int(state.ply) + 1:
                    raise ValueError(f"Ply mismatch at reconstructed continuation: {root_key}")

                pairs = conflict_pairs(move.get("root_action_diagnostics") or [])
                if pairs:
                    # Stable identity includes the complete state and repetition history.
                    identity_payload = json.dumps(
                        [state.to_dict(), [list_key(k) for k in history]],
                        sort_keys=True, separators=(",", ":"),
                    )
                    position_id = hashlib.sha256(identity_payload.encode()).hexdigest()[:16]
                    stratum = (agent, str(row["variant"]), branch)
                    candidate = {
                        "position_id": position_id,
                        "checkpoint_role": agent,
                        "source_game_index": int(row["source_game_index"]),
                        "source_seed": row.get("source_seed"),
                        "source_root_ply": int(row["ply"]),
                        "source_variant": str(row["variant"]),
                        "source_branch": branch,
                        "continuation_index": int(move["continuation_index"]),
                        "position_ply": int(state.ply),
                        "state": state.to_dict(),
                        "recent_state_keys": [list_key(k) for k in history],
                        "conflict_pairs": pairs,
                        "conflict_pair_count": len(pairs),
                        "max_conflict_severity": pairs[0]["severity"],
                        "source_action_diagnostics": move.get("root_action_diagnostics") or [],
                    }
                    previous = candidates[stratum].get(position_id)
                    if previous is None or candidate["max_conflict_severity"] > previous["max_conflict_severity"]:
                        candidates[stratum][position_id] = candidate

                state = apply_action_index(state, int(move["action"]))
                history.append(state_repetition_key(state))

    selected = []
    selection_report = {}
    for stratum, unique in sorted(candidates.items()):
        ordered = sorted(unique.values(), key=lambda x: (
            -x["max_conflict_severity"], x["source_game_index"],
            x["source_root_ply"], x["continuation_index"], x["position_id"]
        ))
        take = ordered[:max_per_stratum]
        selected.extend(take)
        selection_report["/".join(str(x) for x in stratum)] = {
            "available_unique_positions": len(ordered),
            "selected_positions": len(take),
        }
    return selected, selection_report


def replay_one(model, position, budget, avoid_repetition, c_puct, seed, device):
    torch.manual_seed(seed)
    random.seed(seed)
    state = restore_state(position["state"])
    history = [tuple_key(k) for k in position["recent_state_keys"]]
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
        state, add_root_noise=False, recent_state_keys=history,
        record_selection_trace=False,
    )
    stats = search.last_search_stats
    raw = stats.get("root_action_stats") or {}
    actions = []
    for action_text, item in raw.items():
        actions.append({
            "action": int(action_text),
            "prior": float(item.get("network_prior", item.get("prior", 0.0))),
            "selection_prior": float(item.get("selection_prior", item.get("prior", 0.0))),
            "initial_q": 0.0,
            "final_q": float(item.get("q", 0.0)),
            "visits": int(item.get("visits", 0)),
            "active": bool(item.get("active", False)),
            "repetition_blocked": bool(item.get("repetition_blocked", False)),
        })
    total_visits = sum(a["visits"] for a in actions)
    legal = set(legal_action_indices(state))
    if {a["action"] for a in actions} != legal:
        raise RuntimeError(f"Legal-action coverage mismatch at {position['position_id']}")
    if total_visits != budget or int(stats.get("simulations", -1)) != budget:
        raise RuntimeError(
            f"Visit accounting mismatch at {position['position_id']}: "
            f"expected={budget}, visits={total_visits}, simulations={stats.get('simulations')}"
        )
    if set(policy) != {a["action"] for a in actions}:
        raise RuntimeError("Returned policy and root diagnostics have different action sets")

    selected_action = max(actions, key=lambda a: (a["visits"], -a["action"]))["action"]
    for key, getter in (("rank_prior", lambda a: -a["prior"]),
                        ("rank_q", lambda a: -a["final_q"]),
                        ("rank_visits", lambda a: -a["visits"])):
        ranked = sorted(actions, key=lambda a: (getter(a), a["action"]))
        for rank, action in enumerate(ranked, 1):
            action[key] = rank
    return {
        "selected_action": selected_action,
        "selected_visits": next(a["visits"] for a in actions if a["action"] == selected_action),
        "root_visit_total": total_visits,
        "repetition_blocked_actions": int(stats.get("root_repetition_blocked_actions", 0)),
        "repetition_allowed_actions": stats.get("root_repetition_allowed_actions"),
        "actions": sorted(actions, key=lambda a: a["action"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deeper-report", type=Path, required=True)
    parser.add_argument("--source-evaluation", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--incumbent-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--max-per-stratum", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_per_stratum < 1:
        parser.error("--max-per-stratum must be >= 1")

    deeper = json.loads(args.deeper_report.read_text(encoding="utf-8"))
    source_eval = json.loads(args.source_evaluation.read_text(encoding="utf-8"))
    c_puct = float((deeper.get("continuation") or {}).get("c_puct", 1.5))
    positions, selection_report = reconstruct_conflict_positions(
        deeper, source_eval, args.max_per_stratum
    )
    if not positions:
        raise RuntimeError("No positions with observed prior-Q conflict could be reconstructed")

    model_paths = {
        "candidate": args.candidate_checkpoint,
        "incumbent": args.incumbent_checkpoint,
    }
    models = {}
    for role, path in model_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        model = load_model(path, device=args.device)
        model.eval()
        models[role] = model

    output_rows = []
    total = len(positions) * len(REPLAY_VARIANTS) * len(BUDGETS)
    done = 0
    for position in positions:
        model = models[position["checkpoint_role"]]
        for variant in REPLAY_VARIANTS:
            avoid = variant == "with_repetition_avoidance"
            for budget in BUDGETS:
                done += 1
                # Same seed for both budgets and repetition variants within a position.
                seed = args.seed + int(position["position_id"], 16) % 1_000_000
                result = replay_one(model, position, budget, avoid, c_puct, seed, args.device)
                output_rows.append({
                    "position_id": position["position_id"],
                    "checkpoint_role": position["checkpoint_role"],
                    "source_variant": position["source_variant"],
                    "source_branch": position["source_branch"],
                    "source_game_index": position["source_game_index"],
                    "source_root_ply": position["source_root_ply"],
                    "position_ply": position["position_ply"],
                    "continuation_index": position["continuation_index"],
                    "replay_variant": variant,
                    "budget": budget,
                    "selected_action": result["selected_action"],
                    "selected_visits": result["selected_visits"],
                    "root_visit_total": result["root_visit_total"],
                    "repetition_blocked_actions": result["repetition_blocked_actions"],
                    "repetition_allowed_actions": result["repetition_allowed_actions"],
                    "conflict_pairs": position["conflict_pairs"],
                    "actions": result["actions"],
                })
                print(f"[{done}/{total}] {position['position_id']} {position['checkpoint_role']} "
                      f"{variant} budget={budget} selected={result['selected_action']}", flush=True)

    report = {
        "experiment": "prior_q_conflict_validation",
        "protocol": {
            "budgets": list(BUDGETS),
            "replay_conditions": list(REPLAY_VARIANTS),
            "checkpoint_mapping": {k: str(v) for k, v in model_paths.items()},
            "c_puct": c_puct,
            "selection": "highest raw root visit count; deterministic action-id tie break",
            "q_comparison": "only actions with raw_visits > 0 are eligible for prior-Q conflict selection",
            "paired_replay": "same reconstructed state, chronological repetition history, checkpoint and seed across budgets and replay conditions",
            "root_noise": False,
            "progressive_widening": False,
            "category_balanced_prior": False,
            "category_exploration": 0.0,
            "repetition_penalty": 0.0,
            "pawn_progress_bonus": 0.0,
            "checkpoint_or_policy_changes": False,
            "heuristic_or_puct_changes": False,
        },
        "source": {
            "deeper_report": str(args.deeper_report),
            "source_evaluation": str(args.source_evaluation),
            "candidate_selection": "top conflict severity per checkpoint x source condition x source branch",
            "max_per_stratum": args.max_per_stratum,
            "selection_report": selection_report,
        },
        "reconstructed_positions": positions,
        "results": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    columns = [
        "position_id", "checkpoint_role", "source_variant", "source_branch",
        "source_game_index", "source_root_ply", "position_ply",
        "continuation_index", "replay_variant", "budget", "selected_action",
        "selected_visits", "root_visit_total", "repetition_blocked_actions",
        "repetition_allowed_actions", "action", "prior", "selection_prior",
        "initial_q", "final_q", "visits", "rank_prior", "rank_q",
        "rank_visits", "active", "repetition_blocked",
    ]
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in output_rows:
            for action in row["actions"]:
                writer.writerow({
                    **{k: row[k] for k in columns if k in row},
                    **action,
                })
    print(f"JSON: {args.output}")
    print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()
