#!/usr/bin/env python3
"""EXP: Budget Scaling Decision Quality Validation.

Paired 32/128/256 MCTS replays on the exact reconstructed positions from the
prior-Q conflict experiment. Decision quality is measured with a neural-free,
engine-only immediate shortest-path differential. This is an independent
structural proxy, NOT a game-theoretic value or a substitute for match outcomes.
No training or checkpoint writes occur.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import torch

from quoridor.engine import shortest_path_metrics
from quoridor.neural_ai.encoder import apply_action_index, legal_action_indices
from quoridor.neural_ai.mcts import MCTS
from training.audit_post_wall_pawn_regression import load_model, restore_state, tuple_key

BUDGETS = (32, 128, 256)
REPLAY_VARIANTS = ("vanilla_puct", "with_repetition_avoidance")
C_PUCT = 1.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def engine_path_score(state, root_player: int) -> float:
    """Neural-free score from root player's perspective; higher is better."""
    if state.winner is not None:
        return 1.0 if state.winner == root_player else -1.0
    own_distance, _own_routes = shortest_path_metrics(state, root_player)
    opp_distance, _opp_routes = shortest_path_metrics(state, 1 - root_player)
    if own_distance is None or opp_distance is None:
        raise RuntimeError("A legal Quoridor state unexpectedly has no path to goal")
    # Pure distance differential; no tunable route-count weighting.
    return float(opp_distance - own_distance)


def score_all_legal_actions(state) -> dict[int, float]:
    root_player = int(state.current_player)
    scores = {}
    for action in legal_action_indices(state):
        child = apply_action_index(state, int(action))
        scores[int(action)] = engine_path_score(child, root_player)
    if not scores:
        raise RuntimeError("No legal actions in a non-terminal reconstructed position")
    return scores


def rank_visited(actions: list[dict], field: str, descending: bool = True) -> dict[int, int]:
    visited = [a for a in actions if int(a["visits"]) > 0]
    ordered = sorted(
        visited,
        key=lambda a: ((-float(a[field])) if descending else float(a[field]), int(a["action"])),
    )
    return {int(row["action"]): rank for rank, row in enumerate(ordered, 1)}


def replay_one(model, position: dict, budget: int, avoid_repetition: bool,
               seed: int, device: str, engine_scores: dict[int, float]) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    state = restore_state(position["state"])
    history = [tuple_key(k) for k in position["recent_state_keys"]]
    search = MCTS(
        model,
        simulations=budget,
        c_puct=C_PUCT,
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
        action = int(action_text)
        if action not in engine_scores:
            raise RuntimeError(f"Engine evaluation missing legal action {action}")
        actions.append({
            "action": action,
            "prior": float(item.get("network_prior", item.get("prior", 0.0))),
            "final_q": float(item.get("q", 0.0)),
            "visits": int(item.get("visits", 0)),
            "active": bool(item.get("active", False)),
            "repetition_blocked": bool(item.get("repetition_blocked", False)),
            "engine_path_score": engine_scores[action],
        })
    legal = set(legal_action_indices(state))
    if {a["action"] for a in actions} != legal or legal != set(engine_scores):
        raise RuntimeError(f"Legal-action coverage mismatch: {position['position_id']}")
    total_visits = sum(a["visits"] for a in actions)
    if total_visits != budget or int(stats.get("simulations", -1)) != budget:
        raise RuntimeError(
            f"Visit accounting mismatch: expected={budget}, visits={total_visits}, "
            f"simulations={stats.get('simulations')}"
        )
    if set(policy) != legal:
        raise RuntimeError("Returned policy and root diagnostics have different action sets")

    selected = max(actions, key=lambda a: (a["visits"], -a["action"]))
    best_engine_score = max(engine_scores.values())
    best_engine_actions = sorted(a for a, score in engine_scores.items()
                                 if score == best_engine_score)
    prior_ranks = rank_visited(actions, "prior", descending=True)
    q_ranks = rank_visited(actions, "final_q", descending=True)
    selected_action = int(selected["action"])
    selected["rank_prior_visited"] = prior_ranks[selected_action]
    selected["rank_q_visited"] = q_ranks[selected_action]
    selected["rank_visits"] = 1
    selected["engine_regret"] = best_engine_score - engine_scores[selected_action]

    return {
        "selected_action": selected_action,
        "selected_visits": int(selected["visits"]),
        "selected_prior": float(selected["prior"]),
        "selected_q": float(selected["final_q"]),
        "selected_prior_rank_visited": prior_ranks[selected_action],
        "selected_q_rank_visited": q_ranks[selected_action],
        "selected_engine_path_score": engine_scores[selected_action],
        "best_engine_path_score": best_engine_score,
        "best_engine_actions": best_engine_actions,
        "engine_regret": best_engine_score - engine_scores[selected_action],
        "root_visit_total": total_visits,
        "repetition_blocked_actions": int(stats.get("root_repetition_blocked_actions", 0)),
        "actions": sorted(actions, key=lambda a: a["action"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-q-report", type=Path, required=True,
                        help="Validated prior_q_conflict_validation JSON containing 40 source records, 31 unique states, and 32 position-checkpoint-role pairs.")
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--incumbent-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    source = json.loads(args.prior_q_report.read_text(encoding="utf-8"))
    if source.get("experiment") != "prior_q_conflict_validation":
        raise ValueError("Input is not prior_q_conflict_validation")
    positions = source.get("reconstructed_positions") or []
    results_source = source.get("results") or []
    if len(positions) != 40:
        raise ValueError(f"Expected 40 source position records; got {len(positions)}")
    if len(results_source) != 160:
        raise ValueError(f"Expected 160 validated prior-Q replay rows; got {len(results_source)}")

    # position_id identifies the state plus repetition history, but a state can
    # legitimately occur in source strata assigned to different checkpoints.
    # The experimental unit is therefore (position_id, checkpoint_role), not
    # position_id alone. Validate state/history identity across all roles first.
    by_position = {}
    for record in positions:
        pid = str(record["position_id"])
        by_position.setdefault(pid, []).append(record)

    unique_positions = []
    for pid, records in sorted(by_position.items()):
        canonical_identity = json.dumps(
            [records[0]["state"], records[0]["recent_state_keys"]],
            sort_keys=True, separators=(",", ":"),
        )
        for record in records[1:]:
            identity = json.dumps(
                [record["state"], record["recent_state_keys"]],
                sort_keys=True, separators=(",", ":"),
            )
            if identity != canonical_identity:
                raise ValueError(f"Position ID collision with different state/history: {pid}")

        by_role = {}
        for record in records:
            role = str(record["checkpoint_role"])
            by_role.setdefault(role, []).append(record)

        for role, role_records in sorted(by_role.items()):
            canonical = role_records[0]
            unique = dict(canonical)
            unique["source_records"] = [
                {
                    "checkpoint_role": row.get("checkpoint_role"),
                    "source_variant": row.get("source_variant"),
                    "source_branch": row.get("source_branch"),
                    "source_game_index": row.get("source_game_index"),
                    "source_root_ply": row.get("source_root_ply"),
                    "position_ply": row.get("position_ply"),
                    "continuation_index": row.get("continuation_index"),
                }
                for row in role_records
            ]
            unique_positions.append(unique)

    ids = [str(p["position_id"]) for p in positions]
    unique_ids = sorted(by_position)
    if len(by_position) != 31 or len(unique_positions) != 32:
        raise ValueError(
            "Expected 31 unique state/history identities and 32 "
            f"position-checkpoint-role pairs; got {len(by_position)} states "
            f"and {len(unique_positions)} pairs"
        )

    model_paths = {
        "candidate": args.candidate_checkpoint,
        "incumbent": args.incumbent_checkpoint,
    }
    models = {}
    checkpoint_hashes = {}
    for role, path in model_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint_hashes[role] = sha256_file(path)
        model = load_model(path, device=args.device)
        model.eval()
        models[role] = model

    output_rows = []
    engine_cache = {}
    total = len(unique_positions) * len(REPLAY_VARIANTS) * len(BUDGETS)
    done = 0
    for position in positions:
        position_id = str(position["position_id"])
        role = str(position["checkpoint_role"])
        if role not in models:
            raise ValueError(f"Unknown checkpoint role {role}")
        state = restore_state(position["state"])
        # Engine-only evaluation is computed once per state and reused for every paired replay.
        engine_scores = score_all_legal_actions(state)
        engine_cache[position_id] = {
            "legal_action_count": len(engine_scores),
            "best_score": max(engine_scores.values()),
            "best_actions": sorted(a for a, v in engine_scores.items()
                                   if v == max(engine_scores.values())),
        }
        seed = args.seed + int(position_id, 16) % 1_000_000
        model = models[role]
        for variant in REPLAY_VARIANTS:
            avoid = variant == "with_repetition_avoidance"
            for budget in BUDGETS:
                done += 1
                result = replay_one(model, position, budget, avoid, seed,
                                    args.device, engine_scores)
                output_rows.append({
                    "position_id": position_id,
                    "checkpoint_role": role,
                    "checkpoint_sha256": checkpoint_hashes[role],
                    "source_records": position["source_records"],
                    "seed": seed,
                    "replay_variant": variant,
                    "budget": budget,
                    **{k: v for k, v in result.items() if k != "actions"},
                    "actions": result["actions"],
                })
                print(f"[{done}/{total}] {position_id} {role} {variant} "
                      f"budget={budget} selected={result['selected_action']} "
                      f"engine_regret={result['engine_regret']:.3f}", flush=True)

    report = {
        "experiment": "budget_scaling_decision_quality_validation",
        "protocol": {
            "budgets": list(BUDGETS),
            "replay_conditions": list(REPLAY_VARIANTS),
            "positions_source": "reconstructed_positions from validated prior_q_conflict_validation JSON",
            "source_record_count": len(positions),
            "unique_state_count": len(by_position),
            "position_checkpoint_role_pair_count": len(unique_positions),
            "expected_replay_count": total,
            "checkpoint_mapping": {k: str(v) for k, v in model_paths.items()},
            "checkpoint_sha256": checkpoint_hashes,
            "base_seed": args.seed,
            "seed_rule": "base_seed + int(position_id,16) % 1_000_000; same seed across budgets and replay conditions for each position",
            "c_puct": C_PUCT,
            "selection": "highest raw root visit count; deterministic action-id tie break",
            "root_noise": False,
            "progressive_widening": False,
            "category_balanced_prior": False,
            "root_category_prior_mix": 0.0,
            "opening_goal_directed_prior": False,
            "category_exploration": 0.0,
            "repetition_penalty": 0.0,
            "pawn_progress_bonus": 0.0,
            "checkpoint_or_policy_changes": False,
            "heuristic_or_puct_changes": False,
            "independent_evaluator": {
                "name": "engine_only_immediate_shortest_path_differential",
                "formula": "opponent_shortest_distance_after_action - root_player_shortest_distance_after_action",
                "higher_is_better": True,
                "regret": "best_engine_path_score_over_all_legal_root_actions - selected_action_engine_path_score",
                "uses_neural_network": False,
                "limitations": [
                    "One-ply structural proxy only; does not model opponent replies or long-term wall utility.",
                    "Not a game-theoretic value, calibrated win probability, or actual match outcome.",
                    "A zero regret means tied for best under this specific shortest-path metric only."
                ]
            }
        },
        "source_validation": {
            "source_experiment": source.get("experiment"),
            "source_position_record_count": len(positions),
            "source_result_count": len(results_source),
            "unique_state_count": len(by_position),
            "unique_position_checkpoint_role_pair_count": len(unique_positions),
            "duplicate_source_records_within_role": len(positions) - len(unique_positions),
            "source_position_ids": ids,
            "unique_state_position_ids": unique_ids,
        },
        "engine_evaluations": engine_cache,
        "results": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    columns = [
        "position_id", "checkpoint_role", "source_records",
        "seed", "replay_variant", "budget",
        "selected_action", "selected_visits", "selected_prior", "selected_q",
        "selected_prior_rank_visited", "selected_q_rank_visited",
        "selected_engine_path_score", "best_engine_path_score", "engine_regret",
        "root_visit_total", "repetition_blocked_actions", "action", "prior",
        "final_q", "visits", "active", "repetition_blocked", "engine_path_score",
        "engine_regret",
    ]
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in output_rows:
            for action in row["actions"]:
                writer.writerow({
                    **{k: row[k] for k in columns if k in row},
                    "source_records": json.dumps(row["source_records"], separators=(",", ":")),
                    **action,
                    "engine_regret": row["best_engine_path_score"] - action["engine_path_score"],
                })
    print(f"JSON: {args.output}")
    print(f"CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
