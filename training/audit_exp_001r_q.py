#!/usr/bin/env python3
"""EXP-001R-Q: action-level value-head impact audit on the exact 30 replay roots.

Evaluation only. Keeps vanilla PUCT unchanged and reuses the exact probe-row
selection/decode path from evaluate_value_only_root.py.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from quoridor.neural_ai.mcts import MCTS, MOVE_ACTIONS
from quoridor.neural_ai.policy import load_model
from training.evaluate_value_only_root import choose_probe_rows, decode_state


def run_root(model, state, *, simulations: int, c_puct: float, device: str) -> dict:
    search = MCTS(
        model, simulations=simulations, c_puct=c_puct, device=device,
        randomize_selection_ties=False,
        progressive_widening=False, initial_wall_candidates=0,
        category_balanced_prior=False, root_category_prior_mix=0.0,
        opening_goal_directed_prior=False, avoid_recent_state_repetition=False,
        category_exploration=0.0, repetition_penalty=0.0,
        pawn_progress_bonus=0.0, wall_distance_weight=0.0, wall_route_weight=0.0,
    )
    visits = search.search(state, add_root_noise=False)
    stats = search.last_search_stats
    raw = stats["root_action_stats"]
    actions = {}
    for action_text, d in raw.items():
        action = int(action_text)
        if not d.get("active", False):
            raise RuntimeError(f"Vanilla PUCT unexpectedly has inactive root action {action}")
        n = int(d["visits"])
        q = float(d["q"])
        u = float(d["u_score"])
        score = float(d["puct_score"])
        if not math.isclose(score, q + u, rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError(
                f"Unexpected PUCT components for action {action}: "
                f"score={score}, q={q}, u={u}"
            )
        actions[action] = {
            "action": action,
            "category": "pawn" if action < MOVE_ACTIONS else "wall",
            "prior": float(d["network_prior"]),
            "selection_prior": float(d["selection_prior"]),
            "visits": n,
            # MCTS stores Q=0 for an unvisited edge. Preserve that selection
            # value separately from an actually observed edge estimate.
            "q_for_selection": q,
            "q_observed": q if n > 0 else None,
            "u": u,
            "puct_score": score,
        }
    if set(actions) != set(visits):
        raise RuntimeError("Root action diagnostics do not match returned visit keys")
    return {
        "root_value": float(stats["root_value"]),
        "root_visits": int(stats["simulations"]),
        "pawn_visits": int(stats["root_pawn_visits"]),
        "wall_visits": int(stats["root_wall_visits"]),
        "pawn_prior_mass": float(stats["root_network_pawn_prior_mass"]),
        "wall_prior_mass": float(stats["root_network_wall_prior_mass"]),
        "actions": actions,
    }


def ranked(actions: dict[int, dict], score_key: str = "puct_score") -> list[dict]:
    rows = list(actions.values())
    rows.sort(key=lambda x: (-float(x[score_key]), -int(x["visits"]), int(x["action"])))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def rank_map(rows: list[dict]) -> dict[int, int]:
    return {int(row["action"]): int(row["rank"]) for row in rows}


def q_summary(actions: dict[int, dict], category: str) -> dict:
    visited = [a for a in actions.values() if a["category"] == category and a["visits"] > 0]
    if not visited:
        return {"visited_actions": 0, "visits": 0, "mean_q_action_weighted": None,
                "mean_q_visit_weighted": None, "best_q": None, "worst_q": None}
    total_visits = sum(a["visits"] for a in visited)
    return {
        "visited_actions": len(visited),
        "visits": total_visits,
        "mean_q_action_weighted": mean(a["q_observed"] for a in visited),
        "mean_q_visit_weighted": sum(a["q_observed"] * a["visits"] for a in visited) / total_visits,
        "best_q": max(a["q_observed"] for a in visited),
        "worst_q": min(a["q_observed"] for a in visited),
    }


def category_leaders(actions: dict[int, dict]) -> dict:
    result = {}
    for category in ("pawn", "wall"):
        candidates = [a for a in actions.values() if a["category"] == category]
        result[category] = max(
            candidates, key=lambda a: (a["puct_score"], a["visits"], -a["action"])
        ) if candidates else None
    return result


def analyze_position(index: int, sample: dict, incumbent: dict, candidate: dict) -> dict:
    ai, ac = incumbent["actions"], candidate["actions"]
    if set(ai) != set(ac):
        raise RuntimeError(f"Legal action set changed at position {index}")
    # Value-only training must leave policy priors unchanged. Fail closed if
    # this assumption is violated; otherwise PUCT deltas cannot be attributed.
    prior_deltas = [abs(ai[a]["prior"] - ac[a]["prior"]) for a in ai]
    max_prior_delta = max(prior_deltas, default=0.0)
    if max_prior_delta > 1e-7:
        raise RuntimeError(
            f"Policy prior changed at position {index}: max delta={max_prior_delta}"
        )

    ri, rc = ranked(ai), ranked(ac)
    rank_i, rank_c = rank_map(ri), rank_map(rc)

    # Counterfactual decomposition: swap only Q or only final U between runs.
    # Priors are invariant; U also captures changed visit counts/search paths.
    q_swap, u_swap = {}, {}
    for action in ai:
        q_swap[action] = {
            **ac[action],
            "puct_score": ac[action]["q_for_selection"] + ai[action]["u"],
        }
        u_swap[action] = {
            **ai[action],
            "puct_score": ai[action]["q_for_selection"] + ac[action]["u"],
        }
    rq, ru = ranked(q_swap), ranked(u_swap)
    rank_q, rank_u = rank_map(rq), rank_map(ru)

    leaders_i, leaders_c = category_leaders(ai), category_leaders(ac)
    leaders = {}
    for label, actions, ranks in (
        ("incumbent", ai, rank_i), ("candidate", ac, rank_c)
    ):
        pawn, wall = category_leaders(actions)["pawn"], category_leaders(actions)["wall"]
        if pawn is None or wall is None:
            leaders[label] = None
            continue
        q_margin = pawn["q_for_selection"] - wall["q_for_selection"]
        u_margin = pawn["u"] - wall["u"]
        prior_margin = pawn["selection_prior"] - wall["selection_prior"]
        leaders[label] = {
            "best_pawn_action": pawn["action"], "best_wall_action": wall["action"],
            "pawn_rank": ranks[pawn["action"]], "wall_rank": ranks[wall["action"]],
            "pawn_q_observed": pawn["q_observed"], "pawn_visits": pawn["visits"],
            "wall_q_observed": wall["q_observed"], "wall_visits": wall["visits"],
            "q_margin_pawn_minus_wall": q_margin,
            "u_margin_pawn_minus_wall": u_margin,
            "selection_prior_margin_pawn_minus_wall": prior_margin,
            "puct_margin_pawn_minus_wall": pawn["puct_score"] - wall["puct_score"],
            "pawn_loses_on_final_score": pawn["puct_score"] < wall["puct_score"],
            "pawn_loses_on_initial_prior_only": prior_margin < 0,
            "diagnosis": (
                "prior_and_q_disadvantage" if prior_margin < 0 and q_margin < 0
                else "prior_dominant_q_not_worse" if prior_margin < 0 and q_margin >= 0
                else "q_dominant_prior_not_smaller" if prior_margin >= 0 and q_margin < 0
                else "neither_prior_nor_q_disadvantage"
            ),
        }

    action_rows = []
    for action in sorted(ai):
        a, c = ai[action], ac[action]
        action_rows.append({
            "action": action, "category": a["category"],
            "incumbent": {k: a[k] for k in ("prior", "selection_prior", "visits",
                "q_observed", "q_for_selection", "u", "puct_score")},
            "candidate": {k: c[k] for k in ("prior", "selection_prior", "visits",
                "q_observed", "q_for_selection", "u", "puct_score")},
            "delta": {
                "q_for_selection": c["q_for_selection"] - a["q_for_selection"],
                "u": c["u"] - a["u"],
                "puct_score": c["puct_score"] - a["puct_score"],
                "rank_candidate_minus_incumbent": rank_c[action] - rank_i[action],
                "rank_if_candidate_q_only": rank_q[action],
                "rank_if_candidate_u_only": rank_u[action],
            },
            "visits_both_models": a["visits"] > 0 and c["visits"] > 0,
        })

    pawn_rank_moves = [
        row for row in action_rows if row["category"] == "pawn"
    ]
    return {
        "position_index": index,
        "game_id": int(sample["game_id"]),
        "ply": int(sample["ply"]),
        "phase": "opening_0_19" if int(sample["ply"]) < 20 else "post_opening_20_plus",
        "player_to_move": int(sample["player"]),
        "heuristic_winner": int(sample["winner"]),
        "legal_actions": len(ai),
        "max_abs_prior_delta": max_prior_delta,
        "incumbent": {
            "root_value": incumbent["root_value"], "pawn_visits": incumbent["pawn_visits"],
            "wall_visits": incumbent["wall_visits"],
            "pawn_prior_mass": incumbent["pawn_prior_mass"],
            "wall_prior_mass": incumbent["wall_prior_mass"],
            "q_by_category": {c: q_summary(ai, c) for c in ("pawn", "wall")},
            "top_20_by_puct": ri[:20],
        },
        "candidate": {
            "root_value": candidate["root_value"], "pawn_visits": candidate["pawn_visits"],
            "wall_visits": candidate["wall_visits"],
            "pawn_prior_mass": candidate["pawn_prior_mass"],
            "wall_prior_mass": candidate["wall_prior_mass"],
            "q_by_category": {c: q_summary(ac, c) for c in ("pawn", "wall")},
            "top_20_by_puct": rc[:20],
        },
        "category_leader_diagnosis": leaders,
        "pawn_actions": pawn_rank_moves,
        "all_actions": action_rows,
        "ranking_decomposition": {
            "incumbent_q_u": "incumbent Q + incumbent final U",
            "candidate_q_u": "candidate Q + candidate final U",
            "candidate_q_only": "candidate Q + incumbent final U",
            "candidate_u_only": "incumbent Q + candidate final U",
            "pawn_actions": {
                str(row["action"]): {
                    "incumbent_rank": rank_i[row["action"]],
                    "candidate_rank": rank_c[row["action"]],
                    "candidate_q_only_rank": rank_q[row["action"]],
                    "candidate_u_only_rank": rank_u[row["action"]],
                    "rank_delta": rank_c[row["action"]] - rank_i[row["action"]],
                    "q_only_rank_delta": rank_q[row["action"]] - rank_i[row["action"]],
                    "u_only_rank_delta": rank_u[row["action"]] - rank_i[row["action"]],
                } for row in pawn_rank_moves
            },
        },
    }


def summarize(results: list[dict]) -> dict:
    summary = {}
    for phase in ("opening_0_19", "post_opening_20_plus", "all"):
        rows = results if phase == "all" else [r for r in results if r["phase"] == phase]
        if not rows:
            summary[phase] = {"positions": 0}
            continue
        q = {"incumbent": {}, "candidate": {}}
        for model in ("incumbent", "candidate"):
            for category in ("pawn", "wall"):
                vals = [r[model]["q_by_category"][category] for r in rows]
                weighted = [(v["mean_q_visit_weighted"], v["visits"]) for v in vals
                            if v["mean_q_visit_weighted"] is not None]
                q[model][category] = {
                    "positions_with_visited_actions": len(weighted),
                    "total_edge_visits": sum(n for _, n in weighted),
                    "mean_of_position_visit_weighted_q": (
                        mean(v for v, _ in weighted) if weighted else None
                    ),
                    "pooled_visit_weighted_q": (
                        sum(v * n for v, n in weighted) / sum(n for _, n in weighted)
                        if weighted else None
                    ),
                }
        rank_deltas = [
            d["rank_delta"]
            for r in rows
            for d in r["ranking_decomposition"]["pawn_actions"].values()
        ]
        q_only = [
            d["q_only_rank_delta"]
            for r in rows
            for d in r["ranking_decomposition"]["pawn_actions"].values()
        ]
        u_only = [
            d["u_only_rank_delta"]
            for r in rows
            for d in r["ranking_decomposition"]["pawn_actions"].values()
        ]
        diagnoses = defaultdict(int)
        for r in rows:
            for model, item in r["category_leader_diagnosis"].items():
                if item:
                    diagnoses[f"{model}:{item['diagnosis']}"] += 1
        summary[phase] = {
            "positions": len(rows),
            "mean_pawn_wall_prior_mass": {
                model: {
                    category: mean(r[model][f"{category}_prior_mass"] for r in rows)
                    for category in ("pawn", "wall")
                } for model in ("incumbent", "candidate")
            },
            "q_by_category": q,
            "pawn_action_rank_changes": {
                "action_observations": len(rank_deltas),
                "candidate_rank_improved": sum(d < 0 for d in rank_deltas),
                "candidate_rank_worsened": sum(d > 0 for d in rank_deltas),
                "unchanged": sum(d == 0 for d in rank_deltas),
                "mean_rank_delta_candidate_minus_incumbent": mean(rank_deltas) if rank_deltas else None,
                "q_only_improved": sum(d < 0 for d in q_only),
                "q_only_worsened": sum(d > 0 for d in q_only),
                "u_only_improved": sum(d < 0 for d in u_only),
                "u_only_worsened": sum(d > 0 for d in u_only),
            },
            "best_category_action_diagnoses": dict(diagnoses),
            "pawn_selected_by_visits": {
                model: sum(r[model]["pawn_visits"] > 0 for r in rows)
                for model in ("incumbent", "candidate")
            },
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--max-positions", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.simulations != 32 or args.c_puct != 1.5:
        print("WARNING: nonstandard EXP-001R-Q budget/c_puct; compare cautiously")

    samples = choose_probe_rows(args.dataset, args.training_report, args.max_positions)
    if len(samples) != 30:
        raise RuntimeError(f"Expected the same 30 replay roots, found {len(samples)}")
    incumbent_model = load_model(args.incumbent, device=args.device)
    candidate_model = load_model(args.candidate, device=args.device)

    results = []
    for index, sample in enumerate(samples, 1):
        state = decode_state(sample["state"])
        incumbent = run_root(incumbent_model, state, simulations=args.simulations,
                             c_puct=args.c_puct, device=args.device)
        candidate = run_root(candidate_model, state, simulations=args.simulations,
                             c_puct=args.c_puct, device=args.device)
        results.append(analyze_position(index, sample, incumbent, candidate))
        leaders = results[-1]["category_leader_diagnosis"]
        print(
            f"[{index}/30] game={sample['game_id']} ply={sample['ply']} "
            f"pawnQ={q_summary(incumbent['actions'], 'pawn')['mean_q_visit_weighted']}->"
            f"{q_summary(candidate['actions'], 'pawn')['mean_q_visit_weighted']} "
            f"pawnWall={leaders['incumbent']['diagnosis']}->"
            f"{leaders['candidate']['diagnosis']}",
            flush=True,
        )

    output = {
        "experiment": "EXP-001R-Q-action-level-value-head-audit",
        "models": {
            "incumbent": str(args.incumbent),
            "candidate": str(args.candidate),
        },
        "replay_source": str(args.dataset),
        "probe_selection": "exact choose_probe_rows() used by EXP-001R root_puct_comparison",
        "positions": len(results),
        "config": {
            "simulations": args.simulations, "c_puct": args.c_puct,
            "progressive_widening": False, "root_noise": False,
            "category_balanced_prior": False, "root_category_prior_mix": 0.0,
            "opening_goal_directed_prior": False, "avoid_recent_state_repetition": False,
            "category_exploration": 0.0, "repetition_penalty": 0.0,
            "pawn_progress_bonus": 0.0, "wall_distance_weight": 0.0,
            "wall_route_weight": 0.0, "randomize_selection_ties": False,
        },
        "method": {
            "vanilla_puct_unchanged": True,
            "policy_prior_invariance_asserted": True,
            "unvisited_q_reported_as_null_observed_with_zero_selection_default": True,
            "rank_order": "descending PUCT score, then visits, then ascending action id",
            "counterfactual_rank_decomposition": [
                "incumbent Q + incumbent final U",
                "candidate Q + candidate final U",
                "candidate Q + incumbent final U",
                "incumbent Q + candidate final U",
            ],
            "final_U_contains_prior_and_visit_count_effects": True,
        },
        "summary": summarize(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"EXP-001R-Q audit saved: {args.output}")


if __name__ == "__main__":
    main()
