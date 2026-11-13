#!/usr/bin/env python3
"""Trace post-wall pawn regression decisions and replay identical roots.

This is an inference-only diagnostic. It never trains or modifies a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

import torch

from quoridor.engine import GameState, state_repetition_key
from quoridor.neural_ai.encoder import MOVE_ACTIONS, apply_action_index, encode_state
from quoridor.neural_ai.mcts import MCTS
from quoridor.neural_ai.policy import load_model


def reconstruct_positions(source: dict, source_agent: str) -> list[dict]:
    """Replay each recorded game from the initial state and retain eligible roots."""
    positions = []
    for game_index, game in enumerate(source.get("games", [])):
        state = GameState.initial()
        history = [state_repetition_key(state)]
        for action_row in sorted(game.get("action_trace", []), key=lambda row: row["ply"]):
            if state.ply != int(action_row["ply"]):
                raise ValueError(
                    f"Non-contiguous trace in game {game_index}: state ply={state.ply}, "
                    f"trace ply={action_row['ply']}"
                )
            diagnostics = action_row.get("root_diagnostics") or {}
            pawn_actions = diagnostics.get("pawn_actions") or {}
            selected = action_row.get("selected_action_root_stats") or {}
            eligible_progress = [
                item for item in pawn_actions.values()
                if item.get("distance_delta") is not None
                and item["distance_delta"] > 0
                and not item.get("repetition_blocked", False)
            ]
            is_target = (
                state.walls_remaining == (0, 0)
                and action_row.get("agent") == source_agent
                and action_row.get("category") == "pawn"
                and selected.get("distance_delta") is not None
                and selected["distance_delta"] < 0
                and bool(eligible_progress)
            )
            if is_target:
                positions.append({
                    "source_game_index": game_index,
                    "source_seed": game.get("seed"),
                    "source_agent": action_row.get("agent"),
                    "ply": state.ply,
                    "state": state.to_dict(),
                    "recent_state_keys": [list_key(key) for key in history],
                    "source_selected_action": int(action_row["action"]),
                    "source_selected_distance_delta": selected["distance_delta"],
                    "available_progress_actions": [
                        int(action) for action, item in pawn_actions.items()
                        if item.get("distance_delta") is not None
                        and item["distance_delta"] > 0
                        and not item.get("repetition_blocked", False)
                    ],
                })
            state = apply_action_index(state, int(action_row["action"]))
            history.append(state_repetition_key(state))
    return positions


def list_key(key: tuple) -> list:
    pawns, walls, remaining, player, winner = key
    return [
        [list(p) for p in pawns],
        [list(w) for w in walls],
        list(remaining),
        player,
        winner,
    ]


def tuple_key(value: list) -> tuple:
    pawns, walls, remaining, player, winner = value
    return (
        tuple(tuple(p) for p in pawns),
        tuple(tuple(w) for w in walls),
        tuple(remaining),
        player,
        winner,
    )


def restore_state(data: dict) -> GameState:
    from quoridor.engine import Wall
    return GameState(
        pawns=tuple((int(p["x"]), int(p["y"])) for p in data["pawns"]),
        walls=frozenset(Wall(w["orientation"], int(w["x"]), int(w["y"])) for w in data["walls"]),
        walls_remaining=tuple(int(v) for v in data["walls_remaining"]),
        current_player=int(data["current_player"]),
        winner=data.get("winner"),
        ply=int(data["ply"]),
    )


def root_trace_summary(stats: dict) -> dict:
    """Classify whether regression leads initially, after Q updates, or by visits."""
    rows = [row for row in stats.get("selection_trace", []) if row.get("selection_depth") == 0]
    first = rows[0] if rows else None
    first_regression_overtake = None
    initial_q_gap = None
    if first:
        candidates = first.get("candidate_pawn_actions", [])
        regs = [x for x in candidates if x.get("distance_delta") is not None and x["distance_delta"] < 0]
        progs = [x for x in candidates if x.get("distance_delta") is not None and x["distance_delta"] > 0]
        if regs and progs:
            initial_q_gap = max(regs, key=lambda x: x["puct_score"])["q"] - max(progs, key=lambda x: x["puct_score"])["q"]

    for row in rows:
        candidates = row.get("candidate_pawn_actions", [])
        regressions = [x for x in candidates if x.get("distance_delta") is not None and x["distance_delta"] < 0]
        progresses = [x for x in candidates if x.get("distance_delta") is not None and x["distance_delta"] > 0]
        if regressions and progresses:
            best_reg = max(regressions, key=lambda x: x["puct_score"])
            best_prog = max(progresses, key=lambda x: x["puct_score"])
            if best_reg["puct_score"] > best_prog["puct_score"]:
                first_regression_overtake = {
                    "simulation": row["simulation"],
                    "best_regression_before_selection": best_reg,
                    "best_progress_before_selection": best_prog,
                    "q_gap_regression_minus_progress": best_reg["q"] - best_prog["q"],
                    "initial_q_gap_regression_minus_progress": initial_q_gap,
                    "q_gap_improved_since_initial": (
                        initial_q_gap is not None
                        and (best_reg["q"] - best_prog["q"]) > initial_q_gap + 1e-12
                    ),
                }
                break

    actions = stats.get("root_action_stats", {})
    pawn_rows = [(int(a), v) for a, v in actions.items() if v.get("category") == "pawn"]
    regression_rows = [(a, v) for a, v in pawn_rows if v.get("distance_delta") is not None and v["distance_delta"] < 0]
    progress_rows = [(a, v) for a, v in pawn_rows if v.get("distance_delta") is not None and v["distance_delta"] > 0]
    selected = max(pawn_rows, key=lambda item: (item[1].get("visits", 0), item[1].get("prior", 0.0)), default=None)
    best_reg_final = max(regression_rows, key=lambda item: item[1].get("puct_score") or -1e9, default=None)
    best_prog_final = max(progress_rows, key=lambda item: item[1].get("puct_score") or -1e9, default=None)

    category = "insufficient_candidates"
    if first and regression_rows and progress_rows:
        initial = first.get("candidate_pawn_actions", [])
        initial_reg = [x for x in initial if x.get("distance_delta") is not None and x["distance_delta"] < 0]
        initial_prog = [x for x in initial if x.get("distance_delta") is not None and x["distance_delta"] > 0]
        initial_reg_wins = bool(initial_reg and initial_prog) and (
            max(x["puct_score"] for x in initial_reg) > max(x["puct_score"] for x in initial_prog)
        )
        selected_is_regression = selected is not None and selected[1].get("distance_delta", 0) < 0
        final_reg_wins = (
            best_reg_final is not None and best_prog_final is not None
            and best_reg_final[1].get("puct_score", -1e9) > best_prog_final[1].get("puct_score", -1e9)
        )
        if initial_reg_wins:
            category = "regression_ahead_on_first_selection_prior_driven"
        elif first_regression_overtake and first_regression_overtake["q_gap_improved_since_initial"]:
            category = "regression_overtakes_after_q_improves"
        elif selected_is_regression and not final_reg_wins:
            category = "regression_selected_by_visits_despite_lower_final_puct"
        elif first_regression_overtake:
            category = "regression_overtakes_without_q_gap_improvement"
        else:
            category = "regression_never_overtakes_progress"

    return {
        "root_selection_steps": len(rows),
        "first_root_snapshot": first,
        "first_regression_overtake": first_regression_overtake,
        "classification": category,
        "final_selected_pawn_action": (
            {"action": selected[0], **selected[1]} if selected else None
        ),
        "best_final_regression": (
            {"action": best_reg_final[0], **best_reg_final[1]} if best_reg_final else None
        ),
        "best_final_progress": (
            {"action": best_prog_final[0], **best_prog_final[1]} if best_prog_final else None
        ),
        "selection_trace": rows,
    }

def response_probe(model, state: GameState, action: int, budget: int, c_puct: float, device: str) -> dict:
    """Let the opponent search/respond, then evaluate the resulting root-player state."""
    after_our_move = apply_action_index(state, action)
    if after_our_move.winner is not None:
        return {
            "opponent_reply": None,
            "state_after_reply": after_our_move.to_dict(),
            "value_for_original_player": (
                1.0 if after_our_move.winner == state.current_player else -1.0
            ),
            "terminal": True,
            "winner": after_our_move.winner,
        }
    reply_search = MCTS(
        model, simulations=budget, c_puct=c_puct, device=device,
        progressive_widening=False, category_exploration=0.0,
        repetition_penalty=0.0, pawn_progress_bonus=0.0,
        avoid_recent_state_repetition=False, randomize_selection_ties=False,
    )
    visits = reply_search.search(after_our_move, add_root_noise=False, record_selection_trace=False)
    if not visits:
        return {"opponent_reply": None, "terminal": after_our_move.winner is not None}
    reply = max(visits, key=visits.get)
    after_reply = apply_action_index(after_our_move, reply)
    if after_reply.winner is not None:
        value_for_original = 1.0 if after_reply.winner == state.current_player else -1.0
    else:
        with torch.inference_mode():
            tensor = torch.tensor(encode_state(after_reply), dtype=torch.float32, device=device).unsqueeze(0)
            _policy, value = model(tensor)
        value_for_original = float(value[0].item())
    return {
        "opponent_reply": reply,
        "state_after_reply": after_reply.to_dict(),
        "value_for_original_player": value_for_original,
        "terminal": after_reply.winner is not None,
        "winner": after_reply.winner,
    }


def run_replay(model, position: dict, *, budget: int, c_puct: float, device: str,
               response_probe_enabled: bool) -> dict:
    state = restore_state(position["state"])
    history = [tuple_key(item) for item in position["recent_state_keys"]]
    variants = {}
    for avoid in (False, True):
        torch.manual_seed(0)
        random.seed(0)
        search = MCTS(
            model, simulations=budget, c_puct=c_puct, device=device,
            randomize_selection_ties=False, progressive_widening=False,
            category_balanced_prior=False, root_category_prior_mix=0.0,
            opening_goal_directed_prior=False,
            avoid_recent_state_repetition=avoid,
            category_exploration=0.0, repetition_penalty=0.0,
            pawn_progress_bonus=0.0,
        )
        visits = search.search(
            state, add_root_noise=False, recent_state_keys=history,
            record_selection_trace=True,
        )
        summary = root_trace_summary(search.last_search_stats)
        summary.update({
            "avoid_recent_state_repetition": avoid,
            "root_repetition_blocked_actions": search.last_search_stats.get("root_repetition_blocked_actions", 0),
            "root_repetition_allowed_actions": search.last_search_stats.get("root_repetition_allowed_actions"),
            "root_repetition_fallback_all_actions": search.last_search_stats.get("root_repetition_fallback_all_actions", False),
            "root_visits": visits,
        })
        if response_probe_enabled:
            pawn_stats = search.last_search_stats.get("root_action_stats", {})
            fallback_all = search.last_search_stats.get("root_repetition_fallback_all_actions", False)
            eligible = lambda x: (not avoid) or fallback_all or not x.get("repetition_blocked", False)
            regressions = [
                (int(a), x) for a, x in pawn_stats.items()
                if x.get("category") == "pawn" and x.get("distance_delta") is not None
                and x["distance_delta"] < 0 and eligible(x)
            ]
            progresses = [
                (int(a), x) for a, x in pawn_stats.items()
                if x.get("category") == "pawn" and x.get("distance_delta") is not None
                and x["distance_delta"] > 0 and eligible(x)
            ]
            best_reg = max(regressions, key=lambda item: item[1].get("visits", 0), default=None)
            best_prog = max(progresses, key=lambda item: item[1].get("visits", 0), default=None)
            summary["response_probe"] = {
                "best_regression": (
                    {"action": best_reg[0], **response_probe(model, state, best_reg[0], budget, c_puct, device)}
                    if best_reg else None
                ),
                "best_progress": (
                    {"action": best_prog[0], **response_probe(model, state, best_prog[0], budget, c_puct, device)}
                    if best_prog else None
                ),
                "warning": "One-ply opponent MCTS response plus neural value is a diagnostic proxy, not a solved minimax value.",
            }
        variants["with_repetition_avoidance" if avoid else "vanilla_puct"] = summary
    return {
        "source": {k: v for k, v in position.items() if k not in ("state", "recent_state_keys")},
        "full_state": position["state"],
        "variants": variants,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_checkpoint", type=Path)
    parser.add_argument("incumbent_checkpoint", type=Path)
    parser.add_argument("source_evaluation", type=Path,
                        help="Paired-evaluation JSON containing per-game action_trace.")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--c-puct", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-positions", type=int, default=0,
                        help="Optional cap after chronological extraction; zero means all eligible positions.")
    parser.add_argument("--response-probe-limit", type=int, default=20,
                        help="Run opponent-response probes on the first N roots; zero disables them.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.simulations < 1 or args.threads < 1 or args.max_positions < 0 or args.response_probe_limit < 0:
        parser.error("simulation/thread counts must be positive and limits non-negative")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a fresh path.")

    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    source = json.loads(args.source_evaluation.read_text(encoding="utf-8"))
    config = source.get("config", {})
    c_puct = float(args.c_puct if args.c_puct is not None else config.get("c_puct", 1.5))
    agents = ("candidate", "incumbent")
    positions = []
    for agent in agents:
        positions.extend(reconstruct_positions(source, agent))
    positions.sort(key=lambda item: (item["source_game_index"], item["ply"], item["source_agent"]))
    if args.max_positions:
        positions = positions[:args.max_positions]
    if not positions:
        parser.error("No post-wall regression positions with at least one unblocked progress action were found.")

    models = {
        "candidate": load_model(args.candidate_checkpoint, device=args.device),
        "incumbent": load_model(args.incumbent_checkpoint, device=args.device),
    }
    for model in models.values():
        model.eval()
    runs = []
    for index, position in enumerate(positions, 1):
        print(
            f"[{index}/{len(positions)}] game={position['source_game_index']} "
            f"agent={position['source_agent']} ply={position['ply']}",
            flush=True,
        )
        model = models[position["source_agent"]]
        runs.append(run_replay(
            model, position, budget=args.simulations, c_puct=c_puct,
            device=args.device,
            response_probe_enabled=index <= args.response_probe_limit,
        ))

    classifications = {}
    for variant in ("vanilla_puct", "with_repetition_avoidance"):
        counts = {}
        for run in runs:
            label = run["variants"][variant]["classification"]
            counts[label] = counts.get(label, 0) + 1
        classifications[variant] = counts
    report = {
        "experiment": "post_wall_pawn_regression_selection_audit",
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "incumbent_checkpoint": str(args.incumbent_checkpoint),
        "source_evaluation": str(args.source_evaluation),
        "positions": len(runs),
        "simulations": args.simulations,
        "c_puct": c_puct,
        "settings": {
            "same_checkpoint_for_both_variants": True,
            "vanilla_puct_formula_unchanged": True,
            "policy_prior_unchanged": True,
            "value_head_unchanged": True,
            "progressive_widening": False,
            "root_dirichlet_noise": False,
            "category_exploration": 0.0,
            "repetition_penalty": 0.0,
            "pawn_progress_bonus": 0.0,
            "paired_root_states": True,
            "repetition_avoidance_changes_only_root_eligible_action_set": True,
        },
        "classification_counts": classifications,
        "response_probe_limit": args.response_probe_limit,
        "response_probe_interpretation": (
            "For selected best-visit regression and progress moves, the opponent receives "
            "the same vanilla PUCT response budget; the resulting state is evaluated by "
            "the unchanged network from the original player's perspective. This is a "
            "one-ply tactical proxy, not minimax proof."
        ),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Post-wall regression audit: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
