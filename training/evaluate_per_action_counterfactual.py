#!/usr/bin/env python3
"""Per-action counterfactual evaluation for two opponent-threat Quoridor states.

Evaluation-only: does not train, edit checkpoints, or modify the MCTS/PUCT
implementation. Every legal root action is forced for an equal simulation
budget; normal vanilla PUCT is used only for continuations below that action.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from statistics import mean

from quoridor.engine import (
    GameState,
    shortest_distance,
    validate_state,
)
from quoridor.neural_ai.encoder import (
    MOVE_ACTIONS,
    apply_action_index,
    decode_action,
    legal_action_indices,
)
from quoridor.neural_ai.mcts import Edge, MCTS, Node
from quoridor.neural_ai.policy import load_model


def build_threat_position(player0_x: int) -> tuple[GameState, list[int]]:
    """Build a reproducible legal position: opponent can win on the next reply."""
    if player0_x not in (3, 5):
        raise ValueError("player0_x must be 3 or 5")
    state = GameState.initial()
    sequence: list[int] = []
    p0_targets = [(player0_x, y) for y in range(8, 1, -1)]
    p1_targets = [(4, y) for y in range(1, 8)]
    for p0_target, p1_target in zip(p0_targets, p1_targets):
        for target in (p0_target, p1_target):
            action = target[1] * 9 + target[0]
            state = apply_action_index(state, action)
            sequence.append(action)
    validate_state(state)
    if state.current_player != 0:
        raise RuntimeError("Scripted position must leave player 0 to move")
    if shortest_distance(state, 1 - state.current_player) != 1:
        raise RuntimeError("Scripted opponent threat is not exactly one step")
    if not has_immediate_winning_move(state, 1 - state.current_player):
        raise RuntimeError("Opponent does not have a legal immediate winning move")
    return state, sequence


def action_payload(action: int) -> dict:
    kind, payload = decode_action(action)
    if kind == "move":
        return {"type": "pawn", "to": list(payload)}
    return {
        "type": "wall",
        "orientation": payload.orientation,
        "x": payload.x,
        "y": payload.y,
    }


def has_immediate_winning_move(state: GameState, player: int) -> bool:
    """Check whether player has a legal one-action win from this board."""
    if state.winner is not None:
        return state.winner == player
    hypothetical = replace(state, current_player=player)
    for action in legal_action_indices(hypothetical):
        if action >= MOVE_ACTIONS:
            continue
        child = apply_action_index(hypothetical, action)
        if child.winner == player:
            return True
    return False


def make_search(model, *, budget: int, c_puct: float, device: str) -> MCTS:
    return MCTS(
        model,
        simulations=budget,
        c_puct=c_puct,
        device=device,
        randomize_selection_ties=False,
        progressive_widening=False,
        category_balanced_prior=False,
        root_category_prior_mix=0.0,
        opening_goal_directed_prior=False,
        avoid_recent_state_repetition=False,
        category_exploration=0.0,
        repetition_penalty=0.0,
        pawn_progress_bonus=0.0,
        wall_distance_weight=0.0,
        wall_route_weight=0.0,
    )


def forced_root_search(
    search: MCTS,
    state: GameState,
    action: int,
    prior: float,
) -> dict:
    """Run equal-budget PUCT continuations while fixing the root action.

    This deliberately reuses the existing MCTS _select/_expand/_backup
    methods. It does not alter their scoring formula or the production search.
    """
    child_state = apply_action_index(state, action)
    root = Node(state)
    root_edge = Edge(prior=prior, child=Node(child_state))
    root.edges[action] = root_edge

    expanded_nodes = 0
    nn_evaluations = 0
    leaf_depths: list[int] = []

    for _ in range(search.simulations):
        node = root_edge.child
        path = [(root, root_edge)]

        while node.expanded and node.edges:
            search._widen(node)
            _selected_action, edge = search._select(node)
            if edge.child is None:
                edge.child = Node(apply_action_index(node.state, _selected_action))
            path.append((node, edge))
            node = edge.child

        leaf_depths.append(node.state.ply - state.ply)
        if not node.expanded:
            value = search._expand(node)
            expanded_nodes += 1
            if node.state.winner is None:
                nn_evaluations += 1
        else:
            value = (
                1.0 if node.state.winner == node.state.current_player else -1.0
            )
        search._backup(path, node, value)

    return {
        "q_root_perspective": root_edge.q,
        "visits": root_edge.visits,
        "value_sum_root_perspective": root_edge.value_sum,
        "expanded_nodes_below_root": expanded_nodes,
        "nn_evaluations_below_root": nn_evaluations,
        "average_leaf_depth": mean(leaf_depths) if leaf_depths else 0.0,
        "max_leaf_depth": max(leaf_depths, default=0),
    }


def evaluate_position(model, state: GameState, *, label: str, sequence: list[int],
                      budget: int, c_puct: float, device: str) -> dict:
    legal = legal_action_indices(state)
    if not legal:
        raise RuntimeError(f"{label}: no legal actions")

    policy_probe = make_search(model, budget=budget, c_puct=c_puct, device=device)
    priors, root_value = policy_probe._evaluate(state)
    if set(priors) != set(legal):
        raise RuntimeError(f"{label}: network prior/legal action mismatch")
    prior_sum = sum(priors.values())
    if not math.isclose(prior_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError(f"{label}: legal-action prior sums to {prior_sum}")

    root_player = state.current_player
    opponent = 1 - root_player
    own_before = shortest_distance(state, root_player)
    opponent_before = shortest_distance(state, opponent)
    immediate_threat_before = has_immediate_winning_move(state, opponent)

    rows: list[dict] = []
    for index, action in enumerate(legal, 1):
        child = apply_action_index(state, action)
        own_after = shortest_distance(child, root_player)
        opponent_after = shortest_distance(child, opponent)
        immediate_threat_after = has_immediate_winning_move(child, opponent)

        value_probe = make_search(model, budget=budget, c_puct=c_puct, device=device)
        _child_priors, child_value_side_to_move = value_probe._evaluate(child)
        predicted_value_root_perspective = -child_value_side_to_move

        search = make_search(model, budget=budget, c_puct=c_puct, device=device)
        forced = forced_root_search(search, state, action, priors[action])

        kind, payload = decode_action(action)
        rows.append({
            "position": label,
            "action_index": action,
            "action": action_payload(action),
            "category": "pawn" if kind == "move" else "wall",
            "neural_prior_P": priors[action],
            "root_value_head_value": root_value,
            "child_value_head_side_to_move": child_value_side_to_move,
            "child_value_head_root_perspective": predicted_value_root_perspective,
            "own_goal_distance_before": own_before,
            "own_goal_distance_after": own_after,
            "own_goal_distance_delta_after_minus_before": (
                None if own_before is None or own_after is None else own_after - own_before
            ),
            "opponent_goal_distance_before": opponent_before,
            "opponent_goal_distance_after": opponent_after,
            "opponent_goal_distance_delta_after_minus_before": (
                None if opponent_before is None or opponent_after is None
                else opponent_after - opponent_before
            ),
            "opponent_immediate_win_available_before": immediate_threat_before,
            "opponent_immediate_win_available_after": immediate_threat_after,
            "prevents_immediate_opponent_win": (
                immediate_threat_before and not immediate_threat_after
            ),
            "delays_opponent_threat": (
                opponent_before is not None and opponent_after is not None
                and opponent_before <= 2 and opponent_after > opponent_before
            ),
            "child_terminal_winner": child.winner,
            "forced_root_search_Q": forced["q_root_perspective"],
            "forced_root_search_visits": forced["visits"],
            "forced_root_search_value_sum": forced["value_sum_root_perspective"],
            "search_expanded_nodes": forced["expanded_nodes_below_root"],
            "search_nn_evaluations": forced["nn_evaluations_below_root"],
            "search_average_leaf_depth": forced["average_leaf_depth"],
            "search_max_leaf_depth": forced["max_leaf_depth"],
        })
        print(
            f"  [{index}/{len(legal)}] action={action} "
            f"P={priors[action]:.5f} V={predicted_value_root_perspective:+.4f} "
            f"Q={forced['q_root_perspective']:+.4f}",
            flush=True,
        )

    rows.sort(key=lambda row: row["action_index"])
    return {
        "label": label,
        "state": state.to_dict(),
        "generation_action_indices": sequence,
        "player_to_move": root_player,
        "opponent": opponent,
        "own_goal_distance_before": own_before,
        "opponent_goal_distance_before": opponent_before,
        "opponent_immediate_win_available_before": immediate_threat_before,
        "root_value_head_value": root_value,
        "legal_action_count": len(legal),
        "prior_sum": prior_sum,
        "actions": rows,
    }


def write_csv(path: Path, positions: list[dict]) -> None:
    rows = [row for position in positions for row in position["actions"]]
    if not rows:
        raise RuntimeError("No action rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["action"] = json.dumps(flat["action"], separators=(",", ":"))
            writer.writerow(flat)


def summarize(positions: list[dict]) -> dict:
    result = {}
    for position in positions:
        rows = position["actions"]
        best_prior = max(rows, key=lambda row: row["neural_prior_P"])
        best_value = max(rows, key=lambda row: row["child_value_head_root_perspective"])
        best_q = max(rows, key=lambda row: row["forced_root_search_Q"])
        result[position["label"]] = {
            "legal_actions": len(rows),
            "pawn_actions": sum(row["category"] == "pawn" for row in rows),
            "wall_actions": sum(row["category"] == "wall" for row in rows),
            "prior_mass": sum(row["neural_prior_P"] for row in rows),
            "actions_preventing_immediate_win": sum(
                row["prevents_immediate_opponent_win"] for row in rows
            ),
            "actions_delaying_opponent_threat": sum(
                row["delays_opponent_threat"] for row in rows
            ),
            "highest_prior_action": best_prior["action"],
            "highest_prior_P": best_prior["neural_prior_P"],
            "highest_value_action": best_value["action"],
            "highest_value": best_value["child_value_head_root_perspective"],
            "highest_Q_action": best_q["action"],
            "highest_Q": best_q["forced_root_search_Q"],
                    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--budget", type=int, default=32,
                        help="Equal continuation simulations for each forced root action.")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/diagnostics/per_action_counterfactual"))
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.budget < 1 or args.c_puct < 0 or args.threads < 1:
        parser.error("budget/threads must be positive and c-puct non-negative")

    output_json = args.output_dir / "per_action_counterfactual.json"
    output_csv = args.output_dir / "per_action_counterfactual.csv"
    output_summary = args.output_dir / "summary.json"
    for path in (output_json, output_csv, output_summary):
        if path.exists():
            parser.error(f"Output already exists: {path}; choose a fresh output directory")

    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)

    model = load_model(args.checkpoint, device=args.device)
    model.eval()
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()

    definitions = [
        ("opponent_threat_left_flank", 3),
        ("opponent_threat_right_flank", 5),
    ]
    positions = []
    for label, player0_x in definitions:
        state, sequence = build_threat_position(player0_x)
        print(
            f"Evaluating {label}: ply={state.ply}, legal_actions={len(legal_action_indices(state))}, "
            f"opponent_distance={shortest_distance(state, 1 - state.current_player)}",
            flush=True,
        )
        positions.append(evaluate_position(
            model, state, label=label, sequence=sequence,
            budget=args.budget, c_puct=args.c_puct, device=args.device,
        ))

    report = {
        "experiment": "per_action_counterfactual_evaluation",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "settings": {
            "budget_per_forced_root_action": args.budget,
            "c_puct": args.c_puct,
            "device": args.device,
            "threads": args.threads,
            "progressive_widening": False,
            "root_noise": False,
            "category_balanced_prior": False,
            "root_category_prior_mix": 0.0,
            "opening_goal_directed_prior": False,
            "category_exploration": 0.0,
            "repetition_penalty": 0.0,
            "pawn_progress_bonus": 0.0,
            "wall_distance_weight": 0.0,
            "wall_route_weight": 0.0,
            "tie_breaking": "deterministic",
            "forced_root_action": "fixed for every simulation; vanilla PUCT below root",
        },
        "metric_conventions": {
            "distance_delta": "after minus before; positive means that player's shortest path became longer",
            "child_value_head_root_perspective": "negative of child value because the opponent is to move after the root action",
            "forced_root_search_Q": "root player's perspective, backed up through the existing MCTS sign-flip logic",
            "prevents_immediate_opponent_win": "opponent had a legal one-action winning pawn move before the root action and has none after it",
            "delays_opponent_threat": "root was an opponent-distance<=2 threat position and the action increases opponent shortest-path distance",
        },
        "positions": positions,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(output_csv, positions)
    output_summary.write_text(json.dumps({
        "experiment": report["experiment"],
        "checkpoint_sha256": checkpoint_sha256,
        "settings": report["settings"],
        "positions": summarize(positions),
        "total_legal_action_rows": sum(len(p["actions"]) for p in positions),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"JSON: {output_json}")
    print(f"CSV: {output_csv}")
    print(f"Summary: {output_summary}")
    print(f"Total evaluated actions: {sum(len(p['actions']) for p in positions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
