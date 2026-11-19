#!/usr/bin/env python3
"""Independent one-ply tactical-label audit and policy/value head calibration.

Reads the frozen run_001 position set, independently recomputes immediate-win
and defensive labels, then samples separate no-defense and no-threat controls.
Inference only: no training, checkpoint writes, or MCTS/PUCT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean, median

from quoridor.engine import (
    GameState, Wall, apply_move, apply_wall, legal_pawn_moves, legal_walls,
    shortest_distance, validate_state,
)
from quoridor.neural_ai.encoder import (
    ACTION_SIZE, MOVE_ACTIONS, apply_action_index, decode_action,
    encode_state, legal_action_indices,
)
from quoridor.neural_ai.policy import load_model
from training.evaluate_tactical_policy_value_attribution import (
    choose_playout_action, state_fingerprint,
)


def restore_state(raw: dict) -> GameState:
    pawns = tuple((int(p["x"]), int(p["y"])) for p in raw["pawns"])
    walls = frozenset(Wall(w["orientation"], int(w["x"]), int(w["y"])) for w in raw["walls"])
    state = GameState(
        pawns=pawns, walls=walls,
        walls_remaining=tuple(int(v) for v in raw["walls_remaining"]),
        current_player=int(raw["current_player"]),
        winner=raw.get("winner"), ply=int(raw.get("ply", 0)),
    )
    validate_state(state)
    return state


def independent_immediate_win(state: GameState, player: int) -> bool:
    """Recompute winning pawn replies directly from legal moves and transitions."""
    if state.winner is not None:
        return state.winner == player
    probe = GameState(
        pawns=state.pawns, walls=state.walls,
        walls_remaining=state.walls_remaining,
        current_player=player, winner=None, ply=state.ply,
    )
    for destination in legal_pawn_moves(probe, player):
        if apply_move(probe, destination, player).winner == player:
            return True
    return False


def independent_defenses(state: GameState, legal: list[int]) -> set[int]:
    opponent = 1 - state.current_player
    defensive = set()
    for action in legal:
        child = apply_action_index(state, action)
        if child.winner == state.current_player:
            continue
        if not independent_immediate_win(child, opponent):
            defensive.add(action)
    return defensive


def action_desc(index: int) -> dict:
    kind, payload = decode_action(index)
    if kind == "move":
        return {"type": "pawn", "to": list(payload)}
    return {"type": "wall", "orientation": payload.orientation, "x": payload.x, "y": payload.y}


def sample_controls(seed: int, target_each: int, max_games: int, max_plies: int) -> dict[str, list[GameState]]:
    rng = random.Random(seed)
    groups = {"threat_no_defense": [], "no_threat": []}
    seen = set()
    for _game in range(max_games):
        state = GameState.initial()
        for _ in range(max_plies):
            if state.winner is not None:
                break
            key = state_fingerprint(state)
            if key not in seen:
                threat = independent_immediate_win(state, 1 - state.current_player)
                if not threat:
                    group = "no_threat"
                else:
                    legal = legal_action_indices(state)
                    if independent_defenses(state, legal):
                        group = None
                    else:
                        group = "threat_no_defense"
                if group and len(groups[group]) < target_each:
                    groups[group].append(state)
                    seen.add(key)
            action = choose_playout_action(state, rng)
            if action is None:
                break
            state = apply_action_index(state, action)
        if all(len(v) >= target_each for v in groups.values()):
            break
    missing = {k: target_each - len(v) for k, v in groups.items() if len(v) < target_each}
    if missing:
        raise RuntimeError(f"Could not sample requested control cohorts: {missing}; increase --max-games/--max-plies")
    return groups


def evaluate(model, state: GameState, cohort: str, device: str, batch_size: int) -> dict:
    import torch
    validate_state(state)
    legal = legal_action_indices(state)
    if not legal:
        raise RuntimeError("Unexpected terminal state in evaluation cohort")
    threat = independent_immediate_win(state, 1 - state.current_player)
    defenses = independent_defenses(state, legal) if threat else set()

    tensor = torch.tensor(encode_state(state), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.inference_mode():
        logits, root_value = model(tensor)
        masked = torch.full((ACTION_SIZE,), float("-inf"), device=device)
        masked[legal] = logits[0][legal]
        probs = torch.softmax(masked, dim=0)
    priors = {a: float(probs[a].item()) for a in legal}

    children, child_actions, values = [], [], {}
    for action in legal:
        child = apply_action_index(state, action)
        if child.winner is not None:
            values[action] = 1.0 if child.winner == state.current_player else -1.0
        else:
            children.append(child)
            child_actions.append(action)
    for start in range(0, len(children), batch_size):
        batch = torch.tensor([encode_state(s) for s in children[start:start+batch_size]], dtype=torch.float32, device=device)
        with torch.inference_mode():
            _l, out = model(batch)
        for action, val in zip(child_actions[start:start+batch_size], out.detach().cpu().tolist()):
            values[action] = -float(val)

    p_order = sorted(legal, key=lambda a: (-priors[a], a))
    v_order = sorted(legal, key=lambda a: (-values[a], a))
    p_rank = {a:i for i,a in enumerate(p_order, 1)}
    v_rank = {a:i for i,a in enumerate(v_order, 1)}
    rows = [{
        "action_index": a, "action": action_desc(a),
        "is_defensive": a in defenses,
        "prior": priors[a], "policy_rank": p_rank[a],
        "child_value_root": values[a], "value_rank": v_rank[a],
    } for a in legal]
    defensive_rows = [r for r in rows if r["is_defensive"]]
    return {
        "cohort": cohort,
        "state": state.to_dict(),
        "ply": state.ply,
        "wall_count": len(state.walls),
        "legal_action_count": len(legal),
        "opponent_immediate_win": threat,
        "defensive_action_count": len(defenses),
        "defensive_policy_mass": sum(priors[a] for a in defenses),
        "best_defensive_policy_rank": min((r["policy_rank"] for r in defensive_rows), default=None),
        "best_defensive_value_rank": min((r["value_rank"] for r in defensive_rows), default=None),
        "defense_in_policy_top1": any(r["is_defensive"] and r["policy_rank"] == 1 for r in rows),
        "defense_in_policy_top5": any(r["is_defensive"] and r["policy_rank"] <= 5 for r in rows),
        "defense_in_value_top1": any(r["is_defensive"] and r["value_rank"] == 1 for r in rows),
        "defense_in_value_top5": any(r["is_defensive"] and r["value_rank"] <= 5 for r in rows),
        "root_value": float(root_value[0].item()),
        "actions": rows,
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"position_count": 0}
    threatened = [p for p in rows if p["opponent_immediate_win"]]
    with_defense = [p for p in threatened if p["defensive_action_count"] > 0]
    def pct(flag):
        return 100.0 * sum(bool(p[flag]) for p in with_defense) / len(with_defense) if with_defense else None
    ranks_p = [p["best_defensive_policy_rank"] for p in with_defense]
    ranks_v = [p["best_defensive_value_rank"] for p in with_defense]
    return {
        "position_count": len(rows),
        "threat_position_count": len(threatened),
        "threat_with_defense_count": len(with_defense),
        "threat_without_defense_count": len(threatened) - len(with_defense),
        "defense_in_policy_top1_pct": pct("defense_in_policy_top1"),
        "defense_in_policy_top5_pct": pct("defense_in_policy_top5"),
        "defense_in_value_top1_pct": pct("defense_in_value_top1"),
        "defense_in_value_top5_pct": pct("defense_in_value_top5"),
        "mean_best_defensive_policy_rank": mean(ranks_p) if ranks_p else None,
        "median_best_defensive_policy_rank": median(ranks_p) if ranks_p else None,
        "mean_best_defensive_value_rank": mean(ranks_v) if ranks_v else None,
        "median_best_defensive_value_rank": median(ranks_v) if ranks_v else None,
        "mean_defensive_policy_mass": mean(p["defensive_policy_mass"] for p in with_defense) if with_defense else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--source-json", type=Path, default=Path("data/diagnostics/tactical_policy_value_attribution/run_001/tactical_policy_value_attribution.json"))
    parser.add_argument("--control-positions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--max-games", type=int, default=5000)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("data/diagnostics/tactical_label_head_calibration/run_001"))
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.source_json.is_file():
        parser.error("Checkpoint and --source-json must exist")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    import torch
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    source = json.loads(args.source_json.read_text(encoding="utf-8"))
    frozen = source.get("positions", [])
    if not frozen:
        parser.error("Source JSON contains no positions[]")
    model = load_model(args.checkpoint, device=args.device)
    model.eval()
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()

    audited = []
    label_checks = {"positions": 0, "actions_checked": 0, "mismatched_labels": 0, "invalid_states": 0}
    for item in frozen:
        state = restore_state(item["state"])
        validate_state(state)
        legal = legal_action_indices(state)
        fresh_labels = independent_defenses(state, legal)
        stored = {int(a["action_index"]): bool(a["is_defensive"]) for a in item["actions"]}
        mismatches = [a for a in legal if stored.get(a) != (a in fresh_labels)]
        label_checks["positions"] += 1
        label_checks["actions_checked"] += len(legal)
        label_checks["mismatched_labels"] += len(mismatches)
        result = evaluate(model, state, "frozen_threat_with_defense", args.device, args.batch_size)
        result["source_position_id"] = item.get("position_id")
        result["label_mismatch_action_indices"] = mismatches
        audited.append(result)

    controls = sample_controls(args.seed, args.control_positions, args.max_games, args.max_plies)
    for cohort, states in controls.items():
        for state in states:
            audited.append(evaluate(model, state, cohort, args.device, args.batch_size))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohorts = {}
    for name in sorted({p["cohort"] for p in audited}):
        cohorts[name] = summarize([p for p in audited if p["cohort"] == name])
    report = {
        "experiment": "tactical_label_and_head_calibration_audit",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "source_json_sha256": hashlib.sha256(args.source_json.read_bytes()).hexdigest(),
        "label_validation": {
            **label_checks,
            "independent_method": "reconstruct frozen GameState; enumerate legal actions; apply each action; enumerate opponent legal pawn moves and test terminal winner",
            "all_labels_match": label_checks["mismatched_labels"] == 0,
        },
        "controls": {"seed": args.seed, "positions_per_control_cohort": args.control_positions,
                     "sampler": "seeded legal playout; independent threat/defense classifier"},
        "evaluation": {"mcts_or_puct_used": False, "checkpoint_modified": False,
                       "policy": "legal-action masked softmax over raw logits",
                       "value": "child value converted to root-player perspective; terminal child exact +/-1"},
        "cohort_summary": cohorts,
        "positions": audited,
    }
    (args.output_dir / "tactical_label_head_calibration.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps({k:v for k,v in report.items() if k!="positions"}, indent=2)+"\n", encoding="utf-8")
    print(f"Positions: {len(audited)}; frozen labels checked: {label_checks['actions_checked']}; mismatches: {label_checks['mismatched_labels']}")
    print(f"Summary: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
