#!/usr/bin/env python3
"""Attribute immediate-threat tactical failures to the neural policy and value heads.

Evaluation only: no training, checkpoint writes, or MCTS/PUCT execution.
Positions are sampled from deterministic legal random playouts, then every legal
root action is evaluated counterfactually by the fixed neural network.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean, median

from quoridor.engine import (
    GameState,
    apply_move,
    apply_wall,
    legal_pawn_moves,
    legal_walls,
    shortest_distance,
    validate_state,
)
from quoridor.neural_ai.encoder import (
    ACTION_SIZE,
    MOVE_ACTIONS,
    apply_action_index,
    decode_action,
    encode_state,
    legal_action_indices,
)
from quoridor.neural_ai.policy import load_model


def has_immediate_winning_move(state: GameState, player: int) -> bool:
    """Exact one-reply test: does player have a legal pawn move that wins now?"""
    if state.winner is not None:
        return state.winner == player
    hypothetical = replace(state, current_player=player)
    for destination in legal_pawn_moves(hypothetical, player):
        child = apply_move(hypothetical, destination, player)
        if child.winner == player:
            return True
    return False


def defensive_actions(state: GameState, legal: list[int]) -> set[int]:
    """Actions after which the opponent no longer has an immediate winning move."""
    opponent = 1 - state.current_player
    result: set[int] = set()
    for action in legal:
        child = apply_action_index(state, action)
        if child.winner == state.current_player:
            continue
        if not has_immediate_winning_move(child, opponent):
            result.add(action)
    return result


def choose_playout_action(state: GameState, rng: random.Random) -> int | None:
    """Generate varied but goal-directed legal games without consulting the NN."""
    player = state.current_player
    can_wall = state.walls_remaining[player] > 0
    choose_wall = can_wall and rng.random() < 0.32

    if choose_wall:
        walls = legal_walls(state, player)
        if walls:
            wall = rng.choice(walls)
            return MOVE_ACTIONS + (0 if wall.orientation == "H" else 64) + wall.y * 8 + wall.x

    moves = legal_pawn_moves(state, player)
    if not moves:
        return None
    if rng.random() < 0.72:
        scored = []
        for destination in moves:
            child = apply_move(state, destination, player)
            distance = shortest_distance(child, player)
            scored.append((distance if distance is not None else 99, destination))
        best = min(score for score, _ in scored)
        progress_moves = [move for score, move in scored if score == best]
        destination = rng.choice(progress_moves)
    else:
        destination = rng.choice(moves)
    return destination[1] * 9 + destination[0]


def diversity_bucket(state: GameState) -> str:
    walls_used = len(state.walls)
    own_distance = shortest_distance(state, state.current_player)
    opponent_x = state.pawns[1 - state.current_player][0]
    if walls_used == 0:
        wall_band = "0"
    elif walls_used <= 4:
        wall_band = "1-4"
    elif walls_used <= 9:
        wall_band = "5-9"
    else:
        wall_band = "10+"
    if own_distance is None or own_distance <= 4:
        own_band = "1-4"
    elif own_distance <= 8:
        own_band = "5-8"
    else:
        own_band = "9+"
    x_band = "left" if opponent_x <= 2 else "center" if opponent_x <= 6 else "right"
    return f"walls={wall_band}|own_distance={own_band}|opponent_flank={x_band}"


def state_fingerprint(state: GameState) -> str:
    payload = state.to_dict()
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def collect_positions(
    *, target: int, seed: int, max_games: int, max_plies: int,
    pool_multiplier: int,
) -> list[dict]:
    """Collect unique legal threat states, then round-robin across diversity buckets."""
    rng = random.Random(seed)
    pool_limit = target * pool_multiplier
    buckets: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    total_threat_states = 0

    for game_index in range(max_games):
        state = GameState.initial()
        game_threats = 0
        for _ in range(max_plies):
            if state.winner is not None:
                break

            opponent = 1 - state.current_player
            if has_immediate_winning_move(state, opponent):
                total_threat_states += 1
                legal = legal_action_indices(state)
                defenses = defensive_actions(state, legal)
                fingerprint = state_fingerprint(state)
                if defenses and fingerprint not in seen:
                    seen.add(fingerprint)
                    bucket = diversity_bucket(state)
                    buckets[bucket].append({
                        "state": state,
                        "source_game": game_index,
                        "source_ply": state.ply,
                        "bucket": bucket,
                        "defensive_action_count": len(defenses),
                    })
                    game_threats += 1
                    if sum(len(items) for items in buckets.values()) >= pool_limit:
                        break

            action = choose_playout_action(state, rng)
            if action is None:
                break
            state = apply_action_index(state, action)

        if sum(len(items) for items in buckets.values()) >= pool_limit:
            break

    # Round-robin bucket selection prevents one common state family dominating.
    selected: list[dict] = []
    bucket_names = sorted(buckets)
    offsets = {name: 0 for name in bucket_names}
    while len(selected) < target:
        added = False
        for name in bucket_names:
            index = offsets[name]
            if index < len(buckets[name]):
                selected.append(buckets[name][index])
                offsets[name] += 1
                added = True
                if len(selected) == target:
                    break
        if not added:
            break

    if len(selected) < target:
        raise RuntimeError(
            f"Only collected {len(selected)}/{target} unique threat positions "
            f"with at least one defensive action. Increase --max-games or "
            f"--max-plies. Raw threat states seen={total_threat_states}, "
            f"diversity buckets={len(buckets)}."
        )

    return selected


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


def batched_child_values(model, states: list[GameState], *, device: str, batch_size: int) -> list[float]:
    """Return value-head outputs in child side-to-move perspective."""
    import torch

    values: list[float] = []
    for start in range(0, len(states), batch_size):
        batch_states = states[start:start + batch_size]
        tensor = torch.tensor(
            [encode_state(state) for state in batch_states],
            dtype=torch.float32,
            device=device,
        )
        with torch.inference_mode():
            _logits, batch_values = model(tensor)
        values.extend(float(value) for value in batch_values.detach().cpu().tolist())
    return values


def evaluate_position(model, item: dict, *, device: str, batch_size: int) -> dict:
    import torch

    state: GameState = item["state"]
    validate_state(state)
    if not has_immediate_winning_move(state, 1 - state.current_player):
        raise RuntimeError("Sampler emitted a state without an immediate opponent win")
    legal = legal_action_indices(state)
    defenses = defensive_actions(state, legal)
    if not defenses:
        raise RuntimeError("Sampler emitted a threat state without a defensive action")

    root_tensor = torch.tensor(
        encode_state(state), dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.inference_mode():
        root_logits, root_value_tensor = model(root_tensor)
        logits = root_logits[0]
        mask = torch.full((ACTION_SIZE,), float("-inf"), device=logits.device)
        mask[legal] = logits[legal]
        probabilities = torch.softmax(mask, dim=0)
    priors = {action: float(probabilities[action].item()) for action in legal}
    prior_sum = sum(priors.values())
    if abs(prior_sum - 1.0) > 1e-5:
        raise RuntimeError(f"Masked legal prior sum is {prior_sum}")

    child_states: list[GameState] = []
    nonterminal_indices: list[int] = []
    child_values_root: dict[int, float] = {}
    root_player = state.current_player
    for action in legal:
        child = apply_action_index(state, action)
        if child.winner is not None:
            child_values_root[action] = 1.0 if child.winner == root_player else -1.0
        else:
            nonterminal_indices.append(action)
            child_states.append(child)

    side_values = batched_child_values(
        model, child_states, device=device, batch_size=batch_size
    )
    for action, side_value in zip(nonterminal_indices, side_values):
        # Child side-to-move is the opponent; invert to root-player perspective.
        child_values_root[action] = -side_value

    prior_order = sorted(legal, key=lambda a: (-priors[a], a))
    value_order = sorted(legal, key=lambda a: (-child_values_root[a], a))
    prior_ranks = {action: rank for rank, action in enumerate(prior_order, 1)}
    value_ranks = {action: rank for rank, action in enumerate(value_order, 1)}

    rows = []
    for action in legal:
        kind, _payload = decode_action(action)
        rows.append({
            "action_index": action,
            "action": action_payload(action),
            "category": "pawn" if kind == "move" else "wall",
            "is_defensive": action in defenses,
            "policy_prior": priors[action],
            "policy_rank": prior_ranks[action],
            "child_value_root_perspective": child_values_root[action],
            "value_rank": value_ranks[action],
        })

    defensive_rows = [row for row in rows if row["is_defensive"]]
    nondefensive_rows = [row for row in rows if not row["is_defensive"]]
    best_defense_policy = max(defensive_rows, key=lambda row: row["policy_prior"])
    best_defense_value = max(defensive_rows, key=lambda row: row["child_value_root_perspective"])
    best_nondefense_value = max(nondefensive_rows, key=lambda row: row["child_value_root_perspective"]) if nondefensive_rows else None
    top_policy = [row for row in rows if row["policy_rank"] <= 1]
    top_value = [row for row in rows if row["value_rank"] <= 1]

    return {
        "position_id": f"threat_{item['source_game']:05d}_{item['source_ply']:03d}",
        "source_game": item["source_game"],
        "source_ply": item["source_ply"],
        "diversity_bucket": item["bucket"],
        "state": state.to_dict(),
        "player_to_move": root_player,
        "opponent": 1 - root_player,
        "ply": state.ply,
        "wall_count": len(state.walls),
        "walls_remaining": list(state.walls_remaining),
        "own_goal_distance": shortest_distance(state, root_player),
        "opponent_goal_distance": shortest_distance(state, 1 - root_player),
        "opponent_immediate_win_available": True,
        "legal_action_count": len(legal),
        "defensive_action_count": len(defenses),
        "defensive_policy_mass": sum(priors[action] for action in defenses),
        "best_defensive_policy_prior": best_defense_policy["policy_prior"],
        "best_defensive_policy_rank": best_defense_policy["policy_rank"],
        "best_defensive_value": best_defense_value["child_value_root_perspective"],
        "best_defensive_value_rank": best_defense_value["value_rank"],
        "best_nondefensive_value": (
            best_nondefense_value["child_value_root_perspective"]
            if best_nondefense_value else None
        ),
        "best_defensive_vs_nondefensive_value_gap": (
            best_defense_value["child_value_root_perspective"]
            - best_nondefense_value["child_value_root_perspective"]
            if best_nondefense_value else None
        ),
        "defensive_in_policy_top1": any(row["is_defensive"] for row in top_policy),
        "defensive_in_value_top1": any(row["is_defensive"] for row in top_value),
        "defensive_in_policy_top5": any(
            row["is_defensive"] and row["policy_rank"] <= 5 for row in rows
        ),
        "defensive_in_value_top5": any(
            row["is_defensive"] and row["value_rank"] <= 5 for row in rows
        ),
        "root_value_head": float(root_value_tensor[0].item()),
        "actions": rows,
    }


def aggregate(positions: list[dict]) -> dict:
    def pct(count: int, total: int) -> float:
        return 100.0 * count / total if total else 0.0

    count = len(positions)
    all_actions = [row for position in positions for row in position["actions"]]
    defensive = [row for row in all_actions if row["is_defensive"]]
    nondefensive = [row for row in all_actions if not row["is_defensive"]]
    gaps = [
        position["best_defensive_vs_nondefensive_value_gap"]
        for position in positions
        if position["best_defensive_vs_nondefensive_value_gap"] is not None
    ]
    return {
        "position_count": count,
        "total_action_rows": len(all_actions),
        "positions_with_defense_in_policy_top1": pct(
            sum(p["defensive_in_policy_top1"] for p in positions), count
        ),
        "positions_with_defense_in_policy_top5": pct(
            sum(p["defensive_in_policy_top5"] for p in positions), count
        ),
        "positions_with_defense_in_value_top1": pct(
            sum(p["defensive_in_value_top1"] for p in positions), count
        ),
        "positions_with_defense_in_value_top5": pct(
            sum(p["defensive_in_value_top5"] for p in positions), count
        ),
        "mean_defensive_policy_mass": mean(p["defensive_policy_mass"] for p in positions),
        "median_defensive_policy_mass": median(p["defensive_policy_mass"] for p in positions),
        "mean_best_defensive_policy_rank": mean(p["best_defensive_policy_rank"] for p in positions),
        "median_best_defensive_policy_rank": median(p["best_defensive_policy_rank"] for p in positions),
        "mean_best_defensive_value_rank": mean(p["best_defensive_value_rank"] for p in positions),
        "median_best_defensive_value_rank": median(p["best_defensive_value_rank"] for p in positions),
        "mean_best_defensive_vs_nondefensive_value_gap": mean(gaps) if gaps else None,
        "mean_defensive_action_prior": mean(row["policy_prior"] for row in defensive),
        "mean_nondefensive_action_prior": mean(row["policy_prior"] for row in nondefensive),
        "mean_defensive_child_value": mean(row["child_value_root_perspective"] for row in defensive),
        "mean_nondefensive_child_value": mean(row["child_value_root_perspective"] for row in nondefensive),
    }


def write_csv(path: Path, positions: list[dict]) -> None:
    rows = []
    for position in positions:
        for action in position["actions"]:
            rows.append({
                key: value for key, value in {
                    "position_id": position["position_id"],
                    "source_game": position["source_game"],
                    "source_ply": position["source_ply"],
                    "diversity_bucket": position["diversity_bucket"],
                    "wall_count": position["wall_count"],
                    "walls_remaining": json.dumps(position["walls_remaining"]),
                    "own_goal_distance": position["own_goal_distance"],
                    "opponent_goal_distance": position["opponent_goal_distance"],
                    **action,
                    "action": json.dumps(action["action"], separators=(",", ":")),
                }.items()
            })
    if not rows:
        raise RuntimeError("No action rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--positions", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--max-games", type=int, default=2500)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--pool-multiplier", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/diagnostics/tactical_policy_value_attribution/run_001"))
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.positions < 1 or args.max_games < 1 or args.max_plies < 1:
        parser.error("positions, max-games, and max-plies must be positive")
    if args.pool_multiplier < 1 or args.batch_size < 1 or args.threads < 1:
        parser.error("pool-multiplier, batch-size, and threads must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")

    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)

    print(
        f"Sampling {args.positions} unique tactical positions "
        f"(seed={args.seed}, max_games={args.max_games})",
        flush=True,
    )
    sampled = collect_positions(
        target=args.positions,
        seed=args.seed,
        max_games=args.max_games,
        max_plies=args.max_plies,
        pool_multiplier=args.pool_multiplier,
    )
    print(
        f"Collected {len(sampled)} positions across "
        f"{len({item['bucket'] for item in sampled})} diversity buckets",
        flush=True,
    )

    model = load_model(args.checkpoint, device=args.device)
    model.eval()
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()

    evaluated = []
    for index, item in enumerate(sampled, 1):
        print(
            f"[{index}/{len(sampled)}] game={item['source_game']} ply={item['source_ply']} "
            f"bucket={item['bucket']} walls={len(item['state'].walls)}",
            flush=True,
        )
        evaluated.append(evaluate_position(
            model, item, device=args.device, batch_size=args.batch_size
        ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "tactical_policy_value_failure_attribution",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "sampler": {
            "method": "seeded legal random playouts; 32% wall decision probability when walls remain; pawn moves favor shortest-path progress 72% of the time",
            "seed": args.seed,
            "max_games": args.max_games,
            "max_plies_per_game": args.max_plies,
            "pool_multiplier": args.pool_multiplier,
            "positions_requested": args.positions,
            "positions_evaluated": len(evaluated),
            "selection": "round-robin across wall-count, own-distance, and opponent-flank buckets",
        },
        "evaluation": {
            "device": args.device,
            "threads": args.threads,
            "batch_size": args.batch_size,
            "mcts_or_puct_used": False,
            "checkpoint_modified": False,
            "policy": "raw model logits masked to legal root actions, then softmax",
            "value": "child value head converted to root-player perspective; terminal children use exact +/-1",
            "defense_definition": "opponent has an immediate legal winning pawn move before root action and has none after it",
        },
        "aggregate": aggregate(evaluated),
        "positions": evaluated,
    }

    output_json = args.output_dir / "tactical_policy_value_attribution.json"
    output_csv = args.output_dir / "tactical_policy_value_attribution.csv"
    output_summary = args.output_dir / "summary.json"
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(output_csv, evaluated)
    output_summary.write_text(json.dumps({
        "experiment": report["experiment"],
        "checkpoint_sha256": checkpoint_sha256,
        "sampler": report["sampler"],
        "evaluation": report["evaluation"],
        "aggregate": report["aggregate"],
        "diversity_buckets": {
            bucket: aggregate([p for p in evaluated if p["diversity_bucket"] == bucket])
            for bucket in sorted({p["diversity_bucket"] for p in evaluated})
        },
    }, indent=2) + "\n", encoding="utf-8")

    print(f"JSON: {output_json}")
    print(f"CSV: {output_csv}")
    print(f"Summary: {output_summary}")
    print(f"Positions: {len(evaluated)}; action rows: {sum(len(p['actions']) for p in evaluated)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
