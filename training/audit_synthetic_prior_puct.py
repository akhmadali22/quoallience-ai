#!/usr/bin/env python3
"""Audit a checkpoint on deterministic, engine-generated Quoridor positions.

No replay/training data is read. Positions are generated from GameState.initial()
using seeded legal action sequences, then validated by the engine before use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from quoridor.engine import (
    GameState,
    apply_move,
    apply_wall,
    shortest_distance,
    state_repetition_key,
    validate_state,
)
from quoridor.neural_ai.encoder import (
    apply_action_index,
    decode_action,
    encode_state,
    legal_action_indices,
)
from training.audit_opening_puct import run_position
from quoridor.neural_ai.policy import load_model


def apply_recorded_action(state: GameState, action_index: int) -> GameState:
    return apply_action_index(state, action_index)


def action_record(action_index: int) -> dict:
    kind, payload = decode_action(action_index)
    if kind == "move":
        return {"type": "pawn", "to": list(payload)}
    return {
        "type": "wall",
        "orientation": payload.orientation,
        "x": payload.x,
        "y": payload.y,
    }


def make_sample(state: GameState, *, label: str, seed: int | None,
                sequence: list[dict], tactical: dict | None = None) -> dict:
    validate_state(state)
    if state.winner is not None:
        raise ValueError("Terminal states must not enter the audit")
    if shortest_distance(state, 0) is None or shortest_distance(state, 1) is None:
        raise ValueError("Both players must retain a path to their goal")
    return {
        "game_id": -1 if seed is None else seed,
        "seed": seed,
        "ply": state.ply,
        "player": state.current_player,
        "state": encode_state(state),
        "_audit_label": label,
        "_generation": {
            "seed": seed,
            "action_sequence": list(sequence),
            "state": state.to_dict(),
            "tactical_metrics": tactical or {},
        },
    }


def inspect_wall_opportunities(state: GameState, legal: list[int]) -> dict | None:
    """Measure legal wall actions using exact engine path distances."""
    player = state.current_player
    own_before = shortest_distance(state, player)
    opp_before = shortest_distance(state, 1 - player)
    best = None

    for action in legal:
        kind, _ = decode_action(action)
        if kind != "wall":
            continue
        child = apply_recorded_action(state, action)
        own_after = shortest_distance(child, player)
        opp_after = shortest_distance(child, 1 - player)
        if None in (own_before, opp_before, own_after, opp_after):
            continue
        own_cost = own_after - own_before
        opponent_delay = opp_after - opp_before
        score = opponent_delay - own_cost
        row = {
            "action": action_record(action),
            "own_distance_before": own_before,
            "own_distance_after": own_after,
            "opponent_distance_before": opp_before,
            "opponent_distance_after": opp_after,
            "own_distance_cost": own_cost,
            "opponent_distance_delay": opponent_delay,
            "delay_minus_own_cost": score,
        }
        if best is None or (score, opponent_delay, -own_cost) > (
            best["delay_minus_own_cost"],
            best["opponent_distance_delay"],
            -best["own_distance_cost"],
        ):
            best = row
    return best


def add_unique(bucket: list[dict], sample: dict, seen: set[tuple],
               limit: int) -> None:
    state = sample["_generation"]["state"]
    key = (
        tuple((p["x"], p["y"]) for p in state["pawns"]),
        tuple(sorted((w["orientation"], w["x"], w["y"]) for w in state["walls"])),
        tuple(state["walls_remaining"]),
        state["current_player"],
    )
    if key not in seen and len(bucket) < limit:
        seen.add(key)
        bucket.append(sample)


def build_scripted_position(player0_targets: list[tuple[int, int]],
                            player1_targets: list[tuple[int, int]]) -> tuple[GameState, list[dict]]:
    """Build a reproducible midgame position from alternating legal pawn moves."""
    if len(player0_targets) != len(player1_targets):
        raise ValueError("Both scripted players must have the same number of moves")
    state = GameState.initial()
    sequence: list[dict] = []
    for p0_target, p1_target in zip(player0_targets, player1_targets):
        for target in (p0_target, p1_target):
            action = target[1] * 9 + target[0]
            state = apply_recorded_action(state, action)
            sequence.append(action_record(action))
    return state, sequence


def generate_positions(*, seed_start: int, seed_count: int, max_game_plies: int,
                       continuation_plies: list[int], per_category: int) -> list[dict]:
    initial = GameState.initial()
    positions = [make_sample(
        initial, label="standard_initial_position", seed=None, sequence=[]
    )]
    continuation: list[dict] = []
    urgent_blocks: list[dict] = []
    strategic_walls: list[dict] = []
    seen_cont: set[tuple] = set()
    seen_urgent: set[tuple] = set()
    seen_strategic: set[tuple] = set()
    checkpoint_plies = set(continuation_plies)

    for seed in range(seed_start, seed_start + seed_count):
        rng = random.Random(seed)
        state = GameState.initial()
        sequence: list[dict] = []

        for _ in range(max_game_plies):
            if state.winner is not None:
                break

            if state.ply in checkpoint_plies:
                add_unique(
                    continuation,
                    make_sample(state, label="deterministic_legal_continuation",
                                seed=seed, sequence=sequence),
                    seen_cont, max(1, per_category * len(continuation_plies)),
                )

            legal = legal_action_indices(state)
            if not legal:
                break

            # Balance action categories during generation. This is only a
            # position-generation policy, never the policy being audited.
            pawn_actions = [a for a in legal if a < 81]
            wall_actions = [a for a in legal if a >= 81]
            categories = []
            if pawn_actions:
                categories.append(pawn_actions)
            if wall_actions:
                categories.append(wall_actions)
            chosen_pool = rng.choice(categories)
            chosen = rng.choice(chosen_pool)

            # Tactical scans are restricted to selected plies or genuine
            # near-goal threats to keep deterministic generation tractable.
            opponent_distance = shortest_distance(state, 1 - state.current_player)
            scan_tactics = state.ply in checkpoint_plies or (
                opponent_distance is not None and opponent_distance <= 2
            )
            if scan_tactics:
                best_wall = inspect_wall_opportunities(state, legal)
                if best_wall and best_wall["opponent_distance_delay"] >= 1:
                    if (opponent_distance is not None and opponent_distance <= 2
                            and best_wall["own_distance_cost"] <= 1):
                        add_unique(
                            urgent_blocks,
                            make_sample(
                                state, label="tactical_opponent_winning_threat",
                                seed=seed, sequence=sequence,
                                tactical={
                                    "opponent_distance": opponent_distance,
                                    "best_blocking_wall": best_wall,
                                    "criterion": "opponent distance <= 2; legal wall delays opponent by >= 1; own path cost <= 1",
                                },
                            ),
                            seen_urgent, per_category,
                        )
                    elif (opponent_distance is not None and opponent_distance >= 3
                          and best_wall["delay_minus_own_cost"] >= 1):
                        add_unique(
                            strategic_walls,
                            make_sample(
                                state, label="tactical_strategic_wall_opportunity",
                                seed=seed, sequence=sequence,
                                tactical={
                                    "opponent_distance": opponent_distance,
                                    "best_strategic_wall": best_wall,
                                    "criterion": "legal wall delay minus own path cost >= 1",
                                },
                            ),
                            seen_strategic, per_category,
                        )

            sequence.append(action_record(chosen))
            state = apply_recorded_action(state, chosen)
            if state.winner is not None:
                break

    # Do not depend on random trajectories to stumble into rare tactical states.
    # These positions are reached through explicit, alternating legal pawn moves
    # from the standard initial board, so every position has a replayable sequence.
    scripted_urgent = [
        (
            [(3, y) for y in range(8, 2, -1)],
            [(4, y) for y in range(1, 7)],
        ),
        (
            [(5, y) for y in range(8, 2, -1)],
            [(4, y) for y in range(1, 7)],
        ),
    ]
    for offset, (p0_moves, p1_moves) in enumerate(scripted_urgent):
        if len(urgent_blocks) >= per_category:
            break
        state, sequence = build_scripted_position(p0_moves, p1_moves)
        legal = legal_action_indices(state)
        opponent_distance = shortest_distance(state, 1 - state.current_player)
        best_wall = inspect_wall_opportunities(state, legal)
        if (opponent_distance is not None and opponent_distance <= 2
                and best_wall is not None
                and best_wall["opponent_distance_delay"] >= 1
                and best_wall["own_distance_cost"] <= 1):
            add_unique(
                urgent_blocks,
                make_sample(
                    state, label="tactical_opponent_winning_threat",
                    seed=seed_start + seed_count + 100 + offset,
                    sequence=sequence,
                    tactical={
                        "opponent_distance": opponent_distance,
                        "best_blocking_wall": best_wall,
                        "criterion": "opponent distance <= 2; legal wall delays opponent by >= 1; own path cost <= 1",
                        "construction": "scripted alternating legal pawn moves from initial state",
                    },
                ),
                seen_urgent, per_category,
            )

    scripted_strategic = [
        (
            [(3, y) for y in range(8, 3, -1)],
            [(4, y) for y in range(1, 6)],
        ),
        (
            [(5, y) for y in range(8, 3, -1)],
            [(4, y) for y in range(1, 6)],
        ),
    ]
    for offset, (p0_moves, p1_moves) in enumerate(scripted_strategic):
        if len(strategic_walls) >= per_category:
            break
        state, sequence = build_scripted_position(p0_moves, p1_moves)
        legal = legal_action_indices(state)
        opponent_distance = shortest_distance(state, 1 - state.current_player)
        best_wall = inspect_wall_opportunities(state, legal)
        if (opponent_distance is not None and opponent_distance >= 3
                and best_wall is not None
                and best_wall["delay_minus_own_cost"] >= 1):
            add_unique(
                strategic_walls,
                make_sample(
                    state, label="tactical_strategic_wall_opportunity",
                    seed=seed_start + seed_count + 200 + offset,
                    sequence=sequence,
                    tactical={
                        "opponent_distance": opponent_distance,
                        "best_strategic_wall": best_wall,
                        "criterion": "legal wall delay minus own path cost >= 1",
                        "construction": "scripted alternating legal pawn moves from initial state",
                    },
                ),
                seen_strategic, per_category,
            )

    if len(continuation) < per_category:
        raise RuntimeError(
            f"Only found {len(continuation)} continuation positions; "
            f"need {per_category}. Increase --seed-count or --max-game-plies."
        )
    if len(urgent_blocks) < per_category:
        raise RuntimeError(
            f"Only found {len(urgent_blocks)} opponent-threat/block positions; "
            f"need {per_category}. Increase --seed-count or --max-game-plies."
        )
    if len(strategic_walls) < per_category:
        raise RuntimeError(
            f"Only found {len(strategic_walls)} strategic-wall positions; "
            f"need {per_category}. Increase --seed-count or --max-game-plies."
        )

    positions.extend(continuation[:max(1, per_category * len(continuation_plies))])
    positions.extend(urgent_blocks)
    positions.extend(strategic_walls)
    return positions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed-start", type=int, default=20261119)
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--max-game-plies", type=int, default=64)
    parser.add_argument("--continuation-plies", type=int, nargs="+",
                        default=[8, 16, 24, 32])
    parser.add_argument("--positions-per-tactical-category", type=int, default=2)
    parser.add_argument("--budgets", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        default=Path("data/diagnostics/exp001r_synthetic_prior_puct_audit.json"))
    args = parser.parse_args()

    if args.seed_count < 1 or args.max_game_plies < 1:
        parser.error("seed-count and max-game-plies must be positive")
    if args.positions_per_tactical_category < 1:
        parser.error("positions-per-tactical-category must be positive")
    if not args.budgets or any(value < 1 for value in args.budgets):
        parser.error("all budgets must be positive")
    if args.c_puct < 0 or args.threads < 1:
        parser.error("c-puct must be non-negative and threads positive")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose another --output")

    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)

    positions = generate_positions(
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        max_game_plies=args.max_game_plies,
        continuation_plies=args.continuation_plies,
        per_category=args.positions_per_tactical_category,
    )
    model = load_model(args.checkpoint, device=args.device)
    model.eval()

    runs = []
    for index, sample in enumerate(positions, 1):
        generation = sample["_generation"]
        print(
            f"[{index}/{len(positions)}] {sample['_audit_label']} "
            f"seed={sample['seed']} ply={sample['ply']}",
            flush=True,
        )
        for budget in args.budgets:
            result = run_position(
                model, sample, budget=budget, c_puct=args.c_puct,
                device=args.device, root_category_prior_mix=0.0,
                opening_goal_directed_prior=False,
                avoid_recent_state_repetition=False,
                recent_state_keys=None,
            )
            result["sample"]["generation"] = generation
            runs.append(result)
            categories = result["pawn_wall"]
            print(
                f"  sims={budget} pawn_prior={result['network_pawn_prior_mass']:.4f} "
                f"wall_prior={result['network_wall_prior_mass']:.4f} "
                f"pawn_visits={categories['pawn']['visit_ratio']:.4f} "
                f"wall_visits={categories['wall']['visit_ratio']:.4f}",
                flush=True,
            )

    report = {
        "experiment": "synthetic_legal_position_prior_vs_puct_audit",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "generation": {
            "method": "seeded legal random trajectories for continuations; explicit scripted legal pawn sequences as fallback for tactical positions",
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            "max_game_plies": args.max_game_plies,
            "continuation_plies": args.continuation_plies,
            "positions_per_tactical_category": args.positions_per_tactical_category,
            "action_generation_policy": "uniform over available pawn/wall categories, then uniform within selected category",
            "tactical_definitions": {
                "opponent_winning_threat": "opponent shortest path <= 2; at least one legal wall delays opponent by >= 1 while own path cost <= 1",
                "strategic_wall_opportunity": "opponent shortest path >= 3; at least one legal wall has opponent delay minus own path cost >= 1",
            },
            "replay_dependency": False,
            "all_positions_engine_validated": True,
            "all_positions_non_terminal": True,
            "both_goal_paths_required": True,
        },
        "settings": {
            "progressive_widening": False,
            "root_noise": False,
            "root_category_prior_mix": 0.0,
            "category_balanced_prior": False,
            "category_exploration": 0.0,
            "repetition_penalty": 0.0,
            "pawn_progress_bonus": 0.0,
            "c_puct": args.c_puct,
            "randomize_selection_ties": False,
        },
        "position_count": len(positions),
        "budgets": args.budgets,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Synthetic PUCT audit: {args.output}")
    print(f"Completed {len(positions)} positions x {len(args.budgets)} budgets = {len(runs)} searches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
