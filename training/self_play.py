#!/usr/bin/env python3
"""Generate AlphaZero-style Quoridor self-play samples with MCTS visit targets."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from quoridor.engine import GameState, state_repetition_key
from quoridor.neural_ai.encoder import apply_action_index, encode_state
from quoridor.neural_ai.mcts import MOVE_ACTIONS, MCTS
from quoridor.neural_ai.network import QuoridorNet
from quoridor.neural_ai.policy import load_model, save_checkpoint
from training.config import SelfPlayConfig



def compact_root_diagnostics(stats: dict) -> dict:
    """Keep complete pawn diagnostics and a compact wall-prior reference."""
    actions = stats.get("root_action_stats", {})
    pawn_actions = {
        action: details for action, details in actions.items()
        if details.get("category") == "pawn"
    }
    wall_actions = [
        (action, details) for action, details in actions.items()
        if details.get("category") == "wall"
    ]
    top_walls = sorted(
        wall_actions, key=lambda item: item[1].get("prior", 0.0), reverse=True
    )[:5]
    keys = (
        "root_value", "root_prior_entropy", "root_active_prior_entropy",
        "root_visit_entropy", "root_pawn_visit_ratio", "root_wall_visit_ratio",
        "root_active_pawn_prior_mass",
        "root_active_wall_prior_mass", "root_pawn_visits", "root_wall_visits",
        "root_legal_actions", "root_active_actions",
        "root_repetition_history_length", "root_repetition_window",
        "root_current_state_repetition_count",
        "root_pawn_actions_with_repeated_next_state",
        "root_pawn_actions_with_progress", "root_pawn_actions_with_regression",
    )
    compact = {key: stats.get(key) for key in keys}
    compact["pawn_actions"] = pawn_actions
    compact["top_wall_actions_by_prior"] = {
        action: details for action, details in top_walls
    }
    return compact


def play_game(model, *, game_id: int, seed: int, config: SelfPlayConfig, device: str) -> tuple[list[dict], dict]:
    random.seed(seed)
    state = GameState.initial()
    search = MCTS(
        model, simulations=config.simulations, device=device,
        category_exploration=config.category_exploration,
        repetition_penalty=config.repetition_penalty,
        pawn_progress_bonus=config.pawn_progress_bonus,
        repetition_window=config.repetition_window,
        c_puct=config.c_puct,
        progressive_widening=config.progressive_widening,
        initial_wall_candidates=config.initial_wall_candidates,
    )
    samples: list[dict] = []
    action_log: list[dict] = []
    pawn_positions = {0: [state.pawns[0]], 1: [state.pawns[1]]}
    state_history = [state_repetition_key(state)]

    while state.winner is None and state.ply < config.max_plies:
        player = state.current_player
        temperature = config.temperature if state.ply < config.temperature_moves else 0.0
        action, visits = search.choose_action(
            state,
            temperature=temperature,
            add_root_noise=config.dirichlet_noise,
            recent_pawn_positions=pawn_positions[player],
            recent_state_keys=state_history,
        )
        search_stats = search.last_search_stats
        samples.append({
            "state": encode_state(state),
            "policy": {str(key): value for key, value in visits.items()},
            "policy_mask": search_stats.get("root_policy_mask", sorted(visits)),
            "player": player,
            "ply": state.ply,
            "root_value": search_stats.get("root_value"),
            "root_policy_entropy": search_stats.get("root_policy_entropy"),
            "root_pawn_visit_ratio": search_stats.get("root_pawn_visit_ratio"),
            "root_repetition_adjusted_actions": search_stats.get("root_repetition_adjusted_actions"),
            "root_diagnostics": compact_root_diagnostics(search_stats),
        })
        next_state = apply_action_index(state, action)
        if action < MOVE_ACTIONS:
            pawn_positions[player].append(next_state.pawns[player])
        action_log.append({
            "ply": state.ply, "player": player, "action": action,
            "selected_action_root_stats": search_stats.get("root_action_stats", {}).get(str(action)),
            "root_diagnostics": compact_root_diagnostics(search_stats),
        })
        state = next_state
        state_history.append(state_repetition_key(state))
        if state.ply % 10 == 0:
            print(f"  game={game_id} ply={state.ply} winner={state.winner}", flush=True)

    winner = state.winner
    result = "win" if winner is not None else "draw_max_plies"
    for sample in samples:
        sample["value"] = (
            0.0 if winner is None
            else (1.0 if winner == sample["player"] else -1.0)
        )
        sample["game_id"] = game_id
        sample["seed"] = seed

    return samples, {
        "game_id": game_id,
        "seed": seed,
        "winner": winner,
        "result": result,
        "plies": state.ply,
        "samples": len(samples),
        "actions": action_log,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Optional model checkpoint; otherwise use a random initial network.")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=48)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--output-dir", default="data/neural_self_play")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--progressive-widening", action="store_true",
                        help="Enable the optional wall-candidate progressive-widening extension.")
    parser.add_argument("--initial-wall-candidates", type=int, default=8)
    parser.add_argument("--category-exploration", type=float, default=0.0,
                        help="Optional category exploration bonus; zero keeps vanilla PUCT.")
    parser.add_argument("--repetition-penalty", type=float, default=0.0,
                        help="Optional root repetition preference; zero keeps vanilla PUCT.")
    parser.add_argument("--pawn-progress-bonus", type=float, default=0.0)
    parser.add_argument("--repetition-window", type=int, default=4)
    args = parser.parse_args()

    if args.games < 1 or args.simulations < 1 or args.max_plies < 1:
        parser.error("--games, --simulations, and --max-plies must be positive")
    if (args.c_puct < 0 or args.initial_wall_candidates < 0
            or args.category_exploration < 0 or args.repetition_penalty < 0
            or args.pawn_progress_bonus < 0 or args.repetition_window < 1):
        parser.error("PUCT and preference values must be non-negative; repetition-window must be positive")

    import torch
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    model = load_model(args.checkpoint, device=device) if args.checkpoint else QuoridorNet(
        channels=args.channels, residual_blocks=args.residual_blocks
    ).to(device)
    model.eval()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = output / "selfplay_model_snapshot.pt"
    samples_path = output / "replay.jsonl"
    games_path = output / "games.jsonl"
    if samples_path.exists() or games_path.exists():
        parser.error(f"Output already contains replay data: {output}. Use a new --output-dir.")
    save_checkpoint(model, snapshot_path, extra={"source_checkpoint": args.checkpoint, "seed": args.seed})
    config = SelfPlayConfig(
        simulations=args.simulations, max_plies=args.max_plies,
        category_exploration=args.category_exploration,
        repetition_penalty=args.repetition_penalty,
        pawn_progress_bonus=args.pawn_progress_bonus,
        repetition_window=args.repetition_window,
        c_puct=args.c_puct,
        progressive_widening=args.progressive_widening,
        initial_wall_candidates=args.initial_wall_candidates,
    )

    with samples_path.open("a", encoding="utf-8") as samples_file, games_path.open("a", encoding="utf-8") as games_file:
        for index in range(args.games):
            game_id = index + 1
            seed = args.seed + index
            print(f"[{index + 1}/{args.games}] game={game_id} seed={seed}", flush=True)
            samples, summary = play_game(model, game_id=game_id, seed=seed, config=config, device=device)
            for sample in samples:
                samples_file.write(json.dumps(sample, separators=(",", ":")) + "\n")
            samples_file.flush()
            games_file.write(json.dumps(summary, separators=(",", ":")) + "\n")
            games_file.flush()
            print(
                f"  {summary['result']} winner={summary['winner']} "
                f"plies={summary['plies']} samples={summary['samples']}",
                flush=True,
            )
    print(f"Model snapshot: {snapshot_path}")
    print(f"Replay data: {samples_path}")
    print(f"Game summaries: {games_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
