#!/usr/bin/env python3
"""Generate AlphaZero-style Quoridor self-play samples with MCTS visit targets."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from quoridor.engine import GameState
from quoridor.neural_ai.encoder import apply_action_index, encode_state
from quoridor.neural_ai.mcts import MCTS
from quoridor.neural_ai.network import QuoridorNet
from quoridor.neural_ai.policy import load_model, save_checkpoint
from training.config import SelfPlayConfig


def play_game(model, *, game_id: int, seed: int, config: SelfPlayConfig, device: str) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    state = GameState.initial()
    search = MCTS(model, simulations=config.simulations, device=device)
    samples: list[dict] = []
    action_log: list[dict] = []

    while state.winner is None and state.ply < config.max_plies:
        player = state.current_player
        temperature = config.temperature if state.ply < config.temperature_moves else 0.0
        action, visits = search.choose_action(
            state,
            temperature=temperature,
            add_root_noise=config.dirichlet_noise,
        )
        samples.append({
            "state": encode_state(state),
            "policy": {str(key): value for key, value in visits.items()},
            "player": player,
            "ply": state.ply,
        })
        next_state = apply_action_index(state, action)
        action_log.append({"ply": state.ply, "player": player, "action": action})
        state = next_state
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
    args = parser.parse_args()

    if args.games < 1 or args.simulations < 1 or args.max_plies < 1:
        parser.error("--games, --simulations, and --max-plies must be positive")

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
    save_checkpoint(model, snapshot_path, extra={"source_checkpoint": args.checkpoint, "seed": args.seed})
    samples_path = output / "replay.jsonl"
    games_path = output / "games.jsonl"
    config = SelfPlayConfig(simulations=args.simulations, max_plies=args.max_plies)

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
