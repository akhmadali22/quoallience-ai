import torch

from quoridor.engine import GameState
from quoridor.neural_ai.policy import load_model
from quoridor.neural_ai.mcts import MCTS
from quoridor.neural_ai.encoder import decode_action, legal_action_indices

state = GameState.initial()
model = load_model("data/neural_self_play/smoke/selfplay_model_snapshot.pt")
model.eval()

for simulations in [16, 64, 128]:
    mcts = MCTS(
        model=model,
        simulations=simulations,
    )

    visits = mcts.search(state)

    print(f"\n=== Simulations: {simulations} ===")

    ranked = sorted(
        visits.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    pawn_visits = 0
    wall_visits = 0

    for action, count in ranked[:15]:
        decoded = decode_action(action)
        print(f"action={action:3d} visits={count:8.3f} -> {decoded}")

    for action, count in visits.items():
        if action < 81:
            pawn_visits += count
        else:
            wall_visits += count

    total = sum(visits.values())

    pawn_visits = sum(count for action, count in visits.items() if action < 81)

    wall_visits = sum(count for action, count in visits.items() if action >= 81)

    total = sum(visits.values())

    print(f"Total visits: {total:.3f}")
    print(f"Pawn visits: {pawn_visits:.3f}")
    print(f"Wall visits: {wall_visits:.3f}")
    print(f"Pawn ratio: {pawn_visits / total:.2%}" if total else "Pawn ratio: 0%")
