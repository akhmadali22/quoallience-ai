"""Integration checks for wall legality over complete random games."""
import random

from quoridor.engine import GameState
from quoridor.neural_ai.encoder import (
    apply_action_index,
    decode_action,
    legal_action_indices,
)


def test_random_games_never_place_the_same_wall_twice():
    # Multiple independent games exercise changing board geometries and turns.
    for seed in (17, 29, 43):
        rng = random.Random(seed)
        state = GameState.initial()
        placed_walls = set()

        for _ in range(160):
            if state.winner is not None:
                break

            action = rng.choice(legal_action_indices(state))
            kind, payload = decode_action(action)
            previous_wall_count = len(state.walls)

            if kind == "wall":
                assert payload not in placed_walls
                assert payload not in state.walls

            next_state = apply_action_index(state, action)

            if kind == "wall":
                placed_walls.add(payload)
                assert len(next_state.walls) == previous_wall_count + 1
            else:
                assert len(next_state.walls) == previous_wall_count

            assert len(next_state.walls) == len(placed_walls)
            state = next_state
