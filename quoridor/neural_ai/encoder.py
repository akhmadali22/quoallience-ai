"""Perspective-relative Quoridor state and action encoding for AlphaZero-style agents."""
from __future__ import annotations

from typing import TypeAlias

from quoridor.engine import (
    GameState, Wall, apply_move, apply_wall, legal_pawn_moves, legal_walls,
)

BOARD_SIZE = 9
MOVE_ACTIONS = BOARD_SIZE * BOARD_SIZE
WALL_ACTIONS = 2 * 8 * 8
ACTION_SIZE = MOVE_ACTIONS + WALL_ACTIONS
INPUT_CHANNELS = 10
Action: TypeAlias = tuple[str, tuple[int, int] | Wall]


def encode_state(state: GameState) -> list[list[list[float]]]:
    """Return 10x9x9 planes, always from the side-to-move perspective."""
    player = state.current_player
    opponent = 1 - player
    planes = [[[0.0 for _ in range(9)] for _ in range(9)] for _ in range(INPUT_CHANNELS)]

    x, y = state.pawns[player]
    planes[0][y][x] = 1.0
    x, y = state.pawns[opponent]
    planes[1][y][x] = 1.0

    own_goal_y = 0 if player == 0 else 8
    opponent_goal_y = 8 if player == 0 else 0
    for x in range(9):
        planes[2][own_goal_y][x] = 1.0
        planes[3][opponent_goal_y][x] = 1.0

    for wall in state.walls:
        plane = 4 if wall.orientation == "H" else 5
        planes[plane][wall.y][wall.x] = 1.0

    planes[6] = [[state.walls_remaining[player] / 10.0] * 9 for _ in range(9)]
    planes[7] = [[state.walls_remaining[opponent] / 10.0] * 9 for _ in range(9)]
    planes[8] = [[1.0 if state.current_player == 0 else 0.0] * 9 for _ in range(9)]
    planes[9] = [[min(state.ply, 200) / 200.0] * 9 for _ in range(9)]
    return planes


def encode_action(action: Action) -> int:
    kind, payload = action
    if kind == "move":
        x, y = payload  # type: ignore[misc]
        return y * 9 + x
    wall = payload
    if not isinstance(wall, Wall):
        raise TypeError("wall action payload must be a Wall")
    offset = MOVE_ACTIONS if wall.orientation == "H" else MOVE_ACTIONS + 64
    return offset + wall.y * 8 + wall.x


def decode_action(index: int) -> Action:
    if not 0 <= index < ACTION_SIZE:
        raise ValueError(f"Action index out of range: {index}")
    if index < MOVE_ACTIONS:
        return ("move", (index % 9, index // 9))
    index -= MOVE_ACTIONS
    orientation = "H"
    if index >= 64:
        orientation = "V"
        index -= 64
    return ("wall", Wall(orientation, index % 8, index // 8))


def legal_action_indices(state: GameState) -> list[int]:
    if state.winner is not None:
        return []
    actions = [encode_action(("move", move)) for move in legal_pawn_moves(state)]
    actions.extend(encode_action(("wall", wall)) for wall in legal_walls(state))
    return sorted(actions)


def apply_action_index(state: GameState, index: int) -> GameState:
    kind, payload = decode_action(index)
    if kind == "move":
        return apply_move(state, payload)  # type: ignore[arg-type]
    return apply_wall(state, payload)  # type: ignore[arg-type]
