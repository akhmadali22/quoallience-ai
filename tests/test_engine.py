from dataclasses import replace

import pytest

from quoridor.engine import (
    GameState,
    Wall,
    apply_move,
    apply_wall,
    is_legal_wall,
    legal_pawn_moves,
    shortest_distance,
    shortest_path,
    validate_state,
)


def test_initial_state():
    state = GameState.initial()

    assert state.pawns == ((4, 8), (4, 0))
    assert shortest_distance(state, 0) == 8
    assert shortest_distance(state, 1) == 8

    validate_state(state)


def test_basic_move():
    state = GameState.initial()

    assert (4, 7) in legal_pawn_moves(state, 0)

    updated = apply_move(state, (4, 7))

    assert updated.pawns[0] == (4, 7)
    assert updated.current_player == 1
    assert updated.ply == 1


def test_turn_order():
    state = GameState.initial()
    state = apply_move(state, (4, 7))

    with pytest.raises(ValueError):
        apply_move(state, (4, 6), player=0)


def test_wall_placement():
    state = GameState.initial()
    wall = Wall("H", 3, 6)

    assert is_legal_wall(state, wall, 0)

    updated = apply_wall(state, wall)

    assert wall in updated.walls
    assert updated.walls_remaining == (9, 10)
    assert updated.current_player == 1


def test_wall_overlap():
    state = replace(
        GameState.initial(),
        walls=frozenset({Wall("H", 2, 2)}),
    )

    assert not is_legal_wall(state, Wall("H", 3, 2), 0)
    assert not is_legal_wall(state, Wall("V", 2, 2), 0)
    assert is_legal_wall(state, Wall("H", 4, 2), 0)


def test_wall_must_preserve_paths():
    state = GameState.initial()

    assert is_legal_wall(state, Wall("V", 0, 0), 0)

    candidate = replace(
        state,
        walls=state.walls | {Wall("V", 0, 0)},
    )

    assert shortest_path(candidate, 0) is not None
    assert shortest_path(candidate, 1) is not None


def test_straight_jump():
    state = replace(
        GameState.initial(),
        pawns=((4, 5), (4, 4)),
    )

    moves = legal_pawn_moves(state, 0)

    assert (4, 3) in moves


def test_diagonal_jump_when_straight_jump_blocked():
    state = replace(
        GameState.initial(),
        pawns=((4, 5), (4, 4)),
        walls=frozenset({Wall("H", 3, 3)}),
    )

    moves = legal_pawn_moves(state, 0)

    assert (3, 4) in moves
    assert (5, 4) in moves
    assert (4, 3) not in moves


def test_win_detection():
    state = replace(
        GameState.initial(),
        pawns=((4, 1), (0, 0)),
    )

    updated = apply_move(state, (4, 0), player=0)

    assert updated.winner == 0
    assert updated.current_player == 1

    with pytest.raises(ValueError):
        apply_move(updated, (1, 0), player=1)


def test_illegal_move_is_rejected():
    state = GameState.initial()

    with pytest.raises(ValueError):
        apply_move(state, (0, 0))


def test_serialization():
    data = GameState.initial().to_dict()

    assert data["current_player"] == 0
    assert data["walls_remaining"] == [10, 10]
    assert data["pawns"][0] == {"x": 4, "y": 8}
