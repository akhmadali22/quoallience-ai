from dataclasses import replace

import pytest

from quoridor.ai import AIConfig, choose_action
from quoridor.engine import (
    GameState,
    Wall,
    apply_move,
    apply_wall,
    legal_pawn_moves,
    legal_walls,
)


def test_ai_returns_a_legal_action():
    state = GameState.initial()

    decision = choose_action(state)

    assert decision.action.type in ("move", "wall")

    if decision.action.type == "move":
        assert decision.action.destination in legal_pawn_moves(state)
    else:
        assert decision.action.wall in legal_walls(state)


def test_ai_move_can_be_applied():
    state = GameState.initial()
    decision = choose_action(state)

    if decision.action.type == "move":
        updated = apply_move(state, decision.action.destination)
    else:
        updated = apply_wall(state, decision.action.wall)

    assert updated.ply == 1
    assert updated.current_player == 1


def test_ai_returns_explanation_and_score():
    decision = choose_action(GameState.initial())

    assert isinstance(decision.score, float)
    assert decision.explanation
    assert decision.elapsed_ms >= 0
    assert len(decision.candidates) >= 1


def test_ai_rejects_finished_game():
    state = replace(
        GameState.initial(),
        pawns=((4, 0), (0, 0)),
        winner=0,
    )

    with pytest.raises(ValueError):
        choose_action(state)


def test_ai_rejects_invalid_time_limit():
    with pytest.raises(ValueError):
        choose_action(
            GameState.initial(),
            AIConfig(time_limit=0),
        )


def test_ai_handles_exhausted_wall_supply():
    state = replace(
        GameState.initial(),
        walls_remaining=(0, 10),
    )

    decision = choose_action(state)

    assert decision.action.type == "move"
