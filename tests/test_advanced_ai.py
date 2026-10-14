import pytest

from quoridor.engine import (
    GameState,
    apply_move,
    apply_wall,
    legal_pawn_moves,
)
from quoridor.advanced_ai import (
    AIConfig,
    Action,
    QuoridorAI,
)


def test_ai_returns_legal_pawn_move_or_wall():
    state = GameState()

    ai = QuoridorAI(
        AIConfig(
            time_limit=0.5,
            max_depth=1,
        )
    )

    result = ai.choose_action(state)

    assert isinstance(result.action, Action)

    if result.action.kind == "move":
        assert result.action.destination in legal_pawn_moves(
            state,
            state.current_player,
        )

    else:
        assert result.action.kind == "wall"
        assert result.action.wall is not None


def test_ai_move_can_be_applied():
    state = GameState()

    ai = QuoridorAI(
        AIConfig(
            time_limit=0.5,
            max_depth=1,
        )
    )

    result = ai.choose_action(state)

    if result.action.kind == "move":
        next_state = apply_move(
            state,
            result.action.destination,
        )
    else:
        next_state = apply_wall(
            state,
            result.action.wall,
        )

    assert next_state is not None
    assert next_state.current_player != state.current_player


def test_ai_returns_search_metadata():
    state = GameState()

    ai = QuoridorAI(
        AIConfig(
            time_limit=0.5,
            max_depth=1,
        )
    )

    result = ai.choose_action(state)

    assert result.depth >= 0
    assert result.elapsed_ms >= 0
    assert result.nodes >= 0
    assert isinstance(result.timed_out, bool)


def test_ai_rejects_finished_game():
    state = GameState(
        pawns=((4, 0), (4, 8)),
        winner=0,
    )

    ai = QuoridorAI()

    with pytest.raises(ValueError):
        ai.choose_action(state)


def test_ai_rejects_invalid_time_limit():
    ai = QuoridorAI(
        AIConfig(time_limit=0)
    )

    with pytest.raises(ValueError):
        ai.choose_action(GameState())


def test_ai_can_play_without_walls():
    state = GameState(
        walls_remaining=(0, 0),
    )

    ai = QuoridorAI(
        AIConfig(
            time_limit=0.5,
            max_depth=1,
        )
    )

    result = ai.choose_action(state)

    assert result.action.kind == "move"


def test_wall_candidates_are_inside_board():
    path = [
        (4, 8),
        (4, 7),
        (4, 6),
        (5, 6),
    ]

    walls = QuoridorAI._wall_candidates_from_path(path)

    assert walls

    for wall in walls:
        assert wall.orientation in ("H", "V")
        assert 0 <= wall.x <= 7
        assert 0 <= wall.y <= 7