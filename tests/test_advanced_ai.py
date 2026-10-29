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

def test_opponent_reply_considers_walls_against_alternative_routes():
    # P0 has just made the root move; P1 must consider closing not only
    # P0's shortest route but also one of P0's likely reroutes.
    state = apply_move(GameState(), (4, 7))
    ai = QuoridorAI(
        AIConfig(
            time_limit=5,
            max_depth=1,
            max_wall_candidates=8,
            predictive_path_count=3,
            predictive_wall_candidates=2,
        )
    )
    ai.root_player = 0
    ai.root_ply = 0
    ai.deadline = 10**12
    ai.stats = {
        "children_generated": 0,
        "move_candidates": 0,
        "wall_candidates": 0,
        "predictive_wall_candidates": 0,
        "alpha_beta_cutoffs": 0,
    }

    candidates = ai._candidate_walls(state)

    assert candidates
    assert ai.stats["predictive_wall_candidates"] > 0
    assert len(candidates) <= ai.config.max_wall_candidates
