import time

import pytest

from quoridor.engine import (
    GameState,
    Wall,
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


def test_urgent_defense_includes_wall_blocking_immediate_win():
    state = GameState(
        pawns=((1, 2), (0, 7)),
        current_player=0,
    )
    ai = QuoridorAI(
        AIConfig(time_limit=1.0, max_depth=1)
    )
    ai.root_player = 0
    ai.deadline = time.monotonic() + 1.0

    urgent = ai._urgent_defensive_walls(state)

    assert Wall("H", 0, 7) in urgent


def test_fallback_uses_defensive_wall_when_all_moves_allow_immediate_win():
    state = GameState(
        pawns=((1, 2), (0, 7)),
        current_player=0,
    )
    ai = QuoridorAI(
        AIConfig(time_limit=1.0, max_depth=1)
    )
    ai.root_player = 0
    ai.deadline = __import__("time").monotonic() + 1.0

    action = ai._fallback_action(state)

    assert action.kind == "wall"
    assert action.wall == Wall("H", 0, 7)


def test_wall_candidates_preserve_urgent_defense_outside_shortlist():
    state = GameState(
        pawns=((1, 2), (0, 7)),
        current_player=0,
    )
    ai = QuoridorAI(
        AIConfig(
            time_limit=3.0,
            max_depth=1,
            max_wall_candidates=1,
            max_alternative_paths=1,
        )
    )
    ai.root_player = 0
    ai.deadline = __import__("time").monotonic() + 3.0

    candidates = ai._candidate_walls(state)

    assert Wall("H", 0, 7) in candidates

def test_route_flexibility_prefers_center_with_more_near_optimal_options():
    center_state = GameState(
        pawns=((4, 4), (4, 2)),
        current_player=0,
    )
    edge_state = GameState(
        pawns=((0, 4), (4, 2)),
        current_player=0,
    )
    ai = QuoridorAI(
        AIConfig(time_limit=2.0, max_depth=1)
    )
    ai.root_player = 0
    ai.deadline = time.monotonic() + 2.0

    center_flexibility = ai._route_flexibility(center_state, 0)
    edge_flexibility = ai._route_flexibility(edge_state, 0)

    assert center_flexibility > edge_flexibility


def test_static_evaluation_accounts_for_route_flexibility():
    center_state = GameState(
        pawns=((4, 4), (4, 2)),
        current_player=0,
    )
    edge_state = GameState(
        pawns=((0, 4), (4, 2)),
        current_player=0,
    )
    ai = QuoridorAI(
        AIConfig(time_limit=2.0, max_depth=1)
    )
    ai.root_player = 0
    ai.deadline = time.monotonic() + 2.0

    center_score = ai._evaluate(center_state)
    edge_score = ai._evaluate(edge_state)

    # Both pawns have the same shortest distance and wall reserves;
    # the center position should score higher because it retains more routes.
    assert center_score > edge_score

def test_search_extends_horizon_for_near_goal_tactics():
    state = GameState(
        pawns=((4, 1), (0, 7)),
        current_player=1,
    )
    ai = QuoridorAI(
        AIConfig(
            time_limit=1.0,
            max_depth=1,
            endgame_extension_distance=3,
            endgame_extension_plies=2,
        )
    )
    ai.root_player = 0
    ai.deadline = time.monotonic() + 1.0

    score = ai._minimax(
        state,
        depth=0,
        alpha=float("-inf"),
        beta=float("inf"),
        extensions_remaining=2,
    )

    # The opponent can win in one move. A leaf evaluation must not treat
    # this as an ordinary positional state just because nominal depth is 0.
    assert score <= -10_000.0
