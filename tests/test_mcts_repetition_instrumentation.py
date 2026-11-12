from quoridor.engine import GameState, Wall, state_repetition_key
from quoridor.neural_ai.encoder import apply_action_index, legal_action_indices
from quoridor.neural_ai.mcts import MOVE_ACTIONS, MCTS


class UniformPolicy:
    pass


def make_mcts(**kwargs):
    mcts = MCTS(UniformPolicy(), **kwargs)

    def evaluate(state):
        legal = legal_action_indices(state)
        probability = 1.0 / len(legal)
        return {action: probability for action in legal}, 0.0

    mcts._evaluate = evaluate
    return mcts


def test_repetition_key_ignores_ply_but_includes_complete_position():
    state = GameState.initial()
    later_same_position = GameState(
        pawns=state.pawns,
        walls=state.walls,
        walls_remaining=state.walls_remaining,
        current_player=state.current_player,
        winner=state.winner,
        ply=state.ply + 20,
    )
    assert state_repetition_key(state) == state_repetition_key(later_same_position)

    changed_side_to_move = GameState(
        pawns=state.pawns,
        walls=state.walls,
        walls_remaining=state.walls_remaining,
        current_player=1 - state.current_player,
        winner=state.winner,
        ply=state.ply,
    )
    assert state_repetition_key(state) != state_repetition_key(changed_side_to_move)



def test_repetition_key_includes_walls_and_remaining_inventory():
    state = GameState.initial()
    changed_walls = GameState(
        pawns=state.pawns,
        walls=frozenset({Wall("H", 0, 0)}),
        walls_remaining=(9, 10),
        current_player=state.current_player,
        ply=state.ply,
    )
    assert state_repetition_key(state) != state_repetition_key(changed_walls)


def test_root_diagnostics_report_prior_q_u_and_pawn_progress():
    state = GameState.initial()
    mcts = make_mcts(simulations=8, progressive_widening=False)
    mcts.search(state)
    stats = mcts.last_search_stats

    pawn_actions = {
        action: item for action, item in stats["root_action_stats"].items()
        if item["category"] == "pawn"
    }
    assert pawn_actions
    for item in pawn_actions.values():
        assert item["prior"] >= 0
        assert item["selection_prior"] >= 0
        assert item["u_score"] is not None
        assert item["puct_score"] is not None
        assert item["destination"]
        assert item["distance_before"] is not None
        assert item["distance_after"] is not None
        assert item["distance_delta"] is not None
        assert item["next_state_repetition_count"] == 0
        assert item["next_state_repeated"] is False


def test_full_state_repetition_is_reported_without_changing_puct():
    state = GameState.initial()
    pawn_action = next(
        action for action in legal_action_indices(state)
        if action < MOVE_ACTIONS
    )
    repeated_next_state = apply_action_index(state, pawn_action)
    history = [
        state_repetition_key(state),
        state_repetition_key(repeated_next_state),
    ]

    mcts = make_mcts(simulations=8, progressive_widening=False)
    mcts.search(state, recent_state_keys=history)
    stats = mcts.last_search_stats
    item = stats["root_action_stats"][str(pawn_action)]

    assert item["next_state_repetition_count"] == 1
    assert item["next_state_repeated"] is True
    assert stats["root_pawn_actions_with_repeated_next_state"] >= 1
    # Repetition instrumentation is observational: vanilla PUCT stays unadjusted.
    assert item["selection_adjustment"] == 0.0


def test_recent_state_history_is_bounded_by_repetition_window():
    state = GameState.initial()
    mcts = make_mcts(simulations=2, repetition_window=2)
    mcts.search(state, recent_state_keys=[
        ("old",), ("older",), state_repetition_key(state)
    ])
    assert mcts.last_search_stats["root_repetition_history_length"] == 3
    assert mcts.last_search_stats["root_repetition_window"] == 2



def test_root_diagnostics_separate_network_prior_from_dirichlet_prior():
    import torch

    state = GameState.initial()
    mcts = make_mcts(simulations=4, progressive_widening=False)
    torch.manual_seed(17)
    mcts.search(state, add_root_noise=True)
    stats = mcts.last_search_stats
    actions = stats["root_action_stats"]

    network_mass = sum(item["network_prior"] for item in actions.values())
    effective_mass = sum(item["effective_prior"] for item in actions.values())
    assert abs(network_mass - 1.0) < 1e-6
    assert abs(effective_mass - 1.0) < 1e-6
    assert any(
        abs(item["network_prior"] - item["effective_prior"]) > 1e-8
        for item in actions.values()
    )
    assert stats["root_network_prior_entropy"] == stats["root_prior_entropy"]
