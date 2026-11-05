from quoridor.engine import GameState
from quoridor.neural_ai.encoder import legal_action_indices
from quoridor.neural_ai.mcts import MOVE_ACTIONS, Edge, MCTS, Node


class UniformPolicy:
    """Placeholder model; tests replace MCTS._evaluate with deterministic priors."""


def make_mcts(**kwargs):
    mcts = MCTS(UniformPolicy(), **kwargs)

    def evaluate(state):
        legal = legal_action_indices(state)
        probability = 1.0 / len(legal)
        return {action: probability for action in legal}, 0.0

    mcts._evaluate = evaluate
    return mcts


def test_progressive_widening_keeps_every_legal_pawn_move_active():
    state = GameState.initial()
    legal = legal_action_indices(state)
    pawn_actions = {action for action in legal if action < MOVE_ACTIONS}
    wall_actions = {action for action in legal if action >= MOVE_ACTIONS}

    mcts = make_mcts(
        progressive_widening=True,
        initial_wall_candidates=5,
        widening_coefficient=2.0,
        widening_exponent=0.5,
    )
    node = Node(state)

    mcts._expand(node)

    assert set(node.edges) >= pawn_actions
    assert all(action < MOVE_ACTIONS for action in node.edges if action in pawn_actions)
    assert sum(action >= MOVE_ACTIONS for action in node.edges) == 5
    assert {action for action, _ in node.pending_walls} == (
        wall_actions - set(node.edges)
    )


def test_progressive_widening_adds_wall_candidates_as_visits_grow():
    state = GameState.initial()
    mcts = make_mcts(
        progressive_widening=True,
        initial_wall_candidates=4,
        widening_coefficient=2.0,
        widening_exponent=0.5,
    )
    node = Node(state)
    mcts._expand(node)

    assert sum(action >= MOVE_ACTIONS for action in node.edges) == 4

    node.visits = 64
    mcts._widen(node)
    assert sum(action >= MOVE_ACTIONS for action in node.edges) == 16

    node.visits = 256
    mcts._widen(node)
    assert sum(action >= MOVE_ACTIONS for action in node.edges) == 32


def test_progressive_widening_search_returns_all_legal_actions_and_stats():
    state = GameState.initial()
    legal = set(legal_action_indices(state))
    mcts = make_mcts(
        simulations=32,
        progressive_widening=True,
        initial_wall_candidates=4,
    )

    policy = mcts.search(state)

    assert set(policy) == legal
    assert abs(sum(policy.values()) - 1.0) < 1e-8
    assert all(probability >= 0 for probability in policy.values())

    stats = mcts.last_search_stats
    assert stats["simulations"] == 32
    assert stats["expanded_nodes"] >= 2
    assert stats["nn_evaluations"] >= 2
    assert stats["root_active_wall_actions"] < stats["root_legal_actions"]
    assert stats["root_pending_wall_actions"] > 0
    assert stats["pawn_visit_ratio"] + stats["wall_visit_ratio"] == 1.0
    assert stats["max_leaf_depth"] >= 1


def test_disabling_progressive_widening_keeps_all_actions_active():
    state = GameState.initial()
    legal = set(legal_action_indices(state))
    mcts = make_mcts(progressive_widening=False)
    node = Node(state)

    mcts._expand(node)

    assert set(node.edges) == legal
    assert node.pending_walls == []


def _category_selection_node():
    node = Node(GameState.initial(), visits=100)
    # Raw PUCT prefers wall action 81 due to its higher prior. A second
    # wall edge has 100 visits, so the aggregate wall category is visited
    # heavily while the pawn category remains unexplored.
    node.edges = {
        0: Edge(prior=0.2),
        81: Edge(prior=0.8),
        82: Edge(prior=0.0, visits=100),
    }
    return node


def test_category_exploration_bonus_can_override_raw_puct_preference():
    mcts = make_mcts(category_exploration=0.5, c_puct=0.1)
    action, _ = mcts._select(_category_selection_node())
    assert action == 0


def test_zero_category_exploration_preserves_raw_puct_preference():
    mcts = make_mcts(category_exploration=0.0, c_puct=0.1)
    action, _ = mcts._select(_category_selection_node())
    assert action == 81


def test_category_exploration_rejects_negative_value():
    import pytest
    with pytest.raises(ValueError, match='category_exploration'):
        make_mcts(category_exploration=-0.1)

def test_choose_action_uses_seeded_rng_for_equal_visit_ties():
    import random

    state = GameState.initial()
    mcts = make_mcts(simulations=1)
    mcts.search = lambda state, add_root_noise=False: {10: 0.5, 20: 0.5}

    first, _ = mcts.choose_action(state, rng=random.Random(1234))
    second, _ = mcts.choose_action(state, rng=random.Random(1234))

    assert first == second
    assert first in {10, 20}


def test_temperature_sampling_is_seeded_and_varies_across_seeds():
    import random
    state = GameState.initial()
    mcts = make_mcts(simulations=1)
    mcts.search = lambda state, add_root_noise=False: {10: 0.5, 20: 0.5}
    first = [mcts.choose_action(state, temperature=1.0, rng=random.Random(11 + i))[0] for i in range(20)]
    repeated = [mcts.choose_action(state, temperature=1.0, rng=random.Random(11 + i))[0] for i in range(20)]
    assert first == repeated
    assert set(first) == {10, 20}


def test_search_reports_per_action_root_diagnostics():
    mcts = make_mcts(simulations=8, initial_wall_candidates=3)
    mcts.search(GameState.initial())
    stats = mcts.last_search_stats
    assert stats['root_action_stats']
    assert stats['root_pawn_visits'] + stats['root_wall_visits'] == 8
    assert 0.0 <= stats['root_pawn_visit_ratio'] <= 1.0
    assert stats['root_value'] == 0.0
    assert stats['root_policy_entropy'] >= 0.0
    assert all({'category', 'prior', 'q', 'visits', 'active'} <= set(item) for item in stats['root_action_stats'].values())


def test_root_action_adjustments_penalize_recent_pawn_reversal_and_reward_progress():
    from quoridor.neural_ai.encoder import encode_action

    mcts = make_mcts(repetition_penalty=0.75, pawn_progress_bonus=0.10)
    state = GameState.initial()
    # The current pawn is at (4, 8); (4, 7) is a recent prior square.
    adjustments = mcts._root_action_adjustments(
        state, [(4, 7), (4, 8)]
    )
    repeated_destination = encode_action(("move", (4, 7)))
    progress_state_history = mcts._root_action_adjustments(
        state, [(3, 8), (4, 8)]
    )
    assert adjustments[repeated_destination] < 0
    assert progress_state_history[repeated_destination] > 0


def test_root_adjustments_are_empty_without_history():
    mcts = make_mcts()
    assert mcts._root_action_adjustments(GameState.initial(), None) == {}


def test_choose_action_accepts_recent_pawn_history():
    import random
    mcts = make_mcts(simulations=4)
    action, visits = mcts.choose_action(
        GameState.initial(), rng=random.Random(7),
        recent_pawn_positions=[(4, 7), (4, 8)],
    )
    assert action in visits
    assert mcts.last_search_stats["root_repetition_adjusted_actions"] >= 1
