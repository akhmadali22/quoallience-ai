from quoridor.engine import GameState
from quoridor.neural_ai.encoder import legal_action_indices
from quoridor.neural_ai.mcts import MOVE_ACTIONS, MCTS, Node


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


def test_category_balancing_sets_requested_mass_and_preserves_relative_preferences():
    from quoridor.neural_ai.mcts import balance_category_priors
    priors = {0: 0.1, 1: 0.3, 81: 0.2, 82: 0.4}
    balanced = balance_category_priors(priors, pawn_fraction=0.5)
    assert abs(sum(balanced.values()) - 1.0) < 1e-9
    assert abs(balanced[0] + balanced[1] - 0.5) < 1e-9
    assert abs(balanced[81] + balanced[82] - 0.5) < 1e-9
    assert balanced[1] > balanced[0] and balanced[82] > balanced[81]


def test_category_balancing_supports_pawn_biased_fraction():
    from quoridor.neural_ai.mcts import balance_category_priors
    balanced = balance_category_priors({0: 0.1, 1: 0.3, 81: 0.2, 82: 0.4}, 0.7)
    assert abs(balanced[0] + balanced[1] - 0.7) < 1e-9
    assert abs(balanced[81] + balanced[82] - 0.3) < 1e-9


def test_category_balancing_normalizes_when_only_one_category_exists():
    from quoridor.neural_ai.mcts import balance_category_priors
    balanced = balance_category_priors({81: 0.2, 82: 0.8}, 0.5)
    assert abs(sum(balanced.values()) - 1.0) < 1e-9
    assert balanced == {81: 0.2, 82: 0.8}