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
    mcts = make_mcts(simulations=4, repetition_penalty=0.75)
    action, visits = mcts.choose_action(
        GameState.initial(), rng=random.Random(7),
        recent_pawn_positions=[(4, 7), (4, 8)],
    )
    assert action in visits
    assert mcts.last_search_stats["root_repetition_adjusted_actions"] >= 1


def test_root_selection_adjustment_can_avoid_a_recent_reversal():
    node = Node(GameState.initial(), visits=10)
    node.edges = {0: Edge(prior=0.9), 1: Edge(prior=0.1)}
    mcts = make_mcts(c_puct=1.5)
    raw_action, _ = mcts._select(node)
    adjusted_action, _ = mcts._select(node, action_adjustments={0: -5.0})
    assert raw_action == 0
    assert adjusted_action == 1



def test_root_noise_and_ranking_run_before_initial_wall_admission():
    state = GameState.initial()
    legal_walls = [action for action in legal_action_indices(state) if action >= MOVE_ACTIONS]
    mcts = make_mcts(simulations=1, initial_wall_candidates=1, widening_coefficient=0.5)

    def rank_before_widening(search_state, root):
        assert not any(action >= MOVE_ACTIONS for action in root.edges)
        assert {action for action, _ in root.pending_walls} == set(legal_walls)
        root.pending_walls.reverse()
        return {}

    mcts._rank_root_wall_candidates = rank_before_widening
    mcts.search(state)

    active_walls = [
        int(action) for action, stats in mcts.last_search_stats["root_action_stats"].items()
        if stats["category"] == "wall" and stats["active"]
    ]
    assert active_walls == [legal_walls[-1]]


def test_backup_updates_leaf_and_all_ancestor_visits_with_alternating_values():
    root = Node(GameState.initial())
    middle = Node(GameState.initial())
    leaf = Node(GameState.initial())
    root_edge = Edge(prior=1.0)
    middle_edge = Edge(prior=1.0)

    MCTS._backup([(root, root_edge), (middle, middle_edge)], leaf, 0.6)

    assert root.visits == middle.visits == leaf.visits == 1
    assert root_edge.visits == middle_edge.visits == 1
    assert root_edge.value_sum == 0.6
    assert middle_edge.value_sum == -0.6



def test_selection_ties_can_be_randomized_reproducibly():
    import random

    node = Node(GameState.initial(), visits=0)
    node.edges = {
        0: Edge(prior=0.5),
        1: Edge(prior=0.5),
    }
    mcts = make_mcts(category_exploration=0.0)
    seeds = list(range(20))
    first = [
        mcts._select(node, rng=random.Random(seed))[0]
        for seed in seeds
    ]
    repeated = [
        mcts._select(node, rng=random.Random(seed))[0]
        for seed in seeds
    ]

    assert first == repeated
    assert set(first) == {0, 1}

def test_default_selection_uses_vanilla_puct_without_experimental_adjustments():
    mcts = make_mcts()
    assert mcts.category_exploration == 0.0
    assert mcts.repetition_penalty == 0.0
    assert mcts.pawn_progress_bonus == 0.0

    node = Node(GameState.initial(), visits=16)
    node.edges = {
        0: Edge(prior=0.75, visits=2, value_sum=0.5),
        1: Edge(prior=0.25, visits=0),
    }
    action, _ = mcts._select(node)

    q0 = 0.5 / 2
    u0 = 1.5 * 0.75 * (16 ** 0.5) / (1 + 2)
    q1 = 0.0
    u1 = 1.5 * 0.25 * (16 ** 0.5) / (1 + 0)
    assert q0 + u0 > q1 + u1
    assert action == 0


def test_repetition_preferences_are_opt_in_and_do_not_change_game_rules():
    mcts = make_mcts()
    state = GameState.initial()
    adjustments = mcts._root_action_adjustments(
        state, [(4, 7), (4, 8)]
    )
    assert adjustments == {}




def test_category_balanced_prior_splits_mass_equally_between_available_categories():
    import torch

    class ZeroLogitModel:
        def __call__(self, inputs):
            return torch.zeros((1, 193)), torch.zeros((1,))

    mcts = MCTS(ZeroLogitModel(), category_balanced_prior=True)
    priors, _ = mcts._evaluate(GameState.initial())
    pawn_mass = sum(value for action, value in priors.items() if action < MOVE_ACTIONS)
    wall_mass = sum(value for action, value in priors.items() if action >= MOVE_ACTIONS)
    assert abs(pawn_mass - 0.5) < 1e-6
    assert abs(wall_mass - 0.5) < 1e-6
    assert abs(sum(priors.values()) - 1.0) < 1e-6


def test_category_balanced_prior_falls_back_when_only_one_category_exists():
    import torch

    class ZeroLogitModel:
        def __call__(self, inputs):
            return torch.zeros((1, 193)), torch.zeros((1,))

    mcts = MCTS(ZeroLogitModel(), category_balanced_prior=True)
    # A synthetic terminal-adjacent state is not needed: directly exercise
    # the one-category normalization helper through a mocked legal-action set.
    import quoridor.neural_ai.mcts as mcts_module
    original = mcts_module.legal_action_indices
    try:
        mcts_module.legal_action_indices = lambda state: [0, 1, 2]
        priors, _ = mcts._evaluate(GameState.initial())
    finally:
        mcts_module.legal_action_indices = original
    assert abs(sum(priors.values()) - 1.0) < 1e-6
    assert all(value >= 0 for value in priors.values())
