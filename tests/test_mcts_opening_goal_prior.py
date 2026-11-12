from quoridor.engine import GameState, shortest_distance_to_position
from quoridor.neural_ai.encoder import MOVE_ACTIONS, apply_action_index, legal_action_indices
from quoridor.neural_ai.mcts import MCTS


class UniformPolicy:
    pass


def make_mcts(**kwargs):
    mcts = MCTS(UniformPolicy(), progressive_widening=False, **kwargs)

    def evaluate(state):
        legal = legal_action_indices(state)
        probability = 1.0 / len(legal)
        return {action: probability for action in legal}, 0.0

    mcts._evaluate = evaluate
    return mcts


def test_distance_to_center_is_mirrored_for_both_players():
    state = GameState.initial()
    assert shortest_distance_to_position(state, 0, (4, 3)) == 5
    assert shortest_distance_to_position(state, 1, (4, 5)) == 5


def test_goal_directed_opening_prior_prioritizes_pawn_progress_and_keeps_puct():
    state = GameState.initial()
    baseline = make_mcts(simulations=8)
    guided = make_mcts(simulations=8, opening_goal_directed_prior=True)

    baseline.search(state)
    guided.search(state)

    base_stats = baseline.last_search_stats
    stats = guided.last_search_stats
    goal = stats["root_goal_directed_prior"]

    assert goal["active"] is True
    assert goal["mode"] == "pawn_to_center"
    assert goal["target"] == [4, 3]
    assert goal["own_target_distance"] == 5
    assert goal["candidate_actions"] >= 1
    assert all(int(action) < MOVE_ACTIONS for action in goal["candidate_action_weights"])

    base_pawn_mass = sum(
        item["effective_prior"] for item in base_stats["root_action_stats"].values()
        if item["category"] == "pawn"
    )
    guided_pawn_mass = sum(
        item["effective_prior"] for item in stats["root_action_stats"].values()
        if item["category"] == "pawn"
    )
    assert guided_pawn_mass > base_pawn_mass

    for item in stats["root_action_stats"].values():
        if item["active"]:
            assert abs(item["puct_score"] - (
                item["q"] + item["u_score"] + item["selection_adjustment"]
            )) < 1e-9


def test_goal_directed_prior_mirrors_center_target_for_player_one():
    state = GameState(
        pawns=((4, 8), (4, 0)),
        current_player=1,
        ply=1,
    )
    guided = make_mcts(simulations=4, opening_goal_directed_prior=True)
    guided.search(state)
    goal = guided.last_search_stats["root_goal_directed_prior"]

    assert goal["mode"] == "pawn_to_center"
    assert goal["target"] == [4, 5]
    assert goal["own_target_distance"] == 5
    assert all(int(action) < MOVE_ACTIONS for action in goal["candidate_action_weights"])


def test_distance_over_six_prioritizes_walls_that_extend_opponent_path():
    state = GameState(
        pawns=((0, 8), (4, 0)),
        current_player=0,
        ply=0,
    )
    assert shortest_distance_to_position(state, 0, (4, 3)) == 9

    guided = make_mcts(simulations=4, opening_goal_directed_prior=True)
    guided.search(state)
    goal = guided.last_search_stats["root_goal_directed_prior"]

    assert goal["mode"] == "extend_opponent_path"
    assert goal["own_target_distance"] == 9
    assert goal["candidate_actions"] > 0
    assert all(int(action) >= MOVE_ACTIONS for action in goal["candidate_action_weights"])
    assert goal["max_opponent_delay"] is not None


def test_goal_directed_prior_is_disabled_after_opening_window():
    state = GameState(
        pawns=((4, 8), (4, 0)),
        current_player=0,
        ply=10,
    )
    baseline = make_mcts(simulations=4)
    guided = make_mcts(simulations=4, opening_goal_directed_prior=True)
    baseline.search(state)
    guided.search(state)

    assert guided.last_search_stats["root_goal_directed_prior"] == {}
    for action, item in baseline.last_search_stats["root_action_stats"].items():
        guided_item = guided.last_search_stats["root_action_stats"][action]
        assert abs(item["effective_prior"] - guided_item["effective_prior"]) < 1e-12
