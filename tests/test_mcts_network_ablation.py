import torch

from quoridor.neural_ai.encoder import ACTION_SIZE
from training.evaluate_mcts_ablation import AblatedModel, play_game


class FixedModel:
    def __call__(self, inputs):
        batch_size = inputs.shape[0]
        logits = torch.arange(12, dtype=torch.float32).repeat(batch_size, 1)
        values = torch.full((batch_size,), 0.625, dtype=torch.float32)
        return logits, values


def test_full_mode_preserves_policy_and_value_outputs():
    inputs = torch.zeros((2, 1))
    expected_logits, expected_values = FixedModel()(inputs)

    logits, values = AblatedModel(
        FixedModel(), policy_mode="neural", value_mode="neural"
    )(inputs)

    assert torch.equal(logits, expected_logits)
    assert torch.equal(values, expected_values)


def test_uniform_policy_removes_policy_preferences_but_keeps_value():
    inputs = torch.zeros((2, 1))
    logits, values = AblatedModel(
        FixedModel(), policy_mode="uniform", value_mode="neural"
    )(inputs)

    assert torch.count_nonzero(logits).item() == 0
    assert torch.allclose(values, torch.full((2,), 0.625))


def test_zero_value_removes_value_estimates_but_keeps_policy():
    inputs = torch.zeros((2, 1))
    logits, values = AblatedModel(
        FixedModel(), policy_mode="neural", value_mode="zero"
    )(inputs)

    assert torch.equal(logits, FixedModel()(inputs)[0])
    assert torch.count_nonzero(values).item() == 0


def test_combined_ablation_removes_both_network_outputs():
    inputs = torch.zeros((2, 1))
    logits, values = AblatedModel(
        FixedModel(), policy_mode="uniform", value_mode="zero"
    )(inputs)

    assert torch.count_nonzero(logits).item() == 0
    assert torch.count_nonzero(values).item() == 0



class ZeroLogitValueModel:
    def __call__(self, inputs):
        return (
            torch.zeros((inputs.shape[0], ACTION_SIZE), device=inputs.device),
            torch.zeros((inputs.shape[0],), device=inputs.device),
        )


def test_ablation_tournament_executes_a_short_mcts_game():
    result = play_game(
        ZeroLogitValueModel(),
        agent_a="full",
        agent_b="uniform_policy",
        seed=17,
        a_player=0,
        simulations=2,
        max_plies=2,
        device="cpu",
        c_puct=1.5,
        progressive_widening=False,
        initial_wall_candidates=8,
    )

    assert result["plies"] == 2
    assert result["agent_metrics"]["full"]["turns"] == 1
    assert result["agent_metrics"]["uniform_policy"]["turns"] == 1
    assert result["result_for_a"] == "draw"
    assert len(result["action_trace"]) == 2
    assert result["termination"]["draw_reason"] == "max_plies"
    assert result["termination"]["max_plies_reached"] is True
    assert len(result["diagnostics"]["final_shortest_distances"]) == 2
    assert result["diagnostics"]["no_progress_plies"] + result["diagnostics"]["progress_plies"] == 2
    assert all(
        "shortest_distance_before" in ply and "shortest_distance_after" in ply
        for ply in result["action_trace"]
    )




def test_pawn_position_repeat_diagnostic_ignores_wall_turns():
    from training.evaluate_mcts_ablation import _record_pawn_position

    history = {0: [(4, 8)], 1: [(4, 0)]}
    repeats = {0: 0, 1: 0}

    # A wall leaves the pawn stationary but is not another pawn-position visit.
    assert not _record_pawn_position(
        history, repeats, player=0, position=(4, 8), action_category="wall"
    )
    assert history[0] == [(4, 8)]
    assert repeats[0] == 0

    # A pawn move returning to the initial square is a genuine revisit.
    assert _record_pawn_position(
        history, repeats, player=0, position=(4, 8), action_category="pawn"
    )
    assert history[0] == [(4, 8), (4, 8)]
    assert repeats[0] == 1

    # A new pawn destination is recorded without incrementing the repeat count.
    assert not _record_pawn_position(
        history, repeats, player=0, position=(3, 8), action_category="pawn"
    )
    assert repeats[0] == 1


def test_root_action_summary_exposes_competing_puct_candidates():
    from training.evaluate_mcts_ablation import _top_root_actions

    stats = {
        "81": {"category": "wall", "prior": 0.8, "q": 0.2, "visits": 4, "active": True},
        "3": {"category": "pawn", "prior": 0.1, "q": 0.6, "visits": 7, "active": True},
        "4": {"category": "pawn", "prior": 0.4, "q": 0.1, "visits": 7, "active": True},
        "82": {"category": "wall", "prior": 0.9, "q": 0.0, "visits": 0, "active": False},
    }

    top = _top_root_actions(stats, limit=3)

    assert [item["action"] for item in top] == [3, 4, 81]
    assert top[0]["q"] == 0.6
    assert top[2]["active"] is True


def test_root_action_summary_can_rank_by_puct_and_prior():
    from training.evaluate_mcts_ablation import _top_root_actions

    stats = {
        "1": {"prior": 0.8, "visits": 2, "puct_score": 0.3},
        "2": {"prior": 0.1, "visits": 9, "puct_score": 0.7},
        "3": {"prior": 0.4, "visits": 4, "puct_score": 0.5},
    }

    assert [x["action"] for x in _top_root_actions(stats, sort_by="prior")] == [1, 3, 2]
    assert [x["action"] for x in _top_root_actions(stats, sort_by="puct_score")] == [2, 3, 1]
    assert [x["action"] for x in _top_root_actions(stats, sort_by="visits")] == [2, 3, 1]
