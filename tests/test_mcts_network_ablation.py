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
