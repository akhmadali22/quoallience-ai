import torch

from training.evaluate_mcts_ablation import AblatedModel


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
