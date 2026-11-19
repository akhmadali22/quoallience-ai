import pytest
import torch

from quoridor.neural_ai.encoder import ACTION_SIZE
from training.train import build_training_tensors


def test_policy_target_masks_actions_not_admitted_by_progressive_widening():
    sample = {
        "state": [0.0],
        "policy": {"3": 0.75, "4": 0.25, "9": 0.0},
        "policy_mask": [3, 4, 5],
        "value": 1.0,
    }

    _, targets, masks, values = build_training_tensors([sample])

    assert targets[0, 3].item() == pytest.approx(0.75)
    assert targets[0, 4].item() == pytest.approx(0.25)
    assert targets[0].sum().item() == pytest.approx(1.0)
    assert masks[0, 3]
    assert masks[0, 4]
    assert masks[0, 5]
    assert not masks[0, 9]
    assert values[0].item() == pytest.approx(1.0)


def test_legacy_replay_without_policy_mask_keeps_full_action_space():
    sample = {
        "state": [0.0],
        "policy": {"3": 1.0},
        "value": -1.0,
    }

    _, targets, masks, _ = build_training_tensors([sample])

    assert masks.all()
    assert targets[0, 3].item() == pytest.approx(1.0)


def test_policy_target_rejects_probability_outside_search_mask():
    sample = {
        "state": [0.0],
        "policy": {"3": 0.5, "9": 0.5},
        "policy_mask": [3, 4],
        "value": 0.0,
    }

    with pytest.raises(ValueError, match="outside policy_mask"):
        build_training_tensors([sample])


def test_policy_masked_logits_have_no_gradient_for_unsearched_actions():
    logits = torch.zeros((1, ACTION_SIZE), requires_grad=True)
    mask = torch.zeros((1, ACTION_SIZE), dtype=torch.bool)
    mask[0, 3] = True
    mask[0, 4] = True
    targets = torch.zeros((1, ACTION_SIZE))
    targets[0, 3] = 1.0

    log_probs = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=1)
    log_probs = log_probs.masked_fill(~mask, 0.0)
    loss = -(targets * log_probs).sum(dim=1).mean()
    loss.backward()

    assert logits.grad[0, 9].item() == 0.0
    assert logits.grad[0, 3].item() < 0.0
