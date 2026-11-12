def test_policy_loss_components_sum_to_total_cross_entropy():
    import torch
    from training.audit_policy_loss import policy_loss_components

    logits = torch.tensor([[0.0, 0.0, 1.0, -1.0]])
    targets = torch.tensor([[0.25, 0.25, 0.5, 0.0]])
    mask = torch.tensor([[True, True, True, False]])
    total, pawn, wall = policy_loss_components(logits, targets, mask)
    assert torch.allclose(total, pawn + wall)
    assert pawn.item() > 0
    assert wall.item() > 0
