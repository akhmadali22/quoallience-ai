def test_policy_loss_components_sum_to_total_cross_entropy():
    import torch
    from quoridor.neural_ai.encoder import ACTION_SIZE, MOVE_ACTIONS
    from training.audit_policy_loss import policy_loss_components

    logits = torch.zeros((1, ACTION_SIZE))
    targets = torch.zeros((1, ACTION_SIZE))
    mask = torch.zeros((1, ACTION_SIZE), dtype=torch.bool)
    targets[0, 0] = 0.25
    targets[0, MOVE_ACTIONS - 1] = 0.25
    targets[0, MOVE_ACTIONS] = 0.5
    mask[0, 0] = True
    mask[0, MOVE_ACTIONS - 1] = True
    mask[0, MOVE_ACTIONS] = True
    total, pawn, wall = policy_loss_components(logits, targets, mask)
    assert torch.allclose(total, pawn + wall)
    assert pawn.item() > 0
    assert wall.item() > 0
