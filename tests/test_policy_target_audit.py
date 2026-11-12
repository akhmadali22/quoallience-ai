from training.audit_policy_targets import audit_samples


def test_policy_target_audit_reports_pawn_wall_mass_and_mask_counts():
    report = audit_samples([
        {
            "policy": {"0": 0.25, "80": 0.25, "81": 0.5},
            "policy_mask": [0, 80, 81, 82],
        },
        {
            "policy": {"81": 1.0},
            "policy_mask": [0, 81],
        },
    ])
    assert report["samples"] == 2
    assert report["target_mass"]["pawn"]["mean"] == 0.25
    assert report["target_mass"]["wall"]["mean"] == 0.75
    assert report["samples_with_zero_category_target"]["pawn"] == 1
    assert report["policy_mask_actions"]["pawn"]["mean"] == 1.5


def test_policy_target_audit_rejects_mass_outside_search_mask():
    import pytest

    with pytest.raises(ValueError, match="outside policy_mask"):
        audit_samples([{
            "policy": {"0": 0.5, "81": 0.5},
            "policy_mask": [0],
        }])
