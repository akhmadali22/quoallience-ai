# Trajectory Outcome Attribution Audit

## Purpose

Explain how identical EXP-001 encoded states receive opposite terminal targets by tracing the **actual recorded action trajectories** after each occurrence. This is a read-only audit; it does not alter training, loss, checkpoints, data, replay, or PUCT.

## Method

- Pin the exact EXP-001 dataset SHA-256.
- Reconstruct every position and read the one-hot action stored in its original `policy` field.
- Verify every action is legal, every action produces the next recorded state, and the final action produces the recorded winner and terminal ply.
- For each of the 52 exact-state conflict groups, pair every +1 occurrence with every -1 occurrence.
- Compare root actions, action-conditioned outcome counts, and common action-sequence prefixes at horizons 1, 2, 4, 8, and 16 plies (plus full suffix pairwise prefix lengths).

## Interpretation

This is **observational trajectory attribution**, not a counterfactual experiment. It can identify:
- whether opposite outcomes begin with the same or different actions;
- which later ply is the first observed divergence;
- whether particular root actions appear under both outcomes, with their sample support.

It cannot prove that a specific action caused a win or loss. The opponent's response, later decisions, and unobserved history remain part of the trajectory. Do not infer action quality from small-support action groups.

## Run

```bash
bash training/run_trajectory_outcome_attribution_audit.sh
```

Outputs:
- `data/diagnostics/trajectory_outcome_attribution/run_001/summary.json`
- `data/diagnostics/trajectory_outcome_attribution/run_001/trajectory_outcome_attribution.json`

## Safety

Read-only. No training, optimizer, checkpoint, loss, dataset, replay, or PUCT changes.
