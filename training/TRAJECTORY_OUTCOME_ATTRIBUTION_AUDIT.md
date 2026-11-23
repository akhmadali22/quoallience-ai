# Trajectory Outcome Attribution Audit

## Purpose

Explain how identical EXP-001 encoded states receive opposite terminal targets by tracing the observed action trajectories after each occurrence. This is a read-only audit; it does not alter training, loss, checkpoints, data, replay, or PUCT.

## Dataset and action reconstruction

The pinned EXP-001 JSONL artifact contains state and game metadata, but does not contain `policy`, `policy_mask`, or an explicit action field. Therefore, this audit does **not** claim to read the original stored policy action.

Instead, it reconstructs each observed action from the recorded trajectory:

- For every non-final record, enumerate legal actions and apply them; the action is accepted only if exactly one legal action produces the next recorded state.
- For the final record, enumerate legal actions and accept only the unique action that produces the recorded winner at the recorded terminal ply.
- Abort if a transition has zero matching actions or more than one matching action.
- Verify contiguous plies, consistent game metadata, player-to-move, and terminal winner/ply.
- Pin the exact EXP-001 dataset SHA-256.

This makes the inferred action sequence verifiable against the observed states, but it is a reconstruction from transitions, not a direct record of the teacher's policy output.

## Analysis

- For each of the 52 exact-state conflict groups, pair every +1 occurrence with every -1 occurrence.
- Compare root actions, action-conditioned outcome counts, common action-sequence prefixes, first divergence, and sequence overlap at horizons 1, 2, 4, 8, and 16 plies (plus full suffix pairwise prefix lengths).

## Interpretation

This is **observational trajectory attribution**, not a counterfactual experiment. It can identify:
- whether opposite outcomes begin with the same or different reconstructed actions;
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
