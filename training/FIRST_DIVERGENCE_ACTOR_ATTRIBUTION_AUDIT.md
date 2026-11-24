# First Divergence Actor Attribution Audit

## Purpose

Identify which player first makes a different observed decision when two EXP-001 trajectories begin from the same exact encoded state but eventually have opposite terminal outcomes.

This is a read-only diagnostic. It does not modify the source dataset, targets, training code, checkpoints, replay, or PUCT.

## Frozen cohort

The audit pins the EXP-001 dataset SHA-256:

`d9e027c3e4537146dcfa7368baee0ccc01be4fbc067bff01f36fc50f859437b9`

Expected integrity cohort:

- 50 games
- 3,780 records/transitions
- 52 exact-state target-conflict groups
- 1,987 conflict occurrences
- 19,516 positive/negative occurrence pairs

Any mismatch aborts the run.

## Method

1. Reconstruct every action uniquely from the next recorded state; reconstruct the final action from terminal winner and ply.
2. For every positive/negative occurrence pair within each exact root state, compare chronological action suffixes.
3. Find the first offset where actions differ.
4. Verify that both trajectories reach the same exact divergence state after their shared action prefix.
5. Identify the actor from the divergence state's side-to-move, relative to the original root player:
   - `root_player`: the player whose perspective defines the conflicting value target.
   - `opponent`: the other player.
6. Record the positive-trajectory and negative-trajectory actions, classified as `move` or `wall`.
7. Recompute one-ply tactical context at the divergence state using the existing engine-backed `label_position` classifier:
   - `no_threat`
   - `threat_with_defense`
   - `threat_no_defense`
   - `threat_with_immediate_win`
   - strict `forced_loss` flag

Common-prefix length is the number of identical actions before the first different action. It is not the number of plies from the beginning of the game.

## Interpretation limits

- This is observational attribution, not a counterfactual intervention.
- The first actor to differ is not necessarily the actor who caused the eventual win or loss.
- Positive/negative action labels indicate the terminal outcome of the observed trajectory, not intrinsic action quality.
- Tactical labels are one-ply engine classifications, not full strategic evaluations.
- The dataset has no stored action/policy field; all actions are reconstructed from legal transitions.
- Repeated pairs are not independent samples: many pairs reuse the same occurrences and states. Do not interpret pair counts as independent statistical trials.

## Run

```bash
bash training/run_first_divergence_actor_attribution_audit.sh
```

Outputs:

- `data/diagnostics/first_divergence_actor_attribution/run_001/summary.json`
- `data/diagnostics/first_divergence_actor_attribution/run_001/first_divergence_actor_attribution.json`

## Safety

Read-only. No training, optimizer, target, loss, checkpoint, replay, PUCT, or dataset changes.
