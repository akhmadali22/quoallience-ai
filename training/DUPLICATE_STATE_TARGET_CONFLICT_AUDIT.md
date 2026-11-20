# Duplicate-State and Target-Conflict Audit

## Purpose

Audit exact duplicate model inputs in the EXP-001R heuristic dataset, quantify
contradictory final-game targets for identical inputs, and compare the original
game-level split with a state-fingerprint-grouped split.

## Scope and invariants

- Read-only dataset and checkpoint inference.
- No optimizer, training, checkpoint writes, replay mutation, or MCTS/PUCT changes.
- Uses the same dataset and pinned EXP-001R checkpoint as the preceding audit.
- Fingerprints the exact flattened float32 representation consumed by the
  neural network, not JSON formatting. Thus integer/float JSON spelling does
  not create false distinctions.
- Reconstructs and validates every state using the preceding coverage audit.
- Uses the original split function and seed to measure current leakage.
- Assigns each unique state fingerprint to exactly one new partition.

## Important split limitation

A strict game-disjoint split and a strict exact-state-disjoint split cannot
both be achieved if the same encoded state appears in multiple games. In
particular, the initial Quoridor position is shared by all games. Keeping each
game intact would connect every game through that shared state, collapsing the
entire dataset into one connected component.

Therefore, the audit's alternative split groups by exact state fingerprint:
identical network inputs cannot cross train/validation/test, but positions from
the same game may appear in different partitions. The report explicitly counts
games spanning partitions. This is a controlled way to measure exact-state
leakage, not a claim of fully independent game-level generalization.

## Metrics

The report includes:

- Total records, games, unique float32 state fingerprints, and duplicate rows.
- Per-state frequency, game IDs, tactical cohorts, target counts, and original
  split membership.
- Number of state groups with both +1 and -1 targets.
- Conflicts broken down by tactical cohort.
- Original game-split metrics and exact-state leakage.
- State-grouped 80/10/10 metrics, with game overlap disclosed.
- State-level metrics using mean target and majority target, so repeated
  records do not silently receive extra weight.
- Checkpoint predictions and per-record targets for independent inspection.

## Run

```bash
bash training/run_duplicate_state_target_conflict_audit.sh
```

Outputs:

- `data/diagnostics/duplicate_state_target_conflict/run_001/summary.json`
- `data/diagnostics/duplicate_state_target_conflict/run_001/duplicate_state_target_conflict.json`

The full report includes one record per dataset row and one row per unique
state. The checkpoint SHA is pinned; a mismatch stops the audit.
