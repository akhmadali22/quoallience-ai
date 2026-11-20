# EXP State-Weighted Dataset and Purged Holdout Audit

## Purpose

Read-only analysis of duplicate-state weighting, per-state target aggregation,
tactical coverage, and exact-state-purged game holdouts for the EXP-001R dataset.

## Safety and provenance

- No training, optimizer, checkpoint writes, replay mutation, or PUCT changes.
- Uses the same source dataset and pinned EXP-001R checkpoint as prior audits.
- Recomputes targets as undiscounted +/-1 from player-to-move perspective.
- Reconstructs every encoded state and validates its round-trip.
- Uses the original winner-stratified 80/10/10 game split and seed.

## A. State-weighted dataset analysis

Reports state frequency distribution, frequency quantiles, top 1/5/10 percent
state record shares, and frequency Gini coefficient.

Compares:
1. Record-weighted loss: every row has equal weight.
2. State-weighted loss preserving row targets: average loss within each state,
   then average equally across states.
3. State-weighted aggregated-target loss: aggregate each state's +/-1 targets
   to a mean, then average one loss per state.

The second metric retains outcome variation within a state. The third measures
the model against the empirical mean outcome. They answer different questions.

## B. Target aggregation

For each exact float32 model-input fingerprint, reports:
- Record/game frequency.
- Positive and negative target counts.
- Mean target, population variance, standard deviation, and outcome entropy.
- Tactical cohort and player identity consistency.
- High-uncertainty conflicting states.

A target conflict across different games with consistent player identity and
cohort is classified as trajectory outcome variation, not automatically as
corrupt labeling. Player/cohort identity mismatches or a conflict confined to
one game are flagged as possible metadata/encoding inconsistencies.

## C. Purged holdout

The exact original game split is preserved. Validation records whose state
fingerprint appears in training are removed. Test records whose fingerprint
appears in training are removed; test is additionally purged against retained
validation fingerprints so validation and test do not share exact inputs.

The report includes before/after counts, removed records and unique states,
post-purge overlap checks, and cohort coverage. Small remaining samples are
reported as-is; the script does not force a target ratio.

## D. Tactical coverage

Reports records, unique states, games, target balance, conflicting state count,
and state-weighted mean target for each tactical cohort in the full dataset and
purged validation/test sets.

## E. Offline baseline comparison

All three weighting/aggregation approaches are evaluated with the frozen
checkpoint. This is diagnostic inference only, not a simulated retraining
comparison and not evidence that a new training objective will improve results.

## Run

```bash
bash training/run_state_weighted_purged_holdout_audit.sh
```

Outputs:
- `data/diagnostics/state_weighted_purged_holdout/run_001/summary.json`
- `data/diagnostics/state_weighted_purged_holdout/run_001/state_weighted_purged_holdout.json`
