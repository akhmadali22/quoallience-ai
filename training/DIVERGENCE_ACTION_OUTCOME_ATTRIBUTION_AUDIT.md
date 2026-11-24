# Divergence Action Outcome Attribution Audit

## Purpose

Compare the two different observed actions at each exact first-divergence state from EXP-001 conflicting-outcome trajectory pairs. The goal is to test whether the action associated with the positive terminal trajectory is preferred by a deterministic reference evaluator over the action associated with the negative trajectory.

This is a read-only diagnostic. It does not modify the source dataset, targets, training code, checkpoints, replay, or PUCT.

## Frozen cohort and integrity

The audit pins the EXP-001 dataset SHA-256:

`d9e027c3e4537146dcfa7368baee0ccc01be4fbc067bff01f36fc50f859437b9`

Expected source cohort:

- 50 games
- 3,780 records/transitions
- 52 exact-state target-conflict groups
- 1,987 conflict occurrences
- 19,516 positive/negative occurrence pairs

Every trajectory action is reconstructed uniquely from its recorded state transition, including the final terminal action. The first-divergence state is recovered from the identical common prefix. Any frozen-cohort mismatch aborts the audit.

## Evaluation method

For each unique divergence state:

1. Collect the distinct positive-trajectory and negative-trajectory actions actually observed at that state.
2. Apply each distinct action once.
3. Search the resulting position with deterministic alpha-beta minimax to a fixed depth of **2 additional plies after the forced observed action**.
4. Run minimax from the divergence actor's perspective, then sign-normalize the result to the original root state's player-to-move perspective. This keeps the evaluator's root-turn and opponent-reply candidate logic aligned with the actual side to move.
5. Use a fresh evaluator instance per divergence state. Its actual-game history is empty.
6. Use the existing deterministic path-based wall candidate generator, capped at 8 wall candidates per node.
7. Abort if the one-hour safety deadline is reached; do not emit mixed-depth results.

The comparison uses normalized root-perspective scores:

`positive_action_score_root_perspective - negative_action_score_root_perspective`

- Positive: the evaluator prefers the positive-trajectory action for the original root player.
- Negative: the evaluator prefers the negative-trajectory action for the original root player.
- Zero within numeric tolerance: evaluator tie.

Scores are reused across all trajectory pairs that share the same divergence state and observed action.

## Critical limitations

- This is a heuristic reference evaluator, **not** an exact game-theoretic solver or ground truth.
- It shares `QuoridorAI`'s static evaluation and candidate-wall generator with the historical teacher code. Empty history and fixed-depth search remove two important sources of replay variation, but do not make the evaluator fully independent.
- Wall candidates are pruned; the search is not exhaustive over all legal walls.
- Only actions observed in the historical trajectories are compared. This does not rank every legal action.
- The search horizon is shallow. A preferred action may lose its advantage beyond the horizon.
- The 19,516 pair rows are combinatorial and reuse occurrences. Report unique divergence-state/action-pair counts alongside pair-weighted counts; do not treat pairs as independent statistical trials.
- A positive action scoring higher means the evaluator prefers it for the original root player. When the divergence actor is the opponent, this can indicate that the positive-outcome trajectory contains an opponent mistake. It does not prove causality.

## Run

```bash
bash training/run_divergence_action_outcome_attribution_audit.sh
```

Outputs:

- `data/diagnostics/divergence_action_outcome_attribution/run_001/summary.json`
- `data/diagnostics/divergence_action_outcome_attribution/run_001/divergence_action_outcome_attribution.json`

## Safety

Read-only. No training, optimizer, target, loss, checkpoint, replay, PUCT, or source dataset changes.
