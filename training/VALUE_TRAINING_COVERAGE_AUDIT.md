# Value Training Coverage Audit

## Purpose

Determine whether EXP-001R's value-head failure on forced tactical outcomes is
explained by missing tactical examples, contradictory final-game labels, or
poor checkpoint predictions despite adequate data coverage.

## Scope

- Read-only dataset and inference audit.
- No optimizer, training, MCTS/PUCT, checkpoint writes, or replay mutation.
- Audits every row in the exact EXP-001R training dataset.
- Reconstructs the encoded state and validates an encode/decode round trip.
- Reuses the exact game-level split algorithm and seed from strict training.
- Uses engine transitions to classify immediate threats and defensive actions.
- Evaluates the pinned EXP-001R checkpoint in player-to-move perspective.

## Dataset provenance

The strict EXP-001R training report identifies:

- Dataset: `data/experiments/exp_001/heuristic_decisive_games.jsonl`
- 3,780 records across 50 decisive heuristic games.
- Generator: `training/generate_heuristic_games.py`
- Heuristic generator configuration: depth 2, 0.08-second move limit,
  0.99 per-ply discount.
- Strict training discards the stored discounted value and recomputes an
  undiscounted `+1/-1` target from `winner` and `player`.
- Split: 80/10/10 by game, stratified by winner, seed `20261113003`.

## Tactical cohorts

- `no_threat`: opponent has no immediate winning pawn move.
- `threat_with_immediate_win`: opponent has an immediate winning threat, but
  the side to move can win immediately itself.
- `threat_with_defense`: opponent has an immediate winning threat and at least
  one nonterminal legal action removes all immediate winning replies.
- `threat_no_defense`: opponent has an immediate winning threat, no immediate
  root win, and no defensive action.
- `forced_loss`: strict subset where every legal root action is nonterminal
  and leaves the opponent at least one immediate winning reply.

## Main diagnostic

For every cohort and split, report:

- Position/game counts.
- Final-game training target distribution.
- Checkpoint prediction distribution.
- MSE, MAE, and sign accuracy against the strict training target.
- Forced-loss counts, target contradictions, and prediction signs.

A forced-loss state with a positive final-game target is especially important:
the one-ply engine proof says the opponent can win on its next move, while the
recorded game outcome says the side to move eventually won. That is evidence
that the heuristic trajectory failed to take an available win, producing a
misleading value target for that tactical state.

## Run

```bash
bash training/run_value_training_coverage_audit.sh
```

Outputs:

- `data/diagnostics/value_training_coverage/run_001/summary.json`
- `data/diagnostics/value_training_coverage/run_001/value_training_coverage.json`

The full JSON includes one compact row per source position. The report records
both dataset and checkpoint SHA-256 hashes.
