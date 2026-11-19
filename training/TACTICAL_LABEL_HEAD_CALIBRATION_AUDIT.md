# Tactical Label and Head Calibration Audit

## Purpose

Validate the one-ply defensive labels from
`tactical_policy_value_attribution/run_001` independently, then compare the
same fixed checkpoint's policy/value outputs across three cohorts:

1. `frozen_threat_with_defense`: all frozen positions from the prior 80-position run.
2. `threat_no_defense`: opponent has an immediate winning pawn reply, but no legal
   root action removes every such reply.
3. `no_threat`: opponent has no immediate winning pawn reply.

This separates "the network missed an available defense" from "there was no
one-ply defense to select". The no-threat cohort is a control for false tactical
signals and baseline policy/value behavior.

## Safety boundaries

- Inference only. No optimizer, training, replay buffer, or checkpoint writes.
- No MCTS or PUCT.
- Does not change model architecture, checkpoint, search, or training data.
- The frozen source positions are reconstructed from their serialized states.
- The label verifier independently enumerates the opponent's legal pawn moves
  after each root action and checks whether any transition ends the game for
  the opponent. It does not trust the stored `is_defensive` flag.

The engine's legal-action generator and transition rules remain the source of
game legality. Thus this is an independent tactical-label recomputation, not a
second implementation of all Quoridor rules.

## Run

From repository root:

```bash
python -m training.audit_tactical_label_head_calibration \
  models/checkpoints/exp_001r_vonly_strict.pt \
  --source-json data/diagnostics/tactical_policy_value_attribution/run_001/tactical_policy_value_attribution.json \
  --control-positions 30 \
  --seed 20261120 \
  --max-games 5000 \
  --max-plies 160 \
  --batch-size 128 \
  --device cpu \
  --threads 1 \
  --output-dir data/diagnostics/tactical_label_head_calibration/run_001
```

Start with a smaller smoke run using `--control-positions 3` and a fresh
output directory. The output directory must be empty or absent.

## Outputs

- `summary.json`: checkpoint/source hashes, label verification counts, and
  cohort-level summaries.
- `tactical_label_head_calibration.json`: per-position and per-action details,
  including label mismatches and policy/value ranks.

## Interpretation

First gate: `label_validation.all_labels_match` must be true and
`mismatched_labels` must be zero. If not, stop interpretation of the head
metrics and inspect mismatched states/actions.

For `frozen_threat_with_defense`, compare defensive top-1/top-5 rates, best
defensive ranks, and defensive policy mass. The no-defense cohort must not be
counted as a policy failure because there is no move satisfying the one-ply
safety definition. The no-threat cohort is descriptive only; it has no
defensive-action label.

The sampled control cohorts come from seeded legal random playouts, not
competitive self-play. Their metrics are diagnostic and should not be treated
as estimates of in-match frequency. Position-level results matter; individual
legal action rows within one position are not independent samples.

The frozen cohort is a fixed audit set, not a holdout for model selection.
Do not tune or train against it and then report it as an unbiased evaluation.
