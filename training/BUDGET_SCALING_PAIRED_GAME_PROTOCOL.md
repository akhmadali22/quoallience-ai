# EXP: Budget Scaling Paired-Game Outcome Validation

Status: protocol prepared; results not yet run. This is a follow-up to
`budget_scaling_decision_quality_20261117`. The earlier one-ply shortest-path
report remains a diagnostic artifact. It must not trigger MCTS tuning or
checkpoint promotion.

## Objective

Test whether increasing MCTS simulations changes actual game outcomes, rather
than only changing root-action selection or a one-ply structural proxy.

## Hypothesis

At a fixed checkpoint and fixed vanilla-PUCT configuration, the higher-budget
agent scores better in paired games than the lower-budget agent. The key
comparisons are 32 vs 128 and 128 vs 256. The 32 vs 256 comparison is a
secondary end-to-end contrast.

## Design

- Checkpoints: `exp_001r_vonly_strict.pt` and `neural_v1.pt`.
- Comparisons: 32 vs 128, 128 vs 256, and 32 vs 256 simulations.
- 30 paired seeds per checkpoint/comparison (60 games per comparison).
- Each seed is played twice, with the low-budget agent occupying P0 and P1
  once each. This controls seat/first-player advantage.
- Both agents in a game use the same checkpoint. Only simulation budget differs.
- Each paired game starts from the normal initial Quoridor state.
- Maximum 200 plies; a game reaching the limit is scored as a draw.
- Selection temperature 0; deterministic visit-count selection.
- c_puct=1.5; root noise off; progressive widening off.
- Category exploration, repetition penalty, pawn-progress bonus, wall-distance
  weight, and wall-route weight are all zero.
- No checkpoint, network, heuristic, or MCTS policy changes.

Total planned workload: 2 checkpoints x 3 comparisons x 30 pairs x 2 games =
360 games.

## Analysis plan

For each seed pair, aggregate the two seat-swapped games into pair points:
win=1, draw=0.5, loss=0 for the low-budget agent. Report pair wins/losses/ties,
game wins/losses/draws, score rate, plies, and agent search-cost telemetry.

The high-budget agent's paired advantage is assessed from pair-level outcomes,
not by treating the two seat-swapped games as independent observations.
Use an exact one-sided sign test on decisive pairs; report ties and the number
of decisive pairs explicitly. Apply Holm correction across the four primary
tests (32 vs 128 and 128 vs 256, each on candidate and incumbent checkpoints).
Treat 32 vs 256 as secondary/descriptive. Report confidence intervals and raw
counts; do not infer superiority from a small score-rate difference alone.

## Decision rules

- This experiment is evaluation only. Do not change MCTS parameters or
  promote a checkpoint based on the previous diagnostic.
- A statistically supported game-outcome advantage is evidence for that
  budget/checkpoint pairing, not proof that the one-ply metric is valid.
- If the result is inconclusive, increase the number of paired seeds using a
  separately recorded extension; do not silently pool altered protocols.
- If budget 256 does not outperform 128, do not assume additional simulations
  are useful merely because search cost increases.
- Inspect draw rate, game length, repetition/cycle telemetry, and search cost
  alongside outcome.

## Run

```bash
git pull origin experiment/progressive-widening-mcts
python -m py_compile training/evaluate_mcts_scaling.py
bash training/run_budget_scaling_paired_games.sh
```

The runner accepts `PAIRS`, `SEED`, `DEVICE`, `MAX_PLIES`, and `PYTHON`
environment overrides. Keep defaults for the registered run.
