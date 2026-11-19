#!/usr/bin/env bash
set -euo pipefail

# EXP-001R paired-seat stage-1 evaluation.
# 10 pairs = 20 games, candidate and incumbent each play both seats.
# Vanilla PUCT only: no progressive widening or optional search extensions.

python -m training.evaluate_candidate \
  models/checkpoints/exp_001r_vonly_strict.pt \
  models/checkpoints/neural_v1.pt \
  --pairs 10 \
  --simulations 32 \
  --max-plies 200 \
  --seed 20261113 \
  --device cpu \
  --c-puct 1.5 \
  --category-exploration 0.0 \
  --repetition-penalty 0.0 \
  --pawn-progress-bonus 0.0 \
  --repetition-window 4 \
  --initial-wall-candidates 0 \
  --min-decisive-pairs 10 \
  --alpha 0.05 \
  --output data/experiments/exp_001r/paired_match_evaluation_stage1.json
