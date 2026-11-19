#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m training.audit_exp_001r_q \
  --dataset data/experiments/exp_001/heuristic_decisive_games.jsonl \
  --training-report data/experiments/exp_001r/training_report.json \
  --incumbent models/checkpoints/neural_v1.pt \
  --candidate models/checkpoints/exp_001r_vonly_strict.pt \
  --output data/experiments/exp_001r/value_head_action_audit.json \
  --simulations 32 \
  --c-puct 1.5 \
  --max-positions 30 \
  --device cpu
