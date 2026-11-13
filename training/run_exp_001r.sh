#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
INCUMBENT="models/checkpoints/neural_v1.pt"
DATASET="data/experiments/exp_001/heuristic_decisive_games.jsonl"
OUT_DIR="data/experiments/exp_001r"
CANDIDATE="models/checkpoints/exp_001r_vonly_strict.pt"
REPORT="$OUT_DIR/training_report.json"

test -x "$PYTHON" || { echo "Python interpreter not found: $PYTHON" >&2; exit 1; }
test -f "$INCUMBENT" || { echo "Incumbent checkpoint missing: $INCUMBENT" >&2; exit 1; }
test -s "$DATASET" || { echo "Heuristic dataset missing/empty: $DATASET" >&2; exit 1; }
test ! -e "$CANDIDATE" || { echo "Candidate already exists; refusing to overwrite: $CANDIDATE" >&2; exit 1; }

mkdir -p "$OUT_DIR"
"$PYTHON" -m training.train_value_only_strict "$DATASET" \
  --checkpoint "$INCUMBENT" \
  --output "$CANDIDATE" \
  --report "$REPORT" \
  --epochs 10 --batch-size 64 --learning-rate 1e-4 \
  --weight-decay 1e-4 --seed 20261113003 --device cpu

"$PYTHON" -m training.evaluate_value_only_root \
  --dataset "$DATASET" \
  --training-report "$REPORT" \
  --incumbent "$INCUMBENT" \
  --candidate "$CANDIDATE" \
  --output "$OUT_DIR/root_puct_comparison.json" \
  --simulations 32 --c-puct 1.5 --max-positions 30 --device cpu

echo "EXP-001R complete. No paired match evaluation was run."
