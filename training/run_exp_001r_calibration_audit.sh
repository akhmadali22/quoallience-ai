#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DATASET="data/experiments/exp_001/heuristic_decisive_games.jsonl"
OUT_DIR="data/experiments/exp_001r"
INCUMBENT="models/checkpoints/neural_v1.pt"
CANDIDATE="models/checkpoints/exp_001r_vonly_strict.pt"
TRAINING_REPORT="$OUT_DIR/training_report.json"

test -x "$PYTHON" || { echo "Python interpreter not found: $PYTHON" >&2; exit 1; }
test -s "$DATASET" || { echo "Dataset missing/empty: $DATASET" >&2; exit 1; }
test -s "$TRAINING_REPORT" || { echo "EXP-001R training report missing: $TRAINING_REPORT" >&2; exit 1; }
test -f "$INCUMBENT" || { echo "Incumbent checkpoint missing: $INCUMBENT" >&2; exit 1; }
test -f "$CANDIDATE" || { echo "Candidate checkpoint missing: $CANDIDATE" >&2; exit 1; }

"$PYTHON" -m training.audit_value_calibration \
  --dataset "$DATASET" \
  --training-report "$TRAINING_REPORT" \
  --incumbent "$INCUMBENT" \
  --candidate "$CANDIDATE" \
  --output "$OUT_DIR/value_calibration_audit.json" \
  --device cpu --batch-size 128 --bootstrap-samples 10000 --seed 20261113004

echo "EXP-001R paired value calibration audit complete."
