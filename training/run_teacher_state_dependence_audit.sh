#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DATASET="data/experiments/exp_001/heuristic_decisive_games.jsonl"
OUTPUT_DIR="data/diagnostics/teacher_state_dependence/run_001"

test -x "$PYTHON" || { echo "Python interpreter not found: $PYTHON" >&2; exit 1; }
test -s "$DATASET" || { echo "Training dataset missing/empty: $DATASET" >&2; exit 1; }
test ! -e "$OUTPUT_DIR" || { echo "Output directory already exists: $OUTPUT_DIR" >&2; exit 1; }

"$PYTHON" -m training.audit_teacher_state_dependence \
  --dataset "$DATASET" \
  --short-time-limit 0.08 \
  --long-time-limit 0.32 \
  --max-depth 2 \
  --max-occurrences-per-state 4 \
  --seed 20261120 \
  --output-dir "$OUTPUT_DIR"
