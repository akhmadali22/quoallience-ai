#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DATASET="data/experiments/exp_001/heuristic_decisive_games.jsonl"
CHECKPOINT="models/checkpoints/exp_001r_vonly_strict.pt"
OUTPUT_DIR="data/diagnostics/state_weighted_purged_holdout/run_001"

test -x "$PYTHON" || { echo "Python interpreter not found: $PYTHON" >&2; exit 1; }
test -s "$DATASET" || { echo "Training dataset missing/empty: $DATASET" >&2; exit 1; }
test -f "$CHECKPOINT" || { echo "Checkpoint missing: $CHECKPOINT" >&2; exit 1; }
test ! -e "$OUTPUT_DIR" || { echo "Output directory already exists: $OUTPUT_DIR" >&2; exit 1; }

"$PYTHON" -m training.audit_state_weighted_purged_holdout \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --batch-size 128 --device cpu --threads 1 \
  --seed 20261113003 \
  --output-dir "$OUTPUT_DIR"
