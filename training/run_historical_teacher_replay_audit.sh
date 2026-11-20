#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
DATASET="data/experiments/exp_001/heuristic_decisive_games.jsonl"
ROOT="data/diagnostics/historical_teacher_replay/run_001"
RAW_DIR="$ROOT/raw"
FINAL_DIR="$ROOT"

test -x "$PYTHON" || { echo "Python interpreter not found: $PYTHON" >&2; exit 1; }
test -s "$DATASET" || { echo "Dataset missing/empty: $DATASET" >&2; exit 1; }
test ! -e "$ROOT" || { echo "Output directory already exists: $ROOT" >&2; exit 1; }

mkdir -p "$RAW_DIR"

"$PYTHON" -m training.audit_teacher_state_dependence \
  --dataset "$DATASET" \
  --short-time-limit 0.08 \
  --long-time-limit 0.32 \
  --max-depth 2 \
  --max-occurrences-per-state 0 \
  --output-dir "$RAW_DIR"

"$PYTHON" -m training.finalize_historical_teacher_replay \
  --raw-dir "$RAW_DIR" \
  --output-dir "$FINAL_DIR"

echo
echo "Historical Teacher Replay Audit complete."
echo "Summary: $FINAL_DIR/summary.json"
echo "Details: $FINAL_DIR/historical_teacher_replay.json"
