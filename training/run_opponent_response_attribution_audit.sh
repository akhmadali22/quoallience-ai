#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-.venv/bin/python}"
test -x "$PYTHON" || { echo "Python interpreter not found: $PYTHON" >&2; exit 1; }
"$PYTHON" -m training.audit_opponent_response_attribution \
  --dataset data/experiments/exp_001/heuristic_decisive_games.jsonl \
  --output-dir data/diagnostics/opponent_response_attribution/run_001
