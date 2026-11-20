#!/usr/bin/env bash
set -euo pipefail
PYTHON="${PYTHON:-.venv/bin/python}"
test -x "$PYTHON" || { echo "Python interpreter not found: $PYTHON" >&2; exit 1; }
python_args=(--dataset data/experiments/exp_001/heuristic_decisive_games.jsonl --output-dir data/diagnostics/trajectory_outcome_attribution/run_001)
"$PYTHON" -m training.audit_trajectory_outcome_attribution "${python_args[@]}"
