#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON=python; fi
CHECKPOINT="${CHECKPOINT:-models/checkpoints/exp_001r_vonly_strict.pt}"
PAIRS="${PAIRS:-4}"
SEED="${SEED:-20261119}"
DEVICE="${DEVICE:-cpu}"
MAX_PLIES="${MAX_PLIES:-200}"
HEURISTIC_TIME_LIMIT="${HEURISTIC_TIME_LIMIT:-0.5}"
HEURISTIC_MAX_DEPTH="${HEURISTIC_MAX_DEPTH:-5}"
OUT="${OUT:-data/diagnostics/mcts_vs_advanced_heuristic_20261119}"
mkdir -p "$OUT"
for budget in 32 128 256; do
  "$PYTHON" -m training.evaluate_mcts_vs_advanced_heuristic "$CHECKPOINT" \
    --budget "$budget" --pairs "$PAIRS" --seed "$SEED" \
    --max-plies "$MAX_PLIES" --device "$DEVICE" \
    --heuristic-time-limit "$HEURISTIC_TIME_LIMIT" \
    --heuristic-max-depth "$HEURISTIC_MAX_DEPTH" \
    --output "$OUT/candidate_budget_${budget}.json"
done
echo "Pilot reports written to $OUT"
