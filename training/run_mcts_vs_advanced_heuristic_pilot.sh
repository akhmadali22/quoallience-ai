#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON=python; fi
CANDIDATE="${CANDIDATE:-models/checkpoints/exp_001r_vonly_strict.pt}"
INCUMBENT="${INCUMBENT:-models/checkpoints/neural_v1.pt}"
PAIRS="${PAIRS:-4}"
SEED="${SEED:-20261119}"
DEVICE="${DEVICE:-cpu}"
MAX_PLIES="${MAX_PLIES:-200}"
HEURISTIC_TIME_LIMIT="${HEURISTIC_TIME_LIMIT:-0.5}"
HEURISTIC_MAX_DEPTH="${HEURISTIC_MAX_DEPTH:-5}"
OUT="${OUT:-data/diagnostics/mcts_vs_advanced_heuristic_20261119_pilot}"
mkdir -p "$OUT"

run_matchup() {
  local checkpoint="$1" label="$2" budget="$3"
  "$PYTHON" -m training.evaluate_mcts_vs_advanced_heuristic "$checkpoint" \
    --label "$label" --budget "$budget" --pairs "$PAIRS" --seed "$SEED" \
    --max-plies "$MAX_PLIES" --device "$DEVICE" \
    --heuristic-time-limit "$HEURISTIC_TIME_LIMIT" \
    --heuristic-max-depth "$HEURISTIC_MAX_DEPTH" \
    --output "$OUT/${label}_budget_${budget}.json"
}
run_matchup "$CANDIDATE" candidate 32
run_matchup "$CANDIDATE" candidate 128
run_matchup "$CANDIDATE" candidate 256
run_matchup "$INCUMBENT" incumbent 128
echo "Pilot reports written to $OUT"
