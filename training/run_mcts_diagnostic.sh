#!/usr/bin/env bash
# Reproducible MCTS diagnostic smoke run.
# Run from any directory: bash training/run_mcts_diagnostic.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT="${1:-models/checkpoints/neural_v1.pt}"
SIMULATIONS="${MCTS_SIMULATIONS:-16}"
PAIRS="${MCTS_PAIRS:-1}"
MAX_PLIES="${MCTS_MAX_PLIES:-120}"
DEVICE="${MCTS_DEVICE:-cpu}"
SEED="${MCTS_SEED:-20261120}"
C_PUCT="${MCTS_C_PUCT:-1.5}"
INITIAL_WALL_CANDIDATES="${MCTS_INITIAL_WALL_CANDIDATES:-8}"

OUTPUT="${MCTS_OUTPUT:-data/mcts_ablation/diagnostic_120.json}"
PROGRESS_JSONL="${MCTS_PROGRESS_JSONL:-data/mcts_ablation/diagnostic_120.jsonl}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  echo "Pass the checkpoint path as the first argument, or set it at the default location." >&2
  exit 1
fi

echo "MCTS diagnostic configuration:"
echo "  checkpoint: $CHECKPOINT"
echo "  simulations: $SIMULATIONS"
echo "  pairs/matchup: $PAIRS (two games per pair, seats swapped)"
echo "  max plies: $MAX_PLIES"
echo "  device: $DEVICE"
echo "  seed: $SEED"
echo "  c_puct: $C_PUCT"
echo "  progressive widening: disabled"
echo "  output: $OUTPUT"
echo "  progress: $PROGRESS_JSONL"
echo

python training/evaluate_mcts_ablation.py "$CHECKPOINT" \
  --simulations "$SIMULATIONS" \
  --pairs "$PAIRS" \
  --max-plies "$MAX_PLIES" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --c-puct "$C_PUCT" \
  --initial-wall-candidates "$INITIAL_WALL_CANDIDATES" \
  --output "$OUTPUT" \
  --progress-jsonl "$PROGRESS_JSONL"
