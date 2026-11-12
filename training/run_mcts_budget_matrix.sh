#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-models/checkpoints/neural_v1.pt}"
PAIRS="${MCTS_PAIRS:-10}"
SEED="${MCTS_SEED:-20261112}"
MAX_PLIES="${MCTS_MAX_PLIES:-200}"
DEVICE="${MCTS_DEVICE:-cpu}"
OUT_DIR="${MCTS_OUTPUT_DIR:-data/mcts_budget_matrix}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

run_pair() {
  local low="$1"
  local high="$2"
  local output="$OUT_DIR/eval_${low}_vs_${high}.json"
  echo "=== Vanilla PUCT: ${low} vs ${high} simulations; pairs=$PAIRS seed=$SEED ==="
  python -m training.evaluate_mcts_scaling "$CHECKPOINT" \
    --low-simulations "$low" \
    --high-simulations "$high" \
    --pairs "$PAIRS" \
    --seed "$SEED" \
    --max-plies "$MAX_PLIES" \
    --device "$DEVICE" \
    --category-exploration 0 \
    --repetition-penalty 0 \
    --pawn-progress-bonus 0 \
    --disable-progressive-widening \
    --output "$output"
}

# All pairings share checkpoint, seed set, seat swaps, C-PUCT, and max plies.
run_pair 8 32
run_pair 32 64
run_pair 8 64

echo "Budget matrix reports: $OUT_DIR"
