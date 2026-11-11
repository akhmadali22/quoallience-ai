#!/usr/bin/env bash
# Compare vanilla-PUCT search budgets with progressive widening off and on.
# Run: bash training/run_mcts_search_matrix.sh [checkpoint]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CHECKPOINT="${1:-models/checkpoints/neural_v1.pt}"
PAIRS="${MCTS_PAIRS:-10}"
MAX_PLIES="${MCTS_MAX_PLIES:-200}"
DEVICE="${MCTS_DEVICE:-cpu}"
SEED="${MCTS_SEED:-20261120}"
OUTPUT_DIR="${MCTS_OUTPUT_DIR:-data/mcts_search_matrix/seed_${SEED}}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
for widening in disabled enabled; do
  for budgets in "16 64" "64 128"; do
    read -r LOW HIGH <<< "$budgets"
    OUTPUT="$OUTPUT_DIR/${widening}_pw_${LOW}_vs_${HIGH}.json"
    args=(
      "$CHECKPOINT"
      --low-simulations "$LOW"
      --high-simulations "$HIGH"
      --pairs "$PAIRS"
      --max-plies "$MAX_PLIES"
      --device "$DEVICE"
      --seed "$SEED"
      --category-exploration 0
      --repetition-penalty 0
      --pawn-progress-bonus 0
      --output "$OUTPUT"
    )
    if [[ "$widening" == "disabled" ]]; then
      args+=(--disable-progressive-widening)
    fi
    echo
    echo "=== PW=$widening, simulations=$LOW vs $HIGH, pairs=$PAIRS ==="
    python training/evaluate_mcts_scaling.py "${args[@]}"
  done
done

echo
echo "Search matrix reports: $OUTPUT_DIR"
