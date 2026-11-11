#!/usr/bin/env bash
# Re-evaluate neural policy/value after selecting the search configuration.
# Run: bash training/run_mcts_policy_value_ablation.sh [checkpoint]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CHECKPOINT="${1:-models/checkpoints/neural_v1.pt}"
SIMULATIONS="${MCTS_SIMULATIONS:-128}"
PAIRS="${MCTS_PAIRS:-20}"
MAX_PLIES="${MCTS_MAX_PLIES:-200}"
DEVICE="${MCTS_DEVICE:-cpu}"
SEED="${MCTS_SEED:-20261120}"
INITIAL_WALL_CANDIDATES="${MCTS_INITIAL_WALL_CANDIDATES:-8}"
PROGRESSIVE_WIDENING="${MCTS_PROGRESSIVE_WIDENING:-0}"
OUTPUT="${MCTS_OUTPUT:-data/mcts_ablation/policy_value_category_balanced_final.json}"
PROGRESS_JSONL="${MCTS_PROGRESS_JSONL:-data/mcts_ablation/policy_value_category_balanced_final.jsonl}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

args=(
  "$CHECKPOINT"
  --simulations "$SIMULATIONS"
  --pairs "$PAIRS"
  --max-plies "$MAX_PLIES"
  --device "$DEVICE"
  --seed "$SEED"
  --c-puct 1.5
  --initial-wall-candidates "$INITIAL_WALL_CANDIDATES"
  --output "$OUTPUT"
  --progress-jsonl "$PROGRESS_JSONL"
)
if [[ "$PROGRESSIVE_WIDENING" == "1" ]]; then
  args+=(--progressive-widening)
fi

echo "Policy/value ablation configuration:"
echo "  simulations: $SIMULATIONS"
echo "  pairs/matchup: $PAIRS"
echo "  progressive widening: $PROGRESSIVE_WIDENING"
echo "  category-balanced uniform baseline: enabled"
echo "  experimental PUCT bonuses: disabled"
python training/evaluate_mcts_ablation.py "${args[@]}"
