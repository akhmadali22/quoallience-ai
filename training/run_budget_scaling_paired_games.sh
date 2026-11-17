#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON=python
fi

PAIRS="${PAIRS:-30}"
SEED="${SEED:-20261118}"
DEVICE="${DEVICE:-cpu}"
MAX_PLIES="${MAX_PLIES:-200}"
OUT="data/diagnostics/budget_scaling_paired_games_20261118"
mkdir -p "$OUT"

CANDIDATE="models/checkpoints/exp_001r_vonly_strict.pt"
INCUMBENT="models/checkpoints/neural_v1.pt"

run_matchup() {
    local checkpoint="$1"
    local label="$2"
    local low="$3"
    local high="$4"
    local output="$OUT/${label}_${low}_vs_${high}.json"

    echo "=== checkpoint=$label budgets=$low/$high pairs=$PAIRS seed=$SEED ==="
    "$PYTHON" -m training.evaluate_mcts_scaling "$checkpoint" \
        --low-simulations "$low" \
        --high-simulations "$high" \
        --pairs "$PAIRS" \
        --max-plies "$MAX_PLIES" \
        --seed "$SEED" \
        --device "$DEVICE" \
        --c-puct 1.5 \
        --category-exploration 0.0 \
        --repetition-penalty 0.0 \
        --pawn-progress-bonus 0.0 \
        --disable-progressive-widening \
        --output "$output"
}

# Each matchup uses paired seeds and swaps player seats in the two games.
# Same checkpoint on both sides; only the simulation budget differs.
for spec in "32 128" "128 256" "32 256"; do
    read -r low high <<< "$spec"
    run_matchup "$CANDIDATE" candidate "$low" "$high"
    run_matchup "$INCUMBENT" incumbent "$low" "$high"
done

"$PYTHON" -m training.summarize_budget_scaling_paired_games \
    --input-dir "$OUT" \
    --output "data/diagnostics/budget_scaling_paired_games_20261118_summary.json" \
    --csv "data/diagnostics/budget_scaling_paired_games_20261118_summary.csv"

echo "All paired-game evaluations and summary completed: $OUT"
