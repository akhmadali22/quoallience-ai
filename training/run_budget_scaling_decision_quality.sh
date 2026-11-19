#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON=python
fi

"$PYTHON" -m training.budget_scaling_decision_quality \
    --prior-q-report data/diagnostics/prior_q_conflict_validation_instrumented_20261116.json \
    --candidate-checkpoint models/checkpoints/exp_001r_vonly_strict.pt \
    --incumbent-checkpoint models/checkpoints/neural_v1.pt \
    --output data/diagnostics/budget_scaling_decision_quality_20261117.json \
    --csv data/diagnostics/budget_scaling_decision_quality_20261117.csv \
    --seed 20261117 \
    --device cpu
