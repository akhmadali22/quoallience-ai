#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CHECKPOINT="${1:-models/checkpoints/exp_001r_vonly_strict.pt}"
OUTPUT="${2:-data/diagnostics/exp001r_synthetic_prior_puct_audit.json}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

python -m training.audit_synthetic_prior_puct \
  "$CHECKPOINT" \
  --seed-start 20261119 \
  --seed-count 32 \
  --max-game-plies 64 \
  --continuation-plies 8 16 24 32 \
  --positions-per-tactical-category 2 \
  --budgets 32 128 \
  --c-puct 1.5 \
  --device cpu \
  --threads 1 \
  --output "$OUTPUT"
