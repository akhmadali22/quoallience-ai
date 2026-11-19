#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="models/checkpoints/exp_001r_vonly_strict.pt"
SOURCE_JSON="data/diagnostics/tactical_action_value_separation/run_001/tactical_action_value_separation.json"
OUTPUT_DIR="data/diagnostics/forced_outcome_value_consistency/run_001"

python -m training.audit_forced_outcome_value_consistency \
  "$CHECKPOINT" \
  --source-json "$SOURCE_JSON" \
  --batch-size 128 \
  --device cpu \
  --threads 1 \
  --output-dir "$OUTPUT_DIR"
