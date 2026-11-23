# Opponent Response Attribution Audit

## Purpose

Determine where positive/negative EXP-001 trajectories diverge after the same exact root state and, where possible, the same root action. This is a read-only diagnostic. It does not modify training, labels, checkpoints, PUCT, or the source dataset.

## Definitions

For every occurrence of a conflicting state, reconstruct the chronological action suffix to game end:

- Offset 0: root player's action from the repeated state.
- Offset 1: opponent's first response, after the root action.
- Offset 2: root player's next action, after that response.

For each exact state, pair every positive terminal target occurrence with every negative target occurrence. Compare:

1. Whether the root actions are identical.
2. Given identical root actions, whether the opponent responses are identical.
3. Given identical root actions and responses, whether the root player's next actions are identical.
4. Opponent response action distributions conditioned on root action and terminal outcome.
5. Move-versus-wall response counts by outcome.

Actions are uniquely reconstructed from legal state transitions. The pinned JSONL contains no direct policy/action field. Any ambiguous or unmatched transition aborts the audit.

## Interpretation limits

This is observational, not a counterfactual intervention. A response associated with wins or losses is not thereby proven to cause them or to be strategically superior. Results should be interpreted with support counts and conditioned on the exact root state and root action. Later decisions and the full game trajectory still affect the terminal outcome.

## Run

```bash
bash training/run_opponent_response_attribution_audit.sh
```

Outputs:

- `data/diagnostics/opponent_response_attribution/run_001/summary.json`
- `data/diagnostics/opponent_response_attribution/run_001/opponent_response_attribution.json`

## Safety

Read-only. No training, optimizer, checkpoint, loss, target, replay, PUCT, or dataset changes.
