# Tactical Action Preference and Value Separation Audit

## Objective

Measure five one-ply action classes:

1. `immediate_win`: root action wins immediately.
2. `prevents_immediate_loss`: opponent has an immediate winning reply before the action, but none after.
3. `allows_immediate_loss`: opponent has an immediate winning reply both before and after.
4. `creates_immediate_loss`: no immediate opponent win before the action, but one exists after.
5. `no_immediate_terminal_consequence`: child is nonterminal and opponent has no immediate winning reply.

The extra `creates_immediate_loss` class prevents newly introduced blunders from being hidden inside the quiet baseline.

## Safety boundaries

- Inference only: no training, optimizer, replay buffer, or checkpoint writes.
- No MCTS or PUCT; no architecture, checkpoint, or search changes.
- Reuses the frozen tactical positions and seeded `threat_no_defense` / `no_threat` controls.
- Uses the engine's legal-action and transition implementation, not a second implementation of Quoridor rules.

## Metrics

Per position: root value, full legal-action policy entropy and normalized entropy, pawn-vs-wall policy mass, and every action's class, prior, rank, root-perspective child value, and value rank. Per class/cohort: full distributions (count, mean, median, population standard deviation, min, p10, p25, p75, p90, max), policy-mass distributions, and within-position pairwise AUC-style discrimination for defense-vs-allow-loss and immediate-win-vs-quiet.

Pairwise comparisons are restricted to actions from the same position. Action rows are clustered within positions and are not independent game samples.

## Run

```bash
python -m training.audit_tactical_action_value_separation \
  models/checkpoints/exp_001r_vonly_strict.pt \
  --source-json data/diagnostics/tactical_policy_value_attribution/run_001/tactical_policy_value_attribution.json \
  --control-positions 30 --seed 20261121 --max-games 5000 --max-plies 160 \
  --batch-size 128 --device cpu --threads 1 \
  --output-dir data/diagnostics/tactical_action_value_separation/run_001
```

Run a smoke test first with `--control-positions 3` and a fresh output directory.

## Outputs

- `summary.json`: hashes, settings, and cohort-level distribution summaries.
- `tactical_action_value_separation.json`: complete position/action-level results.

## Interpretation

Poor prior ranking or within-position prior discrimination indicates policy failure. Poor child-value ranking/discrimination indicates value-head failure. These can differ. `threat_no_defense` contains no available defensive action and is not a policy failure; `no_threat` has no defensive label. Terminal wins use exact +1, not a learned value prediction. Random-playout controls are diagnostic, not estimates of competitive-game frequency. Do not change PUCT, architecture, losses, or checkpoint based on this audit alone.
