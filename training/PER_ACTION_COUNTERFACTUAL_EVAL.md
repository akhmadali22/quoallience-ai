# Per-Action Counterfactual Evaluation

This is an evaluation-only experiment. It does not train or modify a checkpoint,
and it does not change `quoridor/neural_ai/mcts.py` or the PUCT formula.

## Run

From the repository root:

```bash
python -m training.evaluate_per_action_counterfactual \
  models/checkpoints/exp_001r_vonly_strict.pt \
  --budget 32 \
  --c-puct 1.5 \
  --device cpu \
  --threads 1
```

The evaluator constructs two deterministic, engine-validated positions from
legal pawn moves. In both positions the opponent has a legal pawn move that
wins immediately unless the player to move blocks it.

For every legal action in each position it records:

- raw neural policy prior over all legal actions;
- value-head prediction after applying the action, converted to the root
  player's perspective;
- shortest-path distance changes for both players;
- whether the action removes the opponent's immediate winning move;
- whether the action increases the opponent's shortest-path distance;
- forced-root search Q, with the root action fixed for every simulation and
  ordinary vanilla PUCT used for continuations below the root.

Each forced root action receives the same simulation budget. Root noise,
progressive widening, category balancing, repetition penalties, and other
search extensions are disabled.

## Outputs

Written under `data/diagnostics/per_action_counterfactual/`:

- `per_action_counterfactual.json`: full state, action-level measurements,
  checkpoint hash, and experiment settings;
- `per_action_counterfactual.csv`: one row per legal action;
- `summary.json`: compact per-position comparison of top prior, value, and Q
  actions and the number of defensive actions.

The output directory must not already contain these files. Use a fresh
`--output-dir` when repeating the experiment.

## Interpretation cautions

The child value prediction is negated because the opponent is to move after
the root action. Forced-root Q is also reported from the original root
player's perspective using the existing MCTS backup sign convention.

A distance increase is a path-delay signal, not by itself proof that the
opponent can never reach the goal. The immediate-win flag is the exact
one-reply tactical test for these positions.
