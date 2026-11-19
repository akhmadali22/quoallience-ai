# Forced Outcome Value Consistency Audit

## Objective

Determine whether positive value estimates in one-ply forced-loss positions are caused by:
1. a side-to-move / root-perspective sign conversion error;
2. incorrect terminal-state semantics in the engine or evaluator; or
3. a value head that fails to recognize a forced tactical loss.

The audit reuses the exact frozen threat_no_defense and no_threat positions from the preceding report. It does not resample them.

## Value perspective contract

The inspected implementation establishes:

- QuoridorNet.value_head emits a tanh scalar.
- encode_state() encodes from state.current_player perspective.
- The network therefore predicts value for the side to move.
- At a root state, raw value is already root-player perspective.
- After a nonterminal root action, the opponent is side to move. Its raw value must be negated to express value from the original root player's perspective.
- A terminal state is not evaluated by the network. The engine winner field gives the exact outcome.

For a forced-loss root position:
- S0: root value should be negative.
- S1: after a root action, the opponent has an immediate winning reply; child value from the opponent's side-to-move perspective should be positive.
- S1 converted to root perspective should be negative.
- S2: after the opponent's winning reply, the engine must report the opponent as winner; exact value is -1 from the original root perspective.

## Additional checks

- A synthetic legal winning move verifies that apply_move() sets winner, toggles current_player, and removes legal moves after terminal.
- Horizontal reflection maps pawn x to 8-x and wall anchor x to 7-x. The audit checks legal-action-set equivalence, transition commutation, and value invariance for roots and nonterminal children.
- All opponent immediate winning pawn replies are enumerated for every nonterminal child.
- A position is called a one-ply forced loss only when the opponent has an immediate winning reply before the root action, the root has no immediate winning action, and every legal root action leaves at least one immediate winning reply.

## Safety boundaries

- Inference only. No training, optimizer, replay buffer, checkpoint writes, or architecture changes.
- No MCTS or PUCT.
- Terminal states are adjudicated exactly by the engine and are not fed to the neural network.
- Horizontal reflection is a diagnostic invariance test, not an augmentation or training operation.
- Controls are seeded-playout samples from the preceding report, not estimates of competitive-game frequency.

## Run

Run:

    bash training/run_forced_outcome_value_consistency.sh

Equivalent direct command:

    python -m training.audit_forced_outcome_value_consistency models/checkpoints/exp_001r_vonly_strict.pt \
      --source-json data/diagnostics/tactical_action_value_separation/run_001/tactical_action_value_separation.json \
      --batch-size 128 --device cpu --threads 1 \
      --output-dir data/diagnostics/forced_outcome_value_consistency/run_001

Use a fresh output directory. The script refuses to overwrite a non-empty output directory.

## Outputs

- summary.json: hashes, value-perspective contract, engine and symmetry checks, cohort summaries, and forced-loss metrics.
- forced_outcome_value_consistency.json: complete position/action/reply-level detail.

## Interpretation

First inspect:
1. engine_semantics_check.passed;
2. zero horizontal legal-set and transition mismatches;
3. root and child value sign distributions in forced_loss_summary.

If root values are positive and child side-to-move values are negative despite the opponent having a verified immediate winning reply, while engine and symmetry checks pass, the evidence points to a value-head tactical recognition/calibration failure rather than a sign conversion or transition bug.

If the sign is correct at root but wrong only after conversion, inspect the evaluator's sign handling. If reflection checks fail, investigate encoding/model symmetry before attributing the result to tactical learning.

Do not change the checkpoint, architecture, loss, PUCT, or MCTS based on this diagnostic alone.
