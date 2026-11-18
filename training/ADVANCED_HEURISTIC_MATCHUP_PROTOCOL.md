# EXP: MCTS Budget vs Advanced Heuristic Opponent

Status: design / pilot pending. This is a separate outcome evaluation from the
same-checkpoint MCTS budget-scaling tournament. It does not train or promote a
checkpoint.

## Objective

Determine whether MCTS agents at simulation budgets 32, 128, and 256 can beat
the repository's `quoridor/advanced_ai.py` heuristic opponent, and whether
increasing MCTS budget improves competitive outcomes against this opponent.

No `advanced_ai.pt` is required. Use the heuristic implementation directly as
a fixed opponent. Turning it into a neural model would introduce an additional
training/validation variable and is unnecessary for this experiment.

## Matchups

1. Candidate checkpoint `models/checkpoints/exp_001r_vonly_strict.pt`, budget 32
   vs Advanced Heuristic.
2. Same Candidate checkpoint, budget 128 vs Advanced Heuristic.
3. Same Candidate checkpoint, budget 256 vs Advanced Heuristic.
4. Incumbent checkpoint `models/checkpoints/neural_v1.pt`, budget 128
   vs Advanced Heuristic.

The heuristic configuration must be identical across matchups. The MCTS
configuration is held fixed except for simulation budget and checkpoint where
explicitly listed.

## Seat-swapped paired seeds

Each seed consists of two games:
- MCTS occupies Player 0; heuristic occupies Player 1.
- MCTS occupies Player 1; heuristic occupies Player 0.

Use identical seed IDs across all matchups. Report pair-level outcomes and
score rate, not independent-game-only inference. The heuristic is deterministic
unless the implementation proves otherwise; nevertheless, retain the seed and
seat-swap structure for reproducibility and seat control.

Initial state is the normal Quoridor initial state. Maximum 200 plies; a game
reaching the limit is scored as a draw. Do not silently change the draw rule
between pilot and main run. Record draw rate and repetition/cycle evidence.

## Heuristic opponent runtime

The existing `AIConfig` default `time_limit=55` seconds is too costly for a
tournament. Pilot the heuristic with a fixed per-turn time limit of 0.5 seconds
and `max_depth=5`. Record actual elapsed time, completed depth, nodes, and
timeouts for every heuristic turn.

The 0.5-second cap is a provisional protocol parameter. Before the main run,
inspect the pilot's runtime and achieved depth. If changed, register the new
configuration and rerun pilot matchups; do not mix different heuristic
configurations in one analysis. Do not claim computational parity between
MCTS simulations and heuristic wall-clock time: this is a strength-baseline
comparison, not a compute-matched test.

Keep one heuristic instance per seat/player across that game's turns so its
actual-position/repetition history persists. Clear all instances between games.

## Pilot and full sample

Pilot: 4 paired seeds per matchup (8 games per matchup; 32 games total across
four matchups). Measure runtime, game completion, draws, and obvious protocol
or legality issues.

If pilot is operationally sound, main evaluation: 30 paired seeds per matchup
(60 games per matchup; 240 games total). Pilot seeds may be included in the
30-pair total only if all configurations are unchanged and their reports are
retained; explicitly identify which seed pairs are pilot-originated.

If games overwhelmingly reach the 200-ply draw cap, stop and diagnose the
draw/repetition behavior before spending the remaining compute. Do not extend
the sample merely to collect more capped draws.

## Configuration and reporting

MCTS:
- c_puct = 1.5
- selection temperature = 0 (deterministic max-visit action)
- root noise off
- progressive widening off
- category exploration, repetition penalty, pawn-progress bonus = 0
- wall-distance and wall-route weights = 0

Heuristic:
- `AIConfig(time_limit=0.5, max_depth=5)` provisionally.

Report per matchup:
- MCTS wins / heuristic wins / draws
- pair wins / losses / ties after seat swap
- score rate and 95% CI
- exact paired sign test on decisive seed pairs (with ties reported)
- game length and draw-at-ply-limit rate
- MCTS simulations, NN evaluations, search depth/selected-action metrics
- heuristic elapsed time, completed depth, node count, and timeout rate
- checkpoint SHA256 and full config

The primary comparisons are Candidate 32 vs heuristic and Candidate 128 vs
heuristic. Candidate 256 and Incumbent 128 are secondary/contextual. If formal
hypothesis tests are used, correct the family of primary tests for multiplicity.

## Decision guardrails

- This is evaluation only: no training, tuning, or checkpoint promotion.
- This test measures performance against this heuristic, not general playing
  strength or proof of superiority against all opponents.
- The heuristic's weaknesses can be exploited by MCTS; interpret this as a
  useful baseline but not a balanced universal opponent.
- No `advanced_ai.pt` is created or needed.
- Do not start the full tournament until pilot runtime and draw behavior have
  been reviewed.
