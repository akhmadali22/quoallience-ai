# Neural Quoridor: policy/value + MCTS self-play

This is an initial AlphaZero-style training pipeline. It is deliberately separate
from the existing heuristic agent, which remains the baseline opponent.

## Install

Activate the repository virtual environment, then install the optional PyTorch
dependency:

```bash
source .venv/bin/activate
# Intel CPU / no CUDA:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
# Or, on another supported platform, use the optional dependency file:
# python -m pip install -r requirements-nn.txt
```

Use a PyTorch wheel appropriate for the machine. CPU is the supported baseline;
CUDA can be selected on a compatible NVIDIA host. The current Intel integrated
GPU is not assumed to support PyTorch acceleration.

## Smoke run

Start small. MCTS wall generation is expensive because legal wall checks preserve
both players' paths.

```bash
python -m training.self_play --games 2 --simulations 8 --max-plies 120 \
  --output-dir data/neural_self_play/smoke
```

This writes:
- `replay.jsonl`: one training sample per played position, including encoded
  state, MCTS visit distribution, player, and final outcome.
- `games.jsonl`: completed game summaries and action traces.
- `selfplay_model_snapshot.pt`: exact model weights used to generate this replay.

If no checkpoint is supplied, self-play creates a randomly initialized network.
The snapshot is saved so training can start from those exact weights:

```bash
python -m training.train data/neural_self_play/smoke/replay.jsonl \
  --checkpoint data/neural_self_play/smoke/selfplay_model_snapshot.pt \
  --output models/checkpoints/neural_v1.pt --epochs 3 --batch-size 16
```

Evaluate against the existing heuristic agent, alternating neural seat:

```bash
python -m training.evaluate models/checkpoints/neural_v1.pt \
  --games 10 --simulations 16 --heuristic-time 0.2
```

## Iterative learning loop

For each iteration, generate fresh self-play data from the current checkpoint,
then train a candidate from that same checkpoint. Keep each replay in a separate
directory so old training evidence is not silently overwritten.

```bash
python -m training.self_play --checkpoint models/checkpoints/neural_v1.pt \
  --games 20 --simulations 32 --output-dir data/neural_self_play/iteration_002
python -m training.train data/neural_self_play/iteration_002/replay.jsonl \
  --checkpoint models/checkpoints/neural_v1.pt \
  --output models/checkpoints/neural_candidate_v2.pt --epochs 5
python -m training.evaluate models/checkpoints/neural_candidate_v2.pt --games 20
```

Only promote a candidate after it beats the incumbent and the heuristic baseline
over paired seats and varied seeds. The current evaluation script is a first
functional gate, not a full Elo system or a significance test.

## Representation

- Input: 10 perspective-relative 9x9 planes (own/opponent pawn, both goal rows,
  horizontal/vertical walls, both wall supplies, side-to-move marker, normalized
  ply).
- Action space: 209 indices: 81 destination squares, 64 horizontal wall slots,
  64 vertical wall slots. Legal-action masking is applied before policy softmax.
- Network: residual CNN with policy logits and a tanh value estimate.
- Search: PUCT MCTS, optional root Dirichlet noise, temperature sampling early
  in self-play, visit-count selection with seeded random tie-breaking during evaluation.
- Training: cross-entropy against normalized MCTS visit counts plus MSE against
  final game outcome.

## Current limitations

This first implementation intentionally favors a small, inspectable vertical
slice over a distributed training system. It does not yet include symmetry
augmentation, prioritized replay, batched inference, persistent tree reuse,
model-vs-model promotion gates, or crash-resumable partial-game trajectories.
Random-network self-play is only a bootstrap smoke test; meaningful strength
requires many iterations and evaluation against fixed opponents. The existing
heuristic is retained for that purpose.


## Adaptive category exploration

The optional `--category-exploration` adds an exploration bonus to the less-visited
action category (pawn moves or wall placements). It does not force a fixed pawn/wall
prior split. Raw network priors still rank actions inside each category, while the
category bonus helps prevent a large number of wall actions from completely
suppressing pawn-move exploration. Set it to `0` to disable the bonus.

```bash
# Compare baseline PUCT with adaptive category exploration
python -m training.evaluate models/checkpoints/neural_v1.pt --games 20 --simulations 256 --category-exploration 0 --output data/eval_no_category_bonus.json
python -m training.evaluate models/checkpoints/neural_v1.pt --games 20 --simulations 256 --category-exploration 0.35 --output data/eval_category_bonus.json

# Evaluate against random and heuristic opponents
python -m training.evaluate models/checkpoints/neural_v1.pt --opponent random --games 20 --simulations 128 --seed 2026 --output data/eval_random.json
python -m training.evaluate models/checkpoints/neural_v1.pt --opponent heuristic --games 20 --simulations 128 --seed 2026 --output data/eval_heuristic.json
```

## Evaluation diagnostics and reproducibility

Each game gets `base_seed + game_index`. The default selection temperature is
`0.2`, so root visits are sampled stochastically with the per-game RNG. Set
`--selection-temperature 0` for deterministic max-visit selection (random only
among exact ties). The report records per-ply neural actions, selected prior, Q,
visit count, and root pawn/wall visit counts. Use the trace to inspect whether
the search is exploring both categories, rather than judging only the final
pawn-action ratio.

Run the same base seed across configurations for paired comparisons. A nonzero
temperature is part of the evaluation protocol and must be kept identical when
comparing checkpoints. A smoke run is diagnostic, not a statistically conclusive
strength rating.


### Diagnosing pawn progress and value predictions

Evaluation reports record, for each neural turn, pawn coordinates before and after the action, shortest-path distances for both players, whether a pawn move reverses the preceding pawn move, the root value prediction, and visit-policy entropy. Per-game summaries include pawn progress, regression, reversal, and own shortest-distance change counters. Self-play replay samples also retain root value, policy entropy, and pawn visit ratio as diagnostic metadata. Training targets remain the MCTS visit distribution and final game outcome.


### Repetition-aware root selection

Neural MCTS accepts the current player's recent pawn-position history. At the root only, pawn actions that return to one of the previous positions inside the configured window receive a configurable selection penalty. Pawn actions that reduce the player's shortest path distance receive a small configurable bonus. These are soft search preferences, not legality rules; tactical moves remain available.

Defaults: `--repetition-penalty 0.75`, `--pawn-progress-bonus 0.0` (disabled until a smaller value is validated), and `--repetition-window 4`. Both self-play and evaluation accept these flags. Evaluation also reports root-value mean absolute and squared error against the final game outcome, allowing value calibration to be measured rather than guessed.

Example ablation: set `--repetition-penalty 0 --pawn-progress-bonus 0` to disable both root preferences while keeping the same checkpoint, seeds, and simulation budget.


## Four-way repetition/progress ablation

Run all four combinations with the same checkpoint, seeds, opponent, MCTS
simulation budget, and sampling temperature. The runner executes them sequentially
and writes one JSON report per configuration:

```bash
python -m training.run_ablation models/checkpoints/neural_v1.pt \\
  --games 20 --simulations 128 --seed 2026 \\
  --opponent heuristic --device cpu \\
  --output-dir data/ablation
```

The four reports are `eval_baseline.json`, `eval_repetition_only.json`,
`eval_progress_only.json`, and `eval_combined.json`. Increase to 100 or more
games for a stronger validation after inspecting the initial 20-game results.

Evaluation traces now include `opponent_distance_delta` (positive means the
neural action lengthened the opponent's shortest path). Per-game and aggregate
`strategic_diagnostics` summarize opponent path increases/decreases alongside
own pawn progress, regressions, and reversals. The report also contains aggregate
`value_calibration` MAE/MSE and mean root value.


## Root wall candidate ranking experiment

The optional `--wall-distance-weight` reorders only the root's pending wall
candidates before progressive widening admits them. It combines the neural
policy's prior rank with the immediate path impact:

`opponent shortest-path increase - own shortest-path increase`.

The feature is disabled by default (`0.0`). It evaluates each legal root wall
once per search, rather than adding BFS work to every expanded tree node. This
is a heuristic screening signal, not a full strategic wall evaluation: walls
that preserve current shortest distances can still affect future options.

Compare small weights with identical seeds and settings:

```bash
for weight in 0 0.005 0.01 0.02; do
  python -m training.evaluate models/checkpoints/neural_v1.pt \
    --games 50 --simulations 128 --seed 4026 \
    --repetition-penalty 0.75 --pawn-progress-bonus 0 \
    --wall-distance-weight "$weight" --opponent heuristic --device cpu \
    --output "data/wall_ranking/eval_wall_${weight}.json"
done
```

Keep the baseline weight at zero unless paired-seat results show a repeatable
improvement. Check both win rate and wall diagnostics; an increase in wall
frequency alone is not evidence of stronger play.

## Small progress-bonus sweep and wall-effect diagnostics

The 20-game ablation showed that a progress bonus of 0.10 overwhelms wall
selection. Do not use that value as a default. With repetition avoidance enabled,
sweep smaller values using identical seeds and settings:

```bash
python -m training.run_progress_sweep models/checkpoints/neural_v1.pt \\
  --games 50 --simulations 128 --seed 3026 \\
  --repetition-penalty 0.75 --progress-values 0 0.01 0.02 0.05 \\
  --opponent heuristic --device cpu \\
  --output-dir data/progress_sweep
```

The runner writes one report per progress value. Use at least 50 games for this
screening run, then increase promising configurations to 100–200 games.

Evaluation reports now separate wall placement effects from pawn movement:
- `wall_opponent_distance_delta`: opponent shortest-path increase caused by
  this wall (positive is strategically useful; negative helps the opponent).
- `wall_own_distance_delta`: own shortest-path change caused by this wall
  (positive means the wall also lengthened our route).
- Aggregate counters report wall placements that increased/decreased each
  player's shortest path and the total distance deltas.

These are immediate shortest-path effects, not a complete measure of wall
quality. A wall can be useful by changing future tactical options even when the
opponent's shortest path does not immediately increase. Inspect these metrics
alongside game outcomes and action traces.


## Strategic wall route-redundancy experiment

Shortest distance alone misses walls that remove alternative routes without
changing the current minimum distance. The optional `--wall-route-weight`
adds a root-only route-redundancy signal to wall candidate ranking. It compares
log changes in the number of shortest routes available to the opponent and to
ourselves. The route count is capped, and this remains a local geometric
heuristic rather than a full multi-turn tactical evaluator.

Keep both wall-ranking weights disabled by default. First compare the route
signal in isolation while holding distance ranking at zero:

```bash
for weight in 0 0.01 0.02 0.05; do
  python -m training.evaluate models/checkpoints/neural_v1.pt \
    --games 50 --simulations 128 --seed 5026 \
    --repetition-penalty 0.75 --pawn-progress-bonus 0 \
    --wall-distance-weight 0 --wall-route-weight "$weight" \
    --opponent heuristic --device cpu \
    --output "data/wall_routes/eval_route_${weight}.json"
done
```

Evaluation reports also measure the route impact of each wall actually
selected by the neural agent:
- `wall_opponent_route_reduction`: log1p(shortest routes before) minus
  log1p(shortest routes after). Positive means the wall removes opponent routes.
- `wall_own_route_reduction`: the same reduction for our own routes. Positive
  means the wall sacrifices our alternatives.
- `wall_net_route_score`: opponent route reduction minus own route reduction.
- Per-game and aggregate counters report how often either player's shortest
  route count was reduced or increased, plus summed route scores.

These diagnostics use the same capped shortest-route count as candidate ranking.
They are computed only for wall actions actually selected by the neural agent,
so they do not add route-count BFS work to pawn actions. Compare the selected
wall route metrics with candidate-ranking scores and game outcomes.

Compare action traces, route-score diagnostics, wall path effects, and wins.
Do not promote a weight based only on fewer own routes being sacrificed; it must
also improve opponent disruption and eventually head-to-head outcomes.

## MCTS search-budget evaluation

Use a direct MCTS-vs-MCTS tournament to determine whether additional search
simulations improve playing strength. Both agents use the exact same checkpoint
and search preferences; only their simulation budgets differ. Each seed is
played twice, with the agents swapping seats, so both games in a pair start from
the same initial position and use the same seed.

Example: compare 64 simulations against 128:

```bash
python -m training.evaluate_mcts_scaling models/checkpoints/neural_v1.pt \\
  --low-simulations 64 --high-simulations 128 \\
  --pairs 20 --seed 20261109 --device cpu \\
  --output data/mcts_scaling/eval_64_vs_128.json
```

This runs 40 games (20 paired seeds, two seat assignments per seed). The report
includes paired outcomes, per-agent win/draw/loss scores, and MCTS diagnostics
per decision: expanded nodes, neural-network evaluations, leaf depth, root
policy entropy, pawn/wall visit ratios, and selected-action Q/visit counts.

Run a second comparison, such as 128 vs 256, with the same paired-seed protocol
and compare the score rates. Do not infer that more simulations are better from
search depth or visit counts alone: the higher-budget agent must demonstrate
better paired match results. This experiment isolates search-budget effects
under one checkpoint; it does not independently establish policy/value quality.


## Progressive widening and policy-target correctness

At the root, MCTS now applies Dirichlet noise and optional wall-candidate ranking
before admitting the initial wall set. This ensures that the first admitted walls
are selected using the perturbed/ranked priors rather than the unmodified network
ordering.

Self-play records a `policy_mask` containing the actions admitted to the root
search by the end of that search. Training normalizes the MCTS visit target and
computes policy cross-entropy only over this mask. Wall candidates that remained
in the progressive-widening queue are therefore excluded from the policy
softmax and receive no policy gradient from a search that never examined them.
Older replay files without `policy_mask` remain readable and retain the legacy
full-action-space training behavior.

MCTS backup also increments the leaf node visit count, so every node on a
completed simulation path receives one visit. This keeps node visit counts
consistent with incoming edge visits and with the visit-dependent PUCT and
progressive-widening calculations.

These changes correct search/training bookkeeping; they are not evidence of
stronger play. Regenerate self-play data after this change before training a new
checkpoint. Existing replay data has no searched-action mask and cannot recover
which wall candidates were pending during its original searches.

## MCTS policy/value contribution ablation

Use this tournament to isolate the contribution of the trained policy prior and
value estimate while keeping every agent inside PUCT MCTS. It compares the full
network against three controlled ablations:

- `full`: neural policy + neural value.
- `uniform_policy`: uniform prior over legal actions + neural value.
- `zero_value`: neural policy + zero leaf value.
- `uniform_policy_zero_value`: uniform legal-action prior + zero leaf value.

The default protocol disables progressive widening, category exploration, and
root-only repetition/progress preferences. This is intentional: with progressive
widening enabled, the policy prior also determines which wall candidates enter
the tree, so disabling it keeps the policy ablation focused on PUCT prior
selection rather than candidate-admission order. Exact PUCT score ties are
randomized with reproducible per-agent RNG streams, avoiding a systematic
preference for the first action index under uniform priors. Terminal outcomes
still back up normally when the value head is zeroed.

Run the paired-seat tournament:

```bash
python -m training.evaluate_mcts_ablation models/checkpoints/neural_v1.pt \
  --simulations 128 --pairs 10 --seed 20261110 --device cpu \
  --output data/mcts_ablation/policy_value_ablation.json
```

Each matchup plays every seed twice with seats swapped:
- full vs uniform policy: measures the neural policy's contribution while
  retaining the neural value head.
- full vs zero value: measures the neural value head's contribution while
  retaining the neural policy.
- full vs uniform policy + zero value: measures the combined contribution.

The JSON report includes paired outcomes, per-game action traces, selected
Q/visit counts, root value, policy entropy, expanded nodes, neural evaluations,
and pawn/wall visit ratios. This is a component ablation under a fixed MCTS
protocol, not a claim of absolute strength. After isolating the network
components, repeat with `--progressive-widening` to measure their interaction
with the production candidate-admission mechanism.



### Crash-safe MCTS ablation diagnostics

The policy/value ablation evaluator now appends one complete game record to a
JSON Lines file immediately after each game and flushes it to disk with fsync.
The default path is the report path with a .jsonl extension. The final
aggregate JSON is still written after the full tournament finishes.

Each ply records both players' shortest-path distances before and after the
action, distance deltas, and whether the ply made measurable progress. Per-game
diagnostics include exact full-state repetitions, repeated pawn positions,
repeated action windows of 4/6/8 plies, total plies without progress, the
longest consecutive no-progress streak, and final distances.

Draws caused by reaching --max-plies are labeled max_plies. The engine
currently has no separate terminal-draw rule, so a draw before that limit would
be labeled terminal_without_winner.

The JSONL file is append-only. Use a fresh output path for a fresh experiment,
or retain the same path to preserve records from interrupted runs. Each record
has a run_id so separate invocations can be distinguished.

Example:

    python -m training.evaluate_mcts_ablation models/checkpoints/neural_v1.pt \
      --simulations 128 --pairs 10 --seed 20261110 --device cpu \
      --output data/mcts_ablation/policy_value_ablation.json

The durable game-by-game records are written to
data/mcts_ablation/policy_value_ablation.jsonl.


### Pawn-position repetition diagnostic semantics

In the MCTS policy/value ablation JSONL, `pawn_move_position_repeats` counts
repeated pawn destinations only on actual pawn-move turns. The starting square
is included as the first observed position. Wall-placement turns are excluded,
since a stationary pawn during a wall action is not a pawn-position revisit.
The older `pawn_position_repeats` key is retained as an alias with the corrected
semantics for easier comparison with existing analysis scripts. Per-ply traces
include `pawn_position_repeat` so individual revisits can be inspected.


The MCTS ablation per-ply trace also stores the top five root actions ranked
by visit count (prior breaks ties), including action category, prior, Q value,
visits, and whether each action was active in the search. The selected action's
rank is recorded as `selected_action_rank_by_visits`. This is observational
instrumentation only; it does not alter PUCT selection or root priors.
