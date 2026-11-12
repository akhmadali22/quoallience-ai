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
slice over a distributed training system. It now includes a conservative
paired-seat candidate-vs-incumbent promotion gate, but does not yet include
symmetry augmentation, prioritized replay, batched inference, persistent tree
reuse, a full Elo/rating system, or crash-resumable partial-game trajectories.
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

Defaults: `--repetition-penalty 0.0`, `--pawn-progress-bonus 0.0`, and `--repetition-window 4`. Both preferences are opt-in experiments and remain disabled in the vanilla-PUCT baseline. The pawn-coordinate preference is separate from the full-state repetition instrumentation documented below.

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


### Root policy, visit, and PUCT diagnostics

The evaluator reports three distinct root distributions:
- `root_prior_entropy`: entropy of the network prior over all legal root actions.
- `root_active_prior_entropy`: entropy of priors renormalized over currently active root edges, matching the prior normalization used by PUCT selection.
- `root_visit_entropy`: entropy of the root visit distribution after simulations.

The legacy `root_policy_entropy` and `mean_root_policy_entropy` keys remain as
compatibility aliases for visit entropy; they are not neural policy entropy.
Each action record includes its final-snapshot `puct_score`, computed with the
same Q, normalized active prior, exploration term, category bonus, and root
adjustment as `_select`. The trace includes top root actions ranked by visits
and by this final-snapshot PUCT score. The selected action is chosen by maximum
visit count, so its rank by visits is expected to be first and is not an
independent measure of search quality.



## Root MCTS instrumentation and full-state repetition history

Self-play replay samples and candidate-evaluation game traces now retain root
search diagnostics without changing vanilla PUCT selection. Every legal pawn
action records:
- raw neural-network prior before Dirichlet noise;
- effective prior after optional root noise and normalized active prior used by PUCT;
- Q value, U exploration term, final-snapshot PUCT score, and visit count;
- destination square and shortest-path distance before/after the move;
- whether the resulting full game state appeared in the recent repetition window.

The trace also stores the selected action's own root statistics, root pawn/wall
visit ratios, prior and visit entropies, raw neural prior mass by action
category, and the five highest-prior wall candidates. Per-game summaries count
repeated root positions, selected pawn moves into recently seen states, and
pawn progress/regression moves. This separates a weak network prior from
insufficient exploration or an unfavorable Q estimate.

Repetition identity includes both pawn positions, the complete wall layout,
remaining wall inventory, side to move, and winner. Ply is excluded because
the same board position at a different move number is still the same position.
The game runners pass the chronological state-key history into MCTS. The
instrumentation uses two explicit boundaries. For a candidate next state,
it compares against the current root plus the previous `repetition_window - 1`
positions, so a candidate that returns to a full state from four plies earlier
is visible with the default window of four. For the current root itself, it
compares the root against the previous four positions, which requires retaining
five keys including the current root; this catches a same-side-to-move cycle at
plies `t` and `t+4`. The two counts are reported separately. Repetition remains
observational: it does not declare a draw, remove legal actions, or modify PUCT
scores. The older root-only pawn-coordinate preference
remains a separate optional experiment and is not the full-state repetition
detector.


## Controlled vanilla-PUCT search experiments

The experiment sequence separates search mechanics from network-component attribution.
The default PUCT score remains:

`Q(s,a) + c_puct * P(s,a) * sqrt(max(1, N(s))) / (1 + N(s,a))`

Category exploration, repetition penalties, pawn-progress bonuses, and wall
ranking heuristics remain disabled in these controlled runs. Progressive
widening is an independent MCTS option; it admits every legal pawn action
immediately and adds legal wall actions in descending prior order according
to the configured visit-dependent widening schedule. It does not alter
terminal handling or value backup.

### 1. Progressive widening and simulation-budget matrix

Run matched MCTS-vs-MCTS comparisons at 16 vs 64 and 64 vs 128 simulations,
once with progressive widening disabled and once enabled:

```bash
MCTS_PAIRS=10 MCTS_MAX_PLIES=200 \
  bash training/run_mcts_search_matrix.sh models/checkpoints/neural_v1.pt
```

The runner uses the same checkpoint, seed set, C-PUCT (1.5), vanilla PUCT
settings, and paired seat swaps in every comparison. It writes four distinct
reports under `data/mcts_search_matrix/seed_<seed>/`. Increase
`MCTS_PAIRS` to 20 or more for follow-up evaluation. A result dominated by
max-ply draws is inconclusive; inspect terminal wins, paired outcomes, and
action diagnostics rather than interpreting draw score as win rate.

### 2. Category-balanced uniform policy baseline

The policy/value evaluator now includes two distinct uniform baselines:

- `uniform_policy`: equal prior per legal action. Its total wall prior mass
  can dominate because the opening has many more legal wall placements than
  pawn moves.
- `category_uniform_policy`: 50% total prior mass for pawn moves and 50% for
  wall placements whenever both categories are available, with mass distributed
  within each category. If only one category is available, it receives 100%.

This changes only the prior distribution supplied to PUCT. It does not add a
category bonus to the PUCT score and does not change Q backup. With progressive
widening enabled, the active subset is renormalized for selection, so the
effective active pawn/wall mass can differ from the full legal-action 50:50
prior until more wall candidates are admitted. For a clean policy/value
component comparison, first run with progressive widening disabled.

### 3. Re-evaluate policy and value contributions

After inspecting the search matrix and selecting a simulation budget, run:

```bash
MCTS_SIMULATIONS=128 MCTS_PAIRS=20 MCTS_PROGRESSIVE_WIDENING=0 \
  bash training/run_mcts_policy_value_ablation.sh models/checkpoints/neural_v1.pt
```

The report compares the full neural policy/value model against uniform legal
policy, category-balanced uniform policy, zero value, and the combined
policy/value ablations. All agents still use PUCT MCTS. To measure the
interaction with progressive widening after the isolated comparison, repeat
with `MCTS_PROGRESSIVE_WIDENING=1` and a separate output path.

Do not promote a checkpoint or conclude that a component improves playing
strength from a handful of games or max-ply draws. Keep the checkpoint fixed
during these experiments; retrain only after the search protocol and component
effects have been measured consistently.

## Iterative self-play, candidate evaluation, and promotion

The bounded training loop automates the complete iteration:

1. Generate fresh self-play games from the current incumbent checkpoint.
2. Train a candidate from that same incumbent checkpoint using the new replay.
3. Play the candidate against the incumbent with identical MCTS settings.
4. Swap seats for every seed and apply a conservative paired-result promotion gate.
5. Promote only accepted candidates; otherwise retain the incumbent for the next iteration.

Run from the repository root:

```bash
python -m training.run_training_loop \
  --initial-checkpoint models/checkpoints/neural_v1.pt \
  --iterations 3 \
  --games 20 \
  --selfplay-simulations 32 \
  --evaluation-simulations 64 \
  --evaluation-pairs 50 \
  --min-decisive-pairs 10 \
  --device cpu
```

Each run gets a unique directory under `data/alphazero_runs/`. It preserves
self-play replay, game summaries, candidate checkpoints, evaluation reports,
promoted champion checkpoints, and `training_state.json`. Rejected candidates
are retained for analysis and do not replace the incumbent. Later iterations
train from the latest promoted champion, or from the original checkpoint if no
candidate has passed promotion.

The loop defaults to the controlled vanilla-PUCT baseline: C-PUCT 1.5,
category exploration 0, repetition penalty 0, pawn progress bonus 0, and
progressive widening disabled. To explicitly run the optional progressive
widening variant in both self-play and candidate evaluation, add
`--progressive-widening`. Do not compare candidates evaluated with different
search settings.

The standalone paired-seat evaluator can also be run directly:

```bash
python -m training.evaluate_candidate \
  models/checkpoints/neural_candidate_v2.pt \
  models/checkpoints/neural_v1.pt \
  --pairs 50 --simulations 64 --max-plies 200 \
  --min-decisive-pairs 10 --alpha 0.05 \
  --output data/candidate_evaluation.json
```

Promotion is deliberately conservative. Results are grouped by paired seed
(two games with seats swapped). A paired result is a win only when the candidate
scores more than one point across those two games, a loss when it scores less,
and a tie otherwise. The candidate must have at least the configured number of
decisive pairs, more paired wins than losses, an overall score rate above 0.5,
and a one-sided exact sign-test p-value below alpha. Max-ply draws count as half
a point in the score rate but do not count as decisive pairs. Therefore an
all-draw tournament cannot promote a candidate.

This gate is evidence for promotion, not proof of general playing strength.
Use enough pairs and varied seeds, and keep the evaluation MCTS settings fixed
between candidate and incumbent. The loop is bounded by `--iterations`; it
does not run indefinitely.


## Pawn/wall policy-target audit and vanilla PUCT budget matrix

Audit a self-play replay before using it for training:

```bash
python -m training.audit_policy_targets \\
  data/diagnostics/selfplay/replay.jsonl \\
  --output data/diagnostics/selfplay/policy_target_audit.json
```

The report separates normalized MCTS visit-target mass from raw neural prior
mass and MCTS root visit ratio, plus target entropy, nonzero target actions,
and the number of searched actions in the pawn and wall categories. It includes
ten-ply phase buckets so opening targets are not hidden by late-game pawn-only
positions. It also checks that no nonzero target mass falls
outside the recorded `policy_mask`. These are per-position target statistics,
not counts of legal actions and not the neural network's predicted prior.

During training, the existing policy cross-entropy is now reported as additive
pawn and wall contributions, alongside their average target-mass shares. The
loss formula is unchanged: this is instrumentation, not category reweighting.
No pawn/wall balancing coefficient is introduced.

To measure the current checkpoint's actual pawn/wall policy cross-entropy and
value MSE on a replay without training or modifying the checkpoint, run:

```bash
python -m training.audit_policy_loss \
  models/checkpoints/neural_v1.pt \
  data/diagnostics/selfplay/replay.jsonl \
  --output data/diagnostics/selfplay/policy_loss_audit.json
```

The report includes additive pawn and wall contributions to policy
cross-entropy, value loss, total objective, and the same ten-ply phase buckets.


After the audit, compare all three vanilla PUCT budgets against the same
checkpoint and paired seed set:

```bash
MCTS_PAIRS=10 MCTS_SEED=20261112 MCTS_MAX_PLIES=200 \\
  bash training/run_mcts_budget_matrix.sh models/checkpoints/neural_v1.pt
```

This runs 8-vs-32, 32-vs-64, and 8-vs-64 with seat swaps. Progressive widening,
category exploration, repetition penalty, pawn progress bonus, wall-distance
ranking, and wall-route ranking are disabled. Increase `MCTS_PAIRS` for stronger
evidence. If most games end at the max-ply limit, treat the comparison as
inconclusive rather than calling the higher simulation budget stronger. Keep
the checkpoint fixed until this matrix is reviewed; only then decide whether
the neural policy needs a separate experiment.


## Isolate opening vanilla PUCT decisions

After auditing policy targets and checkpoint loss, inspect why MCTS changes the
pawn/wall allocation on the exact same replay positions. Run:

```bash
python -m training.audit_opening_puct \
  models/checkpoints/neural_v1.pt \
  data/diagnostics/selfplay/replay.jsonl \
  --budgets 8 32 64 \
  --max-ply 19 \
  --output data/diagnostics/selfplay/opening_puct_audit.json
```

The script reconstructs each replay state at ply 0-19 and runs all requested
budgets independently on the same checkpoint and same positions. It records
every legal root action's neural prior, initial Q/U/PUCT score, final visits,
final Q/U/PUCT score, and category. Category summaries include legal action
counts, prior mass, visit share, and visit-weighted Q.

This is a controlled vanilla-PUCT diagnostic: progressive widening, root
Dirichlet noise, category-balanced priors, category exploration, repetition
penalty, and pawn-progress bonus are explicitly disabled. The script does not
train or modify the checkpoint. The initial U is the prior-driven exploration
term before the first simulation; final Q and U are post-search snapshots, not
a historical trace of every selection decision. Use the per-action rows to
distinguish low prior allocation from low backed-up Q and later exploration
effects.



### Self-play opening PUCT selection trace

To capture the actual PUCT decisions in self-play (rather than replaying a fixed position later), rerun the same self-play setup with tracing enabled:

```bash
python -m training.self_play \
  --checkpoint models/checkpoints/neural_v1.pt \
  --games 2 --simulations 8 --max-plies 60 --seed 20261102 \
  --output-dir data/diagnostics/selfplay_selection_trace \
  --record-selection-trace --trace-max-ply 19
```

The trace is enabled only for plies 0-19 by default. Each replay sample stores every root and deeper-node selection event for each simulation. At each selection it records the best overall, best pawn, and best wall candidates, including their PUCT score, Q, U, prior, and visits immediately before selection. The selected edge also stores Q, visit count, value sum, and Q delta after that simulation's backup. The per-position summary identifies the first simulation where the best wall score exceeds the best pawn score, the first simulation that selects a pawn, and whether pawn is never selected at the root.

Tracing is observational only. It does not change the checkpoint, priors, root noise, temperature, PUCT formula, tie-breaking, or backup. Use a fresh output directory. Keep the checkpoint, seed, game count, simulation budget, and other self-play arguments identical to the original run when comparing traces.


### Chronological PUCT selection trace

The opening PUCT audit now records the actual selection decision at every
simulation step and every visited tree depth. Each trace entry captures the
highest-scoring overall candidate, the highest-scoring pawn candidate, the
highest-scoring wall candidate, and the action selected by PUCT. For each
candidate it stores action index/category, normalized selection prior, raw
edge prior, Q, U, category bonus, optional selection adjustment, PUCT score,
and visits immediately before selection.

The trace is observational and uses the existing selection calculation. It
does not change priors, tie-breaking, action availability, Q backup, or the
PUCT formula. Comparing the best pawn and wall scores at the root across
simulation indices shows whether pawn actions lose immediately on prior-driven
U, or only after Q values diverge. Deeper trace entries expose the same process
at subsequent nodes.

Run the audit with the same command as the opening PUCT audit:

```bash
python -m training.audit_opening_puct \
  models/checkpoints/neural_v1.pt \
  data/diagnostics/selfplay/replay.jsonl \
  --budgets 8 32 64 \
  --max-ply 19 \
  --output data/diagnostics/selfplay/opening_puct_selection_trace.json
```

The output contains a `selection_trace` array for each position/budget run.
Root entries have `selection_depth=0`; `simulation` identifies the search
iteration. `top_overall`, `top_pawn`, `top_wall`, and `selected` are
snapshots from immediately before that selection. The trace can be large
because it preserves each actual selection step; use a fresh output filename.


## Search-free neural policy alignment audit

To evaluate the neural policy independently of MCTS, compare the checkpoint's
masked raw policy probabilities directly with the visit targets already stored
in replay:

```bash
python -m training.audit_policy_alignment \
  models/checkpoints/neural_v1.pt \
  data/diagnostics/selfplay/replay.jsonl \
  --output data/diagnostics/selfplay/policy_alignment_audit.json
```

This is inference-only. It does not run MCTS, train the network, modify the
checkpoint, change the loss, or alter the architecture. The recorded
`policy_mask` is applied exactly as in training.

The report includes per-position and ten-ply phase summaries for:
- policy cross-entropy and KL divergence against the replay visit target;
- comparison with a uniform prior over legal actions;
- target and predicted pawn/wall probability mass;
- category-conditional cross-entropy and KL, separating category allocation
  from action ranking within each category;
- rank of the best target-supported action and target mass captured by the
  policy's top 1, 5, and 10 predictions.

Use this report alongside the opening PUCT audit. The alignment audit measures
what the network predicts before search; the PUCT audit measures how search
turns those priors and backed-up values into root visits. Neither metric alone
establishes strategic playing strength.
