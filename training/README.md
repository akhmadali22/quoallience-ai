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


## Action-category prior experiment

The optional `--pawn-prior-fraction` changes the total policy prior mass assigned
to pawn actions while preserving relative network preferences within pawn and wall
categories. The default `None` keeps the raw network policy.

```bash
# Baseline: raw policy
python -m training.evaluate models/checkpoints/neural_v1.pt --games 20 --simulations 256 --output data/eval_raw.json
# Balanced: 50% pawn / 50% wall prior mass
python -m training.evaluate models/checkpoints/neural_v1.pt --games 20 --simulations 256 --pawn-prior-fraction 0.5 --output data/eval_balanced.json
# Pawn-biased: 70% pawn / 30% wall prior mass
python -m training.evaluate models/checkpoints/neural_v1.pt --games 20 --simulations 256 --pawn-prior-fraction 0.7 --output data/eval_pawn70.json
```

The same option is available in `training.self_play`. This changes the search
prior only, not network weights or training targets. Compare visit ratios,
search depth, and paired-seat win rates; a higher pawn visit ratio alone is
not evidence of stronger play.

### Reproducible evaluation

Evaluation assigns each game a distinct seed (`--seed` is the base seed; game N uses base seed + N - 1). With the same checkpoint and base seed, each configuration gets the same game-seed sequence, allowing paired comparisons. Equal-visit MCTS actions are selected with that game's seeded RNG rather than always choosing the first action. The JSON report records the full evaluation configuration, per-game seeds, seat-specific wins/losses/draws, and the neural agent's pawn-versus-wall action counts.

Use the same seed for each prior configuration:

```bash
python -m training.evaluate models/checkpoints/neural_v1.pt --games 20 --simulations 64 --seed 2026 --output data/eval_raw.json
python -m training.evaluate models/checkpoints/neural_v1.pt --games 20 --simulations 64 --seed 2026 --pawn-prior-fraction 0.5 --output data/eval_balanced.json
python -m training.evaluate models/checkpoints/neural_v1.pt --games 20 --simulations 64 --seed 2026 --pawn-prior-fraction 0.7 --output data/eval_pawn70.json
```

For a more robust estimate, repeat the paired comparison with several base seeds and a larger game count. Seeded tie-breaking does not make the heuristic search stochastic; it only prevents equal-visit neural choices from always being resolved by action-index order.
