# Quoridor self-play benchmark

The self-play runner plays the current `QuoridorAI` against an independent instance
of the same AI. It records full before/after state snapshots, chosen actions,
search metadata, shortest-path changes caused by walls, and a CSV summary.

## Run

From the repository root, with the project's Python environment activated:

```bash
python -m benchmarks.self_play --games 100 --max-depth 3 --time-limit 0.2
```

Defaults:
- 100 games
- max search depth 3
- 0.2 seconds per AI turn
- maximum 200 plies per game
- 4 seeded random legal pawn moves before AI play, to avoid repeating the exact same deterministic match
- alternating starting player (P0, then P1)

To benchmark from the normal initial board without randomized opening moves:

```bash
python -m benchmarks.self_play --games 100 --opening-plies 0 --seed 20261029
```

Use a new output directory for each run:

```bash
python -m benchmarks.self_play --games 100 --output-dir data/self_play/baseline_depth3
```

The script refuses to overwrite an existing summary or wall-analysis file. Use a
new `--output-dir` for each experiment. The run is reproducible for a fixed seed,
configuration, code version, and Python runtime.

## Output

Each run creates a timestamped directory under `data/self_play/` unless
`--output-dir` is supplied:

- `run_metadata.json`: settings and seed.
- `games/game_NNNN.jsonl`: every opening action and AI action, including complete
  state snapshots, distances, chosen score, search depth, elapsed time, nodes,
  timeout flag, and search statistics.
- `wall_analysis.jsonl`: one record per AI wall, including shortest-path delay
  for the opponent and detour for the acting AI.
- `game_summary.csv`: one row per match, with winner, turns, wall counts, and
  thinking-time and wall-effect aggregates.

Analyze a run:

```bash
python -m benchmarks.analyze_self_play data/self_play/<run-directory>
python -m benchmarks.analyze_self_play data/self_play/<run-directory> --json
```

## Interpretation

A wall's `opponent_delay` is the opponent's shortest distance after placement
minus the distance before placement. `own_detour` is the same calculation for
the player who placed it. A zero-delay wall is **not automatically a bad move**:
it may close an alternative route or improve a later tactical sequence. Use these
fields as diagnostic signals, not as a complete strategic judgment.

Identical self-play is useful for collecting repeatable trajectories and measuring
seat asymmetry, but its P0/P1 win rate is not an absolute strength score. To
measure improvement, compare a candidate build against a fixed baseline build
using paired seeds and alternating seats. Also test against diverse opponents;
identical agents can share the same blind spots.

The randomized opening consists only of legal pawn moves and is recorded as
`opening_action`, not counted as an AI decision. Set `--opening-plies 0` to
remove this source of variation.
