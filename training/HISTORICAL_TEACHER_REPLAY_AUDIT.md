# Historical Teacher Replay Audit

## Purpose

Exhaustively replay all 1,987 occurrences belonging to the 52 exact-state target-conflict groups in EXP-001. This is a read-only diagnostic. It does not train or modify the model, loss, checkpoint, replay buffer, or PUCT.

## Historical provenance established

The EXP-001 dataset was added in commit:

- Commit: `f40a4ce3df337939cd52a16108b281d4b1e6f4d4`
- Date: 2026-11-13
- Message: `result: exp_001 dan exp_002`

The committed `heuristic_games_summary.json` records:

- 50 decisive games, 3,780 samples
- `time_limit=0.08`
- `max_depth=2`
- `max_plies=300`
- `discount=0.99`
- `seed=20261113`

The following files at that historical commit were compared byte-for-byte with the current branch:

- `quoridor/advanced_ai.py`
- `quoridor/engine.py`
- `quoridor/neural_ai/encoder.py`
- `training/generate_heuristic_games.py`

All four matched exactly. The replay also validates the pinned dataset SHA-256 and the three source hashes recorded in the earlier teacher-state audit. If any provenance check fails, finalization stops rather than labeling the run a historical replay.

## Replay matrix

For every one of the 1,987 conflict occurrences, run:

1. Empty history, 80 ms, depth 2.
2. Reconstructed same-player game history, 80 ms, depth 2.
3. Empty history, 320 ms, depth 2.
4. Reconstructed same-player game history, 320 ms, depth 2.

History reconstruction mirrors the generator's two persistent per-game AI instances and the last-16-turn retention used by `QuoridorAI`.

Total expected searches: 7,948.

## Classifications

Classifications are deliberately non-exclusive:

- **History-influenced:** at least one replay of a board changes action when actual history is supplied.
- **Deadline-sensitive:** at least one replay changes action between 80 ms and 320 ms at fixed depth 2.
- **Same action, opposite outcomes:** the same actual-history/80 ms action appears in occurrences of the same exact board under both +1 and -1 targets.
- **No shared action across outcomes:** sampled/replayed actual-history/80 ms actions do not overlap between target classes.

The last category is not proof that history caused the outcome. It only says the replayed teacher actions diverge across the target classes.

## Run

```bash
bash training/run_historical_teacher_replay_audit.sh
```

Estimated runtime is several minutes. The audit prints progress every 20 occurrences.

## Outputs

- `data/diagnostics/historical_teacher_replay/run_001/summary.json`
- `data/diagnostics/historical_teacher_replay/run_001/historical_teacher_replay.json`
- Raw unclassified replay artifacts are retained under `data/diagnostics/historical_teacher_replay/run_001/raw/`.

## Safety

No training, optimizer, checkpoint, loss, dataset, replay, or PUCT changes.
