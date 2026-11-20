# Teacher State-Dependence Audit

## Purpose
Read-only paired replay of each conflicting state under empty versus reconstructed
teacher history, at the original 80 ms budget and a longer 320 ms budget.

## Method
- Loads EXP-001 decisive-game JSONL and reconstructs every encoded board.
- Identifies exact float32 state fingerprints with both +1 and -1 targets.
- Reconstructs each player's persistent AI history from prior positions in the
  same game on that player's turns, retaining the last 16 entries as
  `QuoridorAI._record_actual_position` does.
- For each selected occurrence, runs four paired conditions:
  1. Empty history, 80 ms.
  2. Actual reconstructed history, 80 ms.
  3. Empty history, 320 ms.
  4. Actual reconstructed history, 320 ms.
- Keeps `max_depth=2`, matching the original EXP-001 generator, so the time
  comparison isolates deadline sensitivity rather than deeper search.
- Default selects up to four evenly spaced occurrences per conflict state,
  covering every conflicting state while controlling runtime. Set the CLI
  occurrence limit to zero to replay every occurrence.

## Measures
Action, root-perspective score, completed depth, timeout, elapsed time, and
visited nodes. Paired comparisons quantify history effects and short-versus-long
budget effects. Per-state target/action divergence is reported.

## Interpretation limits
A changed action demonstrates history dependence but does not alone establish
that history caused the final game outcome. A longer budget at fixed depth
measures deadline sensitivity only. The audit replays the current source
implementation; if it differs from the exact generation revision, results are
reproduction evidence, not a bit-exact historical replay.

## Safety
No training, optimizer, checkpoint, loss, replay, or PUCT changes.

## Run
```bash
bash training/run_teacher_state_dependence_audit.sh
```

Outputs:
- `data/diagnostics/teacher_state_dependence/run_001/summary.json`
- `data/diagnostics/teacher_state_dependence/run_001/teacher_state_dependence.json`
