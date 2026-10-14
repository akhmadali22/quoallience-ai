from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .engine import (
    GameState,
    Position,
    Wall,
    apply_move,
    apply_wall,
    legal_pawn_moves,
    legal_walls,
    shortest_distance,
    validate_state,
)

# Nilai tetap agar modul tidak perlu mengimpor konstanta dari modul lain.
WALLS_PER_PLAYER = 10


@dataclass(frozen=True)
class AIConfig:
    # Reward untuk setiap langkah kemajuan menuju tujuan.
    own_progress_weight: float = 1.0

    # Reward untuk setiap langkah tambahan yang harus ditempuh lawan.
    opponent_delay_weight: float = 1.2

    # Penalti dasar untuk menggunakan sebuah dinding.
    wall_cost: float = 0.25

    # Penalti tambahan saat persediaan dinding semakin sedikit.
    wall_scarcity_weight: float = 0.15

    # Batas waktu evaluasi dalam detik.
    time_limit: float = 0.40


@dataclass(frozen=True)
class Action:
    type: str
    destination: Position | None = None
    wall: Wall | None = None

    def to_dict(self) -> dict:
        if self.type == "move":
            assert self.destination is not None
            return {
                "type": "move",
                "destination": {
                    "x": self.destination[0],
                    "y": self.destination[1],
                },
            }

        assert self.wall is not None
        return {
            "type": "wall",
            "wall": {
                "orientation": self.wall.orientation,
                "x": self.wall.x,
                "y": self.wall.y,
            },
        }


@dataclass(frozen=True)
class Candidate:
    action: Action
    score: float
    own_distance: int
    opponent_distance: int
    explanation: str


@dataclass(frozen=True)
class Decision:
    action: Action
    score: float
    explanation: str
    candidates: tuple[Candidate, ...]
    elapsed_ms: float
    timed_out: bool

    def to_dict(self) -> dict:
        return {
            "action": self.action.to_dict(),
            "score": round(self.score, 4),
            "explanation": self.explanation,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "timed_out": self.timed_out,
            "candidates": [
                {
                    "action": candidate.action.to_dict(),
                    "score": round(candidate.score, 4),
                    "own_distance": candidate.own_distance,
                    "opponent_distance": candidate.opponent_distance,
                    "explanation": candidate.explanation,
                }
                for candidate in self.candidates
            ],
        }


def _distances(state: GameState, player: int) -> tuple[int, int]:
    own = shortest_distance(state, player)
    opponent = shortest_distance(state, 1 - player)

    if own is None or opponent is None:
        raise ValueError("Invalid state: a player has no route to goal")

    return own, opponent


def _evaluate(
    before: GameState,
    after: GameState,
    player: int,
    action: Action,
    config: AIConfig,
) -> Candidate:
    own_before, opponent_before = _distances(before, player)
    own_after, opponent_after = _distances(after, player)

    own_progress = own_before - own_after
    opponent_delay = opponent_after - opponent_before

    score = (
        config.own_progress_weight * own_progress
        + config.opponent_delay_weight * opponent_delay
    )

    if action.type == "wall":
        walls_before = before.walls_remaining[player]

        # Dinding adalah sumber daya terbatas. Penalti meningkat
        # ketika persediaan semakin sedikit.
        scarcity = (
            config.wall_scarcity_weight
            * (WALLS_PER_PLAYER - walls_before)
            / WALLS_PER_PLAYER
        )
        score -= config.wall_cost + scarcity

    explanation = (
        f"Progress sendiri: {own_progress:+d}; "
        f"penundaan lawan: {opponent_delay:+d}; "
        f"jarak sendiri: {own_after}; "
        f"jarak lawan: {opponent_after}."
    )

    return Candidate(
        action=action,
        score=score,
        own_distance=own_after,
        opponent_distance=opponent_after,
        explanation=explanation,
    )

def choose_action(
    state: GameState,
    config: AIConfig | None = None,
) -> Decision:
    """
    Select a legal action using one-ply heuristic evaluation.

    This is a baseline AI, not Minimax. It evaluates the resulting state
    after each candidate action but does not search opponent responses.
    """
    config = config or AIConfig()
    validate_state(state)

    if state.winner is not None:
        raise ValueError("Cannot choose an action after the game has ended")

    if config.time_limit <= 0:
        raise ValueError("time_limit must be positive")

    started = perf_counter()
    deadline = started + config.time_limit
    player = state.current_player

    # Establish a legal fallback before doing expensive evaluations.
    legal_moves = legal_pawn_moves(state, player)
    if not legal_moves:
        raise ValueError("No legal pawn moves available")

    fallback_action = Action(type="move", destination=legal_moves[0])
    fallback_state = apply_move(state, legal_moves[0])
    fallback = _evaluate(
        state, fallback_state, player, fallback_action, config
    )

    best = fallback
    evaluated: list[Candidate] = [fallback]
    timed_out = False

    # Evaluate pawn moves first. This ensures we always have a legal
    # fallback even if wall enumeration consumes the time budget.
    actions = [
        Action(type="move", destination=destination)
        for destination in legal_moves
        if destination != legal_moves[0]
    ]

    for action in actions:
        if perf_counter() >= deadline:
            timed_out = True
            break

        assert action.destination is not None
        next_state = apply_move(state, action.destination)
        candidate = _evaluate(
            state, next_state, player, action, config
        )
        evaluated.append(candidate)

        if candidate.score > best.score:
            best = candidate

    # A wall is considered only when the player's wall supply permits it.
    if not timed_out and state.walls_remaining[player] > 0:
        for wall in legal_walls(state, player):
            if perf_counter() >= deadline:
                timed_out = True
                break

            action = Action(type="wall", wall=wall)
            next_state = apply_wall(state, wall, player)
            candidate = _evaluate(
                state, next_state, player, action, config
            )
            evaluated.append(candidate)

            if candidate.score > best.score:
                best = candidate

    elapsed_ms = (perf_counter() - started) * 1000

    # Report a compact ranking, not every possible wall.
    ranked = tuple(
        sorted(
            evaluated,
            key=lambda candidate: candidate.score,
            reverse=True,
        )[:5]
    )

    return Decision(
        action=best.action,
        score=best.score,
        explanation=best.explanation,
        candidates=ranked,
        elapsed_ms=elapsed_ms,
        timed_out=timed_out,
    )