from __future__ import annotations

from collections import deque
from functools import lru_cache
from time import perf_counter
from dataclasses import dataclass, replace
from typing import Literal

Position = tuple[int, int]
Orientation = Literal["H", "V"]

BOARD_SIZE = 9
WALLS_PER_PLAYER = 10

import time

PROFILE = {
    "has_path_calls": 0,
    "has_path_time": 0.0,
    "blocked_calls": 0,
    "blocked_time": 0.0,
}


def reset_profile() -> None:
    """Reset engine profiling statistics."""
    PROFILE.update(
        {
            "has_path_calls": 0,
            "has_path_time": 0.0,
            "blocked_calls": 0,
            "blocked_time": 0.0,
        }
    )


@dataclass(frozen=True, order=True)
class Wall:
    orientation: Orientation
    x: int
    y: int

    def in_bounds(self) -> bool:
        return self.orientation in ("H", "V") and 0 <= self.x < 8 and 0 <= self.y < 8


@dataclass(frozen=True)
class GameState:
    pawns: tuple[Position, Position] = ((4, 8), (4, 0))
    walls: frozenset[Wall] = frozenset()
    walls_remaining: tuple[int, int] = (10, 10)
    current_player: int = 0
    winner: int | None = None
    ply: int = 0

    @classmethod
    def initial(cls) -> GameState:
        return cls()

    def to_dict(self) -> dict:
        return {
            "pawns": [{"x": x, "y": y} for x, y in self.pawns],
            "walls": [
                {
                    "orientation": w.orientation,
                    "x": w.x,
                    "y": w.y,
                }
                for w in sorted(self.walls)
            ],
            "walls_remaining": list(self.walls_remaining),
            "current_player": self.current_player,
            "winner": self.winner,
            "ply": self.ply,
        }


def inside(p: Position) -> bool:
    x, y = p
    return 0 <= x < 9 and 0 <= y < 9


def _blocked_impl(
    walls: frozenset[Wall] | set[Wall],
    a: Position,
    b: Position,
) -> bool:
    """Check whether an orthogonal step between adjacent squares is blocked."""
    ax, ay = a
    bx, by = b

    if abs(ax - bx) + abs(ay - by) != 1:
        raise ValueError("Positions must be orthogonally adjacent")

    if ax == bx:
        boundary_y = min(ay, by)

        return any(
            w.orientation == "H" and w.y == boundary_y and w.x <= ax <= w.x + 1
            for w in walls
        )

    boundary_x = min(ax, bx)

    return any(
        w.orientation == "V" and w.x == boundary_x and w.y <= ay <= w.y + 1
        for w in walls
    )


def blocked(
    walls: frozenset[Wall] | set[Wall],
    a: Position,
    b: Position,
) -> bool:
    """Check whether a step is blocked and record profiling statistics."""
    started = perf_counter()
    PROFILE["blocked_calls"] += 1

    try:
        return _blocked_impl(walls, a, b)
    finally:
        PROFILE["blocked_time"] += perf_counter() - started


def shortest_path(
    state: GameState,
    player: int,
) -> list[Position] | None:
    """Find a shortest path to the player's target row using BFS."""
    if player not in (0, 1):
        raise ValueError("Player must be 0 or 1")

    start = state.pawns[player]
    goal_y = 0 if player == 0 else 8

    queue = deque([start])
    previous: dict[Position, Position | None] = {start: None}

    while queue:
        current = queue.popleft()

        if current[1] == goal_y:
            path = []
            node: Position | None = current

            while node is not None:
                path.append(node)
                node = previous[node]

            return list(reversed(path))

        x, y = current

        for nxt in (
            (x, y - 1),
            (x + 1, y),
            (x, y + 1),
            (x - 1, y),
        ):
            if not inside(nxt) or nxt in previous:
                continue

            if blocked(state.walls, current, nxt):
                continue

            previous[nxt] = current
            queue.append(nxt)

    return None


@lru_cache(maxsize=32768)
def _has_path_cached(
    walls: frozenset[Wall],
    start: Position,
    goal_y: int,
) -> bool:
    """
    Cache hasil pemeriksaan keterjangkauan.

    Hasil hanya bergantung pada:
    - Konfigurasi dinding.
    - Posisi awal pion.
    - Baris tujuan.

    Posisi pion lawan dan jumlah dinding tersisa tidak
    memengaruhi pemeriksaan jalur pada implementasi engine ini.
    """
    queue = deque([start])
    visited = {start}

    while queue:
        x, y = queue.popleft()

        if y == goal_y:
            return True

        current = (x, y)

        for nxt in (
            (x, y - 1),
            (x + 1, y),
            (x, y + 1),
            (x - 1, y),
        ):
            if not inside(nxt) or nxt in visited:
                continue

            if blocked(walls, current, nxt):
                continue

            visited.add(nxt)
            queue.append(nxt)

    return False


def _has_path_impl(state: GameState, player: int) -> bool:
    """Check reachability using cached BFS results."""
    if player not in (0, 1):
        raise ValueError("Player must be 0 or 1")

    start = state.pawns[player]
    goal_y = 0 if player == 0 else 8

    return _has_path_cached(
        state.walls,
        start,
        goal_y,
    )


def has_path(state: GameState, player: int) -> bool:
    """Check reachability and record profiling statistics."""
    started = perf_counter()
    PROFILE["has_path_calls"] += 1

    try:
        return _has_path_impl(state, player)
    finally:
        PROFILE["has_path_time"] += perf_counter() - started


def shortest_distance(
    state: GameState,
    player: int,
) -> int | None:
    """Hitung jarak minimum tanpa merekonstruksi jalur."""
    if player not in (0, 1):
        raise ValueError("Player must be 0 or 1")

    start = state.pawns[player]
    goal_y = 0 if player == 0 else 8

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current, distance = queue.popleft()

        if current[1] == goal_y:
            return distance

        x, y = current

        for nxt in (
            (x, y - 1),
            (x + 1, y),
            (x, y + 1),
            (x - 1, y),
        ):
            if not inside(nxt) or nxt in visited:
                continue

            if blocked(state.walls, current, nxt):
                continue

            visited.add(nxt)
            queue.append((nxt, distance + 1))

    return None


def legal_pawn_moves(
    state: GameState,
    player: int | None = None,
) -> list[Position]:
    """Generate legal normal moves, straight jumps and diagonal jumps."""
    if player is None:
        player = state.current_player

    if player not in (0, 1):
        raise ValueError("Player must be 0 or 1")

    if state.winner is not None:
        return []

    own = state.pawns[player]
    opponent = state.pawns[1 - player]
    moves: set[Position] = set()

    directions = (
        (0, -1),
        (1, 0),
        (0, 1),
        (-1, 0),
    )

    for dx, dy in directions:
        adjacent = (own[0] + dx, own[1] + dy)

        if not inside(adjacent):
            continue

        if blocked(state.walls, own, adjacent):
            continue

        if adjacent != opponent:
            moves.add(adjacent)
            continue

        # The opponent is adjacent. Try jumping straight over them.
        behind = (opponent[0] + dx, opponent[1] + dy)

        if inside(behind) and not blocked(state.walls, opponent, behind):
            moves.add(behind)
            continue

        # If a straight jump is impossible, try both diagonal sides.
        for sx, sy in ((-dy, dx), (dy, -dx)):
            diagonal = (
                opponent[0] + sx,
                opponent[1] + sy,
            )

            if inside(diagonal) and not blocked(state.walls, opponent, diagonal):
                moves.add(diagonal)

    return sorted(moves, key=lambda p: (p[1], p[0]))


def walls_overlap_or_cross(
    existing: frozenset[Wall],
    candidate: Wall,
) -> bool:
    x = candidate.x
    y = candidate.y

    if candidate.orientation == "H":
        # Dinding horizontal tidak boleh bertumpuk
        # dengan dinding horizontal di posisi bersebelahan.
        if Wall("H", x - 1, y) in existing:
            return True

        if Wall("H", x + 1, y) in existing:
            return True

        # Dinding horizontal tidak boleh bersilangan
        # dengan dinding vertikal di titik yang sama.
        if Wall("V", x, y) in existing:
            return True

    else:
        # Dinding vertikal tidak boleh bertumpuk
        # dengan dinding vertikal di posisi bersebelahan.
        if Wall("V", x, y - 1) in existing:
            return True

        if Wall("V", x, y + 1) in existing:
            return True

        # Dinding vertikal tidak boleh bersilangan
        # dengan dinding horizontal di titik yang sama.
        if Wall("H", x, y) in existing:
            return True

    return False


def is_legal_wall(
    state: GameState,
    wall: Wall,
    player: int | None = None,
) -> bool:
    if player is None:
        player = state.current_player

    if player not in (0, 1) or state.winner is not None:
        return False

    if not wall.in_bounds():
        return False

    if state.walls_remaining[player] <= 0:
        return False

    if walls_overlap_or_cross(state.walls, wall):
        return False

    candidate = replace(
        state,
        walls=state.walls | {wall},
    )

    return has_path(candidate, 0) and has_path(candidate, 1)

def _apply_legal_wall(
    state: GameState,
    wall: Wall,
    player: int | None = None,
) -> GameState:
    """Apply a wall already verified as legal by the caller."""
    if player is None:
        player = state.current_player

    if state.winner is not None:
        raise ValueError("Game is already over")

    if player != state.current_player:
        raise ValueError("It is not this player's turn")

    remaining = list(state.walls_remaining)
    remaining[player] -= 1

    return replace(
        state,
        walls=state.walls | {wall},
        walls_remaining=tuple(remaining),
        current_player=1 - player,
        ply=state.ply + 1,
    )


def generate_wall_candidates(state: GameState, player: int) -> list[Wall]:
    result = []

    for orientation in ("H", "V"):
        for y in range(8):
            for x in range(8):
                wall = Wall(orientation, x, y)

                if is_legal_wall(state, wall, player):
                    result.append(wall)

    return result


def legal_walls(
    state: GameState,
    player: int | None = None,
) -> list[Wall]:
    if player is None:
        player = state.current_player

    if (
        player not in (0, 1)
        or state.winner is not None
        or state.walls_remaining[player] <= 0
    ):
        return []

    result = []

    for orientation in ("H", "V"):
        for y in range(8):
            for x in range(8):
                wall = Wall(orientation, x, y)

                if is_legal_wall(state, wall, player):
                    result.append(wall)

    return result


def apply_move(
    state: GameState,
    destination: Position,
    player: int | None = None,
) -> GameState:
    if player is None:
        player = state.current_player

    if state.winner is not None:
        raise ValueError("Game is already over")

    if player != state.current_player:
        raise ValueError("It is not this player's turn")

    if destination not in legal_pawn_moves(state, player):
        raise ValueError(f"Illegal pawn move: {destination}")

    pawns = list(state.pawns)
    pawns[player] = destination

    goal_y = 0 if player == 0 else 8
    winner = player if destination[1] == goal_y else None

    return replace(
        state,
        pawns=tuple(pawns),
        current_player=1 - player,
        winner=winner,
        ply=state.ply + 1,
    )


def apply_wall(
    state: GameState,
    wall: Wall,
    player: int | None = None,
) -> GameState:
    if player is None:
        player = state.current_player

    if state.winner is not None:
        raise ValueError("Game is already over")

    if player != state.current_player:
        raise ValueError("It is not this player's turn")

    legal = is_legal_wall(state, wall, player)
    if not legal:
        raise ValueError(f"Illegal wall placement: {wall}")

    remaining = list(state.walls_remaining)
    remaining[player] -= 1

    return replace(
        state,
        walls=state.walls | {wall},
        walls_remaining=tuple(remaining),
        current_player=1 - player,
        ply=state.ply + 1,
    )


def validate_state(state: GameState) -> None:
    if len(state.pawns) != 2:
        raise ValueError("Exactly two pawns are required")

    if any(not inside(p) for p in state.pawns):
        raise ValueError("Pawn outside board")

    if state.pawns[0] == state.pawns[1]:
        raise ValueError("Pawns cannot occupy the same square")

    if state.current_player not in (0, 1):
        raise ValueError("Invalid current player")

    if any(
        not isinstance(n, int) or not 0 <= n <= WALLS_PER_PLAYER
        for n in state.walls_remaining
    ):
        raise ValueError("Invalid remaining wall count")

    if any(not wall.in_bounds() for wall in state.walls):
        raise ValueError("Wall outside board")

    ordered = sorted(state.walls)

    for i, wall in enumerate(ordered):
        if walls_overlap_or_cross(frozenset(ordered[:i]), wall):
            raise ValueError("Walls overlap or cross")

    if any(shortest_path(state, player) is None for player in (0, 1)):
        raise ValueError("A player has no path to their goal")


def has_path_cache_info() -> dict:
    """Return BFS cache statistics."""
    info = _has_path_cached.cache_info()

    return {
        "hits": info.hits,
        "misses": info.misses,
        "current_size": info.currsize,
        "max_size": info.maxsize,
    }


def clear_has_path_cache() -> None:
    """Clear cached BFS results."""
    _has_path_cached.cache_clear()
