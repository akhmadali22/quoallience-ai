from __future__ import annotations

import math
import os
import time
from datetime import datetime
from pathlib import Path

from dataclasses import dataclass
from typing import Literal
from collections import defaultdict, deque

from quoridor.engine import (
    GameState,
    Wall,
    apply_move,
    apply_wall,
    legal_pawn_moves,
    shortest_distance,
    shortest_path,
    is_legal_wall,
    blocked,
)

ActionType = Literal["move", "wall"]

MATE_SCORE = 10_000.0
INFINITY = math.inf


class SearchTimeout(Exception):
    """Raised when the search reaches its time budget."""


@dataclass(frozen=True)
class Action:
    kind: ActionType
    destination: tuple[int, int] | None = None
    wall: Wall | None = None

    def to_dict(self) -> dict:
        if self.kind == "move":
            return {
                "type": "move",
                "destination": list(self.destination),
            }

        return {
            "type": "wall",
            "orientation": self.wall.orientation,
            "x": self.wall.x,
            "y": self.wall.y,
        }


@dataclass(frozen=True)
class AIConfig:
    time_limit: float = 55
    max_depth: int = 5

    progress_weight: float = 2.0
    wall_resource_weight: float = 0.25

    # max_wall_candidates: int = 24
    max_wall_candidates: int = 8
    transposition_table_limit: int = 50_000

    own_progress_weight: float = 2.5
    opponent_delay_weight: float = 1.5

    # Reserve a small part of the opponent's immediate reply search for
    # walls that attack our second-best routes, not only the current shortest path.
    predictive_path_count: int = 3
    predictive_wall_candidates: int = 2

    # Softly discourage returning to a recently visited pawn square when
    # doing so does not improve the shortest route.
    repetition_penalty: float = 3.5

    # Extra urgency when the opponent can finish in as few moves as we can.
    opponent_threat_weight: float = 1.5


@dataclass
class SearchResult:
    action: Action
    score: float
    depth: int
    elapsed_ms: float
    timed_out: bool
    nodes: int

    def to_dict(self) -> dict:
        return {
            "action": self.action.to_dict(),
            "score": self.score,
            "depth": self.depth,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "timed_out": self.timed_out,
            "nodes": self.nodes,
        }


@dataclass(frozen=True)
class TTEntry:
    depth: int
    score: float
    flag: Literal["EXACT", "LOWER", "UPPER"]
    best_action: Action | None = None


class QuoridorAI:
    """
    Advanced Quoridor AI.

    Features:
    - Minimax
    - Alpha-beta pruning
    - Iterative deepening
    - Transposition table
    - Candidate wall pruning
    - Configurable search deadline

    The evaluation score is always calculated from the root
    player's perspective.
    """

    def __init__(self, config: AIConfig | None = None):
        self.config = config or AIConfig()

        self.deadline = 0.0
        self.root_player = 0
        self.root_ply = 0
        self.nodes = 0

        # Exact results from fully searched Minimax nodes.
        self.transposition_table: dict[tuple, TTEntry] = {}

        # Cache path distances and static position evaluations.
        self.distance_cache: dict[tuple, int | None] = {}
        self.evaluation_cache: dict[tuple, float] = {}

        # Basic performance counters.
        self.distance_cache_hits = 0
        self.evaluation_cache_hits = 0

        # Real positions observed on this AI instance's actual turns only.
        # Search-tree positions are deliberately never added to this history.
        self.actual_state_history: list[tuple[int, tuple]] = []
        self.actual_pawn_history: list[tuple[int, tuple[int, int], int | None]] = []

        self.profile_calls = defaultdict(int)
        self.profile_time = defaultdict(float)

        self.stats = {
            "children_generated": 0,
            "move_candidates": 0,
            "wall_candidates": 0,
            "predictive_wall_candidates": 0,
            "alpha_beta_cutoffs": 0,
        }

    def _profile_start(self, name: str) -> float:
        self.profile_calls[name] += 1
        return time.perf_counter()

    def debug_position(self, state: GameState) -> dict:
        """Menampilkan jarak BFS dan informasi posisi tanpa batas waktu pencarian AI."""
        player = state.current_player
        opponent = 1 - player

        # Hitung BFS secara langsung agar tidak terkena SearchTimeout
        own_distance = shortest_distance(state, player)
        opponent_distance = shortest_distance(state, opponent)

        return {
            "current_player": player,
            "pawns": state.pawns,
            "walls_remaining": state.walls_remaining,
            "own_distance": own_distance,
            "opponent_distance": opponent_distance,
            "legal_moves": legal_pawn_moves(state, player),
        }

    def _profile_end(self, name: str, start: float) -> None:
        self.profile_time[name] += time.perf_counter() - start

    def profile_report(self) -> dict:
        return {
            name: {
                "calls": self.profile_calls[name],
                "total_ms": round(self.profile_time[name] * 1000, 3),
                "average_ms": round(
                    self.profile_time[name] * 1000 / max(1, self.profile_calls[name]),
                    4,
                ),
            }
            for name in sorted(self.profile_calls)
        }

    def _position_key(self, state: GameState) -> tuple:
        """
        Return a key containing all information that affects
        path distance and static evaluation.
        """
        return self._state_key(state)

    def _check_deadline(self) -> None:
        if time.monotonic() >= self.deadline:
            raise SearchTimeout

    def _cached_distance(
        self,
        state: GameState,
        player: int,
    ) -> int | None:
        self._check_deadline()

        key = (state.pawns[player], state.walls, player)

        try:
            distance = self.distance_cache[key]
        except KeyError:
            distance = shortest_distance(state, player)
            self.distance_cache[key] = distance
            return distance
        else:
            self.distance_cache_hits += 1
            return distance

    @staticmethod
    def _state_key(state: GameState) -> tuple:
        """
        Create a stable key for equivalent game states.

        Ply is included because terminal scores prefer faster wins
        and slower losses.
        """
        pawns = tuple(tuple(position) for position in state.pawns)

        walls = tuple(
            sorted((wall.orientation, wall.x, wall.y) for wall in state.walls)
        )

        return (
            pawns,
            walls,
            tuple(state.walls_remaining),
            state.current_player,
            state.winner,
            state.ply,
        )

    @staticmethod
    def _repetition_key(state: GameState) -> tuple:
        """Position identity without ply, so the same board can be detected later."""
        return QuoridorAI._state_key(state)[:-1]

    def _record_actual_position(self, state: GameState) -> None:
        """Record one real AI-turn position, not simulated minimax nodes."""
        key = self._repetition_key(state)

        if self.actual_state_history:
            last_ply, last_key = self.actual_state_history[-1]
            if state.ply == last_ply and key == last_key:
                # Avoid counting a retried API request as another occurrence.
                return
            if state.ply < last_ply:
                # Defensive reset if this AI object is reused for a new game.
                self.actual_state_history.clear()
                self.actual_pawn_history.clear()
            elif state.ply == last_ply:
                return

        player = state.current_player
        distance = shortest_distance(state, player)
        self.actual_state_history.append((state.ply, key))
        self.actual_pawn_history.append((state.ply, state.pawns[player], distance))

        # A short rolling history is enough to catch oscillation without
        # penalizing a pawn for squares visited many turns ago.
        self.actual_state_history = self.actual_state_history[-16:]
        self.actual_pawn_history = self.actual_pawn_history[-16:]

    def _repetition_penalty(
        self,
        state: GameState,
        own_distance: int | None,
    ) -> float:
        """Penalize repeated AI-turn positions and unproductive pawn oscillation."""
        if own_distance is None:
            return 0.0

        prior_states = [
            key
            for ply, key in self.actual_state_history
            if ply < state.ply and key == self._repetition_key(state)
        ]
        exact_penalty = (
            self.config.repetition_penalty
            * 0.5
            * min(len(prior_states), 2)
        )

        current_pawn = state.pawns[self.root_player]
        prior_distances = [
            distance
            for ply, position, distance in self.actual_pawn_history
            if ply < state.ply
            and position == current_pawn
            and distance is not None
        ]

        pawn_penalty = 0.0
        if prior_distances and own_distance >= min(prior_distances):
            pawn_penalty = (
                self.config.repetition_penalty
                * min(len(prior_distances), 2)
            )

        return exact_penalty + pawn_penalty

    def _evaluate(self, state: GameState) -> float:
        """
        Static evaluation from the root player's perspective.

        Higher scores favor the root player.
        """
        self._check_deadline()

        if state.winner is not None:
            if state.winner == self.root_player:
                return MATE_SCORE

            return -MATE_SCORE

        key = self._position_key(state)

        cached = self.evaluation_cache.get(key)

        if cached is not None:
            self.evaluation_cache_hits += 1
            return cached

        opponent = 1 - self.root_player

        own_distance = self._cached_distance(
            state,
            self.root_player,
        )

        opponent_distance = self._cached_distance(
            state,
            opponent,
        )

        if own_distance is None:
            score = -MATE_SCORE

        elif opponent_distance is None:
            score = MATE_SCORE

        else:
            progress_score = (
                opponent_distance * self.config.opponent_delay_weight
                - own_distance * self.config.own_progress_weight
            )

            own_walls = state.walls_remaining[self.root_player]
            opponent_walls = state.walls_remaining[opponent]

            wall_resource_score = (
                own_walls - opponent_walls
            ) * self.config.wall_resource_weight

            # Do not reward forward movement blindly when the opponent is
            # already at least as close to their goal as we are.
            opponent_threat_penalty = (
                self.config.opponent_threat_weight
                * max(0, own_distance - opponent_distance + 1)
            )

            repetition_penalty = self._repetition_penalty(
                state,
                own_distance,
            )

            score = (
                progress_score
                + wall_resource_score
                - opponent_threat_penalty
                - repetition_penalty
            )

        self.evaluation_cache[key] = score

        return score

    @staticmethod
    def _wall_candidates_from_path(
        path: list[tuple[int, int]],
    ) -> set[Wall]:
        """
        Generate walls that could block edges on a shortest path.

        H wall:
            blocks vertical movement across a row boundary.

        V wall:
            blocks horizontal movement across a column boundary.
        """
        candidates: set[Wall] = set()

        if not path or len(path) < 2:
            return candidates

        for first, second in zip(path, path[1:]):
            x1, y1 = first
            x2, y2 = second

            if x1 == x2 and abs(y1 - y2) == 1:
                # Vertical pawn movement: consider horizontal walls.
                boundary_y = min(y1, y2)

                for wall_x in (x1 - 1, x1):
                    if 0 <= wall_x <= 7:
                        candidates.add(
                            Wall(
                                orientation="H",
                                x=wall_x,
                                y=boundary_y,
                            )
                        )

            elif y1 == y2 and abs(x1 - x2) == 1:
                # Horizontal pawn movement: consider vertical walls.
                boundary_x = min(x1, x2)

                for wall_y in (y1 - 1, y1):
                    if 0 <= wall_y <= 7:
                        candidates.add(
                            Wall(
                                orientation="V",
                                x=boundary_x,
                                y=wall_y,
                            )
                        )

        return candidates

    def _alternative_paths(
        self,
        state: GameState,
        player: int,
        max_paths: int = 3,
    ) -> list[list[tuple[int, int]]]:
        """
        Cari beberapa jalur berbeda dengan menghindari setiap edge
        pada jalur terpendek awal.

        Jalur alternatif tidak harus sama panjang dengan jalur awal.
        """
        base_path = shortest_path(state, player)

        if not base_path:
            return []

        paths = [base_path]
        seen = {tuple(base_path)}
        goal_y = 0 if player == 0 else 8
        start = state.pawns[player]

        for index in range(len(base_path) - 1):
            self._check_deadline()
            if len(paths) >= max_paths:
                break

            first = base_path[index]
            second = base_path[index + 1]
            forbidden_edge = frozenset((first, second))

            queue = deque([start])
            previous = {start: None}

            while queue:
                self._check_deadline()
                current = queue.popleft()

                if current[1] == goal_y:
                    path = []
                    node = current

                    while node is not None:
                        path.append(node)
                        node = previous[node]

                    path.reverse()
                    path_key = tuple(path)

                    if path_key not in seen:
                        seen.add(path_key)
                        paths.append(path)

                    break

                x, y = current

                for nxt in (
                    (x, y - 1),
                    (x + 1, y),
                    (x, y + 1),
                    (x - 1, y),
                ):
                    if not (0 <= nxt[0] < 9 and 0 <= nxt[1] < 9):
                        continue

                    if nxt in previous:
                        continue

                    edge = frozenset((current, nxt))

                    if edge == forbidden_edge:
                        continue

                    if blocked(state.walls, current, nxt):
                        continue

                    previous[nxt] = current
                    queue.append(nxt)

        return paths

    def _candidate_walls(
        self,
        state: GameState,
    ) -> list[Wall]:
        """
        Generate useful walls, including a small set that threatens our
        alternative routes on opponent turns within the search.

        The opponent normally considers walls intersecting our current
        shortest path. That is insufficient when we reroute around one wall
        but lose if a later wall closes the reroute as well. Reserve a few
        candidates for alternative paths so minimax can see that sequence.
        """
        player = state.current_player

        if state.walls_remaining[player] <= 0:
            return []

        self._check_deadline()

        started = self._profile_start("shortest_path")
        own_path = shortest_path(state, player)
        self._profile_end("shortest_path", started)

        self._check_deadline()

        started = self._profile_start("shortest_path")
        opponent_path = shortest_path(state, 1 - player)
        self._profile_end("shortest_path", started)

        own_path_walls = self._wall_candidates_from_path(own_path or [])
        opponent_path_walls = self._wall_candidates_from_path(opponent_path or [])

        # Keep predictive wall candidates on opponent replies as before, but
        # also generate them at the root when the opponent is close enough to
        # threaten our race. Waiting until the opponent's minimax turn can be
        # too late: the root shortlist may otherwise omit the wall that
        # preserves our lead.
        own_distance = shortest_distance(state, player)
        opponent_distance = shortest_distance(state, 1 - player)
        is_root_turn = player == self.root_player and state.ply == self.root_ply
        urgent_root_defense = (
            is_root_turn
            and own_distance is not None
            and opponent_distance is not None
            and (
                opponent_distance <= own_distance + 2
                or opponent_distance <= 4
            )
            and self.config.predictive_path_count > 1
            and self.config.predictive_wall_candidates > 0
        )
        is_opponent_reply = (
            player != self.root_player
            and state.ply > self.root_ply
            and (state.ply - self.root_ply) % 2 == 1
            and self.config.predictive_path_count > 1
            and self.config.predictive_wall_candidates > 0
        )
        use_predictive_walls = is_opponent_reply or urgent_root_defense

        predicted_path_walls: set[Wall] = set()
        if use_predictive_walls:
            # On opponent turns, attack our own alternate routes. On an urgent
            # root turn, attack the opponent's alternate routes before they
            # become the only viable response.
            target_player = self.root_player if is_opponent_reply else 1 - player
            predictive_paths = self._alternative_paths(
                state,
                target_player,
                max_paths=max(2, self.config.predictive_path_count),
            )
            for path in predictive_paths[1:]:
                predicted_path_walls.update(
                    self._wall_candidates_from_path(path)
                )

        # Validate all candidates before ranking. For urgent root defense,
        # rank legal walls by the actual distance change they cause, rather
        # than assuming every wall touching a shortest path is equally useful.
        candidates = own_path_walls | opponent_path_walls | predicted_path_walls
        legal_candidates: list[Wall] = []
        for wall in candidates:
            self._check_deadline()
            if is_legal_wall(state, wall, player):
                legal_candidates.append(wall)

        def wall_priority(wall: Wall) -> tuple:
            if urgent_root_defense:
                wall_state = state.__class__(
                    pawns=state.pawns,
                    walls=state.walls | {wall},
                    walls_remaining=state.walls_remaining,
                    current_player=state.current_player,
                    winner=state.winner,
                    ply=state.ply,
                )
                new_own_distance = shortest_distance(wall_state, player)
                new_opponent_distance = shortest_distance(wall_state, 1 - player)
                own_detour = (
                    99 if new_own_distance is None or own_distance is None
                    else new_own_distance - own_distance
                )
                opponent_delay = (
                    -99 if new_opponent_distance is None or opponent_distance is None
                    else new_opponent_distance - opponent_distance
                )
                # Prefer delaying the opponent, then avoid damaging our own
                # route. Path membership breaks ties deterministically.
                return (
                    -opponent_delay,
                    own_detour,
                    wall not in predicted_path_walls,
                    wall not in opponent_path_walls,
                    wall.orientation,
                    wall.x,
                    wall.y,
                )
            return (
                wall not in opponent_path_walls,
                wall not in predicted_path_walls,
                wall in own_path_walls,
                wall.orientation,
                wall.x,
                wall.y,
            )

        ordered = sorted(legal_candidates, key=wall_priority)
        limit = max(0, self.config.max_wall_candidates)
        if len(ordered) <= limit:
            selected = ordered
        elif use_predictive_walls:
            predictive_legal = [
                wall for wall in ordered
                if wall in predicted_path_walls and wall not in opponent_path_walls
            ]
            predictive_limit = min(
                self.config.predictive_wall_candidates,
                limit,
                len(predictive_legal),
            )
            direct = [
                wall for wall in ordered
                if wall not in predicted_path_walls or wall in opponent_path_walls
            ]
            selected = direct[: limit - predictive_limit]
            selected.extend(predictive_legal[:predictive_limit])
            # Re-sort after reserving slots so urgent root defense retains
            # the distance-based ranking among all selected candidates.
            selected.sort(key=wall_priority)
        else:
            selected = ordered[:limit]

        self.stats.setdefault("predictive_wall_candidates", 0)
        self.stats["predictive_wall_candidates"] += sum(
            wall in predicted_path_walls for wall in selected
        )
        self.stats.setdefault("root_defensive_wall_candidates", 0)
        if urgent_root_defense:
            self.stats["root_defensive_wall_candidates"] += sum(
                wall in predicted_path_walls for wall in selected
            )

        return selected

    def _generate_children(
        self,
        state: GameState,
    ) -> list[tuple[Action, GameState]]:
        """Generate legal successor states."""
        self._check_deadline()

        children: list[tuple[Action, GameState]] = []
        player = state.current_player

        # Pawn moves
        moves = legal_pawn_moves(state, player)
        self.stats["move_candidates"] += len(moves)

        for destination in moves:
            self._check_deadline()

            try:
                next_state = apply_move(state, destination)
            except (ValueError, AssertionError):
                continue

            children.append(
                (
                    Action(
                        kind="move",
                        destination=tuple(destination),
                    ),
                    next_state,
                )
            )

        # Wall candidates
        wall_candidates = self._candidate_walls(state)
        self.stats["wall_candidates"] += len(wall_candidates)

        for wall in wall_candidates:
            self._check_deadline()

            started = self._profile_start("apply_wall")

            try:
                next_state = apply_wall(state, wall)
            except (ValueError, AssertionError):
                continue
            finally:
                self._profile_end("apply_wall", started)

            children.append(
                (
                    Action(kind="wall", wall=wall),
                    next_state,
                )
            )

        self.stats["children_generated"] += len(children)

        return children

    def _ordered_children(
        self,
        children,
        maximizing: bool,
        preferred_action=None,
    ):
        """
        Urutkan kandidat menggunakan heuristik murah.

        Prioritas:
        1. Langkah terbaik dari transposition table.
        2. Langkah pion yang lebih dekat ke garis finis.
        3. Dinding, dengan mempertahankan urutan kandidat awal.
        """
        if not children:
            return []

        moves = []
        walls = []

        for action, child_state in children:
            if action.kind == "move":
                moves.append((action, child_state))
            else:
                walls.append((action, child_state))

        # Identifikasi pemain yang baru saja bergerak.
        # Engine mengganti current_player setelah langkah diterapkan.
        def move_order(item):
            action, child_state = item
            mover = 1 - child_state.current_player
            _, destination_y = action.destination

            # P0 bergerak menuju y=0.
            # P1 bergerak menuju y=8.
            if mover == 0:
                return (destination_y, action.destination[0])

            return (-destination_y, action.destination[0])

        # Urutkan langkah pion berdasarkan kemajuan menuju garis finis.
        # Tidak memanggil BFS atau _evaluate().
        moves.sort(key=move_order)

        # Urutan dinding dari _candidate_walls() dipertahankan.
        # Kandidat itu sudah diprioritaskan berdasarkan jalur lawan.
        ordered = moves + walls

        # Langkah terbaik dari transposition table selalu diutamakan.
        if preferred_action is not None:
            for index, (action, _) in enumerate(ordered):
                if action == preferred_action:
                    if index > 0:
                        ordered.insert(0, ordered.pop(index))
                    break

        return ordered

    def _minimax(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        self._check_deadline()
        self.nodes += 1

        if state.winner is not None:
            if state.winner == self.root_player:
                return MATE_SCORE + depth

            return -MATE_SCORE - depth

        if depth <= 0:
            return self._evaluate(state)

        # Satu key untuk setiap posisi.
        # Kedalaman pencarian disimpan di dalam TTEntry.
        key = self._state_key(state)

        original_alpha = alpha
        original_beta = beta

        # Periksa transposition table sebelum menghasilkan anak.
        entry = self.transposition_table.get(key)

        if entry is not None and entry.depth >= depth:
            if entry.flag == "EXACT":
                return entry.score

            if entry.flag == "LOWER":
                alpha = max(alpha, entry.score)

            elif entry.flag == "UPPER":
                beta = min(beta, entry.score)

            if alpha >= beta:
                return entry.score

        maximizing = state.current_player == self.root_player

        children = self._generate_children(state)

        if not children:
            return self._evaluate(state)

        children = self._ordered_children(
            children,
            maximizing=maximizing,
            preferred_action=(entry.best_action if entry is not None else None),
        )

        best_score = -INFINITY if maximizing else INFINITY
        best_action = None

        for action, child_state in children:
            self._check_deadline()

            score = self._minimax(
                child_state,
                depth - 1,
                alpha,
                beta,
            )

            if maximizing:
                if score > best_score:
                    best_score = score
                    best_action = action

                alpha = max(alpha, best_score)

            else:
                if score < best_score:
                    best_score = score
                    best_action = action

                beta = min(beta, best_score)

            if alpha >= beta:
                self.stats["alpha_beta_cutoffs"] += 1
                break

        # Tentukan jenis hasil berdasarkan window pencarian awal.
        if best_score <= original_alpha:
            flag = "UPPER"
        elif best_score >= original_beta:
            flag = "LOWER"
        else:
            flag = "EXACT"

        # Simpan hasil yang sudah selesai dihitung.
        # Jangan pernah menyimpan nilai awal -inf/+inf.
        existing = self.transposition_table.get(key)

        if existing is None or depth >= existing.depth or flag == "EXACT":
            if (
                existing is not None
                or len(self.transposition_table) < self.config.transposition_table_limit
            ):
                self.transposition_table[key] = TTEntry(
                    depth=depth,
                    score=best_score,
                    flag=flag,
                    best_action=best_action,
                )

        return best_score

    @staticmethod
    def _format_action(action: Action) -> str:
        if action.kind == "move":
            return f"move->{action.destination}"
        wall = action.wall
        return f"wall-{wall.orientation}@({wall.x},{wall.y})"

    def _debug_enabled(self) -> bool:
        return os.getenv("QUORIDOR_AI_DEBUG", "").strip().lower() in {
            "1", "true", "yes", "on"
        }

    def _debug_log(self, message: str) -> None:
        if not self._debug_enabled():
            return

        # Write diagnostics to a dedicated text file because API/server processes
        # may not forward stdout to the terminal that launched the game.
        log_path = Path(
            os.getenv("QUORIDOR_AI_DEBUG_FILE", "quoridor_ai_debug.txt")
        ).expanduser()
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        line = f"{timestamp} [AI-DEBUG] {message}\n"

        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(line)
        except OSError as exc:
            # Keep a visible fallback if the configured log path is not writable.
            print(
                f"[AI-DEBUG] Gagal menulis {log_path}: {exc}; pesan: {message}",
                flush=True,
            )

    def _search_root(
        self,
        state: GameState,
        depth: int,
        fallback: Action,
    ) -> tuple[Action, float]:
        self._check_deadline()

        children = self._generate_children(state)

        if not children:
            return fallback, self._evaluate(state)

        children = self._ordered_children(
            children,
            maximizing=True,
        )

        best_action = fallback
        best_score = -INFINITY
        alpha = -INFINITY
        beta = INFINITY
        root_scores: list[tuple[Action, float]] = []

        for action, child_state in children:
            self._check_deadline()

            score = self._minimax(
                child_state,
                depth - 1,
                alpha,
                beta,
            )
            root_scores.append((action, score))

            if score > best_score:
                best_score = score
                best_action = action

            alpha = max(alpha, best_score)

        if self._debug_enabled():
            ranked = sorted(root_scores, key=lambda item: item[1], reverse=True)
            details = " | ".join(
                f"{self._format_action(action)}={score:.2f}"
                for action, score in ranked[:8]
            )
            self._debug_log(
                f"ROOT depth={depth} candidates={len(root_scores)} "
                f"best={self._format_action(best_action)} score={best_score:.2f}; "
                f"top: {details}"
            )

        return best_action, best_score

    def _fallback_action(self, state: GameState) -> Action:
        """
        Find a legal pawn move to use if the search times out.

        Prefer the move with the lowest remaining path distance.
        """
        player = state.current_player
        moves = legal_pawn_moves(state, player)

        if not moves:
            raise ValueError("Tidak ada langkah legal untuk posisi ini.")

        best_move = moves[0]
        best_distance = INFINITY

        for move in moves:
            try:
                next_state = apply_move(state, move)
                distance = shortest_distance(next_state, player)
            except (ValueError, AssertionError):
                continue

            if distance < best_distance:
                best_distance = distance
                best_move = move

        return Action(
            kind="move",
            destination=tuple(best_move),
        )

    def choose_action(self, state: GameState) -> SearchResult:
        """
        Select an action using iterative deepening.

        The last fully completed search depth is retained if a deeper
        iteration times out.
        """
        if state.winner is not None:
            raise ValueError("Permainan sudah selesai. AI tidak dapat bergerak.")

        if self.config.time_limit <= 0:
            raise ValueError("time_limit harus lebih besar dari 0.")

        if self.config.max_depth < 1:
            raise ValueError("max_depth minimal 1.")

        start_time = time.monotonic()
        self.deadline = start_time + self.config.time_limit
        self.root_player = state.current_player
        self.root_ply = state.ply
        self.nodes = 0

        self.transposition_table = {}
        self.distance_cache = {}
        self.evaluation_cache = {}

        self.distance_cache_hits = 0
        self.evaluation_cache_hits = 0

        # Keep actual-turn history across calls; clear only per-search caches.
        self._record_actual_position(state)

        self.stats = {
            "children_generated": 0,
            "move_candidates": 0,
            "wall_candidates": 0,
            "predictive_wall_candidates": 0,
            "alpha_beta_cutoffs": 0,
        }

        # Ensure the AI has a legal action before starting deeper search.
        fallback = self._fallback_action(state)

        if self._debug_enabled():
            player = state.current_player
            opponent = 1 - player
            own_distance = shortest_distance(state, player)
            opponent_distance = shortest_distance(state, opponent)
            goal_y = 0 if player == 0 else 8
            self._debug_log(
                f"START ply={state.ply} player={player} "
                f"pawns={state.pawns} goal_y={goal_y} "
                f"distance={own_distance} opponent_distance={opponent_distance} "
                f"walls={state.walls_remaining} legal_moves={legal_pawn_moves(state, player)} "
                f"wall_count={len(state.walls)} "
                f"repetition_history={len(self.actual_pawn_history)}"
            )

        best_action = fallback
        best_score = self._evaluate(state)
        completed_depth = 0
        timed_out = False

        for depth in range(1, self.config.max_depth + 1):
            try:
                action, score = self._search_root(
                    state,
                    depth,
                    fallback,
                )

                best_action = action
                best_score = score
                completed_depth = depth

            except SearchTimeout:
                timed_out = True
                break

        elapsed_ms = (time.monotonic() - start_time) * 1000

        if self._debug_enabled():
            player = state.current_player
            opponent = 1 - player
            try:
                if best_action.kind == "move":
                    next_state = apply_move(state, best_action.destination)
                else:
                    next_state = apply_wall(state, best_action.wall)
                next_own_distance = shortest_distance(next_state, player)
                next_opponent_distance = shortest_distance(next_state, opponent)
            except (ValueError, AssertionError):
                next_own_distance = None
                next_opponent_distance = None

            self._debug_log(
                f"CHOSEN ply={state.ply} action={self._format_action(best_action)} "
                f"score={best_score:.2f} depth={completed_depth}/{self.config.max_depth} "
                f"elapsed_ms={elapsed_ms:.1f} nodes={self.nodes} timeout={timed_out} "
                f"distance={shortest_distance(state, player)}->{next_own_distance} "
                f"opponent_distance={shortest_distance(state, opponent)}->{next_opponent_distance} "
                f"cache_hits(distance={self.distance_cache_hits},eval={self.evaluation_cache_hits}) "
                f"tt={len(self.transposition_table)} "
                f"repetition_history={len(self.actual_pawn_history)} "
                f"stats={self.stats}"
            )

        return SearchResult(
            action=best_action,
            score=best_score,
            depth=completed_depth,
            elapsed_ms=elapsed_ms,
            timed_out=timed_out,
            nodes=self.nodes,
        )
