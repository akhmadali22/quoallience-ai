from __future__ import annotations

import math
import time
import heapq

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

    max_wall_candidates: int = 8
    max_alternative_paths: int = 2
    transposition_table_limit: int = 50_000

    own_progress_weight: float = 2.5
    opponent_delay_weight: float = 1.5

    route_robustness_weight: float = 0.35
    max_robustness_wall_candidates: int = 8

    route_impact_weight: float = 0.5
    route_flexibility_weight: float = 0.65
    route_probe_depth: int = 3
    route_detour_allowance: int = 2


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
        self.nodes = 0

        # Exact results from fully searched Minimax nodes.
        self.transposition_table: dict[tuple, TTEntry] = {}

        # Cache path distances, goal-distance maps, route flexibility,
        # and static position evaluations. Goal maps are shared by all
        # pawn positions that have the same wall layout.
        self.distance_cache: dict[tuple, int | None] = {}
        self.goal_distance_map_cache: dict[tuple, dict[tuple[int, int], int]] = {}
        self.route_flexibility_cache: dict[tuple, float] = {}
        self.evaluation_cache: dict[tuple, float] = {}

        # Basic performance counters.
        self.distance_cache_hits = 0
        self.evaluation_cache_hits = 0

        self.profile_calls = defaultdict(int)
        self.profile_time = defaultdict(float)

        self.stats = {
            "children_generated": 0,
            "move_candidates": 0,
            "wall_candidates": 0,
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

    def _wall_blocks_path(
        self,
        wall: Wall,
        path: list[tuple[int, int]],
    ) -> bool:
        """Return True if this wall blocks an edge in the path."""
        wall_set = frozenset({wall})

        return any(
            blocked(wall_set, current, nxt)
            for current, nxt in zip(path, path[1:])
        )

    def _immediate_winning_moves(
        self,
        state: GameState,
        player: int,
    ) -> list[tuple[int, int]]:
        """Return legal pawn moves that win immediately for player."""
        if state.winner is not None:
            return []
        goal_y = 0 if player == 0 else 8
        return [
            tuple(destination)
            for destination in legal_pawn_moves(state, player)
            if destination[1] == goal_y
        ]

    def _opponent_has_immediate_win(self, state: GameState) -> bool:
        """State is assumed to be after our action, with opponent to move."""
        if state.winner is not None:
            return False
        return bool(self._immediate_winning_moves(state, state.current_player))

    def _urgent_defensive_walls(self, state: GameState) -> list[Wall]:
        """Find legal walls that block an opponent's one-move winning edge."""
        if state.winner is not None:
            return []
        player = state.current_player
        opponent = 1 - player
        threats = self._immediate_winning_moves(state, opponent)
        if not threats or state.walls_remaining[player] <= 0:
            return []
        opponent_position = state.pawns[opponent]
        urgent: set[Wall] = set()
        for destination in threats:
            for wall in self._wall_candidates_from_path([opponent_position, destination]):
                if is_legal_wall(state, wall, player):
                    urgent.add(wall)
        return sorted(urgent, key=lambda wall: (wall.orientation, wall.x, wall.y))

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

    def _route_robustness(
        self,
        child_state: GameState,
    ) -> float:
        """
        Evaluate how vulnerable the root player's position is
        to the opponent's strongest candidate wall response.

        Returns the worst static evaluation after a legal
        opponent wall placement, from the root player's perspective.
        """
        self._check_deadline()

        if child_state.winner is not None:
            return self._evaluate(child_state)

        opponent = child_state.current_player

        if child_state.walls_remaining[opponent] <= 0:
            return self._evaluate(child_state)

        baseline_score = self._evaluate(child_state)

        candidates = self._candidate_walls(child_state)

        if not candidates:
            return baseline_score

        worst_score = baseline_score
        tested = 0

        for wall in candidates:
            self._check_deadline()

            try:
                response_state = apply_wall(child_state, wall)
            except (ValueError, AssertionError):
                continue

            response_score = self._evaluate(response_state)

            worst_score = min(worst_score, response_score)

            tested += 1

            if tested >= self.config.max_robustness_wall_candidates:
                break

        return worst_score

    def _goal_distance_map(
        self,
        state: GameState,
        player: int,
    ) -> dict[tuple[int, int], int]:
        """Return distances from every square to the player's goal row."""
        self._check_deadline()
        goal_y = 0 if player == 0 else 8
        key = (state.walls, goal_y)
        cached = self.goal_distance_map_cache.get(key)
        if cached is not None:
            return cached

        distances = {(x, goal_y): 0 for x in range(9)}
        queue = deque(distances)

        # Walls block movement symmetrically, so a multi-source BFS from
        # every goal-row square gives distance-to-goal for every cell.
        while queue:
            self._check_deadline()
            current = queue.popleft()
            x, y = current
            for nxt in ((x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)):
                if not (0 <= nxt[0] < 9 and 0 <= nxt[1] < 9):
                    continue
                if nxt in distances or blocked(state.walls, current, nxt):
                    continue
                distances[nxt] = distances[current] + 1
                queue.append(nxt)

        self.goal_distance_map_cache[key] = distances
        return distances

    def _route_flexibility(self, state: GameState, player: int) -> float:
        """
        Estimate near-optimal route options in the first few moves.

        Count distinct simple route prefixes that can still reach the goal
        within a small detour budget. This rewards positions with alternatives
        without running Yen's full alternative-path search at every leaf.
        """
        self._check_deadline()
        goal_y = 0 if player == 0 else 8
        start = state.pawns[player]
        key = (state.walls, start, goal_y)
        cached = self.route_flexibility_cache.get(key)
        if cached is not None:
            return cached

        distances = self._goal_distance_map(state, player)
        shortest = distances.get(start)
        if shortest is None:
            return 0.0

        max_depth = max(1, self.config.route_probe_depth)
        detour_allowance = max(0, self.config.route_detour_allowance)
        visited = {start}

        def explore(position: tuple[int, int], depth: int) -> float:
            self._check_deadline()
            if depth >= max_depth:
                return 0.0

            x, y = position
            total = 0.0
            for nxt in ((x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)):
                if not (0 <= nxt[0] < 9 and 0 <= nxt[1] < 9):
                    continue
                if nxt in visited or blocked(state.walls, position, nxt):
                    continue

                next_distance = distances.get(nxt)
                if next_distance is None:
                    continue

                next_depth = depth + 1
                detour = next_depth + next_distance - shortest
                if detour > detour_allowance:
                    continue

                # Nearby choices matter most; slight detours remain useful,
                # but should not outweigh a genuinely shorter route.
                total += (0.65 ** depth) / (1.0 + detour)
                visited.add(nxt)
                total += explore(nxt, next_depth)
                visited.remove(nxt)

            return total

        raw_options = explore(start, 0)
        flexibility = math.log1p(raw_options)
        self.route_flexibility_cache[key] = flexibility
        return flexibility

    def _evaluate(self, state: GameState) -> float:
        """
        Static strategic evaluation from the root player's perspective.

        The score combines race progress, phase-sensitive wall reserves, and
        near-optimal route flexibility. Higher scores favor the root player.
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

            # Walls have more option value early, when both players still
            # have long races ahead. Their reserve value tapers as the game
            # approaches the goal, while immediate tactical wall effects are
            # still represented by the search itself.
            average_race_distance = (own_distance + opponent_distance) / 2.0
            phase_multiplier = 0.55 + 0.45 * min(8.0, average_race_distance) / 8.0
            wall_resource_score = (
                (own_walls - opponent_walls)
                * self.config.wall_resource_weight
                * phase_multiplier
            )

            own_flexibility = self._route_flexibility(state, self.root_player)
            opponent_flexibility = self._route_flexibility(state, opponent)
            route_flexibility_score = self.config.route_flexibility_weight * (
                own_flexibility - opponent_flexibility
            )

            score = (
                progress_score
                + wall_resource_score
                + route_flexibility_score
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
        Temukan K jalur sederhana terpendek menggunakan Yen's algorithm.

        Jalur diurutkan berdasarkan jumlah langkah. Setiap jalur:
        - Valid terhadap wall yang ada.
        - Tidak mengunjungi kotak yang sama lebih dari sekali.
        - Unik berdasarkan seluruh urutan koordinat.
        - Dicari dengan tetap menghormati deadline AI.
        """
        self._check_deadline()

        if max_paths <= 0:
            return []

        # Jalur terpendek pertama.
        first_path = shortest_path(state, player)

        if not first_path:
            return []

        accepted_paths = [first_path]
        accepted_keys = {tuple(first_path)}

        # Kandidat disimpan sebagai:
        # (jumlah langkah, urutan koordinat).
        candidates = []
        candidate_keys = set()

        start = state.pawns[player]
        goal_y = 0 if player == 0 else 8

        def find_spur_path(
            spur_start: tuple[int, int],
            forbidden_nodes: set[tuple[int, int]],
            forbidden_edges: set[
                tuple[tuple[int, int], tuple[int, int]]
            ],
        ) -> list[tuple[int, int]] | None:
            """BFS untuk mencari jalur dari spur node ke baris tujuan."""

            queue = deque([spur_start])
            previous = {spur_start: None}

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
                    return path

                x, y = current

                # Urutan deterministik agar hasil pengujian konsisten.
                neighbors = (
                    (x, y - 1),
                    (x - 1, y),
                    (x + 1, y),
                    (x, y + 1),
                )

                for nxt in neighbors:
                    if not (0 <= nxt[0] < 9 and 0 <= nxt[1] < 9):
                        continue

                    if nxt in previous:
                        continue

                    if nxt in forbidden_nodes:
                        continue

                    if (current, nxt) in forbidden_edges:
                        continue

                    if blocked(state.walls, current, nxt):
                        continue

                    previous[nxt] = current
                    queue.append(nxt)

            return None

        # Yen's algorithm:
        # setiap iterasi menghasilkan kandidat setelah jalur terakhir.
        while len(accepted_paths) < max_paths:
            self._check_deadline()

            previous_path = accepted_paths[-1]

            for spur_index in range(len(previous_path) - 1):
                self._check_deadline()

                # Prefix dari start hingga spur node.
                root_path = previous_path[:spur_index + 1]
                spur_node = root_path[-1]

                # Hilangkan edge yang akan menghasilkan prefix yang
                # sama dengan jalur yang sudah diterima.
                forbidden_edges = set()

                for path in accepted_paths:
                    if (
                        len(path) > spur_index + 1
                        and path[:spur_index + 1] == root_path
                    ):
                        forbidden_edges.add(
                            (path[spur_index], path[spur_index + 1])
                        )

                # Cegah jalur spur kembali ke node prefix sebelumnya.
                forbidden_nodes = set(root_path[:-1])

                spur_path = find_spur_path(
                    spur_start=spur_node,
                    forbidden_nodes=forbidden_nodes,
                    forbidden_edges=forbidden_edges,
                )

                if spur_path is None:
                    continue

                # Spur path sudah mencakup spur node.
                total_path = root_path[:-1] + spur_path
                path_key = tuple(total_path)

                if path_key in accepted_keys:
                    continue

                if path_key in candidate_keys:
                    continue

                heapq.heappush(
                    candidates,
                    (len(total_path) - 1, path_key),
                )
                candidate_keys.add(path_key)

            # Tidak ada kandidat lain yang bisa ditemukan.
            if not candidates:
                break

            # Kandidat dengan jumlah langkah paling sedikit dipilih.
            _, next_path_key = heapq.heappop(candidates)
            candidate_keys.remove(next_path_key)

            accepted_paths.append(list(next_path_key))
            accepted_keys.add(next_path_key)

        return accepted_paths
    
    def _candidate_walls(
        self,
        state: GameState,
    ) -> list[Wall]:
        """Generate a bounded set of useful walls, preserving urgent defenses."""
        self._check_deadline()
        player = state.current_player
        if state.walls_remaining[player] <= 0:
            return []

        urgent_walls = self._urgent_defensive_walls(state)
        max_paths = max(1, self.config.max_alternative_paths)

        started = self._profile_start("alternative_paths")
        own_paths = self._alternative_paths(state, player, max_paths=max_paths)
        self._check_deadline()
        opponent_paths = self._alternative_paths(
            state, 1 - player, max_paths=max_paths
        )
        self._profile_end("alternative_paths", started)

        own_path_walls: set[Wall] = set()
        opponent_path_walls: set[Wall] = set()
        for path in own_paths:
            own_path_walls.update(self._wall_candidates_from_path(path))
        for path in opponent_paths:
            opponent_path_walls.update(self._wall_candidates_from_path(path))
        candidates = own_path_walls | opponent_path_walls
        candidates.update(urgent_walls)

        if player == self.root_player:
            root_paths, opponent_paths_for_impact = own_paths, opponent_paths
        else:
            root_paths, opponent_paths_for_impact = opponent_paths, own_paths

        # Rank cheaply by route impact before doing distance-based evaluation.
        impact_ranked: list[tuple[float, Wall]] = []
        for wall in candidates:
            self._check_deadline()
            if not is_legal_wall(state, wall, player):
                continue
            root_blocked = sum(
                self._wall_blocks_path(wall, path) for path in root_paths
            )
            opponent_blocked = sum(
                self._wall_blocks_path(wall, path)
                for path in opponent_paths_for_impact
            )
            impact_ranked.append((opponent_blocked - root_blocked, wall))

        impact_ranked.sort(
            key=lambda item: (
                item[0] if player == self.root_player else -item[0],
                item[1].orientation, item[1].x, item[1].y,
            )
        )

        # Previously every candidate triggered a full static evaluation.
        # Keep that expensive work to a bounded shortlist.
        shortlist_size = max(12, max(0, self.config.max_wall_candidates) * 2)
        shortlist = list(urgent_walls)
        for _, wall in impact_ranked:
            if wall not in shortlist:
                shortlist.append(wall)
            if len(shortlist) >= shortlist_size:
                break

        ranked_candidates: list[tuple[float, Wall]] = []
        for wall in shortlist:
            self._check_deadline()
            if not is_legal_wall(state, wall, player):
                continue
            try:
                child_state = apply_wall(state, wall)
            except (ValueError, AssertionError):
                continue
            impact = (
                sum(self._wall_blocks_path(wall, path) for path in opponent_paths_for_impact)
                - sum(self._wall_blocks_path(wall, path) for path in root_paths)
            )
            score = self._evaluate(child_state)
            score += self.config.route_impact_weight * impact
            ranked_candidates.append((score, wall))

        if player == self.root_player:
            ranked_candidates.sort(key=lambda item: item[0], reverse=True)
        else:
            ranked_candidates.sort(key=lambda item: item[0])

        selected = list(urgent_walls)
        limit = max(0, self.config.max_wall_candidates)
        for _, wall in ranked_candidates:
            if wall not in selected:
                selected.append(wall)
            if len(selected) >= max(limit, len(urgent_walls)):
                break
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

        winning_children = [
            child for child in children if child[1].winner == player
        ]
        if winning_children:
            self.stats["children_generated"] += len(winning_children)
            return winning_children

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

    def _search_root(
        self,
        state: GameState,
        depth: int,
        fallback: Action,
    ) -> tuple[Action, float]:
        """
        Search root actions using standard alpha-beta minimax.

        All scores compared at the root use the same evaluation scale.
        """
        self._check_deadline()

        children = self._generate_children(state)

        if not children:
            return fallback, self._evaluate(state)

        # If any action prevents an immediate opponent win, discard actions
        # that allow one. Tactical safety outranks static route scores.
        safe_children = [
            (action, child_state)
            for action, child_state in children
            if not self._opponent_has_immediate_win(child_state)
        ]
        if safe_children:
            children = safe_children

        winning_children = [
            item for item in children if item[1].winner == self.root_player
        ]
        if winning_children:
            return winning_children[0][0], MATE_SCORE

        children = self._ordered_children(
            children,
            maximizing=True,
        )

        best_action = fallback
        best_score = -INFINITY

        alpha = -INFINITY
        beta = INFINITY

        for action, child_state in children:
            self._check_deadline()

            score = self._minimax(
                child_state,
                depth - 1,
                alpha,
                beta,
            )

            if score > best_score:
                best_score = score
                best_action = action

            alpha = max(alpha, best_score)

        return best_action, best_score


    def _fallback_action(self, state: GameState) -> Action:
        """Quick fallback: win now, avoid immediate loss, then advance."""
        player = state.current_player
        moves = legal_pawn_moves(state, player)
        if not moves:
            raise ValueError("Tidak ada langkah legal untuk posisi ini.")

        goal_y = 0 if player == 0 else 8
        for move in moves:
            if move[1] == goal_y:
                return Action(kind="move", destination=tuple(move))

        safe_moves: list[tuple[int | float, tuple[int, int]]] = []
        all_moves: list[tuple[int | float, tuple[int, int]]] = []
        for move in moves:
            try:
                next_state = apply_move(state, move)
                distance = shortest_distance(next_state, player)
            except (ValueError, AssertionError):
                continue
            item = (distance if distance is not None else INFINITY, move)
            all_moves.append(item)
            if not self._opponent_has_immediate_win(next_state):
                safe_moves.append(item)

        if safe_moves:
            _, move = min(safe_moves, key=lambda item: item[0])
            return Action(kind="move", destination=tuple(move))

        # If every pawn move loses immediately, try a wall that blocks the
        # opponent's winning edge before accepting a losing pawn move.
        for wall in self._urgent_defensive_walls(state):
            try:
                next_state = apply_wall(state, wall)
            except (ValueError, AssertionError):
                continue
            if not self._opponent_has_immediate_win(next_state):
                return Action(kind="wall", wall=wall)

        _, move = min(all_moves, key=lambda item: item[0])
        return Action(kind="move", destination=tuple(move))

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
        self.nodes = 0

        self.transposition_table = {}
        self.distance_cache = {}
        self.goal_distance_map_cache = {}
        self.route_flexibility_cache = {}
        self.evaluation_cache = {}

        self.distance_cache_hits = 0
        self.evaluation_cache_hits = 0

        self.stats = {
            "children_generated": 0,
            "move_candidates": 0,
            "wall_candidates": 0,
            "alpha_beta_cutoffs": 0,
        }

        # Ensure the AI has a legal action before starting deeper search.
        fallback = self._fallback_action(state)

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

        return SearchResult(
            action=best_action,
            score=best_score,
            depth=completed_depth,
            elapsed_ms=elapsed_ms,
            timed_out=timed_out,
            nodes=self.nodes,
        )
