"""PUCT Monte Carlo Tree Search with optional progressive widening."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from quoridor.engine import GameState, shortest_distance
from .encoder import ACTION_SIZE, MOVE_ACTIONS as ENCODER_MOVE_ACTIONS, apply_action_index, encode_state, legal_action_indices


MOVE_ACTIONS = ENCODER_MOVE_ACTIONS


@dataclass
class Edge:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    child: "Node | None" = None

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class Node:
    state: GameState
    visits: int = 0
    edges: dict[int, Edge] = field(default_factory=dict)
    # Wall actions not yet admitted by progressive widening.
    pending_walls: list[tuple[int, Edge]] = field(default_factory=list)
    expanded: bool = False


class MCTS:
    def __init__(
        self,
        model: Any,
        *,
        simulations: int = 64,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_fraction: float = 0.25,
        device: str = "cpu",
        progressive_widening: bool = True,
        initial_wall_candidates: int = 8,
        widening_coefficient: float = 2.0,
        widening_exponent: float = 0.5,
        category_exploration: float = 0.35,
        repetition_penalty: float = 0.75,
        pawn_progress_bonus: float = 0.0,
        repetition_window: int = 4,
    ):
        if simulations < 1:
            raise ValueError("simulations must be positive")
        if initial_wall_candidates < 0:
            raise ValueError("initial_wall_candidates cannot be negative")
        if widening_coefficient <= 0:
            raise ValueError("widening_coefficient must be positive")
        if not 0 < widening_exponent <= 1:
            raise ValueError("widening_exponent must be in (0, 1]")
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_fraction = dirichlet_fraction
        self.device = device
        self.progressive_widening = progressive_widening
        self.initial_wall_candidates = initial_wall_candidates
        self.widening_coefficient = widening_coefficient
        if category_exploration < 0:
            raise ValueError("category_exploration cannot be negative")
        self.category_exploration = category_exploration
        if repetition_penalty < 0 or pawn_progress_bonus < 0:
            raise ValueError("repetition_penalty and pawn_progress_bonus cannot be negative")
        if repetition_window < 1:
            raise ValueError("repetition_window must be positive")
        self.repetition_penalty = repetition_penalty
        self.pawn_progress_bonus = pawn_progress_bonus
        self.repetition_window = repetition_window
        self.widening_exponent = widening_exponent
        self.last_search_stats: dict[str, Any] = {}

    def _evaluate(self, state: GameState) -> tuple[dict[int, float], float]:
        import torch

        legal = legal_action_indices(state)
        if not legal:
            if state.winner is not None:
                return {}, (1.0 if state.winner == state.current_player else -1.0)
            raise RuntimeError("Non-terminal Quoridor state has no legal actions")

        tensor = torch.tensor(
            encode_state(state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.inference_mode():
            logits, value = self.model(tensor)
            logits = logits[0]
            mask = torch.full((ACTION_SIZE,), float("-inf"), device=logits.device)
            mask[legal] = logits[legal]
            probs = torch.softmax(mask, dim=0)
            priors = {action: float(probs[action].item()) for action in legal}
            # Keep raw network priors here. Category balancing belongs in
            # _select(), where it can see the actions currently admitted by
            # progressive widening at this particular node.
            return priors, float(value[0].item())

    def _expand(self, node: Node) -> float:
        if node.state.winner is not None:
            node.expanded = True
            return 1.0 if node.state.winner == node.state.current_player else -1.0

        priors, value = self._evaluate(node.state)
        node.edges = {
            action: Edge(prior=prior)
            for action, prior in priors.items()
            if action < MOVE_ACTIONS
        }
        node.pending_walls = sorted(
            [
                (action, Edge(prior=prior))
                for action, prior in priors.items()
                if action >= MOVE_ACTIONS
            ],
            key=lambda item: item[1].prior,
            reverse=True,
        )
        node.expanded = True

        if not self.progressive_widening:
            node.edges.update(node.pending_walls)
            node.pending_walls.clear()
        else:
            self._widen(node)
        return value

    def _widen(self, node: Node) -> None:
        """Admit more wall actions as this node receives more visits.

        Every legal pawn action is active immediately. Wall candidates are
        admitted in descending policy-prior order.
        """
        if not self.progressive_widening or not node.pending_walls:
            return

        active_wall_count = sum(
            action >= MOVE_ACTIONS for action in node.edges
        )
        target_wall_count = max(
            self.initial_wall_candidates,
            math.ceil(
                self.widening_coefficient
                * max(1, node.visits) ** self.widening_exponent
            ),
        )
        target_wall_count = min(
            target_wall_count,
            active_wall_count + len(node.pending_walls),
        )

        while active_wall_count < target_wall_count and node.pending_walls:
            action, edge = node.pending_walls.pop(0)
            node.edges[action] = edge
            active_wall_count += 1

    def _select(
        self, node: Node, action_adjustments: dict[int, float] | None = None
    ) -> tuple[int, Edge]:
        scale = math.sqrt(max(1, node.visits))
        active_priors = {action: edge.prior for action, edge in node.edges.items()}
        prior_total = sum(active_priors.values())
        if prior_total > 0:
            normalized_priors = {action: prior / prior_total for action, prior in active_priors.items()}
        else:
            uniform = 1.0 / len(active_priors) if active_priors else 0.0
            normalized_priors = {action: uniform for action in active_priors}

        category_visits = {
            "pawn": sum(edge.visits for action, edge in node.edges.items() if action < MOVE_ACTIONS),
            "wall": sum(edge.visits for action, edge in node.edges.items() if action >= MOVE_ACTIONS),
        }
        category_bonus = {
            name: self.category_exploration * math.sqrt(math.log1p(node.visits) / (1 + count))
            for name, count in category_visits.items()
        }
        best_action = -1
        best_edge = None
        best_score = -float("inf")
        for action, edge in node.edges.items():
            normalized_prior = normalized_priors[action]
            u = (
                self.c_puct
                * normalized_prior
                * scale
                / (1 + edge.visits)
            )
            category = "pawn" if action < MOVE_ACTIONS else "wall"
            score = edge.q + u + category_bonus[category]
            if action_adjustments is not None:
                score += action_adjustments.get(action, 0.0)
            if score > best_score:
                best_action, best_edge, best_score = action, edge, score
        assert best_edge is not None
        return best_action, best_edge

    def _all_root_edges(self, root: Node) -> dict[int, Edge]:
        result = dict(root.edges)
        result.update(root.pending_walls)
        return result

    def _root_action_adjustments(
        self, state: GameState, recent_pawn_positions: list[tuple[int, int]] | None
    ) -> dict[int, float]:
        """Apply small root-only preferences against recent pawn cycles.

        Positions are from the current player's own pawn history. The current
        square is excluded; earlier squares in the configured window are
        discouraged, not made illegal. A mild shortest-path delta bonus keeps
        forward progress visible without forbidding tactical sideways moves.
        """
        if not recent_pawn_positions:
            return {}
        history = recent_pawn_positions[-(self.repetition_window + 1):-1]
        repeated = set(history)
        player = state.current_player
        before = shortest_distance(state, player)
        adjustments: dict[int, float] = {}
        from .encoder import decode_action
        for action in legal_action_indices(state):
            if action >= MOVE_ACTIONS:
                continue
            _, destination = decode_action(action)
            next_state = apply_action_index(state, action)
            after = shortest_distance(next_state, player)
            adjustment = 0.0
            if destination in repeated:
                adjustment -= self.repetition_penalty
            if before is not None and after is not None:
                adjustment += self.pawn_progress_bonus * (before - after)
            if adjustment:
                adjustments[action] = adjustment
        return adjustments

    def search(
        self,
        state: GameState,
        *,
        add_root_noise: bool = False,
        recent_pawn_positions: list[tuple[int, int]] | None = None,
    ) -> dict[int, float]:
        if state.winner is not None:
            self.last_search_stats = {}
            return {}

        stats: dict[str, Any] = {
            "simulations": self.simulations,
            "expanded_nodes": 0,
            "nn_evaluations": 0,
            "terminal_nodes": 0,
        }
        expanded_depths: Counter[int] = Counter()
        evaluated_depths: Counter[int] = Counter()
        selection_depths: Counter[int] = Counter()
        leaf_depths: list[int] = []

        def expand_tracked(node: Node) -> float:
            depth = node.state.ply - state.ply
            stats["expanded_nodes"] += 1
            expanded_depths[depth] += 1
            if node.state.winner is not None:
                stats["terminal_nodes"] += 1
            else:
                stats["nn_evaluations"] += 1
                evaluated_depths[depth] += 1
            return self._expand(node)

        root = Node(state)
        root_value = expand_tracked(root)
        root_action_adjustments = self._root_action_adjustments(state, recent_pawn_positions)

        if add_root_noise and root.edges:
            import torch

            all_edges = self._all_root_edges(root)
            actions = list(all_edges)
            noise = torch.distributions.Dirichlet(
                torch.full((len(actions),), self.dirichlet_alpha, device="cpu")
            ).sample().tolist()
            for action, sample in zip(actions, noise):
                edge = all_edges[action]
                edge.prior = (
                    (1 - self.dirichlet_fraction) * edge.prior
                    + self.dirichlet_fraction * sample
                )
            root.pending_walls.sort(
                key=lambda item: item[1].prior, reverse=True
            )

        for _ in range(self.simulations):
            node = root
            path: list[tuple[Node, Edge]] = []

            while node.expanded and node.edges:
                self._widen(node)
                depth = node.state.ply - state.ply
                selection_depths[depth] += 1
                action, edge = self._select(
                    node, root_action_adjustments if node is root else None
                )
                if edge.child is None:
                    edge.child = Node(apply_action_index(node.state, action))
                path.append((node, edge))
                node = edge.child

            leaf_depths.append(node.state.ply - state.ply)
            value = expand_tracked(node) if not node.expanded else (
                1.0 if node.state.winner == node.state.current_player else -1.0
            )
            for parent, edge in reversed(path):
                value = -value
                edge.visits += 1
                edge.value_sum += value
                parent.visits += 1

        all_root_edges = self._all_root_edges(root)
        total = sum(edge.visits for edge in all_root_edges.values())
        if total == 0:
            prior_total = sum(edge.prior for edge in all_root_edges.values())
            probabilities = (
                {action: 0.0 for action in all_root_edges}
                if prior_total <= 0
                else {
                    action: edge.prior / prior_total
                    for action, edge in all_root_edges.items()
                }
            )
        else:
            probabilities = {
                action: edge.visits / total
                for action, edge in all_root_edges.items()
            }

        pawn_visits = sum(
            edge.visits
            for action, edge in all_root_edges.items()
            if action < MOVE_ACTIONS
        )
        wall_visits = total - pawn_visits
        active_root_priors = {action: edge.prior for action, edge in root.edges.items()}
        active_prior_total = sum(active_root_priors.values())
        active_root_selection_priors = (
            {action: prior / active_prior_total for action, prior in active_root_priors.items()}
            if active_prior_total > 0 else
            {action: 1.0 / len(active_root_priors) for action in active_root_priors}
        )
        root_pawn_visits = sum(edge.visits for action, edge in all_root_edges.items() if action < MOVE_ACTIONS)
        root_wall_visits = total - root_pawn_visits
        root_action_stats = {
            str(action): {
                "category": "pawn" if action < MOVE_ACTIONS else "wall",
                "prior": edge.prior, "q": edge.q, "visits": edge.visits,
                "active": action in root.edges,
                "selection_adjustment": root_action_adjustments.get(action, 0.0),
            }
            for action, edge in all_root_edges.items()
        }
        policy_entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities.values()
            if probability > 0
        )
        stats.update({
            "root_value": root_value,
            "root_repetition_penalty": self.repetition_penalty,
            "root_pawn_progress_bonus": self.pawn_progress_bonus,
            "root_repetition_adjusted_actions": sum(1 for v in root_action_adjustments.values() if v < 0),
            "root_policy_entropy": policy_entropy,
            "max_leaf_depth": max(leaf_depths, default=0),
            "average_leaf_depth": (
                sum(leaf_depths) / len(leaf_depths) if leaf_depths else 0.0
            ),
            "expanded_nodes_by_depth": dict(sorted(expanded_depths.items())),
            "nn_evaluations_by_depth": dict(sorted(evaluated_depths.items())),
            "selection_calls_by_depth": dict(sorted(selection_depths.items())),
            "root_legal_actions": len(all_root_edges),
            "root_active_actions": len(root.edges),
            "root_active_pawn_actions": sum(
                action < MOVE_ACTIONS for action in root.edges
            ),
            "root_active_wall_actions": sum(
                action >= MOVE_ACTIONS for action in root.edges
            ),
            "root_pending_wall_actions": len(root.pending_walls),
            "root_active_pawn_prior_mass": sum(
                prior for action, prior in active_root_selection_priors.items()
                if action < MOVE_ACTIONS
            ),
            "root_active_wall_prior_mass": sum(
                prior for action, prior in active_root_selection_priors.items()
                if action >= MOVE_ACTIONS
            ),
            "category_exploration": self.category_exploration,
            "root_pawn_visits": root_pawn_visits,
            "root_wall_visits": root_wall_visits,
            "root_pawn_visit_ratio": root_pawn_visits / total if total else 0.0,
            "root_wall_visit_ratio": root_wall_visits / total if total else 0.0,
            "root_action_stats": root_action_stats,
            "pawn_visits": pawn_visits,
            "wall_visits": wall_visits,
            "pawn_visit_ratio": pawn_visits / total if total else 0.0,
            "wall_visit_ratio": wall_visits / total if total else 0.0,
        })
        self.last_search_stats = stats
        return probabilities

    def choose_action(
        self,
        state: GameState,
        *,
        temperature: float = 0.0,
        add_root_noise: bool = False,
        rng: Any | None = None,
        recent_pawn_positions: list[tuple[int, int]] | None = None,
    ) -> tuple[int, dict[int, float]]:
        """Choose from root visits; an optional RNG makes ties reproducible per game."""
        import random

        rng = rng or random
        if recent_pawn_positions is None:
            visits = self.search(state, add_root_noise=add_root_noise)
        else:
            visits = self.search(
                state, add_root_noise=add_root_noise,
                recent_pawn_positions=recent_pawn_positions,
            )
        if not visits:
            raise ValueError("Cannot choose an action from a terminal state")
        if temperature <= 1e-8:
            max_visits = max(visits.values())
            best_actions = [
                action for action, count in visits.items()
                if count == max_visits
            ]
            action = rng.choice(best_actions)
        else:
            weights = {
                action: probability ** (1.0 / temperature)
                for action, probability in visits.items()
                if probability > 0
            }
            total = sum(weights.values())
            threshold = rng.random() * total
            action = next(
                action
                for action, weight in weights.items()
                if (threshold := threshold - weight) <= 0
            )
        return action, visits
