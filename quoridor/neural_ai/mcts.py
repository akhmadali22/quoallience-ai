"""PUCT Monte Carlo Tree Search guided by a policy/value network."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from quoridor.engine import GameState
from .encoder import ACTION_SIZE, apply_action_index, encode_state, legal_action_indices


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
    ):
        if simulations < 1:
            raise ValueError("simulations must be positive")
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_fraction = dirichlet_fraction
        self.device = device

    def _evaluate(self, state: GameState) -> tuple[dict[int, float], float]:
        import torch

        legal = legal_action_indices(state)
        if not legal:
            if state.winner is not None:
                return {}, (1.0 if state.winner == state.current_player else -1.0)
            raise RuntimeError("Non-terminal Quoridor state has no legal actions")

        tensor = torch.tensor(encode_state(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            logits, value = self.model(tensor)
            logits = logits[0]
            mask = torch.full((ACTION_SIZE,), float("-inf"), device=logits.device)
            mask[legal] = logits[legal]
            probs = torch.softmax(mask, dim=0)
            priors = {action: float(probs[action].item()) for action in legal}
            return priors, float(value[0].item())

    def _expand(self, node: Node) -> float:
        if node.state.winner is not None:
            node.expanded = True
            return 1.0 if node.state.winner == node.state.current_player else -1.0
        priors, value = self._evaluate(node.state)
        node.edges = {action: Edge(prior=prior) for action, prior in priors.items()}
        node.expanded = True
        return value

    def _select(self, node: Node) -> tuple[int, Edge]:
        scale = math.sqrt(max(1, node.visits))
        best_action = -1
        best_edge = None
        best_score = -float("inf")
        for action, edge in node.edges.items():
            u = self.c_puct * edge.prior * scale / (1 + edge.visits)
            score = edge.q + u
            if score > best_score:
                best_action, best_edge, best_score = action, edge, score
        assert best_edge is not None
        return best_action, best_edge

    def search(self, state: GameState, *, add_root_noise: bool = False) -> dict[int, float]:
        if state.winner is not None:
            return {}
        root = Node(state)
        self._expand(root)

        if add_root_noise and root.edges:
            import torch
            actions = list(root.edges)
            noise = torch.distributions.Dirichlet(
                torch.full((len(actions),), self.dirichlet_alpha, device="cpu")
            ).sample().tolist()
            for action, sample in zip(actions, noise):
                edge = root.edges[action]
                edge.prior = (1 - self.dirichlet_fraction) * edge.prior + self.dirichlet_fraction * sample

        for _ in range(self.simulations):
            node = root
            path: list[tuple[Node, Edge]] = []
            while node.expanded and node.edges:
                action, edge = self._select(node)
                if edge.child is None:
                    edge.child = Node(apply_action_index(node.state, action))
                path.append((node, edge))
                node = edge.child

            value = self._expand(node) if not node.expanded else (
                1.0 if node.state.winner == node.state.current_player else -1.0
            )
            for parent, edge in reversed(path):
                value = -value
                edge.visits += 1
                edge.value_sum += value
                parent.visits += 1

        total = sum(edge.visits for edge in root.edges.values())
        if total == 0:
            return {action: edge.prior for action, edge in root.edges.items()}
        return {action: edge.visits / total for action, edge in root.edges.items()}

    def choose_action(self, state: GameState, *, temperature: float = 0.0, add_root_noise: bool = False) -> tuple[int, dict[int, float]]:
        import random
        visits = self.search(state, add_root_noise=add_root_noise)
        if not visits:
            raise ValueError("Cannot choose an action from a terminal state")
        if temperature <= 1e-8:
            action = max(visits, key=visits.get)
        else:
            weights = {a: max(v, 1e-12) ** (1.0 / temperature) for a, v in visits.items()}
            total = sum(weights.values())
            threshold = random.random() * total
            action = next(a for a, w in weights.items() if (threshold := threshold - w) <= 0)
        return action, visits
