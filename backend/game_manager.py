from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from quoridor.engine import GameState
from dataclasses import replace


@dataclass
class GameSession:
    game_id: str
    state: GameState
    human_player: int
    ai_time_limit: float
    ai_max_depth: int
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ai_player(self) -> int:
        return 1 - self.human_player

    def status(self) -> str:
        if self.state.winner is None:
            return "in_progress"

        if self.state.winner == self.human_player:
            return "human_won"

        return "ai_won"

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            **self.state.to_dict(),
            "human_player": self.human_player,
            "ai_player": self.ai_player,
            "status": self.status(),
        }


class GameManager:
    """Penyimpanan pertandingan selama backend berjalan."""

    def __init__(self):
        self._games: dict[str, GameSession] = {}

    def create_game(
        self,
        human_player: int = 0,
        first_player: str = "human",
        ai_time_limit: float = 5.0,
        ai_max_depth: int = 4,
    ) -> GameSession:
        if human_player not in (0, 1):
            raise ValueError("human_player harus 0 atau 1.")

        if first_player not in ("human", "ai"):
            raise ValueError("first_player harus 'human' atau 'ai'.")

        ai_player = 1 - human_player

        current_player = human_player if first_player == "human" else ai_player

        state = replace(
            GameState.initial(),
            current_player=current_player,
        )

        session = GameSession(
            game_id=str(uuid4()),
            state=state,
            human_player=human_player,
            ai_time_limit=ai_time_limit,
            ai_max_depth=ai_max_depth,
        )

        self._games[session.game_id] = session
        return session

    def get_game(self, game_id: str) -> GameSession:
        session = self._games.get(game_id)

        if session is None:
            raise KeyError(f"Game '{game_id}' tidak ditemukan.")

        return session

    def delete_game(self, game_id: str) -> None:
        if game_id not in self._games:
            raise KeyError(f"Game '{game_id}' tidak ditemukan.")

        del self._games[game_id]
