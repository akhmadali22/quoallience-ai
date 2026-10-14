from typing import Literal

from pydantic import BaseModel, Field


class NewGameRequest(BaseModel):
    human_player: Literal[0, 1] = 0
    first_player: Literal["human", "ai"] = "human"

class MoveRequest(BaseModel):
    destination: tuple[int, int]


class WallRequest(BaseModel):
    orientation: Literal["H", "V"]
    x: int = Field(ge=0, le=7)
    y: int = Field(ge=0, le=7)


class ActionResponse(BaseModel):
    type: Literal["move", "wall"]
    destination: tuple[int, int] | None = None
    wall: dict | None = None


class GameStateResponse(BaseModel):
    game_id: str
    pawns: list[tuple[int, int]]
    walls: list[dict]
    walls_remaining: list[int]
    current_player: int
    human_player: int
    ai_player: int
    winner: int | None
    ply: int
    status: Literal["in_progress", "human_won", "ai_won"]


class AIMoveResponse(BaseModel):
    action: ActionResponse
    score: float
    depth: int
    elapsed_ms: float
    timed_out: bool
    nodes: int
    game: GameStateResponse