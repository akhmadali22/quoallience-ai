from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.game_manager import GameManager
from backend.schemas import (
    NewGameRequest,
    MoveRequest,
    WallRequest,
)
from backend.services.game_service import GameService


app = FastAPI(
    title="Quoridor AI API",
    description="API untuk bermain Quoridor melawan AI.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

game_manager = GameManager()
game_services: dict[str, GameService] = {}


@app.get("/")
def root():
    return {
        "name": "Quoridor AI API",
        "status": "running",
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/games")
def create_game(request: NewGameRequest):
    try:
        session = game_manager.create_game(
            human_player=request.human_player,
            ai_time_limit=request.ai_time_limit,
            ai_max_depth=request.ai_max_depth,
        )

        game_services[session.game_id] = GameService(session)

        return session.to_dict()

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.get("/api/games/{game_id}")
def get_game(game_id: str):
    try:
        session = game_manager.get_game(game_id)
        return session.to_dict()

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@app.post("/api/games/{game_id}/move")
def human_move(game_id: str, request: MoveRequest):
    try:
        service = game_services.get(game_id)

        if service is None:
            game_manager.get_game(game_id)
            raise HTTPException(
                status_code=404,
                detail="Game service tidak ditemukan.",
            )

        return service.human_move(request.destination)

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post("/api/games/{game_id}/wall")
def human_wall(game_id: str, request: WallRequest):
    try:
        service = game_services.get(game_id)

        if service is None:
            game_manager.get_game(game_id)
            raise HTTPException(
                status_code=404,
                detail="Game service tidak ditemukan.",
            )

        return service.human_wall(
            orientation=request.orientation,
            x=request.x,
            y=request.y,
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post("/api/games/{game_id}/ai-move")
def ai_move(game_id: str):
    try:
        service = game_services.get(game_id)

        if service is None:
            game_manager.get_game(game_id)
            raise HTTPException(
                status_code=404,
                detail="Game service tidak ditemukan.",
            )

        return service.ai_move()

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.get("/api/games/{game_id}/history")
def get_history(game_id: str):
    try:
        session = game_manager.get_game(game_id)
        return {
            "game_id": game_id,
            "history": session.history,
        }

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@app.delete("/api/games/{game_id}")
def delete_game(game_id: str):
    try:
        game_manager.delete_game(game_id)
        game_services.pop(game_id, None)

        return {
            "game_id": game_id,
            "deleted": True,
        }

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc