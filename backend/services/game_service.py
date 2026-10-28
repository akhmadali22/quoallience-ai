from quoridor.engine import (
    Wall,
    apply_move,
    apply_wall,
    validate_state,
)

from backend.game_manager import GameSession
from backend.services.ai_service import AIService


class GameService:
    def __init__(self, session: GameSession):
        self.session = session
        self.ai_service = AIService(
        )

    def human_move(self, destination: tuple[int, int]) -> dict:
        session = self.session

        if session.status() != "in_progress":
            raise ValueError("Permainan sudah selesai.")

        if session.state.current_player != session.human_player:
            raise ValueError("Sekarang giliran AI.")

        old_state = session.state

        new_state = apply_move(
            old_state,
            destination,
            session.human_player,
        )

        validate_state(new_state)
        session.state = new_state

        session.history.append({
            "ply": new_state.ply,
            "player": session.human_player,
            "action": {
                "type": "move",
                "destination": list(destination),
            },
        })

        return session.to_dict()

    def human_wall(
        self,
        orientation: str,
        x: int,
        y: int,
    ) -> dict:
        session = self.session

        if session.status() != "in_progress":
            raise ValueError("Permainan sudah selesai.")

        if session.state.current_player != session.human_player:
            raise ValueError("Sekarang giliran AI.")

        if orientation not in ("H", "V"):
            raise ValueError("Orientasi dinding harus H atau V.")

        wall = Wall(
            orientation=orientation,
            x=x,
            y=y,
        )

        new_state = apply_wall(
            session.state,
            wall,
            session.human_player,
        )

        validate_state(new_state)
        session.state = new_state

        session.history.append({
            "ply": new_state.ply,
            "player": session.human_player,
            "action": {
                "type": "wall",
                "orientation": orientation,
                "x": x,
                "y": y,
            },
        })

        return session.to_dict()

    def ai_move(self) -> dict:
        session = self.session

        if session.status() != "in_progress":
            raise ValueError("Permainan sudah selesai.")

        if session.state.current_player != session.ai_player:
            raise ValueError("Sekarang giliran human.")

        result = self.ai_service.choose_action(session.state)

        action = result["action"]

        if action["type"] == "move":
            new_state = apply_move(
                session.state,
                tuple(action["destination"]),
                session.ai_player,
            )
        else:
            wall = Wall(
                orientation=action["orientation"],
                x=action["x"],
                y=action["y"],
            )

            new_state = apply_wall(
                session.state,
                wall,
                session.ai_player,
            )

        validate_state(new_state)
        session.state = new_state

        session.history.append({
            "ply": new_state.ply,
            "player": session.ai_player,
            "action": action,
            "analysis": result,
        })

        return {
            "action": action,
            "score": result["score"],
            "depth": result["depth"],
            "elapsed_ms": result["elapsed_ms"],
            "timed_out": result["timed_out"],
            "nodes": result["nodes"],
            "statistics": result["statistics"],
            "game": session.to_dict(),
        }