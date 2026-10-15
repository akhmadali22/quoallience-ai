from quoridor.engine import GameState
from quoridor.advanced_ai import AIConfig, QuoridorAI


class AIService:
    def __init__(
        self,
        time_limit: float = 55,
        max_depth: int = 5,
    ):
        self.config = AIConfig(
            time_limit=time_limit,
            max_depth=max_depth,
        )
        self.ai = QuoridorAI(self.config)

    def choose_action(self, state: GameState) -> dict:
        print("\n========== AI POSITION ==========")
        print(self.ai.debug_position(state))
        print("======== END AI POSITION ========\n")

        result = self.ai.choose_action(state)

        return {
            **result.to_dict(),
            "statistics": {
                "distance_cache_entries": len(self.ai.distance_cache),
                "distance_cache_hits": self.ai.distance_cache_hits,
                "evaluation_cache_entries": len(self.ai.evaluation_cache),
                "evaluation_cache_hits": self.ai.evaluation_cache_hits,
                "transposition_table_entries": len(
                    self.ai.transposition_table
                ),
                **self.ai.stats,
            },
        }