import time

from quoridor.engine import GameState
from quoridor.advanced_ai import QuoridorAI, AIConfig

config = AIConfig(
    time_limit=5,
    max_depth=3,
    max_alternative_paths=3,
    max_wall_candidates=8,
    route_impact_weight=0.5,
)

ai = QuoridorAI(config)
state = GameState()

started = time.perf_counter()
result = ai.choose_action(state)
elapsed = time.perf_counter() - started

print(f"Durasi pencarian: {elapsed:.3f} detik")
print(f"Aksi terpilih: {result.action}")
print(f"Skor: {result.score}")
print(f"Kedalaman: {result.depth}")
print(f"Node dikunjungi: {ai.nodes}")