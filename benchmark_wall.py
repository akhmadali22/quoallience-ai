import statistics
import time

from quoridor.engine import GameState, Wall, apply_wall, is_legal_wall

state = GameState()

walls = [
    Wall(orientation, x, y)
    for orientation in ("H", "V")
    for y in range(8)
    for x in range(8)
]

legal_walls = [
    wall for wall in walls
    if is_legal_wall(state, wall, state.current_player)
]

samples = []

for _ in range(10):
    started = time.perf_counter()

    for wall in legal_walls:
        apply_wall(state, wall)

    samples.append(
        (time.perf_counter() - started) * 1000
    )

print(f"Legal wall candidates: {len(legal_walls)}")
print(f"Average total: {statistics.mean(samples):.3f} ms")
print(f"Minimum total: {min(samples):.3f} ms")
print(f"Maximum total: {max(samples):.3f} ms")