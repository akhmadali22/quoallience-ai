import json

from quoridor.engine import GameState, Wall, apply_wall
from quoridor.advanced_ai import AIConfig
from benchmarks.self_play import measure_wall_effect, play_game


def test_wall_metrics_compare_shortest_distances():
    before = GameState.initial()
    after = apply_wall(before, Wall("H", 3, 6))

    metrics = measure_wall_effect(before, after, player=0)

    assert metrics["own_distance_before"] is not None
    assert metrics["own_distance_after"] is not None
    assert metrics["opponent_distance_before"] is not None
    assert metrics["opponent_distance_after"] is not None
    assert metrics["own_detour"] == (
        metrics["own_distance_after"] - metrics["own_distance_before"]
    )
    assert metrics["opponent_delay"] == (
        metrics["opponent_distance_after"] - metrics["opponent_distance_before"]
    )


def test_self_play_game_records_actions_and_wall_log(tmp_path):
    game_log = tmp_path / "game.jsonl"
    wall_log = tmp_path / "walls.jsonl"

    summary = play_game(
        game_id="smoke_0001",
        seed=123,
        starting_player=0,
        opening_plies=0,
        max_plies=4,
        config=AIConfig(time_limit=0.05, max_depth=1),
        game_log=game_log,
        wall_log=wall_log,
        ai_version="test",
    )

    assert summary["game_id"] == "smoke_0001"
    assert summary["total_plies"] <= 4
    assert game_log.exists()
    records = [json.loads(line) for line in game_log.read_text().splitlines()]
    assert records
    action_records = [record for record in records if record["event"] == "action"]
    assert action_records
    assert all("state_before" in record and "state_after" in record for record in action_records)
    assert records[-1]["event"] == "game_end"
    if wall_log.exists():
        wall_records = [json.loads(line) for line in wall_log.read_text().splitlines()]
        assert all(record["event"] == "wall_analysis" for record in wall_records)
