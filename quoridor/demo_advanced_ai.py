import json

from quoridor.engine import GameState
from quoridor.advanced_ai import AIConfig, QuoridorAI
from quoridor.engine import (
    PROFILE, 
    reset_profile,
    has_path_cache_info,
    clear_has_path_cache,
)


def main():
    state = GameState()

    ai = QuoridorAI(
        AIConfig(
            time_limit=60,
            max_depth=4,
            progress_weight=2.0,
            wall_resource_weight=0.25,
            max_wall_candidates=8,
            transposition_table_limit=50_000,
        )
    )

    reset_profile()
    clear_has_path_cache()

    result = ai.choose_action(state)

    print("Initial state:")
    print(state)

    print("\nAI decision:")
    print(json.dumps(
        result.to_dict(),
        indent=2,
    ))
    print("\nCache statistics:")
    print(json.dumps(
        {
            "distance_cache_entries": len(ai.distance_cache),
            "distance_cache_hits": ai.distance_cache_hits,
            "evaluation_cache_entries": len(ai.evaluation_cache),
            "evaluation_cache_hits": ai.evaluation_cache_hits,
            "transposition_table_entries": len(ai.transposition_table),
        },
        indent=2,
    ))
    print("\nSearch statistics:")
    print(ai.stats)
    print("\nPath cache statistics:")
    print(has_path_cache_info())
    print("\nOperation profiling:")
    print(json.dumps(
        ai.profile_report(),
        indent=2,
    ))
    print("\nEngine profiling:")
    print({
        "has_path_calls": PROFILE["has_path_calls"],
        "has_path_total_ms": round(
            PROFILE["has_path_time"] * 1000, 3
        ),
        "has_path_avg_ms": round(
            PROFILE["has_path_time"] * 1000
            / max(PROFILE["has_path_calls"], 1),
            4,
        ),
        "blocked_calls": PROFILE["blocked_calls"],
        "blocked_total_ms": round(
            PROFILE["blocked_time"] * 1000, 3
        ),
        "blocked_avg_us": round(
            PROFILE["blocked_time"] * 1_000_000
            / max(PROFILE["blocked_calls"], 1),
            3,
        ),
    })


if __name__ == "__main__":
    main()