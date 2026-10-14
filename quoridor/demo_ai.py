import json

from .ai import choose_action
from .engine import GameState


def main():
    state = GameState.initial()

    decision = choose_action(state)

    print("Initial state:")
    print(json.dumps(state.to_dict(), indent=2))

    print("\nAI decision:")
    print(json.dumps(decision.to_dict(), indent=2))


if __name__ == "__main__":
    main()
