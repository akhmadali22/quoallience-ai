from quoridor.engine import GameState, Wall
from quoridor.neural_ai.encoder import (
    ACTION_SIZE, INPUT_CHANNELS, apply_action_index, decode_action,
    encode_action, encode_state, legal_action_indices,
)


def test_encoder_shape_and_perspective():
    state = GameState.initial()
    encoded = encode_state(state)
    assert len(encoded) == INPUT_CHANNELS
    assert all(len(plane) == 9 and all(len(row) == 9 for row in plane) for plane in encoded)
    assert encoded[0][8][4] == 1.0
    assert encoded[1][0][4] == 1.0


def test_action_round_trip():
    for index in range(ACTION_SIZE):
        assert encode_action(decode_action(index)) == index


def test_initial_legal_actions_are_decodable_and_applicable():
    state = GameState.initial()
    actions = legal_action_indices(state)
    assert actions
    assert len(actions) == len(set(actions))
    for action in actions:
        next_state = apply_action_index(state, action)
        assert next_state.ply == 1


def test_wall_action_encoding():
    action = encode_action(("wall", Wall("V", 7, 6)))
    assert decode_action(action) == ("wall", Wall("V", 7, 6))
