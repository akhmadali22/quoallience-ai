import torch

from quoridor.engine import GameState
from quoridor.neural_ai.encoder import (
    MOVE_ACTIONS,
    ACTION_SIZE,
    legal_action_indices,
    encode_state,
)
from quoridor.neural_ai.policy import load_model

CHECKPOINT = "data/neural_self_play/smoke/selfplay_model_snapshot.pt"

model = load_model(CHECKPOINT)
model.eval()

state = GameState.initial()
legal = legal_action_indices(state)

tensor = torch.tensor(encode_state(state), dtype=torch.float32).unsqueeze(0)

with torch.inference_mode():
    logits, value = model(tensor)

logits = logits[0]

masked_logits = torch.full((ACTION_SIZE,), float("-inf"))

masked_logits[legal] = logits[legal]

probs = torch.softmax(masked_logits, dim=0)

pawn_actions = [a for a in legal if a < MOVE_ACTIONS]
wall_actions = [a for a in legal if a >= MOVE_ACTIONS]

pawn_mass = sum(probs[a].item() for a in pawn_actions)
wall_mass = sum(probs[a].item() for a in wall_actions)

print("Value:", value.item())
print("Legal actions:", len(legal))
print("Pawn actions:", len(pawn_actions))
print("Wall actions:", len(wall_actions))

print("\nPolicy prior distribution")
print("Pawn mass:", pawn_mass)
print("Wall mass:", wall_mass)

print("\nPawn action priors:")

for action in pawn_actions:
    print(action, probs[action].item())

print("\nTop 10 wall action priors:")

for action in sorted(wall_actions, key=lambda a: probs[a].item(), reverse=True)[:10]:
    print(action, probs[action].item())
