import json
import torch

from quoridor.neural_ai.policy import load_model
from quoridor.neural_ai.encoder import encode_state, legal_action_indices
from quoridor.engine import GameState

path = "data/neural_self_play/smoke/replay.jsonl"

with open(path) as f:
    samples = [json.loads(line) for line in f]

# Cari sampel posisi awal P0.
sample = next(
    (
        row for row in samples
        if row.get("player") == 0
        and row.get("ply") == 0
    ),
    None,
)

if sample is None:
    print("Sampel posisi awal tidak ditemukan.")
    print("Contoh keys:", samples[0].keys())
    raise SystemExit

state = GameState.initial()
legal = legal_action_indices(state)

model = load_model("models/checkpoints/neural_v1.pt")
model.eval()

x = torch.tensor(
    encode_state(state),
    dtype=torch.float32,
).unsqueeze(0)

with torch.inference_mode():
    logits, value = model(x)

    masked = torch.full_like(logits[0], float("-inf"))
    masked[legal] = logits[0, legal]

    predicted = torch.softmax(masked, dim=0)

target = {
    int(action): float(probability)
    for action, probability in sample["policy"].items()
}

print("Value prediction:", value.item())
print("\nAction       Target       Predicted")

for action in sorted(legal):
    t = target.get(action, 0.0)
    p = predicted[action].item()

    if t > 0 or p > 0.01:
        print(f"{action:3d}       {t:.6f}     {p:.6f}")

print("\nTarget sum:", sum(target.values()))