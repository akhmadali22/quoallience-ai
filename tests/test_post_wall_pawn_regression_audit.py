import ijson
import json
from decimal import Decimal
from collections import Counter

path = "data/diagnostics/post_wall_pawn_regression_audit.json"

with open(path, "rb") as f:
    run = next(ijson.items(f, "runs.item"))

print("RUN KEYS:")
print(json.dumps(list(run.keys()), indent=2))

print("\nVARIANT KEYS:")
for name, variant in run["variants"].items():
    print(f"\n{name}:")
    print(json.dumps(list(variant.keys()), indent=2))

print("\nCLASSIFICATION:")
for name, variant in run["variants"].items():
    print(name, "=", variant.get("classification"))

path = "data/diagnostics/post_wall_pawn_regression_audit.json"

counts = {
    "vanilla_puct": Counter(),
    "with_repetition_avoidance": Counter(),
}

paired = Counter()
total = 0

with open(path, "rb") as f:
    for run in ijson.items(f, "runs.item"):
        total += 1

        a = run["variants"]["vanilla_puct"]["classification"]
        b = run["variants"]["with_repetition_avoidance"]["classification"]

        counts["vanilla_puct"][a] += 1
        counts["with_repetition_avoidance"][b] += 1

        paired[(a, b)] += 1

print("TOTAL POSITIONS:", total)

for variant, counter in counts.items():
    print(f"\n{variant}")
    for classification, count in counter.most_common():
        print(f"{classification}: {count} ({count / total * 100:.2f}%)")

print("\nPAIRED CLASSIFICATION")
for (a, b), count in paired.most_common():
    print(f"{a} -> {b}: {count} ({count / total * 100:.2f}%)")

path = "data/diagnostics/post_wall_pawn_regression_audit.json"

def convert_decimal(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_decimal(v) for v in obj]
    return obj

with open(path, "rb") as f:
    run = next(ijson.items(f, "runs.item"))

for variant_name, variant in run["variants"].items():
    print(f"\n=== {variant_name} ===")

    for field in [
        "first_root_snapshot",
        "first_regression_overtake",
        "best_final_regression",
        "best_final_progress",
        "final_selected_pawn_action",
        "response_probe",
    ]:
        value = convert_decimal(variant.get(field))

        print(f"\n{field}:")
        print(json.dumps(value, indent=2)[:4000])

    print("\nFirst selection trace:")
    trace = variant.get("selection_trace", [])
    if trace:
        print(json.dumps(convert_decimal(trace[0]), indent=2)[:4000])