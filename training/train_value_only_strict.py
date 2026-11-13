#!/usr/bin/env python3
"""Strict EXP-001R: train only the value head on undiscounted heuristic outcomes.

The input is the existing EXP-001 JSONL. Existing discounted "value" fields are
intentionally ignored and recomputed from (winner, player). Splits are made by
game_id, stratified by winner, so positions from one game cannot leak across
train/validation/test.

The model remains in eval() mode during optimization. This freezes BatchNorm
running statistics and all other training-mode buffers. Only value_head
parameters are trainable.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from quoridor.neural_ai.policy import load_model, save_checkpoint


def load_games(path: Path) -> tuple[list[dict], dict]:
    samples: list[dict] = []
    game_meta: dict[int, tuple[int, int]] = {}
    seen: set[tuple[int, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"state", "game_id", "ply", "player", "winner"}
            missing = required - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no}: missing fields {sorted(missing)}")
            game_id = int(row["game_id"])
            ply = int(row["ply"])
            player = int(row["player"])
            winner = int(row["winner"])
            if player not in (0, 1) or winner not in (0, 1):
                raise ValueError(f"{path}:{line_no}: invalid player/winner")
            key = (game_id, ply)
            if key in seen:
                raise ValueError(f"{path}:{line_no}: duplicate game_id/ply {key}")
            seen.add(key)
            prior_meta = game_meta.get(game_id)
            meta = (winner, int(row.get("total_plies", -1)))
            if prior_meta is not None and prior_meta != meta:
                raise ValueError(f"game_id={game_id} has inconsistent game metadata")
            game_meta[game_id] = meta
            # Deliberately discard the old discounted value and any policy target.
            value = 1.0 if player == winner else -1.0
            samples.append({
                "state": row["state"], "value": value,
                "game_id": game_id, "ply": ply, "player": player,
                "winner": winner, "total_plies": meta[1],
            })
    if not samples:
        raise ValueError(f"No samples found in {path}")
    return samples, {"games": game_meta, "records": len(samples)}


def allocate_counts(n: int) -> tuple[int, int, int]:
    """Largest-remainder allocation for 80/10/10, preserving exact totals."""
    raw = [0.8 * n, 0.1 * n, 0.1 * n]
    counts = [int(x) for x in raw]
    remaining = n - sum(counts)
    order = sorted(range(3), key=lambda i: (raw[i] - counts[i], -i), reverse=True)
    for i in order[:remaining]:
        counts[i] += 1
    return counts[0], counts[1], counts[2]


def split_by_game(game_meta: dict[int, tuple[int, int]], seed: int) -> dict[str, list[int]]:
    rng = random.Random(seed)
    result = {"train": [], "validation": [], "test": []}
    for winner in (0, 1):
        ids = sorted(gid for gid, meta in game_meta.items() if meta[0] == winner)
        rng.shuffle(ids)
        n_train, n_val, n_test = allocate_counts(len(ids))
        result["train"].extend(ids[:n_train])
        result["validation"].extend(ids[n_train:n_train + n_val])
        result["test"].extend(ids[n_train + n_val:n_train + n_val + n_test])
    for values in result.values():
        values.sort()
    if not result["train"] or not result["validation"] or not result["test"]:
        raise ValueError("Game-level split produced an empty partition")
    all_ids = result["train"] + result["validation"] + result["test"]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(game_meta):
        raise AssertionError("Game split is not a disjoint partition")
    return result


def tensor_batch(samples: list[dict], indices: list[int], device: str):
    states = torch.tensor([samples[i]["state"] for i in indices], dtype=torch.float32, device=device)
    values = torch.tensor([samples[i]["value"] for i in indices], dtype=torch.float32, device=device)
    return states, values


@torch.inference_mode()
def metrics(model, samples: list[dict], indices: list[int], device: str) -> dict:
    if not indices:
        return {}
    model.eval()
    predictions = []
    targets = []
    for start in range(0, len(indices), 128):
        batch_indices = indices[start:start + 128]
        x, y = tensor_batch(samples, batch_indices, device)
        _, pred = model(x)
        predictions.extend(pred.detach().cpu().tolist())
        targets.extend(y.detach().cpu().tolist())
    errors = [p - y for p, y in zip(predictions, targets)]
    nonzero = [(p, y) for p, y in zip(predictions, targets) if y != 0]
    buckets = {}
    for label, lo, hi in (("0-9", 0, 10), ("10-19", 10, 20), ("20-29", 20, 30),
                          ("30-59", 30, 60), ("60+", 60, 10**9)):
        pairs = [(predictions[j], targets[j]) for j, idx in enumerate(indices)
                 if lo <= samples[idx]["ply"] < hi]
        if pairs:
            buckets[label] = {
                "count": len(pairs),
                "mse": sum((p-y)**2 for p, y in pairs) / len(pairs),
                "mae": sum(abs(p-y) for p, y in pairs) / len(pairs),
                "sign_accuracy": sum((p >= 0) == (y > 0) for p, y in pairs) / len(pairs),
            }
    return {
        "count": len(indices),
        "mse": sum(e*e for e in errors) / len(errors),
        "mae": sum(abs(e) for e in errors) / len(errors),
        "sign_accuracy": sum((p >= 0) == (y > 0) for p, y in nonzero) / len(nonzero) if nonzero else None,
        "prediction_mean": sum(predictions) / len(predictions),
        "prediction_min": min(predictions),
        "prediction_max": max(predictions),
        "target_mean": sum(targets) / len(targets),
        "by_ply_bucket": buckets,
    }


@torch.inference_mode()
def probe_outputs(model, samples: list[dict], indices: list[int], device: str) -> list[dict]:
    """Fixed deterministic probe set, selected from held-out test games."""
    chosen = []
    seen_indices = set()
    wanted = (("0-9", 0, 9), ("10-19", 10, 19), ("20-29", 20, 29),
              ("30-59", 30, 59), ("60+", 60, 10**9))
    test_game_ids = sorted({samples[i]["game_id"] for i in indices})
    # Sample each ply bucket from every held-out game, not repeatedly from the
    # first game in the split. This keeps the fixed probe set game-diverse.
    for label, lo, hi in wanted:
        for game_id in test_game_ids:
            candidates = [
                i for i in indices
                if samples[i]["game_id"] == game_id and lo <= samples[i]["ply"] <= hi
            ]
            if candidates:
                idx = min(candidates, key=lambda i: (abs(samples[i]["ply"] - lo), samples[i]["ply"]))
                if idx not in seen_indices:
                    chosen.append((label, idx))
                    seen_indices.add(idx)
    result = []
    model.eval()
    for label, idx in chosen:
        row = samples[idx]
        x = torch.tensor(row["state"], dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = model(x)
        result.append({
            "bucket": label, "game_id": row["game_id"], "ply": row["ply"],
            "target": row["value"], "value": float(value.item()),
            "policy_logits": logits[0].detach().cpu(),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261113003)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("epochs/batch-size/lr must be positive and weight-decay non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    samples, data_info = load_games(args.replay)
    splits = split_by_game(data_info["games"], args.seed)
    id_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(samples):
        id_to_indices[row["game_id"]].append(i)
    split_indices = {
        name: [i for gid in ids for i in id_to_indices[gid]]
        for name, ids in splits.items()
    }

    model = load_model(args.checkpoint, device=args.device)
    model.eval()
    original_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("value_head."))
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable_names or any(not name.startswith("value_head.") for name in trainable_names):
        raise AssertionError(f"Unexpected trainable parameters: {trainable_names}")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    fixed_probe = probe_outputs(model, samples, split_indices["test"], args.device)
    best_state = None
    best_val_mse = float("inf")
    best_epoch = None
    rng = random.Random(args.seed)
    train_indices = split_indices["train"]

    for epoch in range(1, args.epochs + 1):
        model.eval()  # Never switch to train(): BatchNorm buffers must remain frozen.
        order = list(train_indices)
        rng.shuffle(order)
        weighted_loss = 0.0
        seen = 0
        for start in range(0, len(order), args.batch_size):
            batch_ids = order[start:start + args.batch_size]
            x, y = tensor_batch(samples, batch_ids, args.device)
            optimizer.zero_grad(set_to_none=True)
            _, predicted = model(x)
            loss = F.mse_loss(predicted, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            weighted_loss += float(loss.item()) * len(batch_ids)
            seen += len(batch_ids)
        val_metrics = metrics(model, samples, split_indices["validation"], args.device)
        train_mse = weighted_loss / max(seen, 1)
        print(
            f"epoch={epoch}/{args.epochs} train_mse={train_mse:.6f} "
            f"val_mse={val_metrics['mse']:.6f} val_mae={val_metrics['mae']:.6f} "
            f"val_sign={val_metrics['sign_accuracy']:.4f}", flush=True,
        )
        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best checkpoint state was captured")
    model.load_state_dict(best_state)
    model.eval()

    final_state = model.state_dict()
    frozen_changed = []
    value_changed = []
    for key, before in original_state.items():
        after = final_state[key].detach().cpu()
        if key.startswith("value_head."):
            if not torch.equal(before, after):
                value_changed.append(key)
        elif not torch.equal(before, after):
            frozen_changed.append(key)
    if frozen_changed:
        raise RuntimeError(f"Frozen parameter/buffer state changed: {frozen_changed}")
    if not value_changed:
        raise RuntimeError("No value_head tensor changed; training had no effect")

    post_probe = probe_outputs(model, samples, split_indices["test"], args.device)
    probe_rows = []
    for before, after in zip(fixed_probe, post_probe):
        before_logits = before.pop("policy_logits")
        after_logits = after.pop("policy_logits")
        probe_rows.append({
            **before,
            "value_after": after["value"],
            "policy_logits_max_abs_diff": float((before_logits - after_logits).abs().max().item()),
        })

    test_metrics = metrics(model, samples, split_indices["test"], args.device)
    train_metrics = metrics(model, samples, split_indices["train"], args.device)
    validation_metrics = metrics(model, samples, split_indices["validation"], args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, args.output, step=best_epoch or args.epochs, extra={
        "experiment": "EXP-001R",
        "description": "Strict value-head-only training; undiscounted outcomes; game-level split",
        "incumbent": str(args.checkpoint),
        "replay": str(args.replay),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_mse": best_val_mse,
        "trunk_frozen": True,
        "policy_head_frozen": True,
        "batchnorm_buffers_frozen": True,
        "value_target": "undiscounted +/-1 from player-to-move perspective",
        "split_game_ids": splits,
    })

    report = {
        "experiment": "EXP-001R",
        "config": {
            "incumbent": str(args.checkpoint), "candidate": str(args.output),
            "replay": str(args.replay), "seed": args.seed,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "device": args.device, "trunk_frozen": True, "policy_head_frozen": True,
            "batchnorm_buffers_frozen": True, "value_target": "undiscounted +/-1",
        },
        "data": {
            "records": len(samples), "games": len(data_info["games"]),
            "winner_counts": dict(Counter(meta[0] for meta in data_info["games"].values())),
            "samples_per_split": {k: len(v) for k, v in split_indices.items()},
            "games_per_split": {k: len(v) for k, v in splits.items()},
        },
        "split_game_ids": splits,
        "metrics": {"train": train_metrics, "validation": validation_metrics, "test": test_metrics},
        "best_epoch": best_epoch,
        "best_validation_mse": best_val_mse,
        "integrity": {
            "trainable_parameter_names": trainable_names,
            "changed_value_head_tensors": value_changed,
            "frozen_state_tensors_changed": frozen_changed,
            "all_non_value_state_dict_entries_bitwise_unchanged": not frozen_changed,
            "policy_logits_max_abs_diff": max((r["policy_logits_max_abs_diff"] for r in probe_rows), default=0.0),
        },
        "fixed_test_probes": probe_rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Candidate saved: {args.output}")
    print(f"Report saved: {args.report}")
    print(f"Test MSE={test_metrics['mse']:.6f} MAE={test_metrics['mae']:.6f} "
          f"sign_accuracy={test_metrics['sign_accuracy']:.4f}")
    print("Verified: every non-value state_dict entry, including BatchNorm buffers, is bitwise unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
