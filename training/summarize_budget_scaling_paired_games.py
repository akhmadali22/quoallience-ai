#!/usr/bin/env python3
"""Summarize paired-seat MCTS budget tournaments with exact paired sign tests."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


PRIMARY = {
    "candidate_32_vs_128",
    "candidate_128_vs_256",
    "incumbent_32_vs_128",
    "incumbent_128_vs_256",
}


def upper_binomial_tail(wins: int, decisive: int) -> float:
    if decisive == 0:
        return 1.0
    return sum(math.comb(decisive, k) for k in range(wins, decisive + 1)) / (2 ** decisive)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate percentile of empty sample")
    index = (len(ordered) - 1) * p
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def bootstrap_pair_score_ci(pair_scores: list[float], *, seed: int, samples: int = 10000) -> list[float]:
    rng = random.Random(seed)
    n = len(pair_scores)
    means = []
    for _ in range(samples):
        means.append(sum(pair_scores[rng.randrange(n)] for _ in range(n)) / n)
    return [percentile(means, 0.025), percentile(means, 0.975)]


def summarize(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    config = report["config"]
    if config.get("progressive_widening") is not False:
        raise ValueError(f"{path}: expected progressive_widening=false")
    if config.get("c_puct") != 1.5 or config.get("selection_temperature") != 0.0:
        raise ValueError(f"{path}: unexpected PUCT or selection configuration")
    if config.get("category_exploration") != 0.0 or config.get("repetition_penalty") != 0.0:
        raise ValueError(f"{path}: optional exploration/repetition settings are not zero")
    if config.get("pawn_progress_bonus") != 0.0:
        raise ValueError(f"{path}: pawn_progress_bonus is not zero")

    games = report["games"]
    pairs = report["paired_seed_results"]
    expected_games = 2 * len(pairs)
    if len(games) != expected_games or config.get("games") != expected_games:
        raise ValueError(f"{path}: incomplete games/pair results")

    seat_map: dict[int, list[int]] = {}
    for game in games:
        seat_map.setdefault(int(game["pair"]), []).append(int(game["low_player"]))
    for pair_no, seats in seat_map.items():
        if sorted(seats) != [0, 1]:
            raise ValueError(f"{path}: pair {pair_no} does not contain both swapped seats")

    pair_scores = []
    low_wins = low_losses = pair_ties = 0
    seeds = []
    for pair in pairs:
        outcome = pair["pair_result_for_low"]
        if outcome == "win":
            low_wins += 1
        elif outcome == "loss":
            low_losses += 1
        elif outcome == "draw":
            pair_ties += 1
        else:
            raise ValueError(f"{path}: unknown paired result {outcome!r}")
        pair_scores.append(float(pair["high_points"]) / 2.0)
        seeds.append(int(pair["seed"]))

    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path}: duplicate pair seeds")
    high_wins, high_losses = low_losses, low_wins
    decisive = high_wins + high_losses
    p_value = upper_binomial_tail(high_wins, decisive)
    score_rate = sum(pair_scores) / len(pair_scores)
    ci = bootstrap_pair_score_ci(pair_scores, seed=int(config["base_seed"]) ^ 0xB00757)

    stem = path.stem
    return {
        "comparison": stem,
        "checkpoint": report["checkpoint"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "low_budget": int(config["low_simulations"]),
        "high_budget": int(config["high_simulations"]),
        "pairs": len(pairs),
        "games": len(games),
        "high_pair_wins": high_wins,
        "low_pair_wins": high_losses,
        "paired_ties": pair_ties,
        "decisive_pairs": decisive,
        "high_score_rate": score_rate,
        "high_score_rate_bootstrap_95ci": ci,
        "high_vs_low_one_sided_sign_p": p_value,
        "mean_plies": sum(int(g["plies"]) for g in games) / len(games),
        "game_draws": sum(g["result_for_low"] == "draw" for g in games),
        "base_seed": int(config["base_seed"]),
    }


def holm_adjust(items: list[dict]) -> None:
    selected = [item for item in items if item["comparison"] in PRIMARY]
    if len(selected) != len(PRIMARY):
        missing = sorted(PRIMARY - {item["comparison"] for item in selected})
        raise ValueError(f"Missing primary comparisons: {missing}")
    ordered = sorted(selected, key=lambda x: x["high_vs_low_one_sided_sign_p"])
    running = 0.0
    m = len(ordered)
    for rank, item in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * item["high_vs_low_one_sided_sign_p"])
        running = max(running, adjusted)
        item["holm_adjusted_p_primary_family"] = running
    for item in items:
        item.setdefault("holm_adjusted_p_primary_family", None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=Path("data/diagnostics/budget_scaling_paired_games_20261118"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/diagnostics/budget_scaling_paired_games_20261118_summary.json"))
    parser.add_argument("--csv", type=Path,
                        default=Path("data/diagnostics/budget_scaling_paired_games_20261118_summary.csv"))
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*.json"))
    if len(paths) != 6:
        raise ValueError(f"Expected six matchup reports in {args.input_dir}; found {len(paths)}")
    summaries = [summarize(path) for path in paths]
    holm_adjust(summaries)

    output = {
        "experiment": "budget_scaling_paired_game_outcome_validation",
        "source_reports": [str(path) for path in paths],
        "primary_tests": sorted(PRIMARY),
        "multiple_testing_correction": "Holm step-down across four primary one-sided paired sign tests",
        "confidence_interval": "95% percentile bootstrap over paired-seed high-agent score rates; 10,000 resamples",
        "results": summaries,
        "interpretation_guardrail": (
            "Evaluation only. No MCTS tuning or checkpoint promotion is authorized by this report. "
            "Paired seeds, not individual games, are the inferential units."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    columns = [
        "comparison", "checkpoint_sha256", "low_budget", "high_budget",
        "pairs", "games", "high_pair_wins", "low_pair_wins", "paired_ties",
        "decisive_pairs", "high_score_rate", "high_score_rate_ci_low",
        "high_score_rate_ci_high", "high_vs_low_one_sided_sign_p",
        "holm_adjusted_p_primary_family", "mean_plies", "game_draws",
    ]
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in summaries:
            writer.writerow({
                **row,
                "high_score_rate_ci_low": row["high_score_rate_bootstrap_95ci"][0],
                "high_score_rate_ci_high": row["high_score_rate_bootstrap_95ci"][1],
            })
    print(f"JSON: {args.output}")
    print(f"CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
