#!/usr/bin/env python3
"""Reduce the large post-wall pawn regression audit into compact aggregate metrics.

Uses ijson so the 450 MB diagnostic report is processed incrementally.
No checkpoints, source reports, or experiment outputs are modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import ijson


VARIANTS = ("vanilla_puct", "with_repetition_avoidance")


def number(value):
    """Convert JSON numeric values (including ijson Decimal) to float."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def candidate_rows(snapshot):
    if not isinstance(snapshot, dict):
        return []
    rows = snapshot.get("candidate_pawn_actions")
    return rows if isinstance(rows, list) else []


def best_by(rows, key):
    valid = [row for row in rows if number(row.get(key)) is not None]
    return max(valid, key=lambda row: number(row[key])) if valid else None


def action_id(row):
    value = row.get("action", row.get("action_index"))
    if value is None:
        return None
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def final_candidate_map(variant):
    """Map action -> final distance category using the last root trace snapshot."""
    trace = variant.get("selection_trace") or []
    if not trace:
        return {}
    rows = candidate_rows(trace[-1])
    result = {}
    for row in rows:
        aid = action_id(row)
        delta = number(row.get("distance_delta"))
        if aid is None or delta is None:
            continue
        result[aid] = "regression" if delta < 0 else "progress" if delta > 0 else "neutral"
    return result


def visit_totals(variant):
    """Sum final visits across every root pawn edge.

    Do not assume the last trace row is a complete final snapshot. Trace rows
    are per-selection snapshots; collect the greatest post-backup visit count
    observed for each action across all root-depth rows. Edge visits are
    monotonic, so the maximum is the final count. Fall back to the pre-selection
    visits field for older traces that predate visits_after_backup.
    """
    trace = variant.get("selection_trace") or []
    totals = {"regression": 0, "progress": 0}
    final_by_action = {}
    for snapshot in trace:
        if snapshot.get("selection_depth") != 0:
            continue
        for row in candidate_rows(snapshot):
            aid = action_id(row)
            delta = number(row.get("distance_delta"))
            if aid is None or delta is None:
                continue
            category = "regression" if delta < 0 else "progress" if delta > 0 else None
            if category is None:
                continue
            raw = row.get("visits_after_backup")
            if raw is None:
                raw = row.get("visits", 0)
            count = number(raw)
            if count is None:
                continue
            previous = final_by_action.get(aid)
            if previous is None or count > previous[1]:
                final_by_action[aid] = (category, int(count))
    for category, count in final_by_action.values():
        totals[category] += count
    return totals


def first_selection_comparison(variant):
    rows = candidate_rows(variant.get("first_root_snapshot"))
    regressions = [r for r in rows if (number(r.get("distance_delta")) or 0) < 0]
    progresses = [r for r in rows if (number(r.get("distance_delta")) or 0) > 0]
    best_reg = best_by(regressions, "puct_score")
    best_prog = best_by(progresses, "puct_score")
    if best_reg is None or best_prog is None:
        return None
    reg_score, prog_score = number(best_reg.get("puct_score")), number(best_prog.get("puct_score"))
    return {
        "regression_ahead": reg_score > prog_score,
        "regression_puct": reg_score,
        "progress_puct": prog_score,
        "regression_prior": number(best_reg.get("prior")),
        "progress_prior": number(best_prog.get("prior")),
        "regression_q": number(best_reg.get("q")),
        "progress_q": number(best_prog.get("q")),
    }


def final_comparison(variant):
    reg = variant.get("best_final_regression") or {}
    prog = variant.get("best_final_progress") or {}
    if not reg or not prog:
        return None
    return {
        "regression_prior": number(reg.get("prior")),
        "progress_prior": number(prog.get("prior")),
        "regression_q": number(reg.get("q")),
        "progress_q": number(prog.get("q")),
        "regression_visits": number(reg.get("visits")),
        "progress_visits": number(prog.get("visits")),
        "regression_puct": number(reg.get("puct_score")),
        "progress_puct": number(prog.get("puct_score")),
        "regression_q_better": (
            number(reg.get("q")) is not None and number(prog.get("q")) is not None
            and number(reg.get("q")) > number(prog.get("q"))
        ),
    }


def add_metric(bucket, name, value):
    if value is not None:
        bucket[name].append(value)


def summarize_variant(rows, variant_name):
    metrics = defaultdict(list)
    counts = defaultdict(int)
    position_rows = []
    probe_deltas = []
    for run in rows:
        variant = (run.get("variants") or {}).get(variant_name) or {}
        first = first_selection_comparison(variant)
        final = final_comparison(variant)
        visits = visit_totals(variant)
        counts["positions"] += 1
        if first:
            counts["positions_with_first_category_comparison"] += 1
            counts["first_selection_regression_ahead"] += int(first["regression_ahead"])
            for key in ("regression_prior", "progress_prior"):
                add_metric(metrics, "first_" + key, first[key])
        if final:
            counts["positions_with_final_category_comparison"] += 1
            counts["final_q_regression_better"] += int(final["regression_q_better"])
            for key in (
                "regression_prior", "progress_prior", "regression_q", "progress_q",
                "regression_visits", "progress_visits", "regression_puct", "progress_puct",
            ):
                add_metric(metrics, "final_" + key, final[key])
            if final["regression_prior"] is not None and final["progress_prior"] is not None:
                add_metric(metrics, "final_prior_difference_reg_minus_prog",
                           final["regression_prior"] - final["progress_prior"])
            if final["regression_q"] is not None and final["progress_q"] is not None:
                add_metric(metrics, "final_q_difference_reg_minus_prog",
                           final["regression_q"] - final["progress_q"])
        counts["total_regression_visits"] += visits["regression"]
        counts["total_progress_visits"] += visits["progress"]
        if visits["regression"] + visits["progress"] > 0:
            add_metric(metrics, "per_position_regression_progress_visit_ratio",
                       visits["regression"] / visits["progress"] if visits["progress"] else None)

        probe = variant.get("response_probe") or {}
        reg_probe = probe.get("best_regression") or {}
        prog_probe = probe.get("best_progress") or {}
        reg_value = number(reg_probe.get("value_for_original_player"))
        prog_value = number(prog_probe.get("value_for_original_player"))
        if reg_value is not None and prog_value is not None:
            delta = reg_value - prog_value
            probe_deltas.append(delta)
            counts["positions_with_both_response_probes"] += 1
            counts["response_probe_regression_better"] += int(delta > 0)
            counts["response_probe_progress_better"] += int(delta < 0)
            counts["response_probe_equal"] += int(abs(delta) <= 1e-12)
            position_rows.append({
                "source_game_index": (run.get("source") or {}).get("source_game_index"),
                "ply": (run.get("source") or {}).get("ply"),
                "variant": variant_name,
                "regression_response_value": reg_value,
                "progress_response_value": prog_value,
                "regression_minus_progress": delta,
                "regression_reply": reg_probe.get("opponent_reply"),
                "progress_reply": prog_probe.get("opponent_reply"),
            })

    n_first = counts["positions_with_first_category_comparison"]
    n_final = counts["positions_with_final_category_comparison"]
    total_reg = counts["total_regression_visits"]
    total_prog = counts["total_progress_visits"]
    n_probe = counts["positions_with_both_response_probes"]
    return {
        "positions": counts["positions"],
        "first_selection": {
            "comparable_positions": n_first,
            "regression_ahead_count": counts["first_selection_regression_ahead"],
            "regression_ahead_percent": (
                100 * counts["first_selection_regression_ahead"] / n_first if n_first else None
            ),
            "mean_best_prior": {
                "regression": mean(metrics["first_regression_prior"]) if metrics["first_regression_prior"] else None,
                "progress": mean(metrics["first_progress_prior"]) if metrics["first_progress_prior"] else None,
            },
        },
        "final_search": {
            "comparable_positions": n_final,
            "mean_best_prior": {
                "regression": mean(metrics["final_regression_prior"]) if metrics["final_regression_prior"] else None,
                "progress": mean(metrics["final_progress_prior"]) if metrics["final_progress_prior"] else None,
                "difference_regression_minus_progress": (
                    mean(metrics["final_prior_difference_reg_minus_prog"])
                    if metrics["final_prior_difference_reg_minus_prog"] else None
                ),
            },
            "mean_best_q": {
                "regression": mean(metrics["final_regression_q"]) if metrics["final_regression_q"] else None,
                "progress": mean(metrics["final_progress_q"]) if metrics["final_progress_q"] else None,
                "difference_regression_minus_progress": (
                    mean(metrics["final_q_difference_reg_minus_prog"])
                    if metrics["final_q_difference_reg_minus_prog"] else None
                ),
            },
            "regression_q_better_count": counts["final_q_regression_better"],
            "regression_q_better_percent": (
                100 * counts["final_q_regression_better"] / n_final if n_final else None
            ),
            "mean_best_action_visits": {
                "regression": mean(metrics["final_regression_visits"]) if metrics["final_regression_visits"] else None,
                "progress": mean(metrics["final_progress_visits"]) if metrics["final_progress_visits"] else None,
            },
        },
        "aggregate_pawn_visit_totals": {
            "regression": total_reg,
            "progress": total_prog,
            "regression_to_progress_ratio": total_reg / total_prog if total_prog else None,
            "regression_visit_share_percent": (
                100 * total_reg / (total_reg + total_prog) if total_reg + total_prog else None
            ),
        },
        "response_probe": {
            "comparable_positions": n_probe,
            "mean_regression_value": mean([
                row["regression_response_value"] for row in position_rows
            ]) if position_rows else None,
            "mean_progress_value": mean([
                row["progress_response_value"] for row in position_rows
            ]) if position_rows else None,
            "mean_value_difference_regression_minus_progress": (
                mean(probe_deltas) if probe_deltas else None
            ),
            "regression_better_count": counts["response_probe_regression_better"],
            "progress_better_count": counts["response_probe_progress_better"],
            "equal_count": counts["response_probe_equal"],
            "regression_better_percent": (
                100 * counts["response_probe_regression_better"] / n_probe if n_probe else None
            ),
            "interpretation": (
                "One-ply opponent MCTS response plus unchanged neural value; diagnostic proxy, "
                "not minimax proof. Only roots where both probes were recorded are included."
            ),
        },
        "_response_probe_rows": position_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="post_wall_pawn_regression_audit.json")
    parser.add_argument("--output", type=Path, required=True, help="Compact summary JSON path.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional response-probe per-position CSV.")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input report not found: {args.input}")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a fresh path.")
    if args.csv and args.csv.exists():
        parser.error(f"CSV already exists: {args.csv}; choose a fresh path.")

    summaries = {}
    probe_rows = []
    run_count = 0
    for variant_name in VARIANTS:
        summaries[variant_name] = None

    # Stream each root exactly once; retain only compact accumulators.
    accumulators = {
        name: {
            "metrics": defaultdict(list), "counts": defaultdict(int),
            "probe_deltas": [], "probe_reg_values": [], "probe_prog_values": [],
            "probe_rows": [],
        } for name in VARIANTS
    }

    # Reuse the same reducer logic incrementally by feeding a one-run iterator.
    # This keeps memory bounded even when the source report is hundreds of MB.
    for run in ijson.items(args.input.open("rb"), "runs.item"):
        run_count += 1
        for variant_name in VARIANTS:
            one = summarize_variant([run], variant_name)
            a = accumulators[variant_name]
            # Merge scalar counts and per-position observations.
            for key in (
                "positions", "positions_with_first_category_comparison",
                "first_selection_regression_ahead", "positions_with_final_category_comparison",
                "final_q_regression_better", "total_regression_visits",
                "total_progress_visits", "positions_with_both_response_probes",
                "response_probe_regression_better", "response_probe_progress_better",
                "response_probe_equal",
            ):
                a["counts"][key] += one[key] if key in one else (
                    one["first_selection"].get("comparable_positions", 0) if key == "positions_with_first_category_comparison" else
                    one["first_selection"].get("regression_ahead_count", 0) if key == "first_selection_regression_ahead" else
                    one["final_search"].get("comparable_positions", 0) if key == "positions_with_final_category_comparison" else
                    one["final_search"].get("regression_q_better_count", 0) if key == "final_q_regression_better" else
                    one["aggregate_pawn_visit_totals"].get("regression", 0) if key == "total_regression_visits" else
                    one["aggregate_pawn_visit_totals"].get("progress", 0) if key == "total_progress_visits" else
                    one["response_probe"].get("comparable_positions", 0) if key == "positions_with_both_response_probes" else
                    one["response_probe"].get("regression_better_count", 0) if key == "response_probe_regression_better" else
                    one["response_probe"].get("progress_better_count", 0) if key == "response_probe_progress_better" else
                    one["response_probe"].get("equal_count", 0)
                )
            for section, keys in (
                ("first_selection", ("regression", "progress")),
                ("final_search", ("regression", "progress")),
            ):
                prior = one[section]["mean_best_prior"]
                for cat in keys:
                    value = prior.get(cat)
                    if value is not None:
                        a["metrics"][section + "_prior_" + cat].append(value)
            for cat in ("regression", "progress"):
                for metric in ("mean_best_q", "mean_best_action_visits"):
                    value = one["final_search"][metric].get(cat)
                    if value is not None:
                        a["metrics"]["final_" + metric + "_" + cat].append(value)
            for metric in ("difference_regression_minus_progress",):
                for section in ("mean_best_q",):
                    value = one["final_search"][section].get(metric)
                    if value is not None:
                        a["metrics"]["final_q_diff"].append(value)
                value = one["final_search"]["mean_best_prior"].get(metric)
                if value is not None:
                    a["metrics"]["final_prior_diff"].append(value)
            ratio = one["aggregate_pawn_visit_totals"].get("regression_to_progress_ratio")
            if ratio is not None:
                a["metrics"]["visit_ratio_per_position"].append(ratio)
            probe = one["response_probe"]
            for metric, key in (
                ("probe_reg", "mean_regression_value"),
                ("probe_prog", "mean_progress_value"),
                ("probe_delta", "mean_value_difference_regression_minus_progress"),
            ):
                if probe.get(key) is not None:
                    a["metrics"][metric].append(probe[key])
            a["probe_rows"].extend(one["_response_probe_rows"])

    for variant_name, a in accumulators.items():
        c, m = a["counts"], a["metrics"]
        n_first = c["positions_with_first_category_comparison"]
        n_final = c["positions_with_final_category_comparison"]
        n_probe = c["positions_with_both_response_probes"]
        reg_visits, prog_visits = c["total_regression_visits"], c["total_progress_visits"]
        summaries[variant_name] = {
            "positions": c["positions"],
            "first_selection": {
                "comparable_positions": n_first,
                "regression_ahead_count": c["first_selection_regression_ahead"],
                "regression_ahead_percent": 100*c["first_selection_regression_ahead"]/n_first if n_first else None,
                "mean_best_prior": {
                    "regression": mean(m["first_selection_prior_regression"]) if m["first_selection_prior_regression"] else None,
                    "progress": mean(m["first_selection_prior_progress"]) if m["first_selection_prior_progress"] else None,
                },
            },
            "final_search": {
                "comparable_positions": n_final,
                "mean_best_prior": {
                    "regression": mean(m["final_search_prior_regression"]) if m["final_search_prior_regression"] else None,
                    "progress": mean(m["final_search_prior_progress"]) if m["final_search_prior_progress"] else None,
                    "difference_regression_minus_progress": mean(m["final_prior_diff"]) if m["final_prior_diff"] else None,
                },
                "mean_best_q": {
                    "regression": mean(m["final_mean_best_q_regression"]) if m["final_mean_best_q_regression"] else None,
                    "progress": mean(m["final_mean_best_q_progress"]) if m["final_mean_best_q_progress"] else None,
                    "difference_regression_minus_progress": mean(m["final_q_diff"]) if m["final_q_diff"] else None,
                },
                "regression_q_better_count": c["final_q_regression_better"],
                "regression_q_better_percent": 100*c["final_q_regression_better"]/n_final if n_final else None,
                "mean_best_action_visits": {
                    "regression": mean(m["final_mean_best_action_visits_regression"]) if m["final_mean_best_action_visits_regression"] else None,
                    "progress": mean(m["final_mean_best_action_visits_progress"]) if m["final_mean_best_action_visits_progress"] else None,
                },
            },
            "aggregate_pawn_visit_totals": {
                "regression": reg_visits, "progress": prog_visits,
                "regression_to_progress_ratio": reg_visits/prog_visits if prog_visits else None,
                "regression_visit_share_percent": 100*reg_visits/(reg_visits+prog_visits) if reg_visits+prog_visits else None,
                "mean_per_position_ratio": mean(m["visit_ratio_per_position"]) if m["visit_ratio_per_position"] else None,
            },
            "response_probe": {
                "comparable_positions": n_probe,
                "mean_regression_value": mean(m["probe_reg"]) if m["probe_reg"] else None,
                "mean_progress_value": mean(m["probe_prog"]) if m["probe_prog"] else None,
                "mean_value_difference_regression_minus_progress": mean(m["probe_delta"]) if m["probe_delta"] else None,
                "regression_better_count": c["response_probe_regression_better"],
                "progress_better_count": c["response_probe_progress_better"],
                "equal_count": c["response_probe_equal"],
                "regression_better_percent": 100*c["response_probe_regression_better"]/n_probe if n_probe else None,
                "interpretation": "One-ply opponent MCTS response plus unchanged neural value; diagnostic proxy, not minimax proof.",
            },
        }
        probe_rows.extend(a["probe_rows"])

    report = {
        "experiment": "post_wall_pawn_regression_audit_summary",
        "source_report": str(args.input),
        "positions_processed": run_count,
        "aggregation_notes": {
            "prior_and_q": "Per-root comparison uses the highest-PUCT regression action and highest-PUCT progress action at the final root snapshot; first-selection priors use the highest-PUCT action in each category in the first snapshot.",
            "visit_ratio": "Aggregate visit ratio sums visits across all regression pawn actions and all progress pawn actions using the final trace's distance_delta labels.",
            "response_probe": "Only positions with both regression and progress response probes are compared. This report's probe count may be limited by the audit's --response-probe-limit.",
            "scope": "Metrics are reported separately for vanilla PUCT and PUCT with repetition avoidance.",
        },
        "variants": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "source_game_index", "ply", "variant", "regression_response_value",
            "progress_response_value", "regression_minus_progress",
            "regression_reply", "progress_reply",
        ]
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(probe_rows)
    print(f"Processed {run_count} roots")
    print(f"Summary: {args.output}")
    if args.csv:
        print(f"Response-probe rows: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
