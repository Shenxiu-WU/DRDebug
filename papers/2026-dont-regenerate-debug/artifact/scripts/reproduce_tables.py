#!/usr/bin/env python3
"""Recompute the principal paper and Supplement statistics from packaged CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPRODUCED = ROOT / "reproduced"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def number(value: object) -> float:
    text = str(value).strip()
    return float(text) if text else 0.0


def by_operator(rows: list[dict[str, str]]) -> dict[str, bool]:
    return {row["operator_id"]: truth(row["pass_at_1"]) for row in rows}


def pair_counts(left: dict[str, bool], right: dict[str, bool]) -> dict[str, int | float]:
    if left.keys() != right.keys():
        raise ValueError("Paired result sets do not contain identical operators")
    both = sum(left[key] and right[key] for key in left)
    left_only = sum(left[key] and not right[key] for key in left)
    right_only = sum(not left[key] and right[key] for key in left)
    neither = len(left) - both - left_only - right_only
    return {
        "both_pass": both,
        "left_only": left_only,
        "right_only": right_only,
        "both_fail": neither,
        "exact_mcnemar_p": exact_mcnemar(left_only, right_only),
    }


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def percentile(sorted_values: list[float], probability: float) -> float:
    index = int(probability * (len(sorted_values) - 1))
    return sorted_values[index]


def paired_bootstrap_interval(
    left: dict[str, bool],
    right: dict[str, bool],
    *,
    samples: int = 100_000,
    seed: int = 20260729,
) -> list[float]:
    keys = sorted(left)
    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        draw = [keys[rng.randrange(len(keys))] for _ in keys]
        differences.append(
            sum(float(right[key]) - float(left[key]) for key in draw) / len(draw)
        )
    differences.sort()
    return [percentile(differences, 0.025), percentile(differences, 0.975)]


def token_summary(rows: list[dict[str, str]], success_count: int | None = None) -> dict[str, float | int]:
    attempted = len(rows)
    successes = success_count if success_count is not None else sum(truth(row["pass_at_1"]) for row in rows)
    uncached = sum(number(row["uncached_tokens"]) for row in rows)
    all_tokens = sum(number(row["all_tokens"]) for row in rows)
    return {
        "attempted_runs": attempted,
        "pass_at_1": successes,
        "uncached_tokens_total": int(uncached),
        "all_tokens_total": int(all_tokens),
        "uncached_tokens_per_attempt": uncached / attempted,
        "all_tokens_per_attempt": all_tokens / attempted,
        "uncached_tokens_per_pass": uncached / successes,
        "all_tokens_per_pass": all_tokens / successes,
    }


def rq1_rq2(debug_q: list[dict[str, str]], regen_q: list[dict[str, str]]) -> tuple[dict, dict]:
    debug = by_operator(debug_q)
    trials: dict[int, dict[str, bool]] = {}
    for trial in (1, 2, 3):
        rows = [row for row in regen_q if int(row["trial"]) == trial]
        trials[trial] = by_operator(rows)
    regen_pass3 = {
        operator: any(trials[trial][operator] for trial in trials)
        for operator in debug
    }
    paired = {
        f"debug_vs_regenerate_trial_{trial}": pair_counts(debug, trials[trial])
        for trial in trials
    }
    paired["debug_vs_regenerate_pass_at_3"] = pair_counts(debug, regen_pass3)

    debug_tokens = token_summary(debug_q)
    regen_average_tokens = token_summary(regen_q)
    regen_pass3_tokens = token_summary(regen_q, sum(regen_pass3.values()))
    outcomes = {
        "debug_agent_pass_at_1": sum(debug.values()),
        "regenerate_trial_passes": {str(trial): sum(values.values()) for trial, values in trials.items()},
        "regenerate_run_passes": sum(sum(values.values()) for values in trials.values()),
        "regenerate_pass_at_3": sum(regen_pass3.values()),
        "paired": paired,
    }
    tokens = {
        "debug_agent_qwen": debug_tokens,
        "regenerate_qwen_all_81_runs": regen_average_tokens,
        "regenerate_qwen_pass_at_3_budget": regen_pass3_tokens,
        "debug_all_token_reduction_vs_regenerate_pass_at_3_per_pass": (
            1.0 - debug_tokens["all_tokens_per_pass"] / regen_pass3_tokens["all_tokens_per_pass"]
        ),
    }
    return outcomes, tokens


def rq3(
    debug_q: list[dict[str, str]],
    debug_k: list[dict[str, str]],
    cbd_q: list[dict[str, str]],
    cbd_k: list[dict[str, str]],
) -> tuple[dict, dict]:
    cells = {
        "D-Q": by_operator(debug_q),
        "D-K": by_operator(debug_k),
        "CBD-Q": by_operator(cbd_q),
        "CBD-K": by_operator(cbd_k),
    }
    counts = {cell: sum(values.values()) for cell, values in cells.items()}
    paired = {
        "D-Q_vs_CBD-Q": pair_counts(cells["D-Q"], cells["CBD-Q"]),
        "D-K_vs_CBD-K": pair_counts(cells["D-K"], cells["CBD-K"]),
        "D-Q_vs_D-K": pair_counts(cells["D-Q"], cells["D-K"]),
        "CBD-Q_vs_CBD-K": pair_counts(cells["CBD-Q"], cells["CBD-K"]),
    }
    n = len(cells["D-Q"])
    factorial = {
        "method_main_effect_pp": 100 * (((counts["D-Q"] - counts["CBD-Q"]) + (counts["D-K"] - counts["CBD-K"])) / (2 * n)),
        "model_main_effect_pp": 100 * (((counts["D-K"] - counts["D-Q"]) + (counts["CBD-K"] - counts["CBD-Q"])) / (2 * n)),
        "method_by_model_interaction_pp": 100 * (((counts["D-K"] - counts["CBD-K"]) - (counts["D-Q"] - counts["CBD-Q"])) / n),
    }
    tokens = {
        "D-Q": token_summary(debug_q),
        "D-K": token_summary(debug_k),
        "CBD-Q": token_summary(cbd_q),
        "CBD-K": token_summary(cbd_k),
    }
    return {"cell_pass_at_1": counts, "paired": paired, "factorial": factorial}, tokens


def rq4(ablation: list[dict[str, str]]) -> tuple[dict, dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ablation:
        grouped[row["arm"]].append(row)
    full = by_operator(grouped["full"])
    outcomes = {}
    tokens = {}
    common_success = {}
    for arm in sorted(grouped):
        values = by_operator(grouped[arm])
        paired = pair_counts(full, values)
        paired["paired_bootstrap_95"] = paired_bootstrap_interval(full, values)
        outcomes[arm] = {
            "pass_at_1": sum(values.values()),
            "long_failures": sum(truth(row["long_failure"]) for row in grouped[arm]),
            "paired_vs_full": paired,
        }
        tokens[arm] = token_summary(grouped[arm])
        common = sorted(operator for operator in full if full[operator] and values[operator])
        full_rows = {row["operator_id"]: row for row in grouped["full"]}
        arm_rows = {row["operator_id"]: row for row in grouped[arm]}

        def mean(rows: dict[str, dict[str, str]], field: str) -> float:
            return sum(number(rows[operator][field]) for operator in common) / len(common)

        common_success[arm] = {
            "operators": len(common),
            "full_mean_turns": mean(full_rows, "turns"),
            "arm_mean_turns": mean(arm_rows, "turns"),
            "full_mean_all_tokens": mean(full_rows, "all_tokens"),
            "arm_mean_all_tokens": mean(arm_rows, "all_tokens"),
            "full_mean_recorded_cost_proxy": mean(full_rows, "recorded_cost_proxy"),
            "arm_mean_recorded_cost_proxy": mean(arm_rows, "recorded_cost_proxy"),
        }
    return {"arms": outcomes, "common_success_sensitivity": common_success}, tokens


def validate(summary: dict) -> None:
    checks = {
        "rq1.debug": summary["rq1_rq2"]["debug_agent_pass_at_1"] == 18,
        "rq1.regen_runs": summary["rq1_rq2"]["regenerate_run_passes"] == 21,
        "rq1.regen_pass3": summary["rq1_rq2"]["regenerate_pass_at_3"] == 11,
        "rq3.cells": summary["rq3"]["cell_pass_at_1"] == {"D-Q": 18, "D-K": 21, "CBD-Q": 12, "CBD-K": 16},
        "rq4.full": summary["rq4"]["arms"]["full"]["pass_at_1"] == 18,
        "rq4.minus_kb": summary["rq4"]["arms"]["minus_kb"]["pass_at_1"] == 13,
        "rq4.minus_diagnostic": summary["rq4"]["arms"]["minus_diagnostic"]["pass_at_1"] == 15,
        "rq4.minus_adaptive_scheduling": summary["rq4"]["arms"]["minus_adaptive_scheduling"]["pass_at_1"] == 18,
        "rq4.minus_anti_cheat": summary["rq4"]["arms"]["minus_anti_cheat"]["pass_at_1"] == 14,
        "rq4.minus_full_eval": summary["rq4"]["arms"]["minus_full_evaluation_shadow"]["pass_at_1"] == 12,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"Reproduction checks failed: {failures}")


def markdown(summary: dict) -> str:
    rq1 = summary["rq1_rq2"]
    rq3_data = summary["rq3"]
    lines = [
        "# Reproduced Key Results",
        "",
        "## RQ1--RQ2",
        "",
        "| Endpoint | Passes |",
        "|---|---:|",
        f"| Debug-Agent Qwen Pass@1 | {rq1['debug_agent_pass_at_1']}/27 |",
        f"| Regenerate Qwen run successes | {rq1['regenerate_run_passes']}/81 |",
        f"| Regenerate Qwen Pass@3 | {rq1['regenerate_pass_at_3']}/27 |",
        "",
        "## RQ3 Balanced Method-by-Model Cells",
        "",
        "| Cell | Pass@1 |",
        "|---|---:|",
    ]
    for cell in ("D-Q", "D-K", "CBD-Q", "CBD-K"):
        lines.append(f"| {cell} | {rq3_data['cell_pass_at_1'][cell]}/27 |")
    lines.extend([
        "",
        "## RQ4 Component Configurations",
        "",
        "| Configuration | Pass@1 | Long failures |",
        "|---|---:|---:|",
    ])
    for arm, values in summary["rq4"]["arms"].items():
        lines.append(f"| {arm} | {values['pass_at_1']}/27 | {values['long_failures']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-canonical", action="store_true")
    args = parser.parse_args()

    debug_q = read_csv("rq1_rq2_debug_agent_qwen_27.csv")
    regen_q = read_csv("rq1_rq2_regenerate_qwen_81.csv")
    debug_k = read_csv("rq3_debug_agent_kimi_27.csv")
    cbd_q = read_csv("rq3_cannbot_debug_qwen_27.csv")
    cbd_k = read_csv("rq3_cannbot_debug_kimi_27.csv")
    ablation = read_csv("rq4_ablation_first6_per_task.csv")

    rq12_outcomes, rq12_tokens = rq1_rq2(debug_q, regen_q)
    rq3_outcomes, rq3_tokens = rq3(debug_q, debug_k, cbd_q, cbd_k)
    rq4_outcomes, rq4_tokens = rq4(ablation)
    summary = {
        "schema_version": 1,
        "rq1_rq2": rq12_outcomes,
        "rq3": rq3_outcomes,
        "rq4": rq4_outcomes,
        "token_accounting": {"rq1_rq2": rq12_tokens, "rq3": rq3_tokens, "rq4": rq4_tokens},
    }
    validate(summary)

    REPRODUCED.mkdir(exist_ok=True)
    (REPRODUCED / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (REPRODUCED / "tables.md").write_text(markdown(summary), encoding="utf-8")
    if args.refresh_canonical:
        paired = {
            "schema_version": 1,
            "rq1_rq2": rq12_outcomes["paired"],
            "rq3": rq3_outcomes["paired"],
            "rq3_factorial": rq3_outcomes["factorial"],
            "rq4": {arm: value["paired_vs_full"] for arm, value in rq4_outcomes["arms"].items()},
        }
        (RESULTS / "paired_statistics.json").write_text(json.dumps(paired, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (RESULTS / "token_accounting_summary.json").write_text(
            json.dumps(summary["token_accounting"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print("All canonical reproduction checks passed")


if __name__ == "__main__":
    main()
