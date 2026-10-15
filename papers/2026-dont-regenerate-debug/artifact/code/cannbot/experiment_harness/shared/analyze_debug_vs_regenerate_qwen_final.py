#!/usr/bin/env python3
"""Build the final same-model Debug-Agent vs Regenerate comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [center - half, center + half]


def exact_mcnemar_two_sided(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    smaller = min(left_only, right_only)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (
        2**discordant
    )
    return min(1.0, 2 * tail)


def pct(value: float) -> float:
    return round(100 * value, 4)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate-json", type=Path, required=True)
    parser.add_argument("--debug-posthoc-json", type=Path, required=True)
    parser.add_argument("--debug-cost-csv", type=Path, required=True)
    parser.add_argument("--debug-arm-manifest", type=Path, required=True)
    parser.add_argument("--manual-review-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manual_rows = load_csv(args.manual_review_csv)
    manual: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in manual_rows:
        manual[(row["method"], row["operator"], int(row["trial"]))] = row

    regenerate_payload = load_json(args.regenerate_json)
    regenerate_rows = regenerate_payload["rows"]
    for row in regenerate_rows:
        review = manual.get(
            ("regenerate", str(row["operator"]), int(row["trial"]))
        )
        row["automatic_strict_success"] = (
            row["full_eval_result"] == "pass"
            and row["anticheat_verdict"] == "CLEAN"
        )
        row["manual_anticheat_verdict"] = (
            review["manual_verdict"]
            if review
            else ("NOT_REVIEWED" if row["full_eval_result"] != "pass" else "MISSING")
        )
        row["manual_strict_success"] = (
            row["full_eval_result"] == "pass"
            and row["manual_anticheat_verdict"] == "CLEAN"
        )

    debug_posthoc = load_json(args.debug_posthoc_json)["rows"]
    debug_cost = {row["task"]: row for row in load_csv(args.debug_cost_csv)}
    debug_rows: list[dict[str, Any]] = []
    for row in debug_posthoc:
        operator = str(row["task"])
        review = manual.get(("debug-agent", operator, 1))
        harness_verdict = str(row["anticheat"]["verdict"])
        manual_verdict = (
            review["manual_verdict"]
            if review
            else ("CLEAN" if harness_verdict == "CLEAN" else "NOT_REVIEWED")
        )
        full_eval_pass = bool(row["verification"]["objective_passed"])
        cost = debug_cost[operator]
        debug_rows.append(
            {
                "operator": operator,
                "trial": 1,
                "full_eval_result": "pass" if full_eval_pass else "fail",
                "harness_anticheat_verdict": harness_verdict,
                "manual_anticheat_verdict": manual_verdict,
                "automatic_strict_success": bool(row["posthoc_clean_success"]),
                "manual_strict_success": full_eval_pass
                and manual_verdict == "CLEAN",
                "turns": int(cost["turns"]),
                "input_tokens": int(cost["input_tokens"]),
                "output_tokens": int(cost["output_tokens"]),
                "cache_creation_tokens": int(cost["cache_creation_tokens"]),
                "cache_read_tokens": int(cost["cache_read_tokens"]),
                "total_tokens_uncached": int(cost["input_tokens"])
                + int(cost["output_tokens"]),
                "total_tokens_all": int(cost["total_tokens"]),
                "cli_reported_cost_usd": float(cost["cost_usd"]),
            }
        )

    regenerate_by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in regenerate_rows:
        regenerate_by_operator[str(row["operator"])].append(row)
    debug_by_operator = {str(row["operator"]): row for row in debug_rows}

    operators = sorted(debug_by_operator)
    if operators != sorted(regenerate_by_operator):
        raise SystemExit("Debug and Regenerate operator cohorts do not match")

    paired: list[dict[str, Any]] = []
    for operator in operators:
        regen = sorted(regenerate_by_operator[operator], key=lambda row: row["trial"])
        debug = debug_by_operator[operator]
        paired.append(
            {
                "operator": operator,
                "debug_auto_success": debug["automatic_strict_success"],
                "debug_manual_success": debug["manual_strict_success"],
                "regenerate_auto_pass_at_3": any(
                    row["automatic_strict_success"] for row in regen
                ),
                "regenerate_manual_pass_at_3": any(
                    row["manual_strict_success"] for row in regen
                ),
                "regenerate_manual_successful_trials": sum(
                    row["manual_strict_success"] for row in regen
                ),
                "regenerate_full_eval_pass_trials": sum(
                    row["full_eval_result"] == "pass" for row in regen
                ),
                "regenerate_trial1_manual_success": regen[0][
                    "manual_strict_success"
                ],
                "regenerate_trial2_manual_success": regen[1][
                    "manual_strict_success"
                ],
                "regenerate_trial3_manual_success": regen[2][
                    "manual_strict_success"
                ],
            }
        )

    debug_auto = sum(row["automatic_strict_success"] for row in debug_rows)
    debug_manual = sum(row["manual_strict_success"] for row in debug_rows)
    regen_auto_runs = sum(row["automatic_strict_success"] for row in regenerate_rows)
    regen_manual_runs = sum(row["manual_strict_success"] for row in regenerate_rows)
    regen_auto_ops = sum(row["regenerate_auto_pass_at_3"] for row in paired)
    regen_manual_ops = sum(row["regenerate_manual_pass_at_3"] for row in paired)

    debug_only = sum(
        row["debug_manual_success"] and not row["regenerate_manual_pass_at_3"]
        for row in paired
    )
    regen_only = sum(
        row["regenerate_manual_pass_at_3"] and not row["debug_manual_success"]
        for row in paired
    )
    both = sum(
        row["debug_manual_success"] and row["regenerate_manual_pass_at_3"]
        for row in paired
    )
    neither = len(paired) - debug_only - regen_only - both

    debug_totals = {
        key: sum(row[key] for row in debug_rows)
        for key in (
            "turns",
            "input_tokens",
            "output_tokens",
            "cache_creation_tokens",
            "cache_read_tokens",
            "total_tokens_uncached",
            "total_tokens_all",
            "cli_reported_cost_usd",
        )
    }
    regen_totals = regenerate_payload["summary"]["totals"]

    def safe_ratio(numerator: float, denominator: float) -> float:
        return round(numerator / denominator, 6)

    cost = {
        "debug_total": debug_totals,
        "regenerate_total_3_trials": regen_totals,
        "per_operator": {
            "debug": {
                key: debug_totals[key] / 27 for key in debug_totals
            },
            "regenerate_pass_at_3_budget": {
                key: regen_totals[key] / 27 for key in debug_totals
            },
            "regenerate_single_trial_average": {
                key: regen_totals[key] / 81 for key in debug_totals
            },
        },
        "ratios": {
            "regenerate_pass_at_3_over_debug": {
                key: safe_ratio(regen_totals[key], debug_totals[key])
                for key in debug_totals
            },
            "regenerate_single_trial_over_debug": {
                key: safe_ratio(regen_totals[key] / 3, debug_totals[key])
                for key in debug_totals
            },
        },
        "per_manual_successful_operator": {
            "debug": {
                key: debug_totals[key] / debug_manual for key in debug_totals
            },
            "regenerate": {
                key: regen_totals[key] / regen_manual_ops for key in debug_totals
            },
            "regenerate_over_debug": {
                key: safe_ratio(
                    regen_totals[key] / regen_manual_ops,
                    debug_totals[key] / debug_manual,
                )
                for key in debug_totals
            },
        },
        "monetary_cost_note": (
            "Both sides use Qwen subscription-credit CLI cost estimates; "
            "the Regenerate audit marks 0/81 rows as attributable billing. "
            "Dollar values are retained only as a secondary CLI proxy."
        ),
    }

    debug_manifest = load_json(args.debug_arm_manifest)
    summary = {
        "cohort": {
            "operators": 27,
            "debug_runs": 27,
            "regenerate_runs": 81,
            "model": "qwen3.8-max-preview",
            "regenerate_qwen_control_replacements": 14,
        },
        "coverage": {
            "debug_terminal": len(debug_rows),
            "debug_posthoc_full_eval": len(debug_rows),
            "regenerate_terminal": len(regenerate_rows),
            "regenerate_full_eval": len(regenerate_rows),
            "regenerate_anticheat": len(regenerate_rows),
        },
        "automatic": {
            "debug_strict_success": debug_auto,
            "debug_total": 27,
            "regenerate_strict_success_runs": regen_auto_runs,
            "regenerate_total_runs": 81,
            "regenerate_pass_at_3_operators": regen_auto_ops,
            "regenerate_total_operators": 27,
        },
        "manual": {
            "debug_strict_success": debug_manual,
            "debug_rate_pct": pct(debug_manual / 27),
            "debug_wilson_95": [pct(x) for x in wilson(debug_manual, 27)],
            "regenerate_strict_success_runs": regen_manual_runs,
            "regenerate_run_rate_pct": pct(regen_manual_runs / 81),
            "regenerate_trial_rates": {
                str(trial): {
                    "success": sum(
                        row["manual_strict_success"]
                        and int(row["trial"]) == trial
                        for row in regenerate_rows
                    ),
                    "total": 27,
                }
                for trial in (1, 2, 3)
            },
            "regenerate_pass_at_3_operators": regen_manual_ops,
            "regenerate_pass_at_3_rate_pct": pct(regen_manual_ops / 27),
            "regenerate_pass_at_3_wilson_95": [
                pct(x) for x in wilson(regen_manual_ops, 27)
            ],
            "absolute_rate_difference_debug_minus_regen_pct_points": round(
                100 * (debug_manual - regen_manual_ops) / 27, 4
            ),
            "relative_risk_debug_over_regen": safe_ratio(
                debug_manual / 27, regen_manual_ops / 27
            ),
        },
        "paired": {
            "both_success": both,
            "debug_only_success": debug_only,
            "regenerate_only_success": regen_only,
            "neither_success": neither,
            "paired_odds_ratio_debug_over_regenerate": safe_ratio(
                debug_only, regen_only
            ),
            "mcnemar_exact_two_sided_p": exact_mcnemar_two_sided(
                debug_only, regen_only
            ),
        },
        "anti_cheat": {
            "regenerate_official": regenerate_payload["summary"][
                "official_anticheat"
            ],
            "regenerate_harness": regenerate_payload["summary"]["anticheat"],
            "regenerate_compatibility_overrides": regenerate_payload["summary"][
                "compatibility_override_applied"
            ],
            "regenerate_full_eval_pass_manual_reviewed": regen_manual_runs,
            "regenerate_pass_candidates_confirmed_true_cheat": 0,
            "debug_harness_clean": sum(
                row["harness_anticheat_verdict"] == "CLEAN" for row in debug_rows
            ),
            "debug_harness_cheat": sum(
                row["harness_anticheat_verdict"] == "CHEAT" for row in debug_rows
            ),
            "debug_harness_skipped": sum(
                row["harness_anticheat_verdict"] == "SKIPPED" for row in debug_rows
            ),
            "debug_manual_false_positive": 1,
            "debug_manual_confirmed_true_cheat": 1,
        },
        "cost": cost,
        "debug_configuration": {
            key: debug_manifest.get(key)
            for key in (
                "max_attempts",
                "max_task_turns",
                "max_turns",
                "soft_task_turns",
                "timeout",
                "model_context_window",
                "primary_cost_scope",
                "failed_cycles_excluded_from_primary_cost",
            )
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "paired_operator_summary.csv", paired)
    write_csv(args.output_dir / "debug_agent_qwen_27.csv", debug_rows)
    write_csv(args.output_dir / "regenerate_qwen_81_adjudicated.csv", regenerate_rows)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
