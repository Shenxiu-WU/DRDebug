#!/usr/bin/env python3
"""Build preregistered paired statistics and level-stratified summaries."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, value) for value in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def wilcoxon_exact(differences: list[float]) -> dict[str, Any]:
    nonzero = [value for value in differences if value != 0]
    if not nonzero:
        return {"n": 0, "w_plus": 0.0, "w_minus": 0.0, "two_sided_p": 1.0}
    order = sorted(range(len(nonzero)), key=lambda index: abs(nonzero[index]))
    doubled_ranks = [0] * len(nonzero)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and abs(nonzero[order[end]]) == abs(nonzero[order[cursor]]):
            end += 1
        # Average 1-based rank, multiplied by two to retain integer DP states.
        doubled = (cursor + 1) + end
        for position in range(cursor, end):
            doubled_ranks[order[position]] = doubled
        cursor = end
    positive = sum(rank for rank, value in zip(doubled_ranks, nonzero) if value > 0)
    total = sum(doubled_ranks)
    counts = {0: 1}
    for rank in doubled_ranks:
        updated = dict(counts)
        for subtotal, count in counts.items():
            updated[subtotal + rank] = updated.get(subtotal + rank, 0) + count
        counts = updated
    outcomes = 2 ** len(doubled_ranks)
    lower = sum(count for subtotal, count in counts.items() if subtotal <= positive) / outcomes
    upper = sum(count for subtotal, count in counts.items() if subtotal >= positive) / outcomes
    return {
        "n": len(nonzero),
        "w_plus": positive / 2,
        "w_minus": (total - positive) / 2,
        "two_sided_p": min(1.0, 2.0 * min(lower, upper)),
    }


def attributable_cost(rows: list[dict[str, Any]]) -> float | None:
    token_rows = [row for row in rows if int(row.get("total_tokens_all") or 0) > 0]
    if any(not row.get("cost_attributable") for row in token_rows):
        return None
    return sum(float(row.get("cost_usd") or 0) for row in rows)


def main() -> int:
    rows = json.loads((HERE / "token_cost_summary.json").read_text())["rows"]
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["operator"]][row["arm"]].append(row)

    pairs: list[dict[str, Any]] = []
    mixed_regen_rows: list[dict[str, Any]] = []
    for operator, arms in sorted(grouped.items()):
        debug = [row for row in arms.get("debug", []) if row.get("same_model_eligible")]
        all_regen = arms.get("regenerate", [])
        mixed_regen = sorted(
            [row for row in all_regen if not row.get("is_operational_retry")],
            key=lambda row: int(row.get("trial") or 0),
        )
        mixed_regen_rows.extend(mixed_regen)
        regen = [row for row in mixed_regen if row.get("same_model_eligible")]
        operational = [row for row in all_regen if row.get("is_operational_retry")]
        if not debug or not regen:
            continue
        clean_trials = [int(row["trial"]) for row in regen if row.get("external_clean")]
        first_clean = min(clean_trials) if clean_trials else None
        regen_to_clean = sum(
            int(row.get("total_tokens_uncached") or 0)
            for row in regen
            if first_clean is None or int(row.get("trial") or 0) <= first_clean
        )
        pairs.append(
            {
                "operator": operator,
                "level": operator.split("/", 1)[0],
                "debug_clean": any(row.get("external_clean") for row in debug),
                "debug_uncached_tokens": sum(int(row.get("total_tokens_uncached") or 0) for row in debug),
                "debug_all_tokens": sum(int(row.get("total_tokens_all") or 0) for row in debug),
                "debug_cost_usd": attributable_cost(debug),
                "debug_sessions": len(debug),
                "regen_pass_at_1": 1 in clean_trials,
                "regen_pass_at_3": bool(clean_trials),
                "regen_first_clean_trial": first_clean,
                "regen_uncached_tokens_to_clean_or_3": regen_to_clean,
                "regen_uncached_tokens_all_3": sum(int(row.get("total_tokens_uncached") or 0) for row in regen),
                "regen_all_tokens_all_3": sum(int(row.get("total_tokens_all") or 0) for row in regen),
                "regen_cost_usd_all_3": attributable_cost(regen),
                "regen_trials": len(regen),
                "regen_operational_sessions": len(operational),
                "regen_operational_overhead_tokens": sum(
                    int(row.get("total_tokens_all") or 0) for row in operational
                ),
                "regen_operational_overhead_cost_usd": attributable_cost(operational),
            }
        )

    b = sum(pair["debug_clean"] and not pair["regen_pass_at_3"] for pair in pairs)
    c = sum((not pair["debug_clean"]) and pair["regen_pass_at_3"] for pair in pairs)
    both_clean = [pair for pair in pairs if pair["debug_clean"] and pair["regen_pass_at_3"]]
    differences = [
        float(pair["regen_uncached_tokens_to_clean_or_3"] - pair["debug_uncached_tokens"])
        for pair in both_clean
    ]
    levels: dict[str, dict[str, Any]] = {}
    for level in sorted({pair["level"] for pair in pairs}):
        subset = [pair for pair in pairs if pair["level"] == level]
        levels[level] = {
            "n": len(subset),
            "debug_external_clean": sum(pair["debug_clean"] for pair in subset),
            "regen_pass_at_1": sum(pair["regen_pass_at_1"] for pair in subset),
            "regen_pass_at_3": sum(pair["regen_pass_at_3"] for pair in subset),
        }

    failed = [row for row in rows if row.get("is_failed_trial")]
    fallback = [row for row in rows if row.get("token_source") == "journal_fallback"]
    mixed_by_model: dict[str, int] = {}
    for row in mixed_regen_rows:
        model = str(row.get("model") or "unknown")
        mixed_by_model[model] = mixed_by_model.get(model, 0) + 1
    mixed_clean_trials = [row for row in mixed_regen_rows if row.get("external_clean")]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "operators": len(pairs),
        "same_model": "Kimi K3 only",
        "external_clean": {
            "debug": sum(pair["debug_clean"] for pair in pairs),
            "regenerate_pass_at_1": sum(pair["regen_pass_at_1"] for pair in pairs),
            "regenerate_pass_at_3": sum(pair["regen_pass_at_3"] for pair in pairs),
        },
        "mcnemar_pass_at_3": {
            "debug_only": b,
            "regenerate_only": c,
            "two_sided_exact_p": exact_mcnemar(b, c),
        },
        "tokens_to_clean_both_successful": {
            "metric": "uncached input + output",
            "pairs": len(both_clean),
            "wilcoxon_signed_rank": wilcoxon_exact(differences),
        },
        "level_strata": levels,
        "business_failed_rows": len(failed),
        "business_failed_operational_cost_usd": attributable_cost(failed),
        "journal_fallback_rows": len(fallback),
        "mixed_model_regenerate_descriptive": {
            "causal_comparison_eligible": False,
            "reason": "Regenerate switched from Kimi K3 to qwen3.8-max-preview mid-experiment",
            "rows": len(mixed_regen_rows),
            "models": mixed_by_model,
            "external_clean_rows": len(mixed_clean_trials),
            "token_volume_all": sum(int(row.get("total_tokens_all") or 0) for row in mixed_regen_rows),
            "monetary_cost_usd": attributable_cost(mixed_regen_rows),
            "monetary_cost_note": (
                "unattributed because Token Plan exposes dynamic Credits only in console usage details"
                if attributable_cost(mixed_regen_rows) is None
                else "all rows priced"
            ),
        },
        "pairs": pairs,
    }
    (HERE / "statistical_analysis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    with (HERE / "statistical_analysis_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]) if pairs else ["operator"])
        writer.writeheader()
        writer.writerows(pairs)
    lines = [
        "# Debug vs Regenerate statistical analysis",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Paired operators: {len(pairs)}; model eligibility: Kimi K3 on both arms only.",
        f"- External-Clean: Debug {payload['external_clean']['debug']}/{len(pairs)}, "
        f"Regenerate pass@1 {payload['external_clean']['regenerate_pass_at_1']}/{len(pairs)}, "
        f"pass@3 {payload['external_clean']['regenerate_pass_at_3']}/{len(pairs)}.",
        f"- Exact McNemar (Debug vs Regenerate pass@3): b={b}, c={c}, p={exact_mcnemar(b, c):.8g}.",
        f"- Tokens-to-clean Wilcoxon: n={len(both_clean)}, "
        f"p={payload['tokens_to_clean_both_successful']['wilcoxon_signed_rank']['two_sided_p']:.8g}.",
        f"- Mixed-model Regenerate (descriptive only): {len(mixed_regen_rows)} rows, "
        f"{len(mixed_clean_trials)} External-Clean; models={mixed_by_model}.",
        "- Qwen3.8 Token Plan monetary cost is not assigned per run because the official plan exposes "
        "dynamic Credits only in console usage details; token volumes remain exact.",
        "",
        "## Level strata",
        "",
        "| Level | N | Debug clean | Regen pass@1 | Regen pass@3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for level, item in levels.items():
        lines.append(
            f"| {level} | {item['n']} | {item['debug_external_clean']} | {item['regen_pass_at_1']} | {item['regen_pass_at_3']} |"
        )
    (HERE / "statistical_analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: payload[key] for key in ("operators", "external_clean", "mcnemar_pass_at_3")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
