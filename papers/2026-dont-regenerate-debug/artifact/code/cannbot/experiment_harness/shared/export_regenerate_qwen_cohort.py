#!/usr/bin/env python3
"""Export the final Qwen-only 27x3 Regenerate cohort as auditable CSV/JSON."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(root: Path) -> list[dict[str, Any]]:
    payload = load_json(root / "experiment_harness" / "token_cost_summary.json")
    return list(payload["rows"])


def key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["operator"]), int(row["trial"])


def is_qwen(row: dict[str, Any]) -> bool:
    return "qwen3.8-max-preview" in str(row.get("model") or "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    combined: dict[tuple[str, int], dict[str, Any]] = {}
    for row in load_rows(args.main_root):
        if is_qwen(row):
            combined[key(row)] = {**row, "cohort_source": "main"}
    for row in load_rows(args.supplement_root):
        if is_qwen(row):
            combined[key(row)] = {**row, "cohort_source": "qwen_k3_control"}

    rows: list[dict[str, Any]] = []
    for _, row in sorted(combined.items()):
        cwd = Path(str(row["cwd"]))
        runtime = load_json(cwd / "run_record.json")
        full_eval = load_json(Path(str(row["full_eval_path"])))
        anticheat = load_json(Path(str(row["anticheat_path"])))
        rows.append(
            {
                "operator": row["operator"],
                "trial": int(row["trial"]),
                "run_id": row["run_id"],
                "cohort_source": row["cohort_source"],
                "model": row["model"],
                "provider": row.get("provider"),
                "status": runtime.get("status"),
                "result_subtype": runtime.get("result_subtype"),
                "candidate_materialized": runtime.get("candidate_materialized"),
                "contamination_detected": (
                    runtime.get("contamination_detected")
                    if "contamination_detected" in runtime
                    else runtime.get("regenerate_purity") == "contaminated"
                ),
                "forbidden_debug_skill_calls": runtime.get(
                    "forbidden_debug_skill_calls", []
                ),
                "source_type": runtime.get("source_type", row.get("source_type")),
                "regenerate_purity": runtime.get(
                    "regenerate_purity", row.get("regenerate_purity")
                ),
                "isolation_preflight_passed": runtime.get(
                    "isolation_preflight_passed"
                ),
                "full_eval_result": full_eval.get("full_eval_result"),
                "final_failure_type": full_eval.get("final_failure_type"),
                "matched_count": full_eval.get("matched_count"),
                "total_outputs": full_eval.get("total_outputs"),
                "match_rate_pct": full_eval.get("match_rate_pct"),
                "anticheat_verdict": anticheat.get("verdict"),
                "official_anticheat_verdict": anticheat.get(
                    "official_verdict", anticheat.get("verdict")
                ),
                "compatibility_override_applied": anticheat.get(
                    "compatibility_override_applied", False
                ),
                "anticheat_reasons": anticheat.get("reasons", []),
                "input_tokens": int(row.get("input_tokens") or 0),
                "output_tokens": int(row.get("output_tokens") or 0),
                "cache_creation_tokens": int(row.get("cache_creation_tokens") or 0),
                "cache_read_tokens": int(row.get("cache_read_tokens") or 0),
                "total_tokens_uncached": int(row.get("total_tokens_uncached") or 0),
                "total_tokens_all": int(row.get("total_tokens_all") or 0),
                "turns": int(row.get("num_turns") or 0),
                "cli_reported_cost_usd": float(
                    row.get("cli_reported_cost_usd") or 0.0
                ),
                "cost_attributable": bool(row.get("cost_attributable")),
                "cwd": str(cwd),
            }
        )

    if len(rows) != 81 or len({key(row) for row in rows}) != 81:
        raise SystemExit("Qwen cohort is not exactly 81 unique operator/trial rows")

    totals = {
        name: sum(row[name] for row in rows)
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_creation_tokens",
            "cache_read_tokens",
            "total_tokens_uncached",
            "total_tokens_all",
            "turns",
            "cli_reported_cost_usd",
        )
    }
    summary = {
        "rows": len(rows),
        "unique_operators": len({row["operator"] for row in rows}),
        "unique_trials": sorted({row["trial"] for row in rows}),
        "status": dict(Counter(str(row["status"]) for row in rows)),
        "candidate_materialized_true": sum(
            row["candidate_materialized"] is True for row in rows
        ),
        "contamination_detected_true": sum(
            row["contamination_detected"] is True for row in rows
        ),
        "forbidden_debug_skill_call_count": sum(
            len(row["forbidden_debug_skill_calls"]) for row in rows
        ),
        "isolation_preflight_passed_true": sum(
            row["isolation_preflight_passed"] is True for row in rows
        ),
        "full_eval": dict(Counter(str(row["full_eval_result"]) for row in rows)),
        "full_eval_failure_type": dict(
            Counter(str(row["final_failure_type"]) for row in rows)
        ),
        "anticheat": dict(Counter(str(row["anticheat_verdict"]) for row in rows)),
        "official_anticheat": dict(
            Counter(str(row["official_anticheat_verdict"]) for row in rows)
        ),
        "compatibility_override_applied": sum(
            bool(row["compatibility_override_applied"]) for row in rows
        ),
        "automatic_strict_success_runs": sum(
            row["full_eval_result"] == "pass"
            and row["anticheat_verdict"] == "CLEAN"
            for row in rows
        ),
        "totals": totals,
        "attributable_cost_rows": sum(row["cost_attributable"] for row in rows),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    fields = list(rows[0])
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["anticheat_reasons"] = json.dumps(
                out["anticheat_reasons"], ensure_ascii=False
            )
            out["forbidden_debug_skill_calls"] = json.dumps(
                out["forbidden_debug_skill_calls"], ensure_ascii=False
            )
            writer.writerow(out)

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
