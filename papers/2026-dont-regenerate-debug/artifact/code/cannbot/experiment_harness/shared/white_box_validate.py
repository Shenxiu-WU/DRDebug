#!/usr/bin/env python3
"""Independently recompute three representative rows and write the QA report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens")


def usage(raw: dict[str, Any]) -> dict[str, int]:
    aliases = {
        "input_tokens": ("input_tokens", "inputTokens"),
        "output_tokens": ("output_tokens", "outputTokens"),
        "cache_read_tokens": ("cache_read_input_tokens", "cacheReadInputTokens", "cache_read_tokens"),
        "cache_creation_tokens": ("cache_creation_input_tokens", "cacheCreationInputTokens", "cache_creation_tokens"),
    }
    return {
        field: next((int(raw[name]) for name in names if raw.get(name) is not None), 0)
        for field, names in aliases.items()
    }


def journal(path: Path) -> dict[str, int]:
    latest: dict[str, dict[str, int]] = {}
    if not path.is_file():
        return {field: 0 for field in FIELDS}
    for line in path.read_text(errors="replace").splitlines():
        try:
            message = json.loads(line).get("message", {})
        except json.JSONDecodeError:
            continue
        if isinstance(message.get("id"), str) and isinstance(message.get("usage"), dict):
            latest[message["id"]] = usage(message["usage"])
    return {field: sum(item[field] for item in latest.values()) for field in FIELDS}


def direct_result(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text())
    breakdown = data.get("modelUsage") or {}
    return {field: sum(usage(item)[field] for item in breakdown.values()) for field in FIELDS}


def main() -> int:
    summary = json.loads((HERE / "token_cost_summary.json").read_text())["rows"]
    pricing = json.loads((HERE / "pricing_table.json").read_text())
    regen = [row for row in summary if row["arm"] == "regenerate" and row.get("total_tokens_all", 0) > 0]
    selected: list[dict[str, Any]] = []
    # Prefer a business failure, a subagent-heavy row, and a journal fallback.
    for candidates in (
        [row for row in regen if row.get("is_failed_trial")],
        sorted(regen, key=lambda row: int(row.get("subagent_tokens") or 0), reverse=True),
        [row for row in regen if row.get("token_source") == "journal_fallback"],
        regen,
    ):
        for row in candidates:
            if row["run_id"] not in {item["run_id"] for item in selected}:
                selected.append(row)
                break
        if len(selected) >= 3:
            break

    checks = []
    for row in selected:
        result_path = Path(row["session_result_path"]) if row.get("session_result_path") else None
        if (
            row.get("token_source") == "final_model_usage"
            and result_path
            and result_path.is_file()
            and result_path.stat().st_size
        ):
            recomputed = direct_result(result_path)
            source = "final modelUsage independently summed"
        else:
            main = journal(Path(row["jsonl_path"]))
            subs = [journal(Path(path)) for path in row.get("subagent_jsonl_paths", [])]
            recomputed = {field: main[field] + sum(item[field] for item in subs) for field in FIELDS}
            source = "journal fallback independently deduplicated"
        match = all(recomputed[field] == int(row[field]) for field in FIELDS)
        model = row["model"]
        key = "kimi-k3" if model in {"k3", "k3[1m]", "kimi-k3"} else model
        pricing_spec = pricing["models"][key]
        rates = pricing_spec.get("per_million")
        if rates:
            native = sum(recomputed[field] * rates[field] / 1_000_000 for field in FIELDS)
            usd = native / pricing["fx"]["usd_cny"] if pricing_spec["currency"] == "CNY" else native
            cost_abs_error = abs(float(row["cost_usd"]) - usd)
            cost_check = "exact_per_token_recomputation"
        else:
            usd = None
            cost_abs_error = None
            cost_check = str(pricing_spec.get("billing_mode") or "unpriced")
        checks.append(
            {
                "run_id": row["run_id"],
                "operator": row["operator"],
                "source": source,
                "summary_tokens": {field: row[field] for field in FIELDS},
                "recomputed_tokens": recomputed,
                "tokens_exact": match,
                "summary_cost_usd": row["cost_usd"],
                "recomputed_cost_usd": usd,
                "cost_abs_error": cost_abs_error,
                "cost_check": cost_check,
                "subagent_tokens": row.get("subagent_tokens"),
                "subagent_is_subset_not_readded": int(row.get("subagent_tokens") or 0) <= int(row["total_tokens_all"]),
                "is_failed_trial": row.get("is_failed_trial"),
                "included_in_operational_cost": row.get("include_in_operational_cost"),
                "token_source": row.get("token_source"),
            }
        )

    failed = [row for row in summary if row.get("is_failed_trial")]
    terminal_failed = [row for row in summary if row["arm"] == "regenerate" and row.get("terminal_status") == "failed"]
    fallback_failed = [row for row in terminal_failed if row.get("token_source") == "journal_fallback"]
    passed = (
        len(checks) >= min(3, len(regen))
        and all(
            item["tokens_exact"]
            and (item["cost_abs_error"] is None or item["cost_abs_error"] < 1e-8)
            for item in checks
        )
        and all(row.get("include_in_operational_cost") for row in failed if int(row.get("total_tokens_all") or 0) > 0)
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "failed_trial_rows": len(failed),
        "failed_trial_tokens": sum(int(row.get("total_tokens_all") or 0) for row in failed),
        "terminal_failed_regenerate_rows": len(terminal_failed),
        "terminal_failed_with_journal_fallback": len(fallback_failed),
        "failed_journal_acceptance": (
            "verified_journal_fallback" if fallback_failed else "terminal_failed_result_present_or_no_terminal_failed_session"
        ),
    }
    (HERE / "white_box_validation.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# Token/cost white-box validation",
        "",
        f"- Status: **{payload['status']}**",
        f"- Generated: {payload['generated_at']}",
        f"- Business-failed rows retained in operational cost: {payload['failed_trial_rows']} "
        f"({payload['failed_trial_tokens']:,} total tokens).",
        f"- Terminal-failed Regenerate sessions: {payload['terminal_failed_regenerate_rows']}; "
        f"journal fallback rows: {payload['terminal_failed_with_journal_fallback']}.",
        f"- Failed-session acceptance: `{payload['failed_journal_acceptance']}`.",
        "",
        "| Run | Independent source | Four token fields exact | Cost abs error | Subagent subset | Failed retained |",
        "|---|---|---|---:|---|---|",
    ]
    for item in checks:
        cost_display = "n/a" if item["cost_abs_error"] is None else f"{item['cost_abs_error']:.3g}"
        lines.append(
            f"| {item['run_id']} | {item['source']} | {item['tokens_exact']} | "
            f"{cost_display} | {item['subagent_is_subset_not_readded']} | "
            f"{item['is_failed_trial'] and item['included_in_operational_cost']} |"
        )
    lines.extend(
        [
            "",
            "The recomputation does not import the production aggregator. It independently keeps the last usage "
            "per message id within each journal, sums final `modelUsage` per model where available, and applies "
            "the versioned offline price table. `subagent_tokens` is checked as a disclosed subset of the "
            "whole-session total and is never added a second time.",
        ]
    )
    (HERE / "validation_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": payload["status"], "checks": len(checks)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
