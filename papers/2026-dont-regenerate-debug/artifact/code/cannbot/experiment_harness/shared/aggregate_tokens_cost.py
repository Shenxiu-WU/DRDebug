#!/usr/bin/env python3
"""Aggregate Regenerate and Debug Claude sessions with one reproducible cost model.

The important accounting rule is that Claude Code journals are streaming state:
the same message id may occur several times with evolving usage.  Journal usage
is therefore deduplicated by (absolute journal path, message.id), keeping the
last record.  A final result's modelUsage remains the authoritative whole-session
total when present; journals provide the main/subagent attribution and fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
)
FORBIDDEN_DEBUG_SKILLS = {
    "ascendc-precision-debug",
    "ascendc-crash-debug",
    "ascendc-runtime-debug",
}
CANNBOT_DEBUG_FORBIDDEN_GENERATION_SKILLS = {
    "ops-direct-invoke",
    "ascendc-direct-invoke-template",
    "ascendc-regbase-best-practice",
    "ascendc-blaze-best-practice",
    "torch-ascendc-op-extension",
}

CORE_FIELDS = [
    "operator",
    "arm",
    "trial",
    "cwd",
    "session_id",
    "jsonl_path",
    *TOKEN_FIELDS,
    "total_tokens_uncached",
    "total_tokens_all",
    "num_turns",
    "subagent_tokens",
    "cost_usd",
    "full_eval_result",
    "anticheat_verdict",
    "external_clean",
    "is_failed_trial",
    "error_type",
]

EXTRA_FIELDS = [
    "run_id",
    "provider",
    "model",
    "model_raw_aliases",
    "model_breakdown",
    "session_result_path",
    "subagent_jsonl_paths",
    "full_eval_path",
    "anticheat_path",
    "token_source",
    "main_journal_input_tokens",
    "main_journal_output_tokens",
    "main_journal_cache_read_tokens",
    "main_journal_cache_creation_tokens",
    "subagent_input_tokens",
    "subagent_output_tokens",
    "subagent_cache_read_tokens",
    "subagent_cache_creation_tokens",
    "unattributed_input_tokens",
    "unattributed_output_tokens",
    "unattributed_cache_read_tokens",
    "unattributed_cache_creation_tokens",
    "unattributed_session_tokens",
    "token_reconciliation_status",
    "token_reconciliation_ratio",
    "num_turns_source",
    "canonical_assistant_messages",
    "pricing_id",
    "pricing_source_urls",
    "pricing_captured_at",
    "pricing_status",
    "cost_attributable",
    "cost_native",
    "cost_native_currency",
    "fx_usd_cny",
    "cli_reported_cost_usd",
    "terminal_status",
    "transport_status",
    "error_scope",
    "include_in_operational_cost",
    "include_in_success_denominator",
    "evidence_status",
    "repair_policy",
    "had_internal_failure",
    "internal_repair_count",
    "regenerate_purity",
    "forbidden_debug_skill_calls",
    "required_cannbot_debug_skill_calls",
    "cannbot_debug_skill_observed",
    "same_model_eligible",
    "source_type",
    "is_operational_retry",
    "operational_retry_of",
    "context_window_tokens",
    "provider_retry_count",
    "provider_checkpoints",
    "resume_count",
    "resume_policy",
]

OUTPUT_FIELDS = CORE_FIELDS + EXTRA_FIELDS


def zero_tokens() -> dict[str, int]:
    return {field: 0 for field in TOKEN_FIELDS}


def add_tokens(*parts: dict[str, int]) -> dict[str, int]:
    out = zero_tokens()
    for part in parts:
        for field in TOKEN_FIELDS:
            out[field] += int(part.get(field, 0) or 0)
    return out


def normalize_usage(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return zero_tokens()
    aliases = {
        "input_tokens": ("input_tokens", "inputTokens"),
        "output_tokens": ("output_tokens", "outputTokens"),
        "cache_read_tokens": ("cache_read_input_tokens", "cacheReadInputTokens", "cache_read_tokens"),
        "cache_creation_tokens": (
            "cache_creation_input_tokens",
            "cacheCreationInputTokens",
            "cache_creation_tokens",
        ),
    }
    out = zero_tokens()
    for field, candidates in aliases.items():
        for candidate in candidates:
            if candidate in raw and raw[candidate] is not None:
                try:
                    out[field] = int(raw[candidate])
                except (TypeError, ValueError):
                    out[field] = 0
                break
    return out


def token_totals(tokens: dict[str, int]) -> tuple[int, int]:
    uncached = int(tokens["input_tokens"]) + int(tokens["output_tokens"])
    return uncached, uncached + int(tokens["cache_read_tokens"]) + int(tokens["cache_creation_tokens"])


def classify_token_reconciliation(diffs: dict[str, int], total_tokens: int) -> tuple[str, float]:
    """Classify final-result versus journal usage without hiding field drift.

    Some provider/CLI versions move a small number of tokens between the input,
    output, and cache buckets even though their all-token total reconciles.
    Final ``modelUsage`` remains authoritative; a non-negative aggregate delta
    of at most one percent is therefore accepted and the per-field signed
    deltas stay visible for audit.
    """
    aggregate_delta = sum(diffs.values())
    ratio = aggregate_delta / max(1, total_tokens)
    if all(value == 0 for value in diffs.values()):
        return "exact", ratio
    if all(value >= 0 for value in diffs.values()) and 0 <= ratio <= 0.01:
        return "reconciled_with_overhead", ratio
    if aggregate_delta >= 0 and ratio <= 0.01:
        return "reconciled_with_category_drift", ratio
    return "mismatch", ratio


def load_json(path: Path | None) -> Any:
    if path is None or not path.is_file() or path.stat().st_size == 0:
        return None
    text = path.read_text(errors="replace").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Some launchers prepend diagnostic lines.  Prefer the last complete JSON line.
        for line in reversed(text.splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def model_usage_from_result(result: Any, fallback_model: str) -> tuple[dict[str, dict[str, int]], bool]:
    if not isinstance(result, dict):
        return {}, False
    raw_breakdown = result.get("modelUsage") or result.get("model_usage")
    breakdown: dict[str, dict[str, int]] = {}
    if isinstance(raw_breakdown, dict):
        for model, usage in raw_breakdown.items():
            if isinstance(usage, dict):
                breakdown[str(model)] = normalize_usage(usage)
    if breakdown:
        return breakdown, True
    raw_usage = result.get("usage")
    if isinstance(raw_usage, dict):
        return {fallback_model: normalize_usage(raw_usage)}, False
    return {}, False


def journal_usage(path: Path) -> dict[str, Any]:
    latest: dict[str, dict[str, int]] = {}
    aliases: set[str] = set()
    session_ids: set[str] = set()
    skill_calls: list[str] = []
    invalid_lines = 0
    if not path.is_file():
        return {
            "tokens": zero_tokens(),
            "messages": 0,
            "aliases": aliases,
            "session_ids": session_ids,
            "skill_calls": skill_calls,
            "invalid_lines": invalid_lines,
        }
    for line in path.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            invalid_lines += 1
            continue
        message = record.get("message")
        raw_session = record.get("sessionId") or record.get("session_id")
        if isinstance(raw_session, str) and raw_session:
            session_ids.add(raw_session)
        if not isinstance(message, dict):
            continue
        model = message.get("model")
        if isinstance(model, str) and model:
            aliases.add(model)
        message_id = message.get("id")
        usage = message.get("usage")
        if isinstance(message_id, str) and isinstance(usage, dict):
            latest[message_id] = normalize_usage(usage)
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use" or block.get("name") != "Skill":
                    continue
                tool_input = block.get("input")
                if isinstance(tool_input, dict) and isinstance(tool_input.get("skill"), str):
                    skill_calls.append(tool_input["skill"])
    return {
        "tokens": add_tokens(*latest.values()),
        "messages": len(latest),
        "aliases": aliases,
        "session_ids": session_ids,
        "skill_calls": skill_calls,
        "invalid_lines": invalid_lines,
    }


class Pricing:
    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.pricing_id = str(data["pricing_id"])
        self.captured_at = str(data["captured_at"])
        self.fx_usd_cny = float(data["fx"]["usd_cny"])
        self.models = data["models"]
        self.alias_to_key: dict[str, str] = {}
        for key, spec in self.models.items():
            self.alias_to_key[key] = key
            for alias in spec.get("aliases", []):
                self.alias_to_key[str(alias)] = key

    def canonical(self, model: str) -> str | None:
        return self.alias_to_key.get(model)

    def cost(
        self,
        breakdown: dict[str, dict[str, int]],
    ) -> tuple[float | None, float | None, str, list[str], dict[str, Any]]:
        total_usd = 0.0
        native_parts: list[tuple[str, float]] = []
        sources: set[str] = set()
        enriched: dict[str, Any] = {}
        has_unattributed_subscription = False
        for raw_model, tokens in breakdown.items():
            key = self.canonical(raw_model)
            if key is None:
                raise ValueError(f"no pricing entry for model {raw_model!r}")
            spec = self.models[key]
            sources.add(str(spec["source"]))
            if "per_million" not in spec:
                has_unattributed_subscription = True
                enriched[raw_model] = {
                    "pricing_key": key,
                    "tokens": tokens,
                    "billing_mode": spec.get("billing_mode", "unpriced"),
                    "cost_native": None,
                    "currency": spec.get("currency"),
                    "cost_usd": None,
                }
                continue
            rates = spec["per_million"]
            native = sum(float(tokens[field]) * float(rates[field]) / 1_000_000 for field in TOKEN_FIELDS)
            currency = str(spec["currency"])
            usd = native if currency == "USD" else native / self.fx_usd_cny
            total_usd += usd
            native_parts.append((currency, native))
            enriched[raw_model] = {
                "pricing_key": key,
                "tokens": tokens,
                "billing_mode": spec.get("billing_mode", "per_token"),
                "cost_native": native,
                "currency": currency,
                "cost_usd": usd,
            }
        if has_unattributed_subscription:
            native_currency = (
                "UNATTRIBUTED_SUBSCRIPTION"
                if not native_parts
                else "MIXED_WITH_UNATTRIBUTED_SUBSCRIPTION"
            )
            return None, None, native_currency, sorted(sources), enriched
        currencies = {currency for currency, _ in native_parts}
        if len(currencies) == 1:
            native_currency = next(iter(currencies))
            native_total: float | None = sum(value for _, value in native_parts)
        elif not native_parts:
            native_currency = "USD"
            native_total = 0.0
        else:
            native_currency = "MIXED"
            native_total = None
        return total_usd, native_total, native_currency, sorted(sources), enriched


def parse_full_eval(record: dict[str, Any]) -> tuple[str, str | None]:
    explicit = record.get("full_eval_result")
    path_text = record.get("full_eval_path")
    path = Path(path_text) if isinstance(path_text, str) and path_text else None
    if explicit in {"pass", "fail", "error", "not_run"}:
        return str(explicit), str(path) if path else None
    data = load_json(path)
    if not isinstance(data, dict):
        return "not_run" if path is None else "error", str(path) if path else None
    value = data.get("full_eval_result") or data.get("final_failure_type") or data.get("status") or data.get("result")
    if value in {"pass", "PASS", "success", "passed"}:
        return "pass", str(path)
    if value in {"fail", "FAIL", "failed", "precision_failed", "runtime_error", "import_failed", "timeout"}:
        return "fail", str(path)
    if data.get("full_eval_pass") is True or data.get("success") is True or data.get("pass") is True:
        return "pass", str(path)
    if data.get("full_eval_pass") is False or data.get("success") is False or data.get("pass") is False:
        return "fail", str(path)
    match_rate = data.get("match_rate_pct", data.get("last_full_match_rate"))
    if match_rate is not None:
        try:
            return ("pass" if float(match_rate) == 100.0 else "fail"), str(path)
        except (TypeError, ValueError):
            pass
    return "error", str(path)


def parse_anticheat(record: dict[str, Any]) -> tuple[str, str | None]:
    explicit = record.get("anticheat_verdict")
    path_text = record.get("anticheat_path")
    path = Path(path_text) if isinstance(path_text, str) and path_text else None
    if explicit in {"CLEAN", "CHEAT", "error", "not_run"}:
        return str(explicit), str(path) if path else None
    data = load_json(path)
    if not isinstance(data, dict):
        return "not_run" if path is None else "error", str(path) if path else None
    verdict = data.get("verdict") or data.get("anticheat_verdict")
    if isinstance(verdict, str) and verdict.upper() in {"CLEAN", "CHEAT", "NOT_RUN"}:
        normalized = "not_run" if verdict.upper() == "NOT_RUN" else verdict.upper()
        return normalized, str(path)
    return "error", str(path)


def infer_error(result: Any, full_eval: str, anticheat: str, override: str | None = None) -> tuple[str, str]:
    if override and override != "none":
        scopes = {
            "business_fail": "business",
            "bench_defect": "benchmark",
            "transport_disconnect": "transport",
            "evidence_conflict": "harness",
        }
        return override, scopes.get(override, "provider" if override.startswith("provider_") else "harness")
    text = json.dumps(result, ensure_ascii=False).lower() if result is not None else ""
    for code in (400, 401, 402, 429):
        if re.search(rf"\b{code}\b", text):
            return f"provider_{code}", "provider"
    if "usage policy" in text or (isinstance(result, dict) and str(result.get("stop_reason", "")).lower() == "refusal"):
        return "provider_refusal", "provider"
    if any(needle in text for needle in ("provider error", "overloaded", "internal server error", "api error")):
        return "provider_5xx", "provider"
    if "timeout" in text or "timed out" in text:
        return "timeout", "provider"
    if isinstance(result, dict) and result.get("is_error") is True:
        return "unknown", "harness"
    if full_eval == "fail" or anticheat == "CHEAT":
        return "business_fail", "business"
    if full_eval == "error" or anticheat == "error":
        return "evidence_conflict", "harness"
    return "none", "none"


def result_terminal_status(result: Any) -> str:
    if not isinstance(result, dict):
        return "unknown"
    if result.get("is_error") is True:
        return "failed"
    subtype = str(result.get("subtype", "")).lower()
    if subtype == "success" or result.get("is_error") is False:
        return "success"
    if subtype or result.get("error"):
        return "failed"
    return "unknown"


def dominant_model(breakdown: dict[str, dict[str, int]], fallback: str) -> str:
    if not breakdown:
        return fallback
    return max(breakdown, key=lambda model: token_totals(breakdown[model])[1])


def models_match_expected(
    breakdown: dict[str, dict[str, int]],
    expected_model: str,
    pricing: Pricing,
) -> bool:
    """Return whether every billed model resolves to the manifest model.

    Earlier accounting code hard-coded Kimi K3 because the first paired report
    only covered that model.  C2 also uses Qwen3.8-Max-Preview, so eligibility
    must follow ``model_expected`` rather than one provider-specific model key.
    """
    if not breakdown:
        return False
    expected = pricing.canonical(expected_model)
    actual = {pricing.canonical(model) for model in breakdown}
    return expected is not None and None not in actual and actual == {expected}


def base_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {field: None for field in OUTPUT_FIELDS}
    row.update(
        {
            "operator": record.get("operator", ""),
            "arm": record.get("arm", ""),
            "trial": record.get("trial"),
            "cwd": record.get("cwd") or record.get("task_dir") or "",
            "run_id": record.get("run_id", ""),
            "provider": record.get("provider", "kimi-code"),
            "transport_status": record.get("transport_status", "unknown"),
            "repair_policy": record.get("repair_policy", "repair_allowed" if record.get("arm") == "regenerate" else "not_applicable"),
            "had_internal_failure": bool(record.get("had_internal_failure", False)),
            "internal_repair_count": record.get("internal_repair_count"),
            "regenerate_purity": record.get("regenerate_purity", "unknown" if record.get("arm") == "regenerate" else "not_applicable"),
            "source_type": record.get("source_type", "fresh"),
            "same_model_eligible": record.get("same_model_eligible", True),
            "is_operational_retry": bool(record.get("is_operational_retry", False)),
            "operational_retry_of": record.get("operational_retry_of"),
            "context_window_tokens": record.get("context_window_tokens"),
        }
    )
    return row


def finalize_common(
    row: dict[str, Any],
    record: dict[str, Any],
    result: Any,
    breakdown: dict[str, dict[str, int]],
    pricing: Pricing,
    full_eval: str,
    anticheat: str,
) -> dict[str, Any]:
    totals = add_tokens(*breakdown.values())
    for field in TOKEN_FIELDS:
        row[field] = totals[field]
    row["total_tokens_uncached"], row["total_tokens_all"] = token_totals(totals)
    fallback_model = str(record.get("model_expected") or "k3")
    row["model"] = dominant_model(breakdown, fallback_model)
    row["same_model_eligible"] = bool(
        row.get("same_model_eligible", True)
    ) and models_match_expected(breakdown, fallback_model, pricing)
    row["full_eval_result"] = full_eval
    row["anticheat_verdict"] = anticheat
    row["external_clean"] = full_eval == "pass" and anticheat == "CLEAN"
    error_type, error_scope = infer_error(result, full_eval, anticheat, record.get("error_type"))
    row["error_type"] = error_type
    row["error_scope"] = record.get("error_scope", error_scope)
    row["is_failed_trial"] = row["error_scope"] == "business" and not row["external_clean"]
    row["include_in_operational_cost"] = row["total_tokens_all"] > 0
    row["include_in_success_denominator"] = row["error_scope"] in {"none", "business"}
    row["terminal_status"] = result_terminal_status(result)
    row["pricing_id"] = pricing.pricing_id
    row["pricing_captured_at"] = pricing.captured_at
    row["fx_usd_cny"] = pricing.fx_usd_cny
    if breakdown:
        cost_usd, cost_native, native_currency, sources, enriched = pricing.cost(breakdown)
        if cost_usd is None:
            row["cost_usd"] = None
            row["pricing_status"] = "subscription_credits_unattributed"
            row["cost_attributable"] = False
        else:
            row["cost_usd"] = round(cost_usd, 9)
            row["pricing_status"] = "priced_per_token"
            row["cost_attributable"] = True
        row["cost_native"] = None if cost_native is None else round(cost_native, 9)
        row["cost_native_currency"] = native_currency
        row["pricing_source_urls"] = sources
        row["model_breakdown"] = enriched
    else:
        row["cost_usd"] = 0.0
        row["pricing_status"] = "no_tokens"
        row["cost_attributable"] = True
        row["cost_native"] = 0.0
        row["cost_native_currency"] = "USD"
        row["pricing_source_urls"] = []
        row["model_breakdown"] = {}
    if isinstance(result, dict):
        raw_cli_cost = result.get("total_cost_usd")
        if raw_cli_cost is None:
            raw_cli_cost = sum(
                float(item.get("costUSD", 0) or 0)
                for item in (result.get("modelUsage") or {}).values()
                if isinstance(item, dict)
            )
        row["cli_reported_cost_usd"] = raw_cli_cost
    return row


def regen_row(record: dict[str, Any], pricing: Pricing) -> dict[str, Any]:
    row = base_row(record)
    run_record_path = Path(record["run_record_path"]) if record.get("run_record_path") else None
    run_record = load_json(run_record_path)
    if isinstance(run_record, dict):
        for field in (
            "had_internal_failure",
            "internal_repair_count",
            "regenerate_purity",
            "transport_status",
            "provider_retry_count",
            "provider_checkpoints",
            "resume_count",
            "resume_policy",
        ):
            if field in run_record:
                row[field] = run_record[field]
    result_path = Path(record["final_result_path"]) if record.get("final_result_path") else None
    result = load_json(result_path)
    fallback_model = str(record.get("model_expected") or "k3")
    breakdown, authoritative = model_usage_from_result(result, fallback_model)

    main_path = Path(record["journal_main_path"]) if record.get("journal_main_path") else None
    main = journal_usage(main_path) if main_path else journal_usage(Path("/__missing__"))
    sub_paths: list[Path] = []
    if isinstance(record.get("subagent_jsonl_paths"), list):
        sub_paths.extend(Path(item) for item in record["subagent_jsonl_paths"])
    elif record.get("subagent_dir"):
        sub_dir = Path(record["subagent_dir"])
        if sub_dir.is_dir():
            sub_paths.extend(sorted(sub_dir.rglob("*.jsonl")))
    sub_parts = [journal_usage(path) for path in sub_paths]
    sub_tokens = add_tokens(*(part["tokens"] for part in sub_parts))
    journal_total = add_tokens(main["tokens"], sub_tokens)
    resumed_session = bool(
        isinstance(run_record, dict)
        and (
            int(run_record.get("provider_retry_count", 0) or 0) > 0
            or int(run_record.get("resume_count", 0) or 0) > 0
        )
    )
    if resumed_session and token_totals(journal_total)[1] > 0:
        # A resumed CLI result may describe only the latest invocation or the
        # whole conversation depending on CLI/provider version. The append-only
        # main/subagent journals have stable message IDs, so their deduplicated
        # union is the non-overlapping authority for a resumed session.
        breakdown = {fallback_model: journal_total}
        authoritative = False
    if not breakdown and token_totals(journal_total)[1] > 0:
        breakdown = {fallback_model: journal_total}
        authoritative = False

    full_eval, full_eval_path = parse_full_eval(record)
    anticheat, anticheat_path = parse_anticheat(record)
    finalize_common(row, record, result, breakdown, pricing, full_eval, anticheat)
    if isinstance(run_record, dict) and run_record.get("status") == "failed":
        # Generation-terminal failure evidence lives in run_record when a
        # timeout leaves final_stdout.json intentionally empty.
        row["terminal_status"] = "failed"
    row["session_id"] = (result or {}).get("session_id") if isinstance(result, dict) else record.get("expected_session_id")
    row["session_id"] = row["session_id"] or record.get("expected_session_id", "")
    row["jsonl_path"] = str(main_path) if main_path else None
    row["session_result_path"] = str(result_path) if result_path else None
    row["subagent_jsonl_paths"] = [str(path) for path in sub_paths]
    row["full_eval_path"] = full_eval_path
    row["anticheat_path"] = anticheat_path
    row["num_turns"] = (result or {}).get("num_turns") if isinstance(result, dict) else None
    row["num_turns_source"] = f"{result_path}.num_turns" if result_path and row["num_turns"] is not None else None
    row["token_source"] = (
        "journal_resumed_session"
        if resumed_session and breakdown
        else ("final_model_usage" if authoritative else ("journal_fallback" if breakdown else "missing"))
    )
    row["canonical_assistant_messages"] = main["messages"] + sum(part["messages"] for part in sub_parts)
    row["model_raw_aliases"] = sorted(main["aliases"].union(*(part["aliases"] for part in sub_parts))) if sub_parts else sorted(main["aliases"])
    skill_calls = main["skill_calls"] + [call for part in sub_parts for call in part["skill_calls"]]
    if record.get("arm") == "cannbot-debug":
        row["required_cannbot_debug_skill_calls"] = [
            call for call in skill_calls if call == "ascendc-precision-debug"
        ]
        row["cannbot_debug_skill_observed"] = bool(
            row["required_cannbot_debug_skill_calls"]
        )
        row["forbidden_debug_skill_calls"] = [
            call
            for call in skill_calls
            if call in CANNBOT_DEBUG_FORBIDDEN_GENERATION_SKILLS
        ]
    else:
        row["required_cannbot_debug_skill_calls"] = []
        row["cannbot_debug_skill_observed"] = None
        row["forbidden_debug_skill_calls"] = [
            call for call in skill_calls if call in FORBIDDEN_DEBUG_SKILLS
        ]

    expected_session = str(record.get("expected_session_id") or "")
    result_session = str((result or {}).get("session_id") or "") if isinstance(result, dict) else ""
    main_sessions = set(main.get("session_ids") or set())
    session_conflict = bool(
        expected_session
        and ((result_session and result_session != expected_session) or any(value != expected_session for value in main_sessions))
    )

    for field in TOKEN_FIELDS:
        row[f"main_journal_{field}"] = main["tokens"][field]
        row[f"subagent_{field}"] = sub_tokens[field]
    row["subagent_tokens"] = token_totals(sub_tokens)[1]

    if resumed_session and breakdown:
        for field in TOKEN_FIELDS:
            row[f"unattributed_{field}"] = None
        row["unattributed_session_tokens"] = None
        row["token_reconciliation_ratio"] = None
        row["token_reconciliation_status"] = "resume_journal_aggregate"
    elif authoritative:
        total_tokens = {field: row[field] for field in TOKEN_FIELDS}
        diffs = {field: total_tokens[field] - journal_total[field] for field in TOKEN_FIELDS}
        for field in TOKEN_FIELDS:
            row[f"unattributed_{field}"] = diffs[field]
        row["unattributed_session_tokens"] = sum(diffs.values())
        reconciliation_status, reconciliation_ratio = classify_token_reconciliation(
            diffs,
            row["total_tokens_all"],
        )
        row["token_reconciliation_ratio"] = reconciliation_ratio
        journal_present = bool(main_path and main_path.is_file()) or bool(sub_paths)
        if not journal_present:
            row["token_reconciliation_status"] = "final_result_no_journal"
            row["token_reconciliation_ratio"] = None
            row["unattributed_session_tokens"] = None
            row["subagent_tokens"] = None
            for field in TOKEN_FIELDS:
                row[f"unattributed_{field}"] = None
                row[f"subagent_{field}"] = None
        else:
            row["token_reconciliation_status"] = reconciliation_status
    elif breakdown:
        for field in TOKEN_FIELDS:
            row[f"unattributed_{field}"] = None
        row["unattributed_session_tokens"] = None
        row["token_reconciliation_ratio"] = None
        row["token_reconciliation_status"] = "fallback_no_final_result"
    else:
        row["token_reconciliation_status"] = "missing"
    invalid = main["invalid_lines"] + sum(part["invalid_lines"] for part in sub_parts)
    if result is None and token_totals(journal_total)[1] == 0:
        row["evidence_status"] = "missing"
    elif row["token_reconciliation_status"] == "mismatch" or invalid or session_conflict:
        row["evidence_status"] = "conflict"
    elif row["token_reconciliation_status"] == "final_result_no_journal":
        row["evidence_status"] = "partial"
    elif result is None:
        row["evidence_status"] = "partial"
    else:
        row["evidence_status"] = "complete"
    if record.get("transport_status") == "disconnected_after_result" and result is not None:
        row["terminal_status"] = result_terminal_status(result)
    if session_conflict:
        row["error_type"] = "evidence_conflict"
        row["error_scope"] = "harness"
        row["include_in_success_denominator"] = False
    return row


def debug_rows(record: dict[str, Any], pricing: Pricing) -> list[dict[str, Any]]:
    task_dir = Path(record["task_dir"])
    pattern = record.get("claude_results_glob", "precision_tuning/claude_results/*.json")
    files = sorted(task_dir.glob(pattern))
    if not files:
        files = [None]
    full_eval, full_eval_path = parse_full_eval(record)
    anticheat, anticheat_path = parse_anticheat(record)
    out: list[dict[str, Any]] = []
    for index, path in enumerate(files):
        row = base_row(record)
        row["run_id"] = f"{record.get('run_id', '')}:session{index + 1}"
        result = load_json(path)
        fallback_model = str(record.get("model_expected") or "k3")
        breakdown, authoritative = model_usage_from_result(result, fallback_model)
        finalize_common(row, record, result, breakdown, pricing, full_eval, anticheat)
        filename = path.name if path else ""
        sid_match = re.search(r"_([0-9a-f]{8}-[0-9a-f-]{27,})\.json$", filename, re.I)
        row["session_id"] = (result or {}).get("session_id") if isinstance(result, dict) else None
        row["session_id"] = row["session_id"] or (sid_match.group(1) if sid_match else "")
        row["jsonl_path"] = None
        row["session_result_path"] = str(path) if path else None
        row["subagent_jsonl_paths"] = []
        row["full_eval_path"] = full_eval_path
        row["anticheat_path"] = anticheat_path
        row["num_turns"] = (result or {}).get("num_turns") if isinstance(result, dict) else None
        row["num_turns_source"] = f"{path}.num_turns" if path and row["num_turns"] is not None else None
        row["token_source"] = "final_model_usage" if authoritative else ("result_usage" if breakdown else "missing")
        row["subagent_tokens"] = 0
        for field in TOKEN_FIELDS:
            row[f"main_journal_{field}"] = None
            row[f"subagent_{field}"] = 0
            row[f"unattributed_{field}"] = None
        row["unattributed_session_tokens"] = None
        row["token_reconciliation_ratio"] = None
        row["token_reconciliation_status"] = "not_applicable" if breakdown else "missing"
        row["canonical_assistant_messages"] = None
        row["model_raw_aliases"] = list(breakdown)
        row["forbidden_debug_skill_calls"] = []
        row["evidence_status"] = "complete" if result is not None else "missing"
        out.append(row)
    return out


def serialize_csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_summary(rows: list[dict[str, Any]], output_json: Path, output_csv: Path, manifest: Path, pricing: Pricing) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest.resolve()),
        "pricing_id": pricing.pricing_id,
        "rows": rows,
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: serialize_csv_value(row.get(field)) for field in OUTPUT_FIELDS})


def write_paired(rows: list[dict[str, Any]], path: Path) -> None:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["operator"])][str(row["arm"])].append(row)
    fields = [
        "operator",
        "debug_external_clean",
        "debug_tokens",
        "debug_cost_usd",
        "debug_sessions",
        "debug_same_model_eligible",
        "regen_pass_at_1",
        "regen_pass_at_3",
        "regen_first_clean_trial",
        "regen_tokens_3_trials",
        "regen_cost_usd_3_trials",
        "regen_trials_completed",
        "regen_operational_sessions",
        "regen_operational_overhead_tokens",
        "regen_operational_overhead_cost_usd",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for operator in sorted(grouped):
            debug = grouped[operator].get("debug", []) + grouped[operator].get("cannbot-debug", [])
            all_regen = grouped[operator].get("regenerate", [])
            regen = sorted(
                [
                    row
                    for row in all_regen
                    if not row.get("is_operational_retry") and row.get("same_model_eligible")
                ],
                key=lambda row: int(row.get("trial") or 0),
            )
            operational = [row for row in all_regen if row.get("is_operational_retry")]
            eligible_debug = [row for row in debug if row.get("same_model_eligible")]
            clean_trials = [int(row["trial"]) for row in regen if row.get("external_clean") and row.get("trial") is not None]
            debug_clean = any(row.get("external_clean") for row in eligible_debug)
            writer.writerow(
                {
                    "operator": operator,
                    "debug_external_clean": str(debug_clean).lower(),
                    "debug_tokens": sum(int(row.get("total_tokens_all") or 0) for row in eligible_debug),
                    "debug_cost_usd": round(sum(float(row.get("cost_usd") or 0) for row in eligible_debug), 9),
                    "debug_sessions": len(eligible_debug),
                    "debug_same_model_eligible": str(bool(eligible_debug)).lower(),
                    "regen_pass_at_1": str(1 in clean_trials).lower(),
                    "regen_pass_at_3": str(bool(clean_trials)).lower(),
                    "regen_first_clean_trial": min(clean_trials) if clean_trials else "",
                    "regen_tokens_3_trials": sum(int(row.get("total_tokens_all") or 0) for row in regen),
                    "regen_cost_usd_3_trials": (
                        round(sum(float(row["cost_usd"]) for row in regen), 9)
                        if regen and all(row.get("cost_attributable") for row in regen)
                        else ""
                    ),
                    "regen_trials_completed": sum(row.get("terminal_status") in {"success", "failed"} for row in regen),
                    "regen_operational_sessions": len(operational),
                    "regen_operational_overhead_tokens": sum(
                        int(row.get("total_tokens_all") or 0) for row in operational
                    ),
                    "regen_operational_overhead_cost_usd": round(
                        sum(float(row.get("cost_usd") or 0) for row in operational), 9
                    ),
                    "note": (
                        ""
                        if eligible_debug and regen
                        else "no paired eligible K3 Debug and Regenerate sessions"
                    ),
                }
            )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--manifest", type=Path, default=here / "run_manifest.json")
    parser.add_argument("--pricing", type=Path, default=here / "pricing_table.json")
    parser.add_argument("--output-json", type=Path, default=here / "token_cost_summary.json")
    parser.add_argument("--output-csv", type=Path, default=here / "token_cost_summary.csv")
    parser.add_argument("--paired-output", type=Path, default=here / "debug_vs_regenerate_paired.csv")
    parser.add_argument("--strict", action="store_true", help="fail on reconciliation mismatches or forbidden debug-skill calls")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    manifest_data = load_json(args.manifest)
    pricing_data = load_json(args.pricing)
    if not isinstance(manifest_data, (dict, list)):
        raise SystemExit(f"invalid manifest: {args.manifest}")
    if not isinstance(pricing_data, dict):
        raise SystemExit(f"invalid pricing table: {args.pricing}")
    records = manifest_data if isinstance(manifest_data, list) else manifest_data.get("runs", [])
    pricing = Pricing(pricing_data)
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("arm") in {"regenerate", "cannbot-debug"}:
            rows.append(regen_row(record, pricing))
        elif record.get("arm") == "debug":
            rows.extend(debug_rows(record, pricing))
        else:
            raise ValueError(f"unknown arm in manifest: {record.get('arm')!r}")
    write_summary(rows, args.output_json, args.output_csv, args.manifest, pricing)
    write_paired(rows, args.paired_output)
    mismatches = [row for row in rows if row.get("token_reconciliation_status") == "mismatch"]
    forbidden = [row for row in rows if row.get("forbidden_debug_skill_calls")]
    missing_required = [
        row
        for row in rows
        if row.get("arm") == "cannbot-debug"
        and not row.get("cannbot_debug_skill_observed")
    ]
    print(
        json.dumps(
            {
                "rows": len(rows),
                "mismatches": len(mismatches),
                "forbidden_debug_skill_sessions": len(forbidden),
                "missing_required_cannbot_debug_skill_sessions": len(missing_required),
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
                "paired_output": str(args.paired_output),
            },
            indent=2,
        )
    )
    if args.strict and (mismatches or forbidden or missing_required):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
