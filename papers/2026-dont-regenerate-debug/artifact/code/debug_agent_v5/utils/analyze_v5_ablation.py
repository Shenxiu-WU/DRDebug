#!/usr/bin/env python3
"""Build the evidence-backed V5 ablation closure package."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ARMS = (
    "full",
    "no_kb",
    "no_diagnostic_evidence",
    "no_loopguard",
    "no_anticheat",
    "no_fulleval",
    "baseline",
)
EVIDENCE_CAPABLE_ARMS = {
    "full", "no_kb", "no_diagnostic_evidence", "no_loopguard",
}
DIAGNOSTICS_DISABLED_ARMS = {"no_diagnostic_evidence", "baseline"}
KB_DISABLED_ARMS = {"no_kb", "no_diagnostic_evidence", "baseline"}
RECOVERY_DISABLED_ARMS = {"baseline"}
PROVIDER_ERROR_RE = re.compile(
    r"\b(?:400|401|403|429|5\d\d)\b|refusal|context[_ -]?limit|"
    r"output[_ -]?limit|max[_ -]?turn",
    re.I,
)


def _load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _json_dict(path: Path) -> dict[str, Any] | None:
    value = _load_json(path, None)
    return value if isinstance(value, dict) else None


def _dict_has_keys(value: object, keys: Iterable[str]) -> bool:
    return isinstance(value, dict) and all(key in value for key in keys)


def _valid_probe_policy(path: Path) -> bool:
    value = _json_dict(path)
    return (
        _dict_has_keys(
            value,
            (
                "attempt", "failure_type", "policy", "policy_pass",
                "metadata_complete", "observed_status",
            ),
        )
        and isinstance(value.get("policy_pass"), bool)
        and value.get("metadata_complete") is True
        and value.get("observed_status")
        in {"skipped", "executed", "not_applicable"}
        and isinstance(value.get("ablation_violation", False), bool)
    )


def _valid_direction_artifact(tuning: Path) -> bool:
    directions = _json_dict(tuning / "tuning_directions.json")
    entries = directions.get("entries") if directions else None
    if isinstance(entries, list) and entries:
        return all(
            _dict_has_keys(
                entry,
                (
                    "attempt", "fix_type", "direction_verdict",
                    "direction_reason", "outcome", "evidence",
                ),
            )
            for entry in entries
        )
    summaries = sorted(tuning.glob("diagnosis_summary_attempt_*.json"))
    return bool(summaries) and all(
        _dict_has_keys(
            _json_dict(path),
            ("attempt", "fix_type", "direction_verdict", "direction_reason"),
        )
        for path in summaries
    )


def _valid_kb_trace(path: Path) -> bool:
    value = _load_json(path, None)
    if not isinstance(value, list) or not value:
        return False
    required = (
        "attempt", "retrieved_ids", "injected_ids", "declared_used_ids",
        "declared_unknown_ids", "usage_trace_complete",
    )
    return all(
        _dict_has_keys(entry, required)
        and entry.get("usage_trace_complete") is True
        and all(
            isinstance(entry.get(key), list)
            for key in (
                "retrieved_ids", "injected_ids",
                "declared_used_ids", "declared_unknown_ids",
            )
        )
        for entry in value
    )


def _valid_forensics_report(path: Path) -> bool:
    value = _json_dict(path)
    return (
        _dict_has_keys(value, ("attempt", "status", "primary_hint"))
        and bool(
            value.get("source")
            or value.get("version")
            or value.get("generated_at")
        )
    )


def _valid_checkpoint(tuning: Path) -> bool:
    current = tuning / "history" / "current_best"
    metric = _json_dict(current / "metric.json")
    manifest = _json_dict(current / "manifest.json")
    return (
        _dict_has_keys(
            metric,
            ("attempt", "case_pass_rate", "match_rate", "correctness_passed"),
        )
        and _dict_has_keys(
            manifest,
            ("schema_version", "complete", "attempt", "files"),
        )
        and manifest.get("complete") is True
        and manifest.get("attempt") == metric.get("attempt")
        and isinstance(manifest.get("files"), list)
        and bool(manifest.get("files"))
        and all(
            _dict_has_keys(entry, ("path", "size", "sha256"))
            for entry in manifest.get("files")
        )
        and (current / "src").is_dir()
    )


def _valid_claude_result(path: Path) -> bool:
    value = _json_dict(path)
    return (
        _dict_has_keys(value, ("num_turns", "modelUsage", "total_cost_usd"))
        and isinstance(value.get("modelUsage"), dict)
        and bool(value.get("modelUsage"))
    )


def _valid_audit_context(path: Path) -> bool:
    value = _json_dict(path)
    return (
        _dict_has_keys(
            value,
            (
                "schema_version", "attempt", "generated_by", "generated_at",
                "source_type", "source_path", "source_parseable",
                "primary_hint", "direction_verdict",
            ),
        )
        and value.get("generated_by") == "engine_pre_agent_audit"
        and value.get("source_parseable") is True
    )


def claude_result_paths(task: Path) -> list[Path]:
    archive = sorted(
        (task / "precision_tuning" / "claude_results").glob("*.json")
    )
    if archive:
        return archive
    return sorted(task.glob("_claude_result_attempt*.json"))


def summarize_claude_cost(paths: Iterable[Path]) -> dict[str, Any]:
    totals: dict[str, float] = {
        "turns": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "cost_usd": 0.0,
    }
    models: set[str] = set()
    context_windows: set[int] = set()
    max_output_tokens: set[int] = set()
    provider_errors: list[str] = []
    result_count = 0
    for path in paths:
        payload = _load_json(path, {})
        if not isinstance(payload, dict):
            continue
        result_count += 1
        totals["turns"] += _number(payload.get("num_turns"))
        usage = payload.get("modelUsage")
        usage_cost = 0.0
        if isinstance(usage, dict):
            for model, record in usage.items():
                if isinstance(record, dict):
                    models.add(str(model))
                    if isinstance(record.get("contextWindow"), int):
                        context_windows.add(record["contextWindow"])
                    if isinstance(record.get("maxOutputTokens"), int):
                        max_output_tokens.add(record["maxOutputTokens"])
                    totals["input_tokens"] += _number(record.get("inputTokens"))
                    totals["output_tokens"] += _number(record.get("outputTokens"))
                    totals["cache_creation_tokens"] += _number(
                        record.get("cacheCreationInputTokens"))
                    totals["cache_read_tokens"] += _number(
                        record.get("cacheReadInputTokens"))
                    usage_cost += _number(record.get("costUSD"))
        top_cost = payload.get("total_cost_usd")
        totals["cost_usd"] += (
            _number(top_cost) if top_cost is not None else usage_cost
        )
        error_text = json.dumps({
            key: payload.get(key)
            for key in (
                "api_error_status", "is_error", "terminal_reason",
                "claude_state", "error", "subtype",
            )
            if key in payload
        }, ensure_ascii=False)
        provider_errors.extend(
            match.group(0).lower()
            for match in PROVIDER_ERROR_RE.finditer(error_text)
        )
    totals["turns"] = int(totals["turns"])
    for key in (
        "input_tokens", "output_tokens",
        "cache_creation_tokens", "cache_read_tokens",
    ):
        totals[key] = int(totals[key])
    totals["total_tokens"] = sum(
        totals[key]
        for key in (
            "input_tokens", "output_tokens",
            "cache_creation_tokens", "cache_read_tokens",
        )
    )
    totals["cost_usd"] = round(float(totals["cost_usd"]), 8)
    totals["result_count"] = result_count
    totals["models"] = sorted(models)
    totals["context_windows"] = sorted(context_windows)
    totals["max_output_tokens"] = sorted(max_output_tokens)
    totals["provider_errors"] = sorted(provider_errors)
    return totals


def _coverage_map(root: Path) -> dict[str, dict[str, Any]]:
    audit = _load_json(
        root / "experiment_control" / "dataset_audit.json", {})
    return {
        str(row.get("task")): row.get("full_eval_coverage") or {}
        for row in audit.get("tasks", [])
        if isinstance(row, dict) and row.get("task")
    }


def _posthoc_map(root: Path, arm: str) -> dict[str, dict[str, Any]]:
    payload = _load_json(
        root / "posthoc" / f"arm_{arm}" / "posthoc_results.json", {})
    return {
        str(row.get("task")): row
        for row in payload.get("rows", [])
        if isinstance(row, dict) and row.get("task")
    }


def _full_eval_evidence(task: Path, applicable: bool) -> tuple[bool, str]:
    if not applicable:
        return True, "not_applicable_or_coverage_equivalent"
    paths = sorted(
        (task / "precision_tuning").glob(
            "validation_result_attempt_*_full.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not paths:
        return False, "missing"
    payload = _load_json(paths[-1], {})
    passed = bool(
        payload.get("correctness_passed")
        or payload.get("coverage_equivalent")
    )
    return passed, paths[-1].name


def _event_rows(task: Path) -> tuple[list[dict[str, Any]], bool]:
    path = task / ".debug_events" / "events.jsonl"
    if not path.is_file():
        return [], False
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                return [], False
            rows.append(payload)
    except (OSError, ValueError, TypeError):
        return [], False
    return rows, True


def _evidence_check(
    *,
    required: bool,
    observed: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "required": required,
        "observed": observed,
        "complete": observed or not required,
        "reason": reason,
    }


def _observability(
    task: Path,
    status: dict[str, Any],
    arm: str,
) -> dict[str, Any]:
    tuning = task / "precision_tuning"
    events, events_valid = _event_rows(task)
    started_steps = {
        str((event.get("action") or {}).get("step"))
        for event in events
        if event.get("type") == "action_started"
    }
    completed = [
        event for event in events
        if event.get("type") == "action_completed"
    ]
    completed_steps = {
        str((event.get("action") or {}).get("step"))
        for event in completed
    }

    diagnose_started = "diagnose_and_fix" in started_steps
    validate_started = "validate" in started_steps
    knowledge_started = "knowledge_search" in started_steps
    forensics_started = "forensics" in started_steps
    audit_started = "audit" in started_steps
    checkpoint_started = bool(
        {"baseline_checkpoint", "checkpoint_and_rollback"} & started_steps
    )
    checkpoint_completed = [
        event for event in completed
        if str((event.get("action") or {}).get("step"))
        in {"baseline_checkpoint", "checkpoint_and_rollback"}
    ]
    checkpoint_state_required = any(
        (
            ((event.get("result") or {}).get("checkpoint") or {}).get("success")
            is True
        )
        for event in checkpoint_completed
    )
    rollback_events = [
        event for event in completed
        if str((event.get("action") or {}).get("step"))
        == "checkpoint_and_rollback"
        and (event.get("result") or {}).get("rolled_back") is True
    ]

    status_observed = _dict_has_keys(
        _json_dict(task / "debug_status.json"),
        (
            "session_outcome", "ended_at", "objective_success",
            "reportable_success", "anti_cheat_pass", "ast_degrade_pass",
        ),
    )
    run_summary_observed = _dict_has_keys(
        _json_dict(task / "run_summary.json"),
        (
            "schema_version", "session_outcome", "attempts_used",
            "turns", "gate", "forensics",
        ),
    )
    claude_paths = claude_result_paths(task)
    probe_paths = sorted(tuning.glob("probe_policy_attempt*.json"))
    forensics_paths = sorted(tuning.glob("forensics_report_*.json"))
    audit_paths = sorted(tuning.glob("audit_context_attempt_*.json"))
    direction_observed = _valid_direction_artifact(tuning)
    kb_observed = _valid_kb_trace(tuning / "kb_usage_trace.json")
    forensics_observed = bool(forensics_paths) and all(
        _valid_forensics_report(path) for path in forensics_paths
    )
    checkpoint_observed = _valid_checkpoint(tuning)
    rollback_observed = bool(rollback_events) and all(
        _dict_has_keys(
            event.get("result"),
            ("success", "rolled_back", "from_attempt", "best_attempt"),
        )
        and (event.get("result") or {}).get("success") is True
        for event in rollback_events
    )
    checks = {
        "status": _evidence_check(
            required=True,
            observed=status_observed,
            reason="terminal_artifact",
        ),
        "events": _evidence_check(
            required=True,
            observed=events_valid,
            reason="event_source",
        ),
        "run_summary": _evidence_check(
            required=True,
            observed=run_summary_observed,
            reason="terminal_artifact",
        ),
        "claude_results": _evidence_check(
            required=diagnose_started,
            observed=bool(claude_paths) and all(
                _valid_claude_result(path) for path in claude_paths
            ),
            reason=(
                "diagnose_action_started"
                if diagnose_started else "diagnose_action_not_triggered"
            ),
        ),
        "probe": _evidence_check(
            required=(
                diagnose_started and arm not in DIAGNOSTICS_DISABLED_ARMS
            ),
            observed=bool(probe_paths) and all(
                _valid_probe_policy(path) for path in probe_paths
            ),
            reason=(
                "disabled_by_arm"
                if arm in DIAGNOSTICS_DISABLED_ARMS
                else (
                    "diagnose_action_started"
                    if diagnose_started else "diagnose_action_not_triggered"
                )
            ),
        ),
        "direction": _evidence_check(
            required=validate_started,
            observed=direction_observed,
            reason=(
                "validate_action_started"
                if validate_started else "validate_action_not_triggered"
            ),
        ),
        "kb": _evidence_check(
            required=(
                (knowledge_started or diagnose_started)
                and arm not in KB_DISABLED_ARMS
            ),
            observed=kb_observed,
            reason=(
                "disabled_by_arm"
                if arm in KB_DISABLED_ARMS
                else (
                    "knowledge_or_diagnose_action_started"
                    if knowledge_started or diagnose_started
                    else "knowledge_action_not_triggered"
                )
            ),
        ),
        "forensics": _evidence_check(
            required=(
                forensics_started and arm not in DIAGNOSTICS_DISABLED_ARMS
            ),
            observed=forensics_observed,
            reason=(
                "disabled_by_arm"
                if arm in DIAGNOSTICS_DISABLED_ARMS
                else (
                    "forensics_action_started"
                    if forensics_started else "forensics_action_not_triggered"
                )
            ),
        ),
        "audit": _evidence_check(
            required=audit_started,
            observed=(
                bool(audit_paths)
                and all(_valid_audit_context(path) for path in audit_paths)
            ),
            reason=(
                "audit_action_started"
                if audit_started else "audit_action_not_triggered"
            ),
        ),
        "checkpoint": _evidence_check(
            required=(
                checkpoint_started and arm not in RECOVERY_DISABLED_ARMS
            ),
            observed=(
                bool(checkpoint_completed)
                and (checkpoint_observed or not checkpoint_state_required)
            ),
            reason=(
                "disabled_by_arm"
                if arm in RECOVERY_DISABLED_ARMS
                else (
                    "checkpoint_action_started"
                    if checkpoint_started else "checkpoint_action_not_triggered"
                )
            ),
        ),
        "rollback": _evidence_check(
            required=bool(rollback_events),
            observed=rollback_observed,
            reason=(
                "rollback_recorded_in_action_result"
                if rollback_events else "rollback_not_triggered"
            ),
        ),
    }
    return {
        "complete": all(check["complete"] for check in checks.values()),
        "checks": checks,
        "started_actions": sorted(started_steps - {"None"}),
        "completed_actions": sorted(completed_steps - {"None"}),
    }


def _mechanism_metrics(task: Path) -> dict[str, Any]:
    tuning = task / "precision_tuning"
    directions = _json_dict(tuning / "tuning_directions.json") or {}
    entries = directions.get("entries")
    if not isinstance(entries, list):
        entries = []
    kb_trace = _load_json(tuning / "kb_usage_trace.json", [])
    if not isinstance(kb_trace, list):
        kb_trace = []
    reports = [
        value
        for path in sorted(tuning.glob("forensics_report_*.json"))
        if (value := _json_dict(path)) is not None
    ]
    events, _ = _event_rows(task)
    event_text = json.dumps(events, ensure_ascii=False)
    probe_records = [
        value
        for path in sorted(tuning.glob("probe_policy_attempt*.json"))
        if (value := _json_dict(path)) is not None
    ]
    return {
        "no_improvement_attempts": sum(
            entry.get("outcome") in {"stagnant", "regressed"}
            for entry in entries
            if isinstance(entry, dict)
        ),
        "direction_switches": sum(
            str(entry.get("direction_verdict") or "").lower()
            in {"否", "switch", "changed", "change"}
            for entry in entries
            if isinstance(entry, dict)
        ),
        "kb_retrieved_ids": sorted({
            str(item)
            for entry in kb_trace
            if isinstance(entry, dict)
            for item in (entry.get("retrieved_ids") or [])
        }),
        "kb_injected_ids": sorted({
            str(item)
            for entry in kb_trace
            if isinstance(entry, dict)
            for item in (entry.get("injected_ids") or [])
        }),
        "kb_declared_used_ids": sorted({
            str(item)
            for entry in kb_trace
            if isinstance(entry, dict)
            for item in (entry.get("declared_used_ids") or [])
        }),
        "kb_declared_unknown_ids": sorted({
            str(item)
            for entry in kb_trace
            if isinstance(entry, dict)
            for item in (entry.get("declared_unknown_ids") or [])
        }),
        "kb_usage_trace_complete": bool(kb_trace) and all(
            isinstance(entry, dict)
            and entry.get("usage_trace_complete") is True
            for entry in kb_trace
        ),
        "kb_declared_ids_valid": bool(kb_trace) and all(
            isinstance(entry, dict)
            and not entry.get("declared_unknown_ids")
            for entry in kb_trace
        ),
        "probe_policy_records": len(probe_records),
        "probe_metadata_complete": bool(probe_records) and all(
            record.get("metadata_complete") is True
            for record in probe_records
        ),
        "probe_policy_compliant": bool(probe_records) and all(
            record.get("policy_pass") is True
            and record.get("ablation_violation") is not True
            for record in probe_records
        ),
        "forensics_report_count": len(reports),
        "forensics_cache_hits": sum(
            report.get("cache_hit") is True or report.get("cached") is True
            for report in reports
        ),
        "forensics_reuses": sum(
            report.get("reused_after_rollback") is True
            or report.get("reuse_kind") is not None
            for report in reports
        ),
        "forensics_degraded": sum(
            report.get("degraded_evidence") is True
            or report.get("forensics_degraded") is True
            or report.get("unavailable_reason")
            in {"build_failed", "runtime_error"}
            or report.get("status") in {"build_failed", "runtime_error"}
            for report in reports
        ),
        "rollback_count": sum(
            event.get("type") == "action_completed"
            and str((event.get("action") or {}).get("step"))
            == "checkpoint_and_rollback"
            and (event.get("result") or {}).get("rolled_back") is True
            for event in events
        ),
        "sigterm_count": event_text.upper().count("SIGTERM"),
        "sigkill_count": event_text.upper().count("SIGKILL"),
    }


def task_row(
    root: Path,
    arm: str,
    task: Path,
    *,
    coverage: dict[str, dict[str, Any]],
    posthoc: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rel = str(task.relative_to(root / f"arm_{arm}" / "tasks"))
    status = _load_json(task / "debug_status.json", {})
    run_summary = _load_json(task / "run_summary.json", {})
    cost = summarize_claude_cost(claude_result_paths(task))
    coverage_row = coverage.get(rel, {})
    full_pass, full_source = _full_eval_evidence(
        task, bool(coverage_row.get("full_eval_applicable")))
    posthoc_row = posthoc.get(rel, {})
    objective = status.get("objective_success") is True
    reportable = status.get("reportable_success") is True
    anti = status.get("anti_cheat_pass") is True
    ast = status.get("ast_degrade_pass") is True
    observability = _observability(task, status, arm)
    mechanism_metrics = _mechanism_metrics(task)
    mechanism_compliant = (
        (
            arm in DIAGNOSTICS_DISABLED_ARMS
            or mechanism_metrics["probe_policy_compliant"]
        )
        and (
            arm in KB_DISABLED_ARMS
            or (
                mechanism_metrics["kb_usage_trace_complete"]
                and mechanism_metrics["kb_declared_ids_valid"]
            )
        )
    )
    evidence_backed = (
        arm in EVIDENCE_CAPABLE_ARMS
        and objective and reportable and anti and ast and full_pass
        and observability["complete"]
        and mechanism_compliant
    )
    row: dict[str, Any] = {
        "arm": arm,
        "task": rel,
        "session_outcome": status.get("session_outcome"),
        "entry_failure_type": status.get("entry_failure_type"),
        "final_failure_type": status.get("final_failure_type"),
        "attempts_used": int(_number(status.get("attempts_used"))),
        "terminal_complete": bool(
            status.get("session_outcome") and status.get("ended_at")
        ),
        "objective_success": objective,
        "reportable_success": reportable,
        "anti_cheat_pass": anti,
        "ast_degrade_pass": ast,
        "full_eval_applicable": bool(
            coverage_row.get("full_eval_applicable")),
        "full_eval_pass": full_pass,
        "full_eval_evidence": full_source,
        "evidence_backed_success": evidence_backed,
        "posthoc_clean_success": (
            posthoc_row.get("posthoc_clean_success") is True),
        "posthoc_run_state": posthoc_row.get("run_state"),
        "posthoc_infrastructure_error": posthoc_row.get(
            "infrastructure_error"),
        "posthoc_transient_rechecks": len(
            posthoc_row.get("transient_infrastructure_rechecks") or []
        ),
        "posthoc_build_pass": (
            (posthoc_row.get("build") or {}).get("passed") is True),
        "notes": str(status.get("notes") or ""),
        "observability_complete": observability["complete"],
        "mechanism_compliant": mechanism_compliant,
        "observability": observability,
        "mechanism_metrics": mechanism_metrics,
        **mechanism_metrics,
        **cost,
    }
    if isinstance(run_summary, dict):
        row["no_improvement_signal"] = (
            (run_summary.get("gate") or {}).get("final_loop_signal"))
    return row


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def classify_long_failures(
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Apply one frozen full-arm burden threshold to every arm."""

    full_rows = [row for row in rows if row["arm"] == "full"]
    thresholds = {
        key: percentile(
            [float(row[key]) for row in full_rows if float(row[key]) > 0],
            0.75,
        )
        for key in ("turns", "total_tokens", "cost_usd")
    }
    for row in rows:
        components = [
            key for key, threshold in thresholds.items()
            if threshold > 0 and float(row[key]) >= threshold
        ]
        row["long_failure_components"] = components
        row["long_failure_threshold_source"] = "full_arm_p75"
        row["long_failure"] = (
            not row["posthoc_clean_success"] and len(components) >= 2
        )
    return thresholds


def mcnemar_exact(full: list[bool], treatment: list[bool]) -> dict[str, Any]:
    b = sum(left and not right for left, right in zip(full, treatment))
    c = sum(not left and right for left, right in zip(full, treatment))
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index) * (0.5 ** discordant)
            for index in range(0, min(b, c) + 1)
        )
        p_value = min(1.0, 2.0 * tail)
    return {
        "full_only": b,
        "treatment_only": c,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def paired_bootstrap_ci(
    full: list[bool],
    treatment: list[bool],
    *,
    samples: int = 10000,
    seed: int = 20260724,
) -> dict[str, float]:
    if not full:
        return {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    diffs = []
    n = len(full)
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(
            int(treatment[index]) - int(full[index]) for index in indices
        ) / n)
    return {
        "difference": (
            sum(map(int, treatment)) - sum(map(int, full))) / n,
        "ci_low": percentile(diffs, 0.025),
        "ci_high": percentile(diffs, 0.975),
    }


def _manifest_compliance(
    root: Path,
    arm: str,
    task_rows: list[dict[str, Any]],
    posthoc_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    frozen = _load_json(
        root / "experiment_control" / "arm_manifests" / f"arm_{arm}.json", {})
    runtime = _load_json(root / f"arm_{arm}" / "experiment_manifest.json", {})
    mapping = {
        "task_count": len(runtime.get("targets") or []),
        "containers": str(runtime.get("containers") or "").split(","),
        "npus": [
            int(value) for value in str(runtime.get("npus") or "").split(",")
            if value
        ],
        "max_attempts": runtime.get("max_attempts"),
        "max_turns": int(runtime.get("max_turns") or 0),
        "soft_task_turns": runtime.get("soft_task_turns"),
        "max_task_turns": runtime.get("max_task_turns"),
        "timeout": runtime.get("timeout_sec"),
        "agent": runtime.get("agent"),
        "entry_failure_type": runtime.get("entry_failure_type"),
        "arm": runtime.get("ablate_profile"),
        "kb_path": runtime.get("kb_path"),
        "model": runtime.get("model"),
        "provider_assignment_mode": runtime.get("provider_assignment_mode"),
        "provider_names": runtime.get("providers"),
        "provider_env_storage": runtime.get("provider_env_storage"),
        "usage_query_enabled": runtime.get("usage_query_enabled"),
        "mixed_provider_enabled": runtime.get("mixed_provider_enabled"),
        "kb_read_only": runtime.get("kb_read_only"),
    }
    mismatches = {
        key: {"expected": frozen.get(key), "actual": value}
        for key, value in mapping.items()
        if frozen.get(key) != value
    }
    task_root = root / f"arm_{arm}" / "tasks"
    prohibited: list[str] = []
    if arm in {"no_kb", "no_diagnostic_evidence", "baseline"}:
        for path in task_root.glob("level*/*/precision_tuning/*knowledge*.json"):
            prohibited.append(str(path.relative_to(root)))
    if arm in {"no_diagnostic_evidence", "baseline"}:
        for path in task_root.glob("level*/*/precision_tuning/forensics_report_*.json"):
            prohibited.append(str(path.relative_to(root)))
    if arm in {"no_fulleval", "baseline"}:
        for path in task_root.glob(
            "level*/*/precision_tuning/validation_result_attempt_*_full.json"
        ):
            prohibited.append(str(path.relative_to(root)))
    if arm == "no_anticheat":
        for path in task_root.glob("level*/*/_anticheat.json"):
            prohibited.append(str(path.relative_to(root)))
    task_count = len(task_rows)
    terminal_count = sum(row["terminal_complete"] for row in task_rows)
    observability_complete_count = sum(
        row.get("observability_complete") is True for row in task_rows
    )
    posthoc_count = len(posthoc_rows)
    posthoc_completed_count = sum(
        row.get("run_state") == "completed"
        for row in posthoc_rows.values()
    )
    expected_model = frozen.get("model")
    expected_context = frozen.get("model_context_window")
    model_usage_violations = [
        {
            "task": row["task"],
            "models": row["models"],
            "context_windows": row["context_windows"],
        }
        for row in task_rows
        if row["result_count"] > 0
        and (
            row["models"] != [expected_model]
            or row["context_windows"] != [expected_context]
        )
    ]
    supervisor_state = _load_json(
        root / f"arm_{arm}" / "quota_batch_cc_state.json", {})
    supervisor_completed = (
        supervisor_state.get("event") == "completed"
        and supervisor_state.get("done") == int(frozen.get("task_count") or 0)
        and supervisor_state.get("pending") == 0
    )
    container_contract = _load_json(
        root
        / "experiment_control"
        / "container_contracts"
        / f"{arm}_before.json",
        {},
    )
    expected_kb_mask = arm in KB_DISABLED_ARMS
    expected_forensics_mask = arm in DIAGNOSTICS_DISABLED_ARMS
    container_id = container_contract.get("container_id")
    contract_ids = [
        row.get("container_id")
        for path in sorted(
            (
                root
                / "experiment_control"
                / "container_contracts"
            ).glob("*_before.json")
        )
        if isinstance((row := _load_json(path, {})), dict)
        and row.get("container_id")
    ]
    container_contract_passed = (
        container_contract.get("passed") is True
        and container_contract.get("arm") == arm
        and container_contract.get("fresh_label") is True
        and isinstance(container_id, str)
        and bool(container_id)
        and contract_ids.count(container_id) == 1
        and _dict_has_keys(container_contract, ("created_at", "image_id"))
        and container_contract.get("privileged") is True
        and container_contract.get("kb_masked") is expected_kb_mask
        and container_contract.get("forensics_masked")
        is expected_forensics_mask
        and "SYS_ADMIN" in (container_contract.get("cap_drop") or [])
        and container_contract.get("mount_modes") == {
            "root_rw": False,
            "outputs_rw": True,
            "dataset_rw": False,
            "control_rw": False,
        }
    )
    complete = (
        task_count == int(frozen.get("task_count") or 0)
        and terminal_count == int(frozen.get("task_count") or 0)
        and observability_complete_count == int(frozen.get("task_count") or 0)
        and posthoc_count == int(frozen.get("task_count") or 0)
        and posthoc_completed_count == int(frozen.get("task_count") or 0)
        and supervisor_completed
        and container_contract_passed
    )
    return {
        "arm": arm,
        "passed": (
            complete
            and not mismatches
            and not prohibited
            and not model_usage_violations
        ),
        "task_directory_count": task_count,
        "terminal_task_count": terminal_count,
        "observability_complete_count": observability_complete_count,
        "posthoc_task_count": posthoc_count,
        "posthoc_completed_count": posthoc_completed_count,
        "supervisor_completed": supervisor_completed,
        "container_contract_passed": container_contract_passed,
        "container_contract": container_contract,
        "supervisor_status": supervisor_state.get("event"),
        "expected_task_count": frozen.get("task_count"),
        "manifest_mismatches": mismatches,
        "prohibited_artifacts": prohibited,
        "model_usage_violations": model_usage_violations,
    }


def _failed_cycles(root: Path) -> list[dict[str, Any]]:
    rows = []
    for arm in ARMS:
        for path in sorted(
            (root / f"arm_{arm}" / "retry_evidence").glob(
                "**/archive_manifest.json")
        ):
            payload = _load_json(path, {})
            rows.append({
                "arm": arm,
                "archive": str(path.relative_to(root)),
                "task": payload.get("target") or path.parent.parent.name,
                "reason": payload.get("reason"),
                "excluded_from_primary_cost": (
                    (payload.get("cost_accounting") or {}).get(
                        "excluded_from_primary_cost") is True
                ),
            })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _arm_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        ebs = sum(row["evidence_backed_success"] for row in selected)
        objective = sum(row["objective_success"] for row in selected)
        reportable = sum(row["reportable_success"] for row in selected)
        output.append({
            "arm": arm,
            "tasks": len(selected),
            "objective_success": objective,
            "reportable_success": reportable,
            "evidence_backed_success": ebs,
            "rsir_count": reportable - ebs,
            "rsir": round((reportable - ebs) / len(selected), 6)
            if selected else None,
            "osre_count": objective - ebs,
            "osre": round((objective - ebs) / len(selected), 6)
            if selected else None,
            "posthoc_clean_success": sum(
                row["posthoc_clean_success"] for row in selected),
            "long_failures": sum(row["long_failure"] for row in selected),
            "final_cycle_turns": sum(row["turns"] for row in selected),
            "final_cycle_tokens": sum(row["total_tokens"] for row in selected),
            "final_cycle_cost_usd": round(
                sum(row["cost_usd"] for row in selected), 8),
            "tokens_per_ebs": (
                round(sum(row["total_tokens"] for row in selected) / ebs, 2)
                if ebs else None
            ),
            "cost_per_ebs_usd": (
                round(sum(row["cost_usd"] for row in selected) / ebs, 8)
                if ebs else None
            ),
        })
    return output


def _paired_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_arm = {
        arm: {row["task"]: row for row in rows if row["arm"] == arm}
        for arm in ARMS
    }
    common_posthoc = set.intersection(*[
        {task for task, row in by_arm[arm].items()
         if row["posthoc_clean_success"]}
        for arm in ARMS
    ])
    common_ebs = set.intersection(*[
        {task for task, row in by_arm[arm].items()
         if row["evidence_backed_success"]}
        for arm in sorted(EVIDENCE_CAPABLE_ARMS)
    ])
    paired = []
    pairwise_intersections: dict[str, dict[str, list[str]]] = {}

    def append_cost_rows(
        scope: str,
        tasks: set[str],
        arms: Iterable[str],
    ) -> None:
        for arm in arms:
            selected = [by_arm[arm][task] for task in sorted(tasks)]
            paired.append({
                "scope": scope,
                "arm": arm,
                "task_count": len(selected),
                "turns": sum(row["turns"] for row in selected),
                "tokens": sum(row["total_tokens"] for row in selected),
                "cost_usd": round(
                    sum(row["cost_usd"] for row in selected), 8),
                "tokens_per_task": (
                    round(
                        sum(row["total_tokens"] for row in selected)
                        / len(selected),
                        2,
                    )
                    if selected else None
                ),
                "cost_per_task_usd": (
                    round(
                        sum(row["cost_usd"] for row in selected)
                        / len(selected),
                        8,
                    )
                    if selected else None
                ),
            })

    for arm in ARMS[1:]:
        posthoc_tasks = (
            {
                task for task, row in by_arm["full"].items()
                if row["posthoc_clean_success"]
            }
            & {
                task for task, row in by_arm[arm].items()
                if row["posthoc_clean_success"]
            }
        )
        ebs_tasks: set[str] = set()
        if arm in EVIDENCE_CAPABLE_ARMS:
            ebs_tasks = (
                {
                    task for task, row in by_arm["full"].items()
                    if row["evidence_backed_success"]
                }
                & {
                    task for task, row in by_arm[arm].items()
                    if row["evidence_backed_success"]
                }
            )
        pairwise_intersections[arm] = {
            "common_posthoc_tasks": sorted(posthoc_tasks),
            "common_evidence_backed_tasks": sorted(ebs_tasks),
        }
        append_cost_rows(
            f"full_vs_{arm}_common_posthoc",
            posthoc_tasks,
            ("full", arm),
        )
        if arm in EVIDENCE_CAPABLE_ARMS:
            append_cost_rows(
                f"full_vs_{arm}_common_ebs",
                ebs_tasks,
                ("full", arm),
            )

    for scope, tasks, arms in (
        ("common_posthoc_all_arms", common_posthoc, ARMS),
        ("common_ebs_evidence_arms", common_ebs, sorted(EVIDENCE_CAPABLE_ARMS)),
    ):
        append_cost_rows(scope, tasks, arms)
    return paired, {
        "common_posthoc_tasks": sorted(common_posthoc),
        "common_evidence_backed_tasks": sorted(common_ebs),
        "full_pairwise": pairwise_intersections,
    }


def _paired_success_tests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arm = {
        arm: {row["task"]: row for row in rows if row["arm"] == arm}
        for arm in ARMS
    }
    tasks = sorted(set.intersection(*[
        set(by_arm[arm]) for arm in ARMS
    ]))
    output = []
    for metric in (
        "objective_success", "reportable_success",
        "evidence_backed_success", "posthoc_clean_success",
    ):
        full = [bool(by_arm["full"][task][metric]) for task in tasks]
        for arm in ARMS[1:]:
            treatment = [bool(by_arm[arm][task][metric]) for task in tasks]
            output.append({
                "metric": metric,
                "arm": arm,
                **mcnemar_exact(full, treatment),
                **paired_bootstrap_ci(full, treatment),
            })
    return output


def _write_report(
    output: Path,
    summaries: list[dict[str, Any]],
    compliance: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    intersections: dict[str, Any],
    failed_cycles: list[dict[str, Any]],
) -> None:
    lines = [
        "# V5 Ablation Analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Primary cost includes only each operator's final valid supervisor cycle. "
        "Archived retry cycles are reported separately and are not summed.",
        "",
        "## Arm Summary",
        "",
        "| Arm | OSR | WRSR | EBSR | Post-hoc clean | RSIR | OSRE | "
        "Long failures | Tokens/EBS | Cost/EBS USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        n = row["tasks"]
        rsir = f"{row['rsir']:.4f}" if row["rsir"] is not None else "NA"
        osre = f"{row['osre']:.4f}" if row["osre"] is not None else "NA"
        lines.append(
            f"| {row['arm']} | {row['objective_success']}/{n} | "
            f"{row['reportable_success']}/{n} | "
            f"{row['evidence_backed_success']}/{n} | "
            f"{row['posthoc_clean_success']}/{n} | "
            f"{rsir} | {osre} | "
            f"{row['long_failures']} | "
            f"{row['tokens_per_ebs'] if row['tokens_per_ebs'] is not None else 'NA'} | "
            f"{row['cost_per_ebs_usd'] if row['cost_per_ebs_usd'] is not None else 'NA'} |"
        )
    lines.extend([
        "",
        "## Full-vs-Arm Paired Cost",
        "",
        "| Scope | Arm | Common successes | Tokens/task | Cost/task USD |",
        "|---|---|---:|---:|---:|",
    ])
    for row in paired:
        if not row["scope"].startswith("full_vs_"):
            continue
        lines.append(
            f"| {row['scope']} | {row['arm']} | {row['task_count']} | "
            f"{row['tokens_per_task'] if row['tokens_per_task'] is not None else 'NA'} | "
            f"{row['cost_per_task_usd'] if row['cost_per_task_usd'] is not None else 'NA'} |"
        )
    lines.extend([
        "",
        "The all-arm intersections are supplemental only: "
        f"post-hoc={len(intersections['common_posthoc_tasks'])}, "
        f"evidence-backed={len(intersections['common_evidence_backed_tasks'])}.",
        "",
        "## Contract Closure",
        "",
    ])
    for row in compliance:
        lines.append(
            f"- `{row['arm']}`: {'PASS' if row['passed'] else 'FAIL'}; "
            f"terminal={row['terminal_task_count']}, "
            f"observability={row['observability_complete_count']}, "
            f"posthoc_completed={row['posthoc_completed_count']}, "
            f"supervisor_completed={row['supervisor_completed']}."
        )
    lines.extend([
        "",
        "## Failed Provider Cycles",
        "",
        f"{len(failed_cycles)} archived cycles are listed in `failed_cycles.csv`; "
        "their cost is intentionally excluded from primary aggregates.",
        "",
        "See `per_task.csv`, `paired_success_tests.json`, "
        "`paired_success_cost.csv`, `long_failures.csv`, and "
        "`arm_compliance.json` for auditable evidence.",
    ])
    (output / "V5_ANALYSIS_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def _verify_arm_closure(
    root: Path,
    arm: str,
    expected_tasks: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coverage = _coverage_map(root)
    task_root = root / f"arm_{arm}" / "tasks"
    tasks = sorted(path for path in task_root.glob("level*/*") if path.is_dir())
    posthoc = _posthoc_map(root, arm)
    rows = [
        task_row(
            root,
            arm,
            task,
            coverage=coverage,
            posthoc=posthoc,
        )
        for task in tasks
    ]
    compliance = _manifest_compliance(root, arm, rows, posthoc)
    compliance["passed"] = (
        compliance["passed"]
        and len(rows) == expected_tasks
    )
    return compliance, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=27)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--verify-arm", choices=ARMS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.verify_arm:
        compliance, rows = _verify_arm_closure(
            args.experiment_root,
            args.verify_arm,
            args.expected_tasks,
        )
        (args.output / "arm_closure.json").write_text(
            json.dumps(compliance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_csv(
            args.output / "per_task.csv",
            rows,
            [
                "arm", "task", "terminal_complete", "objective_success",
                "reportable_success", "evidence_backed_success",
                "posthoc_clean_success", "observability_complete",
                "posthoc_run_state", "posthoc_infrastructure_error",
                "posthoc_transient_rechecks",
                "models", "context_windows", "turns", "total_tokens",
                "cost_usd",
            ],
        )
        return 0 if compliance["passed"] else 2

    coverage = _coverage_map(args.experiment_root)
    rows: list[dict[str, Any]] = []
    compliance = []
    for arm in ARMS:
        task_root = args.experiment_root / f"arm_{arm}" / "tasks"
        tasks = sorted(path for path in task_root.glob("level*/*") if path.is_dir())
        posthoc = _posthoc_map(args.experiment_root, arm)
        arm_rows = [task_row(
            args.experiment_root,
            arm,
            task,
            coverage=coverage,
            posthoc=posthoc,
        ) for task in tasks]
        rows.extend(arm_rows)
        compliance.append(_manifest_compliance(
            args.experiment_root, arm, arm_rows, posthoc))

    long_failure_thresholds = classify_long_failures(rows)
    summaries = _arm_summary(rows)
    paired, intersections = _paired_rows(rows)
    success_tests = _paired_success_tests(rows) if rows else []
    failed_cycles = _failed_cycles(args.experiment_root)
    long_failures = [row for row in rows if row["long_failure"]]

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_cost_scope": "final_valid_cycle_only",
        "failed_cycles_excluded_from_primary_cost": True,
        "long_failure_thresholds": {
            "source": "full_arm_p75",
            **long_failure_thresholds,
        },
        "arm_summary": summaries,
        "paired_intersections": intersections,
        "paired_success_tests": success_tests,
        "compliance": compliance,
        "rows": rows,
    }
    (args.output / "v5_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (args.output / "paired_success_tests.json").write_text(
        json.dumps(success_tests, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (args.output / "arm_compliance.json").write_text(
        json.dumps(compliance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    _write_csv(
        args.output / "arm_summary.csv",
        summaries,
        list(summaries[0]) if summaries else ["arm"],
    )
    per_task_fields = [
        "arm", "task", "session_outcome", "entry_failure_type",
        "final_failure_type", "attempts_used", "terminal_complete",
        "objective_success",
        "reportable_success", "evidence_backed_success",
        "posthoc_clean_success", "full_eval_pass", "turns",
        "input_tokens", "output_tokens", "cache_creation_tokens",
        "cache_read_tokens", "total_tokens", "cost_usd",
        "long_failure", "long_failure_components", "models",
        "long_failure_threshold_source",
        "context_windows", "max_output_tokens", "provider_errors", "notes",
        "posthoc_run_state", "posthoc_infrastructure_error",
        "posthoc_transient_rechecks",
        "observability_complete",
        "no_improvement_attempts", "direction_switches",
        "kb_retrieved_ids", "kb_injected_ids", "kb_declared_used_ids",
        "kb_declared_unknown_ids", "kb_usage_trace_complete",
        "kb_declared_ids_valid", "probe_policy_records",
        "probe_metadata_complete", "probe_policy_compliant",
        "mechanism_compliant", "forensics_report_count",
        "forensics_cache_hits", "forensics_reuses", "forensics_degraded",
        "rollback_count", "sigterm_count", "sigkill_count",
    ]
    _write_csv(args.output / "per_task.csv", rows, per_task_fields)
    _write_csv(
        args.output / "paired_success_cost.csv",
        paired,
        list(paired[0]) if paired else ["scope", "arm"],
    )
    _write_csv(
        args.output / "long_failures.csv",
        long_failures,
        per_task_fields,
    )
    _write_csv(
        args.output / "failed_cycles.csv",
        failed_cycles,
        ["arm", "task", "reason", "excluded_from_primary_cost", "archive"],
    )
    _write_report(
        args.output,
        summaries,
        compliance,
        paired,
        intersections,
        failed_cycles,
    )

    complete = (
        all(row["passed"] for row in compliance)
        and all(row["tasks"] == args.expected_tasks for row in summaries)
    )
    return 0 if complete or args.allow_incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
