"""knowledge_search.py — engine-owned KB retrieval before diagnose.

The older workflow described two precision_knowledge.py search calls in
SKILL.md, but those calls were left to the agent.  This module makes retrieval
deterministic and auditable: after forensics succeeds, runner calls this action
once per attempt, appending call-index 0/1 entries to knowledge_search_log.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

_ENGINE_DIR = Path(__file__).resolve().parent
_KB_SCRIPT = _ENGINE_DIR.parent / "scripts" / "precision_knowledge.py"
_SEARCH_TIMEOUT_SEC = 120


def _skip(reason: str, **extra) -> dict:
    return {"success": True, "skipped": True, "reason": reason, **extra}


def _load_forensics_report(task_dir: Path, attempt: int) -> tuple[Optional[dict], Optional[str]]:
    path = task_dir / "precision_tuning" / f"forensics_report_{attempt}.json"
    if not path.exists():
        return None, f"forensics report missing: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, ValueError) as exc:
        return None, f"forensics report parse failed: {exc}"


def _query_fields(report: dict) -> tuple[str, str]:
    op = report.get("L8_operator") or {}
    op_type = op.get("op_type") or "unknown"
    primary_hint = report.get("primary_hint") or "unknown"
    return str(op_type), str(primary_hint)


def _first_failing_output(report: dict) -> dict:
    outputs = report.get("outputs") or []
    for item in outputs:
        if isinstance(item, dict) and item.get("pass_fail") is False:
            return item
    for item in outputs:
        if isinstance(item, dict):
            return item
    return {}


def _infer_position(report: dict) -> Optional[str]:
    """Infer tail/boundary/scattered from forensics output, best-effort."""
    output = _first_failing_output(report)
    tail = output.get("tail_analysis") or {}
    if isinstance(tail, dict):
        text = json.dumps(tail, ensure_ascii=False).lower()
        if "tail" in text and ("mismatch" in text or "spike" in text or "高" in text):
            return "tail"

    shape = output.get("output_shape") or []
    worst = output.get("worst_elements") or []
    if not shape or not worst:
        return None

    tail_hits = 0
    boundary_hits = 0
    seen = 0
    for item in worst:
        idx = item.get("index") if isinstance(item, dict) else None
        if not isinstance(idx, list):
            continue
        for dim, coord in zip(shape, idx):
            if not isinstance(dim, int) or dim <= 1 or not isinstance(coord, int):
                continue
            seen += 1
            if coord >= int(dim * 0.8):
                tail_hits += 1
            elif coord <= max(1, int(dim * 0.2)):
                boundary_hits += 1
    if seen == 0:
        return None
    if tail_hits / seen >= 0.5:
        return "tail"
    if boundary_hits / seen >= 0.5:
        return "boundary"
    return "scattered"


def _read_log_summary(log_path: Path, attempt: int) -> dict:
    if not log_path.exists():
        return {"log_path": str(log_path), "entries": []}
    try:
        data = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"log_path": str(log_path), "entries": []}
    if not isinstance(data, list):
        return {"log_path": str(log_path), "entries": []}
    entries = [
        {
            "attempt": e.get("attempt"),
            "call_index": e.get("call_index"),
            "query": e.get("query"),
            "matched_count": e.get("matched_count"),
            "checklist_count": e.get("checklist_count"),
            "fallback_to_full_load": e.get("fallback_to_full_load"),
            "top_titles": (e.get("top_titles") or [])[:3],
            "top_ids": (e.get("top_ids") or [])[:3],
            "retrieved": (e.get("retrieved") or [])[:3],
            "match_reasons": (e.get("match_reasons") or [])[:3],
        }
        for e in data
        if isinstance(e, dict) and e.get("attempt") == attempt
    ]
    return {"log_path": str(log_path), "entries": entries[-2:]}


def _run_search(
    *,
    kb_path: str,
    task_dir: Path,
    op_name: str,
    attempt: int,
    op_type: str,
    pattern: str,
    position: Optional[str],
    call_index: int,
    top_k: int,
    _run: Callable,
) -> dict:
    cmd = [
        sys.executable,
        str(_KB_SCRIPT),
        "search",
        "--kb-path",
        str(kb_path),
        "--op-type",
        op_type,
        "--pattern",
        pattern,
        "--op-name",
        op_name,
        "--task-dir",
        str(task_dir),
        "--top-k",
        str(top_k),
        "--log-path",
        str(task_dir / "precision_tuning"),
        "--attempt",
        str(attempt),
        "--call-index",
        str(call_index),
    ]
    if position:
        cmd.extend(["--position", position])
    try:
        proc = _run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_SEARCH_TIMEOUT_SEC,
            check=False,
        )
        return {
            "call_index": call_index,
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
        }
    except Exception as exc:  # noqa: BLE001 — retrieval must not kill debug.
        return {"call_index": call_index, "returncode": None, "error": str(exc)}


def run_knowledge_search(
    task_dir,
    *,
    kb_path: Optional[str],
    op_name: str,
    attempt: int,
    top_k: int = 3,
    _run: Callable = subprocess.run,
) -> dict:
    """Run deterministic KB search for one precision attempt.

    Return value is event-friendly.  Any failure is non-fatal: the diagnose
    action should still run, but the failed search remains visible in events.
    """
    task_dir = Path(task_dir)
    if not kb_path:
        return _skip("kb_path 未配置，跳过知识库检索")
    if not _KB_SCRIPT.exists():
        return _skip(f"找不到检索脚本 {_KB_SCRIPT}")

    report, error = _load_forensics_report(task_dir, attempt)
    if report is None:
        return _skip(error or "forensics report unavailable")

    op_type, pattern = _query_fields(report)
    position = _infer_position(report)
    calls = [
        _run_search(
            kb_path=str(kb_path),
            task_dir=task_dir,
            op_name=op_name,
            attempt=attempt,
            op_type=op_type,
            pattern=pattern,
            position=None,
            call_index=0,
            top_k=top_k,
            _run=_run,
        ),
        _run_search(
            kb_path=str(kb_path),
            task_dir=task_dir,
            op_name=op_name,
            attempt=attempt,
            op_type=op_type,
            pattern=pattern,
            position=position,
            call_index=1,
            top_k=top_k,
            _run=_run,
        ),
    ]
    log_summary = _read_log_summary(
        task_dir / "precision_tuning" / "knowledge_search_log.json",
        attempt,
    )
    return {
        "success": all(c.get("returncode") == 0 for c in calls),
        "skipped": False,
        "op_type": op_type,
        "pattern": pattern,
        "position": position,
        "calls": calls,
        "knowledge_search": log_summary,
    }
