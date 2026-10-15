#!/usr/bin/env python3
"""Audit Regenerate isolation, forbidden skills, and natural self-correction."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
FORBIDDEN = {"ascendc-precision-debug", "ascendc-crash-debug", "ascendc-runtime-debug"}
FAILURE_RE = re.compile(
    r"(?:status\s*:\s*fail|\bfailed\b|traceback \(most recent call last\)|runtimeerror|"
    r"error:|command failed|exit code [1-9]|returned non-zero)",
    re.I,
)


def text_of(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(text_of(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(str(value.get(key, "")) for key in ("content", "text", "stdout", "stderr"))
    return ""


def inspect(path: Path) -> dict[str, Any]:
    calls: dict[str, dict[str, Any]] = {}
    events: list[tuple[str, str]] = []
    forbidden: list[str] = []
    if not path.is_file():
        return {"failures": 0, "repairs": 0, "forbidden": forbidden}
    for line in path.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                name = str(block.get("name") or "")
                call_id = str(block.get("id") or "")
                tool_input = block.get("input") if isinstance(block.get("input"), dict) else {}
                calls[call_id] = {"name": name, "input": tool_input}
                if name == "Skill" and tool_input.get("skill") in FORBIDDEN:
                    forbidden.append(str(tool_input["skill"]))
                if name in {"Edit", "Write", "NotebookEdit"}:
                    events.append(("mutation", name))
                elif name == "Bash":
                    command = str(tool_input.get("command") or "")
                    if re.search(r"\b(?:build|cmake|make|pytest|test|verification|run)\b", command, re.I):
                        events.append(("execution", command[:200]))
            elif block.get("type") == "tool_result":
                call = calls.get(str(block.get("tool_use_id") or ""), {})
                body = text_of(block.get("content"))
                if call.get("name") == "Bash" and (block.get("is_error") is True or FAILURE_RE.search(body)):
                    events.append(("failure", body[-300:]))
    failures = sum(kind == "failure" for kind, _ in events)
    repairs = 0
    seen_failure = False
    for kind, _ in events:
        if kind == "failure":
            seen_failure = True
        elif seen_failure and kind in {"mutation", "execution"}:
            repairs += 1
            seen_failure = False
    return {"failures": failures, "repairs": repairs, "forbidden": forbidden}


def main() -> int:
    manifest = json.loads((HERE / "run_manifest.json").read_text())
    results = []
    for record in manifest["runs"]:
        if record.get("arm") != "regenerate":
            continue
        paths = [Path(record["journal_main_path"])]
        subdir = Path(record["subagent_dir"])
        if subdir.is_dir():
            paths.extend(sorted(subdir.rglob("*.jsonl")))
        parts = [inspect(path) for path in paths]
        failures = sum(part["failures"] for part in parts)
        repairs = sum(part["repairs"] for part in parts)
        forbidden = sorted({skill for part in parts for skill in part["forbidden"]})
        run_record_path = Path(record["run_record_path"])
        run_record = json.loads(run_record_path.read_text()) if run_record_path.is_file() else {}
        run_record.update(
            {
                "journal_audited_at": datetime.now(timezone.utc).isoformat(),
                "journal_files_audited": len(paths),
                "had_internal_failure": failures > 0,
                "internal_failure_events": failures,
                "internal_repair_count": repairs,
                "regenerate_purity": "internal_repair_observed" if repairs > 0 else ("pure_first_attempt" if paths[0].is_file() else "unknown"),
                "forbidden_debug_skill_calls": forbidden,
            }
        )
        run_record_path.write_text(json.dumps(run_record, indent=2, ensure_ascii=False) + "\n")
        results.append(
            {
                "run_id": record["run_id"],
                "journal_files": len(paths),
                "failures": failures,
                "repairs": repairs,
                "forbidden_debug_skill_calls": forbidden,
            }
        )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": len(results),
        "forbidden_skill_runs": sum(bool(item["forbidden_debug_skill_calls"]) for item in results),
        "internal_repair_runs": sum(item["repairs"] > 0 for item in results),
        "results": results,
    }
    (HERE / "regenerate_journal_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: payload[key] for key in ("runs", "forbidden_skill_runs", "internal_repair_runs")}, indent=2))
    return 2 if payload["forbidden_skill_runs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
