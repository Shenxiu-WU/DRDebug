#!/usr/bin/env python3
"""Fail when a process still references one V5 arm task tree."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def matching_processes(tasks_root: Path) -> list[dict[str, Any]]:
    target = str(tasks_root.resolve())
    skipped = {os.getpid(), os.getppid()}
    matches = []
    for proc in Path("/proc").glob("[0-9]*"):
        pid = int(proc.name)
        if pid in skipped:
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace")
        except OSError:
            cmdline = ""
        try:
            cwd = str((proc / "cwd").resolve())
        except OSError:
            cwd = ""
        if target in cmdline or cwd == target or cwd.startswith(target + "/"):
            matches.append({"pid": pid, "cwd": cwd, "cmdline": cmdline.strip()})
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    matches = matching_processes(args.tasks_root)
    payload = {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "tasks_root": str(args.tasks_root),
        "passed": not matches,
        "matching_processes": matches,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if matches:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
