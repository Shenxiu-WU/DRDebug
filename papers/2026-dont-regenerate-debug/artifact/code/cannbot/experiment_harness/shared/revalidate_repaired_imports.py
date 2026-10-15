#!/usr/bin/env python3
"""Run zero-LLM corrective validation for rematerialized import failures."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repair-report", required=True, type=Path)
    parser.add_argument("--exclude-repair-report", type=Path)
    parser.add_argument("--validate-script", required=True, type=Path)
    parser.add_argument("--source-pool", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--container", required=True)
    parser.add_argument("--npus", required=True)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--timeout", default=1200, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    repair = json.loads(args.repair_report.read_text())
    excluded: set[str] = set()
    if args.exclude_repair_report:
        excluded = {
            row["run_id"]
            for row in json.loads(args.exclude_repair_report.read_text()).get("results", [])
            if row.get("repair_valid")
        }
    operators = [
        row["operator"]
        for row in repair.get("results", [])
        if row.get("repair_valid") and row.get("run_id") not in excluded
    ]
    if not operators:
        raise SystemExit("repair report is incomplete; refusing corrective validation")

    command = [
        sys.executable,
        str(args.validate_script),
        "--manifest",
        str(args.manifest),
        "--source-pool",
        str(args.source_pool),
        "--repo-root",
        str(args.repo_root),
        "--container",
        args.container,
        "--npus",
        args.npus,
        "--concurrency",
        str(args.concurrency),
        "--timeout",
        str(args.timeout),
        "--arm",
        "cannbot-debug",
        "--force",
    ]
    for operator in operators:
        command.extend(("--only", operator))
    completed = subprocess.run(command, check=False)

    default_summary = args.manifest.parent / "validation_run_summary.json"
    if not default_summary.is_file():
        raise SystemExit("corrective validator did not produce its summary")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(default_summary, args.output)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
