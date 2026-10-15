#!/usr/bin/env python3
"""Package CANNBot-Debug candidate sources and validation evidence for review."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
from pathlib import Path


SOURCE_SUFFIXES = {".asc", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".py", ".json", ".txt"}
EXCLUDED_DIRS = {".git", "__pycache__", "build", "build_out", "logs", "node_modules"}


def copy_tree(source: Path, destination: Path) -> int:
    copied = 0
    if not source.exists():
        return copied
    for dirpath, dirnames, filenames in os.walk(source):
        dirnames[:] = [name for name in dirnames if name not in EXCLUDED_DIRS]
        directory = Path(dirpath)
        for filename in filenames:
            path = directory / filename
            if path.suffix.lower() not in SOURCE_SUFFIXES and filename not in {
                "CMakeLists.txt",
                "Makefile",
            }:
                continue
            relative = path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            copied += 1
    return copied


def copy_file(source: Path, destination: Path) -> int:
    if not source.is_file():
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    with tempfile.TemporaryDirectory(prefix="cannbot-debug-audit-") as temporary:
        stage = Path(temporary) / "cannbot_debug_kimi_manual_audit"
        stage.mkdir()
        index = []
        file_count = 0
        for run in manifest.get("runs", []):
            cwd = Path(run["cwd"])
            safe = run["operator"].replace("/", "__")
            target = stage / safe
            file_count += copy_tree(cwd / "candidate_task", target / "candidate_task")
            file_count += copy_tree(cwd / "candidate_adapter", target / "candidate_adapter")
            file_count += copy_tree(cwd / "input_evidence", target / "input_evidence")
            file_count += copy_file(cwd / "run_record.json", target / "run_record.json")
            file_count += copy_file(
                cwd / "validation" / "full_eval.json", target / "validation" / "full_eval.json"
            )
            file_count += copy_file(
                cwd / "validation" / "anticheat.json", target / "validation" / "anticheat.json"
            )
            file_count += copy_file(
                cwd / "validation" / "work" / "_external_eval_manifest.json",
                target / "validation" / "_external_eval_manifest.json",
            )
            index.append(
                {
                    "run_id": run["run_id"],
                    "operator": run["operator"],
                    "trial": run["trial"],
                    "source": str(cwd),
                    "packaged_as": safe,
                }
            )
        (stage / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_manifest": str(args.manifest),
                    "runs": len(index),
                    "files": file_count,
                    "results": index,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.output, "w:gz") as archive:
            archive.add(stage, arcname=stage.name)
    print(json.dumps({"runs": len(index), "files": file_count, "output": str(args.output)}))
    return 0 if len(index) == 27 else 1


if __name__ == "__main__":
    raise SystemExit(main())
