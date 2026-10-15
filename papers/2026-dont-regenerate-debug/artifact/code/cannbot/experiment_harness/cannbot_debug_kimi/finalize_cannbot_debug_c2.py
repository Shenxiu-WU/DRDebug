#!/usr/bin/env python3
"""Wait for the C2 CANNBot Debug arm, validate it, and aggregate evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TERMINAL = {"completed", "failed"}
NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".asc"}
GENERATION_MARKERS = {
    "ops-direct-invoke",
    "ascendc-direct-invoke-template",
    "torch-ascendc-op-extension",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        cmdline = (
            Path("/proc") / str(pid) / "cmdline"
        ).read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return "run_regenerate.py" in cmdline and "--arm cannbot-debug" in cmdline


def generation_state(manifest: Path, runner_pid: int) -> dict[str, Any]:
    data = load(manifest)
    rows = data.get("runs", []) if isinstance(data, dict) else []
    counts: dict[str, int] = {}
    incomplete = []
    for row in rows:
        record = load(Path(row["run_record_path"]))
        status = str(record.get("status") if isinstance(record, dict) else "missing")
        counts[status] = counts.get(status, 0) + 1
        if status not in TERMINAL:
            incomplete.append(row["run_id"])
    return {
        "updated_at": now(),
        "runner_alive": pid_alive(runner_pid),
        "expected": len(rows),
        "terminal": len(rows) - len(incomplete),
        "counts": counts,
        "incomplete": incomplete,
    }


def source_hashes(root: Path) -> dict[str, str]:
    import hashlib

    hashes = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in NATIVE_SUFFIXES:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[str(path.relative_to(root))] = digest
    return hashes


def journal_skill_markers(journal: Path) -> set[str]:
    markers: set[str] = set()
    if not journal.is_file():
        return markers
    for line in journal.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            serialized = json.dumps(block, ensure_ascii=False)
            for marker in GENERATION_MARKERS | {"ascendc-precision-debug"}:
                if marker in serialized:
                    markers.add(marker)
    return markers


def audit_lineage(manifest: Path, output: Path) -> dict[str, Any]:
    data = load(manifest)
    rows = data.get("runs", []) if isinstance(data, dict) else []
    audits = []
    for row in rows:
        frozen = Path(row["frozen_input_path"])
        task = Path(row["task_dir"])
        before = source_hashes(frozen)
        after = source_hashes(task)
        deleted = sorted(set(before) - set(after))
        added = sorted(set(after) - set(before))
        modified = sorted(
            path for path in set(before) & set(after) if before[path] != after[path]
        )
        markers = sorted(journal_skill_markers(Path(row["journal_main_path"])))
        deletion_ratio = len(deleted) / max(1, len(before))
        possible_violation = bool(
            deletion_ratio >= 0.8
            or any(marker in GENERATION_MARKERS for marker in markers)
        )
        audits.append(
            {
                "run_id": row["run_id"],
                "operator": row["operator"],
                "native_sources_before": len(before),
                "native_sources_after": len(after),
                "deleted": deleted,
                "added": added,
                "modified": modified,
                "deletion_ratio": round(deletion_ratio, 6),
                "observed_skill_markers": markers,
                "precision_debug_skill_observed": "ascendc-precision-debug" in markers,
                "possible_regeneration_violation": possible_violation,
                "manual_review_required": possible_violation,
            }
        )
    payload = {
        "generated_at": now(),
        "requested": len(audits),
        "possible_regeneration_violations": sum(
            row["possible_regeneration_violation"] for row in audits
        ),
        "missing_precision_debug_skill": sum(
            not row["precision_debug_skill_observed"] for row in audits
        ),
        "rows": audits,
    }
    atomic_json(output, payload)
    return payload


def ensure_validation_container(
    *, name: str, root: Path, repo_root: Path, image: str
) -> None:
    subprocess.run(
        ["docker", "rm", "--force", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    command = [
        "docker",
        "run",
        "--detach",
        "--rm",
        "--name",
        name,
        "--privileged",
        "--security-opt",
        "label=disable",
        "--shm-size",
        "64m",
        "--mount",
        f"type=bind,src={repo_root},dst={repo_root},readonly",
        "--mount",
        f"type=bind,src={root},dst={root}",
        "--mount",
        "type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver",
        "--mount",
        "type=bind,src=/usr/local/Ascend/firmware,dst=/usr/local/Ascend/firmware",
        "--mount",
        "type=bind,src=/usr/local/bin/npu-smi,dst=/usr/local/bin/npu-smi",
        image,
        "sleep",
        "infinity",
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def stop_validation_container(name: str) -> None:
    subprocess.run(
        ["docker", "rm", "--force", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def busy_npus(targets: set[str]) -> dict[str, list[int]]:
    proc = subprocess.run(
        ["npu-smi", "info"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    busy: dict[str, list[int]] = {}
    for line in proc.stdout.splitlines():
        match = re.match(r"^\|\s*(\d+)\s+\d+\s+\|\s*(\d+)\s+\|", line)
        if not match:
            continue
        npu, pid = match.groups()
        if npu in targets:
            busy.setdefault(npu, []).append(int(pid))
    return busy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("/workspace/cannbot_debug_experiment"),
    )
    parser.add_argument("--runner-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--validation-npus", default="0,1,2,3,4")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(
            "/workspace/regenerate_experiment/frozen_debug_engine"
        ),
    )
    parser.add_argument("--container", default="cannbot-runtimebot_c2_validation")
    parser.add_argument("--container-image", default="cannbot-regenerate-runtime:release")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.experiment_root.resolve()
    harness = root / "experiment_harness"
    manifest = harness / "run_manifest.json"
    status_path = harness / "c2_finalizer_status.json"
    state = generation_state(manifest, args.runner_pid)
    while state["runner_alive"]:
        atomic_json(status_path, {"phase": "waiting_for_generation", **state})
        time.sleep(max(10, args.poll_seconds))
        state = generation_state(manifest, args.runner_pid)
    if state["terminal"] != state["expected"]:
        atomic_json(status_path, {"phase": "generation_incomplete", **state})
        return 3

    lineage = audit_lineage(manifest, harness / "source_lineage_audit.json")
    validation_targets = {
        value.strip() for value in args.validation_npus.split(",") if value.strip()
    }
    occupied = busy_npus(validation_targets)
    while occupied:
        atomic_json(
            status_path,
            {
                "phase": "waiting_for_validation_npus",
                "updated_at": now(),
                "busy_npus": occupied,
                "generation": state,
            },
        )
        time.sleep(max(10, args.poll_seconds))
        occupied = busy_npus(validation_targets)
    ensure_validation_container(
        name=args.container,
        root=root,
        repo_root=args.repo_root,
        image=args.container_image,
    )
    stages = []
    try:
        commands = [
            (
                "validate",
                [
                    "python3",
                    str(harness / "validate_manifest.py"),
                    "--manifest",
                    str(manifest),
                    "--arm",
                    "cannbot-debug",
                    "--repo-root",
                    str(args.repo_root),
                    "--container",
                    args.container,
                    "--npus",
                    args.validation_npus,
                    "--concurrency",
                    "5",
                ],
            ),
            (
                "aggregate",
                [
                    "python3",
                    str(harness / "aggregate_tokens_cost.py"),
                    "--manifest",
                    str(manifest),
                    "--pricing",
                    str(harness / "pricing_table.json"),
                    "--strict",
                ],
            ),
        ]
        for name, command in commands:
            started = now()
            proc = subprocess.run(command, cwd=root)
            stage = {
                "stage": name,
                "started_at": started,
                "ended_at": now(),
                "returncode": proc.returncode,
            }
            stages.append(stage)
            if proc.returncode != 0:
                raise RuntimeError(f"{name} failed with rc={proc.returncode}")
    finally:
        stop_validation_container(args.container)

    validation = load(harness / "validation_run_summary.json") or {}
    completion = {
        "phase": "complete",
        "completed_at": now(),
        "generation": state,
        "lineage": {
            "possible_regeneration_violations": lineage[
                "possible_regeneration_violations"
            ],
            "missing_precision_debug_skill": lineage[
                "missing_precision_debug_skill"
            ],
        },
        "validation_requested": validation.get("requested"),
        "validation_errors": validation.get("errors"),
        "stages": stages,
    }
    atomic_json(status_path, completion)
    atomic_json(harness / "c2_completion.json", completion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
