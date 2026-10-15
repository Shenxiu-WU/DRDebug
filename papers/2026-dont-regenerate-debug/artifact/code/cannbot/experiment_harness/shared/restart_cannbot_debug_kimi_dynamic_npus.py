#!/usr/bin/env python3
"""Safely restart the Kimi C2 runner after enabling all configured NPUs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, data: Any, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    if mode is not None:
        temporary.chmod(mode)
    temporary.replace(path)
    if mode is not None:
        path.chmod(mode)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def stop_process_group(pid: int, wait_seconds: int = 15) -> None:
    if not pid_alive(pid):
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + wait_seconds
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.25)
    if pid_alive(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def immutable_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            continue
        path.chmod(0o500 if path.is_dir() else 0o400)
    root.chmod(0o500)


def snapshot_workspace(cwd: Path, destination: Path) -> None:
    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = set()
        if Path(directory).name == ".home":
            ignored.add(".provider.env")
        ignored.update(name for name in names if name == "__pycache__")
        return ignored

    shutil.copytree(cwd, destination, symlinks=True, ignore=ignore)
    immutable_tree(destination)


def replace_runner_pid(command: list[str], runner_pid: int) -> list[str]:
    updated = list(command)
    try:
        index = updated.index("--runner-pid")
    except ValueError as exc:
        raise RuntimeError("finalizer command lacks --runner-pid") from exc
    updated[index + 1] = str(runner_pid)
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("/workspace/cannbot_debug_experiment"),
    )
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    harness = root / "experiment_harness"
    restart_path = harness / "cannbot_debug_kimi_isolated_restart.json"
    restart = load(restart_path)
    if not str(restart.get("generation_service_state") or "").startswith("running"):
        raise SystemExit("Kimi experiment is not in a running state")

    records = []
    for path in (root / "runs").rglob("run_record.json"):
        record = load(path)
        if record.get("status") == "running":
            records.append((path, record))
    if not records:
        raise SystemExit("no active Kimi workers require controlled restart")

    repair_root = (
        harness
        / "control_repairs"
        / f"dynamic_npu_candidate_pool_{stamp()}"
    )
    checkpoints = repair_root / "checkpoints"
    provider_hold = repair_root / "provider_hold"
    checkpoints.mkdir(parents=True)
    provider_hold.mkdir(parents=True)
    provider_hold.chmod(0o700)

    key_config = Path(restart["key_config"])
    held_config = provider_hold / key_config.name
    shutil.copy2(key_config, held_config)
    held_config.chmod(0o600)
    live_config = load(key_config)
    empty_config = {
        key: value
        for key, value in live_config.items()
        if key != "keys"
    }
    empty_config["keys"] = []
    atomic_json(key_config, empty_config, mode=0o600)
    atomic_json(
        repair_root / "dispatch_barrier.json",
        {
            "created_at": now(),
            "reason": "load patched eight-card candidate queue",
            "runner_pid": restart.get("runner_pid"),
            "finalizer_pid": restart.get("finalizer_pid"),
            "active_runs": [record["run_id"] for _, record in records],
            "default_key_pool_empty": True,
        },
        mode=0o400,
    )

    old_runner_pid = int(restart.get("runner_pid") or 0)
    old_finalizer_pid = int(restart.get("finalizer_pid") or 0)
    stop_process_group(old_finalizer_pid)
    stop_process_group(old_runner_pid)
    for _, record in records:
        container = str(record.get("trial_container_name") or "")
        if container.startswith("cannbot-regen-"):
            subprocess.run(
                ["docker", "rm", "--force", container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        (Path(record["cwd"]) / ".home" / ".provider.env").unlink(
            missing_ok=True
        )

    checkpoint_paths: dict[str, str] = {}
    for record_path, record in records:
        safe = "".join(
            character.lower() if character.isalnum() else "-"
            for character in str(record["run_id"])
        ).strip("-")
        destination = checkpoints / safe
        snapshot_workspace(Path(record["cwd"]), destination / "workspace")
        checkpoint = {
            "created_at": now(),
            "reason": "harness_control_dynamic_npu_candidate_pool_fix",
            "run_id": record["run_id"],
            "operator": record.get("operator"),
            "trial": record.get("trial"),
            "attempt_index": record.get("attempt_index"),
            "npu": record.get("npu"),
            "credential_name": record.get("credential_name"),
            "expected_session_id": record.get("expected_session_id"),
            "journal_main_path": record.get("journal_main_path"),
            "subagent_dir": record.get("subagent_dir"),
            "candidate_task_path": record.get("candidate_task_path"),
            "same_session_same_workspace": True,
            "invalid_attempt_excluded": True,
            "workspace_snapshot": str(destination / "workspace"),
        }
        checkpoint_path = destination / "checkpoint.json"
        atomic_json(checkpoint_path, checkpoint, mode=0o400)
        destination.chmod(0o500)
        checkpoint_paths[record["run_id"]] = str(checkpoint_path)

        updated = dict(record)
        invalid = list(updated.get("invalid_control_attempts") or [])
        invalid.append(
            {
                "attempt_index": record.get("attempt_index"),
                "interrupted_at": now(),
                "reason": "dynamic_npu_candidate_pool_fix",
                "checkpoint": str(checkpoint_path),
                "count_as_model_failure": False,
                "count_as_valid_internal_attempt": False,
            }
        )
        provider_checkpoints = list(updated.get("provider_checkpoints") or [])
        provider_checkpoints.append(str(checkpoint_path))
        updated.update(
            {
                "status": "provider_interrupted",
                "error_type": "harness_control_interrupted_dynamic_npu_pool_fix",
                "error_scope": "harness_control",
                "include_in_success_denominator": False,
                "resume_available": Path(
                    str(record["journal_main_path"])
                ).is_file(),
                "resume_policy": (
                    "same_session_same_workspace_after_dynamic_npu_pool_fix"
                ),
                "same_session_same_workspace": True,
                "invalid_attempt_excluded": True,
                "invalid_control_attempts": invalid,
                "provider_checkpoints": provider_checkpoints,
                "trial_container_removed": True,
                "control_interrupted_at": now(),
            }
        )
        updated.pop("ended_at", None)
        atomic_json(record_path, updated)

    shutil.copy2(held_config, key_config)
    key_config.chmod(0o600)

    runner_log = Path(restart["runner_log"])
    runner_handle = runner_log.open("a")
    runner = subprocess.Popen(
        list(restart["runner_command"]),
        cwd=root,
        stdout=runner_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    runner_handle.close()
    finalizer_command = replace_runner_pid(
        list(restart["finalizer_command"]), runner.pid
    )
    finalizer_log = Path(restart["finalizer_log"])
    finalizer_handle = finalizer_log.open("a")
    finalizer = subprocess.Popen(
        finalizer_command,
        cwd=root,
        stdout=finalizer_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    finalizer_handle.close()

    updated_restart = dict(restart)
    updated_restart.update(
        {
            "updated_at": now(),
            "generation_service_state": (
                "running_after_dynamic_eight_npu_candidate_pool_fix"
            ),
            "runner_pid": runner.pid,
            "finalizer_pid": finalizer.pid,
            "finalizer_command": finalizer_command,
            "dynamic_npu_pool_fix": {
                "applied_at": now(),
                "old_runner_pid": old_runner_pid,
                "old_finalizer_pid": old_finalizer_pid,
                "configured_npus": list(range(8)),
                "candidate_queue_uses_all_configured_npus": True,
                "concurrency": 5,
                "invalid_attempts_excluded": len(records),
                "checkpoint_root": str(checkpoints),
                "checkpoint_paths": checkpoint_paths,
                "provider_hold": str(held_config),
                "key_pool_restored": True,
            },
        }
    )
    atomic_json(restart_path, updated_restart)
    time.sleep(5)
    if not pid_alive(runner.pid) or not pid_alive(finalizer.pid):
        raise SystemExit("replacement runner/finalizer failed launch verification")
    print(
        json.dumps(
            {
                "runner_pid": runner.pid,
                "finalizer_pid": finalizer.pid,
                "interrupted_runs": len(records),
                "checkpoint_root": str(checkpoints),
                "all_eight_npus_in_candidate_queue": True,
                "concurrency": 5,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
