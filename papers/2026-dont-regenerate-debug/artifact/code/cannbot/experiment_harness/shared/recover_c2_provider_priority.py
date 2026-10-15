#!/usr/bin/env python3
"""Recover orphaned C2 workers and enforce a primary/fallback provider pool."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time


def atomic_json(path: pathlib.Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def set_option(command: list[str], flag: str, value: str) -> None:
    if flag in command:
        command[command.index(flag) + 1] = value
    else:
        command.extend([flag, value])


def one_option(command: list[str], flag: str, value: str) -> list[str]:
    result: list[str] = []
    seen = False
    index = 0
    while index < len(command):
        if command[index] == flag:
            if not seen:
                result.extend([flag, value])
                seen = True
            index += 2
            continue
        result.append(command[index])
        index += 1
    if not seen:
        result.extend([flag, value])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=pathlib.Path,
        default=pathlib.Path(
            "/workspace/cannbot_debug_experiment"
        ),
    )
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    harness = root / "experiment_harness"
    sys.path.insert(0, str(harness))
    import run_regenerate

    manifest = json.loads((harness / "run_manifest.json").read_text())
    running: list[tuple[dict, pathlib.Path, dict]] = []
    for row in manifest["runs"]:
        record_path = pathlib.Path(row["run_record_path"])
        record = json.loads(record_path.read_text())
        if record.get("status") == "running":
            running.append((row, record_path, record))

    # The original supervising runner is gone. Quiesce only its exact C2
    # containers before taking immutable, consistent checkpoints.
    for _, _, record in running:
        container = record.get("trial_container_name")
        if container:
            subprocess.run(
                ["docker", "rm", "--force", str(container)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
    time.sleep(2)
    for proc in pathlib.Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmdline = (
                (proc / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
        except OSError:
            continue
        if str(root) in cmdline and (
            "docker exec" in cmdline or cmdline.startswith("timeout ")
        ):
            try:
                os.kill(int(proc.name), signal.SIGTERM)
            except ProcessLookupError:
                pass

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for row, record_path, record in running:
        checkpoint = run_regenerate.snapshot_provider_abort(
            row,
            {
                **record,
                "status": "provider_interrupted",
                "provider_abort_reason": (
                    "controlled_supervisor_reload_interruption"
                ),
                "error_scope": "harness_control",
            },
            1,
            root,
        )
        (pathlib.Path(row["cwd"]) / ".home/.provider.env").unlink(
            missing_ok=True
        )
        updated = dict(record)
        updated.update(
            {
                "status": "provider_interrupted",
                "attempt_ended_at": timestamp,
                "provider_aborted": False,
                "provider_abort_reason": (
                    "controlled_supervisor_reload_interruption"
                ),
                "error_scope": "harness_control",
                "include_in_success_denominator": False,
                "resume_available": pathlib.Path(
                    row["journal_main_path"]
                ).is_file(),
                "resume_policy": (
                    "same_session_same_workspace_after_supervisor_reload"
                ),
                "provider_retry_count": 0,
                "provider_checkpoints": [str(checkpoint)],
                "model_invocation_count": 0,
                "task_turns_consumed": 0,
                "task_active_seconds": 0,
                "trial_container_removed": True,
            }
        )
        updated.pop("ended_at", None)
        atomic_json(record_path, updated)

    provider_base = root / ".provider_env"
    if provider_base.exists():
        shutil.rmtree(provider_base)

    wall = time.time()
    key2_reset = datetime.datetime.fromisoformat(
        "2026-07-27T13:59:00+00:00"
    ).timestamp()
    provider_state = {
        "schema_version": 1,
        "updated_at": timestamp,
        "providers": {
            "provider-slot-a": {
                "reason": "explicit_quota_exhausted",
                "basis": {
                    "policy": "actual_request_quota_error_without_reset",
                    "source": "minimal_actual_generation_probe",
                    "http_status": 429,
                },
                "unavailable_until_epoch": wall + 21600,
                "unavailable_until": datetime.datetime.fromtimestamp(
                    wall + 21600, datetime.timezone.utc
                ).isoformat(),
            },
            "provider-slot-b": {
                "reason": "explicit_quota_exhausted",
                "basis": {
                    "policy": "actual_generation_429_weekly_reset",
                    "source": "actual_generation_and_reprobe",
                    "http_status": 429,
                    "window": "weekly",
                },
                "unavailable_until_epoch": key2_reset,
                "unavailable_until": "2026-07-27T13:59:00+00:00",
            },
        },
    }
    atomic_json(harness / "provider_runtime_state.json", provider_state)

    restart_path = harness / "cannbot_debug_isolated_restart.json"
    restart = json.loads(restart_path.read_text())
    runner = list(restart["runner_command"])
    set_option(runner, "--preferred-provider", "provider-slot-a")
    set_option(
        runner,
        "--provider-env-name",
        "c2_cannbot_debug_qwen3_8_priority_pool",
    )
    runner = one_option(
        runner, "--fallback-provider", "provider-slot-a"
    )
    runner_log = harness / "cannbot_debug_runner.log"
    with runner_log.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "provider_priority_pool_reload",
                    "at": timestamp,
                    "primary": [
                        "provider-slot-a",
                        "provider-slot-b",
                    ],
                    "fallback": ["provider-slot-a"],
                    "resumable_sessions": len(running),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()
        runner_proc = subprocess.Popen(
            runner,
            cwd=root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(3)
    if runner_proc.poll() is not None:
        raise RuntimeError(f"runner exited rc={runner_proc.returncode}")

    finalizer = list(restart["finalizer_command"])
    set_option(finalizer, "--runner-pid", str(runner_proc.pid))
    finalizer_log = harness / "cannbot_debug_finalizer.log"
    with finalizer_log.open("a") as handle:
        finalizer_proc = subprocess.Popen(
            finalizer,
            cwd=root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(2)
    if finalizer_proc.poll() is not None:
        raise RuntimeError(f"finalizer exited rc={finalizer_proc.returncode}")

    restart.update(
        {
            "updated_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "generation_service_state": (
                "running_with_workerb_primary_workerd_fallback"
            ),
            "generation_resume_requires_user_confirmation": False,
            "runner_pid": runner_proc.pid,
            "finalizer_pid": finalizer_proc.pid,
            "runner_command": runner,
            "finalizer_command": finalizer,
            "credential_name": "provider_priority_pool",
            "key_config": str(
                root / ".secrets/qwen3_8_max_preview.json"
            ),
            "provider_priority": {
                "primary": [
                    "provider-slot-a",
                    "provider-slot-b",
                ],
                "fallback": ["provider-slot-a"],
                "fallback_admission": (
                    "only_when_no_primary_provider_slot_available"
                ),
                "key1_latest_probe": (
                    "429_quota_exhausted_unspecified_window"
                ),
                "key2_latest_probe": "429_weekly_quota_exhausted",
                "workerd_latest_probe": "200_available",
            },
            "supervisor_reload": {
                "at": timestamp,
                "interrupted_runs": [
                    row["run_id"] for row, _, _ in running
                ],
                "same_session_same_workspace": True,
                "invalid_invocation_excluded": True,
            },
        }
    )
    atomic_json(restart_path, restart)
    print(
        json.dumps(
            {
                "runner_pid": runner_proc.pid,
                "finalizer_pid": finalizer_proc.pid,
                "resumable_sessions": len(running),
                "state": restart["generation_service_state"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
