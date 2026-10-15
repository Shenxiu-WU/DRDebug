#!/usr/bin/env python3
"""Recover a timeout-misclassified trial3, validate trial3 gaps, then resume.

This is a one-shot operational controller for a live rolling runner that was
started before attempt timeouts entered the resumable infrastructure path.  It
drains only future provider refill, preserves every active trial2 container,
resumes the timed-out trial3 from its existing session on the first NPU that
becomes free, runs missing trial3 validation, and restores the provider pool.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import trial3_incremental_validation_gate as gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("/workspace/regenerate_experiment"),
    )
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--validation-timeout", type=int, default=1200)
    parser.add_argument(
        "--recovery-npu",
        action="append",
        dest="recovery_npus",
        default=[],
        help="Additional NPU to prefer for timeout recovery; may be repeated.",
    )
    return parser.parse_args()


def update_phase(
    gate_path: Path,
    restart_path: Path,
    phase: dict[str, Any],
    service_state: str,
) -> None:
    gate.atomic_json(gate_path, phase, mode=0o644)
    gate.update_json(
        restart_path,
        {
            "generation_service_state": service_state,
            "trial3_incremental_validation_gate": phase,
        },
    )
    print(json.dumps(phase, ensure_ascii=False), flush=True)


def timeout_interrupted(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    interrupted = []
    for record in records:
        state = gate.run_state(record)
        is_old_timeout_failure = (
            state.get("status") == "failed" and state.get("timed_out") is True
        )
        is_reclassified_timeout = (
            state.get("status") == "provider_interrupted"
            and state.get("provider_abort_reason") == "attempt_timeout"
            and state.get("resume_available") is True
        )
        if is_old_timeout_failure or is_reclassified_timeout:
            interrupted.append(record)
    return interrupted


def mark_timeout_resumable(record: dict[str, Any]) -> None:
    path = Path(record["run_record_path"])
    state = gate.run_state(record)
    journal = Path(record["journal_main_path"])
    if not journal.is_file():
        raise RuntimeError(f"cannot resume timeout without journal: {record['run_id']}")
    if (
        state.get("status") == "provider_interrupted"
        and state.get("provider_abort_reason") == "attempt_timeout"
        and state.get("resume_available") is True
    ):
        return
    if state.get("status") != "failed" or state.get("timed_out") is not True:
        raise RuntimeError(f"run is no longer a timeout failure: {record['run_id']}")
    state.update(
        {
            "status": "provider_interrupted",
            "provider_aborted": True,
            "provider_abort_reason": "attempt_timeout",
            "error_type": "attempt_timeout",
            "error_scope": "infrastructure",
            "include_in_success_denominator": False,
            "resume_available": True,
            "resume_policy": "same_session_same_workspace_attempt_timeout",
            "timeout_reclassified_at": gate.now(),
        }
    )
    state.pop("ended_at", None)
    gate.atomic_json(path, state)


def wait_for_free_npus(
    records: list[dict[str, Any]], npus: list[str], poll_seconds: float
) -> list[str]:
    while True:
        active = gate.active_generation_npus(records)
        free = [npu for npu in npus if npu not in active]
        if free:
            return free
        time.sleep(max(0.25, poll_seconds))


def dedicated_resume_command(
    harness: Path,
    hold_path: Path,
    record: dict[str, Any],
    npu: str,
) -> list[str]:
    return [
        "python3",
        str(harness / "run_regenerate.py"),
        "--model",
        "qwen3.8-max-preview",
        "--key-config",
        str(hold_path),
        "--npus",
        npu,
        "--concurrency",
        "1",
        "--max-turns",
        "240",
        "--timeout",
        "21600",
        "--preferred-provider",
        "qwen3.8-max-preview-primary",
        "--min-provider-remaining",
        "1",
        "--slots-per-key",
        "1",
        "--provider-retries",
        "0",
        "--deliverable-retries",
        "3",
        "--repair-invalid-completed",
        "--provider-abort-cooldown-seconds",
        "300",
        "--provider-quota-cooldown-seconds",
        "21600",
        "--quota-timeout",
        "20",
        "--provider-refresh-seconds",
        "60",
        "--quota-retry-wait",
        "60",
        "--provider-env-name",
        "regenerate_qwen3_8_trial3_timeout_recovery",
        "--queue-order",
        "reverse_trial_then_operator",
        "--only",
        record["run_id"],
    ]


def main() -> int:
    args = parse_args()
    root = args.experiment_root.resolve()
    harness = root / "experiment_harness"
    manifest_path = harness / "run_manifest.json"
    restart_path = harness / "regenerate_isolated_restart.json"
    gate_path = harness / "trial3_incremental_validation_gate.json"
    lock_path = harness / ".trial3_incremental_validation_gate.lock"
    controller_log = root / "orchestration_logs" / "trial3_incremental_validation.log"
    recovery_log = root / "orchestration_logs" / "trial3_timeout_recovery_runner.log"

    manifest = gate.load_json(manifest_path)
    restart = gate.load_json(restart_path)
    if not isinstance(manifest, dict) or not isinstance(restart, dict):
        raise SystemExit("manifest or restart JSON is invalid")
    all_records = gate.regenerate_records(manifest)
    trial3 = gate.regenerate_records(manifest, trial=3)
    interrupted = timeout_interrupted(trial3)
    if not interrupted:
        raise SystemExit("no timeout-misclassified trial3 record found")

    key_path = Path(restart["key_config"])
    hold_path = key_path.with_name(key_path.name + ".trial3_validation_hold")
    configured_npus = gate.configured_npus(restart)
    npus = list(dict.fromkeys([*args.recovery_npus, *configured_npus]))
    original = b""
    restore_status = "not_engaged"
    validation_rc: int | None = None

    lock_path.touch(mode=0o600, exist_ok=True)
    with lock_path.open("r+") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("trial3 gate controller is still active")

        phase: dict[str, Any] = {
            "phase": "draining_for_trial3_timeout_recovery",
            "controller_pid": os.getpid(),
            "started_at": gate.now(),
            "runner_pid": restart.get("runner_pid"),
            "finalizer_pid": restart.get("finalizer_pid"),
            "timeout_run_ids": [record["run_id"] for record in interrupted],
            "interrupt_active_generation": False,
            "preserve_same_session_workspace": True,
        }
        try:
            current_key_config = gate.load_json(key_path)
            if (
                hold_path.is_file()
                and isinstance(current_key_config, dict)
                and current_key_config.get("transition")
                == "trial3_incremental_validation_drain"
                and current_key_config.get("keys") == []
            ):
                original = hold_path.read_bytes()
                phase["reused_existing_drain"] = True
            else:
                original = gate.engage_provider_drain(key_path, hold_path)
                phase["reused_existing_drain"] = False
            phase["key_pool_drained"] = True
            update_phase(gate_path, restart_path, phase, phase["phase"])

            for record in interrupted:
                mark_timeout_resumable(record)

            free = wait_for_free_npus(all_records, npus, args.poll_seconds)
            recovery_npu = free[0]
            phase.update(
                {
                    "phase": "resuming_trial3_timeout",
                    "recovery_npu": recovery_npu,
                    "recovery_started_at": gate.now(),
                }
            )
            update_phase(gate_path, restart_path, phase, phase["phase"])
            with recovery_log.open("a", encoding="utf-8") as log:
                completed = subprocess.run(
                    dedicated_resume_command(harness, hold_path, interrupted[0], recovery_npu),
                    cwd=root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            recovered_state = gate.run_state(interrupted[0])
            phase.update(
                {
                    "recovery_finished_at": gate.now(),
                    "recovery_runner_returncode": completed.returncode,
                    "recovered_status": recovered_state.get("status"),
                    "recovered_attempt_index": recovered_state.get("attempt_index"),
                    "recovered_candidate_materialized": recovered_state.get("candidate_materialized"),
                }
            )
            if recovered_state.get("status") != "completed":
                raise RuntimeError(
                    f"trial3 timeout recovery did not complete: {recovered_state.get('status')}"
                )

            free = wait_for_free_npus(all_records, npus, args.poll_seconds)
            missing = gate.missing_trial3_validation(trial3)
            phase.update(
                {
                    "phase": "validating_missing_trial3",
                    "validation_started_at": gate.now(),
                    "validation_npus": free,
                    "missing_run_ids": [record["run_id"] for record in missing],
                    "missing_count": len(missing),
                    "validation_before": gate.validation_snapshot(trial3),
                }
            )
            update_phase(gate_path, restart_path, phase, phase["phase"])

            summary_path: Path | None = None
            if missing:
                command = [
                    "python3",
                    str(harness / "validate_manifest.py"),
                    "--manifest",
                    str(manifest_path),
                    "--arm",
                    "regenerate",
                    "--npus",
                    ",".join(free),
                    "--concurrency",
                    str(len(free)),
                    "--timeout",
                    str(args.validation_timeout),
                ]
                for record in missing:
                    command.extend(["--only", record["run_id"]])
                with controller_log.open("a", encoding="utf-8") as log:
                    validated = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                validation_rc = validated.returncode
                generated = harness / "validation_run_summary.json"
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                summary_path = harness / f"validation_run_summary_incremental_trial3_{stamp}.json"
                if generated.is_file():
                    shutil.copy2(generated, summary_path)
            else:
                validation_rc = 0

            phase.update(
                {
                    "phase": "validation_finished",
                    "validation_finished_at": gate.now(),
                    "validation_returncode": validation_rc,
                    "validation_summary": str(summary_path) if summary_path else None,
                    "validation_after": gate.validation_snapshot(trial3),
                    "still_missing_run_ids": [
                        record["run_id"] for record in gate.missing_trial3_validation(trial3)
                    ],
                }
            )
            update_phase(gate_path, restart_path, phase, phase["phase"])
        except Exception as exc:
            phase.update(
                {
                    "phase": "recovery_or_validation_error",
                    "failed_at": gate.now(),
                    "error": repr(exc),
                }
            )
            validation_rc = 1
            update_phase(gate_path, restart_path, phase, phase["phase"])
        finally:
            if original:
                restore_status = gate.restore_provider_pool(key_path, hold_path, original)
            phase.update(
                {
                    "phase": "completed" if validation_rc == 0 else "completed_with_errors",
                    "completed_at": gate.now(),
                    "provider_pool_restore_status": restore_status,
                    "generation_resumed": restore_status in {
                        "restored_original_pool",
                        "kept_operator_updated_pool",
                    },
                }
            )
            update_phase(
                gate_path,
                restart_path,
                phase,
                "running_after_trial3_incremental_validation"
                if validation_rc == 0
                else "running_after_trial3_recovery_error",
            )

    return 0 if validation_rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
