#!/usr/bin/env python3
"""Finalize MoeInitRouting trial3 as timeout, validate trial3 gaps, then resume."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import trial3_incremental_validation_gate as gate


RUN_ID = "regen:level2/005_MoeInitRouting:t3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("/workspace/regenerate_experiment"),
    )
    parser.add_argument("--validation-npu", default="6")
    parser.add_argument("--validation-timeout", type=int, default=1200)
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


def main() -> int:
    args = parse_args()
    root = args.experiment_root.resolve()
    harness = root / "experiment_harness"
    manifest_path = harness / "run_manifest.json"
    restart_path = harness / "regenerate_isolated_restart.json"
    gate_path = harness / "trial3_incremental_validation_gate.json"
    lock_path = harness / ".trial3_incremental_validation_gate.lock"
    validation_log = root / "orchestration_logs" / "trial3_incremental_validation.log"

    manifest = gate.load_json(manifest_path)
    restart = gate.load_json(restart_path)
    if not isinstance(manifest, dict) or not isinstance(restart, dict):
        raise SystemExit("manifest or restart JSON is invalid")
    trial3 = gate.regenerate_records(manifest, trial=3)
    by_id = {record["run_id"]: record for record in trial3}
    target = by_id[RUN_ID]
    record_path = Path(target["run_record_path"])
    state = gate.run_state(target)

    key_path = Path(restart["key_config"])
    hold_path = key_path.with_name(key_path.name + ".trial3_validation_hold")
    if not hold_path.is_file():
        raise SystemExit("provider hold is missing; refusing to validate without drain")
    original = hold_path.read_bytes()
    validation_rc: int | None = None
    restore_status = "not_attempted"

    lock_path.touch(mode=0o600, exist_ok=True)
    with lock_path.open("r+") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("previous trial3 controller still owns the gate lock")

        phase: dict[str, Any] = {
            "phase": "finalizing_trial3_timeout",
            "controller_pid": os.getpid(),
            "started_at": gate.now(),
            "terminal_timeout_run_id": RUN_ID,
            "validation_npus": [args.validation_npu],
            "key_pool_drained": True,
            "preserve_candidate_workspace": True,
        }
        update_phase(gate_path, restart_path, phase, phase["phase"])

        try:
            now = gate.now()
            state.update(
                {
                    "status": "failed",
                    "result_subtype": "timeout",
                    "timed_out": True,
                    "terminal_timeout": True,
                    "user_directed_terminal_timeout": True,
                    "provider_aborted": False,
                    "provider_abort_reason": None,
                    "error_type": "attempt_timeout",
                    "error_scope": "trial",
                    "include_in_success_denominator": True,
                    "resume_available": False,
                    "resume_policy": "terminal_timeout_no_resume",
                    "attempt_ended_at": now,
                    "ended_at": now,
                }
            )
            gate.atomic_json(record_path, state)

            missing = gate.missing_trial3_validation(trial3)
            phase.update(
                {
                    "phase": "validating_missing_trial3",
                    "validation_started_at": gate.now(),
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
                    args.validation_npu,
                    "--concurrency",
                    "1",
                    "--timeout",
                    str(args.validation_timeout),
                ]
                for record in missing:
                    command.extend(["--only", record["run_id"]])
                with validation_log.open("a", encoding="utf-8") as log:
                    completed = subprocess.run(
                        command, cwd=root, stdout=log, stderr=subprocess.STDOUT
                    )
                validation_rc = completed.returncode
                generated = harness / "validation_run_summary.json"
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                summary_path = (
                    harness / f"validation_run_summary_incremental_trial3_{stamp}.json"
                )
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
                        record["run_id"]
                        for record in gate.missing_trial3_validation(trial3)
                    ],
                }
            )
            update_phase(gate_path, restart_path, phase, phase["phase"])
        except Exception as exc:
            validation_rc = 1
            phase.update(
                {
                    "phase": "terminal_timeout_validation_error",
                    "failed_at": gate.now(),
                    "error": repr(exc),
                }
            )
            update_phase(gate_path, restart_path, phase, phase["phase"])
        finally:
            restore_status = gate.restore_provider_pool(key_path, hold_path, original)
            phase.update(
                {
                    "phase": "completed" if validation_rc == 0 else "completed_with_errors",
                    "completed_at": gate.now(),
                    "provider_pool_restore_status": restore_status,
                    "generation_resumed": restore_status
                    in {"restored_original_pool", "kept_operator_updated_pool"},
                }
            )
            update_phase(
                gate_path,
                restart_path,
                phase,
                "running_after_trial3_incremental_validation"
                if validation_rc == 0
                else "running_after_trial3_validation_error",
            )

    return 0 if validation_rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
