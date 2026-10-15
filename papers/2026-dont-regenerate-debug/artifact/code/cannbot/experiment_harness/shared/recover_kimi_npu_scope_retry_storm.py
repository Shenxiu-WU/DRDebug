#!/usr/bin/env python3
"""Recover the Kimi C2 run after the NPU-scope isolation retry storm."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/cannbot_debug_experiment")
HARNESS = ROOT / "experiment_harness"
RESTART_PATH = HARNESS / "cannbot_debug_kimi_isolated_restart.json"
CHECKPOINT_ROOT = (
    HARNESS
    / "control_repairs"
    / "dynamic_npu_candidate_pool_20260727T180816+0800"
    / "checkpoints"
)
REPAIR_ROOT = (
    HARNESS
    / "control_repairs"
    / "npu_scope_retry_storm_20260727T181251+0800"
)
HYENA_LIVE = (
    ROOT
    / "runs"
    / "level2_023_HyenaFftSizePaddingRfft"
    / "cannbot-debug"
    / "trial1"
)
HYENA_CHECKPOINT = (
    CHECKPOINT_ROOT
    / "cannbot-debug-kimi-level2-023-hyenafftsizepaddingrfft-t1"
    / "workspace"
)
ACTIVE_RUNS = (
    "level2_006_MoeFinalizeRouting",
    "level2_011_DequantSwigluQuant",
    "level2_023_HyenaFftSizePaddingRfft",
    "level2_025_MaskedSoftmaxWithAttentionDropoutBackward",
    "level7_004_NormRopeConcat",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, data: Any, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
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


def restore_tree_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        dirs_exist_ok=True,
    )


def make_trial_writable(root: Path, uid: int, gid: int) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            continue
        os.chown(path, uid, gid)
        current = path.stat().st_mode & 0o777
        path.chmod(current | (0o700 if path.is_dir() else 0o600))


def mark_interrupted(record_path: Path, *, retry_count: int) -> dict[str, Any]:
    record = load(record_path)
    history = list(record.get("interruption_history") or [])
    history.append(
        {
            "status": "provider_interrupted",
            "error_type": "harness_control_interrupted_npu_scope_retry_storm",
            "error_scope": "harness_control",
            "attempt_index": record.get("attempt_index"),
            "recorded_at": now(),
            "count_as_model_failure": False,
            "count_as_valid_internal_attempt": False,
            "same_session_same_workspace": True,
        }
    )
    invalid = list(record.get("invalid_control_attempts") or [])
    invalid.append(
        {
            "attempt_index": record.get("attempt_index"),
            "interrupted_at": now(),
            "reason": "npu_scope_retry_storm_control_recovery",
            "count_as_model_failure": False,
            "count_as_valid_internal_attempt": False,
        }
    )
    record.update(
        {
            "status": "provider_interrupted",
            "attempt_ended_at": now(),
            "error_type": "harness_control_interrupted_npu_scope_retry_storm",
            "error_scope": "harness_control",
            "include_in_success_denominator": False,
            "resume_available": Path(record["journal_main_path"]).is_file(),
            "resume_policy": "same_session_same_workspace_after_harness_repair",
            "same_session_same_workspace": True,
            "invalid_attempt_excluded": True,
            "invalid_control_attempts": invalid,
            "interruption_history": history,
            "trial_container_removed": True,
            "count_as_model_failure": False,
            "count_as_valid_internal_attempt": False,
            "retry_storm_invalid_retries_excluded": retry_count,
        }
    )
    record.pop("ended_at", None)
    atomic_json(record_path, record)
    return record


def set_option(command: list[str], option: str, value: str) -> None:
    index = command.index(option)
    command[index + 1] = value


def import_runner_module() -> Any:
    path = HARNESS / "run_regenerate.py"
    spec = importlib.util.spec_from_file_location("kimi_run_regenerate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import patched Kimi runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    restart = load(RESTART_PATH)
    if restart.get("generation_service_state") != (
        "safe_halted_after_kimi_npu_scope_retry_storm"
    ):
        raise RuntimeError("Kimi experiment is not in the expected safe halt")
    if restart.get("runner_pid") or restart.get("finalizer_pid"):
        raise RuntimeError("Kimi PID fields are not empty")
    if subprocess.run(
        [
            "bash",
            "-lc",
            "docker ps --format '{{.Names}}' | "
            "grep '^cannbot-regen-cannbot-debug-kimi'",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0:
        raise RuntimeError("Kimi containers still exist")

    bad_live = REPAIR_ROOT / "hyena_live_after_retry_storm"
    if bad_live.exists():
        partial = REPAIR_ROOT / "hyena_partial_recovery_failed_copy"
        if partial.exists():
            raise RuntimeError(f"partial recovery evidence already exists: {partial}")
        if HYENA_LIVE.exists():
            HYENA_LIVE.rename(partial)
    else:
        HYENA_LIVE.rename(bad_live)
    HYENA_LIVE.mkdir(parents=True, mode=0o700)

    # The retry loop only clears these runner-owned paths.  Keep the live task,
    # work_tmp and debug_summary from the preserved bad tree, then restore the
    # deleted conversation/deliverable state from the pre-migration checkpoint.
    for name in ("task", "work_tmp", "debug_summary.md", "spec", "input_evidence"):
        source = bad_live / name
        if not source.exists() and not source.is_symlink():
            continue
        target = HYENA_LIVE / name
        if source.is_dir() and not source.is_symlink():
            shutil.copytree(source, target, symlinks=True)
        else:
            shutil.copy2(source, target, follow_symlinks=False)
    for name in (
        ".home",
        ".claude",
        "evidence",
        "validation",
        "candidate_adapter",
        "candidate_task",
        "operators",
        "npu_watch",
    ):
        source = HYENA_CHECKPOINT / name
        if not source.exists() and not source.is_symlink():
            continue
        target = HYENA_LIVE / name
        if source.is_dir() and not source.is_symlink():
            restore_tree_contents(source, target)
        else:
            shutil.copy2(source, target, follow_symlinks=False)
    shutil.copy2(HYENA_CHECKPOINT / "run_record.json", HYENA_LIVE / "run_record.json")

    hyena_record = load(HYENA_LIVE / "run_record.json")
    uid = int(hyena_record["container_uid"])
    gid = int(hyena_record["container_gid"])
    os.chown(HYENA_LIVE, uid, gid)
    HYENA_LIVE.chmod(0o700)
    for name in (
        "task",
        "work_tmp",
        "debug_summary.md",
        ".home",
        ".claude",
        "evidence",
        "validation",
        "candidate_adapter",
        "candidate_task",
        "operators",
        "npu_watch",
    ):
        path = HYENA_LIVE / name
        if path.exists():
            make_trial_writable(path, uid, gid)

    # Verify the exact symlink case inside the real ephemeral trial container
    # before any provider credential is restored or model request is possible.
    runner_module = import_runner_module()
    home = HYENA_LIVE / ".home"
    plugin = Path(hyena_record["plugin_runtime"])
    devkit = Path(hyena_record["devkit_runtime"])
    container = runner_module.start_trial_container(
        record=hyena_record,
        cwd=HYENA_LIVE,
        home=home,
        plugin=plugin,
        devkit=devkit,
        npu="5",
        image=restart.get("container_image") or "cannbot-regenerate-runtime:release",
    )
    try:
        preflight = runner_module.verify_trial_container_isolation(
            container=container,
            cwd=HYENA_LIVE,
            experiment_root=ROOT,
            plugin=plugin,
            devkit=devkit,
            uid=uid,
            gid=gid,
        )
    finally:
        runner_module.stop_trial_container(container)
    if not preflight.get("passed"):
        raise RuntimeError(f"patched preflight did not pass: {preflight}")

    retry_count = int(
        (restart.get("retry_storm_halt") or {}).get("retry_count")
        or restart.get("retry_storm_invalid_retries_excluded")
        or 101
    )
    records: list[dict[str, Any]] = []
    for slug in ACTIVE_RUNS:
        record_path = (
            ROOT / "runs" / slug / "cannbot-debug" / "trial1" / "run_record.json"
        )
        records.append(mark_interrupted(record_path, retry_count=retry_count))
    if not all(record.get("resume_available") for record in records):
        raise RuntimeError("one or more preserved sessions lack their main journal")

    key_config = Path(restart["key_config"])
    held_config = Path(restart["key_config_pause_hold"])
    if not held_config.is_file():
        raise RuntimeError("held Kimi key configuration is missing")
    shutil.copy2(held_config, key_config)
    key_config.chmod(0o600)

    runner_command = list(restart["runner_command"])
    set_option(runner_command, "--npus", "0,1,2,3,4,5")
    set_option(runner_command, "--concurrency", "5")
    runner_log = Path(restart["runner_log"])
    with runner_log.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "resume_after_npu_scope_isolation_repair",
                    "at": now(),
                    "configured_npus": [0, 1, 2, 3, 4, 5],
                    "invalid_control_retries_excluded": retry_count,
                    "same_session_same_workspace": True,
                }
            )
            + "\n"
        )
        handle.flush()
        runner = subprocess.Popen(
            runner_command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(3)
    if runner.poll() is not None:
        raise RuntimeError(f"replacement runner exited rc={runner.returncode}")

    finalizer_command = list(restart["finalizer_command"])
    set_option(finalizer_command, "--runner-pid", str(runner.pid))
    set_option(finalizer_command, "--validation-npus", "0,1,2,3,4,5")
    finalizer_log = Path(restart["finalizer_log"])
    with finalizer_log.open("a") as handle:
        finalizer = subprocess.Popen(
            finalizer_command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(3)
    if finalizer.poll() is not None:
        raise RuntimeError(f"replacement finalizer exited rc={finalizer.returncode}")

    restart.update(
        {
            "updated_at": now(),
            "generation_service_state": (
                "running_after_kimi_npu_0_5_isolation_repair"
            ),
            "generation_resume_requires_user_confirmation": False,
            "runner_pid": runner.pid,
            "finalizer_pid": finalizer.pid,
            "runner_command": runner_command,
            "finalizer_command": finalizer_command,
            "repair_required_before_resume": False,
            "npu_scope": {
                "configured_npus": [0, 1, 2, 3, 4, 5],
                "dynamic_free_card_selection": True,
                "forbidden_for_kimi": [6, 7],
            },
            "isolation_symlink_fix": {
                "applied_at": now(),
                "lexical_workspace_membership": True,
                "isolation_failure_clean_retry_disabled": True,
                "real_hyena_preflight": preflight,
                "invalid_control_retries_excluded": retry_count,
                "evidence_root": str(REPAIR_ROOT),
            },
        }
    )
    atomic_json(RESTART_PATH, restart)
    atomic_json(
        REPAIR_ROOT / "recovery_result.json",
        {
            "recovered_at": now(),
            "runner_pid": runner.pid,
            "finalizer_pid": finalizer.pid,
            "resumed_run_ids": [record["run_id"] for record in records],
            "same_session_same_workspace": True,
            "configured_npus": [0, 1, 2, 3, 4, 5],
            "hyena_preflight": preflight,
            "invalid_control_retries_excluded": retry_count,
            "qwen_modified": False,
        },
        mode=0o400,
    )
    print(
        json.dumps(
            {
                "runner_pid": runner.pid,
                "finalizer_pid": finalizer.pid,
                "resumed_runs": len(records),
                "configured_npus": [0, 1, 2, 3, 4, 5],
                "hyena_preflight_passed": preflight.get("passed"),
                "invalid_control_retries_excluded": retry_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
