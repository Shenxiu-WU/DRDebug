#!/usr/bin/env python3
"""Start the independent Kimi K3 CANNBot Debug C2 runner and finalizer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXCLUDED_PROVIDER = "providera-kimi-key"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("/workspace/cannbot_debug_experiment"),
    )
    parser.add_argument(
        "--source-key-config",
        type=Path,
        default=Path(
            "/workspace/regenerate_experiment/"
            ".secrets/provider_config.json"
        ),
    )
    parser.add_argument(
        "--qwen-reference",
        type=Path,
        default=Path("/workspace/cannbot_debug_experiment"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.experiment_root.resolve()
    harness = root / "experiment_harness"
    manifest_path = harness / "run_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing prepared manifest: {manifest_path}")
    manifest = load(manifest_path)
    rows = manifest.get("runs", []) if isinstance(manifest, dict) else []
    if len(rows) != 27 or any(row.get("status") != "prepared" for row in rows):
        raise SystemExit("Kimi experiment must start from exactly 27 prepared rows")

    qwen_restart_path = (
        args.qwen_reference
        / "experiment_harness"
        / "cannbot_debug_isolated_restart.json"
    )
    qwen = load(qwen_restart_path)
    qwen_state = str(qwen.get("generation_service_state") or "")
    if not qwen_state.startswith("paused_"):
        raise SystemExit(f"Qwen reference is not paused: {qwen_state}")
    if qwen.get("runner_pid") or qwen.get("finalizer_pid"):
        raise SystemExit("Qwen reference still records a live runner/finalizer")

    source_config = load(args.source_key_config)
    source_keys = source_config.get("keys", [])
    filtered_keys = [
        item
        for item in source_keys
        if isinstance(item, dict)
        and str(item.get("name") or "").strip().lower()
        != EXCLUDED_PROVIDER
    ]
    if not filtered_keys:
        raise SystemExit("no Kimi credentials remain after explicit exclusion")
    target_config = {
        **source_config,
        "model": "k3",
        "base_url": str(
            source_config.get("base_url") or "https://api.kimi.com/coding/"
        ),
        "quota_probe": "kimi_usage",
        "keys": filtered_keys,
    }
    secrets = root / ".secrets"
    secrets.mkdir(parents=True, exist_ok=True)
    secrets.chmod(0o700)
    key_config = secrets / "kimi_keys_agent2.json"
    atomic_json(key_config, target_config, mode=0o600)

    existing_restart = harness / "cannbot_debug_kimi_isolated_restart.json"
    if existing_restart.is_file():
        existing = load(existing_restart)
        for field in ("runner_pid", "finalizer_pid"):
            pid = int(existing.get(field) or 0)
            if pid and pid_alive(pid):
                raise SystemExit(f"refusing duplicate start; {field} {pid} is alive")

    runner_command = [
        "python3",
        str(harness / "run_regenerate.py"),
        "--manifest",
        str(manifest_path),
        "--experiment-root",
        str(root),
        "--key-config",
        str(key_config),
        "--model",
        "k3",
        "--arm",
        "cannbot-debug",
        "--npus",
        "0,1,2,3,4,5,6,7",
        "--concurrency",
        "5",
        "--max-turns",
        "240",
        "--soft-task-turns",
        "480",
        "--max-task-turns",
        "600",
        "--max-internal-attempts",
        "5",
        "--timeout",
        "43200",
        "--preferred-provider",
        "provider-slot-a",
        "--slots-per-key",
        "5",
        "--exclude-provider",
        EXCLUDED_PROVIDER,
        "--min-provider-remaining",
        "20",
        "--quota-timeout",
        "20",
        "--provider-retries",
        "0",
        "--deliverable-retries",
        "3",
        "--repair-invalid-completed",
        "--provider-abort-cooldown-seconds",
        "300",
        "--provider-quota-cooldown-seconds",
        "21600",
        "--quota-retry-wait",
        "300",
        "--provider-refresh-seconds",
        "300",
        "--provider-env-name",
        "cannbot_debug_kimi_k3",
        "--queue-order",
        "reverse_trial_then_operator",
    ]
    runner_log = harness / "cannbot_debug_kimi_runner.log"
    finalizer_log = harness / "cannbot_debug_kimi_finalizer.log"
    runner_handle = runner_log.open("a")
    runner = subprocess.Popen(
        runner_command,
        cwd=root,
        stdout=runner_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    runner_handle.close()

    finalizer_command = [
        "python3",
        str(harness / "finalize_cannbot_debug_c2.py"),
        "--experiment-root",
        str(root),
        "--runner-pid",
        str(runner.pid),
        "--poll-seconds",
        "60",
        "--validation-npus",
        "0,1,2,3,4,5,6,7",
        "--container",
        "cannbot-runtimebot_c2_kimi_validation",
    ]
    finalizer_handle = finalizer_log.open("a")
    finalizer = subprocess.Popen(
        finalizer_command,
        cwd=root,
        stdout=finalizer_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    finalizer_handle.close()

    restart = {
        "schema_version": 1,
        "updated_at": now(),
        "generation_service_state": "running_cannbot_debug_kimi_k3_c2",
        "generation_resume_requires_user_confirmation": False,
        "experiment_root": str(root),
        "reference_qwen_experiment": str(args.qwen_reference.resolve()),
        "reference_qwen_state_verified": qwen_state,
        "runner_pid": runner.pid,
        "finalizer_pid": finalizer.pid,
        "runner_command": runner_command,
        "finalizer_command": finalizer_command,
        "runner_log": str(runner_log),
        "finalizer_log": str(finalizer_log),
        "key_config": str(key_config),
        "key_config_mode": "0600",
        "key_config_source_name": "kimi_keys_agent2.json",
        "excluded_providers": [EXCLUDED_PROVIDER],
        "model": "k3",
        "context_window_tokens": 1_048_576,
        "effort": "max",
        "queue_order_scope": "reverse_trial_then_operator",
        "configured_npus": list(range(8)),
        "dynamic_free_npu_gate": True,
        "concurrency": 5,
        "slots_per_key": 5,
        "provider_policy": "live_remaining_capacity_diverse_assignment",
        "provider_retries": "unlimited",
        "deliverable_retries": 3,
        "per_invocation_max_turns": 240,
        "soft_task_turns": 480,
        "max_task_turns": 600,
        "max_internal_attempts": 5,
        "timeout_seconds": 43200,
    }
    atomic_json(existing_restart, restart)
    time.sleep(3)
    if not pid_alive(runner.pid) or not pid_alive(finalizer.pid):
        failed = dict(restart)
        failed.update(
            {
                "updated_at": now(),
                "generation_service_state": "failed_to_start",
                "runner_alive": pid_alive(runner.pid),
                "finalizer_alive": pid_alive(finalizer.pid),
            }
        )
        atomic_json(existing_restart, failed)
        raise SystemExit("runner or finalizer exited during launch preflight")
    print(
        json.dumps(
            {
                "experiment_root": str(root),
                "runner_pid": runner.pid,
                "finalizer_pid": finalizer.pid,
                "qwen_reference_state": qwen_state,
                "configured_npus": list(range(8)),
                "concurrency": 5,
                "excluded_provider": EXCLUDED_PROVIDER,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
