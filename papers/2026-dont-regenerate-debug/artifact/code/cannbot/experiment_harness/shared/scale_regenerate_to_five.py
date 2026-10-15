#!/usr/bin/env python3
"""Scale the live Regenerate arm from NPUs 1-3 to the full 1-5 pool.

The old runner is in the middle of a wave.  This controller preserves that
expensive in-flight trial, immediately starts two far-tail trials on NPUs 4/5,
then performs a clean runner handoff at the next wave boundary.  A temporary
empty key-pool file prevents the old runner from launching another wave during
the handoff; the exact original key-pool bytes are restored before the new
five-slot runner starts.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_regenerate


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def atomic_bytes(path: Path, content: bytes, mode: int = 0o600) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.chmod(mode)
    temporary.replace(path)


def atomic_json(path: Path, data: Any, mode: int = 0o600) -> None:
    atomic_bytes(path, (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode(), mode)


def pid_matches(pid: int, marker: str) -> bool:
    try:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return marker in command


def terminate(pid: int, marker: str, timeout: int = 20) -> None:
    if not pid_matches(pid, marker):
        return
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_matches(pid, marker):
            return
        time.sleep(0.5)
    os.kill(pid, signal.SIGKILL)


def running_run_ids(manifest: dict[str, Any]) -> set[str]:
    active = set()
    for record in manifest["runs"]:
        if record.get("arm") != "regenerate":
            continue
        state = load_json(Path(record["run_record_path"]))
        if isinstance(state, dict) and state.get("status") == "running":
            active.add(record["run_id"])
    return active


def usable_scaleout_keys(key_data: dict[str, Any], active_slots: list[str], minimum: float) -> list[dict[str, Any]]:
    base_url = str(key_data.get("base_url") or "https://api.kimi.com/coding/")
    states = []
    for item in key_data.get("keys", []):
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(item.get("name", "")))
        if any(safe_name and safe_name in slot for slot in active_slots):
            continue
        quota = run_regenerate.query_kimi_capacity(item, base_url, timeout=20)
        capacity = quota.get("capacity_percent")
        if quota.get("success") and isinstance(capacity, (int, float)) and capacity >= minimum:
            states.append((float(capacity), item, quota))
    states.sort(key=lambda value: value[0], reverse=True)
    if len(states) < 2:
        raise RuntimeError(f"only {len(states)} non-active providers have at least {minimum:g}% remaining")
    return [item for _, item, _ in states[:2]]


def choose_scaleout_records(manifest: dict[str, Any], count: int) -> list[dict[str, Any]]:
    candidates = []
    for record in reversed(manifest["runs"]):
        if record.get("arm") != "regenerate":
            continue
        final_path = Path(record["final_result_path"])
        state = load_json(Path(record["run_record_path"]))
        if final_path.is_file() and final_path.stat().st_size > 0:
            continue
        if isinstance(state, dict) and state.get("status") == "running":
            continue
        candidates.append(record)
        if len(candidates) == count:
            break
    if len(candidates) != count:
        raise RuntimeError(f"could not select {count} untouched tail records")
    return list(reversed(candidates))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("/workspace/regenerate_experiment"),
    )
    parser.add_argument("--old-runner-pid", type=int, required=True)
    parser.add_argument("--old-finalizer-pid", type=int, required=True)
    parser.add_argument("--debug-pid", type=int, default=4038504)
    parser.add_argument("--poll-seconds", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.experiment_root.resolve()
    harness = root / "experiment_harness"
    manifest_path = harness / "run_manifest.json"
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("runs"), list):
        raise SystemExit(f"invalid manifest: {manifest_path}")
    if not pid_matches(args.old_runner_pid, "run_regenerate.py"):
        raise SystemExit(f"old runner is not alive: {args.old_runner_pid}")

    key_path = root / ".secrets" / "kimi_keys_agent2.json"
    backup_path = root / ".secrets" / "kimi_keys_agent2.json.scale_to_five_backup"
    scaleout_key_path = root / ".secrets" / "kimi_keys_scaleout_45.json"
    status_path = harness / "regenerate_scale_transition.json"
    controller_log = root / "orchestration_logs" / "regenerate_scale_transition.log"
    original_keys = key_path.read_bytes()
    key_data = json.loads(original_keys)
    if backup_path.exists():
        raise SystemExit(f"refusing to overwrite existing transition backup: {backup_path}")

    active_ids = running_run_ids(manifest)
    if not active_ids:
        raise SystemExit("no active old-runner trial found; direct restart should be used")
    active_slots = []
    for record in manifest["runs"]:
        if record.get("run_id") not in active_ids:
            continue
        state = load_json(Path(record["run_record_path"]))
        if isinstance(state, dict):
            active_slots.append(str(state.get("provider_slot") or ""))

    scaleout_records = choose_scaleout_records(manifest, 2)
    scaleout_keys = usable_scaleout_keys(key_data, active_slots, minimum=20.0)
    filtered = {key: value for key, value in key_data.items() if key != "keys"}
    filtered["keys"] = scaleout_keys
    filtered["model"] = "k3"

    backup_path.write_bytes(original_keys)
    backup_path.chmod(0o600)
    atomic_json(scaleout_key_path, filtered)
    atomic_json(
        key_path,
        {
            "base_url": key_data.get("base_url") or "https://api.kimi.com/coding/",
            "model": "k3",
            "keys": [],
            "transition": "temporarily block old runner from starting another wave",
            "created_at": now(),
        },
    )

    transition: dict[str, Any] = {
        "phase": "scaleout_started",
        "started_at": now(),
        "old_runner_pid": args.old_runner_pid,
        "old_finalizer_pid": args.old_finalizer_pid,
        "preserved_active_run_ids": sorted(active_ids),
        "scaleout_run_ids": [record["run_id"] for record in scaleout_records],
        "scaleout_npus": ["4", "5"],
        "target_npus": ["1", "2", "4", "5", "3"],
        "target_concurrency": 5,
    }
    atomic_json(status_path, transition, mode=0o644)

    scaleout_command = [
        "python3",
        str(harness / "run_regenerate.py"),
        "--key-config",
        str(scaleout_key_path),
        "--npus",
        "4,5",
        "--concurrency",
        "2",
        "--provider-env-name",
        "regenerate_scaleout_45",
        "--quota-retry-wait",
        "60",
        "--max-quota-waits",
        "3",
        "--provider-retries",
        "2",
    ]
    for record in scaleout_records:
        scaleout_command.extend(["--only", record["run_id"]])

    scaleout_log_path = root / "orchestration_logs" / "regenerate_scaleout_45.log"
    swapped = True
    try:
        with scaleout_log_path.open("a", encoding="utf-8") as scaleout_log:
            scaleout = subprocess.Popen(
                scaleout_command,
                cwd=root,
                stdout=scaleout_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        transition.update({"scaleout_pid": scaleout.pid, "phase": "waiting_for_wave_boundary"})
        atomic_json(status_path, transition, mode=0o644)
        print(json.dumps(transition, ensure_ascii=False), flush=True)

        # The old runner will process its current wave, delete the invalid
        # provider-aborted attempt, then encounter the temporary empty pool.
        log_path = root / "orchestration_logs" / "regenerate_full.log"
        initial_size = log_path.stat().st_size if log_path.exists() else 0
        boundary_seen = False
        while not boundary_seen:
            if not pid_matches(args.old_runner_pid, "run_regenerate.py"):
                boundary_seen = True
                break
            if log_path.exists() and log_path.stat().st_size > initial_size:
                with log_path.open("rb") as handle:
                    handle.seek(initial_size)
                    boundary_seen = b"waiting_for_provider_quota" in handle.read()
            if not boundary_seen:
                time.sleep(max(2, args.poll_seconds))

        scaleout_rc = scaleout.wait()
        transition.update(
            {
                "phase": "handoff",
                "old_wave_boundary_seen": boundary_seen,
                "scaleout_returncode": scaleout_rc,
                "scaleout_ended_at": now(),
            }
        )
        atomic_json(status_path, transition, mode=0o644)

        # Stop the old finalizer first so it cannot mistake the intentional
        # runner handoff for an incomplete experiment.
        terminate(args.old_finalizer_pid, "finalize_step3.py")
        terminate(args.old_runner_pid, "run_regenerate.py")
    finally:
        if swapped:
            atomic_bytes(key_path, original_keys)
        backup_path.unlink(missing_ok=True)
        scaleout_key_path.unlink(missing_ok=True)

    main_log_path = root / "orchestration_logs" / "regenerate_full.log"
    with main_log_path.open("a", encoding="utf-8") as main_log:
        new_runner = subprocess.Popen(
            [
                "python3",
                str(harness / "run_regenerate.py"),
                "--npus",
                "1,2,4,5,3",
                "--concurrency",
                "5",
                "--provider-env-name",
                "regenerate_rolling_12453",
            ],
            cwd=root,
            stdout=main_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finalizer_log_path = root / "orchestration_logs" / "step3_finalizer.log"
    with finalizer_log_path.open("a", encoding="utf-8") as finalizer_log:
        new_finalizer = subprocess.Popen(
            [
                "python3",
                str(harness / "finalize_step3.py"),
                "--regenerate-pid",
                str(new_runner.pid),
                "--debug-pid",
                str(args.debug_pid),
                "--poll-seconds",
                "60",
            ],
            cwd=root,
            stdout=finalizer_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    transition.update(
        {
            "phase": "complete",
            "completed_at": now(),
            "new_runner_pid": new_runner.pid,
            "new_finalizer_pid": new_finalizer.pid,
            "restored_default_key_pool": str(key_path),
        }
    )
    atomic_json(status_path, transition, mode=0o644)
    controller_log.write_text(json.dumps(transition, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(transition, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
