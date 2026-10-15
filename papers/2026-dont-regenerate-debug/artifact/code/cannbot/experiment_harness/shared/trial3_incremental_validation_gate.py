#!/usr/bin/env python3
"""Drain rolling generation at the trial3 boundary, validate gaps, and resume.

The live runner loads provider credentials for each newly scheduled worker,
while already-running trial containers use a private per-trial environment.
This controller therefore waits until every Regenerate trial3 row is terminal,
atomically replaces the default key pool with an empty pool, and lets any
already-running trial2 workers continue.  As soon as at least one NPU is free,
it validates only completed trial3 rows missing full-eval or anti-cheat.  The
original key configuration is restored afterwards so the unchanged runner can
continue the rolling trial2 queue.

No generation workspace, journal, checkpoint, or candidate is deleted.
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


TRIAL_TERMINAL = {"completed", "failed"}


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


def update_json(path: Path, updates: dict[str, Any], mode: int = 0o600) -> None:
    current = load_json(path)
    if not isinstance(current, dict):
        current = {}
    current.update(updates)
    atomic_json(path, current, mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("/workspace/regenerate_experiment"),
    )
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--validation-timeout", type=int, default=1200)
    parser.add_argument("--expected-trial3", type=int, default=27)
    return parser.parse_args()


def regenerate_records(manifest: dict[str, Any], trial: int | None = None) -> list[dict[str, Any]]:
    records = [record for record in manifest.get("runs", []) if record.get("arm") == "regenerate"]
    if trial is not None:
        records = [record for record in records if int(record.get("trial", 0) or 0) == trial]
    return records


def run_state(record: dict[str, Any]) -> dict[str, Any]:
    value = load_json(Path(record["run_record_path"]))
    return value if isinstance(value, dict) else {}


def trial3_snapshot(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    nonterminal: list[str] = []
    for record in records:
        state = run_state(record)
        status = str(state.get("status") or "not_started")
        # A harness wall-clock timeout is an infrastructure interruption, not
        # evidence that the model completed the trial.  Keep the gate armed so
        # monitoring can restore the same session/workspace and resume it.
        if status == "failed" and state.get("timed_out") is True:
            status = "timeout_interrupted"
        counts[status] = counts.get(status, 0) + 1
        if status not in TRIAL_TERMINAL:
            nonterminal.append(record["run_id"])
    return {"counts": counts, "nonterminal": nonterminal}


def configured_npus(restart: dict[str, Any]) -> list[str]:
    values = restart.get("npu_order")
    if isinstance(values, list) and values:
        return [str(value) for value in values]
    command = restart.get("runner_command")
    if isinstance(command, list) and "--npus" in command:
        index = command.index("--npus")
        if index + 1 < len(command):
            return [value.strip() for value in str(command[index + 1]).split(",") if value.strip()]
    return ["0", "1", "2", "3", "4"]


def active_generation_npus(records: list[dict[str, Any]]) -> set[str]:
    active = set()
    for record in records:
        state = run_state(record)
        if state.get("status") == "running" and state.get("npu") is not None:
            active.add(str(state["npu"]))
    return active


def missing_trial3_validation(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing = []
    for record in records:
        state = run_state(record)
        if state.get("status") != "completed" or state.get("candidate_materialized") is not True:
            continue
        full_path = Path(record["full_eval_path"])
        anti_path = Path(record["anticheat_path"])
        if not full_path.is_file() or not anti_path.is_file():
            missing.append(record)
    return missing


def validation_snapshot(records: list[dict[str, Any]]) -> dict[str, Any]:
    full: dict[str, int] = {}
    anti: dict[str, int] = {}
    both = 0
    for record in records:
        state = run_state(record)
        if state.get("status") != "completed":
            continue
        full_data = load_json(Path(record["full_eval_path"]))
        anti_data = load_json(Path(record["anticheat_path"]))
        full_value = "missing"
        anti_value = "missing"
        if isinstance(full_data, dict):
            full_value = str(full_data.get("full_eval_result") or full_data.get("final_failure_type") or "unknown")
        if isinstance(anti_data, dict):
            anti_value = str(anti_data.get("verdict") or "unknown")
        full[full_value] = full.get(full_value, 0) + 1
        anti[anti_value] = anti.get(anti_value, 0) + 1
        if full_value in {"pass", "success"} and anti_value == "CLEAN":
            both += 1
    return {"full_eval": full, "anticheat": anti, "pass_and_clean": both}


def engage_provider_drain(key_path: Path, hold_path: Path) -> bytes:
    if hold_path.exists():
        raise RuntimeError(f"refusing to overwrite existing key hold: {hold_path}")
    original = key_path.read_bytes()
    data = json.loads(original)
    if not isinstance(data, dict) or not isinstance(data.get("keys"), list) or not data["keys"]:
        raise RuntimeError("default Qwen key config has no schedulable keys before gate")
    hold_path.write_bytes(original)
    hold_path.chmod(0o600)
    drained = dict(data)
    drained["keys"] = []
    drained["transition"] = "trial3_incremental_validation_drain"
    drained["drained_at"] = now()
    atomic_json(key_path, drained)
    return original


def restore_provider_pool(key_path: Path, hold_path: Path, original: bytes) -> str:
    current = load_json(key_path)
    if isinstance(current, dict) and current.get("transition") == "trial3_incremental_validation_drain":
        atomic_bytes(key_path, original)
        hold_path.unlink(missing_ok=True)
        return "restored_original_pool"
    if isinstance(current, dict) and current.get("keys"):
        # An operator rotated the credential during validation.  Preserve the
        # newer live configuration instead of overwriting it with the hold.
        hold_path.unlink(missing_ok=True)
        return "kept_operator_updated_pool"
    return "restore_blocked_unexpected_key_config"


def main() -> int:
    args = parse_args()
    root = args.experiment_root.resolve()
    harness = root / "experiment_harness"
    manifest_path = harness / "run_manifest.json"
    restart_path = harness / "regenerate_isolated_restart.json"
    gate_path = harness / "trial3_incremental_validation_gate.json"
    lock_path = harness / ".trial3_incremental_validation_gate.lock"
    log_path = root / "orchestration_logs" / "trial3_incremental_validation.log"

    lock_path.touch(mode=0o600, exist_ok=True)
    with lock_path.open("r+") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another trial3 validation gate controller is active")

        manifest = load_json(manifest_path)
        restart = load_json(restart_path)
        if not isinstance(manifest, dict) or not isinstance(restart, dict):
            raise SystemExit("manifest or restart JSON is invalid")
        trial3 = regenerate_records(manifest, trial=3)
        all_regenerate = regenerate_records(manifest)
        if len(trial3) != args.expected_trial3:
            raise SystemExit(f"expected {args.expected_trial3} trial3 records, found {len(trial3)}")

        key_path = Path(restart["key_config"])
        hold_path = key_path.with_name(key_path.name + ".trial3_validation_hold")
        initial = {
            "phase": "armed_waiting_for_trial3",
            "armed_at": now(),
            "controller_pid": os.getpid(),
            "runner_pid": restart.get("runner_pid"),
            "finalizer_pid": restart.get("finalizer_pid"),
            "policy": {
                "interrupt_active_generation": False,
                "block_only_future_refill": True,
                "validate_only_missing_completed_trial3": True,
                "restore_generation_after_validation": True,
            },
            "trial3": trial3_snapshot(trial3),
        }
        atomic_json(gate_path, initial, mode=0o644)
        update_json(
            restart_path,
            {
                "trial3_incremental_validation_gate": initial,
                "generation_service_state": "running_until_trial3_validation_gate",
            },
        )

        last_snapshot: dict[str, Any] | None = None
        while True:
            snapshot = trial3_snapshot(trial3)
            if snapshot != last_snapshot:
                print(json.dumps({"at": now(), "phase": "waiting_for_trial3", **snapshot}, ensure_ascii=False), flush=True)
                initial["trial3"] = snapshot
                initial["updated_at"] = now()
                atomic_json(gate_path, initial, mode=0o644)
                last_snapshot = snapshot
            if not snapshot["nonterminal"]:
                break
            time.sleep(max(0.05, args.poll_seconds))

        original = b""
        restore_status = "not_engaged"
        validation_rc: int | None = None
        summary_path: Path | None = None
        try:
            original = engage_provider_drain(key_path, hold_path)
            phase = {
                **initial,
                "phase": "draining_future_refill",
                "trial3_terminal_at": now(),
                "trial3": trial3_snapshot(trial3),
                "key_pool_drained": True,
            }
            atomic_json(gate_path, phase, mode=0o644)
            update_json(
                restart_path,
                {
                    "generation_service_state": "draining_for_trial3_incremental_validation",
                    "trial3_incremental_validation_gate": phase,
                },
            )
            print(json.dumps({"at": now(), "phase": phase["phase"]}, ensure_ascii=False), flush=True)

            npus = configured_npus(restart)
            free: list[str] = []
            while not free:
                active = active_generation_npus(all_regenerate)
                free = [npu for npu in npus if npu not in active]
                if not free:
                    time.sleep(max(0.25, args.poll_seconds))

            missing = missing_trial3_validation(trial3)
            before = validation_snapshot(trial3)
            phase.update(
                {
                    "phase": "validating_missing_trial3",
                    "validation_started_at": now(),
                    "validation_npus": free,
                    "missing_run_ids": [record["run_id"] for record in missing],
                    "missing_count": len(missing),
                    "validation_before": before,
                }
            )
            atomic_json(gate_path, phase, mode=0o644)
            update_json(restart_path, {"trial3_incremental_validation_gate": phase})

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
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as log:
                    completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                validation_rc = completed.returncode
                generated_summary = harness / "validation_run_summary.json"
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                summary_path = harness / f"validation_run_summary_incremental_trial3_{stamp}.json"
                if generated_summary.is_file():
                    shutil.copy2(generated_summary, summary_path)
            else:
                validation_rc = 0

            after = validation_snapshot(trial3)
            still_missing = [record["run_id"] for record in missing_trial3_validation(trial3)]
            phase.update(
                {
                    "phase": "validation_finished",
                    "validation_finished_at": now(),
                    "validation_returncode": validation_rc,
                    "validation_summary": str(summary_path) if summary_path else None,
                    "validation_after": after,
                    "still_missing_run_ids": still_missing,
                }
            )
            atomic_json(gate_path, phase, mode=0o644)
        finally:
            if original:
                restore_status = restore_provider_pool(key_path, hold_path, original)
            final = load_json(gate_path)
            if not isinstance(final, dict):
                final = dict(initial)
            final.update(
                {
                    "phase": "completed" if validation_rc == 0 else "completed_with_validation_errors",
                    "completed_at": now(),
                    "provider_pool_restore_status": restore_status,
                    "generation_resumed": restore_status in {
                        "restored_original_pool",
                        "kept_operator_updated_pool",
                    },
                }
            )
            atomic_json(gate_path, final, mode=0o644)
            update_json(
                restart_path,
                {
                    "generation_service_state": "running_after_trial3_incremental_validation",
                    "trial3_incremental_validation_gate": final,
                },
            )
            print(json.dumps(final, ensure_ascii=False), flush=True)

        return 0 if validation_rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
