#!/usr/bin/env python3
"""Wait for both STEP-3 generation arms and run the frozen final pipeline.

This process is deliberately conservative: it never launches or resets a model
run.  It waits for the two already-launched runners, verifies that all expected
records have terminal artifacts, then runs the shared external validator and
the preregistered accounting/analysis checks.  If a runner exits early, the
finalizer records the incomplete state and stops instead of fabricating output.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TERMINAL_REGENERATE = {"completed", "failed"}
RETRYABLE_DEBUG = {"provider_api_error"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def pid_alive(pid: int, command_marker: str) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return command_marker in command


def nonempty(path_value: Any) -> bool:
    if not path_value:
        return False
    path = Path(str(path_value))
    return path.is_file() and path.stat().st_size > 0


def regenerate_record_terminal(record: Any, final_result_path: Any) -> bool:
    """Accept completed results or auditable terminal failures.

    A timed-out launcher may leave an intentionally empty ``final_stdout.json``.
    Requiring that file to be non-empty incorrectly reclassifies a formal
    timeout as an unfinished generation.  Provider interruptions remain
    non-terminal, while failed rows require durable timeout/exit evidence.
    """
    if not isinstance(record, dict):
        return False
    if record.get("provider_aborted") is True:
        return False
    status = record.get("status")
    if status == "completed":
        return nonempty(final_result_path)
    if status != "failed":
        return False
    return bool(
        record.get("ended_at")
        and (
            record.get("timed_out") is True
            or record.get("terminal_timeout") is True
            or (
                isinstance(record.get("launcher_returncode"), int)
                and record.get("launcher_returncode") != 0
            )
        )
    )


def debug_retryable(status: dict[str, Any]) -> bool:
    outcome = status.get("session_outcome")
    if outcome in RETRYABLE_DEBUG:
        return True
    notes = str(status.get("notes") or "")
    if outcome == "crashed" and (
        status.get("ended_at") is None
        or "无终态事件" in notes
        or "异常中断" in notes
    ):
        return True
    if outcome == "crashed" and status.get("entry_failure_type") is None and "failure_type" in notes:
        return True
    return False


def generation_state(manifest_path: Path, regenerate_pid: int, debug_pid: int) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("runs"), list):
        raise RuntimeError(f"invalid manifest: {manifest_path}")

    regenerate = [row for row in manifest["runs"] if row.get("arm") == "regenerate"]
    fresh_debug = [
        row
        for row in manifest["runs"]
        if row.get("arm") == "debug" and row.get("source_type") == "fresh_initial_near_miss"
    ]
    regenerate_terminal: list[str] = []
    regenerate_incomplete: list[str] = []
    for row in regenerate:
        record = load_json(Path(row["run_record_path"]))
        ready = regenerate_record_terminal(record, row.get("final_result_path"))
        (regenerate_terminal if ready else regenerate_incomplete).append(row["run_id"])

    debug_terminal: list[str] = []
    debug_incomplete: list[str] = []
    debug_outcomes: dict[str, int] = {}
    for row in fresh_debug:
        status = load_json(Path(row["task_dir"]) / "debug_status.json")
        outcome = status.get("session_outcome") if isinstance(status, dict) else None
        if outcome:
            debug_outcomes[str(outcome)] = debug_outcomes.get(str(outcome), 0) + 1
        ready = isinstance(status, dict) and bool(outcome) and not debug_retryable(status)
        (debug_terminal if ready else debug_incomplete).append(row["run_id"])

    return {
        "updated_at": now(),
        "runner_alive": {
            "regenerate": pid_alive(regenerate_pid, "run_regenerate.py"),
            "debug": pid_alive(debug_pid, "run_debug_fresh.py"),
        },
        "regenerate": {
            "expected": len(regenerate),
            "terminal": len(regenerate_terminal),
            "incomplete": regenerate_incomplete,
        },
        "fresh_debug": {
            "expected": len(fresh_debug),
            "terminal": len(debug_terminal),
            "outcomes": debug_outcomes,
            "incomplete": debug_incomplete,
        },
    }


def run_stage(name: str, command: list[str], cwd: Path, stages: list[dict[str, Any]]) -> None:
    print(json.dumps({"stage": name, "status": "started", "at": now(), "command": command}), flush=True)
    started = now()
    proc = subprocess.run(command, cwd=cwd)
    item = {"stage": name, "started_at": started, "ended_at": now(), "returncode": proc.returncode}
    stages.append(item)
    print(json.dumps(item), flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"stage {name} failed with return code {proc.returncode}")


def source_is_unchanged(root: Path) -> dict[str, Any]:
    repo = Path("/workspace/debug_agent")
    evidence = root / "evidence"
    before_head = (evidence / "ascendopgenagent_head_before_step3.txt").read_text()
    before_status = (evidence / "ascendopgenagent_status_before_step3.txt").read_bytes()
    current_head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    current_status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    return {
        "repo": str(repo),
        "head_before": before_head.strip(),
        "head_after": current_head.strip(),
        "head_unchanged": current_head == before_head,
        "status_bytes_unchanged": current_status == before_status,
        "unchanged": current_head == before_head and current_status == before_status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("/workspace/regenerate_experiment"),
    )
    parser.add_argument("--regenerate-pid", type=int, required=True)
    parser.add_argument("--debug-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.experiment_root.resolve()
    harness = root / "experiment_harness"
    manifest = harness / "run_manifest.json"
    status_path = harness / "step3_finalizer_status.json"

    state = generation_state(manifest, args.regenerate_pid, args.debug_pid)
    if args.status_only:
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return 0

    while state["runner_alive"]["regenerate"] or state["runner_alive"]["debug"]:
        atomic_json(status_path, {"phase": "waiting_for_generation", **state})
        print(
            json.dumps(
                {
                    "at": state["updated_at"],
                    "regenerate": f"{state['regenerate']['terminal']}/{state['regenerate']['expected']}",
                    "fresh_debug": f"{state['fresh_debug']['terminal']}/{state['fresh_debug']['expected']}",
                    "runner_alive": state["runner_alive"],
                }
            ),
            flush=True,
        )
        time.sleep(max(5, args.poll_seconds))
        state = generation_state(manifest, args.regenerate_pid, args.debug_pid)

    generation_ready = (
        state["regenerate"]["terminal"] == state["regenerate"]["expected"]
        and state["fresh_debug"]["terminal"] == state["fresh_debug"]["expected"]
    )
    if not generation_ready:
        atomic_json(status_path, {"phase": "generation_incomplete", **state})
        print(json.dumps({"status": "generation_incomplete", **state}, ensure_ascii=False), flush=True)
        return 3

    stages: list[dict[str, Any]] = []
    validation_npus = os.environ.get("CANNBOT_VALIDATION_NPUS", "0,1,2,3,4")
    try:
        run_stage(
            "validate_regenerate",
            [
                "python3",
                str(harness / "validate_manifest.py"),
                "--arm",
                "regenerate",
                "--npus",
                validation_npus,
                "--concurrency",
                "5",
            ],
            root,
            stages,
        )
        shutil.copy2(
            harness / "validation_run_summary.json",
            harness / "validation_run_summary_regenerate.json",
        )
        run_stage(
            "validate_debug",
            [
                "python3",
                str(harness / "validate_manifest.py"),
                "--arm",
                "debug",
                "--npus",
                validation_npus,
                "--concurrency",
                "5",
            ],
            root,
            stages,
        )
        shutil.copy2(
            harness / "validation_run_summary.json",
            harness / "validation_run_summary_debug.json",
        )

        # Provider/infrastructure retry evidence is intentionally not archived.
        # Valid business-failure attempts remain in each task directory.
        retry_evidence = root / "debug_fresh_k3" / "agent2" / "retry_evidence"
        deleted_retry_evidence = retry_evidence.exists()
        if deleted_retry_evidence:
            shutil.rmtree(retry_evidence)

        for name, script, extra in (
            ("audit_regenerate_journals", "audit_regenerate_journals.py", []),
            ("aggregate_tokens_cost", "aggregate_tokens_cost.py", ["--strict"]),
            ("analyze_results", "analyze_results.py", []),
            ("white_box_validate", "white_box_validate.py", []),
        ):
            run_stage(name, ["python3", str(harness / script), *extra], root, stages)

        source_check = source_is_unchanged(root)
        if not source_check["unchanged"]:
            raise RuntimeError("/workspace/debug_agent changed during STEP 3")
        summary = load_json(harness / "token_cost_summary.json")
        rows = summary.get("rows", []) if isinstance(summary, dict) else []
        non_k3 = [
            row.get("run_id")
            for row in rows
            if row.get("same_model_eligible") and str(row.get("model", "")).lower() not in {"k3", "k3[1m]", "kimi-k3"}
        ]
        if non_k3:
            raise RuntimeError(f"same-model output contains non-K3 rows: {non_k3[:5]}")

        completion = {
            "phase": "complete",
            "completed_at": now(),
            "generation": state,
            "stages": stages,
            "deleted_operational_retry_evidence": deleted_retry_evidence,
            "source_isolation": source_check,
            "summary_rows": len(rows),
            "same_model_non_k3_rows": len(non_k3),
        }
        atomic_json(status_path, completion)
        atomic_json(harness / "step3_completion.json", completion)
        print(json.dumps(completion, indent=2, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - durable finalizer must retain exact failure state
        atomic_json(
            status_path,
            {
                "phase": "finalization_failed",
                "failed_at": now(),
                "error": repr(exc),
                "generation": state,
                "stages": stages,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
