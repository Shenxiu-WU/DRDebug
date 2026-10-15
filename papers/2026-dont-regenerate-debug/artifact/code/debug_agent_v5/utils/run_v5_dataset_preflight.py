#!/usr/bin/env python3
"""Validate all frozen V5 tasks on isolated copies across NPU 3-7."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEBUGGABLE_FAILURE_TYPES = {
    "build_failed",
    "import_failed",
    "runtime_error",
    "timeout",
    "precision_failed",
}
RUNTIME_NAMES = {
    ".debug_events",
    ".verify_logs",
    ".verify_status",
    "__pycache__",
    "build",
    "debug_status.json",
    "debug_trace.md",
    "precision_tuning",
}


@dataclass(frozen=True)
class Target:
    rel: str
    source: Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ignore_copy(_directory: str, names: list[str]) -> set[str]:
    return {
        name for name in names
        if name in RUNTIME_NAMES
        or name.startswith("_claude_result")
        or name.startswith("_provider")
        or name == "_anticheat.json"
        or name.endswith(".pyc")
    }


def prepare_output_dir(output: Path) -> None:
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise SystemExit(f"refuse existing preflight output: {output}")
    else:
        output.mkdir(parents=True)


def prepare_isolated_task(target: Target, work_dir: Path) -> None:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(target.source, work_dir, ignore=_ignore_copy)


def _docker_build_and_verify(
    *,
    container: str,
    npu: str,
    repo_root: Path,
    task_dir: Path,
    tilelang_env: str,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    shell = (
        'set +e; [ -f "$1" ] && source "$1"; cd "$2"; '
        f"timeout --signal=TERM --kill-after=30 {timeout} "
        'bash -lc \'python3 utils/build_ascendc.py "$1" --clean '
        '&& python3 utils/verification_ascendc.py "$1"\' _ "$3"'
    )
    return subprocess.run(
        [
            "docker", "exec",
            "-e", f"ASCEND_RT_VISIBLE_DEVICES={npu}",
            container,
            "bash", "-lc", shell, "_",
            tilelang_env, str(repo_root), str(task_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout + 90,
    )


def _classify(
    *,
    repo_root: Path,
    task_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    exit_code: int,
) -> tuple[dict[str, Any], subprocess.CompletedProcess[str]]:
    proc = subprocess.run(
        [
            "python3", str(repo_root / "utils" / "classify_verify_result.py"),
            "--exit-code", str(exit_code),
            "--stdout-path", str(stdout_path),
            "--stderr-path", str(stderr_path),
            "--task-dir", str(task_dir),
            "--phase", "4",
            "--attempt", "0",
            "--write-status",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {
            "failure_type": "classifier_error",
            "classifier_stdout": proc.stdout,
            "classifier_stderr": proc.stderr,
        }
    return payload, proc


def evaluate_one(
    *,
    target: Target,
    output: Path,
    repo_root: Path,
    container: str,
    npu: str,
    tilelang_env: str,
    timeout: int,
) -> dict[str, Any]:
    work_dir = output / "work" / target.rel
    logs_dir = output / "logs" / target.rel
    result_path = output / "results" / f"{target.rel.replace('/', '__')}.json"
    logs_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    prepare_isolated_task(target, work_dir)

    verify = _docker_build_and_verify(
        container=container,
        npu=npu,
        repo_root=repo_root,
        task_dir=work_dir,
        tilelang_env=tilelang_env,
        timeout=timeout,
    )
    stdout_path = logs_dir / "verify.stdout"
    stderr_path = logs_dir / "verify.stderr"
    stdout_path.write_text(verify.stdout, encoding="utf-8")
    stderr_path.write_text(verify.stderr, encoding="utf-8")
    status, classifier = _classify(
        repo_root=repo_root,
        task_dir=work_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        exit_code=verify.returncode,
    )
    (logs_dir / "classifier.stderr").write_text(
        classifier.stderr, encoding="utf-8")
    failure_type = status.get("failure_type")
    row = {
        "task": target.rel,
        "source": str(target.source),
        "work_dir": str(work_dir),
        "npu": int(npu),
        "evaluated_at": _now(),
        "verification_exit_code": verify.returncode,
        "classifier_exit_code": classifier.returncode,
        "failure_type": failure_type,
        "preflight_passed": (
            failure_type in DEBUGGABLE_FAILURE_TYPES
            and classifier.returncode in (0, 1)
        ),
        "status": status,
    }
    result_path.write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return row


def _is_transient_npu_runtime(row: dict[str, Any]) -> bool:
    if row.get("failure_type") != "runtime_error":
        return False
    status = row.get("status") or {}
    evidence = "\n".join(
        str(status.get(key) or "")
        for key in ("exit_signal", "stdout_tail", "stderr_tail")
    )
    return (
        "507015" in evidence
        or "NPU_AICORE_EXCEPTION" in evidence
    )


def _archive_transient_attempt(
    output: Path,
    target: Target,
    attempt: int,
    row: dict[str, Any],
) -> None:
    archive = output / "transient_rechecks" / target.rel / f"attempt_{attempt}"
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "result.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logs = output / "logs" / target.rel
    if logs.is_dir():
        shutil.copytree(logs, archive / "logs", dirs_exist_ok=True)


def evaluate_with_transient_rechecks(
    *,
    target: Target,
    output: Path,
    repo_root: Path,
    container: str,
    npu: str,
    tilelang_env: str,
    timeout: int,
    transient_rechecks: int,
) -> dict[str, Any]:
    transient_rows = []
    row: dict[str, Any] = {}
    for attempt in range(transient_rechecks + 1):
        row = evaluate_one(
            target=target,
            output=output,
            repo_root=repo_root,
            container=container,
            npu=npu,
            tilelang_env=tilelang_env,
            timeout=timeout,
        )
        if not _is_transient_npu_runtime(row):
            break
        _archive_transient_attempt(output, target, attempt, row)
        transient_rows.append({
            "attempt": attempt,
            "failure_type": row.get("failure_type"),
            "exit_signal": (row.get("status") or {}).get("exit_signal"),
            "archive": str(
                Path("transient_rechecks")
                / target.rel
                / f"attempt_{attempt}"
            ),
        })
    if transient_rows:
        row["transient_infrastructure_rechecks"] = transient_rows
        row["stable_result_after_transient_recheck"] = (
            not _is_transient_npu_runtime(row)
        )
        result_path = (
            output / "results" / f"{target.rel.replace('/', '__')}.json"
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return row


def _evaluate_partition(
    *,
    targets: list[Target],
    output: Path,
    repo_root: Path,
    container: str,
    npu: str,
    tilelang_env: str,
    timeout: int,
    transient_rechecks: int,
) -> list[dict[str, Any]]:
    return [
        evaluate_with_transient_rechecks(
            target=target,
            output=output,
            repo_root=repo_root,
            container=container,
            npu=npu,
            tilelang_env=tilelang_env,
            timeout=timeout,
            transient_rechecks=transient_rechecks,
        )
        for target in targets
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--container", default="v5_cann")
    parser.add_argument("--npus", default="3,4,5,6,7")
    parser.add_argument(
        "--tilelang-env",
        default="/usr/local/Ascend/ascend-toolkit/set_env.sh",
    )
    parser.add_argument("--timeout", type=int, default=43200)
    parser.add_argument("--expected-tasks", type=int, default=27)
    parser.add_argument(
        "--transient-rechecks",
        type=int,
        default=2,
        help="Extra clean-build retries for explicit 507015/AICore transients.",
    )
    args = parser.parse_args()

    prepare_output_dir(args.output)
    targets = [
        Target(
            rel=f"{path.parent.name}/{path.name}",
            source=path,
        )
        for path in sorted(args.source_root.glob("level*/*"))
        if path.is_dir()
    ]
    if len(targets) != args.expected_tasks:
        raise SystemExit(
            f"expected {args.expected_tasks} source tasks, found {len(targets)}")
    npus = [item.strip() for item in args.npus.split(",") if item.strip()]
    if not npus:
        raise SystemExit("at least one NPU is required")

    partitions = [
        targets[index::len(npus)]
        for index in range(len(npus))
    ]
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(npus)) as pool:
        futures = [
            pool.submit(
                _evaluate_partition,
                targets=partition,
                output=args.output,
                repo_root=args.repo_root,
                container=args.container,
                npu=npus[index],
                tilelang_env=args.tilelang_env,
                timeout=args.timeout,
                transient_rechecks=max(0, args.transient_rechecks),
            )
            for index, partition in enumerate(partitions)
            if partition
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda row: row["task"])
    used_npus = sorted({row["npu"] for row in rows})
    summary = {
        "schema_version": 1,
        "generated_at": _now(),
        "source_root": str(args.source_root),
        "task_count": len(rows),
        "passed_count": sum(bool(row["preflight_passed"]) for row in rows),
        "used_npus": used_npus,
        "required_npus": [int(npu) for npu in npus],
        "all_required_npus_exercised": used_npus == sorted(map(int, npus)),
        "transient_recheck_task_count": sum(
            bool(row.get("transient_infrastructure_rechecks"))
            for row in rows
        ),
        "failure_type_counts": {
            failure_type: sum(
                row["failure_type"] == failure_type for row in rows)
            for failure_type in sorted({
                str(row["failure_type"]) for row in rows
            })
        },
        "rows": rows,
    }
    summary["passed"] = (
        summary["passed_count"] == args.expected_tasks
        and summary["all_required_npus_exercised"]
    )
    (args.output / "dataset_preflight_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
