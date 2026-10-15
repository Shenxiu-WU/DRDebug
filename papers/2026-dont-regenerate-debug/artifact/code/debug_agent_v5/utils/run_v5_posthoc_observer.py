#!/usr/bin/env python3
"""Evaluate one completed V5 arm on isolated task copies.

The observer never writes into ``arm_output/tasks``.  Each terminal candidate is
deep-copied, rebuilt and verified against the frozen source task's broadest case
set, then scanned by the common anti-cheat verifier.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CASE_RE = re.compile(r"^case\[\d+\]: output(?:\[[^\]]+\])+:\s*(.*)$", re.M)
STATUS_RE = re.compile(r"^Status\s*:\s*(\w+)", re.M)
INFRASTRUCTURE_RE = re.compile(
    r"507015|NPU_AICORE_EXCEPTION|ACL stream synchronize failed|"
    r"(?:NPU|device).*(?:unavailable|not available|lost)|"
    r"(?:driver|runtime).*(?:unavailable|initiali[sz]ation failed)",
    re.I,
)
RUNTIME_NAMES = {
    ".bench_baseline",
    ".debug_events",
    ".verify_logs",
    ".verify_status",
    "__pycache__",
    "build",
    "debug_status.json",
    "debug_trace.md",
    "precision_tuning",
}
CASE_EXCLUDES = {
    "debug_status.json",
    "experiment_manifest.json",
    "model.json",
    "round_summary.json",
    "run_summary.json",
}
WRAPPER_FILES = (
    "model.py",
    "model_new_ascendc.py",
    "model_new_tilelang.py",
)


@dataclass(frozen=True)
class Target:
    rel: str
    treatment: Path
    source: Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _count_cases(path: Path) -> int:
    return sum(1 for line in path.read_text(errors="replace").splitlines() if line.strip())


def _case_set_digest(path: Path) -> str:
    rows: list[str] = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            rows.append(line)
        else:
            rows.append(json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    digest = hashlib.sha256()
    for row in sorted(rows):
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _looks_like_cases(path: Path) -> bool:
    if path.name.startswith(("_", ".")) or path.name in CASE_EXCLUDES:
        return False
    if not path.name.endswith((".json", ".json.bak", ".json.full")):
        return False
    try:
        first = next(
            line for line in path.read_text(errors="replace").splitlines()
            if line.strip()
        )
        value = json.loads(first)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(value, dict) and "inputs" in value


def _active_name(path: Path) -> str:
    if path.name.endswith(".json.bak"):
        return path.name[:-4]
    if path.name.endswith(".json.full"):
        return path.name[:-5]
    return path.name


def choose_full_cases(source: Path) -> Path | None:
    candidates = [
        path for path in source.iterdir()
        if path.is_file() and _looks_like_cases(path)
    ]
    if not candidates:
        return None

    def rank(path: Path) -> tuple[int, int, str]:
        suffix = 2 if path.name.endswith((".json.bak", ".json.full")) else 1
        return (_count_cases(path), suffix, path.name)

    return max(candidates, key=rank)


def _ignore_copy(_directory: str, names: list[str]) -> set[str]:
    ignored = {
        name for name in names
        if name in RUNTIME_NAMES
        or name.startswith("_claude_result")
        or name.startswith("_provider")
        or name == "_anticheat.json"
        or name.endswith(".pyc")
    }
    return ignored


def prepare_isolated_task(
    target: Target,
    work_dir: Path,
) -> dict[str, Any]:
    """Deep-copy a treatment task and inject the frozen full case set."""
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(target.treatment, work_dir, ignore=_ignore_copy)

    source_model = target.source / "model.py"
    if source_model.is_file():
        shutil.copy2(source_model, work_dir / "model.py")
    baseline_dir = work_dir / ".bench_baseline"
    baseline_dir.mkdir()
    for name in WRAPPER_FILES:
        source_wrapper = target.source / name
        if not source_wrapper.is_file():
            continue
        baseline_copy = baseline_dir / name
        shutil.copy2(source_wrapper, baseline_copy)
        (baseline_dir / f"{name}.sha256").write_text(
            hashlib.sha256(source_wrapper.read_bytes()).hexdigest() + "\n",
            encoding="utf-8",
        )

    full_cases = choose_full_cases(target.source)
    if full_cases is None:
        return {
            "available": False,
            "reason": "missing_case_jsonl",
            "work_dir": str(work_dir),
        }

    active_source = target.source / _active_name(full_cases)
    active_count = _count_cases(active_source) if active_source.is_file() else 0
    active_digest = _case_set_digest(active_source) if active_source.is_file() else None
    full_count = _count_cases(full_cases)
    full_digest = _case_set_digest(full_cases)
    active_work = work_dir / _active_name(full_cases)
    active_work.write_bytes(full_cases.read_bytes())
    active_aliases: list[str] = []
    source_model_json = target.source / "model.json"
    if (
        source_model_json.is_file()
        and active_source.is_file()
        and source_model_json != active_source
        and _count_cases(source_model_json) == active_count
        and _case_set_digest(source_model_json) == active_digest
    ):
        (work_dir / "model.json").write_bytes(full_cases.read_bytes())
        active_aliases.append("model.json")
    return {
        "available": True,
        "work_dir": str(work_dir),
        "active_json": active_work.name,
        "active_json_aliases": active_aliases,
        "active_cases": active_count,
        "active_case_set_sha256": active_digest,
        "full_json_source": str(full_cases),
        "full_cases": full_count,
        "full_case_set_sha256": full_digest,
        "coverage_equivalent": (
            active_count == full_count and active_digest == full_digest
        ),
        "anticheat_baseline_source": str(target.source),
    }


def _docker_run(
    *,
    container: str,
    npu: str,
    repo_root: Path,
    task_dir: Path,
    tilelang_env: str,
    timeout: int,
    command: str,
) -> subprocess.CompletedProcess[str]:
    rendered_command = (
        command.replace("{task}", '"$3"')
        if "{task}" in command
        else f'{command} "$3"'
    )
    shell = (
        'set +e; [ -f "$1" ] && source "$1"; cd "$2"; '
        f'timeout --signal=TERM --kill-after=30 {timeout} '
        f"{rendered_command}"
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


def _classify_verify(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    comparisons = CASE_RE.findall(proc.stdout)
    total = len(comparisons)
    passed = sum(value.strip().startswith("matched") for value in comparisons)
    status = STATUS_RE.search(proc.stdout)
    status_value = status.group(1).upper() if status else ""
    infrastructure_error = _infrastructure_error(proc)
    return {
        "return_code": proc.returncode,
        "status": status_value,
        "passed_outputs": passed,
        "total_outputs": total,
        "match_rate": round(100.0 * passed / total, 4) if total else None,
        "objective_passed": proc.returncode == 0 and status_value == "PASS",
        "infrastructure_error": infrastructure_error,
    }


def _classify_build(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "return_code": proc.returncode,
        "passed": proc.returncode == 0,
        "infrastructure_error": _infrastructure_error(proc),
    }


def _infrastructure_error(
    proc: subprocess.CompletedProcess[str],
) -> str | None:
    text = "\n".join((proc.stdout or "", proc.stderr or ""))
    match = INFRASTRUCTURE_RE.search(text)
    return match.group(0) if match else None


def _evaluate_one(
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
    coverage = prepare_isolated_task(target, work_dir)
    row: dict[str, Any] = {
        "task": target.rel,
        "npu": int(npu),
        "treatment_task": str(target.treatment),
        "evaluated_at": _now(),
        "coverage": coverage,
    }
    if not coverage["available"]:
        row.update({
            "run_state": "missing_case_jsonl",
            "posthoc_clean_success": False,
        })
        result_path.write_text(json.dumps(row, ensure_ascii=False, indent=2))
        return row

    build = _docker_run(
        container=container,
        npu=npu,
        repo_root=repo_root,
        task_dir=work_dir,
        tilelang_env=tilelang_env,
        timeout=timeout,
        command="python3 utils/build_ascendc.py {task} --clean",
    )
    (logs_dir / "build.stdout").write_text(build.stdout, encoding="utf-8")
    (logs_dir / "build.stderr").write_text(build.stderr, encoding="utf-8")
    build_result = _classify_build(build)

    if build_result["passed"]:
        verify = _docker_run(
            container=container,
            npu=npu,
            repo_root=repo_root,
            task_dir=work_dir,
            tilelang_env=tilelang_env,
            timeout=timeout,
            command="python3 utils/verification_ascendc.py",
        )
        (logs_dir / "verify.stdout").write_text(verify.stdout, encoding="utf-8")
        (logs_dir / "verify.stderr").write_text(verify.stderr, encoding="utf-8")
        verify_result = _classify_verify(verify)
    else:
        verify_result = {
            "return_code": None,
            "status": "SKIPPED",
            "passed_outputs": 0,
            "total_outputs": 0,
            "match_rate": None,
            "objective_passed": False,
            "skipped_reason": "clean_build_failed",
        }

    infrastructure_error = (
        build_result.get("infrastructure_error")
        or verify_result.get("infrastructure_error")
    )
    if infrastructure_error:
        anticheat_payload = {
            "verdict": "SKIPPED",
            "reasons": ["posthoc_infrastructure_error"],
            "return_code": None,
        }
    else:
        anticheat = _docker_run(
            container=container,
            npu=npu,
            repo_root=repo_root,
            task_dir=work_dir,
            tilelang_env=tilelang_env,
            timeout=min(timeout, 900),
            command=(
                "python3 skills/ascendc/ascendc-debug/scripts/anticheat.py "
                "verify {task} --json"
            ),
        )
        (logs_dir / "anticheat.stdout").write_text(
            anticheat.stdout, encoding="utf-8")
        (logs_dir / "anticheat.stderr").write_text(
            anticheat.stderr, encoding="utf-8")
        try:
            anticheat_payload = json.loads(anticheat.stdout)
        except json.JSONDecodeError:
            anticheat_payload = {
                "verdict": "ERROR",
                "reasons": [
                    anticheat.stderr.strip() or anticheat.stdout.strip()
                ],
            }
        anticheat_payload["return_code"] = anticheat.returncode

    row.update({
        "run_state": (
            "infrastructure_error" if infrastructure_error else "completed"
        ),
        "infrastructure_error": infrastructure_error,
        "build": build_result,
        "verification": verify_result,
        "anticheat": anticheat_payload,
        "posthoc_clean_success": (
            build_result["passed"]
            and verify_result["objective_passed"]
            and anticheat_payload.get("verdict") == "CLEAN"
        ),
    })
    result_path.write_text(json.dumps(row, ensure_ascii=False, indent=2))
    return row


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
        row = _evaluate_one(
            target=target,
            output=output,
            repo_root=repo_root,
            container=container,
            npu=npu,
            tilelang_env=tilelang_env,
            timeout=timeout,
        )
        if row.get("run_state") != "infrastructure_error":
            break
        _archive_transient_attempt(output, target, attempt, row)
        transient_rows.append({
            "attempt": attempt,
            "signal": row.get("infrastructure_error"),
            "archive": str(
                Path("transient_rechecks")
                / target.rel
                / f"attempt_{attempt}"
            ),
        })
    if transient_rows:
        persistent_aicore_runtime = (
            row.get("run_state") == "infrastructure_error"
            and any(
                token in str(row.get("infrastructure_error") or "").lower()
                for token in (
                    "507015",
                    "npu_aicore_exception",
                    "acl stream synchronize failed",
                )
            )
        )
        if persistent_aicore_runtime:
            row["run_state"] = "completed"
            row["infrastructure_error"] = None
            row["persistent_runtime_failure"] = True
            row["persistent_runtime_failure_type"] = "runtime_error"
            row["posthoc_clean_success"] = False
            verification = row.setdefault("verification", {})
            verification["objective_passed"] = False
            verification["failure_type"] = "runtime_error"
            verification["infrastructure_error"] = None
        row["transient_infrastructure_rechecks"] = transient_rows
        row["stable_result_after_transient_recheck"] = (
            row.get("run_state") != "infrastructure_error"
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


def _write_summary(output: Path, rows: list[dict[str, Any]]) -> None:
    payload = {
        "generated_at": _now(),
        "task_count": len(rows),
        "completed": sum(row.get("run_state") == "completed" for row in rows),
        "infrastructure_errors": sum(
            row.get("run_state") == "infrastructure_error" for row in rows
        ),
        "build_success": sum(
            bool((row.get("build") or {}).get("passed")) for row in rows
        ),
        "objective_success": sum(
            bool((row.get("verification") or {}).get("objective_passed"))
            for row in rows
        ),
        "anticheat_clean": sum(
            (row.get("anticheat") or {}).get("verdict") == "CLEAN"
            for row in rows
        ),
        "posthoc_clean_success": sum(
            bool(row.get("posthoc_clean_success")) for row in rows
        ),
        "rows": rows,
    }
    (output / "posthoc_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "posthoc_results.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "task", "npu", "run_state", "build_passed", "objective_passed",
            "anticheat_verdict", "posthoc_clean_success",
            "full_cases", "coverage_equivalent", "infrastructure_error",
            "transient_rechecks",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "task": row["task"],
                "npu": row["npu"],
                "run_state": row.get("run_state"),
                "build_passed": (row.get("build") or {}).get("passed"),
                "objective_passed": (
                    row.get("verification") or {}).get("objective_passed"),
                "anticheat_verdict": (
                    row.get("anticheat") or {}).get("verdict"),
                "posthoc_clean_success": row.get("posthoc_clean_success"),
                "full_cases": (row.get("coverage") or {}).get("full_cases"),
                "coverage_equivalent": (
                    row.get("coverage") or {}).get("coverage_equivalent"),
                "infrastructure_error": row.get("infrastructure_error"),
                "transient_rechecks": len(
                    row.get("transient_infrastructure_rechecks") or []
                ),
            })


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
    parser.add_argument("--arm-output", type=Path, required=True)
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
    parser.add_argument("--transient-rechecks", type=int, default=2)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refuse existing post-hoc output: {args.output}")
    tasks_root = args.arm_output / "tasks"
    targets = [
        Target(
            rel=f"{path.parent.name}/{path.name}",
            treatment=path,
            source=args.source_root / path.parent.name / path.name,
        )
        for path in sorted(tasks_root.glob("level*/*"))
        if path.is_dir()
    ]
    if len(targets) != args.expected_tasks:
        raise SystemExit(
            f"expected {args.expected_tasks} treatment tasks, found {len(targets)}")
    missing_sources = [target.rel for target in targets if not target.source.is_dir()]
    if missing_sources:
        raise SystemExit(f"missing frozen source tasks: {missing_sources}")

    npus = [item.strip() for item in args.npus.split(",") if item.strip()]
    if not npus:
        raise SystemExit("at least one NPU is required")
    args.output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    partitions = [
        targets[index::len(npus)]
        for index in range(len(npus))
    ]
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
    _write_summary(args.output, rows)
    return 0 if all(row.get("run_state") == "completed" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
