#!/usr/bin/env python3
"""Run the same frozen full-eval and no-baseline anti-cheat on both arms.

The source task is used only for the reference model and frozen case JSONL.
The candidate implementation is overlaid from the manifest task directory.
All evaluation work directories live under the isolated experiment root so the
dedicated container never needs access to the existing AscendOpGenAgent tree.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CASE_LINE_RE = re.compile(r"^case\[\d+\]: output(?:\[[^\]]+\])+:\s*(.*)$", re.M)
STATUS_RE = re.compile(r"^Status\s*:\s*(\w+)", re.M)
MODEL_JSON_LITERAL_RE = re.compile(r"[\"']model\.json[\"']")
SELF_NAMED_JSON_RE = re.compile(r"__file__.{0,256}[\"']\.json[\"']", re.S)
PRINT_LOCK = threading.Lock()
CASE_SUFFIXES = (".json", ".json.bak", ".json.full")
SOURCE_EXCLUDES = {
    "build",
    "design",
    "kernel",
    "precision_tuning",
    ".bench_baseline",
    ".debug_events",
    ".verify_logs",
    ".verify_status",
    "__pycache__",
}
SOURCE_FILE_EXCLUDES = {
    "debug_status.json",
    "debug_trace.md",
    "run_summary.json",
    "trace.md",
    "_anticheat.json",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def looks_like_cases(path: Path) -> bool:
    if not path.is_file() or not path.name.endswith(CASE_SUFFIXES):
        return False
    if path.name.startswith(("_", ".")) or path.name in {"debug_status.json", "run_summary.json"}:
        return False
    try:
        first = next(line for line in path.read_text(errors="replace").splitlines() if line.strip())
        value = json.loads(first)
    except (StopIteration, json.JSONDecodeError, OSError):
        return False
    return isinstance(value, dict) and "inputs" in value


def active_name(path: Path) -> str:
    if path.name.endswith(".json.bak"):
        return path.name[:-4]
    if path.name.endswith(".json.full"):
        return path.name[:-5]
    return path.name


def referenced_case_aliases(model_path: Path) -> set[str]:
    """Return case filenames that the frozen reference resolves implicitly.

    Several frozen tasks derive their case path from ``model.py`` itself, for
    example ``splitext(basename(__file__))[0] + '.json'``.  Their archived full
    cases retain an operator-specific filename such as ``20_Foo.json.bak``.
    Staging only the active operator filename therefore leaves the evaluator
    looking for a nonexistent ``model.json``.  Detect the two supported
    reference forms and stage an auditable alias to the same frozen full data.
    """
    try:
        source = model_path.read_text(errors="replace")
    except OSError:
        return set()
    if MODEL_JSON_LITERAL_RE.search(source) or SELF_NAMED_JSON_RE.search(source):
        return {"model.json"}
    return set()


def line_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(errors="replace").splitlines())


def embedded_case_count(model_path: Path) -> int:
    try:
        tree = ast.parse(model_path.read_text())
    except (OSError, SyntaxError):
        return 0
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name) and target.id == "INPUT_CASES":
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    return 0
                return len(value) if isinstance(value, list) else 0
    return 0


def choose_full_cases(source: Path) -> tuple[Path | None, int]:
    candidates = [path for path in source.iterdir() if looks_like_cases(path)]
    if not candidates:
        count = embedded_case_count(source / "model.py")
        if count == 0:
            raise FileNotFoundError(f"no frozen case JSONL or embedded INPUT_CASES in {source}")
        return None, count

    def rank(path: Path) -> tuple[int, int, str]:
        suffix_score = 2 if path.name.endswith((".json.bak", ".json.full")) else 1
        return line_count(path), suffix_score, path.name

    selected = max(candidates, key=rank)
    return selected, line_count(selected)


def copy_source_reference(source: Path, evaluation: Path) -> None:
    """Copy reference-side files while explicitly excluding the near-miss."""
    for item in source.iterdir():
        if item.name in SOURCE_EXCLUDES or item.name in SOURCE_FILE_EXCLUDES or item.name.startswith("_"):
            continue
        if item.name.startswith("model_new_") or looks_like_cases(item):
            continue
        destination = evaluation / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                destination,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "model_new_*"),
            )
        elif item.is_file():
            shutil.copy2(item, destination)
    if not (evaluation / "model.py").is_file():
        raise FileNotFoundError(f"reference model.py missing after source copy: {source}")


def overlay_candidate(candidate: Path, evaluation: Path) -> None:
    """Expose candidate artifacts without allowing it to replace reference/cases."""
    for item in candidate.iterdir():
        if item.name == "model.py" or looks_like_cases(item):
            continue
        if item.name.startswith(".") or item.name in {"precision_tuning", "debug_status.json", "debug_trace.md", "trace.md"}:
            continue
        destination = evaluation / item.name
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        # Keep adapter __file__ rooted in the isolated evaluation workdir,
        # and make kernel/build available there without mutating the candidate.
        if item.name == "model_new_ascendc.py" and item.is_file():
            shutil.copy2(item, destination)
        elif item.name == "kernel" and item.is_dir():
            shutil.copytree(
                item,
                destination,
                ignore=shutil.ignore_patterns("build", "__pycache__", "*.pyc"),
            )
        else:
            # Both endpoints are inside the single dedicated experiment mount.
            destination.symlink_to(item.resolve(), target_is_directory=item.is_dir())


def stage_legacy_kernel_build(candidate: Path, evaluation: Path) -> str | None:
    """Expose legacy operator_project/kernel/build as work/kernel/build."""
    legacy_build = candidate / "operator_project" / "kernel" / "build"
    if not legacy_build.is_dir() or not any(legacy_build.glob("*.so")):
        return None
    kernel_dir = evaluation / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    alias = kernel_dir / "build"
    if alias.exists() or alias.is_symlink():
        if alias.is_dir() and not alias.is_symlink():
            shutil.rmtree(alias)
        else:
            alias.unlink()
    staged_target = evaluation / "operator_project" / "kernel" / "build"
    alias.symlink_to(staged_target, target_is_directory=True)
    return str(alias)


def staged_library_paths(candidate: Path, evaluation: Path) -> list[str]:
    """Return evaluation-visible directories containing candidate libraries.

    Contract-v2 materialization keeps legacy compiled extensions below the
    candidate's ``operator_project`` symlink while many frozen adapters import
    from ``kernel/build`` relative to their source file. Because the adapter is
    staged as a symlink, ``Path(__file__).resolve()`` points back to the
    candidate mirror and cannot see the legacy build directory. Supplying the
    actual extension parent directories through PYTHONPATH/LD_LIBRARY_PATH
    preserves the candidate unchanged and makes both layouts importable.
    """
    operator_project = candidate / "operator_project"
    if not operator_project.exists():
        return []
    project_root = operator_project.resolve()
    parents: set[Path] = set()
    for library in project_root.rglob("*.so"):
        try:
            relative_parent = library.parent.relative_to(project_root)
        except ValueError:
            continue
        parents.add(evaluation / "operator_project" / relative_parent)
    return [str(item) for item in sorted(parents, key=str)]


def prepare_eval_dir(
    source: Path,
    candidate: Path,
    evaluation: Path,
) -> tuple[Path | None, int, list[str], str | None]:
    if evaluation.exists():
        shutil.rmtree(evaluation)
    evaluation.mkdir(parents=True)
    copy_source_reference(source, evaluation)
    full_cases, cases = choose_full_cases(source)
    source_case_names = {active_name(path) for path in source.iterdir() if looks_like_cases(path)}
    reference_aliases = referenced_case_aliases(source / "model.py")
    names = source_case_names | reference_aliases
    if full_cases is not None:
        full_text = full_cases.read_text(errors="replace")
        for name in sorted(names):
            if name.endswith(".json"):
                (evaluation / name).write_text(full_text)
    if candidate.is_dir():
        overlay_candidate(candidate, evaluation)
    build_alias = stage_legacy_kernel_build(candidate, evaluation)
    library_paths = staged_library_paths(candidate, evaluation)
    atomic_json(
        evaluation / "_external_eval_manifest.json",
        {
            "created_at": now(),
            "source": str(source),
            "candidate": str(candidate),
            "full_cases_source": str(full_cases or (source / "model.py")),
            "full_case_lines": cases,
            "staged_case_names": sorted(names),
            "reference_case_aliases": sorted(reference_aliases),
            "staged_library_paths": library_paths,
            "staged_kernel_build_alias": build_alias,
            "near_miss_source_overlayed": False,
        },
    )
    return full_cases, cases, library_paths, build_alias


def docker_run(
    *,
    container: str,
    npu: str,
    timeout: int,
    command: str,
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=30",
                str(timeout),
                "docker",
                "exec",
                "-e",
                f"ASCEND_RT_VISIBLE_DEVICES={npu}",
                container,
                "bash",
                "-lc",
                command,
            ],
            stdout=stdout,
            stderr=stderr,
        )
    return proc.returncode


def container_is_running(container: str) -> tuple[bool, str]:
    """Return whether the validation container is available for docker exec."""
    proc = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", container],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    detail = (proc.stdout or proc.stderr).strip()
    return proc.returncode == 0 and detail.lower() == "true", detail


def classify_full_eval(stdout: str, stderr: str, returncode: int) -> dict[str, Any]:
    comparisons = CASE_LINE_RE.findall(stdout)
    total = len(comparisons)
    matched = sum(value.strip().startswith("matched") for value in comparisons)
    status_match = STATUS_RE.search(stdout)
    status = status_match.group(1).upper() if status_match else ""
    combined = f"{stdout}\n{stderr}"
    if returncode == 124:
        failure = "timeout"
        result = "fail"
    elif status == "PASS":
        failure = "success"
        result = "pass"
    elif total:
        failure = "precision_failed"
        result = "fail"
    elif any(mark in combined for mark in ("ModuleNotFoundError", "ImportError", "No module named")):
        failure = "import_failed"
        result = "fail"
    elif returncode != 0 or any(mark in combined for mark in ("RuntimeError", "Traceback", "Exception")):
        failure = "runtime_error"
        result = "fail"
    else:
        failure = "unclassified"
        result = "error"
    return {
        "full_eval_result": result,
        "final_failure_type": failure,
        "matched_count": matched if total else None,
        "total_outputs": total if total else None,
        "match_rate_pct": round(100.0 * matched / total, 6) if total else None,
        "returncode": returncode,
    }


def parse_last_json(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _strip_cpp_comments_and_strings(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    source = re.sub(r"//[^\n]*", "", source)
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', source)


def adjudicate_direct_invoke_anticheat(task_dir: Path, official: dict[str, Any]) -> dict[str, Any]:
    """Narrow compatibility check for torch.ops + AscendC .asc launchers.

    The frozen checker intentionally remains untouched and its raw verdict is
    always retained.  An override is possible only when its *only* findings
    are the two known direct-invoke blind spots: AST type1 and
    NO_KERNEL_LAUNCH.  All other C++/AST violations remain CHEAT.
    """
    result: dict[str, Any] = {
        "applicable": False,
        "passed": False,
        "reason": "official verdict does not match the narrow compatibility case",
    }
    if official.get("verdict") != "CHEAT":
        return result
    details = official.get("details") if isinstance(official.get("details"), dict) else {}
    ast_detail = details.get("ast") if isinstance(details.get("ast"), dict) else {}
    cpp_detail = details.get("cpp") if isinstance(details.get("cpp"), dict) else {}
    violations = cpp_detail.get("violations") if isinstance(cpp_detail.get("violations"), list) else []
    other_cpp = [item for item in violations if not isinstance(item, dict) or item.get("type") != "NO_KERNEL_LAUNCH"]
    if ast_detail.get("regression_type") != 1 or other_cpp:
        return result
    result["applicable"] = True

    wrapper = task_dir / "model_new_ascendc.py"
    try:
        tree = ast.parse(wrapper.read_text())
    except (OSError, SyntaxError) as exc:
        result["reason"] = f"wrapper parse failed: {exc!r}"
        return result

    load_library = False
    op_aliases: set[str] = set()
    op_calls: list[str] = []
    forbidden_forward_nodes: list[str] = []
    forward: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward":
            forward = node
        if isinstance(node, ast.Call) and _dotted_name(node.func) == "torch.ops.load_library":
            load_library = True
    if forward is not None:
        for node in ast.walk(forward):
            if isinstance(node, ast.Assign):
                value_name = _dotted_name(node.value)
                if value_name and value_name.startswith("torch.ops."):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            op_aliases.add(target.id)
            elif isinstance(node, ast.Call):
                call_name = _dotted_name(node.func)
                if call_name and call_name.startswith("torch.ops.") and call_name != "torch.ops.load_library":
                    op_calls.append(call_name)
                elif isinstance(node.func, ast.Name) and node.func.id in op_aliases:
                    op_calls.append(node.func.id)
                elif call_name and call_name.startswith("torch.") and not call_name.startswith("torch.ops."):
                    forbidden_forward_nodes.append(call_name)
            elif isinstance(node, (ast.BinOp, ast.MatMult)):
                forbidden_forward_nodes.append(type(node).__name__)

    kernel_dir = task_dir / "kernel"
    launch_files: list[str] = []
    scanned_asc: list[str] = []
    if kernel_dir.exists():
        for path in sorted(kernel_dir.rglob("*")):
            if not path.is_file() or path.suffix not in {".asc", ".cpp", ".cc", ".cxx", ".h", ".hpp"}:
                continue
            if path.suffix == ".asc":
                scanned_asc.append(str(path.relative_to(task_dir)))
            source = _strip_cpp_comments_and_strings(path.read_text(errors="replace"))
            if re.search(r"<<<[^<>]+>>>\s*\(", source) or re.search(
                r"\b(?:aclrtLaunchKernel|ACLRT_LAUNCH_KERNEL|Launch[A-Za-z]*Kernel)\b", source
            ):
                launch_files.append(str(path.relative_to(task_dir)))

    operator_project = task_dir / "operator_project"
    libraries = sorted(str(path) for path in operator_project.resolve().rglob("*.so")) if operator_project.exists() else []
    checks = {
        "torch_ops_load_library": load_library,
        "torch_ops_forward_calls": op_calls,
        "forbidden_forward_nodes": forbidden_forward_nodes,
        "asc_files_scanned": scanned_asc,
        "kernel_launch_files": launch_files,
        "shared_libraries": libraries,
        "official_other_cpp_violations": other_cpp,
    }
    passed = bool(load_library and op_calls and launch_files and libraries and not forbidden_forward_nodes and not other_cpp)
    result.update(
        {
            "passed": passed,
            "reason": "direct-invoke runtime and launcher independently confirmed" if passed else "direct-invoke evidence incomplete",
            "checks": checks,
        }
    )
    return result


def candidate_for(record: dict[str, Any]) -> Path:
    if record["arm"] in {"regenerate", "cannbot-debug"}:
        return Path(record["cwd"]) / "candidate_task"
    return Path(record["task_dir"])


def candidate_preflight(record: dict[str, Any], candidate: Path) -> tuple[bool, str]:
    if not candidate.is_dir():
        return False, "candidate directory missing"
    if record["arm"] not in {"regenerate", "cannbot-debug"}:
        return True, "debug candidate directory present"
    adapter = candidate / "model_new_ascendc.py"
    operator_project = candidate / "operator_project"
    if not adapter.is_file():
        return False, "candidate adapter missing"
    if not operator_project.exists():
        return False, "candidate operator_project missing"
    if not any(operator_project.resolve().rglob("*.so")):
        return False, "candidate shared library missing"
    return True, f"{record['arm']} candidate materialized"


def validate_one(record: dict[str, Any], npu: str, args: argparse.Namespace) -> dict[str, Any]:
    full_path = Path(record["full_eval_path"])
    anti_path = Path(record["anticheat_path"])
    if full_path.is_file() and anti_path.is_file() and not args.force:
        return {"run_id": record["run_id"], "status": "skipped_existing"}

    final_result_path = Path(record["final_result_path"]) if record.get("final_result_path") else None
    final_result = load(final_result_path) if final_result_path and final_result_path.is_file() and final_result_path.stat().st_size else None
    final_text = json.dumps(final_result, ensure_ascii=False).lower() if isinstance(final_result, dict) else ""
    provider_aborted = isinstance(final_result, dict) and final_result.get("is_error") is True and (
        "usage policy" in final_text
        or final_result.get("api_error_status") is not None
        or any(marker in final_text for marker in ("quota", "usage limit", "overloaded", "api error"))
    )
    if provider_aborted:
        atomic_json(
            full_path,
            {
                "generated_at": now(),
                "run_id": record["run_id"],
                "full_eval_result": "not_run",
                "final_failure_type": "provider_aborted_before_candidate",
            },
        )
        atomic_json(
            anti_path,
            {
                "generated_at": now(),
                "run_id": record["run_id"],
                "verdict": "not_run",
                "reason": "provider_aborted_before_candidate",
                "mode": "no_baseline_ast_cpp_scan",
            },
        )
        return {"run_id": record["run_id"], "status": "skipped_provider_abort"}

    candidate = candidate_for(record)
    candidate_ready, candidate_note = candidate_preflight(record, candidate)
    if not candidate_ready:
        atomic_json(
            full_path,
            {
                "generated_at": now(),
                "run_id": record["run_id"],
                "operator": record["operator"],
                "arm": record["arm"],
                "trial": record.get("trial"),
                "full_eval_result": "fail",
                "final_failure_type": "candidate_missing",
                "candidate": str(candidate),
                "reason": candidate_note,
            },
        )
        atomic_json(
            anti_path,
            {
                "generated_at": now(),
                "run_id": record["run_id"],
                "verdict": "not_run",
                "reason": "candidate_missing",
                "candidate_note": candidate_note,
                "mode": "no_baseline_ast_cpp_scan",
            },
        )
        return {
            "run_id": record["run_id"],
            "status": "validated",
            "full_eval_result": "fail",
            "anticheat_verdict": "not_run",
            "candidate_note": candidate_note,
        }

    operator = record["operator"]
    source = args.source_pool / operator
    validation_root = full_path.parent
    evaluation = validation_root / "work"
    logs = validation_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        full_cases, full_case_lines, library_paths, build_alias = prepare_eval_dir(
            source, candidate, evaluation
        )
    except Exception as exc:  # noqa: BLE001 - preserve a per-run evidence file
        payload = {
            "generated_at": now(),
            "run_id": record["run_id"],
            "full_eval_result": "error",
            "final_failure_type": "harness_prepare_error",
            "error": repr(exc),
            "source": str(source),
            "candidate": str(candidate),
        }
        atomic_json(full_path, payload)
        atomic_json(anti_path, {"generated_at": now(), "verdict": "error", "error": "evaluation preparation failed"})
        return {"run_id": record["run_id"], "status": "prepare_error", "error": repr(exc)}

    verify_stdout = logs / "full_eval.stdout"
    verify_stderr = logs / "full_eval.stderr"
    library_path_value = ":".join(library_paths)
    library_exports = ""
    if library_path_value:
        quoted_library_paths = shlex.quote(library_path_value)
        library_exports = (
            f"export PYTHONPATH={quoted_library_paths}:${{PYTHONPATH:-}}; "
            f"export LD_LIBRARY_PATH={quoted_library_paths}:${{LD_LIBRARY_PATH:-}}; "
        )
    verify_command = (
        f"source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true; "
        f"{library_exports}"
        f"cd {args.repo_root}; python3 utils/verification_ascendc.py {evaluation}"
    )
    verify_rc = docker_run(
        container=args.container,
        npu=npu,
        timeout=args.timeout,
        command=verify_command,
        stdout_path=verify_stdout,
        stderr_path=verify_stderr,
    )
    verify_data = classify_full_eval(
        verify_stdout.read_text(errors="replace"),
        verify_stderr.read_text(errors="replace"),
        verify_rc,
    )
    verify_data.update(
        {
            "generated_at": now(),
            "run_id": record["run_id"],
            "operator": operator,
            "arm": record["arm"],
            "trial": record.get("trial"),
            "npu": npu,
            "candidate": str(candidate),
            "eval_task_dir": str(evaluation),
            "full_json_source": str(full_cases or (source / "model.py")),
            "full_json_cases": full_case_lines,
            "staged_library_paths": library_paths,
            "staged_kernel_build_alias": build_alias,
            "stdout": str(verify_stdout),
            "stderr": str(verify_stderr),
        }
    )
    atomic_json(full_path, verify_data)

    anti_stdout = logs / "anticheat.stdout"
    anti_stderr = logs / "anticheat.stderr"
    anti_command = (
        f"cd {args.repo_root}; python3 skills/ascendc/ascendc-debug/scripts/anticheat.py verify "
        f"{evaluation} --baseline-name .external_no_baseline --json"
    )
    anti_rc = docker_run(
        container=args.container,
        npu=npu,
        timeout=min(args.timeout, 600),
        command=anti_command,
        stdout_path=anti_stdout,
        stderr_path=anti_stderr,
    )
    official_data = parse_last_json(anti_stdout.read_text(errors="replace")) or {
        "verdict": "error",
        "parse_error": True,
    }
    official_record = dict(official_data)
    official_record.update(
        {
            "generated_at": now(),
            "run_id": record["run_id"],
            "returncode": anti_rc,
            "mode": "frozen_anticheat_no_baseline_ast_cpp_scan",
            "baseline_name": ".external_no_baseline",
            "stdout": str(anti_stdout),
            "stderr": str(anti_stderr),
        }
    )
    official_path = validation_root / "anticheat_official.json"
    atomic_json(official_path, official_record)

    compatibility = adjudicate_direct_invoke_anticheat(evaluation, official_data)
    anti_data = dict(official_data)
    official_verdict = official_data.get("verdict", "error")
    if compatibility.get("passed"):
        anti_data["verdict"] = "CLEAN"
        anti_data["reasons"] = []
    anti_data.update(
        {
            "generated_at": now(),
            "run_id": record["run_id"],
            "returncode": anti_rc,
            "mode": "frozen_anticheat_plus_direct_invoke_compatibility",
            "baseline_name": ".external_no_baseline",
            "stdout": str(anti_stdout),
            "stderr": str(anti_stderr),
            "official_verdict": official_verdict,
            "official_reasons": official_data.get("reasons", []),
            "official_result_path": str(official_path),
            "compatibility_adjudication": compatibility,
            "compatibility_override_applied": bool(compatibility.get("passed")),
        }
    )
    atomic_json(anti_path, anti_data)
    return {
        "run_id": record["run_id"],
        "status": "validated",
        "full_eval_result": verify_data["full_eval_result"],
        "anticheat_verdict": anti_data.get("verdict", "error"),
    }


def validate_partition(
    records: list[dict[str, Any]],
    npu: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Validate one partition serially on a worker permanently bound to an NPU."""
    results: list[dict[str, Any]] = []
    for record in records:
        try:
            result = validate_one(record, npu, args)
        except Exception as exc:  # noqa: BLE001
            result = {
                "run_id": record["run_id"],
                "status": "validator_exception",
                "error": repr(exc),
                "npu": npu,
            }
        results.append(result)
        with PRINT_LOCK:
            print(json.dumps(result, ensure_ascii=False), flush=True)
    return results


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=here / "run_manifest.json")
    parser.add_argument(
        "--source-pool",
        type=Path,
        default=Path("/workspace/debug_agent/outputs/precision_failed_clean33_compare_input_rebuilt_20260522_161422/cases"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("/workspace/regenerate_experiment/frozen_debug_engine"))
    parser.add_argument(
        "--container",
        default=os.environ.get("CANNBOT_VALIDATION_CONTAINER", "cannbot-regenerate-runtime"),
    )
    parser.add_argument("--npus", default="1,2,3,4,5")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--arm", choices=("regenerate", "debug", "cannbot-debug"))
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    records = load(args.manifest)["runs"]
    if args.arm:
        records = [record for record in records if record.get("arm") == args.arm]
    if args.only:
        records = [
            record
            for record in records
            if any(term in record.get("run_id", "") or term in record.get("operator", "") for term in args.only)
        ]
    if args.limit:
        records = records[: args.limit]
    container_running, container_detail = container_is_running(args.container)
    if not container_running:
        results = [
            {
                "run_id": record["run_id"],
                "status": "validator_preflight_error",
                "error": "validation container is not running",
                "container": args.container,
                "container_detail": container_detail,
            }
            for record in records
        ]
        summary = {
            "generated_at": now(),
            "requested": len(records),
            "validated": 0,
            "errors": len(records),
            "preflight_error": "validation container is not running",
            "container": args.container,
            "container_detail": container_detail,
            "results": results,
        }
        atomic_json(args.manifest.parent / "validation_run_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 1
    npus = [value.strip() for value in args.npus.split(",") if value.strip()]
    concurrency = min(args.concurrency, len(npus))
    results: list[dict[str, Any]] = []
    partitions: list[list[dict[str, Any]]] = [[] for _ in range(concurrency)]
    for index, record in enumerate(records):
        partitions[index % concurrency].append(record)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(validate_partition, partition, npus[index], args): npus[index]
            for index, partition in enumerate(partitions)
            if partition
        }
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "run_id": None,
                        "status": "validator_partition_exception",
                        "error": repr(exc),
                        "npu": futures[future],
                    }
                )
    summary = {
        "generated_at": now(),
        "requested": len(records),
        "validated": sum(
            result.get("status") in {"validated", "skipped_existing", "skipped_provider_abort"}
            for result in results
        ),
        "errors": sum(
            result.get("status") not in {"validated", "skipped_existing", "skipped_provider_abort"}
            for result in results
        ),
        "results": results,
    }
    atomic_json(args.manifest.parent / "validation_run_summary.json", summary)
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
