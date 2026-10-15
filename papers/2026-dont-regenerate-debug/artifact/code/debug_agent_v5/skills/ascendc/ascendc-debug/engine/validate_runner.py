"""Objective validation runner for engine-owned Gate-V inputs.

The debug engine owns the validate step.  A validate action must first rebuild
the AscendC extension, run the canonical verification script, classify that
result into .verify_status/phase8_attemptN.json, and then let precision_gate.py
consume those artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_ENGINE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ENGINE_DIR.parents[3]

_MISMATCH_RATIO_RE = re.compile(r"mismatch_ratio=([0-9.]+)%")
_MAX_DIFF_RE = re.compile(r"max_abs_diff=([0-9.eE+\-]+|inf|-inf|nan)", re.IGNORECASE)
_CASE_RE = re.compile(r"^case\[(\d+)\]:\s*(.+)$", re.MULTILINE)
_MODEL_JSON_EXCLUDES = {
    "debug_status.json",
    "run_summary.json",
    "experiment_manifest.json",
}

# 全量复验用 case JSONL 排除集 (与 _MODEL_JSON_EXCLUDES 合并，覆盖批跑脚本同名集)。
_CASE_JSON_EXCLUDE = _MODEL_JSON_EXCLUDES | {"round_summary.json"}


# ---------------------------------------------------------------------------
# 全量复验: case JSONL 文件选择 (从 utils/run_unified_final_verify_full_eval.py 复制，
# 行为对齐其 choose_full_json 选最全集合的逻辑)。utils/ 是批跑工具非引擎库，import
# 会耦合其 argparse/main，故外科手术复制这几个纯文件函数。
# ---------------------------------------------------------------------------
def _count_nonempty_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(errors="replace").splitlines() if line.strip())


def _normalized_case_set_digest(path: Path) -> str:
    """Hash a JSONL case multiset independent of formatting and line order."""
    normalized: list[str] = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            normalized.append(line)
        else:
            normalized.append(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
    digest = hashlib.sha256()
    for line in sorted(normalized):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _looks_like_case_jsonl(path: Path) -> bool:
    """首行为含 'inputs' 键的 JSON dict 才算 case JSONL (排除 model.json/状态文件)。"""
    if path.name.startswith(("_", ".")):
        return False
    if path.name in _CASE_JSON_EXCLUDE or path.name == "model.json":
        return False
    if not (path.name.endswith(".json") or path.name.endswith(".json.bak")
            or path.name.endswith(".json.full")):
        return False
    try:
        first = next(line for line in path.read_text(errors="replace").splitlines()
                     if line.strip())
        data = json.loads(first)
    except Exception:  # noqa: BLE001 — 非法/空文件一律判否
        return False
    return isinstance(data, dict) and "inputs" in data


def _active_json_name(path: Path) -> str:
    """全量备份文件 (<op>.json.bak / .json.full) 对应的生效文件名 <op>.json。"""
    name = path.name
    if name.endswith(".json.bak"):
        return name[:-4]
    if name.endswith(".json.full"):
        return name[:-5]
    return name


def _choose_full_json(source_dir: Path) -> tuple[Optional[Path], int]:
    """选任务目录里最全的 case JSONL: 行数→后缀分(.bak/.full=2 > .json=1)→名字。"""
    candidates = [p for p in Path(source_dir).iterdir()
                  if p.is_file() and _looks_like_case_jsonl(p)]
    if not candidates:
        return None, 0

    def rank(path: Path) -> tuple[int, int, str]:
        lines = _count_nonempty_lines(path)
        suffix_score = 2 if path.name.endswith((".json.bak", ".json.full")) else 1
        return (lines, suffix_score, path.name)

    best = max(candidates, key=rank)
    return best, _count_nonempty_lines(best)


def _full_eval_enabled() -> bool:
    """全量复验闸开关。默认开；ABLATE_FULL_EVAL=1 (或 ASCENDC_ABLATE_FULL_EVAL=1) 关。"""
    for key in ("ABLATE_FULL_EVAL", "ASCENDC_ABLATE_FULL_EVAL"):
        if os.environ.get(key) == "1":
            return False
    return True


def _full_eval_trigger_threshold() -> float:
    """轻量 match_rate ≥ 此阈值才切全量复验 (默认 99.0，对齐 nearly_success 线)。"""
    try:
        return float(os.environ.get("ASCENDC_DEBUG_FULL_EVAL_TRIGGER", "99.0"))
    except (TypeError, ValueError):
        return 99.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    prepend = [str(repo_root)]
    archive = repo_root / "archive_tasks"
    if archive.is_dir():
        prepend.insert(0, str(archive))
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = ":".join(prepend + ([old] if old else []))
    return env


def _append_header(path: Path, title: str, cmd: list[str]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {title} start={_now_iso()} =====\n")
        f.write("$ " + " ".join(cmd) + "\n")


def _append_footer(path: Path, title: str, rc: int) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(f"===== {title} end={_now_iso()} rc={rc} =====\n")


def _run_logged(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    title: str,
    timeout: Optional[float],
) -> int:
    _append_header(stdout_path, title, cmd)
    _append_header(stderr_path, title, cmd)
    with stdout_path.open("a", encoding="utf-8") as out, stderr_path.open(
        "a", encoding="utf-8"
    ) as err:
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                stdout=out,
                stderr=err,
                text=True,
                timeout=timeout,
                check=False,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired as exc:
            rc = 124
            err.write(f"\n[engine_validate] timeout after {timeout}s: {exc}\n")
    _append_footer(stdout_path, title, rc)
    _append_footer(stderr_path, title, rc)
    return rc


def _read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_first_json(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        return {}
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _public_task_json_candidates(task_dir: Path) -> list[Path]:
    """Top-level task input JSON candidates for legacy model.py that reads model.json."""
    out: list[Path] = []
    for path in sorted(Path(task_dir).glob("*.json")):
        name = path.name
        if name == "model.json":
            continue
        if name in _MODEL_JSON_EXCLUDES:
            continue
        if name.startswith(("_", ".")):
            continue
        out.append(path)
    return out


def _ensure_model_json_alias(task_dir: Path) -> dict:
    """Create model.json from the unique public task JSON when legacy model.py requires it.

    Some archived tasks have model.py hard-coded to read "model.json", while the
    directory only contains an operator-named JSON such as
    "20_FusedRopeWithQkNormAndKvCacheUpdate.json". Forensics and objective
    validation both import model.py, so normalize this input before either path.
    """
    task_dir = Path(task_dir)
    target = task_dir / "model.json"
    if target.exists():
        return {"created": False, "reason": "model_json_exists", "target": str(target)}
    if not (task_dir / "model.py").exists():
        return {"created": False, "reason": "missing_model_py", "target": str(target)}

    candidates = _public_task_json_candidates(task_dir)
    if len(candidates) != 1:
        return {
            "created": False,
            "reason": "ambiguous_or_missing_task_json",
            "target": str(target),
            "candidates": [p.name for p in candidates],
        }
    source = candidates[0]
    try:
        shutil.copy2(source, target)
    except OSError as exc:
        return {
            "created": False,
            "reason": "copy_failed",
            "target": str(target),
            "source": str(source),
            "error": str(exc),
        }

    info = {
        "created": True,
        "reason": "missing_model_json_unique_task_json",
        "target": str(target),
        "source": str(source),
    }
    try:
        tuning = task_dir / "precision_tuning"
        tuning.mkdir(parents=True, exist_ok=True)
        (tuning / "input_normalization.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return info


def _case_stats(stdout_text: str) -> tuple[int, int, float]:
    total = 0
    passed = 0
    for match in _CASE_RE.finditer(stdout_text):
        total += 1
        detail = match.group(2)
        if "matched" in detail:
            passed += 1
    rate = (passed * 100.0 / total) if total else 0.0
    return passed, total, rate


def _validation_metrics(stdout_text: str, exit_code: int) -> dict:
    ratios = [float(x) for x in _MISMATCH_RATIO_RE.findall(stdout_text)]
    max_diffs = _MAX_DIFF_RE.findall(stdout_text)
    passed_cases, total_cases, case_pass_rate = _case_stats(stdout_text)
    if ratios:
        match_rate = max(0.0, 100.0 - (sum(ratios) / len(ratios)))
        source = "avg_mismatch_ratio"
    elif exit_code == 0:
        match_rate = 100.0
        source = "verification_exit_code"
    elif total_cases:
        match_rate = case_pass_rate
        source = "case_pass_rate_fallback"
    else:
        match_rate = 0.0
        source = "no_case_data"
    return {
        "match_rate": f"{match_rate:.2f}",
        "match_rate_source": source,
        "max_diff": max_diffs[-1] if max_diffs else "0.0",
        "passed_cases": passed_cases,
        "total_cases": total_cases,
        "case_pass_rate": round(case_pass_rate, 4),
    }


_ERROR_LINE_RE = re.compile(
    r"(error|fail|mismatch|traceback|exception|assert|❌|not match)", re.IGNORECASE
)


def _extract_first_error(stdout_text: str, *, max_lines: int = 30) -> list[str]:
    """从 stdout 摘出首个错误相关行起的最多 max_lines 行，作为 first_error_lines 指针。

    修复 4a: validation_result 不再内嵌完整 stdout/stderr (数千行)，改存这段紧凑摘要 +
    stdout_path/stderr_path 指针，需要全文时 agent 按指针读盘。无错误标记时回退到
    stdout 末尾 max_lines 行 (失败信息通常在尾部)。
    """
    lines = stdout_text.splitlines()
    if not lines:
        return []
    for i, line in enumerate(lines):
        if _ERROR_LINE_RE.search(line):
            return lines[i:i + max_lines]
    return lines[-max_lines:]


def _write_validation_result(
    task_dir: Path,
    *,
    attempt: int,
    exit_code: int,
    stdout_text: str,
    stderr_text: str,
    stdout_path: Path,
    stderr_path: Path,
    full_eval: Optional[dict] = None,
) -> Path:
    tuning_dir = task_dir / "precision_tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    metrics = _validation_metrics(stdout_text, exit_code)
    payload = {
        "attempt": attempt,
        "correctness_passed": exit_code == 0,
        "match_rate": metrics["match_rate"],
        "max_diff": metrics["max_diff"],
        "match_rate_source": metrics["match_rate_source"],
        "passed_cases": metrics["passed_cases"],
        "total_cases": metrics["total_cases"],
        "case_pass_rate": metrics["case_pass_rate"],
        "source": "engine_objective_validate",
        # 修复 4a: 删 evaluate_stdout/evaluate_stderr 全文字段，改紧凑摘要 + 路径指针。
        "first_error_lines": _extract_first_error(stdout_text),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    # 全量复验产物 (§2.1)。默认 None → 不写键，旧 schema 不变 (向后兼容已跑产物/UT)。
    if full_eval is not None:
        payload["full_eval"] = full_eval
    path = tuning_dir / f"validation_result_attempt_{attempt}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _full_eval_state_path(task_dir: Path) -> Path:
    return task_dir / "precision_tuning" / ".full_eval_state.json"


def _upsert_full_eval_state(task_dir: Path, *, attempt: int, match_rate: float) -> None:
    """跨轮全量复验收敛态 (best-effort)。供 branch_precision 判"是否已做过/有无改善"。

    attempts_done: 已做过全量复验的 attempt 列表 (去重升序)；
    best_full_match_rate: 历史最高全量 match_rate (判无改善)；
    last_full_match_rate / last_attempt: 最近一次。
    """
    path = _full_eval_state_path(task_dir)
    state = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                state = loaded
        except (ValueError, OSError):
            state = {}
    done = set(state.get("attempts_done", []))
    done.add(attempt)
    prev_best = state.get("best_full_match_rate")
    best = match_rate if prev_best is None else max(float(prev_best), match_rate)
    state.update({
        "attempts_done": sorted(done),
        "best_full_match_rate": best,
        "last_full_match_rate": match_rate,
        "last_attempt": attempt,
    })
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _run_full_eval(
    task_dir: Path,
    *,
    attempt: int,
    repo_root: Path,
    env: dict[str, str],
    clean_build: bool,  # noqa: ARG001 — 全量不重 build (.so 已在)，保留签名对齐
    timeout: Optional[float],
) -> Optional[dict]:
    """轻量达阈值后用最全 case 集复验一次 (§2.1)。

    物理机制 (对齐 utils/run_unified_final_verify_full_eval.py): 备份当前生效 <op>.json →
    用 .json.bak/.json.full 覆盖 → 跑同一 verification (不重 build) → finally 恢复。
    返回 None = 无独立备份集合 (only_py 算子 / 无 .bak)，下游 graceful 回退轻量口径。
    case 数相同时比较规范化 case-set 摘要：内容不同仍复验；完全等价则返回显式
    coverage_equivalent 记录，不重复执行相同用例。
    """
    task_dir = Path(task_dir)
    full_json, full_lines = _choose_full_json(task_dir)
    if full_json is None:
        return None
    active_name = _active_json_name(full_json)
    active_path = task_dir / active_name
    # full_json 即生效文件本身 = 无独立备份，无法形成额外覆盖证据。
    cur_lines = _count_nonempty_lines(active_path) if active_path.exists() else 0
    if active_path == full_json or full_lines < cur_lines:
        return None
    active_digest = (
        _normalized_case_set_digest(active_path) if active_path.exists() else None
    )
    active_aliases: list[Path] = []
    model_json = task_dir / "model.json"
    if (
        model_json.is_file()
        and model_json != active_path
        and active_path.is_file()
        and _count_nonempty_lines(model_json) == cur_lines
        and _normalized_case_set_digest(model_json) == active_digest
    ):
        active_aliases.append(model_json)
    full_digest = _normalized_case_set_digest(full_json)
    if full_lines == cur_lines and active_digest == full_digest:
        result = {
            "ran": False,
            "coverage_equivalent": True,
            "equivalence_basis": "normalized_case_set_sha256",
            "active_json": active_path.name,
            "active_json_aliases": [path.name for path in active_aliases],
            "active_json_cases": cur_lines,
            "active_case_set_sha256": active_digest,
            "full_json_source": full_json.name,
            "full_json_cases": full_lines,
            "full_case_set_sha256": full_digest,
        }
        try:
            (
                task_dir
                / "precision_tuning"
                / f"validation_result_attempt_{attempt}_full.json"
            ).write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass
        return result

    # 备份用独立后缀，避开 _choose_full_json 只认的 .json/.json.bak/.json.full。
    backup_path = task_dir / f"{active_name}.lightweight_bak_attempt{attempt}"
    logs_dir = task_dir / ".verify_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"phase8_attempt{attempt}_full.stdout"
    stderr_path = logs_dir / f"phase8_attempt{attempt}_full.stderr"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    active_targets = [active_path, *active_aliases]
    original_bytes = {
        path: path.read_bytes() if path.exists() else None
        for path in active_targets
    }
    try:
        backup_path.write_bytes(original_bytes[active_path] or b"")
        full_bytes = full_json.read_bytes()
        for path in active_targets:
            path.write_bytes(full_bytes)
        verify_cmd = [
            sys.executable,
            str(repo_root / "utils" / "verification_ascendc.py"),
            str(task_dir),
        ]
        rc = _run_logged(
            verify_cmd,
            cwd=repo_root,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            title="verification_ascendc_full",
            timeout=timeout,
        )
    finally:
        for path, content in original_bytes.items():
            try:
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(content)
            except OSError:
                pass
        try:
            backup_path.unlink()
        except OSError:
            pass

    stdout_text = _read(stdout_path)
    metrics = _validation_metrics(stdout_text, rc)
    total_cases = metrics["total_cases"]
    crashed = rc != 0 and total_cases == 0  # 跑不起来 vs 精度失败 的区分
    match_rate = float(metrics["match_rate"])
    result = {
        "ran": True,
        "coverage_equivalent": False,
        "equivalence_basis": "normalized_case_set_sha256",
        "active_json": active_path.name,
        "active_json_aliases": [path.name for path in active_aliases],
        "active_json_cases": cur_lines,
        "active_case_set_sha256": active_digest,
        "full_json_source": full_json.name,
        "full_json_cases": full_lines,
        "full_case_set_sha256": full_digest,
        "match_rate": match_rate,
        "passed_cases": metrics["passed_cases"],
        "total_cases": total_cases,
        "correctness_passed": rc == 0,
        "crashed": crashed,
        "stdout_path": str(stdout_path),
        "first_error_lines": _extract_first_error(stdout_text) if rc != 0 else [],
    }
    # 单独落盘 + 更新跨轮收敛态。
    try:
        (task_dir / "precision_tuning" / f"validation_result_attempt_{attempt}_full.json"
         ).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    _upsert_full_eval_state(task_dir, attempt=attempt, match_rate=match_rate)
    return result


def run_objective_validation(
    task_dir: Path,
    *,
    attempt: int,
    repo_root: Optional[Path] = None,
    timeout: Optional[float] = None,
) -> dict:
    """Run build + verification + classify for phase8 attempt artifacts."""
    task_dir = Path(task_dir).resolve()
    repo_root = Path(repo_root or _REPO_ROOT).resolve()
    input_normalization = _ensure_model_json_alias(task_dir)
    logs_dir = task_dir / ".verify_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"phase8_attempt{attempt}.stdout"
    stderr_path = logs_dir / f"phase8_attempt{attempt}.stderr"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    env = _repo_env(repo_root)
    soc_version = env.get("ASCENDC_SOC_VERSION", "Ascend910B3")
    clean_build = env.get("ASCENDC_DEBUG_CLEAN_BUILD", "1") != "0"

    build_cmd = [
        sys.executable,
        str(repo_root / "utils" / "build_ascendc.py"),
        str(task_dir),
        "-v",
        soc_version,
    ]
    if clean_build:
        build_cmd.append("--clean")
    build_rc = _run_logged(
        build_cmd,
        cwd=repo_root,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        title="build_ascendc",
        timeout=timeout,
    )

    verification_ran = False
    verify_rc = build_rc
    if build_rc == 0:
        verification_ran = True
        verify_cmd = [
            sys.executable,
            str(repo_root / "utils" / "verification_ascendc.py"),
            str(task_dir),
        ]
        verify_rc = _run_logged(
            verify_cmd,
            cwd=repo_root,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            title="verification_ascendc",
            timeout=timeout,
        )

    stdout_text = _read(stdout_path)
    stderr_text = _read(stderr_path)

    # 全量复验闸 (§2.1): 轻量 verify 全过 (rc=0) 或近通过 (match_rate≥触发阈值) 时，
    # 用最全 case 集复验一次。结果落 validation_result 的 full_eval 子字段供 Gate-V 判信号。
    # ABLATE_FULL_EVAL=1 关闸；无更全集合 (only_py 算子) 自动 graceful no-op (返回 None)。
    full_eval = None
    if _full_eval_enabled() and verification_ran:
        lm = _validation_metrics(stdout_text, verify_rc)
        if verify_rc == 0 or float(lm["match_rate"]) >= _full_eval_trigger_threshold():
            full_eval = _run_full_eval(
                task_dir,
                attempt=attempt,
                repo_root=repo_root,
                env=env,
                clean_build=clean_build,
                timeout=timeout,
            )

    validation_result_path = _write_validation_result(
        task_dir,
        attempt=attempt,
        exit_code=verify_rc,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        full_eval=full_eval,
    )

    classify_cmd = [
        sys.executable,
        str(repo_root / "utils" / "classify_verify_result.py"),
        "--exit-code",
        str(verify_rc),
        "--stdout-path",
        str(stdout_path),
        "--stderr-path",
        str(stderr_path),
        "--task-dir",
        str(task_dir),
        "--phase",
        "8",
        "--attempt",
        str(attempt),
        "--write-status",
    ]
    classify_proc = subprocess.run(
        classify_cmd,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    status = _extract_first_json(classify_proc.stdout)
    status_path = task_dir / ".verify_status" / f"phase8_attempt{attempt}.json"

    return {
        "success": classify_proc.returncode == 0,
        "build_exit_code": build_rc,
        "verification_exit_code": verify_rc,
        "verification_ran": verification_ran,
        "classify_exit_code": classify_proc.returncode,
        "failure_type": status.get("failure_type"),
        "import_subtype": status.get("import_subtype"),
        "failed_step": status.get("failed_step"),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "verify_status_path": str(status_path),
        "validation_result_path": str(validation_result_path),
        "input_normalization": input_normalization,
        "full_eval": full_eval,
    }


_FORENSICS_SRC_GLOBS = ("*.cpp", "*.h", "*.hpp", "*.py")


def _forensics_input_hash(task_dir: Path) -> str:
    """对取证输入 (kernel/ 下源文件 + model_new_ascendc.py) 算合并内容 hash。

    建议A (6.11 文档 Tier 1): forensics 子进程 (含 OperatorExecutor) 昂贵 (上限 1800s)，
    若本轮 kernel 源码与上轮完全一致 (agent 修复未落盘 / 仅改了无关文件)，重跑取证产出
    必然相同，可直接复用上轮 report。hash 覆盖会影响取证结果的全部源文件: kernel/ 下
    .cpp/.h/.hpp/.py 与 task_dir 根的 model_new_ascendc.py。文件名也并入 hash，使增删
    文件 (而非仅改内容) 同样判为变化。
    """
    h = hashlib.sha256()
    files: list[Path] = []
    kernel_dir = task_dir / "kernel"
    if kernel_dir.is_dir():
        for pat in _FORENSICS_SRC_GLOBS:
            files.extend(kernel_dir.rglob(pat))
    model_new = task_dir / "model_new_ascendc.py"
    if model_new.exists():
        files.append(model_new)
    model_json = task_dir / "model.json"
    if model_json.exists():
        files.append(model_json)
    files.extend(_public_task_json_candidates(task_dir))
    for f in sorted(set(files), key=lambda p: str(p)):
        try:
            h.update(str(f.relative_to(task_dir)).encode("utf-8"))
            h.update(b"\0")
            h.update(f.read_bytes())
            h.update(b"\0")
        except OSError:
            continue
    return h.hexdigest()


def _forensics_cache_load(task_dir: Path) -> dict:
    path = task_dir / "precision_tuning" / ".forensics_cache.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (ValueError, OSError):
        return {}


def _forensics_cache_save(task_dir: Path, *, src_hash: str, attempt: int) -> None:
    path = task_dir / "precision_tuning" / ".forensics_cache.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"src_hash": src_hash, "attempt": attempt}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def _kernel_build_ready(task_dir: Path) -> bool:
    build_dir = task_dir / "kernel" / "build"
    if not build_dir.is_dir():
        return False
    try:
        return any(build_dir.rglob("*.so"))
    except OSError:
        return False


def _run_forensics_prebuild(
    task_dir: Path,
    *,
    attempt: int,
    repo_root: Path,
    timeout: Optional[float],
) -> dict:
    logs_dir = task_dir / ".verify_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"forensics_build_attempt{attempt}.stdout"
    stderr_path = logs_dir / f"forensics_build_attempt{attempt}.stderr"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    env = _repo_env(repo_root)
    soc_version = env.get("ASCENDC_SOC_VERSION", "Ascend910B3")
    clean_build = env.get("ASCENDC_DEBUG_FORENSICS_CLEAN_BUILD", "1") != "0"
    build_cmd = [
        sys.executable,
        str(repo_root / "utils" / "build_ascendc.py"),
        str(task_dir),
        "-v",
        soc_version,
    ]
    if clean_build:
        build_cmd.append("--clean")

    rc = _run_logged(
        build_cmd,
        cwd=repo_root,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        title="forensics_prebuild_ascendc",
        timeout=timeout,
    )
    stdout_text = _read(stdout_path)
    stderr_text = _read(stderr_path)
    return {
        "exit_code": rc,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_tail": stdout_text[-2000:],
        "stderr_tail": stderr_text[-2000:],
        "first_error_lines": (
            _extract_first_error(stdout_text + "\n" + stderr_text) if rc != 0 else []
        ),
    }


def _write_forensics_unavailable_report(
    task_dir: Path,
    *,
    attempt: int,
    status: str,
    primary_hint: str,
    error: str,
    build_result: Optional[dict] = None,
) -> Path:
    tuning_dir = task_dir / "precision_tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "2.0",
        "op_name": task_dir.name,
        "attempt": attempt,
        "status": status,
        "error": error,
        "outputs": [],
        "primary_hint": primary_hint,
        "primary_confidence": 0.0,
        "primary_evidence": error,
        "source": "engine_forensics_prebuild",
        "generated_at": _now_iso(),
    }
    if build_result is not None:
        payload["build_exit_code"] = build_result.get("exit_code")
        payload["stdout_path"] = build_result.get("stdout_path")
        payload["stderr_path"] = build_result.get("stderr_path")
        payload["first_error_lines"] = build_result.get("first_error_lines", [])
        payload["stderr_tail"] = build_result.get("stderr_tail", "")
    path = tuning_dir / f"forensics_report_{attempt}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _latest_rollback_from_attempt(task_dir: Path) -> Optional[dict]:
    """Return rollback evidence from either the legacy event or replayable action."""
    try:
        from engine.best_rollback import latest_rollback_record
        return latest_rollback_record(task_dir)
    except (ImportError, OSError, ValueError):
        return None


def _copy_forensics_report(
    source: Path,
    report: Path,
    *,
    attempt: int,
    provenance: dict,
) -> bool:
    """复制缓存报告并把内部 attempt/provenance 改成当前轮。

    Gate-F 同时校验文件名和 JSON 内的 attempt。字节复制历史报告会导致
    ``forensics_report_N.json`` 内仍是旧 attempt，真实流水线会连续重试并停止。
    """
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return False
        payload["attempt"] = attempt
        payload.update(provenance)
        report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError):
        return False
    return report.exists()


def _reuse_forensics_after_rollback(
    task_dir: Path, attempt: int, report: Path
) -> Optional[dict]:
    """回滚后复用与 best 源码相符的 forensics report。

    每轮顺序是 forensics -> agent 修改 -> validate，因此 ``forensics_report_N`` 描述的是
    attempt N 修改前的源码，而 current_best(attempt N) 保存的是修改后的源码。与 best
    源码对应的是下一轮开始时生成的 ``forensics_report_{N+1}``，不是 report_N。

    best report 缺失/复制失败 → 返回 None 回退正常执行 (向正确性倾斜)。
    """
    rb = _latest_rollback_from_attempt(task_dir)
    if rb is None or rb.get("from_attempt") != attempt - 1:
        return None
    best_attempt = rb.get("best_attempt")
    if best_attempt is None:
        return None
    try:
        report_attempt = int(best_attempt) + 1
    except (TypeError, ValueError):
        return None
    best_report = task_dir / "precision_tuning" / f"forensics_report_{report_attempt}.json"
    if not best_report.exists():
        return None
    if not _copy_forensics_report(
        best_report,
        report,
        attempt=attempt,
        provenance={
            "reused_after_rollback": True,
            "reused_from_attempt": report_attempt,
            "reused_best_attempt": best_attempt,
        },
    ):
        return None
    return {
        "success": True,
        "exit_code": 0,
        "report_path": str(report),
        "error": None,
        "cached": True,
        "reused_after_rollback": True,
        "reused_from_attempt": report_attempt,
        "reused_best_attempt": best_attempt,
        "cache_hit": True,
        "reuse_kind": "rollback",
        "forensics_reused": True,
        "forensics_executed": False,
        "forensics_completed": True,
        "prebuild_executed": False,
        "build_skipped": True,
        "build_skip_reason": "rollback_forensics_reuse",
    }


def _existing_build_failure_evidence(task_dir: Path) -> Optional[dict]:
    """Load the compile log already produced by objective validation.

    A build-failed branch cannot run precision forensics until the source
    compiles.  Rebuilding unchanged source only repeats a deterministic error,
    so the existing build log is the correct diagnostic input for the Agent.
    """
    status_path = task_dir / ".verify_status" / "latest.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if status.get("failure_type") != "build_failed":
        return None
    raw_log_path = status.get("log_path")
    if not raw_log_path:
        return None
    log_path = Path(raw_log_path)
    if not log_path.is_absolute():
        log_path = task_dir / log_path
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return {
        "exit_code": status.get("verification_exit_code", 1),
        "stdout_path": None,
        "stderr_path": str(log_path),
        "stdout_tail": "",
        "stderr_tail": log_text[-2000:],
        "first_error_lines": _extract_first_error(log_text),
        "source_status_path": str(status_path),
    }


_DETERMINISTIC_RUNTIME_ERROR_SIGNATURES = (
    "aicore exception",
    "acl stream synchronize failed",
    "rtdevicesynchronize",
    "runtime result = 507",
    "error code:507",
)


def _degrade_deterministic_runtime_forensics(
    report: Path,
    *,
    stderr_tail: Optional[str],
) -> Optional[dict]:
    """Turn a device runtime crash report into usable diagnostic evidence.

    Precision forensics executes the same broken kernel as objective validation.
    An AICore/ACL execution exception is therefore deterministic evidence for the
    runtime-fix Agent, not a reason to execute the unchanged kernel three times.
    Missing/malformed reports and unclassified child failures remain retryable.
    """
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("status") != "error":
        return None

    evidence_text = "\n".join(
        str(value)
        for value in (
            payload.get("error"),
            payload.get("traceback"),
            payload.get("primary_evidence"),
            stderr_tail,
        )
        if value
    )
    lowered = evidence_text.lower()
    classified_device_fault = (
        "aicore exception" in lowered
        or (
            "acl stream synchronize failed" in lowered
            and (
                "runtime result = 507" in lowered
                or "error code:507" in lowered
            )
        )
    )
    if not classified_device_fault:
        return None
    matched = [
        signature
        for signature in _DETERMINISTIC_RUNTIME_ERROR_SIGNATURES
        if signature in lowered
    ]

    payload.update({
        "forensics_degraded": True,
        "forensics_unavailable": True,
        "unavailable_reason": "runtime_error",
        "diagnostic_evidence_kind": "runtime_error_log",
        "proceed_to_agent": True,
        "diagnostic_signatures": matched,
        "degraded_at": _now_iso(),
    })
    try:
        report.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        return None
    return {
        "error": payload.get("error") or stderr_tail,
        "traceback": payload.get("traceback"),
        "diagnostic_signatures": matched,
        "source_report_path": str(report),
    }


def run_forensics(
    task_dir: Path,
    *,
    attempt: int,
    failure_type: Optional[str] = None,
    repo_root: Optional[Path] = None,
    timeout: Optional[float] = None,
) -> dict:
    """Engine-owned: 主动调用 precision_forensics.py 产出 forensics_report。

    修复 1 (6.11 文档): forensics step 此前从不由 engine 编排，磁盘上的 report 是
    agent 在 diagnose 内自发产出、时机不受 Gate-F 控制。此函数让 engine 在 forensics
    step 主动产出 report，使 Gate-F 当轮可读到。

    与 run_objective_validation 对齐: 同样 cwd=repo_root + _repo_env(PYTHONPATH) 以便
    precision_forensics.py 的 _forensics_child 子进程能 import 任务模块。timeout 默认
    None——脚本内部 OperatorExecutor 子进程自带 1800s 上限，外层不再叠加更短的硬杀，
    避免误杀合法长取证后触发无谓重试。
    """
    task_dir = Path(task_dir).resolve()
    repo_root = Path(repo_root or _REPO_ROOT).resolve()
    input_normalization = _ensure_model_json_alias(task_dir)
    report = task_dir / "precision_tuning" / f"forensics_report_{attempt}.json"

    build_evidence = (
        _existing_build_failure_evidence(task_dir)
        if failure_type == "build_failed"
        else None
    )
    if build_evidence is not None:
        error = "using existing objective-validation build log as diagnostic evidence"
        report_path = _write_forensics_unavailable_report(
            task_dir,
            attempt=attempt,
            status="build_failed",
            primary_hint="build_error",
            error=error,
            build_result=build_evidence,
        )
        return {
            "success": True,
            "exit_code": build_evidence.get("exit_code"),
            "report_path": str(report_path),
            "error": error,
            "cached": False,
            "forensics_unavailable": True,
            "unavailable_reason": "build_failed",
            "forensics_degraded": True,
            "diagnostic_evidence_kind": "existing_build_log",
            "proceed_to_agent": True,
            "build_result": build_evidence,
            "cache_hit": False,
            "reuse_kind": None,
            "forensics_reused": False,
            "forensics_executed": False,
            "forensics_completed": False,
            "prebuild_executed": False,
            "build_skipped": True,
            "build_skip_reason": "existing_build_failure_evidence",
            "input_normalization": input_normalization,
        }

    # 问题 7: 回滚后复用。若本轮 (attempt) 紧跟一次 kernel 回滚 (best_rollback 已把源码恢复
    # 成 best)，则本轮取证输入 == best validate 后的源码，重跑必产同结果，且 build/ 残留的是
    # 改坏代码的 .so (与回滚后源码不一致，重跑会用错 .so 失真)。故复用 best 后下一轮开始
    # 时生成、与 best 源码对应的 forensics_report，
    # 跳过 prebuild + OperatorExecutor (省编译+取证)。best report 缺失 → 回退正常执行。
    reused = _reuse_forensics_after_rollback(task_dir, attempt, report)
    if reused is not None:
        reused["input_normalization"] = input_normalization
        return reused

    # 建议A: staleness 缓存。当前取证输入 (kernel 源码 + model_new) hash 命中上轮缓存，
    # 且上轮 report 仍在 → 复用，省一次昂贵子进程 (上限 1800s)。复用时把上轮 report 复制
    # 成当前 attempt 名 (Gate-F 按 forensics_report_{attempt}.json 精确读取)。
    # env ASCENDC_DEBUG_FORENSICS_NO_CACHE=1 禁用。源码或缓存任一异常 → 回退到重跑 (向
    # 正确性倾斜: 宁可多跑一次，不复用可能过期的取证)。
    use_cache = os.environ.get("ASCENDC_DEBUG_FORENSICS_NO_CACHE", "0") != "1"
    src_hash = _forensics_input_hash(task_dir) if use_cache else ""
    if use_cache and src_hash:
        cache = _forensics_cache_load(task_dir)
        prev_attempt = cache.get("attempt")
        if cache.get("src_hash") == src_hash and prev_attempt is not None:
            prev_report = task_dir / "precision_tuning" / f"forensics_report_{prev_attempt}.json"
            if prev_report.exists():
                if _copy_forensics_report(
                    prev_report,
                    report,
                    attempt=attempt,
                    provenance={
                        "cached": True,
                        "cached_from_attempt": prev_attempt,
                    },
                ):
                    _forensics_cache_save(task_dir, src_hash=src_hash, attempt=attempt)
                    return {
                        "success": True,
                        "exit_code": 0,
                        "report_path": str(report),
                        "error": None,
                        "cached": True,
                        "cached_from_attempt": prev_attempt,
                        "cache_hit": True,
                        "reuse_kind": "input_hash",
                        "forensics_reused": True,
                        "forensics_executed": False,
                        "forensics_completed": True,
                        "prebuild_executed": False,
                        "build_skipped": True,
                        "build_skip_reason": "forensics_cache_hit",
                        "input_normalization": input_normalization,
                    }

    prebuild_result = None
    prebuild_enabled = os.environ.get("ASCENDC_DEBUG_FORENSICS_PREBUILD", "1") != "0"
    build_ready_before = _kernel_build_ready(task_dir)
    prebuild_executed = False
    if prebuild_enabled:
        if not build_ready_before:
            prebuild_executed = True
            prebuild_result = _run_forensics_prebuild(
                task_dir,
                attempt=attempt,
                repo_root=repo_root,
                timeout=timeout,
            )
            if int(prebuild_result.get("exit_code", 1)) != 0:
                error = (
                    "forensics prebuild failed; treating build log as diagnostic "
                    "evidence and continuing to agent"
                )
                report_path = _write_forensics_unavailable_report(
                    task_dir,
                    attempt=attempt,
                    status="build_failed",
                    primary_hint="build_error",
                    error=error,
                    build_result=prebuild_result,
                )
                return {
                    "success": True,
                    "exit_code": prebuild_result.get("exit_code"),
                    "report_path": str(report_path),
                    "error": error,
                    "cached": False,
                    "forensics_unavailable": True,
                    "unavailable_reason": "build_failed",
                    "forensics_degraded": True,
                    "diagnostic_evidence_kind": "prebuild_log",
                    "proceed_to_agent": True,
                    "build_result": prebuild_result,
                    "cache_hit": False,
                    "reuse_kind": None,
                    "forensics_reused": False,
                    "forensics_executed": False,
                    "forensics_completed": False,
                    "prebuild_executed": True,
                    "build_skipped": False,
                    "build_skip_reason": None,
                    "input_normalization": input_normalization,
                }

    script = repo_root / "skills" / "ascendc" / "ascendc-debug" / "scripts" / "precision_forensics.py"
    # 第一位置参 = task_name (目录名)，不是 op_name；--task-dir 给绝对路径。
    cmd = [sys.executable, str(script), task_dir.name,
           "--attempt", str(attempt), "--task-dir", str(task_dir)]
    env = _repo_env(repo_root)
    try:
        proc = subprocess.run(
            cmd, cwd=repo_root, env=env,
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        exit_code = proc.returncode
        stderr_tail = proc.stderr[-500:] if proc.returncode != 0 else None
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stderr_tail = f"forensics timeout after {timeout}s: {exc}"
    success = exit_code == 0 and report.exists()
    runtime_evidence = None
    if not success and failure_type == "runtime_error":
        runtime_evidence = _degrade_deterministic_runtime_forensics(
            report,
            stderr_tail=stderr_tail,
        )
    if runtime_evidence is not None:
        return {
            "success": True,
            "exit_code": exit_code,
            "report_path": str(report),
            "error": runtime_evidence.get("error"),
            "cached": False,
            "forensics_unavailable": True,
            "unavailable_reason": "runtime_error",
            "forensics_degraded": True,
            "diagnostic_evidence_kind": "runtime_error_log",
            "proceed_to_agent": True,
            "runtime_result": runtime_evidence,
            "build_result": prebuild_result,
            "cache_hit": False,
            "reuse_kind": None,
            "forensics_reused": False,
            "forensics_executed": True,
            "forensics_completed": False,
            "prebuild_executed": prebuild_executed,
            "build_skipped": not prebuild_executed,
            "build_skip_reason": (
                "build_artifact_ready" if build_ready_before
                else "prebuild_disabled" if not prebuild_enabled
                else None
            ),
            "input_normalization": input_normalization,
        }
    # 重跑成功 → 记录本轮 hash，使下轮 (源码未变时) 命中缓存。
    if success and use_cache and src_hash:
        _forensics_cache_save(task_dir, src_hash=src_hash, attempt=attempt)
    return {
        "success": success,
        "exit_code": exit_code,
        "report_path": str(report) if report.exists() else None,
        "error": stderr_tail,
        "cached": False,
        "build_result": prebuild_result,
        "cache_hit": False,
        "reuse_kind": None,
        "forensics_reused": False,
        "forensics_executed": True,
        "forensics_completed": success,
        "prebuild_executed": prebuild_executed,
        "build_skipped": not prebuild_executed,
        "build_skip_reason": (
            "build_artifact_ready" if build_ready_before
            else "prebuild_disabled" if not prebuild_enabled
            else None
        ),
        "input_normalization": input_normalization,
    }
