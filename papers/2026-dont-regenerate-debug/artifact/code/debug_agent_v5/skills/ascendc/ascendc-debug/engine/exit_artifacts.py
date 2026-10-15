"""exit_artifacts.py — 退出产物从 events.jsonl 重建 (不靠 LLM 手写)。

REWRITE_PLAN §2.7: runner 抵达终态时调用，从事件流重放生成两份强制退出产物:
  debug_status.json — 机器可读 verdict + 成功分层 (schema 见 exit-protocols.md §7.2)
  debug_trace.md    — 4 节叙事 (入口快照 / 迭代历史 / Verdict / 产物清单，§7.1)

determinism 对论文 reproducibility 的直接贡献: 退出产物不再靠 LLM 手写 (消除漏写/编造
风险)，而是从权威事实源 events.jsonl 确定性重建。timeout/crashed 终态同样从 events 重建，
不依赖进程正常退出。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from engine.events import read_events
from engine.state import DebugState

SCHEMA_VERSION = 1

# failure_type → session_branch 标签 (exit-protocols.md §7.2: 填入口分支，后续漂移不更新)。
_BRANCH_LABEL = {
    "precision_failed": "1-P",
    "build_failed": "1-B",
    "import_failed": "1-I",
    "runtime_error": "1-R",
    "timeout": "1-T",
}

_TERMINAL_TYPES = {"session_done", "session_aborted", "session_escalated"}

# 项 12b: kernel 源文件快照 (per-file hash)，runner 在 diagnose 派发前后各取一次，
# diff 出本轮真实改动文件名 → diagnosis_summary.changed_files。globs 与 forensics
# 输入对齐 (validate_runner._FORENSICS_SRC_GLOBS)，覆盖 kernel/ 下 .cpp/.h/.hpp/.py
# + task_dir 根 model_new_ascendc.py。code_snapshot/ 不可用 (仅 Gate-A 写，引擎不调)。
_KERNEL_SRC_GLOBS = ("*.cpp", "*.h", "*.hpp", "*.py")

# 反作弊: 受保护产物 (评测基准/参考实现/算子配置)。纳入快照后，正常诊断只该改 kernel/，
# 一旦这些文件出现在 changed_files 即受保护产物被篡改的信号 (与 anticheat hash 校验互补，
# 这里给出具体文件名)。<op>.json / <op>.json.bak 需 op_name 精确定位，避免 glob *.json
# 误纳入引擎自身产物 (debug_status.json / _anticheat.json / validation_result_*.json)。
_PROTECTED_ROOT_FILES = ("model.py", "model_new_ascendc.py", "model_new_tilelang.py")


def _kernel_file_hashes(task_dir: Path, op_name: Optional[str] = None) -> dict[str, str]:
    """{相对 task_dir 的源文件路径: sha256}。读异常的文件跳过 (best-effort)。

    op_name 给定时额外纳入 <op>.json / <op>.json.bak (受保护算子配置)。
    """
    task_dir = Path(task_dir)
    out: dict[str, str] = {}
    files: list[Path] = []
    kernel_dir = task_dir / "kernel"
    if kernel_dir.is_dir():
        for pat in _KERNEL_SRC_GLOBS:
            files.extend(kernel_dir.rglob(pat))
    for name in _PROTECTED_ROOT_FILES:
        p = task_dir / name
        if p.exists():
            files.append(p)
    if op_name:
        for name in (f"{op_name}.json", f"{op_name}.json.bak"):
            p = task_dir / name
            if p.exists():
                files.append(p)
    for f in sorted(set(files), key=str):
        # 排除点文件: macOS AppleDouble (._foo.cpp，归档经 mac 中转的真实残留，见
        # hyena 快照) 及其他隐藏文件，不是真实源码，会污染 changed_files 文件名列表。
        if f.name.startswith("."):
            continue
        try:
            rel = f.relative_to(task_dir).as_posix()
            out[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
        except OSError:
            continue
    return out


def diff_changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """两次 _kernel_file_hashes 快照 diff → 新增/修改/删除的文件名 (排序)。"""
    changed = set()
    for path, h in after.items():
        if before.get(path) != h:
            changed.add(path)
    for path in before:
        if path not in after:
            changed.add(path)
    return sorted(changed)


def _iso(ts_ns: Optional[int]) -> Optional[str]:
    """事件 seq (time.time_ns()) → ISO8601 UTC。None 透传。"""
    if ts_ns is None:
        return None
    return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _duration(start_ns: Optional[int], end_ns: Optional[int]) -> str:
    if start_ns is None or end_ns is None:
        return "(未记录)"
    elapsed = max(0.0, (end_ns - start_ns) / 1e9)
    if elapsed < 10:
        return f"{elapsed:.3f}s"
    return f"{elapsed:.1f}s"


def _md_cell(value: Any) -> str:
    text = str(value) if value is not None else ""
    return text.replace("\n", " ").replace("|", "\\|")


def _action_label(action: dict) -> str:
    step = action.get("step") or action.get("name") or action.get("kind")
    return str(step or "?")


def _result_summary(result: dict) -> str:
    fields = []
    for key in (
        "loop_signal",
        "stop_reason_code",
        "failure_type",
        "claude_state",
        "api_error_status",
        "status",
        "returncode",
        "error",
    ):
        value = result.get(key)
        if value not in (None, ""):
            fields.append(f"{key}={value}")
    return ", ".join(fields) or "ok"


def _final_failure_type(events: list[dict]) -> Optional[str]:
    """最后一个 attempt_started 的 failure_type；无 attempt 则 None (skipped_* 场景)。"""
    for e in reversed(events):
        if e.get("type") == "attempt_started":
            return e.get("failure_type")
    return None


def _terminal_event(events: list[dict]) -> Optional[dict]:
    for e in reversed(events):
        if e.get("type") in _TERMINAL_TYPES:
            return e
    return None


def _latest_existing_phase8(task_dir: Path, attempts: int) -> Optional[str]:
    status_dir = Path(task_dir) / ".verify_status"
    for attempt in range(attempts - 1, -1, -1):
        path = status_dir / f"phase8_attempt{attempt}.json"
        if path.exists():
            return str(path)
    latest = status_dir / "latest.json"
    if latest.exists():
        return str(latest)
    return None


def _session_outcome(events: list[dict]) -> str:
    """从终态事件取 session_outcome；无终态事件 (异常中断) 归 crashed。"""
    term = _terminal_event(events)
    if term is None:
        return "crashed"
    if term.get("type") == "session_done":
        return term.get("session_outcome", "failed")
    if term.get("type") == "session_aborted":
        details = term.get("details") or {}
        return details.get("session_outcome", "crashed")
    return "crashed"  # escalated (debug 不用) 保守归 crashed


def _latest_validate_result(events: list[dict]) -> Optional[dict]:
    """最后一次 validate action 的 result；无则 None。"""
    for e in reversed(events):
        if e.get("type") != "action_completed":
            continue
        if (e.get("action") or {}).get("step") == "validate":
            result = e.get("result") or {}
            return result if isinstance(result, dict) else None
    return None


def _read_json(path: Path) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _objective_success_from_validation_file(task_dir: Path, attempts: int) -> Optional[bool]:
    tuning = Path(task_dir) / "precision_tuning"
    for attempt in range(attempts - 1, -1, -1):
        data = _read_json(tuning / f"validation_result_attempt_{attempt}.json")
        if data is None:
            continue
        if "correctness_passed" in data:
            return bool(data.get("correctness_passed"))
    return None


def _objective_success_from_verify_status(final_status_path: Optional[str]) -> Optional[bool]:
    if not final_status_path:
        return None
    data = _read_json(Path(final_status_path))
    if data is None:
        return None
    verify = data.get("verify") or {}
    if data.get("failure_type") == "success" and verify.get("status") == "passed":
        return True
    if data.get("failure_type") is not None or verify.get("status") is not None:
        return False
    return None


def _objective_success(
    task_dir: Path,
    events: list[dict],
    attempts: int,
    final_status_path: Optional[str],
) -> bool:
    """客观数值成功，不等价于 clean/reportable success。"""
    latest_validate = _latest_validate_result(events) or {}
    objective = latest_validate.get("objective_validation") or {}
    if isinstance(objective, dict):
        rc = objective.get("verification_exit_code")
        if isinstance(rc, int):
            return rc == 0
        if "success" in objective:
            return bool(objective.get("success")) and objective.get("failure_type") == "success"
    if isinstance(latest_validate.get("verification_exit_code"), int):
        return latest_validate["verification_exit_code"] == 0

    from_validation = _objective_success_from_validation_file(task_dir, attempts)
    if from_validation is not None:
        return from_validation
    from_status = _objective_success_from_verify_status(final_status_path)
    return bool(from_status) if from_status is not None else False


def _latest_validate_checks(events: list[dict]) -> dict:
    latest_validate = _latest_validate_result(events) or {}
    checks = latest_validate.get("checks") or {}
    return checks if isinstance(checks, dict) else {}


def _anti_cheat_pass(task_dir: Path) -> bool:
    """confirmed violation 为 false；warning 代表 unknown，不算 confirmed cheat。"""
    return _anti_cheat_summary(task_dir).get("violations", 0) == 0


def _post_anticheat_result(task_dir: Path) -> dict:
    # no_anticheat is a strict execution ablation. A copied/reset task may
    # still contain a prior run's wrapper result; never let that stale evidence
    # alter this arm's exit classification.
    if os.environ.get("ABLATE_ANTICHEAT") == "1":
        return {}
    path = Path(task_dir) / "_anticheat.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _ast_degrade_pass_from_post_anticheat(task_dir: Path) -> Optional[bool]:
    data = _post_anticheat_result(task_dir)
    ast = ((data.get("details") or {}).get("ast") or {})
    status = ast.get("status")
    if status == "pass":
        return True
    if status == "fail":
        return False
    if status in {"validator_missing", "parse_error"}:
        return None
    return None


def _ast_degrade_pass(task_dir: Path, events: list[dict]) -> Optional[bool]:
    if os.environ.get("ABLATE_ANTICHEAT") == "1":
        return None
    checks = _latest_validate_checks(events)
    if "ast_degrade_pass" not in checks:
        return _ast_degrade_pass_from_post_anticheat(task_dir)
    if checks.get("ast_validator_errored"):
        return None
    return bool(checks.get("ast_degrade_pass"))


def _probe_policy_summary(task_dir: Path) -> dict:
    tuning = Path(task_dir) / "precision_tuning"
    records = []
    try:
        paths = sorted(tuning.glob("probe_policy_attempt*.json"))
    except OSError:
        paths = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict):
            records.append({
                "attempt": record.get("attempt"),
                "policy": record.get("policy"),
                "observed_status": record.get("observed_status"),
                "policy_pass": record.get("policy_pass"),
                "source_audit_pass": record.get("source_audit_pass"),
                "source_audit_path": record.get("source_audit_path"),
                "ablation_violation": record.get(
                    "ablation_violation", False),
                "record_path": str(path),
            })
    violations = sum(record.get("policy_pass") is False for record in records)
    unknown = sum(record.get("policy_pass") is None for record in records)
    if not records or unknown:
        compliant = None if violations == 0 else False
    else:
        compliant = violations == 0
    return {
        "probe_policy_compliant": compliant,
        "probe_policy_violations": violations,
        "probe_policy_unknown": unknown,
        "probe_policy_records": records,
    }


def _reportable_success(
    session_outcome: str,
    objective_success: bool,
    anti_cheat_pass: bool,
    ast_degrade_pass: Optional[bool],
    task_dir: Path,
) -> bool:
    # KB finalize 的既有口径: cheat_history 中 warning/violation 任一存在都不是 clean。
    cheat_clean = _anti_cheat_summary(task_dir).get("total", 0) == 0
    ast_clean = ast_degrade_pass is True
    return (
        session_outcome == "success"
        and objective_success
        and anti_cheat_pass
        and cheat_clean
        and ast_clean
    )


def _success_category(
    objective_success: bool,
    anti_cheat_pass: bool,
    ast_degrade_pass: Optional[bool],
    reportable_success: bool,
) -> str:
    """退出分层枚举 (互斥且穷尽)，论文按此口径统计 clean-success 率与作弊率。

      failed                   — 客观数值未通过 (无论是否作弊，数值没过就是失败)。
      cheat_or_invalid         — 数值通过但确证作弊 (wrapper hash 破坏 / AST 退化确证 /
                                 C++ 绕过)，「假成功」不计入有效成功。
      clean_kernel_success     — 数值通过 + reportable (无任何作弊/warning，AST clean)，
                                 论文口径的真·干净 kernel 成功。
      wrapper_assisted_success — 数值通过 + 无确证作弊，但未达 reportable (AST validator
                                 异常导致未知 / 有 warning / session 未正常 success)，灰色
                                 中间态: 不算确证作弊，也不计入干净成功，单列供人工复核。

    confirmed_cheat 口径与 gates.common.run_common 一致: anti_cheat 确证 violation，或 AST
    退化确证 fail (ast_degrade_pass is False；None=validator errored 不算确证)。
    """
    if not objective_success:
        return "failed"
    confirmed_cheat = (not anti_cheat_pass) or (ast_degrade_pass is False)
    if confirmed_cheat:
        return "cheat_or_invalid"
    if reportable_success:
        return "clean_kernel_success"
    return "wrapper_assisted_success"


def build_debug_status(task_dir: Path) -> dict:
    """从 events 派生 debug_status.json verdict + 成功分层字段。"""
    events = read_events(task_dir)
    state = DebugState.load(task_dir)
    entry_ft = state.entry_failure_type
    final_ft = _final_failure_type(events) or entry_ft

    started_at = _iso(events[0].get("seq")) if events else None
    term = _terminal_event(events)
    ended_at = _iso(term.get("seq")) if term else None

    final_status_path: Optional[str] = None
    if final_ft is not None and state.total_attempts > 0:
        final_status_path = _latest_existing_phase8(task_dir, state.total_attempts)

    outcome = _session_outcome(events)
    objective_success = _objective_success(
        task_dir, events, state.total_attempts, final_status_path)
    anti_cheat_pass = _anti_cheat_pass(task_dir)
    ast_degrade_pass = _ast_degrade_pass(task_dir, events)
    reportable_success = _reportable_success(
        outcome, objective_success, anti_cheat_pass, ast_degrade_pass, task_dir)
    probe_summary = _probe_policy_summary(task_dir)

    status = {
        "schema_version": SCHEMA_VERSION,
        "session_outcome": outcome,
        "session_branch": _BRANCH_LABEL.get(entry_ft or "", None),
        "started_at": started_at,
        "ended_at": ended_at,
        "attempts_used": state.total_attempts,
        "entry_failure_type": entry_ft,
        "final_failure_type": final_ft,
        "final_verify_status_path": final_status_path,
        "objective_success": objective_success,
        "anti_cheat_pass": anti_cheat_pass,
        "ast_degrade_pass": ast_degrade_pass,
        "reportable_success": reportable_success,
        "success_category": _success_category(
            objective_success, anti_cheat_pass, ast_degrade_pass, reportable_success),
        "notes": _terminal_reason(term),
    }
    status.update(probe_summary)
    return status


def _terminal_reason(term: Optional[dict]) -> str:
    if term is None:
        return "无终态事件 (异常中断)"
    return term.get("reason") or ""


def write_debug_status(task_dir: Path) -> Path:
    status = build_debug_status(task_dir)
    path = Path(task_dir) / "debug_status.json"
    path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 诊断摘要 (6.11 文档 Tier 2 项 12b): engine 每轮 validate 后产出独立的
# diagnosis_summary_attempt_{N}.json，数据源 = agent final_response + changed_files
# (均来自 diagnose_and_fix 事件 result) + validation summary (validation_result_attempt_N
# .json)，**不依赖 precision_audit_N.md** (该文件实测缺失率 100%: Gate-A 不被引擎调用、
# 约束层不强制、无 CLI 引擎生成——正是 12b 要解决的问题)。precision_audit_N.md 若存在
# 则降级为可选 enrichment (折叠 root_cause/fix_plan)。
#
# 范式 (方案 B, runner = owner): 诊断"内容"是 agent 推理产物 (final_response)，"汇编"
# 是 engine 确定性搬运。每轮 diagnose 后即写，贴 attempt_N 语义、CONTINUE 时也留档。
# best-effort: 任何文件缺失/读异常 → None/降级，绝不影响终态产物落地。
# ---------------------------------------------------------------------------

DIAGNOSIS_SCHEMA_VERSION = 2

# 诊断摘要单 section 截断上限 (防 ROOT_CAUSE 长篇撑爆 trace；trace 是概览非全文)。
_DIAGNOSIS_SECTION_MAXLEN = 600


def _extract_audit_section(content: str, tag: str) -> Optional[str]:
    """从 audit 文本抽取 `[TAG]` section 正文 (与 gate 侧 _extract_section 同源)。

    audit 文件历史上两种 marker 格式并存: 裸 `[ROOT_CAUSE]` (SKILL.md 模板规定，多数
    产物) 与 `## [ROOT_CAUSE]` (部分产物带 markdown 标题)。裸 `[{tag}]` 是两者公共子串，
    故用裸前缀定位即可同时命中。边界: 到下一 section 头 (`\n[` 或 `\n## [`) 或
    `=== END AUDIT ===` 或文末。返回去首尾空白的正文；未找到 marker 或正文空 → None。
    """
    marker = f"[{tag}]"
    idx = content.find(marker)
    if idx == -1:
        return None
    body_start = idx + len(marker)
    rest = content[body_start:]
    # 找下一个 section 头或结束标记 (兼容裸 [ 与 ## [ 两种格式)。
    end = len(rest)
    for sentinel in ("\n[", "\n## [", "=== END AUDIT ==="):
        pos = rest.find(sentinel)
        if pos != -1:
            end = min(end, pos)
    body = rest[:end].strip()
    return body or None


def _extract_direction_metadata(content: str) -> dict[str, str]:
    """Extract structured repair-direction fields from Agent/audit text."""
    if not content:
        return {}
    out: dict[str, str] = {}
    fix_scope = _extract_audit_section(content, "FIX_PLAN") or content
    fix_match = re.search(r"\bFIX_PRECISION_[A-Z0-9_]+\b", fix_scope, re.IGNORECASE)
    if fix_match:
        out["fix_type"] = fix_match.group(0).upper()

    direction = _extract_audit_section(content, "DIRECTION_ASSESSMENT") or content
    for line in direction.splitlines():
        stripped = line.strip().lstrip("-* ")
        if "本轮是否延续上一轮方向" in stripped or "本轮是否延续" in stripped:
            value = re.split(r"[:：]", stripped, maxsplit=1)
            if len(value) == 2:
                verdict = value[1].strip()[:1]
                if verdict in ("是", "否"):
                    out["direction_verdict"] = verdict
        if "换方向理由" in stripped:
            value = re.split(r"[:：]", stripped, maxsplit=1)
            if len(value) == 2 and value[1].strip():
                reason = value[1].strip()
                if len(reason) > _DIAGNOSIS_SECTION_MAXLEN:
                    reason = reason[:_DIAGNOSIS_SECTION_MAXLEN].rstrip() + " …(截断)"
                out["direction_reason"] = reason
    return out


def _diagnose_result_for_attempt(events: list[dict], attempt: int) -> dict:
    """取指定 attempt 的 diagnose_and_fix 事件 result (final_response/changed_files 源)。

    按 attempt_started 分组定位；同 attempt 多次 diagnose (重派) 取最后一次。无 → {}。
    """
    cur: Optional[int] = None
    found: dict = {}
    for e in events:
        t = e.get("type")
        if t == "attempt_started":
            cur = e.get("attempt")
        elif t == "action_completed" and cur == attempt:
            if (e.get("action") or {}).get("step") == "diagnose_and_fix":
                found = e.get("result") or {}
    return found


def _validation_summary_for_attempt(task_dir: Path, attempt: int) -> Optional[dict]:
    """读 validation_result_attempt_{N}.json 的核心字段 (engine 每轮 validate 产出)。"""
    path = Path(task_dir) / "precision_tuning" / f"validation_result_attempt_{attempt}.json"
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    out: dict[str, Any] = {}
    for key in ("correctness_passed", "match_rate", "max_diff",
                "passed_cases", "total_cases", "first_error_lines"):
        if data.get(key) is not None:
            out[key] = data[key]
    return out or None


def _audit_enrichment(task_dir: Path, attempt: int) -> dict:
    """可选 enrichment: precision_audit_{N}.md 若存在则折叠 root_cause/fix_plan。

    audit 实测缺失率 100% (Gate-A 不被引擎调用)，故仅作锦上添花，缺失静默跳过。
    """
    path = Path(task_dir) / "precision_tuning" / f"precision_audit_{attempt}.md"
    out: dict[str, str] = {}
    try:
        if not path.exists():
            return out
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return out
    for tag, key in (("ROOT_CAUSE", "root_cause"), ("FIX_PLAN", "fix_plan")):
        sec = _extract_audit_section(content, tag)
        if sec:
            if len(sec) > _DIAGNOSIS_SECTION_MAXLEN:
                sec = sec[:_DIAGNOSIS_SECTION_MAXLEN].rstrip() + " …(截断)"
            out[key] = sec
    out.update(_extract_direction_metadata(content))
    return out


def build_diagnosis_summary(
    task_dir: Path, attempt: int, events: Optional[list[dict]] = None,
) -> Optional[dict]:
    """汇编单轮诊断摘要 (项 12b)。三源全缺 → None。

    final_response/changed_files ← diagnose_and_fix 事件 result;
    validation ← validation_result_attempt_{N}.json;
    root_cause/fix_plan ← precision_audit_{N}.md (可选 enrichment)。
    """
    task_dir = Path(task_dir)
    if events is None:
        events = read_events(task_dir)
    diag = _diagnose_result_for_attempt(events, attempt)

    out: dict[str, Any] = {
        "schema_version": DIAGNOSIS_SCHEMA_VERSION,
        "attempt": attempt,
    }
    structured = diag.get("attempt_metadata")
    if isinstance(structured, dict):
        out["attempt_metadata"] = structured
    for key in ("fix_type", "direction_verdict", "direction_reason",
                "probe_status", "probe_policy", "probe_policy_pass"):
        value = diag.get(key)
        if value is None and isinstance(structured, dict):
            value = structured.get(key)
        if value is not None:
            out[key] = value
    final_response = diag.get("final_response")
    if final_response:
        out["final_response"] = final_response
        for key, value in _extract_direction_metadata(final_response).items():
            out.setdefault(key, value)
    changed = diag.get("changed_files")
    if changed is not None:
        out["changed_files"] = changed
    validation = _validation_summary_for_attempt(task_dir, attempt)
    if validation:
        out["validation"] = validation
    for key, value in _audit_enrichment(task_dir, attempt).items():
        out.setdefault(key, value)

    # 仅有 schema_version/attempt 两个骨架键 → 本轮无任何实质内容 → None。
    if set(out) <= {"schema_version", "attempt"}:
        return None
    return out


def write_diagnosis_summary(task_dir: Path, attempt: int) -> Optional[Path]:
    """落 diagnosis_summary_attempt_{N}.json。best-effort: 失败返回 None，绝不抛异常。"""
    try:
        summary = build_diagnosis_summary(task_dir, attempt)
        if summary is None:
            return None
        tuning = Path(task_dir) / "precision_tuning"
        tuning.mkdir(parents=True, exist_ok=True)
        path = tuning / f"diagnosis_summary_attempt_{attempt}.json"
        path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
    except Exception:  # noqa: BLE001
        return None


def _load_diagnosis_summary(task_dir: Path, attempt: int) -> Optional[dict]:
    """读已归档的 diagnosis_summary_attempt_{N}.json (debug_trace §2 消费)。缺/坏 → None。"""
    path = Path(task_dir) / "precision_tuning" / f"diagnosis_summary_attempt_{attempt}.json"
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _iter_attempts(events: list[dict]) -> list[dict]:
    """把事件按 attempt 分组，并配对 action started/completed 时间戳。"""
    attempts: list[dict[str, Any]] = []
    cur: Optional[dict[str, Any]] = None
    last_seq: Optional[int] = None

    def close_current(end_ns: Optional[int]) -> None:
        if cur is not None and cur.get("ended_ns") is None:
            cur["ended_ns"] = end_ns

    for e in events:
        t = e.get("type")
        if t == "attempt_started":
            close_current(last_seq)
            cur = {
                "attempt": e.get("attempt"),
                "failure_type": e.get("failure_type"),
                "started_ns": e.get("seq"),
                "ended_ns": None,
                "last_event_ns": e.get("seq"),
                "completed": [],
                "actions": [],
                "_actions_by_id": {},
            }
            attempts.append(cur)
        elif t == "action_started" and cur is not None:
            action = e.get("action") or {}
            action_id = e.get("action_id") or f"anon_{len(cur['actions'])}"
            item = {
                "action_id": action_id,
                "step": _action_label(action),
                "started_ns": e.get("seq"),
                "ended_ns": None,
                "result": {},
            }
            cur["actions"].append(item)
            cur["_actions_by_id"][action_id] = item
            cur["last_event_ns"] = e.get("seq")
        elif t == "action_completed" and cur is not None:
            action = e.get("action") or {}
            action_id = e.get("action_id") or f"anon_{len(cur['actions'])}"
            step = (e.get("action") or {}).get("step")
            result = e.get("result") or {}
            cur["completed"].append({
                "step": step,
                "loop_signal": result.get("loop_signal"),
                "stop_reason_code": result.get("stop_reason_code"),
            })
            item = cur["_actions_by_id"].get(action_id)
            if item is None:
                item = {
                    "action_id": action_id,
                    "step": _action_label(action),
                    "started_ns": None,
                    "ended_ns": None,
                    "result": {},
                }
                cur["actions"].append(item)
                cur["_actions_by_id"][action_id] = item
            item["step"] = item.get("step") or _action_label(action)
            item["ended_ns"] = e.get("seq")
            item["result"] = result
            cur["last_event_ns"] = e.get("seq")
        elif t in _TERMINAL_TYPES:
            close_current(e.get("seq"))
            if cur is not None:
                cur["last_event_ns"] = e.get("seq")
        elif cur is not None:
            cur["last_event_ns"] = e.get("seq")
        last_seq = e.get("seq") or last_seq

    # 异常中断无终态时，保留 last_event_at，但不伪造 ended_at。
    for attempt in attempts:
        attempt.pop("_actions_by_id", None)
    return attempts


def build_debug_trace(task_dir: Path) -> str:
    """从 events 重建 debug_trace.md (4 节强制叙事)。"""
    events = read_events(task_dir)
    status = build_debug_status(task_dir)
    lines: list[str] = ["# AscendC Debug Trace", ""]

    # 1. 调用入口快照
    lines += ["## 1. 调用入口快照", ""]
    lines.append(f"- 调用时间: {status['started_at'] or '(未记录)'}")
    lines.append(f"- task_dir: {task_dir}")
    lines.append(f"- session_branch: {status['session_branch'] or '(未路由)'}")
    lines.append(f"- entry_failure_type: {status['entry_failure_type'] or '(无)'}")
    lines.append("")

    # 2. 迭代历史 (每轮一节)
    lines += ["## 2. 迭代历史", ""]
    attempts = _iter_attempts(events)
    if not attempts:
        lines.append("- (无 attempt: 入口即退出，见 Verdict)")
        lines.append("")
    for a in attempts:
        lines.append(f"### Attempt {a['attempt']}")
        lines.append(f"- failure_type: {a['failure_type']}")
        lines.append(f"- started_at: {_iso(a.get('started_ns')) or '(未记录)'}")
        if a.get("ended_ns") is not None:
            lines.append(f"- ended_at: {_iso(a.get('ended_ns')) or '(未记录)'}")
            lines.append(f"- duration: {_duration(a.get('started_ns'), a.get('ended_ns'))}")
        elif a.get("last_event_ns") is not None:
            lines.append(f"- last_event_at: {_iso(a.get('last_event_ns')) or '(未记录)'}")
        steps = ", ".join(c["step"] or "?" for c in a["completed"]) or "(无完成步骤)"
        lines.append(f"- 已完成步骤: {steps}")
        if a["actions"]:
            lines.append("")
            lines.append("| step | started_at | ended_at | duration | result |")
            lines.append("|------|------------|----------|----------|--------|")
            for action in a["actions"]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _md_cell(action.get("step")),
                            _md_cell(_iso(action.get("started_ns")) or "(未记录)"),
                            _md_cell(_iso(action.get("ended_ns")) or "(未记录)"),
                            _md_cell(_duration(action.get("started_ns"), action.get("ended_ns"))),
                            _md_cell(_result_summary(action.get("result") or {})),
                        ]
                    )
                    + " |"
                )
        # 取本轮 validate 的 loop_signal / stop_reason_code (若有)。
        for c in a["completed"]:
            if c["step"] == "validate" and c["loop_signal"]:
                extra = f"- Gate-V: loop_signal={c['loop_signal']}"
                if c["stop_reason_code"]:
                    extra += f", stop_reason_code={c['stop_reason_code']}"
                lines.append(extra)
        # 诊断摘要 (engine 每轮 validate 后归档的独立 JSON，项 12b；精度分支有，余分支降级)。
        diag = _load_diagnosis_summary(task_dir, a["attempt"])
        if diag is None:
            lines.append("- 诊断摘要: (无)")
        else:
            lines.append("- 诊断摘要:")
            if diag.get("changed_files") is not None:
                files = ", ".join(diag["changed_files"]) or "(无改动)"
                lines.append(f"  - changed_files: {files}")
            val = diag.get("validation") or {}
            if val.get("correctness_passed") is not None:
                lines.append(f"  - correctness_passed: {val['correctness_passed']}")
            if val.get("match_rate") is not None:
                lines.append(f"  - match_rate: {val['match_rate']}")
            if diag.get("root_cause"):
                lines.append(f"  - root_cause: {_md_cell(diag['root_cause'])}")
            if diag.get("fix_plan"):
                lines.append(f"  - fix_plan: {_md_cell(diag['fix_plan'])}")
            if diag.get("final_response"):
                lines.append(f"  - final_response: {_md_cell(diag['final_response'])}")
        lines.append("")

    # 3. 最终 Verdict
    lines += ["## 3. 最终 Verdict", ""]
    lines.append(f"- session_outcome: {status['session_outcome']}")
    lines.append(f"- attempts_used: {status['attempts_used']}")
    lines.append(f"- final_failure_type: {status['final_failure_type'] or '(无)'}")
    if status["notes"]:
        lines.append(f"- 原因: {status['notes']}")
    lines.append("")

    # 4. 产物清单
    lines += ["## 4. 产物清单", ""]
    lines.append("- debug_status.json")
    lines.append(f"- .debug_events/events.jsonl ({len(events)} 条事件)")
    if status["final_verify_status_path"]:
        rel = Path(status["final_verify_status_path"]).relative_to(task_dir)
        lines.append(f"- {rel}")
    lines.append("")

    return "\n".join(lines)


def write_debug_trace(task_dir: Path) -> Path:
    path = Path(task_dir) / "debug_trace.md"
    path.write_text(build_debug_trace(task_dir), encoding="utf-8")
    return path


def write_exit_artifacts(task_dir: Path) -> tuple[Path, Path]:
    """runner 终态调用: 同时落地 debug_status.json + debug_trace.md。

    顺序: 先 status (trace 引用其字段)，再 trace。
    """
    status_path = write_debug_status(task_dir)
    trace_path = write_debug_trace(task_dir)
    return status_path, trace_path


# ===========================================================================
# run_summary.json — 批跑后处理直接消费的机器摘要 (6.11 文档 Tier 2 项 11)。
#
# 设计: 独立 JSON，不混入 debug_trace.md (trace 是人读叙事)。聚合 turns / gate /
# anti_cheat / knowledge_candidate / kb_finalize / forensics，免 batch report 再扫各处日志。
#
# 成本口径只用 turns (agent_backend 透传的 num_turns)，模型无关，跨模型可比。
# best-effort: 任何子项读取失败降级为空/None，绝不影响终态。
# ===========================================================================

RUN_SUMMARY_SCHEMA_VERSION = 1


def _turns_summary(events: list[dict]) -> dict:
    """累加 diagnose_and_fix 步的 agent_turns，并按 attempt 拆分。

    事件流是唯一事实源 (与 next_action._total_agent_turns 同源)。缺失/非正整数按
    0 计 (timeout/spawn_failed 早返回路径无此字段)。
    """
    by_attempt: dict[int, int] = {}
    total = 0
    cur_attempt: Optional[int] = None
    for e in events:
        t = e.get("type")
        if t == "attempt_started":
            cur_attempt = e.get("attempt")
            by_attempt.setdefault(cur_attempt, 0)
        elif t == "action_completed":
            if (e.get("action") or {}).get("step") != "diagnose_and_fix":
                continue
            turns = (e.get("result") or {}).get("agent_turns")
            if isinstance(turns, int) and turns > 0:
                total += turns
                if cur_attempt is not None:
                    by_attempt[cur_attempt] = by_attempt.get(cur_attempt, 0) + turns
    return {
        "total": total,
        "by_attempt": {str(k): v for k, v in sorted(by_attempt.items())},
    }


def _gate_summary(events: list[dict]) -> dict:
    """各轮 validate 步的 loop_signal/stop_reason_code，及最终一轮的终判信号。"""
    by_attempt: list[dict] = []
    cur_attempt: Optional[int] = None
    for e in events:
        t = e.get("type")
        if t == "attempt_started":
            cur_attempt = e.get("attempt")
        elif t == "action_completed":
            if (e.get("action") or {}).get("step") != "validate":
                continue
            result = e.get("result") or {}
            sig = result.get("loop_signal")
            if sig is None:
                continue
            by_attempt.append({
                "attempt": cur_attempt,
                "loop_signal": sig,
                "stop_reason_code": result.get("stop_reason_code"),
            })
    final = by_attempt[-1] if by_attempt else {}
    return {
        "final_loop_signal": final.get("loop_signal"),
        "final_stop_reason_code": final.get("stop_reason_code"),
        "by_attempt": by_attempt,
    }


def _anti_cheat_summary(task_dir: Path) -> dict:
    """cheat_history.json 的 cheating_attempts 计数 (violation/warning 分列)。

    后置 anticheat.py 写出的 _anticheat.json 若为 CHEAT，也计入 violation。
    文件不存在 = 从未触发任何检查 = 全 0 (clean)。解析失败同样降级全 0。
    """
    if os.environ.get("ABLATE_ANTICHEAT") == "1":
        return {"total": 0, "violations": 0, "warnings": 0}
    path = Path(task_dir) / "precision_tuning" / "cheat_history.json"
    total = 0
    violations = 0
    warnings = 0
    if not path.exists():
        attempts = []
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            data = {}
        attempts = data.get("cheating_attempts") or []
    total = len(attempts)
    violations = sum(1 for a in attempts if a.get("severity") == "violation")
    warnings = sum(1 for a in attempts if a.get("severity") == "warning")
    post = _post_anticheat_result(task_dir)
    if post.get("verdict") == "CHEAT" and violations == 0:
        total += 1
        violations += 1
    return {"total": total, "violations": violations, "warnings": warnings}


def _forensics_summary(task_dir: Path, attempts: int) -> dict:
    """哪些 attempt 落了 forensics_report (cache 复用也会复制成当轮名，一并计入)。"""
    tuning = Path(task_dir) / "precision_tuning"
    reports = [
        a for a in range(max(attempts, 0))
        if (tuning / f"forensics_report_{a}.json").exists()
    ]
    return {"reports": reports}


def build_run_summary(task_dir: Path, status: Optional[dict] = None) -> dict:
    """从 events + 各产物文件聚合 run_summary (batch report 程序化消费)。

    status 可由调用方 (runner._terminate) 传入已构建好的 debug_status (含 kb_finalize)，
    避免重复 build；未传则现场 build。
    """
    task_dir = Path(task_dir)
    events = read_events(task_dir)
    if status is None:
        status = build_debug_status(task_dir)
    attempts_used = status.get("attempts_used") or 0
    return {
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "session_outcome": status.get("session_outcome"),
        "session_branch": status.get("session_branch"),
        "attempts_used": attempts_used,
        "success_category": status.get("success_category"),
        "turns": _turns_summary(events),
        "gate": _gate_summary(events),
        "anti_cheat": _anti_cheat_summary(task_dir),
        "knowledge_candidate": status.get("knowledge_candidate"),
        "kb_finalize": status.get("kb_finalize"),
        "forensics": _forensics_summary(task_dir, attempts_used),
    }


def write_run_summary(task_dir: Path, status: Optional[dict] = None) -> Optional[Path]:
    """落 run_summary.json。best-effort: 失败返回 None，绝不抛异常影响终态。"""
    try:
        summary = build_run_summary(task_dir, status)
        path = Path(task_dir) / "run_summary.json"
        path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
    except Exception:
        return None
