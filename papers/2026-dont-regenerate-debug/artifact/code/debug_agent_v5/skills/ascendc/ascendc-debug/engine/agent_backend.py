"""agent_backend.py — diagnose_and_fix 的真实执行体 (方案 C 的唯一 spawn 点)。

引擎 next_action 里 forensics/validate 是确定性 py_action (runner 直接跑 precision_gate.py)，
**唯一需要拉起 agent 的是 diagnose_and_fix**。本模块把该 Action 翻译成一次 `claude --bare -p`
调用 (命令形态对齐批处理脚本的 Claude CLI 调用约定)，做「单个 attempt 的
诊断 + 改 kernel」，改完进程退出，控制权交回 runner (runner 随后跑 Gate-V 客观判定这一轮)。

方案 C 的核心边界 (防 reward-hack):
  - backend 只判断「claude 这一轮进程是否正常跑完」(读 --output-format json 的
    is_error/stop_reason/api_error_status)，返回的 success **不是「修好了」**。
  - 「修没修好」由 runner 之后的 validate gate 客观判定，agent 自报不算数。
  - 收窄 prompt 显式禁止 agent 自己跑 MAX_ATTEMPTS 循环 / 读 loop_signal 自决 / 写
    debug_status.json (这些编排职责已收归引擎)。

可注入: claude 命令通过 runner 参数构造，subprocess 调用点 (_run) 可被 UT monkeypatch，
UT 不真跑 claude。
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from engine.types import Action

_PROVIDER_LIMIT_STATUSES = {401, 402, 403, 429}

# 项 12b: agent final response 截断上限 (诊断摘要的主数据源——claude --output-format
# json 的顶层 `result` 字段是 agent 本轮的最终诊断陈述，实测含完整根因/证据链/修复说明，
# 通常 1~2KB)。透传给 next 层做 diagnosis_summary，超限截断防撑爆事件流/产物。
_FINAL_RESPONSE_MAXLEN = 4000

# cc 脚本默认值 (utils/run_ascendc_debug_batch_cc.sh L44-47)，保持一致。
_DEFAULT_CLAUDE_BIN = "claude"
_DEFAULT_ALLOWED_TOOLS = "Bash,Read,Write,Edit,Glob,Grep,Skill"
_DEFAULT_AGENT = "ascendc-debug-agent-constructive"  # AAAI 主力 (cc 默认是 discovery，这里默认 constructive)
_EFFORT_ENV = "ASCENDC_DEBUG_AGENT_EFFORT"

_ATTEMPT_METADATA_TAG = "ENGINE_ATTEMPT_METADATA"
_ATTEMPT_METADATA_KEYS = {
    "fix_type",
    "direction_verdict",
    "direction_reason",
    "probe_status",
    "probe_reason",
    "kb_used_ids",
}
_DIRECTION_VERDICTS = {"initial", "continue", "switch", "uncertain"}
_PROBE_STATUSES = {"skipped", "executed", "not_applicable", "unknown"}

# 单轮约束覆盖层: 方案 C 下 agent 只做一个 attempt，编排归引擎。
# 不改 agent spec 本体 (留 Step 7)，在 prompt 末尾追加此约束覆盖自驱散文。
_SINGLE_ROUND_CONSTRAINT = """

═══════════════════════════════════════════════════════════════
【本次调用 = 单个 attempt，由确定性引擎驱动，务必遵守以下覆盖约束】
═══════════════════════════════════════════════════════════════
本次调用只负责**一个 attempt 的诊断 + 修复**，循环与终态由外层引擎掌控:
- 禁止自己跑 MAX_ATTEMPTS 循环；禁止读 Gate loop_signal 自行决定继续/停止/跳 Step5/Step6。
- 禁止写 debug_status.json / debug_trace.md (退出产物由引擎从事件流确定性重建)。
- 【受保护文件，严禁以任何方式修改/删除/重命名/覆写】: model.py、参考实现 wrapper
  (model_new_ascendc.py / model_new_tilelang.py)、算子配置 <op>.json 及其 .json.bak 备份、
  .verify_status/ 目录下任何文件。这些是客观评测的基准与产物，改动它们等同作弊 (会被反作弊
  hash 校验与 C++/AST 扫描捕获并判定 cheat_detected，本轮成果作废)。只允许修改
  {task_dir}/kernel/ 下的 .cpp/.h/.hpp 算子源码。
- 不要把自测/自评结果写成最终结论；最终 build/eval/classify/precision_gate 由引擎
  在本轮结束后统一执行。诊断中如确有必要，可以运行局部 build/eval/forensics 命令，
  但其结果只作为本轮根因分析证据，不作为流程终态。
- 读取上下文采用按需原则: 不要全文读取 SKILL.md、branch_precision.py、CANN API 文档、
  build/eval 日志或 .json 大文件，优先用 Grep/Read offset+limit/sed/head/tail 定位
  当前根因相关片段；只有诊断必须时才扩大读取范围。
- 知识库检索由 engine 在本轮 diagnose 前确定性执行，并已在 prompt 中注入摘要；
  不要重复运行 precision_knowledge.py search。若没有摘要，按当前 forensics/code 证据继续。
- Bash 命令应有明确诊断目的；长输出请重定向到文件并查看摘要/关键片段，避免完整递归
  grep、完整编译/验证输出反复进入上下文。
- 修复保持聚焦、最小且可解释；可以按根因需要修改 {task_dir}/kernel/ 下相关文件。
  完成本轮预期修改后停止，把验证交回引擎。
本次只做: 按当前 failure_type 走对应分支方法论 (精度走 Phase A→B→C 等) → 诊断根因 →
最小化修改 {task_dir}/kernel/ 下文件 → 完成即停 (不要追加任何收尾动作)。
如需参考 SKILL.md，只读取/遵循当前 failure_type 对应的诊断/修复方法论片段。
不要读取或执行 SKILL.md 中的全局调度、MAX_ATTEMPTS/loop_signal、Gate 验证、
Step5/Step6/Step7、退出协议、归档、批处理、report 生成等流程章节；这些由 engine 负责。
═══════════════════════════════════════════════════════════════
"""

_ATTEMPT_METADATA_CONSTRAINT = """

═══════════════════════════════════════════════════════════════
【必须输出的结构化 attempt 元数据】
═══════════════════════════════════════════════════════════════
最终回复末尾必须逐行输出以下 section；它由 engine 解析并写入事件流，不替代正常诊断总结:
[ENGINE_ATTEMPT_METADATA]
fix_type: <本轮主要修复方向的简短稳定名称>
direction_verdict: <initial|continue|switch|uncertain>
direction_reason: <为何延续/切换方向；首轮说明初始依据>
probe_status: <skipped|executed|not_applicable>
probe_reason: <与本轮 engine probe policy 一致的理由或实测探针说明>
kb_used_ids: <实际采用的 KB_ID，逗号分隔；未采用写 none>

不得把“检索到”写成“已采用”；只有修复确实使用了对应知识时才列入 kb_used_ids。
═══════════════════════════════════════════════════════════════
"""

_PROBE_POLICY_CONSTRAINT = """

═══════════════════════════════════════════════════════════════
【Engine probe policy】
═══════════════════════════════════════════════════════════════
- policy: {policy}
- reason: {reason}
- primary_hint: {primary_hint}
{instruction}
本轮最终回复中的 probe_status/probe_reason 必须如实记录实际行为；不得把执行过的探针写成跳过。
═══════════════════════════════════════════════════════════════
"""

# 消融注入: 关闭插桩 (no_probe / baseline arm)。ABLATE_PROBE=1 时在 prompt 末尾追加，
# 指示 agent 跳过 SKILL.md Sub-step 2.6 插桩定位，[L5_PROBE] 写消融跳过理由版
# (复用 SKILL.md:824/506 "状态: 跳过（理由: ...）" 口径，section 仍存在不触发 Gate-A 缺失)。
_NOPROBE_CONSTRAINT = """

═══════════════════════════════════════════════════════════════
【消融开关 ABLATE_PROBE: 本次调用禁用插桩定位 (L5 Probe)】
═══════════════════════════════════════════════════════════════
- 跳过 SKILL.md Sub-step 2.6 插桩定位 (printf/DumpTensor 二分搜索)，不得在 kernel 中
  插入任何调试打印探针。
- 仍须产出 [L5_PROBE] section，但写跳过版:
  「状态: 跳过（理由: 消融实验关闭插桩 ABLATE_PROBE=1）」。
- 根因定位仅依据 forensics/code 静态证据 + Phase A/B 既有分析，不依赖 L5 实测中间值。
═══════════════════════════════════════════════════════════════
"""

_NOKB_CONSTRAINT = """

═══════════════════════════════════════════════════════════════
【消融开关 ABLATE_KB: 本次调用禁用知识库】
═══════════════════════════════════════════════════════════════
- 不得读取、检索或引用任何 precision knowledge base 及历史知识条目。
- 只能依据当前 task 的源码、验证日志和本轮客观产物诊断。
- kb_used_ids 必须写 none。
═══════════════════════════════════════════════════════════════
"""

_NOFORENSICS_CONSTRAINT = """

═══════════════════════════════════════════════════════════════
【消融开关 ABLATE_FORENSICS: 本次调用禁用前置取证】
═══════════════════════════════════════════════════════════════
- 不得读取或复用 precision_tuning/forensics_report_*.json。
- 不得自行运行 precision_forensics.py 重建被消融的证据。
- 只能依据当前源码与原始验证/编译/运行日志诊断。
═══════════════════════════════════════════════════════════════
"""

_NORECOVERY_CONSTRAINT = """

═══════════════════════════════════════════════════════════════
【消融开关 ABLATE_RECOVERY: 本次调用禁用恢复脚手架】
═══════════════════════════════════════════════════════════════
- 不读取或使用 current_best、rollback、tuning_directions 等跨轮恢复/方向历史。
- 只依据当前 task 状态与当前 attempt 的客观证据完成一次修复。
═══════════════════════════════════════════════════════════════
"""

_PROBE_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cuh"}
_PROBE_SOURCE_PATTERNS = {
    "printf_call": re.compile(r"(?<![A-Za-z0-9_])printf\s*\("),
    "ascendc_printf": re.compile(r"\bAscendCPrintf\s*\("),
    "dump_tensor": re.compile(r"\bDumpTensor\s*\("),
    "dump_data": re.compile(r"\b(?:DataDump|DumpData)\s*\("),
}


def _snapshot_probe_sources(task_dir: Path) -> dict:
    """Count durable probe APIs in editable kernel sources."""
    kernel_dir = task_dir / "kernel"
    files: dict[str, dict[str, int]] = {}
    if kernel_dir.is_dir():
        for path in sorted(kernel_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _PROBE_SOURCE_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            counts = {
                name: len(pattern.findall(text))
                for name, pattern in _PROBE_SOURCE_PATTERNS.items()
            }
            if any(counts.values()):
                files[str(path.relative_to(task_dir))] = counts
    return {"files": files}


def _audit_probe_sources(task_dir: Path, attempt: int, before: dict,
                         after: dict) -> dict:
    """Persist an independent before/after audit for newly added probe calls."""
    additions = []
    before_files = before.get("files", {})
    after_files = after.get("files", {})
    for probe_name in _PROBE_SOURCE_PATTERNS:
        old_total = sum(
            int(counts.get(probe_name, 0))
            for counts in before_files.values()
        )
        new_total = sum(
            int(counts.get(probe_name, 0))
            for counts in after_files.values()
        )
        delta = new_total - old_total
        if delta > 0:
            additions.append({
                "probe": probe_name,
                "added_occurrences": delta,
                "files_with_increased_count": [
                    file_name
                    for file_name, counts in sorted(after_files.items())
                    if int(counts.get(probe_name, 0))
                    > int(before_files.get(file_name, {}).get(probe_name, 0))
                ],
            })
    record = {
        "schema_version": 1,
        "attempt": attempt,
        "before": before,
        "after": after,
        "new_probe_occurrences": additions,
        "passed": not additions,
        "limitation": (
            "Detects probe calls left in source after the Agent returns; "
            "temporary probes removed before return and count-balanced "
            "replacement of an existing probe are not observable."
        ),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = task_dir / "precision_tuning" / f"probe_source_audit_attempt{attempt}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    record["record_path"] = str(path)
    return record


def _attach_probe_observability(result: dict, task_dir: Path, attempt: int,
                                policy: dict, policy_path: Path,
                                source_before: dict) -> dict:
    """Attach probe contract and independent source evidence on every exit."""
    source_audit = _audit_probe_sources(
        task_dir, attempt, source_before, _snapshot_probe_sources(task_dir))
    probe_record = _finalize_probe_policy(
        task_dir, policy, result.get("final_response"),
        result.get("attempt_metadata") or {}, source_audit)
    result["probe_policy"] = probe_record.get("policy")
    result["probe_status"] = probe_record.get("observed_status")
    result["probe_policy_pass"] = probe_record.get("policy_pass")
    result["probe_record_path"] = str(
        probe_record.get("record_path") or policy_path)
    result["probe_source_audit_path"] = source_audit.get("record_path")
    result["ablation_violation"] = probe_record.get(
        "ablation_violation", False)
    return result


def _stable_kb_id(title: str) -> str:
    """Return the same deterministic fallback ID used by precision_knowledge.py."""
    normalized = " ".join(str(title).strip().lower().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"kb-{digest}"


def _read_forensics_primary_hint(task_dir: Path, attempt: int) -> str:
    if os.environ.get("ABLATE_FORENSICS") == "1":
        return "unknown"
    path = task_dir / "precision_tuning" / f"forensics_report_{attempt}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unknown"
    hint = data.get("primary_hint") if isinstance(data, dict) else None
    return str(hint).strip() if hint else "unknown"


def _probe_policy(task_dir: Path, failure_type: str, attempt: int) -> dict:
    """Build the engine-owned probe policy for one Agent attempt.

    The policy is deliberately separate from the Agent's self-report.  It records
    what the engine required before spawn; post-run evidence is appended later.
    """
    primary_hint = _read_forensics_primary_hint(task_dir, attempt)
    if os.environ.get("ABLATE_PROBE") == "1":
        policy = "skip"
        reason = "ablation_probe_disabled"
        instruction = "不得执行任何 L5 probe/插桩；[L5_PROBE] 记录消融跳过。"
    elif failure_type != "precision_failed":
        policy = "not_applicable"
        reason = f"failure_type={failure_type}"
        instruction = "本分支不要求 precision L5 probe，probe_status 写 not_applicable。"
    elif attempt == 0 and primary_hint != "nan_inf_contamination":
        policy = "skip"
        reason = "first_round_fast_path"
        instruction = (
            "首轮禁止执行 printf/DumpTensor/GM dump 等 probe 或临时插桩；"
            "仅使用 forensics/code 静态证据，probe_status 写 skipped。"
        )
    elif attempt == 0:
        policy = "required"
        reason = "nan_inf_contamination_exception"
        instruction = "首轮属于 NaN/Inf 例外，必须执行定位首现点的 L5 probe。"
    else:
        policy = "conditional"
        reason = "post_first_round_default_probe"
        instruction = (
            "默认执行 L5 probe；仅在 SKILL.md 的四项跳过条件全部有证据时才可跳过。"
        )
    return {
        "schema_version": 1,
        "attempt": attempt,
        "failure_type": failure_type,
        "policy": policy,
        "reason": reason,
        "primary_hint": primary_hint,
        "instruction": instruction,
        "enforcement": "prompt_contract_with_posthoc_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_probe_policy(task_dir: Path, policy: dict) -> Path:
    tuning = task_dir / "precision_tuning"
    tuning.mkdir(parents=True, exist_ok=True)
    path = tuning / f"probe_policy_attempt{policy['attempt']}.json"
    path.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _parse_attempt_metadata(final_response: Optional[str]) -> dict:
    """Parse the required metadata footer without depending on Chinese prose."""
    if not isinstance(final_response, str) or not final_response.strip():
        return {}
    marker = f"[{_ATTEMPT_METADATA_TAG}]"
    start = final_response.rfind(marker)
    if start < 0:
        return {}
    section = final_response[start + len(marker):]
    out: dict[str, object] = {}
    for raw_line in section.splitlines():
        line = raw_line.strip().lstrip("-* ")
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key not in _ATTEMPT_METADATA_KEYS or not value:
            continue
        if key == "direction_verdict":
            normalized = value.lower()
            if normalized in _DIRECTION_VERDICTS:
                out[key] = normalized
        elif key == "probe_status":
            normalized = value.lower()
            if normalized in _PROBE_STATUSES:
                out[key] = normalized
        elif key == "kb_used_ids":
            if value.lower() in {"none", "null", "n/a", "无", "未使用"}:
                out[key] = []
            else:
                ids = re.findall(r"\bkb-[0-9a-f]{12}\b", value.lower())
                out[key] = list(dict.fromkeys(ids))
        else:
            out[key] = value[:600]
    return out


def _infer_probe_status(final_response: Optional[str], metadata: dict) -> tuple[str, str]:
    declared = metadata.get("probe_status")
    if declared in _PROBE_STATUSES:
        return str(declared), "structured_metadata"
    text = final_response or ""
    if re.search(r"L5_PROBE.{0,120}(?:状态\s*[:：]?\s*跳过|首轮快路径|未插桩)",
                 text, re.IGNORECASE | re.DOTALL):
        return "skipped", "final_response_heuristic"
    if re.search(
        r"(?:L5_PROBE.{0,120}(?:已执行|实测)|(?:执行|增加|插入|使用).{0,30}(?:探针|插桩)|"
        r"(?:GM\s*dump|DumpTensor|printf).{0,30}(?:探针|插桩|定位))",
        text,
        re.IGNORECASE | re.DOTALL,
    ):
        return "executed", "final_response_heuristic"
    return "unknown", "no_evidence"


def _finalize_probe_policy(task_dir: Path, policy: dict,
                           final_response: Optional[str], metadata: dict,
                           source_audit: Optional[dict] = None) -> dict:
    status, source = _infer_probe_status(final_response, metadata)
    expected = policy.get("policy")
    if expected == "skip":
        passed = True if status == "skipped" else False if status == "executed" else None
    elif expected == "required":
        passed = True if status == "executed" else False if status == "skipped" else None
    elif expected == "not_applicable":
        passed = status in {"not_applicable", "skipped"} if status != "unknown" else None
    else:
        passed = status in {"executed", "skipped"} if status != "unknown" else None
    source_violation = bool(
        source_audit
        and expected == "skip"
        and source_audit.get("passed") is False
    )
    if source_violation:
        passed = False
    record = dict(policy)
    record.update({
        "observed_status": status,
        "observation_source": source,
        "policy_pass": passed,
        "metadata_complete": _ATTEMPT_METADATA_KEYS.issubset(metadata),
        "source_audit_pass": (
            source_audit.get("passed") if source_audit is not None else None
        ),
        "source_audit_path": (
            source_audit.get("record_path") if source_audit is not None else None
        ),
        "ablation_violation": source_violation,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    path = _write_probe_policy(task_dir, record)
    record["record_path"] = str(path)
    return record


def _cheat_warning(task_dir: Path) -> str:
    """读 cheat_history.json，若上一轮被判作弊则生成警告注入下一轮 prompt (修复4 场景A)。

    仅取最近一条 violation (非 warning——validator 异常不该警告 agent 它作弊)。无则返回空串。
    """
    if os.environ.get("ANTICHEAT_DETECT_ONLY") == "1":
        return ""
    path = task_dir / "precision_tuning" / "cheat_history.json"
    if not path.exists():
        return ""
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return ""
    violations = [e for e in history.get("cheating_attempts", [])
                  if e.get("severity") != "warning"]
    if not violations:
        return ""
    last = violations[-1]
    return (
        f"\n⚠️ 上一轮 (attempt {last.get('attempt')}) 的修复被检测为作弊:\n"
        f"  - 类型: {last.get('cheat_type')}\n"
        f"  - 要求: {last.get('instruction')}\n"
        f"  - 本轮必须用真实 AscendC kernel 实现，禁止用 torch 原生算子绕过。\n"
    )


def _rollback_warning(task_dir: Path, attempt: int) -> str:
    """问题 7: 若上一轮 (attempt-1) 触发了 kernel 回滚，注入提示到本轮 prompt。

    回滚 = 连续无改善后 kernel 已被恢复到 current_best。告知 agent 当前代码是历史最佳
    (非上一轮改坏的版本)，且最近方向未见效，须换思路——避免回滚后重走死路 (决策 4)。
    失败方向明细由 agent 按 SKILL.md 指令自读 tuning_directions.json。
    """
    try:
        from engine.best_rollback import latest_rollback_record
        last = latest_rollback_record(task_dir)
    except Exception:  # noqa: BLE001
        return ""
    if last is None or last.get("from_attempt") != attempt - 1:
        return ""
    direction_summary = _failed_direction_summary(task_dir, attempt - 1)
    return (
        f"\n🔄 上一轮 (attempt {attempt - 1}) 连续无改善，kernel 已回滚到历史最佳"
        f" (attempt {last.get('best_attempt')}，case_pass_rate={last.get('best_case_pass_rate')})。\n"
        f"  - 当前 kernel/ 是目前最好的版本，不是上一轮改坏的版本。\n"
        f"  - 最近几轮的修复方向未见效，本轮请换一个方向，勿重复。\n"
        f"{direction_summary}"
        f"  - 完整历史见 tuning_directions.json。\n"
    )


def _failed_direction_summary(task_dir: Path, through_attempt: int) -> str:
    """Render recent structured regressed/stagnant directions into the prompt."""
    path = task_dir / "precision_tuning" / "tuning_directions.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    entries = [
        entry for entry in (data.get("entries") or [])
        if isinstance(entry, dict)
        and isinstance(entry.get("attempt"), int)
        and entry["attempt"] <= through_attempt
        and entry.get("outcome") in ("regressed", "stagnant")
        and any(entry.get(key) for key in (
            "fix_type", "direction_verdict", "direction_reason"))
    ][-3:]
    if not entries:
        return ""
    lines = ["  - 已确认的近期无效方向:"]
    for entry in entries:
        details = [f"outcome={entry.get('outcome')}"]
        for key in ("fix_type", "direction_verdict", "direction_reason"):
            if entry.get(key):
                details.append(f"{key}={entry[key]}")
        lines.append(f"    - attempt {entry['attempt']}: " + ", ".join(details))
    return "\n".join(lines) + "\n"


def _direction_history_context(task_dir: Path, attempt: int) -> str:
    """Inject recent engine-owned direction/outcome history on every later attempt."""
    if attempt <= 0:
        return ""
    path = task_dir / "precision_tuning" / "tuning_directions.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    entries = [
        entry for entry in (data.get("entries") or [])
        if isinstance(entry, dict)
        and isinstance(entry.get("attempt"), int)
        and entry["attempt"] < attempt
    ][-3:]
    if not entries:
        return ""
    lines = ["\n【引擎注入的近期修复方向与客观结果】"]
    for entry in entries:
        details = []
        for key in ("fix_type", "direction_verdict", "direction_reason",
                    "outcome", "case_pass_rate", "improvement_ratio"):
            if entry.get(key) is not None:
                details.append(f"{key}={entry[key]}")
        lines.append(
            f"- attempt {entry['attempt']}: "
            + (", ".join(details) if details else "无结构化方向元数据")
        )
    lines.append("- 本轮不得重复已确认 stagnant/regressed 的方向；如延续，必须给出新的证据。")
    return "\n".join(lines) + "\n"


def _knowledge_search_context(task_dir: Path, attempt: int) -> str:
    """Return a compact KB retrieval summary injected by engine, if present."""
    if os.environ.get("ABLATE_KB") == "1":
        return ""
    path = task_dir / "precision_tuning" / "knowledge_search_log.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return ""
    if not isinstance(data, list):
        return ""
    entries = [
        e for e in data
        if isinstance(e, dict) and e.get("attempt") == attempt
    ][-2:]
    if not entries:
        return ""

    lines = [
        "\n【引擎知识库检索摘要】",
        f"- log_path: {path}",
        "- 你应参考这些命中项，但根因仍必须由 forensics/code 证据验证。",
    ]
    for entry in entries:
        query = entry.get("query") or {}
        lines.append(
            f"- call_index={entry.get('call_index')} "
            f"op_type={query.get('op_type')} pattern={query.get('pattern')} "
            f"position={query.get('position')} "
            f"matched={entry.get('matched_count')} checklist={entry.get('checklist_count')} "
            f"fallback={entry.get('fallback_to_full_load')}"
        )
        for item in (entry.get("match_reasons") or [])[:3]:
            title = item.get("title")
            kb_id = item.get("knowledge_id") or _stable_kb_id(title or "")
            score = item.get("score")
            reason = ",".join(str(x) for x in (item.get("reason") or [])[:4])
            if title:
                lines.append(
                    f"  - [KB_ID={kb_id}] {title} (score={score}, reason={reason})")
        if not (entry.get("match_reasons") or []) and entry.get("top_titles"):
            top_ids = entry.get("top_ids") or []
            for idx, title in enumerate((entry.get("top_titles") or [])[:3]):
                kb_id = top_ids[idx] if idx < len(top_ids) else _stable_kb_id(title)
                lines.append(f"  - [KB_ID={kb_id}] {title}")
    lines.append(
        "- 最终回复必须在 [ENGINE_ATTEMPT_METADATA] 的 kb_used_ids 中列出实际采用的 KB_ID；"
        "未采用任何命中项写 none。"
    )
    return "\n".join(lines) + "\n"


def _audit_context(task_dir: Path, attempt: int) -> str:
    """Inject the engine-owned Gate-A evidence packet into the Agent prompt."""
    if os.environ.get("ABLATE_GATE_A") == "1":
        return ""
    path = (
        task_dir
        / "precision_tuning"
        / f"audit_context_attempt_{attempt}.json"
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(value, dict):
        return ""
    return (
        "\n[ENGINE_GATE_A_AUDIT]\n【Gate-A 引擎预审证据】\n"
        + json.dumps(value, ensure_ascii=False, indent=2)[:6000]
        + "\n请显式确认或推翻 primary_hint，并让修改方向与该证据一致。\n"
    )


def _build_prompt(task_dir: Path, op_name: str, failure_type: str,
                  attempt: int, npu: Optional[str]) -> str:
    """构造收窄到「单个 attempt」的 diagnose prompt。

    agent spec 的 System Prompt 由 --agent 加载 (含分支方法论)，这里只传本轮参数 +
    单轮约束覆盖层。对齐 cc 脚本 PROMPT_TEMPLATE 的 "debug {task_dir} npu=" 入口形态。
    """
    npu_line = f"npu={npu}" if npu is not None else ""
    head = (
        f"debug {task_dir} {npu_line}\n\n"
        f"本轮上下文 (由引擎注入，勿自行改动):\n"
        f"  task_dir: {task_dir}\n"
        f"  op_name: {op_name}\n"
        f"  failure_type: {failure_type}\n"
        f"  attempt: {attempt}\n"
    )
    probe_policy = _probe_policy(task_dir, failure_type, attempt)
    probe_constraint = _PROBE_POLICY_CONSTRAINT.format(**probe_policy)
    recovery_context = ""
    if os.environ.get("ABLATE_RECOVERY") != "1":
        recovery_context = (
            _rollback_warning(task_dir, attempt)
            + _direction_history_context(task_dir, attempt)
        )
    return (head + _cheat_warning(task_dir)
            + recovery_context
            + _knowledge_search_context(task_dir, attempt)
            + _audit_context(task_dir, attempt)
            + _SINGLE_ROUND_CONSTRAINT.replace("{task_dir}", str(task_dir))
            + probe_constraint
            + (_NOPROBE_CONSTRAINT if os.environ.get("ABLATE_PROBE") == "1" else "")
            + (_NOKB_CONSTRAINT if os.environ.get("ABLATE_KB") == "1" else "")
            + (_NOFORENSICS_CONSTRAINT
               if os.environ.get("ABLATE_FORENSICS") == "1" else "")
            + (_NORECOVERY_CONSTRAINT if os.environ.get("ABLATE_RECOVERY") == "1" else "")
            + _ATTEMPT_METADATA_CONSTRAINT)


def _classify_claude_result(result_file: Path) -> dict:
    """读 claude --output-format json 结果，判定本轮进程是否正常完成。

    success=进程正常跑完本轮 (非「修好了」——后者由 runner 的 Gate-V 判定)。
    对齐 cc 脚本 read_claude_state / read_fatal_claude_error 的字段语义。
    """
    if not result_file.exists():
        return {"success": False, "claude_state": "missing_claude_result",
                "error": f"claude 未产出结果文件 {result_file.name}"}
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        return {"success": False, "claude_state": "invalid_claude_result",
                "error": f"claude 结果文件无法解析: {e}"}

    api_status = data.get("api_error_status")
    stop_reason = data.get("stop_reason")
    subtype = data.get("subtype")
    is_error = bool(data.get("is_error"))
    # api_error_status 更具体且是 fatal 判定依据 (对齐 cc 脚本 read_fatal_claude_error)，
    # 优先于泛化的 is_error。真实 claude API 错误两者常同时出现。
    if api_status is not None:
        state = f"api_error_{api_status}"
    elif is_error and subtype == "error_max_turns":
        state = "max_turns_exceeded"
    elif is_error:
        state = "claude_error"
    elif stop_reason == "pause_turn":
        state = "claude_pause_turn"  # 本轮未自然结束 (引擎可据此判断重试/收尾)
    else:
        state = "ok"
    result = {
        "success": state == "ok",
        "claude_state": state,
        "stop_reason": stop_reason,
        "subtype": subtype,
        "api_error_status": api_status,
        "error": None if state == "ok" else f"claude_state={state}",
    }
    # 透传本轮 agentic turn 数 (claude --output-format json 顶层 num_turns)。
    # next_action 跨 attempt 累加做任务级 turns 硬闸；非整数/缺失置 None (累加层按 0 计)。
    turns = data.get("num_turns")
    result["agent_turns"] = turns if isinstance(turns, int) else None
    # 项 12b: 透传 agent 本轮最终诊断陈述 (顶层 `result` 字段)，作为 diagnosis_summary
    # 的主数据源 (取代 100% 缺失的 precision_audit_N.md)。
    # 关键: 仅 state=="ok" 才提取——真实产物核实 (artifacts 20 样例) 表明 is_error/
    # stop_sequence 态的 `result` 是错误串 ("API Error: 400 ... token limit")，非诊断
    # 内容；error 态恒置 None，避免把 API 错误噪声塞进诊断摘要。非 str/空同样 None；超限截断。
    final = data.get("result")
    if state == "ok" and isinstance(final, str) and final.strip():
        final = final.strip()
        metadata = _parse_attempt_metadata(final)
        result["attempt_metadata"] = metadata
        for key in ("fix_type", "direction_verdict", "direction_reason"):
            if metadata.get(key) is not None:
                result[key] = metadata[key]
        if len(final) > _FINAL_RESPONSE_MAXLEN:
            final = final[:_FINAL_RESPONSE_MAXLEN].rstrip() + " …(截断)"
        result["final_response"] = final
    else:
        result["final_response"] = None
        result["attempt_metadata"] = {}
    if api_status in _PROVIDER_LIMIT_STATUSES:
        result["fatal"] = True
        result["provider_error"] = True
    return result


def _write_kb_usage_trace(task_dir: Path, attempt: int,
                          final_response: Optional[str],
                          attempt_metadata: Optional[dict] = None) -> None:
    """Persist retrieved/injected/declared-used KB evidence separately.

    ``used_ids`` is based on the explicit structured footer.  Exact-title matches
    remain as a compatibility heuristic and are never promoted to declared use.
    """
    log_path = task_dir / "precision_tuning" / "knowledge_search_log.json"
    trace_path = task_dir / "precision_tuning" / "kb_usage_trace.json"
    kb_disabled = os.environ.get("ABLATE_KB") == "1"
    try:
        data = (
            []
            if kb_disabled
            else json.loads(log_path.read_text(encoding="utf-8"))
            if log_path.exists()
            else []
        )
    except (ValueError, OSError):
        data = []
    injected: list[dict] = []
    for e in data:
        if isinstance(e, dict) and e.get("attempt") == attempt:
            reasons = e.get("match_reasons") or []
            if reasons:
                for item in reasons:
                    title = item.get("title")
                    if title:
                        injected.append({
                            "knowledge_id": item.get("knowledge_id") or _stable_kb_id(title),
                            "title": title,
                        })
            else:
                top_ids = e.get("top_ids") or []
                for idx, title in enumerate(e.get("top_titles") or []):
                    injected.append({
                        "knowledge_id": (
                            top_ids[idx] if idx < len(top_ids) else _stable_kb_id(title)
                        ),
                        "title": title,
                    })
    # deduplicate preserving rank/order
    seen: set[str] = set()
    injected = [item for item in injected
                if not (item["knowledge_id"] in seen or seen.add(item["knowledge_id"]))]  # type: ignore[func-returns-value]

    text = (final_response or "").lower()
    mentioned = [item["knowledge_id"] for item in injected
                 if item["title"].lower() in text]
    metadata = attempt_metadata or {}
    injected_ids = [item["knowledge_id"] for item in injected]
    raw_declared = list(metadata.get("kb_used_ids") or [])
    declared = [kb_id for kb_id in raw_declared if kb_id in injected_ids]
    unknown_declared = [kb_id for kb_id in raw_declared if kb_id not in injected_ids]
    injected_titles = [item["title"] for item in injected]
    entry = {
        "attempt": attempt,
        "retrieved_ids": injected_ids,
        "injected_ids": injected_ids,
        "declared_used_ids": declared,
        "declared_unknown_ids": unknown_declared,
        "used_ids": declared,
        "mentioned_title_ids": mentioned,
        "usage_trace_complete": "kb_used_ids" in metadata,
        "ablation_disabled": kb_disabled,
        "ablation_reason": (
            "no_diagnostic_evidence_dependency"
            if os.environ.get("ABLATE_DIAGNOSTIC_EVIDENCE") == "1"
            else "no_kb"
            if kb_disabled
            else None
        ),
        "injected_titles": injected_titles,
        # Compatibility fields: exact-title mention, not a reliable use signal.
        "cited_titles": [item["title"] for item in injected
                         if item["knowledge_id"] in mentioned],
        "citation_rate": round(len(mentioned) / len(injected), 3) if injected else None,
    }
    try:
        existing = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else []
        if not isinstance(existing, list):
            existing = []
    except (ValueError, OSError):
        existing = []
    existing = [e for e in existing if not (isinstance(e, dict) and e.get("attempt") == attempt)]
    existing.append(entry)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_claude_cmd(prompt: str, *, claude_bin: str, model: Optional[str],
                      agent_name: str, session_id: str, workdir: Path,
                      allowed_tools: str, result_file: Path,
                      max_turns: Optional[str] = None,
                      effort: Optional[str] = None) -> list:
    """构造项目隔离的 `claude -p` 命令。

    Claude Code 2.1.204 的 ``--bare`` 不加载项目自定义 agent，和 ``--agent`` 组合会
    立即失败。改用 ``--setting-sources project``：只加载项目 agent/skill/settings，
    不加载 user/local 配置，仍保持实验隔离。输出由调用方重定向到 result_file。
    """
    cmd = [claude_bin, "-p", "--setting-sources", "project"]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
    # --max-turns: 单 attempt agentic turn 数硬闸，模型无关。
    if max_turns:
        cmd += ["--max-turns", str(max_turns)]
    cmd += [
        "--agent", agent_name,
        "--session-id", session_id,
        "--add-dir", str(workdir),
        "--allowedTools", allowed_tools,
        "--output-format", "json",
        prompt,
    ]
    return cmd


def _effective_session_max_turns(
    configured: Optional[str],
    task_remaining: object,
) -> Optional[str]:
    """Clamp a Claude session to the task budget remaining at action creation."""
    limits: list[int] = []
    for raw in (configured, task_remaining):
        try:
            value = int(raw) if raw is not None and raw != "" else 0
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            limits.append(value)
    return str(min(limits)) if limits else None


def _task_budget_boundary_hit(
    claude_state: object,
    applied_max_turns: object,
    task_remaining: object,
) -> bool:
    """Return true when Claude stopped at the dynamically clamped task boundary."""
    if claude_state != "max_turns_exceeded":
        return False
    try:
        applied = int(applied_max_turns)
        remaining = int(task_remaining)
    except (TypeError, ValueError):
        return False
    return remaining > 0 and applied >= remaining


def _session_result_path(task_dir: Path, attempt: int, session_id: str) -> Path:
    """Return an append-only result path for a single Claude session.

    `_claude_result_attemptN.json` is kept as the compatibility/latest file for
    existing consumers. The per-session archive avoids losing same-attempt
    retries caused by provider/API errors.
    """
    safe_session = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in session_id)
    return task_dir / "precision_tuning" / "claude_results" / (
        f"attempt{attempt}_{safe_session}.json"
    )


def _sync_latest_result_file(src: Path, dst: Path) -> None:
    """Best-effort copy from append-only archive to the legacy latest path."""
    try:
        if src.exists():
            shutil.copyfile(src, dst)
    except OSError:
        pass


def spawn_diagnose_agent(
    action: Action,
    task_dir: Path,
    op_name: str,
    attempt: int,
    *,
    agent_name: str = _DEFAULT_AGENT,
    claude_bin: str = _DEFAULT_CLAUDE_BIN,
    model: Optional[str] = None,
    allowed_tools: str = _DEFAULT_ALLOWED_TOOLS,
    workdir: Optional[Path] = None,
    npu: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    max_turns: Optional[str] = None,
    effort: Optional[str] = None,
    _run: Callable = subprocess.run,
) -> dict:
    """把 diagnose_and_fix Action 翻译成一次 claude 调用，返回本轮 result dict。

    _run 注入点: 默认 subprocess.run，UT 用 fake 替换 (不真跑 claude)。
    result dict 进 events.jsonl 的 action_completed.result (success/claude_state/...)。
    """
    args = action.skill_args or {}
    failure_type = args.get("failure_type", "")
    applied_max_turns = _effective_session_max_turns(
        max_turns, args.get("task_turns_remaining"))
    wd = Path(workdir) if workdir is not None else Path(task_dir)
    session_id = str(uuid.uuid4())
    result_file = _session_result_path(Path(task_dir), attempt, session_id)
    latest_result_file = Path(task_dir) / f"_claude_result_attempt{attempt}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)

    probe_policy = _probe_policy(Path(task_dir), failure_type, attempt)
    probe_record_path = _write_probe_policy(Path(task_dir), probe_policy)
    probe_source_before = _snapshot_probe_sources(Path(task_dir))
    prompt = _build_prompt(Path(task_dir), op_name, failure_type, attempt, npu)
    effort = effort if effort is not None else os.environ.get(_EFFORT_ENV)
    if effort == "":
        effort = None
    cmd = _build_claude_cmd(
        prompt, claude_bin=claude_bin, model=model, agent_name=agent_name,
        session_id=session_id, workdir=wd, allowed_tools=allowed_tools,
        result_file=result_file,
        max_turns=applied_max_turns, effort=effort)

    stderr_file = result_file.with_suffix(".stderr")
    proc = None
    try:
        with result_file.open("w", encoding="utf-8") as out:
            proc = _run(cmd, stdout=out, stderr=subprocess.PIPE, timeout=timeout_sec,
                        check=False, text=True)
    except subprocess.TimeoutExpired:
        _sync_latest_result_file(result_file, latest_result_file)
        return _attach_probe_observability(
            {"success": False, "claude_state": "timeout", "fatal": False,
             "error": f"claude 调用超时 (>{timeout_sec}s)",
             "max_turns_applied": applied_max_turns,
             "session_id": session_id,
             "result_path": str(result_file),
             "stderr_path": str(stderr_file),
             "latest_result_path": str(latest_result_file)},
            Path(task_dir), attempt, probe_policy, probe_record_path,
            probe_source_before)
    except Exception as e:  # noqa: BLE001 — 拉起失败转结构化结果，不让 runner 裸死
        return _attach_probe_observability(
            {"success": False, "claude_state": "spawn_failed", "fatal": True,
             "error": f"拉起 claude 失败: {e}",
             "max_turns_applied": applied_max_turns,
             "session_id": session_id,
             "result_path": str(result_file),
             "stderr_path": str(stderr_file),
             "latest_result_path": str(latest_result_file)},
            Path(task_dir), attempt, probe_policy, probe_record_path,
            probe_source_before)

    stderr_text = getattr(proc, "stderr", "") or ""
    if not isinstance(stderr_text, str):
        stderr_text = str(stderr_text)
    try:
        stderr_file.write_text(stderr_text, encoding="utf-8")
    except OSError:
        pass
    _sync_latest_result_file(result_file, latest_result_file)
    result = _classify_claude_result(result_file)
    result["claude_return_code"] = getattr(proc, "returncode", None)
    result["max_turns_applied"] = applied_max_turns
    for key in ("task_turns_used", "task_turns_limit", "task_turns_remaining",
                "task_turn_budget_tier"):
        if key in args:
            result[key] = args[key]
    result["session_turn_cap_hit"] = (
        result.get("claude_state") == "max_turns_exceeded"
    )
    result["task_budget_boundary_hit"] = _task_budget_boundary_hit(
        result.get("claude_state"),
        applied_max_turns,
        args.get("task_turns_remaining"),
    )
    result["stderr_path"] = str(stderr_file)
    if not result.get("success") and stderr_text.strip():
        result["error"] = (
            f"{result.get('error') or 'claude failed'}; "
            f"stderr_tail={stderr_text.strip()[-4000:]}"
        )
    result["session_id"] = session_id
    result["result_path"] = str(result_file)
    result["latest_result_path"] = str(latest_result_file)
    _attach_probe_observability(
        result, Path(task_dir), attempt, probe_policy, probe_record_path,
        probe_source_before)
    _write_kb_usage_trace(
        Path(task_dir), attempt, result.get("final_response"),
        result.get("attempt_metadata") or {})
    return result


def make_agent_callback(
    *,
    agent_name: str = _DEFAULT_AGENT,
    claude_bin: str = _DEFAULT_CLAUDE_BIN,
    model: Optional[str] = None,
    allowed_tools: str = _DEFAULT_ALLOWED_TOOLS,
    workdir: Optional[Path] = None,
    npu: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    max_turns: Optional[str] = None,
    effort: Optional[str] = None,
    _run: Callable = subprocess.run,
) -> Callable[[Action, Path, str, int], dict]:
    """造一个符合 runner AgentCallback 签名 (action, task_dir, op_name, attempt)->dict
    的闭包，把上述配置固化进去，供 run_debug_session(agent_callback=...) 注入。"""
    def _cb(action: Action, task_dir: Path, op_name: str, attempt: int) -> dict:
        return spawn_diagnose_agent(
            action, task_dir, op_name, attempt,
            agent_name=agent_name, claude_bin=claude_bin, model=model,
            allowed_tools=allowed_tools, workdir=workdir, npu=npu,
            timeout_sec=timeout_sec,
            max_turns=max_turns,
            effort=effort, _run=_run)
    return _cb
