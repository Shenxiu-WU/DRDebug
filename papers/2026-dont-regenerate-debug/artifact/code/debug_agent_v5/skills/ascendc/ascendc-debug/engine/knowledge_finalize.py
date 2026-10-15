"""knowledge_finalize.py — session 成功终态时把候选知识入库 (修复问题 6)。

runner 主循环只有 forensics→diagnose_and_fix→validate，无 KB finalize 步；agent 产出的
candidate_kb_entry.json 一直留在磁盘未入库 (文档问题 6 实证: DeepSeek 2 个候选均未入库)。
本模块在 runner._terminate 终态出口被调用，以子进程方式跑 scripts/precision_knowledge.py
dump 完成入库 (沿用其全部字段校验，不在此重复)。

入库前置 (文档 6.5 成功分层 reportable_success = objective_success && anti_cheat_pass
&& ast_degrade_pass)，映射到引擎事实:
  - session_outcome == "success"      → Gate-V PASS，含 objective_success
  - cheat_history.json 无任何记录       → anti_cheat_pass && ast_degrade_pass
      · violation = 作弊 (绕过 kernel)
      · warning   = AST validator 异常 (ast_status=unknown，文档 6.5: 不等价 pass →
                    success_unverified)，保守同样不入库
三者全真才入库；否则带 reason skip。

默认不启用: kb_path=None (env ASCENDC_DEBUG_KB_PATH 未设) → 直接 skip，向后兼容。
绝不抛异常: 入库是 best-effort 副产物，任何失败都吞成 skip，不影响 session 终态。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from engine.events import read_events

# engine/knowledge_finalize.py → parent=engine/ → parent.parent=<skill 根>/ → scripts/...
_ENGINE_DIR = Path(__file__).resolve().parent
_KB_SCRIPT = _ENGINE_DIR.parent / "scripts" / "precision_knowledge.py"

_DUMP_TIMEOUT_SEC = 120  # dump 是纯文件操作，120s 充裕；防子进程异常卡死。

# §2.2: 入库判据改全量——全量复验下 matched_ratio ≥ 此阈值且不 crash 才入库。
# 0.95 保住 50/51(0.98)/49/50(0.98) 近满分高价值条目，挡 runtime/crash 类污染。
_KB_FULL_EVAL_THRESHOLD = 0.95


def _skip(reason: str, **extra) -> dict:
    return {"finalized": False, "skipped": True, "reason": reason, **extra}


def _session_op_name(task_dir: Path) -> Optional[str]:
    """从 events 的 session_started 取 op_name (传给 dump 的 --op-name)。"""
    for e in read_events(task_dir):
        if e.get("type") == "session_started":
            return e.get("op_name")
    return None


def _cheat_history_clean(task_dir: Path) -> bool:
    """cheat_history.json 无任何记录才算 clean (anti_cheat_pass && ast_degrade_pass)。

    文件不存在 = 从未触发任何检查 = clean。任何 cheating_attempts (violation 或
    warning) 都阻止入库。解析失败 = 状态未知 = 保守阻止 (绝不让损坏态污染 KB)。
    """
    path = task_dir / "precision_tuning" / "cheat_history.json"
    if not path.exists():
        return True
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False
    return not (data.get("cheating_attempts") or [])


def _kb_full_eval_threshold() -> float:
    try:
        return float(os.environ.get("ASCENDC_DEBUG_KB_FULL_EVAL_THRESHOLD",
                                    str(_KB_FULL_EVAL_THRESHOLD)))
    except (TypeError, ValueError):
        return _KB_FULL_EVAL_THRESHOLD


def _latest_full_eval(task_dir: Path) -> Optional[dict]:
    """取最大 attempt 的 validation_result 的 full_eval 子字段 (终态轮的全量复验产物)。

    无 validation_result / 均无 full_eval → None (全量闸关 / only_py 算子 / 旧产物)。
    """
    tuning = task_dir / "precision_tuning"
    if not tuning.is_dir():
        return None
    best_attempt, best_fe = -1, None
    for path in tuning.glob("validation_result_attempt_*.json"):
        m = re.search(r"validation_result_attempt_(\d+)\.json$", path.name)
        if not m:
            continue
        attempt = int(m.group(1))
        if attempt <= best_attempt:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        fe = data.get("full_eval")
        if isinstance(fe, dict):
            best_attempt, best_fe = attempt, fe
    return best_fe


def _full_eval_admits(task_dir: Path) -> tuple[bool, str]:
    """§2.2 入库判据: 全量复验下不 crash 且 matched_ratio ≥ 阈值才放行。

    无 full_eval / 未 ran → 放行 (向后兼容旧产物/无全量集的 only_py 算子，不引第三套数据源)。
    crashed → 挡 (更广输入下根本跑不起来，非测试不全)。ratio < 阈值 → 挡。
    """
    fe = _latest_full_eval(task_dir)
    if fe is None or not fe.get("ran"):
        return True, "无全量复验产物，按轻量口径放行 (向后兼容)"
    if fe.get("crashed"):
        return False, "全量复验 crash (更广输入跑不起来)"
    total = fe.get("total_cases", 0) or 0
    passed = fe.get("passed_cases", 0) or 0
    if total <= 0:
        return True, "全量复验无 case 数据，放行"
    ratio = passed / total
    threshold = _kb_full_eval_threshold()
    if ratio < threshold:
        return False, f"全量 matched_ratio={ratio:.3f} < {threshold}"
    return True, f"全量 matched_ratio={ratio:.3f} 达标"


def _candidate_action(task_dir: Path) -> tuple[str, Optional[str]]:
    """读候选的 action/merge_target_title (均为可选字段)；缺失/非法默认 new。

    agent 可在 Step 5.2 据 check 子命令的相似度结果，把 new/merge/abandon 决策写进
    候选；引擎只透传不重判 (语义决策属 agent)。读不出就保守 new (追加，不覆盖既有)。
    """
    path = task_dir / "precision_tuning" / "candidate_kb_entry.json"
    try:
        cand = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return "new", None
    action = cand.get("action", "new")
    if action not in ("new", "merge", "abandon"):
        action = "new"
    return action, cand.get("merge_target_title")


def _write_result(task_dir: Path, result: dict) -> None:
    """落 precision_tuning/kb_finalize_result.json 供追溯 (best-effort)。"""
    try:
        out = task_dir / "precision_tuning" / "kb_finalize_result.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    except OSError:
        pass


def _finalize_impl(task_dir: Path, kb_path: str, session_outcome: Optional[str],
                   op_name: Optional[str], _run: Callable) -> dict:
    if session_outcome != "success":
        return _skip(f"session_outcome={session_outcome} 非 success，不入库")
    if not _KB_SCRIPT.exists():
        return _skip(f"找不到入库脚本 {_KB_SCRIPT}")

    candidate = task_dir / "precision_tuning" / "candidate_kb_entry.json"
    if not candidate.exists():
        return _skip("无 candidate_kb_entry.json (agent 未产出候选)")
    if not _cheat_history_clean(task_dir):
        return _skip("cheat_history 非空 (reportable_success=false)，保守不入库")

    full_ok, full_reason = _full_eval_admits(task_dir)
    if not full_ok:
        return _skip(f"全量复验判据未过: {full_reason}")

    action, merge_target = _candidate_action(task_dir)
    if action == "abandon":
        return _skip("候选标记 action=abandon，跳过入库", action=action)

    op = op_name or _session_op_name(task_dir) or task_dir.name
    cmd = [sys.executable, str(_KB_SCRIPT), "dump",
           "--kb-path", str(kb_path),
           "--task-name", task_dir.name,
           "--task-dir", str(task_dir),
           "--op-name", op,
           "--action", action]
    if action == "merge" and merge_target:
        cmd += ["--merge-target-title", merge_target]

    try:
        proc = _run(cmd, capture_output=True, text=True,
                    timeout=_DUMP_TIMEOUT_SEC, check=False)
    except Exception as e:  # noqa: BLE001 — 子进程异常转 skip，不裸抛
        return _skip(f"入库子进程异常: {e}", action=action)

    if proc.returncode == 0:
        return {"finalized": True, "skipped": False, "reason": "已入库",
                "action": action, "op_name": op}
    return _skip(f"入库子进程失败 (exit={proc.returncode}): "
                 f"{(proc.stderr or '').strip()[:200]}", action=action)


def finalize_knowledge(
    task_dir,
    *,
    kb_path: Optional[str],
    session_outcome: Optional[str],
    op_name: Optional[str] = None,
    _run: Callable = subprocess.run,
) -> dict:
    """session 成功且无作弊时把候选知识入库，返回编排结果 dict (绝不抛异常)。

    kb_path: 知识库 JSON 路径；None/空 → skip (默认不启用)。
    session_outcome: 终态 outcome (取自 debug_status)；非 "success" → skip。
    op_name: 算子名 (传给 dump)；None 时从 events.session_started 读，再退化为目录名。
    _run: subprocess.run 注入点 (UT 用 fake 替换，不真跑子进程)。
    """
    task_dir = Path(task_dir)
    if os.environ.get("ABLATE_KB") == "1":
        result = _skip("ABLATE_KB=1 (no_kb arm)")
        _write_result(task_dir, result)
        return result
    if os.environ.get("ASCENDC_DEBUG_KB_READ_ONLY") == "1":
        result = _skip("ASCENDC_DEBUG_KB_READ_ONLY=1 (frozen experiment KB)")
        _write_result(task_dir, result)
        return result
    if not kb_path:
        return _skip("kb_path 未配置 (默认不启用)")
    result = _finalize_impl(task_dir, kb_path, session_outcome, op_name, _run)
    _write_result(task_dir, result)
    return result
