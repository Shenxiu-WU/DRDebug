"""next_action.py — 引擎核心纯函数决策器。

把原本散落在 SKILL.md 散文 + LLM 自驱的编排决策，全部代码化为一个无副作用纯函数:
    debug_next_action(state, gate_result=None) -> Action | Continue | Done | Abort | Escalate

决策来源 (均已精读源码/文档核实，非臆造):
  - 路由表          ← SKILL.md Step 0.3 分支路由表 (L123-132)
  - loop_signal 派发 ← SKILL.md L1027-1046 (PASS→Step5 / CONTINUE→Step0.3 / STOP→Step6)
  - 白名单/黑名单    ← SKILL.md L29-30 + agent.md L103 (import_env_side→skipped_env_issue)
  - 三道闸预算       ← REWRITE_PLAN §2.3b (全局 MAX_ATTEMPTS + 分支硬上限 + wall-clock)
  - stop_reason_code → session_outcome 映射 ← 用户确认 (按语义映射 + nearly_success 归 failed)

纯函数契约: 输入 (DebugState 投影 + 可选 GateResult)，输出一个 decision，无副作用、不写文件、
不跑子进程。所有 I/O (跑 gate、spawn agent、落事件) 由 runner (Step 5) 承担。
"""
from __future__ import annotations

import os
from typing import Optional

from engine.gate_adapter import GateResult, parse_gate_output
from engine.state import DebugState
from engine.types import (
    DEBUGGABLE_FAILURE_TYPES,
    PIPELINE_ORDER,
    Abort,
    Action,
    Continue,
    Done,
)

# ---------------------------------------------------------------------------
# 预算配置 (REWRITE_PLAN §2.3b 三道闸的前两道；第三道 wall-clock 在 runner)。
# 全局 MAX_ATTEMPTS 与现有 gates/common.py 同源 (env 可覆盖)，保持引擎与 gate 一致。
# ---------------------------------------------------------------------------
def _max_attempts() -> int:
    try:
        v = int(os.environ.get("ASCENDC_DEBUG_MAX_ATTEMPTS", "5"))
        return v if v >= 1 else 5
    except (TypeError, ValueError):
        return 5


def _max_agent_retries_per_attempt() -> int:
    try:
        v = int(os.environ.get("ASCENDC_DEBUG_MAX_AGENT_RETRIES_PER_ATTEMPT", "2"))
        return v if v >= 0 else 2
    except (TypeError, ValueError):
        return 2


def _max_forensics_retries_per_attempt() -> int:
    try:
        v = int(os.environ.get("ASCENDC_DEBUG_MAX_FORENSICS_RETRIES", "2"))
        return v if v >= 0 else 2
    except (TypeError, ValueError):
        return 2


def _max_task_turns() -> Optional[int]:
    """任务级累计 agentic turn 数硬闸 (修复 4b-B 方案B)。

    env 未设时不启用；engine/batch CLI 当前默认会设置硬上限 600，
    并由 ASCENDC_DEBUG_SOFT_TASK_TURNS=480 控制无客观改善时的软预算。
    用 turns 而非金额: turns 模型无关，与单 attempt --max-turns 同量纲，批跑切模型时不漂移。
    """
    raw = os.environ.get("ASCENDC_DEBUG_MAX_TASK_TURNS")
    if raw is None or raw == "":
        return None
    try:
        v = int(raw)
        return v if v >= 1 else None
    except (TypeError, ValueError):
        return None


def _soft_task_turns() -> Optional[int]:
    """Return the evidence-gated task turn budget.

    The soft budget prevents stagnant/provider-error retries from consuming the
    full hard allowance.  A task may use the hard allowance only after durable
    objective evidence shows that continuing is justified.
    """
    raw = os.environ.get("ASCENDC_DEBUG_SOFT_TASK_TURNS")
    if raw is None or raw == "":
        return None
    try:
        v = int(raw)
        return v if v >= 1 else None
    except (TypeError, ValueError):
        return None


def _max_degenerate_rounds() -> Optional[int]:
    """N6 复合早停阈值: 连续 N 轮退化空转 (作弊/audit 缺产物兜底无改善) 即停。

    默认启用 (返回 2)——N6 是 12a/修复4 的「CONTINUE 叠加烧预算」配套防御，二者已
    默认生效，故其防护也默认生效。
    env ASCENDC_DEBUG_MAX_DEGENERATE_ROUNDS=<N>: N>=1 设阈值；<=0 (或非法) 视为禁用
    (返回 None)，留给需要跑满 branch_cap 观察退化全过程的实验。
    """
    raw = os.environ.get("ASCENDC_DEBUG_MAX_DEGENERATE_ROUNDS")
    if raw is None or raw == "":
        return 2
    try:
        v = int(raw)
        return v if v >= 1 else None
    except (TypeError, ValueError):
        return 2


def _total_agent_turns(state: DebugState) -> int:
    """跨所有 attempt 累加 diagnose agent 的 agent_turns (agent_backend 透传的 num_turns)。

    只数 diagnose_and_fix 步的 action_completed.result.agent_turns；缺失/None 按 0 计
    (timeout/spawn_failed 等早返回路径无此字段，不误杀)。事件流是唯一事实源，重放即得。
    """
    total = 0
    for e in state.events:
        if e.get("type") != "action_completed":
            continue
        if (e.get("action") or {}).get("step") != "diagnose_and_fix":
            continue
        turns = (e.get("result") or {}).get("agent_turns")
        if isinstance(turns, int) and turns > 0:
            total += turns
    return total


def _has_turn_budget_extension_evidence(state: DebugState) -> bool:
    """Whether objective progress justifies extending soft turns to the hard cap.

    Accepted evidence is deliberately engine-owned and durable:
      * checkpoint_and_rollback promoted a post-baseline best implementation;
      * full-eval reached at least 90% passing cases or 99.9% match rate.

    Agent prose, KB hits, and diagnose success are not sufficient because none
    of them proves correctness progress.
    """
    for event in state.events:
        if event.get("type") != "action_completed":
            continue
        action = event.get("action") or {}
        result = event.get("result") or {}
        step = action.get("step")
        if step == "checkpoint_and_rollback":
            checkpoint = result.get("checkpoint") or {}
            best_attempt = result.get("best_attempt")
            if best_attempt is None:
                best_attempt = (checkpoint.get("best_metric") or {}).get("attempt")
            if checkpoint.get("updated") is True and isinstance(best_attempt, int) \
                    and best_attempt >= 0:
                return True
        if step != "validate":
            continue
        objective = result.get("objective_validation") or {}
        full_eval = objective.get("full_eval") or {}
        passed = full_eval.get("passed_cases")
        total = full_eval.get("total_cases")
        if isinstance(passed, int) and isinstance(total, int) and total > 0 \
                and passed / total >= 0.9:
            return True
        match_rate = full_eval.get("match_rate")
        try:
            if match_rate is not None and float(match_rate) >= 99.9:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _task_turn_budget(state: DebugState) -> tuple[Optional[int], str]:
    """Return the active task budget and its tier (disabled/soft/hard)."""
    hard = _max_task_turns()
    if hard is None:
        return None, "disabled"
    # no_loopguard removes adaptive early termination.  Keep the declared hard
    # safety cap, but do not let the evidence-gated soft tier change this arm's
    # behavior.
    if os.environ.get("ABLATE_LOOP_GUARD") == "1":
        return hard, "hard_no_loopguard"
    soft = _soft_task_turns()
    if soft is None or soft >= hard:
        return hard, "hard"
    if _has_turn_budget_extension_evidence(state):
        return hard, "hard_extended"
    return soft, "soft"


def _task_turn_budget_metadata(state: DebugState) -> dict:
    """Build immutable per-action budget metadata for backend enforcement."""
    used = _total_agent_turns(state)
    limit, tier = _task_turn_budget(state)
    remaining = None if limit is None else max(0, limit - used)
    return {
        "task_turns_used": used,
        "task_turns_limit": limit,
        "task_turns_remaining": remaining,
        "task_turn_budget_tier": tier,
    }


def _is_degenerate_round(result: dict) -> bool:
    """单轮 validate result 是否「退化空转」(N6)。二者之一即是:

      a. audit 方向评估缺失兜底: stop_reason_code == "stagnant_audit_missing"
         (12a: mismatch 连续未改善 + audit 缺失 → 保守 CONTINUE)。该 code 本身已隐含
         「连续未改善」(只在 _count_stagnant >= MAX_STAGNANT_ROUNDS 分支产生)。
      b. 确证作弊轮 (仅 violation，不含 N5 的 validator-errored warning):
         anticheat_pass==False 或 (ast_degrade_pass==False 且非 ast_validator_errored)。
         与 common.run_common 的 confirmed_cheat 同源——fail+cheat 的 A 场景返 CONTINUE，
         checks 经 to_gate_output→parse_gate_output 落进 events，故纯函数可重建，不读
         cheat_history.json (恪守 next_action 不碰文件的契约)。

    数据全部取自 events 持久化的 validate result，无文件/子进程 I/O。
    """
    checks = result.get("checks") or {}
    if result.get("stop_reason_code") == "stagnant_audit_missing" or \
            checks.get("stop_reason_code") == "stagnant_audit_missing":
        return True
    anticheat_failed = checks.get("anticheat_pass", True) is False
    ast_failed = checks.get("ast_degrade_pass", True) is False
    confirmed_cheat = anticheat_failed or (
        ast_failed and not checks.get("ast_validator_errored")
    )
    return confirmed_cheat


def _degenerate_continue_streak(state: DebugState) -> int:
    """从最近一轮往前数，连续「退化空转」(见 _is_degenerate_round) 的 attempt 数 (N6)。

    按 attempt 分组取每轮最后一个 validate result；某轮无 validate 或非退化即中断计数
    (退化必须是「连续」的末尾游程，单次退化夹在正常轮间不触发早停)。事件流是唯一事实源。
    """
    # 收集各 attempt 末尾的 validate result，按 attempt 出现序。
    per_attempt: list[Optional[dict]] = []
    cur_has_attempt = False
    for e in state.events:
        t = e.get("type")
        if t == "attempt_started":
            per_attempt.append(None)
            cur_has_attempt = True
        elif t == "action_completed" and cur_has_attempt and \
                (e.get("action") or {}).get("step") == "validate":
            per_attempt[-1] = e.get("result") or {}

    streak = 0
    for result in reversed(per_attempt):
        if result is None or not _is_degenerate_round(result):
            break
        streak += 1
    return streak


# 分支硬上限: 撞上即停 session (stopped_by_branch_limit)。precision 比其他分支宽，因为
# 精度调优本就需要更多轮次试探；build/import/runtime/timeout 是确定性错误，3 轮够。
# env ASCENDC_DEBUG_BRANCH_CAP_<FT>=<N> 可逐分支覆盖。
_DEFAULT_BRANCH_CAP = {
    "precision_failed": 5,
    "build_failed": 3,
    "import_failed": 3,
    "runtime_error": 3,
    "timeout": 3,
}


def _branch_cap(failure_type: str) -> int:
    env_key = f"ASCENDC_DEBUG_BRANCH_CAP_{failure_type.upper()}"
    try:
        v = int(os.environ[env_key])
        if v >= 1:
            return v
    except (KeyError, TypeError, ValueError):
        pass
    return _DEFAULT_BRANCH_CAP.get(failure_type, 3)


# ---------------------------------------------------------------------------
# 路由白名单/黑名单 (SKILL.md Step 0.3)。
# ---------------------------------------------------------------------------
# 5 个可调试分支 → 分支标签 (debug_status.json 的 session_branch 字段)。
_BRANCH_LABEL = {
    "precision_failed": "1-P",
    "build_failed": "1-B",
    "import_failed": "1-I",
    "runtime_error": "1-R",
    "timeout": "1-T",
}

# 不在白名单 → 直接退出的 failure_type → session_outcome。
# success 单独处理 (走 PASS 收尾，不在此表)。
_NONWHITELIST_OUTCOME = {
    "degraded": "skipped_unsupported_type",
    "no_kernel": "skipped_unsupported_type",
    "tilelang_only_failed": "skipped_unsupported_type",
    "execution_aborted": "skipped_unsupported_type",
}

# ---------------------------------------------------------------------------
# stop_reason_code → session_outcome 映射 (按语义)。
# 未列出的 stop_reason_code (含 None) 归 failed 兜底。
# ---------------------------------------------------------------------------
_STOP_OUTCOME = {
    "prerequisite_failure": "stopped_by_gate",
    "max_attempts_reached": "stopped_by_attempt_limit",
    # 以下均归 failed；nearly_success / fp16_precision_ceiling 由 stop_reason_code
    # 字段在论文统计时单独拆「准通过」子类，session_outcome 仍是 failed。
    "harmful_regression": "failed",
    "stagnant_same_direction": "failed",
    "stagnant_new_direction": "failed",
    "nearly_success": "failed",
    "fp16_precision_ceiling": "failed",
    "validation_failed": "failed",
    # 修复 4 (6.11 文档): objective success 但作弊 (绕过 kernel 的「假成功」)。
    # session_outcome 归 failed；reportable_success 由 stop_reason_code 在统计层区分
    # (clean success 须 anti_cheat_pass，见文档 6.5 成功分层)。
    "cheat_detected": "failed",
}


# ---------------------------------------------------------------------------
# 轮内步骤序列 (一个 attempt 内引擎逐步驱动的 Action 链)。
# 对齐 agent.md 分工边界: 取证=py_action / 诊断+修复=spawn_agent / Gate-V=py_action。
#   precision_failed: forensics → knowledge_search → audit → diagnose_and_fix → validate
#   other branches:   forensics → diagnose_and_fix → validate
# 引擎据「本轮 attempt_started 之后已完成哪些 step」决定下一步。
# audit (Gate-A) 仅 precision 分支有，位于修复前: 审上一轮遗留 + 本轮取证，产出
# 指导本轮修复; 缺产物降级 (passed=False 不阻断，见 _completed_steps_this_attempt)。
# ---------------------------------------------------------------------------
_ROUND_SEQUENCE = ("forensics", "diagnose_and_fix", "validate")
_PRECISION_ROUND_SEQUENCE = ("forensics", "knowledge_search", "audit",
                             "diagnose_and_fix", "validate")


def _round_sequence(failure_type: str) -> tuple[str, ...]:
    seq = _PRECISION_ROUND_SEQUENCE if failure_type == "precision_failed" else _ROUND_SEQUENCE
    skip = set()
    if os.environ.get("ABLATE_FORENSICS") == "1":
        # knowledge_search consumes the structured forensics report.  Leaving it
        # enabled would make no_forensics depend on manually/stale-created
        # reports and would no longer be a strict ablation.
        skip.update(("forensics", "knowledge_search"))
    if os.environ.get("ABLATE_KB") == "1":
        skip.add("knowledge_search")
    if os.environ.get("ABLATE_GATE_A") == "1":
        skip.add("audit")
    return tuple(s for s in seq if s not in skip) if skip else seq


def _completed_steps_this_attempt(state: DebugState) -> set:
    """本轮 (最后一个 attempt_started 之后) 已完成的 step 名集合。"""
    done: set = set()
    # 反向扫到最近的 attempt_started，收集其后的 action_completed.step。
    for e in reversed(state.events):
        if e.get("type") == "attempt_started":
            break
        if e.get("type") == "action_completed":
            step = (e.get("action") or {}).get("step")
            result = e.get("result") or {}
            if step == "diagnose_and_fix" and result.get("success") is False:
                # A session clipped exactly by the remaining task-turn budget
                # may already have written a useful kernel.  Count diagnose as
                # exhausted (never retry it), then allow the deterministic
                # validate/checkpoint steps to judge those changes once.
                if _is_task_budget_boundary_turn_cap(result):
                    done.add(step)
                continue
            if step == "checkpoint_and_rollback" and result.get("success") is False:
                continue
            # Deterministic compile/device-runtime failures cannot be repaired by
            # rerunning forensics against unchanged source.  validate_runner
            # persists their logs as degraded diagnostic evidence; treat that as
            # a completed prerequisite and let the matching Agent consume it.
            if step == "forensics" and result.get("passed") is False:
                if _is_usable_degraded_forensics(result):
                    done.add(step)
                continue
            # audit (Gate-A) 降级豁免: 无论 passed 与否都计入 completed (不进上面
            # 的 forensics 排除分支)。Gate-A 缺 section → passed=False，但它只产审计
            # md + 落 checks 供统计/消融，不阻断；越过它推进到 diagnose_and_fix。
            # (与 forensics passed=False 重派语义相反——故不加排除分支，此处仅注明。)
            if step:
                done.add(step)
    return done


def _is_usable_degraded_forensics(result: dict) -> bool:
    """Whether failed Gate-F still produced usable deterministic evidence."""
    forensics = result.get("forensics") or {}
    unavailable = (
        result.get("forensics_unavailable") is True
        or forensics.get("forensics_unavailable") is True
    )
    reason = result.get("unavailable_reason") or forensics.get("unavailable_reason")
    proceed = result.get("proceed_to_agent")
    if proceed is None:
        proceed = forensics.get("proceed_to_agent")
    report_path = result.get("report_path") or forensics.get("report_path")
    build_result = result.get("build_result") or forensics.get("build_result")
    evidence_kind = (
        result.get("diagnostic_evidence_kind")
        or forensics.get("diagnostic_evidence_kind")
    )
    # ``proceed_to_agent`` is explicit in new events.  The report fallback also
    # makes old interrupted build streams resumable after this fix.  Runtime
    # degradation is deliberately stricter: it requires the explicit marker
    # and classified evidence kind written by validate_runner.
    if unavailable and reason == "build_failed":
        return proceed is True or (
            bool(report_path) and isinstance(build_result, dict)
        )
    if unavailable and reason == "runtime_error":
        return (
            proceed is True
            and bool(report_path)
            and evidence_kind == "runtime_error_log"
        )
    return False


def _failed_forensics_calls_this_attempt(state: DebugState) -> int:
    count = 0
    for e in reversed(state.events):
        if e.get("type") == "attempt_started":
            break
        if e.get("type") != "action_completed":
            continue
        if (e.get("action") or {}).get("step") != "forensics":
            continue
        result = e.get("result") or {}
        if result.get("passed") is False and not _is_usable_degraded_forensics(result):
            count += 1
    return count


def _failed_diagnose_calls_this_attempt(state: DebugState) -> int:
    count = 0
    for e in reversed(state.events):
        if e.get("type") == "attempt_started":
            break
        if e.get("type") != "action_completed":
            continue
        if (e.get("action") or {}).get("step") != "diagnose_and_fix":
            continue
        result = e.get("result") or {}
        if result.get("success") is False:
            count += 1
    return count


def _diagnose_budget_exceeded_this_attempt(state: DebugState) -> Optional[int]:
    """Return turns from the latest diagnose max-turns hit in the current attempt."""
    for e in reversed(state.events):
        if e.get("type") == "attempt_started":
            return None
        if e.get("type") != "action_completed":
            continue
        if (e.get("action") or {}).get("step") != "diagnose_and_fix":
            continue
        result = e.get("result") or {}
        if result.get("claude_state") == "max_turns_exceeded":
            turns = result.get("agent_turns")
            return turns if isinstance(turns, int) else 0
    return None


def _is_task_budget_boundary_turn_cap(result: dict) -> bool:
    """Whether max-turns was the dynamic task-budget remainder, not session cap."""
    if result.get("claude_state") != "max_turns_exceeded":
        return False
    if result.get("task_budget_boundary_hit") is True:
        return True
    try:
        applied = int(result.get("max_turns_applied"))
        remaining = int(result.get("task_turns_remaining"))
    except (TypeError, ValueError):
        return False
    return remaining > 0 and applied >= remaining


def _task_budget_boundary_turn_cap_this_attempt(state: DebugState) -> Optional[dict]:
    """Return the current attempt's boundary-clipped diagnose result, if any."""
    for event in reversed(state.events):
        if event.get("type") == "attempt_started":
            return None
        if event.get("type") != "action_completed":
            continue
        if (event.get("action") or {}).get("step") != "diagnose_and_fix":
            continue
        result = event.get("result") or {}
        return result if _is_task_budget_boundary_turn_cap(result) else None
    return None


def _has_attempt_started_event(state: DebugState) -> bool:
    return any(e.get("type") == "attempt_started" for e in state.events)


def _successful_action_exists(state: DebugState, step: str) -> bool:
    return any(
        event.get("type") == "action_completed"
        and (event.get("action") or {}).get("step") == step
        and (event.get("result") or {}).get("success") is not False
        for event in state.events
    )


def _latest_successful_action_result(state: DebugState, step: str) -> Optional[dict]:
    """Return the latest successful result for a named engine step."""
    for event in reversed(state.events):
        if event.get("type") != "action_completed":
            continue
        if (event.get("action") or {}).get("step") != step:
            continue
        result = event.get("result") or {}
        if result.get("success") is not False:
            return result
    return None


def _protection_actions_enabled(state: DebugState) -> bool:
    """Production event streams start with session_started.

    Keeping synthetic counter-only states compatible is useful for pure routing
    tests and external callers that use next_action as a signal mapper.  Every
    runner-owned session, including old sessions resumed after this upgrade,
    has session_started and therefore gets the recovery barrier.
    """
    if os.environ.get("ABLATE_RECOVERY") == "1":
        return False
    return any(event.get("type") == "session_started" for event in state.events)


def _failed_action_calls(state: DebugState, step: str, *, current_attempt_only: bool) -> int:
    count = 0
    for event in reversed(state.events):
        if current_attempt_only and event.get("type") == "attempt_started":
            break
        if event.get("type") != "action_completed":
            continue
        if (event.get("action") or {}).get("step") != step:
            continue
        if (event.get("result") or {}).get("success") is False:
            count += 1
    return count


def _protection_action(step: str, failure_type: str, attempt: int) -> Action:
    return Action(
        kind="py_action",
        name=step,
        step=step,
        skill_args={"failure_type": failure_type, "attempt": attempt},
    )


def _budget_limit_decision(state: DebugState, failure_type: str) -> Optional[Done]:
    if state.total_attempts >= _max_attempts():
        return Done(session_outcome="stopped_by_attempt_limit",
                    reason=f"达全局 MAX_ATTEMPTS={_max_attempts()}")
    if state.branch_attempt(failure_type) >= _branch_cap(failure_type):
        return Done(session_outcome="stopped_by_branch_limit",
                    reason=f"分支 {failure_type} 撞硬上限 {_branch_cap(failure_type)}")
    # 修复 4b-B: 任务级累计 turns 闸 (env 未设时不启用)。放在 loop_limit 之后——
    # 撞轮次上限优先归对应 attempt/branch outcome，仅未撞轮次但累计 turns 超标时
    # 才归 stopped_by_budget。本函数只在续跑路径 (gate CONTINUE / 兜底) 前被调用，
    # gate PASS/STOP 直接 return 不经此 → 天然满足「终判优先于预算闸」(H1)。
    return _turn_budget_limit_decision(state)


def _turn_budget_limit_decision(state: DebugState) -> Optional[Done]:
    """Check only the task-level turn budget.

    A newly persisted ``attempt_started`` event already consumed the decision to
    enter that attempt.  Rechecking MAX_ATTEMPTS before its first action would
    count the boundary but skip the actual Agent round (for example, configured
    5 would execute only attempts 0-3).  The global/branch limits therefore stay
    in the pre-Continue path, while this narrower check protects only turns at
    the start of an already-authorized attempt.
    """
    limit, tier = _task_turn_budget(state)
    if limit is not None:
        used = _total_agent_turns(state)
        if used >= limit:
            if tier == "soft":
                reason = (
                    f"任务累计 agentic turns={used} 达软预算 {limit}，"
                    "且无客观改善/near-success/best checkpoint 更新证据"
                )
            else:
                reason = f"任务累计 agentic turns={used} 达硬上限 {limit}"
            return Done(session_outcome="stopped_by_budget",
                        reason=reason)
    return None


def _route_label(failure_type: Optional[str]) -> Optional[str]:
    return _BRANCH_LABEL.get(failure_type or "")


def _make_action_for_step(step: str, failure_type: str, attempt: int,
                          *, budget_metadata: Optional[dict] = None,
                          action_metadata: Optional[dict] = None) -> Action:
    """把轮内 step 名构造成具体 Action。

    forensics / validate → py_action (跑确定性脚本: precision_gate.py --step)。
    audit → py_action (Gate-A 质量门控: precision_gate.py --step audit, 复用同派发器)。
    knowledge_search → py_action (跑 precision_knowledge.py search, 写检索日志)。
    diagnose_and_fix → spawn_agent (拉起 constructive/discovery worker 诊断+改 kernel)。
    """
    common_args = {"failure_type": failure_type, "attempt": attempt}
    if action_metadata:
        common_args.update(action_metadata)
    if step == "diagnose_and_fix":
        skill_args = dict(common_args)
        if budget_metadata:
            skill_args.update(budget_metadata)
        return Action(
            kind="spawn_agent",
            name="debug_worker",
            step=step,
            skill_args=skill_args,
        )
    if step == "knowledge_search":
        return Action(
            kind="py_action",
            name="knowledge_search",
            step=step,
            skill_args=common_args,
        )
    # forensics/audit/validate 共用 precision_gate 派发器，仅 gate_step 不同。
    if step == "forensics":
        gate_step = "forensics"
    elif step == "audit":
        gate_step = "audit"
    else:
        gate_step = "validate"
    return Action(
        kind="py_action",
        name="precision_gate",
        step=step,
        skill_args={**common_args, "gate_step": gate_step},
    )


def debug_next_action(state: DebugState, gate_result: Optional[GateResult] = None):
    """引擎核心决策。返回 Action | Continue | Done | Abort。

    gate_result: 若上一步是 validate (Gate-V)，runner 把解析好的 GateResult 传入，
    据其 loop_signal/stop_reason_code 决定 PASS/CONTINUE/STOP。其余情况传 None；
    None 且本轮 validate 已完成时，从事件流重建 (crash-resume，见 _resolve_gate)。
    """
    ft = state.current_failure_type

    # ---- 0. 不可恢复前置: 无 failure_type → crashed ----
    if ft is None:
        return Abort(category="missing_failure_type",
                     reason="无 failure_type (verify_status 缺失或未初始化)",
                     details={"session_outcome": "crashed"})

    # ---- 1. 路由白名单/黑名单 (SKILL.md Step 0.3) ----
    if ft == "success":
        # 不经 attempt 直接 success 入口 (极少见): 视为已通过。
        return Done(session_outcome="success", reason="入口即 success")
    if ft == "import_failed":
        subtype = _latest_import_subtype(state)
        if subtype == "import_env_side":
            return Done(session_outcome="skipped_env_issue",
                        reason="import_env_side: 环境库问题，本 skill 不处理")
        # import_kernel_side (或未知子类，保守按 kernel_side 进入 1-I)。
    if ft in _NONWHITELIST_OUTCOME:
        return Done(session_outcome=_NONWHITELIST_OUTCOME[ft],
                    reason=f"failure_type={ft} 不在白名单")
    if ft not in DEBUGGABLE_FAILURE_TYPES:
        return Done(session_outcome="skipped_unsupported_type",
                    reason=f"未知 failure_type={ft}")

    # ---- 2. 本轮 validate 已完成 → gate 终判优先于预算闸 (H1) ----
    # 本轮是终态轮时，gate 的 PASS/STOP 才是真实结局，不能被预算闸覆盖成
    # attempt/branch limit。gate_result 缺失 (crash-resume) 时从事件流重建 (H2)。
    completed = _completed_steps_this_attempt(state)

    if (
        state.total_attempts > 0
        and "diagnose_and_fix" not in completed
        and "validate" not in completed
    ):
        exceeded_turns = _diagnose_budget_exceeded_this_attempt(state)
        if exceeded_turns is not None:
            return Done(
                session_outcome="stopped_by_budget",
                reason=(
                    "diagnose agent hit --max-turns"
                    f" (agent_turns={exceeded_turns})，停止本任务防重复重试"
                ),
            )
        # This attempt boundary was already authorized by the previous
        # validate/CONTINUE decision.  Only the task-turn budget may stop it
        # before diagnose; MAX_ATTEMPTS is checked before recording the next
        # boundary, otherwise the final configured attempt becomes a ghost.
        limited = _turn_budget_limit_decision(state)
        if limited is not None:
            return limited
        failed_agent_calls = _failed_diagnose_calls_this_attempt(state)
        if failed_agent_calls > _max_agent_retries_per_attempt():
            return Abort(
                category="agent_invocation_failed",
                reason=(
                    "diagnose agent 连续失败 "
                    f"{failed_agent_calls} 次，未计入有效修复轮次"
                ),
                details={
                    "session_outcome": "crashed",
                    "failed_agent_calls": failed_agent_calls,
                },
            )

    if "validate" in completed:
        # Gate terminal/continue decisions may only happen after the durable
        # checkpoint/rollback action.  On resume a dangling or failed action is
        # replayed, so validate evidence can never outrun its source side effect.
        if _protection_actions_enabled(state) and "checkpoint_and_rollback" not in completed:
            failures = _failed_action_calls(
                state, "checkpoint_and_rollback", current_attempt_only=True)
            if failures > 2:
                return Abort(
                    category="checkpoint_recovery_failed",
                    reason=f"checkpoint/rollback 恢复连续失败 {failures} 次",
                    details={"session_outcome": "crashed", "attempt": state.total_attempts - 1},
                )
            return _protection_action(
                "checkpoint_and_rollback", ft, state.total_attempts - 1)
        gate = _resolve_gate(state, gate_result)
        if gate is not None:
            sig = gate.loop_signal
            if sig in ("PASS", "STOP"):
                # 终判: 直接收尾，预算闸不介入。
                return _dispatch_loop_signal(state, gate)
            if sig == "CONTINUE":
                limited = _budget_limit_decision(state, ft)
                if limited is not None:
                    return limited
                # N6 复合早停: 连续多轮退化空转 (作弊/audit 缺产物兜底无改善) → 早停，
                # 防 12a/修复4 的 CONTINUE 叠加把退化任务兜到撞满 branch_cap 才停 (实测
                # $49 量级)。放在 loop_limit/budget 闸之后——撞硬上限优先归更精确的 outcome。
                # 此刻 events 已含当前轮 validate (runner 先 record_completed 再 reload)，
                # 故 streak 含当前轮。
                if os.environ.get("ABLATE_LOOP_GUARD") != "1":
                    degen_limit = _max_degenerate_rounds()
                    if degen_limit is not None:
                        streak = _degenerate_continue_streak(state)
                        if streak >= degen_limit:
                            return Done(
                                session_outcome="degenerate_no_progress",
                                reason=(f"连续 {streak} 轮退化空转 (作弊/audit 缺产物兜底无"
                                        f"改善) 达上限 {degen_limit}，早停防烧预算"),
                            )
                return _dispatch_loop_signal(state, gate)
            # sig is None: validate 已完成但 gate 未给 loop_signal = gate 协议错误，
            # 不能当作「需继续」无脑兜底 continue (问题 3)。gate 为 None (crash-resume
            # 无法重建) 时不进此分支，仍落到 ---- 5 的兜底 continue。
            return Abort(
                category="gate_protocol_error",
                reason="validate 完成但 gate 未给 loop_signal（gate 实现缺陷）",
                details={"session_outcome": "crashed"},
            )

    # ---- 3. 尚未开始任何 attempt → 客观验证并保存原始 baseline，再起第一轮 ----
    if state.total_attempts == 0:
        if (_protection_actions_enabled(state)
                and not _successful_action_exists(state, "baseline_checkpoint")):
            failures = _failed_action_calls(
                state, "baseline_checkpoint", current_attempt_only=False)
            if failures > 2:
                return Abort(
                    category="baseline_checkpoint_failed",
                    reason=f"原始 kernel baseline 连续验证/保存失败 {failures} 次",
                    details={"session_outcome": "crashed"},
                )
            return _protection_action("baseline_checkpoint", ft, -1)
        baseline = _latest_successful_action_result(state, "baseline_checkpoint") or {}
        baseline_ft = baseline.get("failure_type") or ft
        if baseline_ft == "success":
            # baseline_checkpoint includes objective validation plus common
            # anti-cheat/AST/structure checks, so this is a clean pre-Agent pass.
            common = baseline.get("baseline_common_gate") or {}
            if common.get("passed") is True:
                return Done(
                    session_outcome="success",
                    reason="原始 kernel baseline 已 clean success",
                )
            return Abort(
                category="baseline_clean_evidence_missing",
                reason="baseline objective success，但缺少 common clean 证据",
                details={"session_outcome": "crashed"},
            )
        if baseline_ft in _NONWHITELIST_OUTCOME:
            return Done(session_outcome=_NONWHITELIST_OUTCOME[baseline_ft],
                        reason=f"baseline failure_type={baseline_ft} 不在白名单")
        if baseline_ft not in DEBUGGABLE_FAILURE_TYPES:
            return Done(session_outcome="skipped_unsupported_type",
                        reason=f"未知 baseline failure_type={baseline_ft}")
        return Continue(next_attempt=0, next_failure_type=baseline_ft,
                        reason="session 首轮")

    # 人工构造/损坏状态可能只有 counters 没有 attempt_started 事件，此时只能按预算闸收敛。
    if not _has_attempt_started_event(state):
        limited = _budget_limit_decision(state, ft)
        if limited is not None:
            return limited
        return Continue(next_attempt=state.total_attempts, next_failure_type=ft,
                        reason="无当前 attempt 事件，进入下一轮")

    # ---- 4. 轮内推进: 找本轮下一个未完成的 step ----
    # forensics 失败重试上限 (镜像 diagnose 的 _max_agent_retries)：本轮 forensics
    # gate 连续 passed=False 超限 → 无法产出取证数据，Done(stopped_by_gate)。
    # 失败的 forensics 已被 _completed_steps_this_attempt 排除出 completed，故会
    # 反复重派 (修复 2 dispatcher 每次重跑 run_forensics)，由此计数收敛。
    failed_forensics = _failed_forensics_calls_this_attempt(state)
    if failed_forensics > _max_forensics_retries_per_attempt():
        return Done(
            session_outcome="stopped_by_gate",
            reason=f"forensics 连续失败 {failed_forensics} 次，无法产出取证数据",
        )

    for step in _round_sequence(ft):
        if step not in completed:
            if step == "diagnose_and_fix":
                limited = _turn_budget_limit_decision(state)
                if limited is not None:
                    return limited
                return _make_action_for_step(
                    step,
                    ft,
                    state.total_attempts - 1,
                    budget_metadata=_task_turn_budget_metadata(state),
                )
            if step == "validate":
                clipped = _task_budget_boundary_turn_cap_this_attempt(state)
                if clipped is not None:
                    return _make_action_for_step(
                        step,
                        ft,
                        state.total_attempts - 1,
                        action_metadata={
                            "validation_after_task_turn_cap": True,
                            "task_turns_used": _total_agent_turns(state),
                            "task_turns_limit": clipped.get("task_turns_limit"),
                            "max_turns_applied": clipped.get("max_turns_applied"),
                        },
                    )
            return _make_action_for_step(step, ft, state.total_attempts - 1)

    # ---- 5. 本轮全部 step 完成但无 gate 信号 (异常) → 兜底当作需继续 ----
    limited = _budget_limit_decision(state, ft)
    if limited is not None:
        return limited
    return Continue(next_attempt=state.total_attempts, next_failure_type=ft,
                    reason="本轮步骤完成，进入下一轮")


def _resolve_gate(state: DebugState, gate_result: Optional[GateResult]):
    """取本轮 validate 的 GateResult: 优先用 runner 传入的；缺失 (crash-resume) 时
    从事件流最后一个 validate 的 action_completed.result 重建 (H2)。

    事件已持久化 loop_signal/stop_reason_code (_gate_result_to_dict)，故重放可还原，
    符合 event-sourcing「内存态可由事实源重建」契约。重建不出则返回 None。
    """
    if gate_result is not None:
        return gate_result
    result = _validate_result_this_attempt(state)
    if result is None:
        return None
    return parse_gate_output(result)


def _validate_result_this_attempt(state: DebugState) -> Optional[dict]:
    """本轮 (最后一个 attempt_started 之后) 最近一个 validate 步的 result dict。"""
    for e in reversed(state.events):
        if e.get("type") == "attempt_started":
            return None
        if e.get("type") == "action_completed" and \
                (e.get("action") or {}).get("step") == "validate":
            return e.get("result") or {}
    return None


def _dispatch_loop_signal(state: DebugState, gate: GateResult):
    """读 Gate-V 的 loop_signal 派发终态/续跑 (SKILL.md L1027-1046)。"""
    sig = gate.loop_signal
    if sig == "PASS":
        return Done(session_outcome="success", reason="Gate-V loop_signal=PASS")
    if sig == "STOP":
        outcome = _STOP_OUTCOME.get(gate.stop_reason_code or "", "failed")
        return Done(session_outcome=outcome,
                    reason=f"Gate-V STOP: {gate.stop_reason_code or 'unspecified'}")
    # CONTINUE: objective validation is the authority for the next branch.  The
    # state projection is event-only and therefore still contains the current
    # attempt's type at this point; using it would silently discard branch drift.
    next_failure_type = gate.failure_type
    if next_failure_type not in DEBUGGABLE_FAILURE_TYPES:
        next_failure_type = state.current_failure_type
    return Continue(next_attempt=state.total_attempts,
                    next_failure_type=next_failure_type,
                    reason=f"Gate-V CONTINUE: {gate.loop_reason or ''}")


def _latest_import_subtype(state: DebugState) -> Optional[str]:
    """从最近的 action_completed.result 里取 import_subtype (runner 跑 verify 后写入)。

    首轮无 action_completed 时返回 None → 调用方保守按 import_kernel_side 进入 1-I。
    runner (Step 5) 须在 session_started / 首次 verify 把 import_subtype 写进事件 result，
    否则首轮 import_env_side 会漏判进 1-I (纯函数不读文件，靠 runner 喂数据)。
    """
    last = state.last_action_completed()
    if last is None:
        return None
    return (last.get("result") or {}).get("import_subtype")
