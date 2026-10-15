"""runner.py — debug session 通用主循环 (方案 B: runner 是 owner)。

镜像 lingxi runtime/engine/runner.py 的 run_autonomous 形态，裁剪到 debug 所需 +
加 lingxi 没有的两件 debug 特有逻辑:
  1. Continue 决策 = attempt 边界 (落 attempt_started 事件，lingxi 无此概念)。
  2. wall-clock timeout 闸 (REWRITE_PLAN §2.3b 闸 3): 超时主动 emit 终态事件，不裸死。

主循环每拍:
    decision = debug_next_action(state, gate_result)
      ├ Done/Abort        → record_decision + write_exit_artifacts + return
      ├ Continue          → record_attempt_started + reload
      └ Action            → record_started → dispatch → record_completed → reload
                            (validate 步的 GateResult 缓存，喂给下一拍 next_action)

dispatcher 解耦 (可注入 mock，UT 不需 NPU): 默认实现 py_action→run_gate / spawn_agent→
agent_callback。resume = DebugState.load 重放，无额外逻辑。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from engine import transition
from engine.exit_artifacts import write_exit_artifacts
from engine.gate_adapter import GateResult, parse_gate_output, run_gate
from engine.next_action import debug_next_action
from engine.state import DebugState
from engine.types import Abort, Action, Continue, Done
from engine.validate_runner import run_objective_validation

# agent_callback 签名: (action, task_dir, op_name, attempt) -> result dict。
AgentCallback = Callable[[Action, Path, str, int], dict]
# dispatcher 签名: (action, task_dir, op_name, agent_callback) -> result dict。
Dispatcher = Callable[[Action, Path, str, Optional[AgentCallback]], dict]


class RunnerError(RuntimeError):
    pass


# 引擎可派发的 Action 白名单 (对应 lingxi validate_action)。
_AVAILABLE_PY_ACTIONS = (
    "precision_gate",
    "knowledge_search",
    "baseline_checkpoint",
    "checkpoint_and_rollback",
)
_AVAILABLE_AGENTS = ("debug_worker",)


def validate_action(action: Action) -> None:
    if action.kind == "py_action":
        if action.name not in _AVAILABLE_PY_ACTIONS:
            raise RunnerError(
                f"py_action '{action.name}' 不在白名单 {_AVAILABLE_PY_ACTIONS}")
    elif action.kind == "spawn_agent":
        if action.name not in _AVAILABLE_AGENTS:
            raise RunnerError(
                f"spawn_agent '{action.name}' 不在白名单 {_AVAILABLE_AGENTS}")
    else:
        raise RunnerError(f"未知 action.kind={action.kind!r}")


def _default_dispatcher(
    action: Action,
    task_dir: Path,
    op_name: str,
    agent_callback: Optional[AgentCallback],
) -> dict:
    """默认派发: py_action 跑 run_gate / spawn_agent 调 agent_callback。

    py_action(precision_gate): 跑 gate CLI，把 GateResult 摊平进 result，并履行
    import_subtype 喂数据契约 (Step 4 标注: next_action 靠 result.import_subtype 判
    import_env_side)。
    spawn_agent(debug_worker): 委托 agent_callback (NPU 上拉起 agent 进程)。
    """
    args = action.skill_args or {}
    if action.kind == "py_action":
        gate_step = args.get("gate_step", "validate")
        attempt = int(args.get("attempt", 0))
        if action.name == "baseline_checkpoint":
            # attempt=-1 is deliberately outside the Agent attempt budget.  It
            # objectively scores and snapshots the untouched input kernel.
            from engine import best_rollback as br
            objective = run_objective_validation(task_dir, attempt=-1)
            # A baseline is trusted only after the same common anti-cheat/AST/
            # structure checks used by validate.  This also makes an objective
            # baseline success eligible for direct clean-success termination.
            from scripts.gates.common import run_common
            common_output = run_common(
                "validate", task_dir, op_name, -1).to_gate_output()
            common_gate = parse_gate_output(common_output)
            if br.checkpoint_allowed(common_gate) and common_gate.passed:
                checkpoint = br.ensure_current_best(task_dir, -1, force=True)
            else:
                checkpoint = {
                    "success": False,
                    "updated": False,
                    "error": "baseline_common_gate_failed",
                }
            return {
                "success": bool(checkpoint.get("success")),
                "attempt": -1,
                "objective_validation": objective,
                "checkpoint": checkpoint,
                "baseline_common_gate": common_output,
                "failure_type": objective.get("failure_type"),
                "import_subtype": objective.get("import_subtype"),
                "verification_exit_code": objective.get("verification_exit_code"),
            }
        if action.name == "checkpoint_and_rollback":
            from engine import best_rollback as br
            state = DebugState.load(task_dir)
            validate_result = _validate_result_for_attempt(state, attempt)
            if validate_result is None:
                return {"success": False, "error": "missing_validate_event",
                        "attempt": attempt}
            return br.process_validation_checkpoint(
                task_dir, attempt, parse_gate_output(validate_result))
        # 修复 2 (6.11 文档): forensics step 先由 engine 主动产出 report，使下一拍
        # Gate-F 当轮可读到。执行失败只落 passed=False (方案 3 统一路径)——由
        # next_action 的 _completed_steps_this_attempt 排除出 completed → 重派，
        # 连续失败超限才 Done(stopped_by_gate)。不在此直接终止。
        forensics_result = None
        if gate_step == "forensics":
            from engine.validate_runner import run_forensics
            forensics_result = run_forensics(
                task_dir,
                attempt=attempt,
                failure_type=args.get("failure_type"),
            )
            if not forensics_result["success"]:
                return {
                    "passed": False,
                    "error": forensics_result.get("error"),
                    "gate": "GATE-FORENSICS-EXEC",
                    "forensics": forensics_result,
                }
        if action.name == "knowledge_search":
            from engine.knowledge_search import run_knowledge_search
            return run_knowledge_search(
                task_dir,
                kb_path=args.get("kb_path"),
                op_name=op_name,
                attempt=attempt,
            )
        objective = None
        if gate_step == "validate":
            objective = run_objective_validation(task_dir, attempt=attempt)
            # Persist direction metadata before Gate-V writes round_summary and
            # tuning_directions.  The diagnose event and validation JSON already
            # exist at this point; a second post-event write remains idempotent.
            from engine.exit_artifacts import write_diagnosis_summary
            write_diagnosis_summary(task_dir, attempt)
        gr = run_gate(task_dir, step=gate_step, op_name=op_name, attempt=attempt)
        result = _gate_result_to_dict(gr)
        if forensics_result is not None:
            result["forensics"] = forensics_result
            for key in (
                "cache_hit", "reuse_kind", "forensics_reused",
                "forensics_executed", "forensics_completed",
                "prebuild_executed", "build_skipped", "build_skip_reason",
                "reused_after_rollback", "reused_from_attempt",
                "reused_best_attempt", "cached_from_attempt", "report_path",
                "forensics_degraded", "diagnostic_evidence_kind",
                "proceed_to_agent", "forensics_unavailable", "unavailable_reason",
            ):
                if key in forensics_result:
                    result[key] = forensics_result[key]
        if objective is not None:
            result["objective_validation"] = objective
            result["failure_type"] = objective.get("failure_type")
            result["verification_exit_code"] = objective.get("verification_exit_code")
        if args.get("validation_after_task_turn_cap") is True:
            result["validation_after_task_turn_cap"] = True
            result["task_turns_used_at_cap"] = args.get("task_turns_used")
            result["task_turns_limit_at_cap"] = args.get("task_turns_limit")
            result["max_turns_applied_at_cap"] = args.get("max_turns_applied")
        return result
    if action.kind == "spawn_agent":
        if agent_callback is None:
            raise RunnerError("spawn_agent 需要 agent_callback，但未提供")
        attempt = int(args.get("attempt", 0))
        # 项 12b: diagnose 派发前后各取 kernel 快照，diff 出本轮真实改动文件名 →
        # result.changed_files (diagnosis_summary 的来源之一)。best-effort: 快照失败
        # 不阻断诊断本身。
        from engine.exit_artifacts import _kernel_file_hashes, diff_changed_files
        try:
            before = _kernel_file_hashes(task_dir, op_name)
        except Exception:  # noqa: BLE001
            before = None
        result = agent_callback(action, task_dir, op_name, attempt)
        if before is not None:
            try:
                result["changed_files"] = diff_changed_files(
                    before, _kernel_file_hashes(task_dir, op_name))
            except Exception:  # noqa: BLE001
                pass
        return result
    raise RunnerError(f"无法派发 action.kind={action.kind!r}")


def _gate_result_to_dict(gr: GateResult) -> dict:
    """GateResult → 事件 result dict。摊平 loop_signal/stop_reason_code +
    透传 import_subtype / l5_probe_* (从 checks 提取，履行喂数据契约)。"""
    checks = gr.checks or {}
    return {
        "gate": gr.gate,
        "passed": gr.passed,
        "loop_signal": gr.loop_signal,
        "loop_reason": gr.loop_reason,
        "stop_reason_code": gr.stop_reason_code,
        "prerequisite_error": gr.prerequisite_error,
        "failure_type": gr.failure_type,
        "import_subtype": checks.get("import_subtype"),
        # 真探针回退兜底标记 (供统计/消融区分"真探针通过"vs"连续失败回退")。
        "l5_probe_degraded": checks.get("l5_probe_degraded"),
        "l5_probe_source": checks.get("l5_probe_source"),
        "checks": gr.checks,
    }


def _result_to_gate(result: dict) -> Optional[GateResult]:
    """把 validate 步的 result dict 还原成 GateResult，喂给下一拍 next_action。
    非 gate result (无 loop_signal 字段) 返回 None。"""
    if "loop_signal" not in result and "gate" not in result:
        return None
    return parse_gate_output(result)


# 终态事件保护: 防止 Action 数远超合理上限时无限循环 (next_action 应已收敛，这是
# 防御性硬上限，按 MAX_ATTEMPTS * 每轮步数 * 安全系数估算)。
_MAX_TICKS = 200


def run_debug_session(
    task_dir: Path,
    *,
    op_name: str,
    agent: str = "constructive",
    entry_failure_type: Optional[str] = None,
    agent_callback: Optional[AgentCallback] = None,
    dispatcher: Optional[Dispatcher] = None,
    deadline_sec: Optional[float] = None,
    kb_path: Optional[str] = None,
    _now: Callable[[], float] = time.monotonic,
) -> dict:
    """驱动一个 debug session 到终态。返回 debug_status dict。

    task_dir: session 工作目录 (events.jsonl 落在 .debug_events/ 下)。
    op_name: 算子名 (传给 gate / agent)。
    agent: constructive | discovery (写进 session_started，供追溯)。
    entry_failure_type: 入口 failure_type；None 时由首拍 reload 决定 (resume 场景)。
    agent_callback: spawn_agent 的执行体 (NPU 上拉起 agent)；mock 时注入。
    dispatcher: 自定义派发器 (UT 注入 mock)；默认 _default_dispatcher。
    deadline_sec: wall-clock 总时长上限 (秒)；None 不限。超时主动 emit timeout 终态。
    kb_path: 知识库 JSON 路径；非空时 precision_failed 每轮 forensics 后做确定性
        检索，并在成功终态把候选知识入库。None (默认) 不启用。入库前置见
        knowledge_finalize (outcome==success 且无作弊)。
    _now: 单调时钟注入点 (UT 控制时间)。

    幂等/resume: 若 events.jsonl 已有 session_started，不重复写 (据已有事件续跑)。
    """
    task_dir = Path(task_dir)
    dispatcher = dispatcher or _default_dispatcher
    start = _now()

    # 5 处终态出口统一固化 op_name/kb_path (KB finalize 用)，避免逐处传参。
    def _term(decision) -> dict:
        return _terminate(task_dir, decision, op_name=op_name, kb_path=kb_path)

    # 初始化 / resume。
    existing = DebugState.load(task_dir)
    if not existing.events:
        # 全新 session: 写 session_started。
        transition.record_session_started(
            task_dir, op_name=op_name, agent=agent,
            entry_failure_type=entry_failure_type,
        )
    else:
        # H4: 已终态 session 不重启——直接返回原 status，绝不在终态事件后追加。
        if existing.is_terminal():
            from engine.exit_artifacts import build_debug_status
            return build_debug_status(task_dir)
        # H3: resume 时 reconcile 崩溃残留的 dangling action_started。补一条
        # action_completed{success:false}，使下一轮是「干净重派」而非续跑叠加
        # (尤其 spawn_agent 重派 = 二次改 kernel)。
        _reconcile_dangling(task_dir, existing)

    gate_result: Optional[GateResult] = None
    tick = 0
    while True:
        tick += 1
        if tick > _MAX_TICKS:
            return _term(Abort(
                category="state_machine_stuck",
                reason=f"超过最大 tick 数 {_MAX_TICKS}，疑似决策不收敛",
                details={"session_outcome": "crashed"}))

        # 闸 3: wall-clock timeout。超时主动 emit 终态，不裸死。
        if deadline_sec is not None and (_now() - start) >= deadline_sec:
            return _term(Done(
                session_outcome="timeout",
                reason=f"wall-clock 超时 (>{deadline_sec}s)"))

        state = DebugState.load(task_dir)
        decision = debug_next_action(state, gate_result)

        if isinstance(decision, (Done, Abort)):
            return _term(decision)

        if isinstance(decision, Continue):
            transition.record_attempt_started(task_dir, decision)
            gate_result = None  # 新一轮，清空上一轮 gate 缓存
            continue

        if isinstance(decision, Action):
            if kb_path and decision.step == "knowledge_search":
                decision.skill_args = dict(decision.skill_args or {})
                decision.skill_args["kb_path"] = kb_path
            validate_action(decision)
            action_id = transition.new_action_id()
            transition.record_action_started(task_dir, decision, action_id)
            try:
                result = dispatcher(decision, task_dir, op_name, agent_callback)
            except Exception as exc:  # noqa: BLE001 — dispatch 失败转 Abort，不裸死
                transition.record_action_completed(
                    task_dir, decision, action_id,
                    {"success": False, "error": str(exc), "fatal": True})
                return _term(Abort(
                    category="dispatch_error",
                    reason=f"派发 {decision.kind}:{decision.name} 失败: {exc}",
                    details={"session_outcome": "crashed"}))
            transition.record_action_completed(task_dir, decision, action_id, result)
            if (
                decision.step == "diagnose_and_fix"
                and result.get("ablation_violation")
            ):
                return _term(Abort(
                    category="ablation_violation",
                    reason=(
                        "Agent added probe instrumentation while the engine "
                        "probe policy required skip"
                    ),
                    details={
                        "session_outcome": "ablation_violation",
                        "probe_record_path": result.get("probe_record_path"),
                        "probe_source_audit_path": result.get(
                            "probe_source_audit_path"),
                    }))
            if (
                decision.step == "diagnose_and_fix"
                and result.get("provider_error")
            ):
                return _term(Abort(
                    category="provider_api_error",
                    reason=(
                        f"provider API error during diagnose: "
                        f"{result.get('claude_state')}"
                    ),
                    details={
                        "session_outcome": "provider_api_error",
                        "claude_state": result.get("claude_state"),
                        "api_error_status": result.get("api_error_status"),
                    }))
            # validate only records objective/gate evidence.  Best checkpoint
            # and rollback are a separate replayable action selected next.
            if decision.step == "validate":
                gate_result = _result_to_gate(result)
                # 项 12b: 本轮 validate 完成 (final_response/changed_files/validation
                # 三源齐备) → 归档 diagnosis_summary_attempt_N.json。每轮即写，CONTINUE
                # 也留档。best-effort，绝不影响主循环。
                from engine.exit_artifacts import write_diagnosis_summary
                attempt = int((decision.skill_args or {}).get("attempt", 0))
                write_diagnosis_summary(task_dir, attempt)
            else:
                gate_result = None
            # Preserve the historical human-readable rollback event.  The
            # action_completed result is authoritative, so a crash before this
            # convenience event does not lose recovery semantics.
            if decision.step == "checkpoint_and_rollback" and result.get("rolled_back"):
                transition.record_rollback(
                    task_dir,
                    from_attempt=int(result["from_attempt"]),
                    best_metric={
                        "attempt": result.get("best_attempt"),
                        "case_pass_rate": result.get("best_case_pass_rate"),
                        "match_rate": result.get("best_match_rate"),
                    },
                )
            continue

        raise RunnerError(f"未知 decision 类型: {type(decision).__name__}")


def _reconcile_dangling(task_dir: Path, state: DebugState) -> None:
    """resume 时收尾崩溃残留: 对每个有 started 无 completed 的 action_id，补一条
    action_completed{success:false, reason:crash-recovery}。

    意义: 让 next_action 的「本轮已完成 step」集合把残留步算作「已尝试且失败」，
    下一拍重新派发是干净重试，而不是把半完成的 spawn_agent 当作未开始再拉一次
    (二次改 kernel)。补的 completed 不带 loop_signal，故不会被误当作 Gate-V 结果。
    """
    for ev in state.dangling_started():
        payload = ev.get("action") or {}
        action = Action(
            kind=payload.get("kind", "py_action"),
            name=payload.get("name", ""),
            step=payload.get("step"),
            skill_args=payload.get("skill_args"),
        )
        transition.record_action_completed(
            task_dir, action, ev.get("action_id"),
            {"success": False, "reason": "crash-recovery: 上次进程在此步崩溃，已收尾"})


def _validate_result_for_attempt(state: DebugState, attempt: int) -> Optional[dict]:
    """Return the latest persisted validate result for an exact attempt."""
    current_attempt = None
    found = None
    for event in state.events:
        if event.get("type") == "attempt_started":
            current_attempt = event.get("attempt")
            continue
        if event.get("type") != "action_completed" or current_attempt != attempt:
            continue
        if (event.get("action") or {}).get("step") == "validate":
            found = event.get("result") or {}
    return found


def _terminate(task_dir: Path, decision, *,
               op_name: Optional[str] = None,
               kb_path: Optional[str] = None) -> dict:
    """落终态事件 + 生成退出产物 + (成功时) KB 入库 + 返回 debug_status dict。

    KB finalize (修复问题 6): kb_path 配置且 outcome==success 且无作弊时把候选知识
    入库；默认 kb_path=None 不启用。best-effort，绝不抛异常 (finalize_knowledge 内部
    已吞所有异常)，不影响 session 终态。在终态出口统一调用，H4 已终态早返回路径不经此
    (上次终态时已 finalize)，天然不重复入库。
    """
    transition.record_decision(task_dir, decision)
    write_exit_artifacts(task_dir)
    from engine.exit_artifacts import build_debug_status
    status = build_debug_status(task_dir)
    if kb_path:
        from engine.knowledge_candidate import ensure_candidate_entry
        status["knowledge_candidate"] = ensure_candidate_entry(
            task_dir, kb_path=kb_path, status=status, op_name=op_name)
        from engine.knowledge_finalize import finalize_knowledge
        status["kb_finalize"] = finalize_knowledge(
            task_dir, kb_path=kb_path,
            session_outcome=status.get("session_outcome"), op_name=op_name)
    # run_summary (项 11): kb_finalize 挂好后产出，供 batch report 直接消费。
    # best-effort，失败不影响终态。
    from engine.exit_artifacts import write_run_summary
    write_run_summary(task_dir, status)
    return status
