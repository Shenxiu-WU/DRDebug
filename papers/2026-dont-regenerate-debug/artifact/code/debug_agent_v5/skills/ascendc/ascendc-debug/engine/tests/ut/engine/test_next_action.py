"""test_next_action.py — 引擎核心决策器: 五分支 × 三信号矩阵 + 漂移 + 预算闸 + 路由。

验收点 (REWRITE_PLAN §6.4): 五分支×三信号 + failure_type 漂移路由 + 双层预算闸 +
白名单/黑名单 + 各 stop_reason_code。喂 events/state fixture → 断言 decision 类型与字段。

注: next_action 是纯函数，这里直接构造 DebugState (不落盘)，独立于 events I/O。
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from engine.gate_adapter import GateResult
from engine.next_action import debug_next_action
from engine.state import Counters, DebugState
from engine.types import Abort, Action, Continue, Done


def _state(*, current_ft, total_attempts=0, per_branch=None, events=None) -> DebugState:
    cf = Counters(
        total_attempts=total_attempts,
        current_failure_type=current_ft,
        entry_failure_type=current_ft,
        per_branch_attempt=dict(per_branch or {}),
    )
    return DebugState(task_dir=None, cf=cf, events=list(events or []))


def _gate(loop_signal, stop_reason_code=None, failure_type=None) -> GateResult:
    return GateResult(gate="GATE-V", passed=(loop_signal == "PASS"),
                      loop_signal=loop_signal, stop_reason_code=stop_reason_code,
                      failure_type=failure_type)


def _attempt_ev(ft):
    return {"type": "attempt_started", "failure_type": ft}


def _completed_ev(step, result=None):
    return {"type": "action_completed", "action": {"step": step}, "result": result or {}}


def _session_ev(ft="precision_failed"):
    return {"type": "session_started", "entry_failure_type": ft}


class TestRouting(unittest.TestCase):
    """SKILL.md Step 0.3 白名单/黑名单路由。"""

    def test_missing_failure_type_aborts_crashed(self) -> None:
        d = debug_next_action(_state(current_ft=None))
        self.assertIsInstance(d, Abort)
        self.assertEqual(d.details["session_outcome"], "crashed")

    def test_success_entry(self) -> None:
        d = debug_next_action(_state(current_ft="success"))
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "success")

    def test_import_env_side_skipped(self) -> None:
        # 最近 action_completed 带 import_subtype=import_env_side。
        st = _state(current_ft="import_failed", total_attempts=1,
                    per_branch={"import_failed": 1},
                    events=[_completed_ev("forensics",
                                          {"import_subtype": "import_env_side"})])
        d = debug_next_action(st)
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "skipped_env_issue")

    def test_import_kernel_side_continues(self) -> None:
        # kernel_side 不退出，进入轮内推进 (返回 Action，非 Done)。
        st = _state(current_ft="import_failed", total_attempts=1,
                    per_branch={"import_failed": 1},
                    events=[_attempt_ev("import_failed"),
                            _completed_ev("x", {"import_subtype": "import_kernel_side"})])
        d = debug_next_action(st)
        self.assertNotIsInstance(d, Done)

    def test_import_first_round_no_subtype_conservatively_enters_1I(self) -> None:
        # 首轮无 action_completed → import_subtype=None → 保守进 1-I (非 skipped_env_issue)。
        # 锁定缺口行为: runner 须喂 import_subtype，否则首轮 env_side 漏判。
        st = _state(current_ft="import_failed", total_attempts=1,
                    per_branch={"import_failed": 1},
                    events=[_attempt_ev("import_failed")])
        d = debug_next_action(st)
        self.assertNotIsInstance(d, Done)  # 不退出，进入轮内推进
        self.assertIsInstance(d, Action)
        self.assertEqual(d.skill_args["failure_type"], "import_failed")

    def test_nonwhitelist_skipped_unsupported(self) -> None:
        for ft in ("degraded", "no_kernel", "tilelang_only_failed", "execution_aborted"):
            d = debug_next_action(_state(current_ft=ft))
            self.assertIsInstance(d, Done, ft)
            self.assertEqual(d.session_outcome, "skipped_unsupported_type", ft)

    def test_unknown_failure_type_skipped(self) -> None:
        d = debug_next_action(_state(current_ft="weird_new_type"))
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "skipped_unsupported_type")


class TestBudgetGates(unittest.TestCase):
    """三道闸的前两道 (全局 MAX_ATTEMPTS + 分支硬上限)。"""

    def test_global_max_attempts(self) -> None:
        st = _state(current_ft="precision_failed", total_attempts=5,
                    per_branch={"precision_failed": 5})
        d = debug_next_action(st)
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "stopped_by_attempt_limit")

    def test_fifth_started_attempt_executes_before_budget_stop(self) -> None:
        # MAX_ATTEMPTS=5 means five real Agent rounds (attempt 0-4), not four
        # rounds plus a ghost attempt_started=4 immediately followed by Done.
        ft = "precision_failed"
        st = _state(
            current_ft=ft,
            total_attempts=5,
            per_branch={ft: 5},
            events=[_attempt_ev(ft) for _ in range(5)],
        )
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "forensics")
        self.assertEqual(d.skill_args["attempt"], 4)

    def test_branch_hard_cap_build(self) -> None:
        # build cap=3，撞上即停 (此时全局 total=3 < 5，不触发闸1)。
        st = _state(current_ft="build_failed", total_attempts=3,
                    per_branch={"build_failed": 3})
        d = debug_next_action(st)
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "stopped_by_branch_limit")

    def test_precision_cap_higher_than_build(self) -> None:
        # precision 在 build cap(3) 处不停 (precision cap=5)，应继续推进。
        st = _state(current_ft="precision_failed", total_attempts=3,
                    per_branch={"precision_failed": 3},
                    events=[_attempt_ev("precision_failed")])
        d = debug_next_action(st)
        self.assertNotIsInstance(d, Done)


class TestLoopSignalDispatch(unittest.TestCase):
    """五分支 × 三信号: validate 完成后读 gate_result 派发。"""

    def _state_after_validate(self, ft, total=1):
        return _state(current_ft=ft, total_attempts=total,
                      per_branch={ft: total},
                      events=[_attempt_ev(ft),
                              _completed_ev("forensics"),
                              _completed_ev("diagnose_and_fix"),
                              _completed_ev("validate")])

    def test_pass_to_success_all_branches(self) -> None:
        for ft in ("precision_failed", "build_failed", "import_failed",
                   "runtime_error", "timeout"):
            d = debug_next_action(self._state_after_validate(ft), _gate("PASS"))
            self.assertIsInstance(d, Done, ft)
            self.assertEqual(d.session_outcome, "success", ft)

    def test_continue_all_branches(self) -> None:
        for ft in ("precision_failed", "build_failed", "import_failed",
                   "runtime_error", "timeout"):
            d = debug_next_action(self._state_after_validate(ft), _gate("CONTINUE"))
            self.assertIsInstance(d, Continue, ft)

    def test_stop_prerequisite_to_gate(self) -> None:
        d = debug_next_action(self._state_after_validate("precision_failed"),
                              _gate("STOP", "prerequisite_failure"))
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "stopped_by_gate")

    def test_stop_max_attempts_to_attempt_limit(self) -> None:
        d = debug_next_action(self._state_after_validate("precision_failed"),
                              _gate("STOP", "max_attempts_reached"))
        self.assertEqual(d.session_outcome, "stopped_by_attempt_limit")

    def test_stop_nearly_success_to_failed(self) -> None:
        # 用户确认: nearly_success / fp16_ceiling 归 failed (stop_code 区分)。
        # cheat_detected (修复4: success+作弊) 同样归 failed，不计 clean success。
        for code in ("nearly_success", "fp16_precision_ceiling",
                     "harmful_regression", "stagnant_same_direction",
                     "validation_failed", "cheat_detected"):
            d = debug_next_action(self._state_after_validate("precision_failed"),
                                  _gate("STOP", code))
            self.assertEqual(d.session_outcome, "failed", code)

    def test_stop_unknown_code_to_failed(self) -> None:
        d = debug_next_action(self._state_after_validate("precision_failed"),
                              _gate("STOP", None))
        self.assertEqual(d.session_outcome, "failed")


class TestRoundSequence(unittest.TestCase):
    """轮内 step 序列推进 forensics → diagnose_and_fix → validate。"""

    def test_first_attempt_starts_continue(self) -> None:
        # 全新 session (total=0) → Continue(0)。
        d = debug_next_action(_state(current_ft="precision_failed", total_attempts=0))
        self.assertIsInstance(d, Continue)
        self.assertEqual(d.next_attempt, 0)

    def test_runner_event_stream_requires_baseline_before_attempt0(self) -> None:
        st = _state(current_ft="precision_failed", total_attempts=0,
                    events=[_session_ev()])
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "baseline_checkpoint")
        st.events.append(_completed_ev("baseline_checkpoint", {"success": True}))
        d = debug_next_action(st)
        self.assertIsInstance(d, Continue)
        self.assertEqual(d.next_attempt, 0)

    def test_recovery_ablation_skips_baseline_checkpoint(self) -> None:
        st = _state(current_ft="precision_failed", total_attempts=0,
                    events=[_session_ev()])
        with mock.patch.dict(os.environ, {"ABLATE_RECOVERY": "1"}):
            d = debug_next_action(st)
        self.assertIsInstance(d, Continue)
        self.assertEqual(d.next_attempt, 0)

    def test_attempt0_uses_baseline_failure_type_drift(self) -> None:
        st = _state(current_ft="precision_failed", total_attempts=0,
                    events=[_session_ev(), _completed_ev(
                        "baseline_checkpoint",
                        {"success": True, "failure_type": "runtime_error"})])
        d = debug_next_action(st)
        self.assertIsInstance(d, Continue)
        self.assertEqual(d.next_failure_type, "runtime_error")

    def test_clean_baseline_terminates_before_agent(self) -> None:
        st = _state(current_ft="precision_failed", total_attempts=0,
                    events=[_session_ev(), _completed_ev(
                        "baseline_checkpoint",
                        {"success": True, "failure_type": "success",
                         "baseline_common_gate": {"passed": True}})])
        d = debug_next_action(st)
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "success")

    def test_clean_baseline_without_common_evidence_aborts(self) -> None:
        st = _state(current_ft="precision_failed", total_attempts=0,
                    events=[_session_ev(), _completed_ev(
                        "baseline_checkpoint",
                        {"success": True, "failure_type": "success"})])
        d = debug_next_action(st)
        self.assertIsInstance(d, Abort)
        self.assertEqual(d.category, "baseline_clean_evidence_missing")

    def test_unsupported_baseline_does_not_start_attempt(self) -> None:
        st = _state(current_ft="precision_failed", total_attempts=0,
                    events=[_session_ev(), _completed_ev(
                        "baseline_checkpoint",
                        {"success": True, "failure_type": "execution_aborted"})])
        d = debug_next_action(st)
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "skipped_unsupported_type")

    def test_validate_requires_replayable_protection_action_before_pass(self) -> None:
        events = [_session_ev(), _attempt_ev("precision_failed"),
                  _completed_ev("validate", {"loop_signal": "PASS", "gate": "GATE-V"})]
        st = _state(current_ft="precision_failed", total_attempts=1,
                    per_branch={"precision_failed": 1}, events=events)
        d = debug_next_action(st, _gate("PASS"))
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "checkpoint_and_rollback")
        st.events.append(_completed_ev("checkpoint_and_rollback", {"success": False}))
        self.assertEqual(debug_next_action(st, _gate("PASS")).step,
                         "checkpoint_and_rollback")
        st.events.append(_completed_ev("checkpoint_and_rollback", {"success": True}))
        d = debug_next_action(st, _gate("PASS"))
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "success")

    def test_step_progression(self) -> None:
        ft = "precision_failed"
        # attempt 已起 (total=1)，本轮未做任何 step → forensics。
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft)])
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "forensics")
        self.assertEqual(d.kind, "py_action")

    def test_step_after_forensics_is_knowledge_search(self) -> None:
        # precision_failed 序列: forensics → knowledge_search → audit → diagnose_and_fix → validate
        ft = "precision_failed"
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft), _completed_ev("forensics")])
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "knowledge_search")

    def test_no_forensics_also_removes_dependent_knowledge_search(self) -> None:
        ft = "precision_failed"
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft)])
        with mock.patch.dict(os.environ, {"ABLATE_FORENSICS": "1"}):
            d = debug_next_action(st)
        self.assertEqual(d.step, "audit")

    def test_no_kb_removes_knowledge_search_only(self) -> None:
        ft = "precision_failed"
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft), _completed_ev("forensics")])
        with mock.patch.dict(os.environ, {"ABLATE_KB": "1"}):
            d = debug_next_action(st)
        self.assertEqual(d.step, "audit")
        self.assertEqual(d.kind, "py_action")

    def test_step_after_knowledge_search_is_audit(self) -> None:
        # knowledge_search 后是 audit (Gate-A 接回，修复前): py_action/gate_step=audit。
        ft = "precision_failed"
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft), _completed_ev("forensics"),
                            _completed_ev("knowledge_search")])
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "audit")
        self.assertEqual(d.kind, "py_action")
        self.assertEqual(d.name, "precision_gate")
        self.assertEqual(d.skill_args["gate_step"], "audit")

    def test_step_after_audit_is_spawn(self) -> None:
        # audit 完成后推进到 diagnose_and_fix (spawn_agent)。
        ft = "precision_failed"
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft), _completed_ev("forensics"),
                            _completed_ev("knowledge_search"),
                            _completed_ev("audit")])
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "diagnose_and_fix")
        self.assertEqual(d.kind, "spawn_agent")

    def test_audit_failed_does_not_block(self) -> None:
        # Gate-A 降级豁免: audit passed=False 仍计入 completed → 越过它推进到 diagnose,
        # 不卡死 (区别于 forensics passed=False 的重派语义)。
        ft = "precision_failed"
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft), _completed_ev("forensics"),
                            _completed_ev("knowledge_search"),
                            _completed_ev("audit", {"passed": False})])
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "diagnose_and_fix")

    def test_step_after_fix_is_validate(self) -> None:
        ft = "precision_failed"
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft), _completed_ev("forensics"),
                            _completed_ev("knowledge_search"),
                            _completed_ev("audit"),
                            _completed_ev("diagnose_and_fix")])
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "validate")
        self.assertEqual(d.kind, "py_action")


class TestDriftRouting(unittest.TestCase):
    """failure_type 漂移: next_action 按最新 ft 派发对应分支。"""

    def test_drift_routes_by_latest_ft(self) -> None:
        # 上一轮 precision，本轮漂到 build。current_ft=build → 走 build 分支步骤。
        st = _state(current_ft="build_failed", total_attempts=2,
                    per_branch={"precision_failed": 1, "build_failed": 1},
                    events=[_attempt_ev("precision_failed"),
                            _attempt_ev("build_failed")])
        d = debug_next_action(st)
        # 本轮 (build) 未做 step → forensics，skill_args.failure_type=build_failed。
        self.assertIsInstance(d, Action)
        self.assertEqual(d.skill_args["failure_type"], "build_failed")

    def test_continue_carries_current_ft(self) -> None:
        st = _state(current_ft="build_failed", total_attempts=1,
                    per_branch={"build_failed": 1},
                    events=[_attempt_ev("build_failed"),
                            _completed_ev("forensics"),
                            _completed_ev("diagnose_and_fix"),
                            _completed_ev("validate")])
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Continue)
        self.assertEqual(d.next_failure_type, "build_failed")

    def test_continue_uses_objective_validation_failure_type(self) -> None:
        # The current attempt is precision, but objective validation found that
        # the Agent introduced a compile failure.  The next attempt must enter
        # the build branch instead of inheriting stale precision state.
        st = _state(current_ft="precision_failed", total_attempts=1,
                    per_branch={"precision_failed": 1},
                    events=[_attempt_ev("precision_failed"),
                            _completed_ev("forensics"),
                            _completed_ev("diagnose_and_fix"),
                            _completed_ev("validate")])
        d = debug_next_action(
            st, _gate("CONTINUE", failure_type="build_failed"))
        self.assertIsInstance(d, Continue)
        self.assertEqual(d.next_failure_type, "build_failed")


class TestGatePrecedenceOverBudget(unittest.TestCase):
    """H1: 本轮 gate 终判 (PASS/STOP) 优先于预算闸。

    第 5 轮 validate 已 PASS/STOP 时，不能被全局闸 (total>=5) 覆盖成
    stopped_by_attempt_limit——那一轮本就是终态轮，gate 的终判才是真实结局。
    """

    def _state_fifth_validate(self, ft="precision_failed"):
        evs = [_attempt_ev(ft) for _ in range(5)]
        evs += [_completed_ev("forensics"), _completed_ev("diagnose_and_fix"),
                _completed_ev("validate")]
        return _state(current_ft=ft, total_attempts=5, per_branch={ft: 5}, events=evs)

    def test_fifth_round_pass_is_success_not_loop_limit(self) -> None:
        d = debug_next_action(self._state_fifth_validate(), _gate("PASS"))
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "success")

    def test_fifth_round_stop_maps_by_code(self) -> None:
        # 第 5 轮 STOP nearly_success → failed (按 code)，不被全局闸覆盖为 loop_limit。
        d = debug_next_action(self._state_fifth_validate(),
                              _gate("STOP", "nearly_success"))
        self.assertEqual(d.session_outcome, "failed")

    def test_continue_at_budget_limit_is_loop_limit(self) -> None:
        # 反向: 第 5 轮 CONTINUE 想续跑但已达 attempt 上限 → stopped_by_attempt_limit。
        d = debug_next_action(self._state_fifth_validate(), _gate("CONTINUE"))
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "stopped_by_attempt_limit")


class TestForensicsRetry(unittest.TestCase):
    """修复 2+3: forensics gate passed=False 不计入 completed → 重派；超限 → Done。"""

    def _forensics_fail_ev(self):
        return _completed_ev("forensics", {"passed": False,
                                           "gate": "GATE-FORENSICS-EXEC"})

    def test_forensics_fail_not_completed_reissues(self) -> None:
        # 本轮 forensics 失败一次 → 不计入 completed → 再次派 forensics。
        ft = "precision_failed"
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft), self._forensics_fail_ev()])
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "forensics")

    def test_forensics_fail_within_limit_still_reissues(self) -> None:
        # 失败 2 次 (= 默认上限)，未超限 → 仍重派，不 Done。
        ft = "precision_failed"
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft), self._forensics_fail_ev(),
                            self._forensics_fail_ev()])
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "forensics")

    def test_forensics_fail_over_limit_stops_by_gate(self) -> None:
        # 失败 3 次 (> 默认上限 2) → Done(stopped_by_gate)。
        ft = "precision_failed"
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft), self._forensics_fail_ev(),
                            self._forensics_fail_ev(), self._forensics_fail_ev()])
        d = debug_next_action(st)
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "stopped_by_gate")

    def test_forensics_success_then_progresses_to_knowledge_search(self) -> None:
        # 失败后又成功一次: 成功的 forensics 计入 completed，失败计数仍 1 (≤2)
        # 不触发超限 → precision_failed 序列下推进到 knowledge_search。
        ft = "precision_failed"
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft), self._forensics_fail_ev(),
                            _completed_ev("forensics", {"passed": True})])
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "knowledge_search")

    def test_build_failure_degraded_forensics_advances_to_agent(self) -> None:
        # A deterministic compile failure has no runnable kernel for precision
        # forensics.  The existing build log is usable evidence, so do not
        # rerun unchanged compilation and do not block the build-fix Agent.
        ft = "build_failed"
        degraded = _completed_ev("forensics", {
            "passed": False,
            "gate": "GATE-BUILD-F",
            "forensics": {
                "forensics_unavailable": True,
                "unavailable_reason": "build_failed",
                "proceed_to_agent": True,
                "report_path": "/tmp/forensics_report_1.json",
            },
        })
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft), degraded])
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "diagnose_and_fix")
        self.assertEqual(d.skill_args["failure_type"], "build_failed")

    def test_degraded_build_evidence_is_not_counted_as_retry_failure(self) -> None:
        ft = "build_failed"
        events = [_attempt_ev(ft)]
        for _ in range(3):
            events.append(_completed_ev("forensics", {
                "passed": False,
                "forensics_unavailable": True,
                "unavailable_reason": "build_failed",
                "proceed_to_agent": True,
                "report_path": "/tmp/build-evidence.json",
            }))
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=events)
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "diagnose_and_fix")

    def test_runtime_error_log_degradation_advances_to_agent(self) -> None:
        ft = "runtime_error"
        degraded = _completed_ev("forensics", {
            "passed": False,
            "gate": "GATE-RUNTIME-F",
            "forensics": {
                "forensics_unavailable": True,
                "unavailable_reason": "runtime_error",
                "forensics_degraded": True,
                "diagnostic_evidence_kind": "runtime_error_log",
                "proceed_to_agent": True,
                "report_path": "/tmp/forensics_report_1.json",
            },
        })
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft), degraded])
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "diagnose_and_fix")
        self.assertEqual(d.skill_args["failure_type"], "runtime_error")

    def test_unclassified_runtime_forensics_failure_still_retries(self) -> None:
        ft = "runtime_error"
        failed = _completed_ev("forensics", {
            "passed": False,
            "gate": "GATE-FORENSICS-EXEC",
            "forensics": {
                "forensics_unavailable": True,
                "unavailable_reason": "runtime_error",
                "report_path": "/tmp/forensics_report_1.json",
            },
        })
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft), failed])
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "forensics")


class TestValidateNoSignalAborts(unittest.TestCase):
    """修复 3 ④: validate 完成但 gate loop_signal=None = gate 协议错误 → Abort，
    不再无脑兜底 continue (问题 3)。

    注: 「gate 为 None → 落 §5 兜底 continue」是防御性死分支——只要 validate 在
    completed 集合里，_validate_result_this_attempt 至少返回 {}，parse_gate_output({})
    恒为非 None GateResult，故正常事件流无法把 gate 重建成 None。不为不可达路径写测试。"""

    def test_validate_done_no_signal_aborts(self) -> None:
        # 事件里 validate 已完成但 result 无 loop_signal (含 gate 字段使其可重建为
        # loop_signal=None 的 GateResult)。
        ft = "precision_failed"
        result = {"gate": "GATE-COMMON-validate", "passed": False,
                  "loop_signal": None}
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                    events=[_attempt_ev(ft), _completed_ev("forensics"),
                            _completed_ev("diagnose_and_fix"),
                            _completed_ev("validate", result)])
        d = debug_next_action(st, None)
        self.assertIsInstance(d, Abort)
        self.assertEqual(d.category, "gate_protocol_error")
        self.assertEqual(d.details["session_outcome"], "crashed")


class TestResumeReconstructsGate(unittest.TestCase):
    """H2: gate_result 未传入 (crash-resume) 时，从事件流最后一个 validate result 重建。

    validate 的 action_completed 已持久化 loop_signal/stop_reason_code，但 runner 内存态
    gate_result 在崩溃后丢失。next_action 须从事件重建，不能无视已算出的终判径直续跑。
    """

    def _state_validate_persisted(self, loop_signal, stop_code=None, ft="precision_failed"):
        result = {"gate": "GATE-V", "passed": loop_signal == "PASS",
                  "loop_signal": loop_signal, "stop_reason_code": stop_code}
        return _state(current_ft=ft, total_attempts=1, per_branch={ft: 1},
                      events=[_attempt_ev(ft), _completed_ev("forensics"),
                              _completed_ev("diagnose_and_fix"),
                              _completed_ev("validate", result)])

    def test_resume_pass_reconstructed_to_success(self) -> None:
        # gate_result=None (resume)，事件里 validate loop_signal=PASS → 重建派发 success。
        d = debug_next_action(self._state_validate_persisted("PASS"), None)
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "success")

    def test_resume_stop_reconstructed_to_outcome(self) -> None:
        d = debug_next_action(
            self._state_validate_persisted("STOP", "prerequisite_failure"), None)
        self.assertEqual(d.session_outcome, "stopped_by_gate")

    def test_resume_continue_reconstructed(self) -> None:
        d = debug_next_action(self._state_validate_persisted("CONTINUE"), None)
        self.assertIsInstance(d, Continue)
        self.assertEqual(d.next_failure_type, "precision_failed")


class TestTaskTurnsBudget(unittest.TestCase):
    """跨 attempt 累计 turns 闸。

    用 num_turns 累加 (agent_backend 透传的 agent_turns)，turns 模型无关。撞轮次上限优先归 loop_limit，
    仅未撞轮次但累计 turns 超标才归 stopped_by_budget；gate PASS/STOP 不经此闸 (H1)。
    """

    _ENV = "ASCENDC_DEBUG_MAX_TASK_TURNS"
    _SOFT_ENV = "ASCENDC_DEBUG_SOFT_TASK_TURNS"

    def setUp(self) -> None:
        self._saved = os.environ.get(self._ENV)
        self._saved_soft = os.environ.get(self._SOFT_ENV)
        os.environ.pop(self._ENV, None)
        os.environ.pop(self._SOFT_ENV, None)

    def tearDown(self) -> None:
        if self._saved is None:
            os.environ.pop(self._ENV, None)
        else:
            os.environ[self._ENV] = self._saved
        if self._saved_soft is None:
            os.environ.pop(self._SOFT_ENV, None)
        else:
            os.environ[self._SOFT_ENV] = self._saved_soft

    def _state_continue_with_turns(self, turns_per_attempt, *, ft="precision_failed"):
        """构造 N 轮已完成、最后一轮 validate 完成的 state；各轮 diagnose 带 agent_turns。

        total_attempts=len(turns) (< 5 全局闸)，per_branch < cap，故不撞 loop_limit；
        仅 turns 累计可能超阈。最后一轮 validate 已完成 → 配 _gate("CONTINUE") 触发闸。
        """
        n = len(turns_per_attempt)
        evs = []
        for t in turns_per_attempt:
            evs.append(_attempt_ev(ft))
            evs.append(_completed_ev("diagnose_and_fix", {"success": True,
                                                          "agent_turns": t}))
        evs += [_completed_ev("forensics"), _completed_ev("validate")]
        return _state(current_ft=ft, total_attempts=n, per_branch={ft: n}, events=evs)

    def test_disabled_by_default_continues(self) -> None:
        # env 未设 → 闸不启用，即使累计 turns 巨大也照常 CONTINUE。
        st = self._state_continue_with_turns([500, 500])
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Continue)

    def test_under_limit_continues(self) -> None:
        os.environ[self._ENV] = "100"
        st = self._state_continue_with_turns([30, 40])  # 累计 70 < 100
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Continue)

    def test_at_limit_stops_by_budget(self) -> None:
        os.environ[self._ENV] = "100"
        st = self._state_continue_with_turns([60, 50])  # 累计 110 >= 100
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "stopped_by_budget")

    def test_gate_pass_overrides_budget(self) -> None:
        # H1: 即使累计 turns 超阈，本轮 gate PASS 仍判 success (终判优先于预算闸)。
        os.environ[self._ENV] = "50"
        st = self._state_continue_with_turns([60, 60])  # 累计 120 >> 50
        d = debug_next_action(st, _gate("PASS"))
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "success")

    def test_loop_limit_precedence_over_budget(self) -> None:
        # 同时撞全局轮次上限 (5) 与 turns 超阈 → 归 attempt_limit，非 budget。
        os.environ[self._ENV] = "10"
        st = self._state_continue_with_turns([40, 40, 40, 40, 40])  # total=5 撞闸
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "stopped_by_attempt_limit")

    def test_no_loopguard_uses_hard_budget_not_soft_budget(self) -> None:
        os.environ[self._ENV] = "600"
        os.environ[self._SOFT_ENV] = "480"
        st = self._state_continue_with_turns([240, 240])
        with mock.patch.dict(os.environ, {"ABLATE_LOOP_GUARD": "1"}):
            d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Continue)

    def test_missing_turns_counted_as_zero(self) -> None:
        # diagnose result 缺 agent_turns (timeout/spawn_failed 早返回) → 按 0 计，不误杀。
        os.environ[self._ENV] = "10"
        ft = "precision_failed"
        evs = [_attempt_ev(ft),
               _completed_ev("diagnose_and_fix", {"success": True}),  # 无 agent_turns
               _completed_ev("forensics"), _completed_ev("validate")]
        st = _state(current_ft=ft, total_attempts=1, per_branch={ft: 1}, events=evs)
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Continue)  # 累计 0 < 10，不停

    def test_soft_budget_stops_without_objective_progress(self) -> None:
        os.environ[self._ENV] = "600"
        os.environ[self._SOFT_ENV] = "480"
        st = self._state_continue_with_turns([240, 240])
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "stopped_by_budget")
        self.assertIn("软预算 480", d.reason)

    def test_best_checkpoint_extends_soft_budget_to_hard_cap(self) -> None:
        os.environ[self._ENV] = "600"
        os.environ[self._SOFT_ENV] = "480"
        st = self._state_continue_with_turns([256, 252])
        st.events.append(_completed_ev("checkpoint_and_rollback", {
            "success": True,
            "checkpoint": {
                "updated": True,
                "best_metric": {"attempt": 1, "case_pass_rate": 90.0},
            },
            "best_attempt": 1,
        }))
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Continue)

    def test_extended_task_stops_at_hard_cap(self) -> None:
        os.environ[self._ENV] = "600"
        os.environ[self._SOFT_ENV] = "480"
        st = self._state_continue_with_turns([300, 300])
        st.events.append(_completed_ev("checkpoint_and_rollback", {
            "success": True,
            "checkpoint": {
                "updated": True,
                "best_metric": {"attempt": 1, "case_pass_rate": 90.0},
            },
            "best_attempt": 1,
        }))
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "stopped_by_budget")
        self.assertIn("硬上限 600", d.reason)

    def test_near_success_full_eval_extends_soft_budget(self) -> None:
        os.environ[self._ENV] = "600"
        os.environ[self._SOFT_ENV] = "480"
        st = self._state_continue_with_turns([256, 252])
        st.events.append(_completed_ev("validate", {
            "objective_validation": {
                "full_eval": {"passed_cases": 9, "total_cases": 10,
                              "match_rate": 99.99},
            },
        }))
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Continue)

    def test_diagnose_action_carries_remaining_soft_budget(self) -> None:
        os.environ[self._ENV] = "600"
        os.environ[self._SOFT_ENV] = "480"
        ft = "precision_failed"
        st = _state(
            current_ft=ft,
            total_attempts=1,
            per_branch={ft: 1},
            events=[
                _attempt_ev(ft),
                _completed_ev("forensics"),
                _completed_ev("knowledge_search"),
                _completed_ev("audit"),
                _completed_ev("diagnose_and_fix", {
                    "success": False,
                    "claude_state": "api_error_400",
                    "agent_turns": 404,
                }),
            ],
        )
        d = debug_next_action(st)
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "diagnose_and_fix")
        self.assertEqual(d.skill_args["task_turns_used"], 404)
        self.assertEqual(d.skill_args["task_turns_remaining"], 76)
        self.assertEqual(d.skill_args["task_turn_budget_tier"], "soft")

    def _boundary_capped_state(self):
        ft = "precision_failed"
        events = [
            _attempt_ev(ft),
            _completed_ev("diagnose_and_fix", {
                "success": True,
                "claude_state": "ok",
                "agent_turns": 404,
            }),
            _completed_ev("validate", {"loop_signal": "CONTINUE"}),
            _completed_ev("checkpoint_and_rollback", {
                "success": True,
                "checkpoint": {"updated": False},
                "best_attempt": -1,
            }),
            _attempt_ev(ft),
            _completed_ev("forensics"),
            _completed_ev("knowledge_search"),
            _completed_ev("audit"),
            _completed_ev("diagnose_and_fix", {
                "success": False,
                "claude_state": "max_turns_exceeded",
                "agent_turns": 76,
                "max_turns_applied": "76",
                "task_turns_used": 404,
                "task_turns_limit": 480,
                "task_turns_remaining": 76,
                "task_turn_budget_tier": "soft",
                "session_turn_cap_hit": True,
                "task_budget_boundary_hit": True,
            }),
        ]
        return _state(
            current_ft=ft,
            total_attempts=2,
            per_branch={ft: 2},
            events=events,
        )

    def test_task_boundary_max_turns_runs_validation_once(self) -> None:
        os.environ[self._ENV] = "600"
        os.environ[self._SOFT_ENV] = "480"
        d = debug_next_action(self._boundary_capped_state())
        self.assertIsInstance(d, Action)
        self.assertEqual(d.step, "validate")
        self.assertTrue(d.skill_args["validation_after_task_turn_cap"])
        self.assertEqual(d.skill_args["task_turns_used"], 480)
        self.assertEqual(d.skill_args["task_turns_limit"], 480)
        self.assertEqual(d.skill_args["max_turns_applied"], "76")

    def test_task_boundary_validation_without_progress_stops(self) -> None:
        os.environ[self._ENV] = "600"
        os.environ[self._SOFT_ENV] = "480"
        st = self._boundary_capped_state()
        st.events.extend([
            _completed_ev("validate", {
                "loop_signal": "CONTINUE",
                "objective_validation": {
                    "full_eval": {"passed_cases": 4, "total_cases": 10,
                                  "match_rate": 40.0},
                },
            }),
            _completed_ev("checkpoint_and_rollback", {
                "success": True,
                "checkpoint": {"updated": False},
                "best_attempt": -1,
            }),
        ])
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "stopped_by_budget")
        self.assertIn("软预算 480", d.reason)

    def test_task_boundary_validation_progress_extends_to_hard_budget(self) -> None:
        os.environ[self._ENV] = "600"
        os.environ[self._SOFT_ENV] = "480"
        st = self._boundary_capped_state()
        st.events.extend([
            _completed_ev("validate", {
                "loop_signal": "CONTINUE",
                "objective_validation": {
                    "full_eval": {"passed_cases": 9, "total_cases": 10,
                                  "match_rate": 99.99},
                },
            }),
            _completed_ev("checkpoint_and_rollback", {
                "success": True,
                "checkpoint": {
                    "updated": True,
                    "best_metric": {"attempt": 1, "case_pass_rate": 90.0},
                },
                "best_attempt": 1,
            }),
        ])
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Continue)

    def test_ordinary_session_max_turns_still_stops_without_validation(self) -> None:
        os.environ[self._ENV] = "600"
        os.environ[self._SOFT_ENV] = "480"
        ft = "precision_failed"
        st = _state(
            current_ft=ft,
            total_attempts=1,
            per_branch={ft: 1},
            events=[
                _attempt_ev(ft),
                _completed_ev("forensics"),
                _completed_ev("knowledge_search"),
                _completed_ev("audit"),
                _completed_ev("diagnose_and_fix", {
                    "success": False,
                    "claude_state": "max_turns_exceeded",
                    "agent_turns": 240,
                    "max_turns_applied": "240",
                    "task_turns_used": 0,
                    "task_turns_limit": 480,
                    "task_turns_remaining": 480,
                    "task_turn_budget_tier": "soft",
                    "session_turn_cap_hit": True,
                    "task_budget_boundary_hit": False,
                }),
            ],
        )
        d = debug_next_action(st)
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "stopped_by_budget")


def _audit_missing_attempt(ft="precision_failed"):
    """一轮: diagnose + forensics + validate(stop_reason_code=stagnant_audit_missing)。"""
    return [
        _attempt_ev(ft),
        _completed_ev("forensics"),
        _completed_ev("diagnose_and_fix", {"success": True}),
        _completed_ev("validate", {"loop_signal": "CONTINUE",
                                   "stop_reason_code": "stagnant_audit_missing"}),
    ]


def _cheat_attempt(ft="precision_failed", *, errored=False):
    """一轮: validate result.checks 带确证作弊 (anticheat_pass=False)。

    errored=True 时改为 AST validator 异常 (ast_degrade_pass=False + ast_validator_errored)
    = N5 的 warning，N6 不应计入。
    """
    if errored:
        checks = {"ast_degrade_pass": False, "ast_validator_errored": True}
    else:
        checks = {"anticheat_pass": False}
    return [
        _attempt_ev(ft),
        _completed_ev("forensics"),
        _completed_ev("diagnose_and_fix", {"success": True}),
        _completed_ev("validate", {"loop_signal": "CONTINUE", "checks": checks}),
    ]


def _normal_attempt(ft="precision_failed"):
    """一轮正常未通过 (无退化标记) validate。"""
    return [
        _attempt_ev(ft),
        _completed_ev("forensics"),
        _completed_ev("diagnose_and_fix", {"success": True}),
        _completed_ev("validate", {"loop_signal": "CONTINUE", "stop_reason_code": None}),
    ]


class TestDegenerateEarlyStop(unittest.TestCase):
    """N6 复合早停: 连续 N 轮退化空转 (作弊/audit 缺产物兜底) → degenerate_no_progress。

    默认启用 (阈值 2)，仅 violation 计入 (N5 的 validator-errored warning 不计)，
    放在 loop_limit/budget 闸之后。数据全取自 events 持久化的 validate result。
    """

    _ENV = "ASCENDC_DEBUG_MAX_DEGENERATE_ROUNDS"

    def setUp(self) -> None:
        self._saved = os.environ.get(self._ENV)
        os.environ.pop(self._ENV, None)

    def tearDown(self) -> None:
        if self._saved is None:
            os.environ.pop(self._ENV, None)
        else:
            os.environ[self._ENV] = self._saved

    def _state_of(self, attempts_events, ft="precision_failed"):
        evs = []
        for block in attempts_events:
            evs += block
        n = len(attempts_events)
        return _state(current_ft=ft, total_attempts=n, per_branch={ft: n}, events=evs)

    def test_two_audit_missing_stops(self) -> None:
        # 默认阈值 2: 连续 2 轮 audit 缺失兜底 → 早停。
        st = self._state_of([_audit_missing_attempt(), _audit_missing_attempt()])
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Done)
        self.assertEqual(d.session_outcome, "degenerate_no_progress")

    def test_two_cheat_violation_stops(self) -> None:
        st = self._state_of([_cheat_attempt(), _cheat_attempt()])
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertEqual(d.session_outcome, "degenerate_no_progress")

    def test_mixed_audit_and_cheat_stops(self) -> None:
        # 退化口径是「二者之一」，混合连续命中同样累计。
        st = self._state_of([_cheat_attempt(), _audit_missing_attempt()])
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertEqual(d.session_outcome, "degenerate_no_progress")

    def test_one_degenerate_continues(self) -> None:
        # 仅 1 轮退化 (< 阈值 2) → 照常 CONTINUE。
        st = self._state_of([_audit_missing_attempt()])
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Continue)

    def test_normal_round_breaks_streak(self) -> None:
        # 退化轮被正常轮打断 → 末尾游程仅 1，不早停。
        st = self._state_of([_audit_missing_attempt(), _normal_attempt(),
                             _audit_missing_attempt()])
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Continue)

    def test_validator_errored_warning_not_counted(self) -> None:
        # N5: AST validator 异常是 warning，非确证作弊 → 不计退化，不早停。
        st = self._state_of([_cheat_attempt(errored=True),
                             _cheat_attempt(errored=True)])
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Continue)

    def test_disabled_by_env_continues(self) -> None:
        # env<=0 禁用 → 即使连续退化也照常 CONTINUE。
        os.environ[self._ENV] = "0"
        st = self._state_of([_audit_missing_attempt(), _audit_missing_attempt(),
                             _audit_missing_attempt()])
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertIsInstance(d, Continue)

    def test_custom_threshold(self) -> None:
        os.environ[self._ENV] = "3"
        # 2 轮退化 < 3 → 继续。
        st = self._state_of([_audit_missing_attempt(), _audit_missing_attempt()])
        self.assertIsInstance(debug_next_action(st, _gate("CONTINUE")), Continue)

    def test_pass_overrides_degenerate(self) -> None:
        # 即便历史连续退化，本轮 gate PASS → success (终判优先)。
        st = self._state_of([_cheat_attempt(), _cheat_attempt()])
        d = debug_next_action(st, _gate("PASS"))
        self.assertEqual(d.session_outcome, "success")

    def test_loop_limit_precedence(self) -> None:
        # 撞满 5 轮全局闸 + 连续退化 → 归 attempt_limit (在 N6 之前判)。
        blocks = [_cheat_attempt() for _ in range(5)]
        st = self._state_of(blocks)
        d = debug_next_action(st, _gate("CONTINUE"))
        self.assertEqual(d.session_outcome, "stopped_by_attempt_limit")


if __name__ == "__main__":
    unittest.main()
