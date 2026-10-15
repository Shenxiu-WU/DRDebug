"""test_runner.py — mock dispatcher 跑完整 session 到各终态 + timeout 闸。

验收点 (REWRITE_PLAN §6.5): mock agent 跑完整 session 到各终态；timeout 闸 (超时主动
emit 终态事件，不裸死)。不依赖 NPU——dispatcher 注入 mock。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine.events import read_events
from engine.runner import RunnerError, run_debug_session, validate_action
from engine.state import DebugState
from engine.types import Action, Continue


class _MockDispatcher:
    """可编程 dispatcher: 按 (step) 返回预设 result。validate 步返回指定 loop_signal。"""

    def __init__(self, validate_results):
        # validate_results: list[dict] 依次作为每次 validate 的 result。
        self._validate = list(validate_results)
        self._vi = 0
        self.calls = []

    def __call__(self, action: Action, task_dir, op_name, agent_callback):
        self.calls.append((action.kind, action.step))
        if action.step == "validate":
            r = self._validate[min(self._vi, len(self._validate) - 1)]
            self._vi += 1
            return dict(r)
        if action.kind == "spawn_agent":
            return {"success": True, "marker": {"outcome": "needs_retry"}}
        # forensics 等 py_action
        return {"success": True}


def _gate(loop_signal, stop_reason_code=None, import_subtype=None):
    d = {"gate": "GATE-V", "passed": loop_signal == "PASS",
         "loop_signal": loop_signal, "stop_reason_code": stop_reason_code}
    if import_subtype:
        d["checks"] = {"import_subtype": import_subtype}
    return d


class TestChangedFilesDiff(unittest.TestCase):
    """项 12b: _default_dispatcher 在 spawn_agent 前后快照 kernel，diff 出 changed_files。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)
        (self.task_dir / "kernel").mkdir()
        (self.task_dir / "kernel" / "k.h").write_text("v0", encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _dispatch(self, mutate):
        from engine.runner import _default_dispatcher
        action = Action(kind="spawn_agent", name="debug_worker",
                        step="diagnose_and_fix", skill_args={"attempt": 0})

        def _cb(a, td, on, at):
            mutate()
            return {"success": True}
        return _default_dispatcher(action, self.task_dir, "add", _cb)

    def test_changed_files_detected(self) -> None:
        r = self._dispatch(
            lambda: (self.task_dir / "kernel" / "k.h").write_text("v1", encoding="utf-8"))
        self.assertEqual(r["changed_files"], ["kernel/k.h"])

    def test_no_change_empty_list(self) -> None:
        r = self._dispatch(lambda: None)
        self.assertEqual(r["changed_files"], [])

    def test_new_file_detected(self) -> None:
        r = self._dispatch(
            lambda: (self.task_dir / "kernel" / "new.cpp").write_text("x", encoding="utf-8"))
        self.assertEqual(r["changed_files"], ["kernel/new.cpp"])


class TestForensicsObservability(unittest.TestCase):
    def test_default_dispatcher_preserves_reuse_metadata(self) -> None:
        import engine.runner as runner
        from engine.gate_adapter import GateResult

        action = Action(
            kind="py_action", name="precision_gate", step="forensics",
            skill_args={"gate_step": "forensics", "attempt": 3},
        )
        forensics_result = {
            "success": True,
            "cache_hit": True,
            "reuse_kind": "rollback",
            "forensics_reused": True,
            "forensics_executed": False,
            "forensics_completed": True,
            "prebuild_executed": False,
            "build_skipped": True,
            "build_skip_reason": "rollback_forensics_reuse",
            "reused_after_rollback": True,
            "reused_from_attempt": 1,
            "report_path": "/tmp/forensics_report_3.json",
        }
        with tempfile.TemporaryDirectory() as d, \
                mock.patch("engine.validate_runner.run_forensics",
                           return_value=forensics_result), \
                mock.patch.object(
                    runner, "run_gate",
                    return_value=GateResult(gate="GATE-F", passed=True)):
            result = runner._default_dispatcher(action, Path(d), "FakeOp", None)

        self.assertTrue(result["forensics_reused"])
        self.assertEqual(result["reuse_kind"], "rollback")
        self.assertTrue(result["build_skipped"])
        self.assertEqual(result["forensics"], forensics_result)

    def test_default_dispatcher_preserves_runtime_degrade_metadata(self) -> None:
        import engine.runner as runner
        from engine.gate_adapter import GateResult

        action = Action(
            kind="py_action", name="precision_gate", step="forensics",
            skill_args={
                "gate_step": "forensics",
                "attempt": 1,
                "failure_type": "runtime_error",
            },
        )
        forensics_result = {
            "success": True,
            "forensics_unavailable": True,
            "unavailable_reason": "runtime_error",
            "forensics_degraded": True,
            "diagnostic_evidence_kind": "runtime_error_log",
            "proceed_to_agent": True,
            "forensics_executed": True,
            "forensics_completed": False,
            "report_path": "/tmp/forensics_report_1.json",
        }
        with tempfile.TemporaryDirectory() as d, \
                mock.patch("engine.validate_runner.run_forensics",
                           return_value=forensics_result), \
                mock.patch.object(
                    runner, "run_gate",
                    return_value=GateResult(gate="GATE-F", passed=False)):
            result = runner._default_dispatcher(action, Path(d), "FakeOp", None)

        self.assertTrue(result["forensics_degraded"])
        self.assertTrue(result["proceed_to_agent"])
        self.assertEqual(result["unavailable_reason"], "runtime_error")
        self.assertEqual(result["diagnostic_evidence_kind"], "runtime_error_log")
        self.assertEqual(result["forensics"], forensics_result)

    def test_result_to_gate_preserves_objective_failure_type(self) -> None:
        import engine.runner as runner

        gate = runner._result_to_gate({
            "gate": "GATE-V",
            "passed": True,
            "loop_signal": "CONTINUE",
            "failure_type": "build_failed",
        })
        self.assertIsNotNone(gate)
        self.assertEqual(gate.failure_type, "build_failed")


class TestValidateAction(unittest.TestCase):
    def test_unknown_py_action_rejected(self) -> None:
        with self.assertRaises(RunnerError):
            validate_action(Action(kind="py_action", name="evil"))

    def test_unknown_agent_rejected(self) -> None:
        with self.assertRaises(RunnerError):
            validate_action(Action(kind="spawn_agent", name="evil"))

    def test_known_actions_ok(self) -> None:
        validate_action(Action(kind="py_action", name="precision_gate"))
        validate_action(Action(kind="spawn_agent", name="debug_worker"))


class TestFullSession(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, validate_results, **kw):
        disp = _MockDispatcher(validate_results)
        status = run_debug_session(
            self.task_dir, op_name="add", agent="constructive",
            entry_failure_type=kw.pop("entry", "precision_failed"),
            dispatcher=disp, **kw)
        return status, disp

    def test_session_to_success(self) -> None:
        # 首轮 PASS → success。
        status, disp = self._run([_gate("PASS")])
        self.assertEqual(status["session_outcome"], "success")
        # 原始 baseline → round → durable checkpoint/rollback barrier。
        steps = [s for _, s in disp.calls]
        self.assertEqual(steps, ["baseline_checkpoint", "forensics",
                                 "knowledge_search", "audit", "diagnose_and_fix",
                                 "validate", "checkpoint_and_rollback"])

    def test_baseline_profile_has_no_hidden_recovery_or_evidence_steps(self) -> None:
        env = {
            "ABLATE_FORENSICS": "1",
            "ABLATE_KB": "1",
            "ABLATE_GATE_A": "1",
            "ABLATE_RECOVERY": "1",
        }
        with mock.patch.dict("os.environ", env):
            status, dispatcher = self._run([_gate("PASS")])
        self.assertEqual(status["session_outcome"], "success")
        self.assertEqual(
            [step for _, step in dispatcher.calls],
            ["diagnose_and_fix", "validate"],
        )

    def test_session_continue_then_pass(self) -> None:
        # 第一轮 CONTINUE，第二轮 PASS → success，2 轮。
        status, _ = self._run([_gate("CONTINUE"), _gate("PASS")])
        self.assertEqual(status["session_outcome"], "success")
        self.assertEqual(status["attempts_used"], 2)

    def test_session_stop_nearly_success_to_failed(self) -> None:
        status, _ = self._run([_gate("STOP", "nearly_success")])
        self.assertEqual(status["session_outcome"], "failed")

    def test_session_stop_prerequisite_to_gate(self) -> None:
        status, _ = self._run([_gate("STOP", "prerequisite_failure")])
        self.assertEqual(status["session_outcome"], "stopped_by_gate")

    def test_session_stop_max_attempts(self) -> None:
        status, _ = self._run([_gate("STOP", "max_attempts_reached")])
        self.assertEqual(status["session_outcome"], "stopped_by_attempt_limit")

    def test_probe_policy_violation_terminates_ablation_arm(self) -> None:
        class _D(_MockDispatcher):
            def __call__(self, action, task_dir, op_name, agent_callback):
                self.calls.append((action.kind, action.step))
                if action.kind == "spawn_agent":
                    return {
                        "success": True,
                        "ablation_violation": True,
                        "probe_record_path": "/tmp/policy.json",
                        "probe_source_audit_path": "/tmp/source.json",
                    }
                return {"success": True}

        dispatcher = _D([_gate("PASS")])
        status = run_debug_session(
            self.task_dir,
            op_name="add",
            agent="constructive",
            entry_failure_type="precision_failed",
            dispatcher=dispatcher,
        )
        self.assertEqual(status["session_outcome"], "ablation_violation")

    def test_exit_artifacts_written(self) -> None:
        self._run([_gate("PASS")])
        self.assertTrue((self.task_dir / "debug_status.json").exists())
        self.assertTrue((self.task_dir / "debug_trace.md").exists())

    def test_events_self_consistent_no_dangling(self) -> None:
        # 正常跑完，无悬挂 started。
        self._run([_gate("PASS")])
        st = DebugState.load(self.task_dir)
        self.assertEqual(st.dangling_started(), [])
        self.assertTrue(st.is_terminal())

    def test_import_env_side_skipped_via_feed(self) -> None:
        # 首轮 forensics 喂回 import_subtype=env_side → 第二拍 next_action 判 skipped。
        # mock: forensics 步返回带 import_subtype 的 result。
        class _D(_MockDispatcher):
            def __call__(self, action, task_dir, op_name, agent_callback):
                self.calls.append((action.kind, action.step))
                if action.step == "forensics":
                    return {"success": True, "import_subtype": "import_env_side"}
                return super().__call__(action, task_dir, op_name, agent_callback)
        disp = _D([_gate("PASS")])
        status = run_debug_session(
            self.task_dir, op_name="add", entry_failure_type="import_failed",
            dispatcher=disp)
        self.assertEqual(status["session_outcome"], "skipped_env_issue")

    def test_kb_finalize_hooked_on_success(self) -> None:
        # kb_path 传入时 _terminate 接入 finalize_knowledge，status 带 kb_finalize 字段。
        # 无候选 → skip (不真跑 dump 子进程)，验证 hook 接通即可。
        status, _ = self._run([_gate("PASS")], kb_path="kb.json")
        self.assertEqual(status["session_outcome"], "success")
        self.assertIn("kb_finalize", status)
        self.assertFalse(status["kb_finalize"]["finalized"])

    def test_kb_finalize_absent_without_kb_path(self) -> None:
        # 不传 kb_path (默认) → status 无 kb_finalize 字段 (向后兼容)。
        status, _ = self._run([_gate("PASS")])
        self.assertNotIn("kb_finalize", status)

    def test_run_summary_written_on_terminate(self) -> None:
        # 项 11: 终态出口落 run_summary.json，且不含 usd 字段。
        import json
        self._run([_gate("CONTINUE"), _gate("PASS")])
        path = self.task_dir / "run_summary.json"
        self.assertTrue(path.exists())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["session_outcome"], "success")
        self.assertEqual(data["attempts_used"], 2)
        self.assertEqual(data["gate"]["final_loop_signal"], "PASS")
        self.assertNotIn("usd", json.dumps(data).lower())


class TestTimeoutGate(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_deadline_emits_timeout_terminal(self) -> None:
        # 注入单调时钟: 第一次取时刻 0，之后每次 +100s → 立刻超 deadline。
        ticks = iter([0, 100, 200, 300, 400, 500])
        clock = lambda: next(ticks)
        disp = _MockDispatcher([_gate("CONTINUE")] * 10)
        status = run_debug_session(
            self.task_dir, op_name="add", entry_failure_type="precision_failed",
            dispatcher=disp, deadline_sec=50, _now=clock)
        self.assertEqual(status["session_outcome"], "timeout")
        # timeout 是主动 emit 的终态事件，events 自洽。
        evs = read_events(self.task_dir)
        self.assertEqual(evs[-1]["type"], "session_done")
        self.assertEqual(evs[-1]["session_outcome"], "timeout")

    def test_timeout_midway_has_no_dangling(self) -> None:
        # 先跑若干拍 (deadline 较大)，中途超时 → 终态前无悬挂 started，events 自洽。
        # 时钟: start=0, 之后阶梯增长，在跑过一轮后越过 deadline。
        seq = [0, 1, 2, 3, 4, 5, 100, 100, 100, 100]
        it = iter(seq)
        clock = lambda: next(it)
        disp = _MockDispatcher([_gate("CONTINUE")] * 10)
        status = run_debug_session(
            self.task_dir, op_name="add", entry_failure_type="precision_failed",
            dispatcher=disp, deadline_sec=50, _now=clock)
        self.assertEqual(status["session_outcome"], "timeout")
        st = DebugState.load(self.task_dir)
        # 每个 action_started 都有配对 completed (timeout 在拍间检查，不打断 action)。
        self.assertEqual(st.dangling_started(), [])
        self.assertTrue(st.is_terminal())

    def test_no_deadline_runs_normally(self) -> None:
        disp = _MockDispatcher([_gate("PASS")])
        status = run_debug_session(
            self.task_dir, op_name="add", entry_failure_type="precision_failed",
            dispatcher=disp, deadline_sec=None)
        self.assertEqual(status["session_outcome"], "success")


class TestResume(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_resume_does_not_duplicate_session_started(self) -> None:
        # 第一次跑到 success。
        run_debug_session(self.task_dir, op_name="add",
                          entry_failure_type="precision_failed",
                          dispatcher=_MockDispatcher([_gate("PASS")]))
        n_session_started = sum(
            1 for e in read_events(self.task_dir) if e["type"] == "session_started")
        # 再次调用 (resume): 不应重复 session_started。
        run_debug_session(self.task_dir, op_name="add",
                          dispatcher=_MockDispatcher([_gate("PASS")]))
        n_after = sum(
            1 for e in read_events(self.task_dir) if e["type"] == "session_started")
        self.assertEqual(n_session_started, 1)
        self.assertEqual(n_after, 1)

    def test_terminal_session_reinvoke_appends_nothing(self) -> None:
        # H4: 已终态 session 再次调用 → 不追加任何事件，直接返回原 status。
        run_debug_session(self.task_dir, op_name="add",
                          entry_failure_type="precision_failed",
                          dispatcher=_MockDispatcher([_gate("PASS")]))
        n_before = len(read_events(self.task_dir))
        status = run_debug_session(self.task_dir, op_name="add",
                                   dispatcher=_MockDispatcher([_gate("PASS")]))
        n_after = len(read_events(self.task_dir))
        self.assertEqual(n_before, n_after)  # 终态后不追加
        self.assertEqual(status["session_outcome"], "success")
        # events 仍以终态事件结尾。
        self.assertEqual(read_events(self.task_dir)[-1]["type"], "session_done")

    def test_resume_dangling_spawn_agent_not_re_dispatched(self) -> None:
        # H3: 崩在 diagnose_and_fix(spawn_agent) 的 started 之后、completed 之前。
        # resume 时 runner 须 reconcile 该 dangling (补 failed completed)，
        # 而非重新拉起 agent → 第二次改 kernel。
        import engine.transition as tr
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_attempt_started(self.task_dir, Continue(
            next_attempt=0, next_failure_type="precision_failed", reason="x"))
        # forensics 正常完成。
        fa = Action(kind="py_action", name="precision_gate", step="forensics")
        fid = tr.new_action_id()
        tr.record_action_started(self.task_dir, fa, fid)
        tr.record_action_completed(self.task_dir, fa, fid, {"success": True})
        # diagnose_and_fix 起了但没完成 (崩溃残留)。
        da = Action(kind="spawn_agent", name="debug_worker", step="diagnose_and_fix")
        did = tr.new_action_id()
        tr.record_action_started(self.task_dir, da, did)

        # resume: 编程 dispatcher 记录每次 spawn_agent 调用。
        spawn_calls = []

        class _Track(_MockDispatcher):
            def __call__(self, action, task_dir, op_name, agent_callback):
                if action.kind == "spawn_agent":
                    spawn_calls.append(action.step)
                return super().__call__(action, task_dir, op_name, agent_callback)

        status = run_debug_session(self.task_dir, op_name="add",
                                   dispatcher=_Track([_gate("PASS")]))
        # dangling 的那次 diagnose_and_fix 不应被「续上」——它已被 reconcile 为 failed，
        # 下一轮是干净重派。本轮 spawn_agent 至多被调用一次 (干净重试)，绝不是续跑叠加。
        st = DebugState.load(self.task_dir)
        self.assertEqual(st.dangling_started(), [])  # 无悬挂残留
        self.assertEqual(status["session_outcome"], "success")


class TestResumeDanglingReconcile(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_dangling_reconciled_with_failed_completed(self) -> None:
        # 单测 reconcile 本身: 崩在 spawn_agent started 后，resume 入口补一条
        # action_completed{success:false}，使该 action_id 不再 dangling。
        import engine.transition as tr
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_attempt_started(self.task_dir, Continue(
            next_attempt=0, next_failure_type="precision_failed", reason="x"))
        da = Action(kind="spawn_agent", name="debug_worker", step="diagnose_and_fix")
        did = tr.new_action_id()
        tr.record_action_started(self.task_dir, da, did)
        # 崩溃前确认确有 dangling。
        self.assertEqual(len(DebugState.load(self.task_dir).dangling_started()), 1)
        run_debug_session(self.task_dir, op_name="add",
                          dispatcher=_MockDispatcher([_gate("PASS")]))
        # resume 后该 dangling 被补 completed，且该补的 completed 标 success=False。
        evs = read_events(self.task_dir)
        recon = [e for e in evs if e.get("type") == "action_completed"
                 and e.get("action_id") == did]
        self.assertEqual(len(recon), 1)
        self.assertFalse(recon[0]["result"].get("success"))


if __name__ == "__main__":
    unittest.main()
