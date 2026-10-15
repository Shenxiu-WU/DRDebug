"""test_state.py — 重放派生正确性 + 双层预算 + 跨分支重置 + resume 幂等。

验收点 (REWRITE_PLAN §6.2): 双层预算派生 (全局+分支计数)、dangling_started 悬挂检测、
resume 幂等。外加 §2.3b 点名必测的 precision↔build 振荡 fixture。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from engine import transition as tr
from engine.events import EventWriter, read_events
from engine.state import Counters, DebugState, derive_counters
from engine.types import Action, Continue


def _attempt(ft: str, n: int) -> dict:
    return {"type": "attempt_started", "attempt": n, "failure_type": ft}


class TestDeriveCounters(unittest.TestCase):
    def test_empty(self) -> None:
        cf = derive_counters([])
        self.assertEqual(cf.total_attempts, 0)
        self.assertEqual(cf.per_branch_attempt, {})
        self.assertIsNone(cf.current_failure_type)

    def test_session_started_sets_entry(self) -> None:
        cf = derive_counters([
            {"type": "session_started", "entry_failure_type": "precision_failed"},
        ])
        self.assertEqual(cf.entry_failure_type, "precision_failed")
        self.assertEqual(cf.current_failure_type, "precision_failed")

    def test_total_attempts_counts_attempt_started(self) -> None:
        cf = derive_counters([
            _attempt("precision_failed", 0),
            _attempt("precision_failed", 1),
            _attempt("precision_failed", 2),
        ])
        self.assertEqual(cf.total_attempts, 3)
        self.assertEqual(cf.per_branch_attempt, {"precision_failed": 3})
        self.assertEqual(cf.current_failure_type, "precision_failed")

    def test_forward_drift_resets_passed_stage(self) -> None:
        cf = derive_counters([
            _attempt("build_failed", 0),
            _attempt("build_failed", 1),
            _attempt("import_failed", 2),
        ])
        # build→import 是「前进」(编译过了、import 才失败): import order(1) > build(0)，
        # 进 import 重置更靠前的 build → build 归零。这正是「旧阶段计数不累计」语义。
        self.assertEqual(cf.per_branch_attempt, {"build_failed": 0, "import_failed": 1})
        self.assertEqual(cf.total_attempts, 3)


class TestCrossBranchReset(unittest.TestCase):
    """REWRITE_PLAN §2.3b 跨分支计数重置 + precision 单调不变量。"""

    def test_enter_precision_resets_upstream(self) -> None:
        # build/import/runtime 各跑一轮后进 precision → 三者归零。
        cf = derive_counters([
            _attempt("build_failed", 0),
            _attempt("import_failed", 1),
            _attempt("runtime_error", 2),
            _attempt("precision_failed", 3),
        ])
        self.assertEqual(cf.per_branch_attempt["build_failed"], 0)
        self.assertEqual(cf.per_branch_attempt["import_failed"], 0)
        self.assertEqual(cf.per_branch_attempt["runtime_error"], 0)
        self.assertEqual(cf.per_branch_attempt["precision_failed"], 1)

    def test_enter_runtime_resets_only_build_import(self) -> None:
        cf = derive_counters([
            _attempt("build_failed", 0),
            _attempt("import_failed", 1),
            _attempt("precision_failed", 2),  # precision 先涨到 1
            _attempt("runtime_error", 3),     # 回 runtime: 重置 build/import，不动 precision
        ])
        self.assertEqual(cf.per_branch_attempt["build_failed"], 0)
        self.assertEqual(cf.per_branch_attempt["import_failed"], 0)
        self.assertEqual(cf.per_branch_attempt["runtime_error"], 1)
        # precision order(3) > runtime order(2) → 不被重置，保持 1。
        self.assertEqual(cf.per_branch_attempt["precision_failed"], 1)

    def test_enter_build_resets_nothing(self) -> None:
        cf = derive_counters([
            _attempt("precision_failed", 0),
            _attempt("build_failed", 1),
        ])
        # build 最靠前，无更靠前分支可重置；precision 不动。
        self.assertEqual(cf.per_branch_attempt["precision_failed"], 1)
        self.assertEqual(cf.per_branch_attempt["build_failed"], 1)

    def test_precision_build_oscillation_monotone(self) -> None:
        # §2.3b 点名 fixture: precision→build→precision→build…
        # 断言 build 计数每次归零、precision 单调递增。
        events = [
            _attempt("precision_failed", 0),  # P=1
            _attempt("build_failed", 1),      # P=1(不动), B=1
            _attempt("precision_failed", 2),  # 进 P 重置 B → B=0, P=2
            _attempt("build_failed", 3),      # P=2(不动), B=1
            _attempt("precision_failed", 4),  # 进 P 重置 B → B=0, P=3
        ]
        # 逐前缀断言 precision 单调不减、build 进 precision 即归零。
        prec_seq = []
        for i in range(1, len(events) + 1):
            cf = derive_counters(events[:i])
            prec_seq.append(cf.per_branch_attempt.get("precision_failed", 0))
        self.assertEqual(prec_seq, [1, 1, 2, 2, 3])  # 单调不减
        # 终态: build 已归零，precision=3。
        cf = derive_counters(events)
        self.assertEqual(cf.per_branch_attempt["build_failed"], 0)
        self.assertEqual(cf.per_branch_attempt["precision_failed"], 3)

    def test_runtime_timeout_same_stage_no_mutual_reset(self) -> None:
        # runtime 与 timeout 同级 (order 都=2)，互切不应重置对方。
        cf = derive_counters([
            _attempt("runtime_error", 0),
            _attempt("timeout", 1),
            _attempt("runtime_error", 2),
        ])
        self.assertEqual(cf.per_branch_attempt["runtime_error"], 2)
        self.assertEqual(cf.per_branch_attempt["timeout"], 1)


class TestLoopSignal(unittest.TestCase):
    def test_last_loop_signal_and_stop_code(self) -> None:
        cf = derive_counters([
            {"type": "action_completed", "result": {"loop_signal": "CONTINUE"}},
            {"type": "action_completed", "result": {"loop_signal": "STOP",
                                                    "stop_reason_code": "fp16_precision_ceiling"}},
        ])
        self.assertEqual(cf.last_loop_signal, "STOP")
        self.assertEqual(cf.last_stop_reason_code, "fp16_precision_ceiling")


class TestDebugStateOnDisk(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_load_from_real_events(self) -> None:
        # 用 transition 写侧 → state 读侧，端到端往返。
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_attempt_started(self.task_dir, Continue(0, "precision_failed"))
        st = DebugState.load(self.task_dir)
        self.assertEqual(st.total_attempts, 1)
        self.assertEqual(st.current_failure_type, "precision_failed")
        self.assertEqual(st.entry_failure_type, "precision_failed")
        self.assertEqual(st.branch_attempt("precision_failed"), 1)

    def test_dangling_started_detects_crash_point(self) -> None:
        act = Action(kind="py_action", name="forensics", step="forensics")
        aid = tr.new_action_id()
        tr.record_action_started(self.task_dir, act, aid)
        # 故意不写 completed → 模拟崩溃。
        st = DebugState.load(self.task_dir)
        dangling = st.dangling_started()
        self.assertEqual(len(dangling), 1)
        self.assertEqual(dangling[0]["action_id"], aid)
        self.assertEqual(dangling[0]["action"]["name"], "forensics")

    def test_dangling_cleared_after_completed(self) -> None:
        act = Action(kind="py_action", name="forensics", step="forensics")
        aid = tr.new_action_id()
        tr.record_action_started(self.task_dir, act, aid)
        tr.record_action_completed(self.task_dir, act, aid, {"ok": True})
        st = DebugState.load(self.task_dir)
        self.assertEqual(st.dangling_started(), [])

    def test_dangling_dedup_same_action_id(self) -> None:
        # 防御: 同一 action_id 两次 started 无 completed → 只返回一个 (去重)。
        EventWriter(self.task_dir).append({
            "type": "action_started", "action_id": "act_dup",
            "action": {"name": "forensics"}})
        EventWriter(self.task_dir).append({
            "type": "action_started", "action_id": "act_dup",
            "action": {"name": "forensics"}})
        dangling = DebugState.load(self.task_dir).dangling_started()
        self.assertEqual(len(dangling), 1)
        self.assertEqual(dangling[0]["action_id"], "act_dup")

    def test_is_terminal(self) -> None:
        from engine.types import Done
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive")
        self.assertFalse(DebugState.load(self.task_dir).is_terminal())
        tr.record_decision(self.task_dir, Done(session_outcome="success", reason="ok"))
        self.assertTrue(DebugState.load(self.task_dir).is_terminal())

    def test_resume_idempotent(self) -> None:
        # 重放幂等: 同一 events.jsonl 多次 load 得到等价 Counters。
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="build_failed")
        tr.record_attempt_started(self.task_dir, Continue(0, "build_failed"))
        tr.record_attempt_started(self.task_dir, Continue(1, "precision_failed"))
        s1 = DebugState.load(self.task_dir)
        s2 = DebugState.load(self.task_dir)
        self.assertEqual(s1.cf, s2.cf)
        # 派生不写回事件 (load 是只读投影)。
        n_before = len(read_events(self.task_dir))
        DebugState.load(self.task_dir)
        self.assertEqual(len(read_events(self.task_dir)), n_before)

    def test_load_equals_derive_on_same_events(self) -> None:
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_attempt_started(self.task_dir, Continue(0, "precision_failed"))
        st = DebugState.load(self.task_dir)
        self.assertEqual(st.cf, derive_counters(read_events(self.task_dir)))


if __name__ == "__main__":
    unittest.main()
