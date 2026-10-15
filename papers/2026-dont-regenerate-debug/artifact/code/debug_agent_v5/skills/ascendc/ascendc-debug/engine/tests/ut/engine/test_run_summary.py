"""test_run_summary.py — 项 11: run_summary.json 聚合 (turns/gate/anti_cheat/
kb_finalize/forensics)，供 batch report 程序化消费。

验收点 (6.11 文档 Tier 2 项 11):
  - turns 聚合只用 num_turns (agent_turns)，模型无关，跨模型可比。
  - turns 按 attempt 拆分，缺失/非正整数按 0 计。
  - gate 取各轮 validate 的 loop_signal/stop_reason_code + 最终终判信号。
  - anti_cheat 按 severity 计数 violation/warning。
  - forensics 报告哪些 attempt 落了 forensics_report。
  - kb_finalize 透传 status (未配置 kb_path 时为 None)。
  - best-effort: write_run_summary 失败返回 None，不抛异常。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine import transition as tr
from engine.exit_artifacts import build_run_summary, write_run_summary
from engine.types import Action, Continue, Done


def _completed(task_dir, step, result):
    a = Action(kind="py_action", name=step, step=step)
    tr.record_action_started(task_dir, a, f"{step}_id")
    tr.record_action_completed(task_dir, a, f"{step}_id", result)


class TestRunSummary(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)
        (self.task_dir / "precision_tuning").mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _two_attempt_session(self) -> None:
        """造 2 轮 session: 各轮 diagnose 带 agent_turns，validate 带 loop_signal。"""
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_attempt_started(self.task_dir, Continue(0, "precision_failed"))
        _completed(self.task_dir, "forensics", {"success": True})
        _completed(self.task_dir, "diagnose_and_fix", {"success": True, "agent_turns": 30})
        _completed(self.task_dir, "validate",
                   {"loop_signal": "CONTINUE", "stop_reason_code": None})
        tr.record_attempt_started(self.task_dir, Continue(1, "precision_failed"))
        _completed(self.task_dir, "forensics", {"success": True})
        _completed(self.task_dir, "diagnose_and_fix", {"success": True, "agent_turns": 45})
        _completed(self.task_dir, "validate",
                   {"loop_signal": "PASS", "stop_reason_code": None})
        tr.record_decision(self.task_dir, Done(session_outcome="success", reason="ok"))

    def test_turns_total_and_by_attempt(self) -> None:
        self._two_attempt_session()
        s = build_run_summary(self.task_dir)
        self.assertEqual(s["turns"]["total"], 75)
        self.assertEqual(s["turns"]["by_attempt"], {"0": 30, "1": 45})

    def test_no_usd_field_anywhere(self) -> None:
        self._two_attempt_session()
        s = build_run_summary(self.task_dir)
        # 护栏: 输出字段不得包含 usd/cost 字样。
        blob = json.dumps(s, ensure_ascii=False).lower()
        self.assertNotIn("usd", blob)
        self.assertNotIn("cost", blob)

    def test_turns_missing_counts_zero(self) -> None:
        # diagnose 无 agent_turns (timeout 早返回) → 该轮 0，不报错。
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_attempt_started(self.task_dir, Continue(0, "precision_failed"))
        _completed(self.task_dir, "diagnose_and_fix", {"success": False})
        tr.record_decision(self.task_dir, Done(session_outcome="crashed", reason="x"))
        s = build_run_summary(self.task_dir)
        self.assertEqual(s["turns"]["total"], 0)
        self.assertEqual(s["turns"]["by_attempt"], {"0": 0})

    def test_gate_final_and_by_attempt(self) -> None:
        self._two_attempt_session()
        s = build_run_summary(self.task_dir)
        self.assertEqual(s["gate"]["final_loop_signal"], "PASS")
        self.assertEqual(len(s["gate"]["by_attempt"]), 2)
        self.assertEqual(s["gate"]["by_attempt"][0]["loop_signal"], "CONTINUE")

    def test_anti_cheat_counts(self) -> None:
        self._two_attempt_session()
        (self.task_dir / "precision_tuning" / "cheat_history.json").write_text(
            json.dumps({"cheating_attempts": [
                {"attempt": 0, "severity": "violation"},
                {"attempt": 1, "severity": "warning"},
            ]}), encoding="utf-8")
        s = build_run_summary(self.task_dir)
        self.assertEqual(s["anti_cheat"], {"total": 2, "violations": 1, "warnings": 1})

    def test_anti_cheat_absent_is_clean(self) -> None:
        self._two_attempt_session()
        s = build_run_summary(self.task_dir)
        self.assertEqual(s["anti_cheat"], {"total": 0, "violations": 0, "warnings": 0})

    def test_forensics_reports(self) -> None:
        self._two_attempt_session()  # attempts_used=2
        tuning = self.task_dir / "precision_tuning"
        (tuning / "forensics_report_0.json").write_text("{}", encoding="utf-8")
        # 第 1 轮无 report (模拟漏产)。
        s = build_run_summary(self.task_dir)
        self.assertEqual(s["forensics"]["reports"], [0])

    def test_kb_finalize_passthrough(self) -> None:
        self._two_attempt_session()
        status = {"session_outcome": "success", "attempts_used": 2,
                  "session_branch": "1-P", "kb_finalize": {"finalized": True}}
        s = build_run_summary(self.task_dir, status)
        self.assertEqual(s["kb_finalize"], {"finalized": True})

    def test_kb_finalize_none_when_absent(self) -> None:
        self._two_attempt_session()
        s = build_run_summary(self.task_dir)
        self.assertIsNone(s["kb_finalize"])

    def test_write_run_summary_lands_file(self) -> None:
        self._two_attempt_session()
        path = write_run_summary(self.task_dir)
        self.assertIsNotNone(path)
        self.assertTrue((self.task_dir / "run_summary.json").exists())
        data = json.loads((self.task_dir / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["session_outcome"], "success")

    def test_write_run_summary_best_effort_on_bad_path(self) -> None:
        # task_dir 不存在 → 写入失败 → 返回 None，不抛异常。
        bad = self.task_dir / "nonexistent_subdir_xyz"
        # build 仍可 (read_events 对缺失目录返回空)，但 write 落点目录不存在 → None。
        result = write_run_summary(bad)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
