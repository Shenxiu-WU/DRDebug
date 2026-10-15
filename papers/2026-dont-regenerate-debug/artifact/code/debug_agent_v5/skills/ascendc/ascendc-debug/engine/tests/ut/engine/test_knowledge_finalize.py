"""test_knowledge_finalize.py — KB 入库编排 (修复问题 6)。

验收: finalize 条件门控 (outcome==success && cheat_history clean && 候选存在) +
action 透传 (new/merge/abandon) + 子进程失败/缺脚本降级为 skip + 默认 kb_path=None 禁用。
用 fake _run 替换 subprocess.run，不真跑 precision_knowledge.py。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine.knowledge_finalize import finalize_knowledge


class _FakeRun:
    """记录 dump 子进程调用；可编程 returncode/stderr。不真跑脚本。"""

    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)

        class _Proc:
            pass
        p = _Proc()
        p.returncode = self.returncode
        p.stderr = self.stderr
        p.stdout = ""
        return p


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


class TestKnowledgeFinalize(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)
        self.tuning = self.task_dir / "precision_tuning"
        # 默认放一个合法候选 (个别用例覆盖/删除)。
        _write(self.tuning / "candidate_kb_entry.json",
               {"title": "t", "type": "FIX_PRECISION_TAIL"})

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _call(self, *, outcome="success", kb_path="kb.json", run=None):
        run = run or _FakeRun()
        res = finalize_knowledge(self.task_dir, kb_path=kb_path,
                                 session_outcome=outcome, op_name="add", _run=run)
        return res, run

    def test_no_kb_path_disabled(self) -> None:
        res, run = self._call(kb_path=None)
        self.assertFalse(res["finalized"])
        self.assertEqual(len(run.calls), 0)

    def test_non_success_skips(self) -> None:
        res, run = self._call(outcome="failed")
        self.assertFalse(res["finalized"])
        self.assertEqual(len(run.calls), 0)

    def test_missing_candidate_skips(self) -> None:
        (self.tuning / "candidate_kb_entry.json").unlink()
        res, run = self._call()
        self.assertFalse(res["finalized"])
        self.assertEqual(len(run.calls), 0)

    def test_cheat_violation_blocks(self) -> None:
        _write(self.tuning / "cheat_history.json",
               {"cheating_attempts": [{"attempt": 1, "severity": "violation"}]})
        res, run = self._call()
        self.assertFalse(res["finalized"])
        self.assertEqual(len(run.calls), 0)

    def test_ast_warning_blocks(self) -> None:
        # AST validator 异常 (severity=warning, ast_status=unknown) 也阻止入库
        # (文档 6.5: unknown 不等价 pass → success_unverified，保守不入库)。
        _write(self.tuning / "cheat_history.json",
               {"cheating_attempts": [{"attempt": 1, "severity": "warning",
                                       "cheat_type": "AST_VALIDATOR_ERROR"}]})
        res, run = self._call()
        self.assertFalse(res["finalized"])
        self.assertEqual(len(run.calls), 0)

    def test_success_clean_runs_dump(self) -> None:
        res, run = self._call()
        self.assertTrue(res["finalized"])
        self.assertEqual(len(run.calls), 1)
        cmd = run.calls[0]
        self.assertIn("dump", cmd)
        self.assertIn("--action", cmd)
        self.assertIn("new", cmd)

    def test_abandon_candidate_skips(self) -> None:
        _write(self.tuning / "candidate_kb_entry.json",
               {"title": "t", "type": "FIX_PRECISION_TAIL", "action": "abandon"})
        res, run = self._call()
        self.assertFalse(res["finalized"])
        self.assertEqual(len(run.calls), 0)

    def test_merge_passes_target_title(self) -> None:
        _write(self.tuning / "candidate_kb_entry.json",
               {"title": "t", "type": "FIX_PRECISION_TAIL", "action": "merge",
                "merge_target_title": "existing-entry"})
        res, run = self._call()
        self.assertTrue(res["finalized"])
        cmd = run.calls[0]
        self.assertIn("--merge-target-title", cmd)
        self.assertIn("existing-entry", cmd)

    def test_dump_failure_skips(self) -> None:
        res, run = self._call(run=_FakeRun(returncode=1, stderr="boom"))
        self.assertFalse(res["finalized"])
        self.assertEqual(len(run.calls), 1)  # 调了但失败
        self.assertIn("boom", res["reason"])

    def test_result_written_to_disk(self) -> None:
        self._call()
        out = self.task_dir / "precision_tuning" / "kb_finalize_result.json"
        self.assertTrue(out.exists())

    def test_read_only_kb_never_invokes_dump(self) -> None:
        with mock.patch.dict(
                os.environ, {"ASCENDC_DEBUG_KB_READ_ONLY": "1"}):
            res, run = self._call()
        self.assertTrue(res["skipped"])
        self.assertIn("READ_ONLY", res["reason"])
        self.assertEqual(run.calls, [])

    def test_no_kb_ablation_never_invokes_dump(self) -> None:
        with mock.patch.dict(os.environ, {"ABLATE_KB": "1"}):
            res, run = self._call()
        self.assertTrue(res["skipped"])
        self.assertIn("ABLATE_KB", res["reason"])
        self.assertEqual(run.calls, [])


class TestFullEvalAdmits(unittest.TestCase):
    """§2.2 入库判据: 全量复验下不 crash 且 matched_ratio ≥ 0.95 才入库。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)
        self.tuning = self.task_dir / "precision_tuning"
        self.tuning.mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_vr(self, attempt: int, full_eval) -> None:
        vr = {"attempt": attempt, "correctness_passed": True}
        if full_eval is not None:
            vr["full_eval"] = full_eval
        (self.tuning / f"validation_result_attempt_{attempt}.json").write_text(
            json.dumps(vr), encoding="utf-8")

    def _admits(self):
        from engine.knowledge_finalize import _full_eval_admits  # noqa: PLC0415
        return _full_eval_admits(self.task_dir)

    def test_no_full_eval_admits(self) -> None:
        self._write_vr(0, None)
        ok, _r = self._admits()
        self.assertTrue(ok)

    def test_crash_rejected(self) -> None:
        self._write_vr(0, {"ran": True, "crashed": True,
                           "passed_cases": 0, "total_cases": 0})
        ok, _r = self._admits()
        self.assertFalse(ok)

    def test_below_threshold_rejected(self) -> None:
        self._write_vr(0, {"ran": True, "crashed": False,
                           "passed_cases": 45, "total_cases": 51})  # 0.882 < 0.95
        ok, _r = self._admits()
        self.assertFalse(ok)

    def test_above_threshold_admits(self) -> None:
        self._write_vr(0, {"ran": True, "crashed": False,
                           "passed_cases": 50, "total_cases": 51})  # 0.980 ≥ 0.95
        ok, _r = self._admits()
        self.assertTrue(ok)

    def test_latest_attempt_wins(self) -> None:
        # 多 attempt 取最大 attempt 的 full_eval (终态轮)。早轮达标、终轮不达标 → 挡。
        self._write_vr(0, {"ran": True, "crashed": False,
                           "passed_cases": 51, "total_cases": 51})
        self._write_vr(2, {"ran": True, "crashed": False,
                           "passed_cases": 40, "total_cases": 51})  # 0.784
        ok, _r = self._admits()
        self.assertFalse(ok)

    def test_finalize_skips_when_full_eval_fails(self) -> None:
        # 集成: 全量未过 → finalize_knowledge skip (不调 dump 子进程)。
        _write(self.tuning / "candidate_kb_entry.json", {"title": "t"})
        self._write_vr(0, {"ran": True, "crashed": True,
                           "passed_cases": 0, "total_cases": 0})
        run = _FakeRun()
        res = finalize_knowledge(self.task_dir, kb_path="kb.json",
                                 session_outcome="success", op_name="add", _run=run)
        self.assertTrue(res["skipped"])
        self.assertIn("全量复验判据未过", res["reason"])
        self.assertEqual(run.calls, [])  # 未触达 dump


if __name__ == "__main__":
    unittest.main()
