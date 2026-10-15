"""test_common_anticheat.py — 修复4: anti-cheat 分场景 (A/B) + N5 + 去重。

验收点 (6.11 文档修复4):
  - objective success + 作弊 → STOP + stop_reason_code=cheat_detected (B 场景)
  - objective fail + 作弊    → CONTINUE (A 场景，保留诊断成本)
  - validator 异常 (errored) → 不确证作弊，不据此终止/续跑 (N5)
  - 非 validate step (forensics) → 仅记录，不派终判信号
  - cheat_history 按 (attempt, cheat_type) 去重
  - cheat STOP 信号经 to_gate_output → parse_gate_output 能提升 stop_reason_code

gates 在 scripts/ 下 (非 engine 包)，测试自插 scripts 到 sys.path (镜像
precision_gate.py 的 bootstrap)，独立于 PYTHONPATH。
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from gates import common  # noqa: E402
from gates.common import run_common  # noqa: E402


def _make_task(tmp: Path, *, correctness_passed=None) -> Path:
    """造一个最小合法 task_dir: kernel/ + model_new_ascendc.py + verify_status + audit。

    correctness_passed 非 None 时写 validation_result_attempt_0.json (供 validate step
    读 objective 结果)。
    """
    task = tmp / "005_FakeOp"
    (task / "kernel").mkdir(parents=True)
    (task / "model_new_ascendc.py").write_text("x = 1\n", encoding="utf-8")
    vs = task / ".verify_status"
    vs.mkdir()
    (vs / "latest.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    tuning = task / "precision_tuning"
    tuning.mkdir()
    (tuning / "precision_audit_0.md").write_text("x" * 300, encoding="utf-8")
    if correctness_passed is not None:
        (tuning / "validation_result_attempt_0.json").write_text(
            json.dumps({"correctness_passed": correctness_passed}), encoding="utf-8")
    return task


class TestAntiCheatBranching(unittest.TestCase):
    """A/B 分场景: objective success/fail × 作弊。"""

    def _run_validate_with_cheat(self, correctness_passed, *, errored=False):
        """跑 validate step，并把 ast_degrade_pass 强制为 fail (模拟作弊)。"""
        with tempfile.TemporaryDirectory() as d:
            task = _make_task(Path(d), correctness_passed=correctness_passed)
            orig = common.check_ast_degrade
            def fake_ast(_task_dir):
                out = {"ast_validator_present": True, "ast_degrade_pass": False}
                if errored:
                    out["ast_validator_errored"] = True
                return out
            common.check_ast_degrade = fake_ast
            try:
                outcome = run_common("validate", task, "FakeOp", 0)
                history = task / "precision_tuning" / "cheat_history.json"
                hist = (json.loads(history.read_text()) if history.exists()
                        else {"cheating_attempts": []})
                return outcome, hist
            finally:
                common.check_ast_degrade = orig

    def test_success_plus_cheat_is_stop_cheat_detected(self) -> None:
        # B 场景: objective success + 作弊 → STOP + cheat_detected。
        outcome, _ = self._run_validate_with_cheat(True)
        out = outcome.to_gate_output()
        self.assertFalse(out["passed"])
        self.assertEqual(out["loop_signal"], "STOP")
        self.assertEqual(out["checks"]["stop_reason_code"], "cheat_detected")

    def test_fail_plus_cheat_is_continue(self) -> None:
        # A 场景: objective fail + 作弊 → CONTINUE (保留诊断成本)。
        outcome, _ = self._run_validate_with_cheat(False)
        out = outcome.to_gate_output()
        self.assertEqual(out["loop_signal"], "CONTINUE")
        self.assertNotIn("stop_reason_code", out.get("checks", {}))

    def test_validator_errored_not_confirmed_cheat(self) -> None:
        # N5: validator 异常 + objective success → 不确证作弊，不派 STOP/CONTINUE。
        outcome, hist = self._run_validate_with_cheat(True, errored=True)
        out = outcome.to_gate_output()
        # 不据 errored 终止: loop_signal 不应是 cheat 的 STOP。
        self.assertNotEqual(out.get("checks", {}).get("stop_reason_code"), "cheat_detected")
        # 但 N5 要求仍记 warning，不静默吞。
        entries = hist["cheating_attempts"]
        self.assertTrue(any(e["cheat_type"] == "AST_VALIDATOR_ERROR" for e in entries))
        self.assertTrue(any(e["severity"] == "warning" for e in entries))

    def test_records_cheat_history(self) -> None:
        _, hist = self._run_validate_with_cheat(False)
        entries = hist["cheating_attempts"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["cheat_type"], "AST_DEGRADE")
        self.assertEqual(entries[0]["severity"], "violation")


class TestCppRegressionGate(unittest.TestCase):
    """C++ 扫描进 Gate 热路径: success+cpp_cheat→STOP / fail+cpp_cheat→CONTINUE，仅 validate step。"""

    def _run_with_cpp(self, step, correctness_passed, *, cpp_status="fail"):
        """跑指定 step，把 check_cpp_regression 强制为指定 status (模拟 C++ 作弊)。"""
        with tempfile.TemporaryDirectory() as d:
            task = _make_task(Path(d), correctness_passed=correctness_passed)
            orig = common.check_cpp_regression
            common.check_cpp_regression = lambda _t: {
                "cpp_checker_present": True,
                "cpp_regression_pass": cpp_status != "fail",
                "cpp_violations": 2 if cpp_status == "fail" else 0,
            }
            try:
                outcome = run_common(step, task, "FakeOp", 0)
                history = task / "precision_tuning" / "cheat_history.json"
                hist = (json.loads(history.read_text()) if history.exists()
                        else {"cheating_attempts": []})
                return outcome, hist
            finally:
                common.check_cpp_regression = orig

    def test_cpp_success_plus_cheat_is_stop(self) -> None:
        # objective success + C++ 作弊 → STOP + cheat_detected。
        outcome, hist = self._run_with_cpp("validate", True)
        out = outcome.to_gate_output()
        self.assertFalse(out["passed"])
        self.assertEqual(out["loop_signal"], "STOP")
        self.assertEqual(out["checks"]["stop_reason_code"], "cheat_detected")
        self.assertTrue(any(e["cheat_type"] == "CPP_REGRESSION"
                            for e in hist["cheating_attempts"]))

    def test_cpp_fail_plus_cheat_is_continue(self) -> None:
        # objective fail + C++ 作弊 → CONTINUE (保留诊断成本)。
        outcome, _ = self._run_with_cpp("validate", False)
        out = outcome.to_gate_output()
        self.assertEqual(out["loop_signal"], "CONTINUE")
        self.assertNotIn("stop_reason_code", out.get("checks", {}))

    def test_cpp_scan_skipped_on_forensics(self) -> None:
        # 非 validate step 不跑 C++ 扫描: forensics 不该读到 cpp_regression_pass。
        outcome, _ = self._run_with_cpp("forensics", None)
        self.assertNotIn("cpp_regression_pass", outcome.checks)

    def test_cpp_pass_no_cheat(self) -> None:
        # C++ 扫描通过 → 不记 cheat，validate step 正常通过。
        outcome, hist = self._run_with_cpp("validate", True, cpp_status="pass")
        out = outcome.to_gate_output()
        self.assertTrue(out["passed"])
        self.assertEqual(hist["cheating_attempts"], [])


class TestForensicsStepDefersVerdict(unittest.TestCase):
    """非 validate step (forensics) 检测到作弊只记录，不派终判信号 (无 objective 结论)。"""

    def test_forensics_cheat_records_no_signal(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            task = _make_task(Path(d))
            orig = common.check_ast_degrade
            common.check_ast_degrade = lambda _t: {
                "ast_validator_present": True, "ast_degrade_pass": False}
            try:
                outcome = run_common("forensics", task, "FakeOp", 0)
                self.assertIsNone(outcome.loop_signal)
                history = task / "precision_tuning" / "cheat_history.json"
                self.assertTrue(history.exists())  # 仍记录
            finally:
                common.check_ast_degrade = orig


class TestCheatHistoryDedup(unittest.TestCase):
    """同一 (attempt, cheat_type) 去重 (同轮 forensics+validate 两次 run_common)。"""

    def test_dedup_same_attempt_type(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            task = _make_task(Path(d), correctness_passed=False)
            checks = {"ast_degrade_pass": False, "anticheat_pass": True}
            common._record_cheat_attempt(task, 0, checks, cheat_type="AST_DEGRADE")
            common._record_cheat_attempt(task, 0, checks, cheat_type="AST_DEGRADE")
            hist = json.loads(
                (task / "precision_tuning" / "cheat_history.json").read_text())
            self.assertEqual(len(hist["cheating_attempts"]), 1)

    def test_different_type_not_deduped(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            task = _make_task(Path(d), correctness_passed=False)
            checks = {"ast_degrade_pass": False, "anticheat_pass": False}
            common._record_cheat_attempt(task, 0, checks, cheat_type="AST_DEGRADE")
            common._record_cheat_attempt(task, 0, checks, cheat_type="WRAPPER_HASH")
            hist = json.loads(
                (task / "precision_tuning" / "cheat_history.json").read_text())
            self.assertEqual(len(hist["cheating_attempts"]), 2)


class TestAntiCheatDetectOnly(unittest.TestCase):
    """detect-only (baseline arm): ANTICHEAT_DETECT_ONLY=1 检测+记录照常，但不阻断。

    带作弊的 success 放行 (不再 STOP cheat_detected)，cheat_history 仍留证。与默认
    阻断态对比 → 量化反作弊拦截的虚假成功数。env 用 addCleanup 隔离防泄漏污染其他用例。
    """

    def _set_detect_only(self, value) -> None:
        import os
        prev = os.environ.get("ANTICHEAT_DETECT_ONLY")
        if value is None:
            os.environ.pop("ANTICHEAT_DETECT_ONLY", None)
        else:
            os.environ["ANTICHEAT_DETECT_ONLY"] = value

        def _restore() -> None:
            if prev is None:
                os.environ.pop("ANTICHEAT_DETECT_ONLY", None)
            else:
                os.environ["ANTICHEAT_DETECT_ONLY"] = prev
        self.addCleanup(_restore)

    def _run_validate_cheat(self, *, correctness_passed=True):
        with tempfile.TemporaryDirectory() as d:
            task = _make_task(Path(d), correctness_passed=correctness_passed)
            orig = common.check_ast_degrade
            common.check_ast_degrade = lambda _t: {
                "ast_validator_present": True, "ast_degrade_pass": False}
            try:
                outcome = run_common("validate", task, "FakeOp", 0)
                history = task / "precision_tuning" / "cheat_history.json"
                hist = (json.loads(history.read_text()) if history.exists()
                        else {"cheating_attempts": []})
                return outcome, hist
            finally:
                common.check_ast_degrade = orig

    def test_detect_only_success_plus_cheat_not_blocked(self) -> None:
        # detect-only: success + 作弊 → 不 STOP，放行 (与 B 场景默认阻断对照)。
        self._set_detect_only("1")
        outcome, hist = self._run_validate_cheat()
        out = outcome.to_gate_output()
        self.assertNotEqual(out.get("checks", {}).get("stop_reason_code"), "cheat_detected")
        self.assertNotEqual(out.get("loop_signal"), "STOP")
        # 检测+记录仍在: cheat_history 留证 + 标记 would_block。
        self.assertTrue(any(e["cheat_type"] == "AST_DEGRADE"
                            for e in hist["cheating_attempts"]))
        self.assertTrue(out["checks"].get("would_block_cheat_detected"))

    def test_detect_only_failure_plus_cheat_does_not_force_continue(self) -> None:
        self._set_detect_only("1")
        outcome, hist = self._run_validate_cheat(correctness_passed=False)
        out = outcome.to_gate_output()
        self.assertIsNone(out.get("loop_signal"))
        self.assertFalse(out["checks"].get("would_block_cheat_detected"))
        self.assertTrue(hist["cheating_attempts"])

    def test_default_still_blocks_when_detect_only_off(self) -> None:
        # 对照: 不设 detect-only → 维持原 STOP cheat_detected。
        self._set_detect_only(None)
        outcome, _ = self._run_validate_cheat()
        out = outcome.to_gate_output()
        self.assertEqual(out["loop_signal"], "STOP")
        self.assertEqual(out["checks"]["stop_reason_code"], "cheat_detected")


if __name__ == "__main__":
    unittest.main()
