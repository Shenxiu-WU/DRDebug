"""test_gate_adapter.py — gate 输出解析 (纯函数主测面) + subprocess 链路 smoke。

验收点 (REWRITE_PLAN §6.3): 解析现有 precision_gate 输出，不重算 loop_signal。
主测 parse_gate_output 对两种 schema 形状的吸收 + 全部 stop_reason_code；
subprocess 链路用真实脚本做一个 smoke (验证 stdout 多行 JSON + 退出码路径打通)。
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

from engine.gate_adapter import (
    GateResult,
    _extract_first_json,
    parse_gate_output,
    run_gate,
)
from precision_gate import _dispatch, _merge_common_checks  # noqa: E402


class TestParseBuildShape(unittest.TestCase):
    """build/import/runtime/timeout 形状: 顶层 loop_signal，checks 无诊断键。"""

    def test_gate_v_continue(self) -> None:
        raw = {
            "gate": "GATE-BUILD-V",
            "passed": True,
            "checks": {"curr_present": True, "curr_failed_step": "execute"},
            "loop_signal": "CONTINUE",
            "loop_reason": "build progress tracked via failed_step transition",
            "failure_type": "build_failed",
        }
        r = parse_gate_output(raw)
        self.assertEqual(r.gate, "GATE-BUILD-V")
        self.assertTrue(r.passed)
        self.assertEqual(r.loop_signal, "CONTINUE")
        self.assertEqual(r.failure_type, "build_failed")
        # build 分支无 stop_reason_code。
        self.assertIsNone(r.stop_reason_code)
        self.assertIsNone(r.attempt)

    def test_gate_f_no_loop_signal(self) -> None:
        # forensics/audit step 无 loop_signal。
        raw = {"gate": "GATE-BUILD-F", "passed": False,
               "checks": {"latest_present": False}}
        r = parse_gate_output(raw)
        self.assertIsNone(r.loop_signal)
        self.assertFalse(r.passed)


class TestParsePrecisionShape(unittest.TestCase):
    """precision Gate-V 形状: stop_reason_code/attempt/max_attempts 塞在 checks 里。"""

    def _precision_raw(self, stop_code, loop_signal, passed=False) -> dict:
        return {
            "gate": "GATE-V",
            "passed": passed,
            "checks": {
                "prerequisite_code": True,
                "precision_passed": passed,
                "stop_reason_code": stop_code,
                "attempt": 1,
                "max_attempts": 5,
            },
            "loop_signal": loop_signal,
            "loop_reason": "...",
        }

    def test_promotes_stop_reason_code_from_checks(self) -> None:
        r = parse_gate_output(self._precision_raw("nearly_success", "STOP"))
        self.assertEqual(r.stop_reason_code, "nearly_success")
        self.assertEqual(r.attempt, 1)
        self.assertEqual(r.max_attempts, 5)
        self.assertEqual(r.loop_signal, "STOP")

    def test_pass_precision_passed(self) -> None:
        r = parse_gate_output(self._precision_raw("precision_passed", "PASS", passed=True))
        self.assertTrue(r.passed)
        self.assertEqual(r.loop_signal, "PASS")
        self.assertEqual(r.stop_reason_code, "precision_passed")

    def test_all_stop_reason_codes_round_trip(self) -> None:
        # 全部已核实的 stop_reason_code 闭集，确保解析不丢值。
        codes = [
            "precision_passed", "nearly_success", "fp16_precision_ceiling",
            "max_attempts_reached", "harmful_regression",
            "stagnant_new_direction", "stagnant_same_direction",
            "prerequisite_failure", "validation_failed",
        ]
        for c in codes:
            r = parse_gate_output(self._precision_raw(c, "STOP"))
            self.assertEqual(r.stop_reason_code, c)

    def test_prerequisite_failure_shape(self) -> None:
        # 前置失败: prerequisite_error 也在 checks 里 (经 _legacy_to_outcome)。
        raw = {
            "gate": "GATE-V",
            "passed": False,
            "checks": {
                "prerequisite_code": False,
                "prerequisite_error": "kernel 未修改",
                "stop_reason_code": "prerequisite_failure",
            },
            "loop_signal": "STOP",
            "loop_reason": "前置条件不满足: kernel 未修改",
        }
        r = parse_gate_output(raw)
        self.assertEqual(r.prerequisite_error, "kernel 未修改")
        self.assertEqual(r.stop_reason_code, "prerequisite_failure")
        self.assertEqual(r.loop_signal, "STOP")


class TestTopLevelPrecedence(unittest.TestCase):
    def test_top_level_wins_over_checks(self) -> None:
        # 若顶层与 checks 同时有某键 (理论上不会)，顶层优先。
        raw = {
            "gate": "G", "passed": False,
            "stop_reason_code": "top",
            "checks": {"stop_reason_code": "nested"},
        }
        self.assertEqual(parse_gate_output(raw).stop_reason_code, "top")

    def test_checks_default_independent(self) -> None:
        r1 = parse_gate_output({"gate": "a", "passed": True})
        r2 = parse_gate_output({"gate": "b", "passed": True})
        r1.checks["x"] = 1
        self.assertEqual(r2.checks, {})


class TestPrecisionGateCommonMerge(unittest.TestCase):
    def test_branch_output_keeps_common_checks(self) -> None:
        class CommonOutcome:
            checks = {
                "ast_degrade_pass": True,
                "anticheat_pass": True,
                "hash_model_new_ascendc.py": True,
            }

        merged = _merge_common_checks(
            {
                "gate": "GATE-V",
                "passed": True,
                "loop_signal": "PASS",
                "checks": {
                    "precision_passed": True,
                    "stop_reason_code": "precision_passed",
                },
            },
            CommonOutcome(),
        )

        self.assertEqual(merged["gate"], "GATE-V")
        self.assertEqual(merged["loop_signal"], "PASS")
        self.assertTrue(merged["checks"]["ast_degrade_pass"])
        self.assertTrue(merged["checks"]["anticheat_pass"])
        self.assertTrue(merged["checks"]["hash_model_new_ascendc.py"])
        self.assertTrue(merged["checks"]["precision_passed"])
        self.assertEqual(merged["checks"]["stop_reason_code"], "precision_passed")


class TestPreAgentAuditDispatch(unittest.TestCase):
    def _task(self, root: Path, failure_type: str) -> Path:
        task = root / failure_type
        (task / "kernel").mkdir(parents=True)
        (task / "model_new_ascendc.py").write_text(
            "# candidate\n", encoding="utf-8")
        verify = task / ".verify_status"
        verify.mkdir()
        (verify / "latest.json").write_text(json.dumps({
            "schema_version": 1,
            "phase": 8,
            "attempt": 0,
            "failure_type": failure_type,
            "duration_sec": 1.0,
            "exit_code": 1,
            "log_path": str(task / ".verify_logs" / "attempt0.log"),
        }), encoding="utf-8")
        if failure_type == "precision_failed":
            tuning = task / "precision_tuning"
            tuning.mkdir()
            (tuning / "forensics_report_0.json").write_text(json.dumps({
                "status": "completed",
                "attempt": 0,
                "primary_hint": "tail writes are unmasked",
                "version": "2.0",
            }), encoding="utf-8")
        return task

    def test_dispatch_generates_branch_specific_audit_before_common_gate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            expected = {
                "precision_failed": "[FORENSICS_SUMMARY]",
                "build_failed": "[COMPILE_ERROR_CITATION]",
                "import_failed": "[IMPORT_TRACEBACK_CITATION]",
                "runtime_error": "[RUNTIME_ERROR_CITATION]",
                "timeout": "[SYNC_POINT_ANALYSIS]",
            }
            for failure_type, marker in expected.items():
                with self.subTest(failure_type=failure_type):
                    task = self._task(root, failure_type)
                    result = _dispatch("audit", task, "FakeOp", 0)
                    audit = (
                        task
                        / "precision_tuning"
                        / "precision_audit_0.md"
                    )
                    context = (
                        task
                        / "precision_tuning"
                        / "audit_context_attempt_0.json"
                    )
                    self.assertTrue(result["passed"], result)
                    self.assertIn(
                        marker, audit.read_text(encoding="utf-8"))
                    self.assertEqual(
                        json.loads(context.read_text(encoding="utf-8"))[
                            "failure_type"
                        ],
                        failure_type,
                    )


class TestExtractFirstJson(unittest.TestCase):
    def test_multiline_json_then_human_line(self) -> None:
        # 复刻 precision_gate.py main(): indent=2 多行 JSON + 后续人类可读行。
        stdout = (
            '{\n'
            '  "gate": "GATE-BUILD-V",\n'
            '  "passed": true,\n'
            '  "loop_signal": "CONTINUE"\n'
            '}\n'
            '\n[GATE-BUILD-V] ✅ PASSED\n'
        )
        obj = _extract_first_json(stdout)
        self.assertEqual(obj["gate"], "GATE-BUILD-V")
        self.assertEqual(obj["loop_signal"], "CONTINUE")

    def test_no_json_raises(self) -> None:
        with self.assertRaises(ValueError):
            _extract_first_json("traceback: boom\n")


class TestRunGateSmoke(unittest.TestCase):
    """subprocess 链路真实打通 (不依赖 NPU): 缺产物 → common 层 FAILED (退出码 1)。"""

    def test_run_gate_common_failure(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            task_dir = Path(d)
            (task_dir / "kernel").mkdir()
            # 缺 model_new_ascendc.py → common 层 has_model_new_ascendc=False → FAILED。
            r = run_gate(task_dir, step="forensics", op_name="add", attempt=0, timeout=60)
        self.assertEqual(r.gate, "GATE-COMMON-forensics")
        self.assertFalse(r.passed)
        # common 层失败无 loop_signal。
        self.assertIsNone(r.loop_signal)
        self.assertFalse(r.checks.get("has_model_new_ascendc"))


if __name__ == "__main__":
    unittest.main()
