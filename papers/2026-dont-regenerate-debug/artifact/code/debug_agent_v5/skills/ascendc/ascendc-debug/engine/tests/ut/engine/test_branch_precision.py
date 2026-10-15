"""test_branch_precision.py — 12a: precision_audit 方向评估缺失兜底。

验收点 (6.11 文档 Tier 2 项 12a):
  - audit 产出缺失 (文件/marker/section 不存在) → _check_direction_assessment 返回
    "missing"，stagnant 分支映射 CONTINUE (stop_reason_code=stagnant_audit_missing)，
    不据缺失误判 STOP 提前掐断诊断。
  - audit 存在但答案模糊 (非 是/否) → "unknown" → 沿用同方向 STOP (原行为不变)。
  - audit 明确 否/是 → "continue"/"stop" (原行为不变)。

gates 在 scripts/ 下 (非 engine 包)，测试自插 scripts 到 sys.path (镜像
test_common_anticheat.py 的 bootstrap)，独立于 PYTHONPATH。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from gates.branch_precision import _LegacyPrecisionChecker  # noqa: E402

# stagnant 分支触发所需: mismatch 连续未改善，且未达 fp16/max_attempts 早停。
_STAGNANT_FORENSICS = {
    "history_trend": {
        "mismatch_improving": False,
        "trend": [
            {"mismatch_ratio": 0.05},
            {"mismatch_ratio": 0.05},
            {"mismatch_ratio": 0.05},
        ],
    }
}


def _audit_path(task: Path, attempt: int) -> Path:
    return task / "precision_tuning" / f"precision_audit_{attempt}.md"


class TestDirectionAssessmentState(unittest.TestCase):
    """_check_direction_assessment 四态判定。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task = Path(self._tmp.name) / "005_FakeOp"
        (self.task / "precision_tuning").mkdir(parents=True)
        self.checker = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=2)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_file_absent_is_missing(self) -> None:
        self.assertEqual(self.checker._check_direction_assessment(), "missing")

    def test_no_marker_is_missing(self) -> None:
        _audit_path(self.task, 2).write_text("无方向 section 的正文\n", encoding="utf-8")
        self.assertEqual(self.checker._check_direction_assessment(), "missing")

    def test_empty_section_is_missing(self) -> None:
        _audit_path(self.task, 2).write_text(
            "[DIRECTION_ASSESSMENT]\n[NEXT]\n", encoding="utf-8")
        self.assertEqual(self.checker._check_direction_assessment(), "missing")

    def test_ambiguous_answer_is_unknown(self) -> None:
        _audit_path(self.task, 2).write_text(
            "[DIRECTION_ASSESSMENT]\n本轮是否延续上一轮方向: 不确定，待定\n",
            encoding="utf-8")
        self.assertEqual(self.checker._check_direction_assessment(), "unknown")

    def test_answer_no_is_continue(self) -> None:
        _audit_path(self.task, 2).write_text(
            "[DIRECTION_ASSESSMENT]\n本轮是否延续上一轮方向: 否 换新方向\n",
            encoding="utf-8")
        self.assertEqual(self.checker._check_direction_assessment(), "continue")

    def test_answer_yes_is_stop(self) -> None:
        _audit_path(self.task, 2).write_text(
            "[DIRECTION_ASSESSMENT]\n本轮是否延续上一轮方向: 是 继续深挖\n",
            encoding="utf-8")
        self.assertEqual(self.checker._check_direction_assessment(), "stop")


class TestStagnantLoopSignalMapping(unittest.TestCase):
    """stagnant 分支 direction 态 → loop_signal 映射 (12a 核心)。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task = Path(self._tmp.name) / "005_FakeOp"
        (self.task / "precision_tuning").mkdir(parents=True)
        # attempt=2: 未达 MAX_ATTEMPTS(5) 早停; 无上一轮 forensics 文件 → fp16 不触发。
        self.checker = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=2)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _signal(self) -> tuple:
        return self.checker._compute_loop_signal(
            passed=False, match_rate=80.0, forensics_data=_STAGNANT_FORENSICS)

    def test_missing_audit_continues(self) -> None:
        # 不写 audit → missing → 兜底 CONTINUE。
        sig, _reason, code = self._signal()
        self.assertEqual(sig, "CONTINUE")
        self.assertEqual(code, "stagnant_audit_missing")

    def test_unknown_audit_stops(self) -> None:
        _audit_path(self.task, 2).write_text(
            "[DIRECTION_ASSESSMENT]\n本轮是否延续上一轮方向: 模糊回答\n",
            encoding="utf-8")
        sig, _reason, code = self._signal()
        self.assertEqual(sig, "STOP")
        self.assertEqual(code, "stagnant_same_direction")

    def test_continue_audit_continues(self) -> None:
        _audit_path(self.task, 2).write_text(
            "[DIRECTION_ASSESSMENT]\n本轮是否延续上一轮方向: 否\n", encoding="utf-8")
        sig, _reason, code = self._signal()
        self.assertEqual(sig, "CONTINUE")
        self.assertEqual(code, "stagnant_new_direction")

    def test_stop_audit_stops(self) -> None:
        _audit_path(self.task, 2).write_text(
            "[DIRECTION_ASSESSMENT]\n本轮是否延续上一轮方向: 是\n", encoding="utf-8")
        sig, _reason, code = self._signal()
        self.assertEqual(sig, "STOP")
        self.assertEqual(code, "stagnant_same_direction")

    def test_no_loopguard_disables_stagnant_direction_stop(self) -> None:
        _audit_path(self.task, 2).write_text(
            "[DIRECTION_ASSESSMENT]\n本轮是否延续上一轮方向: 是\n",
            encoding="utf-8")
        previous = os.environ.get("ABLATE_LOOP_GUARD")
        os.environ["ABLATE_LOOP_GUARD"] = "1"
        try:
            sig, _reason, code = self._signal()
        finally:
            if previous is None:
                os.environ.pop("ABLATE_LOOP_GUARD", None)
            else:
                os.environ["ABLATE_LOOP_GUARD"] = previous
        self.assertEqual(sig, "CONTINUE")
        self.assertIsNone(code)

    def test_no_loopguard_disables_fp16_ceiling_and_harmful_regression(self) -> None:
        previous = os.environ.get("ABLATE_LOOP_GUARD")
        os.environ["ABLATE_LOOP_GUARD"] = "1"
        original_fp16 = self.checker._check_fp16_ceiling
        original_regression = self.checker._detect_harmful_regression
        self.checker._check_fp16_ceiling = lambda _data: True
        self.checker._detect_harmful_regression = lambda _trend: True
        try:
            sig, _reason, code = self._signal()
        finally:
            self.checker._check_fp16_ceiling = original_fp16
            self.checker._detect_harmful_regression = original_regression
            if previous is None:
                os.environ.pop("ABLATE_LOOP_GUARD", None)
            else:
                os.environ["ABLATE_LOOP_GUARD"] = previous
        self.assertEqual(sig, "CONTINUE")
        self.assertIsNone(code)


# ---------------------------------------------------------------------------
# L5_PROBE 真探针优先治理 (优先真探针 / 连续失败才回退 / 回退留痕)
# ---------------------------------------------------------------------------
_L5_EXECUTED = ("[L5_PROBE]\n状态: 已执行\n"
                "P1 (CopyIn 后, DeQue 后读取): x0=0.5000 x1=0.7031 len=128\n"
                "P2 (计算中点): x0=1.2500\n"
                "P3 (CopyOut 前): N/A\n")
_L5_SKIPPED = ("[L5_PROBE]\n状态: 跳过（理由: 上一轮已覆盖该疑似阶段，性质未变）\n"
               "P1: N/A\nP2: N/A\nP3: N/A\n")
_L5_MISSING_VALUES = ("[L5_PROBE]\n状态: 已执行\n"
                      "P1: N/A\nP2: N/A\nP3: N/A\n")
# SKILL 模板原样残留: 状态行同含"已执行 / 跳过"两词 + 值行 <...> 未填占位。
# 子串匹配两头不讨好——治理须把这种残留判为 missing (逼真探针)，不得误判 skipped/executed。
_L5_TEMPLATE_RESIDUE = ("[L5_PROBE]\n状态: 已执行 / 跳过（理由: <若跳过填理由>）\n"
                        "P1 (CopyIn 后): <x[0]=... x[1]=... len=...>\n"
                        "P2 (计算中点): <x[0]=...>\n"
                        "P3 (CopyOut 前): <x[0]=...>\n")
# 混合行: 同一阶段行既有真实测值又含 N/A (部分阶段无值)。不得被整行作废误判 missing。
_L5_MIXED_VALUE = ("[L5_PROBE]\n状态: 已执行\n"
                   "P1 (CopyIn 后): x[0]=0.5000 x[1]=0.7031 / P3 阶段 N/A\n"
                   "P2: N/A\nP3: N/A\n")
_INSTRUMENTATION = ("[INSTRUMENTATION_FINDINGS]\n"
                    "对 case[4] 多 seed 运行，candidate 与 FP32 模拟 max diff = 0。\n")


def _full_audit(l5_block: str, *, instrumentation: bool = False) -> str:
    """构造一份各 section 齐备 (除 L5_PROBE 由 l5_block 控制) 的 audit md。

    attempt>0 也能 passed：含合法 DIRECTION_ASSESSMENT (二元 否)。
    """
    parts = [
        "=== PRECISION AUDIT REPORT ===",
        "[FORENSICS_SUMMARY]\nprimary_hint: magnitude" + " pad" * 40,
        "[COMPUTATION_DECOMPOSITION]\n拆解",
        "[REFERENCE_IMPL_SPEC]\n参考",
        "[KERNEL_STEP_TRACE]\nK1...",
    ]
    if l5_block:
        parts.append(l5_block)
    parts += [
        "[ROOT_CAUSE]\n根因",
        "[CAUSAL_CHAIN_ANALYSIS]\n链路",
        "[FIX_PLAN]\nFIX_PRECISION_CAST",
        "[TARGET_FILES]\nkernel/k.h",
        "[EXPERIMENT_RESULTS]\n实验",
        "[DIRECTION_ASSESSMENT]\n本轮是否延续上一轮方向: 否 换方向\n",
    ]
    if instrumentation:
        parts.append(_INSTRUMENTATION)
    return "\n\n".join(parts) + "\n"


class TestL5ProbeStatus(unittest.TestCase):
    """_l5_probe_status 三态: executed(有实测值) / skipped(带理由) / missing。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task = Path(self._tmp.name) / "005_FakeOp"
        (self.task / "precision_tuning").mkdir(parents=True)
        self.checker = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=0)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_executed_with_measured_value(self) -> None:
        self.assertEqual(self.checker._l5_probe_status(_full_audit(_L5_EXECUTED)),
                         "executed")

    def test_skipped_with_reason(self) -> None:
        self.assertEqual(self.checker._l5_probe_status(_full_audit(_L5_SKIPPED)),
                         "skipped")

    def test_no_section_is_missing(self) -> None:
        self.assertEqual(self.checker._l5_probe_status(_full_audit("")), "missing")

    def test_executed_but_all_na_is_missing(self) -> None:
        # 状态写"已执行"但 P1/P2/P3 全 N/A → 无真数据 → missing (防模板占位糊弄)。
        self.assertEqual(
            self.checker._l5_probe_status(_full_audit(_L5_MISSING_VALUES)), "missing")


class TestConsecutiveProbeFailures(unittest.TestCase):
    """_consecutive_probe_failures: 回看历史轮连续 missing 数 (不含本轮)。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task = Path(self._tmp.name) / "005_FakeOp"
        (self.task / "precision_tuning").mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, attempt: int, l5_block: str) -> None:
        _audit_path(self.task, attempt).write_text(_full_audit(l5_block),
                                                    encoding="utf-8")

    def test_zero_when_no_history(self) -> None:
        c = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=0)
        self.assertEqual(c._consecutive_probe_failures(), 0)

    def test_counts_consecutive_missing(self) -> None:
        self._write(0, "")          # missing
        self._write(1, "")          # missing
        c = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=2)
        self.assertEqual(c._consecutive_probe_failures(), 2)

    def test_executed_breaks_streak(self) -> None:
        self._write(0, "")              # missing (更早)
        self._write(1, _L5_EXECUTED)    # executed → break
        c = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=2)
        self.assertEqual(c._consecutive_probe_failures(), 0)

    def test_absent_history_file_breaks(self) -> None:
        # attempt_1 缺文件 (provider/budget 异常) → break，不误计。
        self._write(0, "")
        c = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=2)
        self.assertEqual(c._consecutive_probe_failures(), 0)


class TestCheckAuditL5ProbeFirst(unittest.TestCase):
    """check_audit 真探针优先 + 连续失败才回退 + degraded 留痕 (集成)。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task = Path(self._tmp.name) / "005_FakeOp"
        (self.task / "precision_tuning").mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _forensics(self, attempt: int) -> None:
        path = self.task / "precision_tuning" / f"forensics_report_{attempt}.json"
        path.write_text(json.dumps({"status": "completed", "attempt": attempt}),
                        encoding="utf-8")

    def _audit(self, attempt: int, l5_block: str, *, instrumentation: bool = False) -> None:
        _audit_path(self.task, attempt).write_text(
            _full_audit(l5_block, instrumentation=instrumentation), encoding="utf-8")

    def test_executed_passes_no_degraded(self) -> None:
        self._forensics(0)
        self._audit(0, _L5_EXECUTED)
        c = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=0)
        r = c.check_audit()
        self.assertTrue(r["passed"])
        self.assertTrue(r["checks"]["has_l5_probe"])
        self.assertNotIn("l5_probe_degraded", r)

    def test_single_missing_not_passed_no_fallback(self) -> None:
        # 仅本轮 missing (无历史) → 不回退 → has_l5_probe=False → passed=False。
        self._forensics(0)
        self._audit(0, "", instrumentation=True)
        c = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=0)
        r = c.check_audit()
        self.assertFalse(r["checks"]["has_l5_probe"])
        self.assertNotIn("l5_probe_degraded", r)

    def test_consecutive_missing_triggers_fallback(self) -> None:
        # 上一轮 missing + 本轮 missing (连续 2 = 阈值) + 有 INSTRUMENTATION → 回退 + degraded。
        self._audit(0, "")               # 历史 missing
        self._forensics(1)
        self._audit(1, "", instrumentation=True)
        c = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=1)
        r = c.check_audit()
        self.assertTrue(r["checks"]["has_l5_probe"])
        self.assertTrue(r.get("l5_probe_degraded"))
        self.assertEqual(r.get("l5_probe_source"), "INSTRUMENTATION_FINDINGS")

    def test_fallback_without_instrumentation_stays_failed(self) -> None:
        # 连续 missing 但连 INSTRUMENTATION 都没有 → 无可兜底 → 仍失败。
        self._audit(0, "")
        self._forensics(1)
        self._audit(1, "")               # 无 instrumentation
        c = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=1)
        r = c.check_audit()
        self.assertFalse(r["checks"]["has_l5_probe"])
        self.assertNotIn("l5_probe_degraded", r)


class TestEngineGeneratedGateAAudit(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task = Path(self._tmp.name) / "005_FakeOp"
        (self.task / "precision_tuning").mkdir(parents=True)
        (self.task / "kernel").mkdir()
        (self.task / "kernel" / "fake.cpp").write_text(
            "// candidate\n", encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_forensics_packet_generates_gate_a_and_context_before_agent(
        self,
    ) -> None:
        tuning = self.task / "precision_tuning"
        (tuning / "forensics_report_0.json").write_text(json.dumps({
            "status": "completed",
            "attempt": 0,
            "primary_hint": "tail writes are unmasked",
            "version": "2.0",
        }), encoding="utf-8")

        result = _LegacyPrecisionChecker(
            "FakeOp", str(self.task), attempt=0).check_audit()

        self.assertTrue(result["passed"])
        self.assertTrue((tuning / "precision_audit_0.md").is_file())
        context = json.loads(
            (tuning / "audit_context_attempt_0.json").read_text(
                encoding="utf-8")
        )
        self.assertEqual(context["generated_by"], "engine_pre_agent_audit")
        self.assertEqual(context["primary_hint"], "tail writes are unmasked")
        self.assertTrue(context["source_parseable"])

    def test_no_forensics_uses_raw_validation_packet(self) -> None:
        verify = self.task / ".verify_status"
        verify.mkdir()
        (verify / "latest.json").write_text(json.dumps({
            "failure_type": "precision_failed",
            "objective_passed": False,
        }), encoding="utf-8")

        with mock.patch.dict(os.environ, {"ABLATE_FORENSICS": "1"}):
            result = _LegacyPrecisionChecker(
                "FakeOp", str(self.task), attempt=0).check_audit()

        self.assertTrue(result["passed"])
        context = json.loads(
            (
                self.task
                / "precision_tuning"
                / "audit_context_attempt_0.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(context["source_type"], "raw_validation")
        self.assertEqual(context["primary_hint"], "precision_failed")


class TestWriteAuditIndexFallback(unittest.TestCase):
    """_write_audit_index: degraded 时 l5_probe 取兜底来源内容 + section_sources 留痕。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task = Path(self._tmp.name) / "005_FakeOp"
        (self.task / "precision_tuning").mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _index(self, attempt: int) -> dict:
        path = self.task / "precision_tuning" / f"round_summary_{attempt}.json"
        return json.loads(path.read_text(encoding="utf-8"))["index"]

    def test_normal_no_section_sources(self) -> None:
        c = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=0)
        c._write_audit_index(_full_audit(_L5_EXECUTED))
        idx = self._index(0)
        self.assertIsNotNone(idx["sections"]["l5_probe"])
        self.assertNotIn("section_sources", idx)

    def test_fallback_indexes_instrumentation_and_records_source(self) -> None:
        c = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=1)
        content = _full_audit("", instrumentation=True)
        c._write_audit_index(content, l5_probe_source="INSTRUMENTATION_FINDINGS")
        idx = self._index(1)
        # l5_probe 索引非空 (内容取自 INSTRUMENTATION_FINDINGS)。
        self.assertIsNotNone(idx["sections"]["l5_probe"])
        self.assertEqual(idx["section_sources"]["l5_probe"], "INSTRUMENTATION_FINDINGS")
        # 落盘的 l5_probe 小文件内容应来自 INSTRUMENTATION_FINDINGS。
        sec_file = self.task / "precision_tuning" / "history" / "attempt_1" / "sections" / "l5_probe.md"
        self.assertIn("INSTRUMENTATION_FINDINGS", sec_file.read_text(encoding="utf-8"))


class TestL5ProbeStatusEdgeCases(unittest.TestCase):
    """B/C/占位 修复的状态级针对性回归 (模板残留 / 混合行 / <> 占位)。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task = Path(self._tmp.name) / "005_FakeOp"
        (self.task / "precision_tuning").mkdir(parents=True)
        self.checker = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=0)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_template_residue_is_missing(self) -> None:
        # B+占位: 状态行"已执行 / 跳过"两词俱在 + 值行全 <...> 占位 → 不得误判 → missing。
        self.assertEqual(
            self.checker._l5_probe_status(_full_audit(_L5_TEMPLATE_RESIDUE)), "missing")

    def test_mixed_value_line_is_executed(self) -> None:
        # C: P1 行同含真值与"N/A"子串 → 不得整行作废 → executed。
        self.assertEqual(
            self.checker._l5_probe_status(_full_audit(_L5_MIXED_VALUE)), "executed")

    def test_skipped_breaks_consecutive_count(self) -> None:
        # 合法 skipped 轮夹在 missing 之间 → 打断连续 missing 计数。
        _audit_path(self.task, 0).write_text(_full_audit(""), encoding="utf-8")            # missing (更早)
        _audit_path(self.task, 1).write_text(_full_audit(_L5_SKIPPED), encoding="utf-8")   # skipped → break
        c = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=2)
        self.assertEqual(c._consecutive_probe_failures(), 0)


class TestCheckAuditSkippedAndDegradedChecks(unittest.TestCase):
    """check_audit: 合法 skipped 集成通过 + degraded 不进 checks 不污染 passed。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task = Path(self._tmp.name) / "005_FakeOp"
        (self.task / "precision_tuning").mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _forensics(self, attempt: int) -> None:
        path = self.task / "precision_tuning" / f"forensics_report_{attempt}.json"
        path.write_text(json.dumps({"status": "completed", "attempt": attempt}),
                        encoding="utf-8")

    def _audit(self, attempt: int, l5_block: str, *, instrumentation: bool = False) -> None:
        _audit_path(self.task, attempt).write_text(
            _full_audit(l5_block, instrumentation=instrumentation), encoding="utf-8")

    def test_skipped_passes_via_check_audit(self) -> None:
        # 合法跳过 (带理由) 经 check_audit 集成 → has_l5_probe=True, passed, 无 degraded。
        self._forensics(0)
        self._audit(0, _L5_SKIPPED)
        r = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=0).check_audit()
        self.assertTrue(r["checks"]["has_l5_probe"])
        self.assertTrue(r["passed"])
        self.assertNotIn("l5_probe_degraded", r)

    def test_degraded_marker_not_in_checks(self) -> None:
        # degraded/source 记 gate_result 顶层，绝不进 checks (否则污染 all(checks.values()))。
        self._audit(0, "")
        self._forensics(1)
        self._audit(1, "", instrumentation=True)
        r = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=1).check_audit()
        self.assertTrue(r.get("l5_probe_degraded"))
        self.assertNotIn("l5_probe_degraded", r["checks"])
        self.assertNotIn("l5_probe_source", r["checks"])
        # degraded 兜底后该轮整体应能 passed (其余 section 齐备)。
        self.assertTrue(r["passed"])


class TestL5ProbeDegradedReachesEvents(unittest.TestCase):
    """盲区根源: run_gate_a → to_gate_output → parse_gate_output → _gate_result_to_dict
    端到端，degraded/source 标记必须落到 events result dict (UT 直调 check_audit 时
    绕过了这条适配器链，故此前 5 个 bug 中最隐蔽的一个在这层)。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task = Path(self._tmp.name) / "005_FakeOp"
        (self.task / "precision_tuning").mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _forensics(self, attempt: int) -> None:
        path = self.task / "precision_tuning" / f"forensics_report_{attempt}.json"
        path.write_text(json.dumps({"status": "completed", "attempt": attempt}),
                        encoding="utf-8")

    def _audit(self, attempt: int, l5_block: str, *, instrumentation: bool = False) -> None:
        _audit_path(self.task, attempt).write_text(
            _full_audit(l5_block, instrumentation=instrumentation), encoding="utf-8")

    def _events_dict(self, attempt: int) -> dict:
        from gates.branch_precision import PrecisionBranch  # noqa: PLC0415
        from engine.gate_adapter import parse_gate_output    # noqa: PLC0415
        from engine.runner import _gate_result_to_dict       # noqa: PLC0415
        outcome = PrecisionBranch("FakeOp").run_gate_a(self.task, attempt)
        gr = parse_gate_output(outcome.to_gate_output())
        return _gate_result_to_dict(gr)

    def test_degraded_source_reaches_events(self) -> None:
        self._audit(0, "")
        self._forensics(1)
        self._audit(1, "", instrumentation=True)
        ev = self._events_dict(1)
        self.assertTrue(ev["l5_probe_degraded"])
        self.assertEqual(ev["l5_probe_source"], "INSTRUMENTATION_FINDINGS")
        # checks 通道也应携带 (顶层摊平与 checks 透传双保险)。
        self.assertEqual(ev["checks"]["l5_probe_source"], "INSTRUMENTATION_FINDINGS")

    def test_real_probe_events_no_degraded_marker(self) -> None:
        # 真探针通过 → events 中 degraded/source 应为 None (不误标兜底)。
        self._forensics(0)
        self._audit(0, _L5_EXECUTED)
        ev = self._events_dict(0)
        self.assertTrue(ev["passed"])
        self.assertIsNone(ev["l5_probe_degraded"])
        self.assertIsNone(ev["l5_probe_source"])


# ---------------------------------------------------------------------------
# 全量复验闸 (§2.1): _compute_loop_signal 读 validation_result 的 full_eval 子字段
# 决定 PASS / CONTINUE(full_eval_regression) / STOP(nearly_success)。
# ---------------------------------------------------------------------------
class TestFullEvalLoopSignal(unittest.TestCase):
    """轻量结果经全量复验复核: 消假阳性 / 推 near-success / 防爆回落。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task = Path(self._tmp.name) / "005_FakeOp"
        self.tuning = self.task / "precision_tuning"
        self.tuning.mkdir(parents=True)
        self.checker = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=0)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_full_eval(self, attempt: int, full_eval, *, state=None) -> None:
        vr = {"attempt": attempt, "correctness_passed": False}
        if full_eval is not None:
            vr["full_eval"] = full_eval
        (self.tuning / f"validation_result_attempt_{attempt}.json").write_text(
            json.dumps(vr), encoding="utf-8")
        if state is not None:
            (self.tuning / ".full_eval_state.json").write_text(
                json.dumps(state), encoding="utf-8")

    def _sig(self, passed, match_rate=None):
        return self.checker._compute_loop_signal(passed=passed, match_rate=match_rate,
                                                 forensics_data=None)

    def test_passed_no_full_eval_graceful_pass(self) -> None:
        # 轻量过 + 无全量产物 (闸关/only_py) → 维持原 PASS。
        sig, _r, code = self._sig(True)
        self.assertEqual((sig, code), ("PASS", "precision_passed"))

    def test_passed_full_eval_100_pass(self) -> None:
        self._write_full_eval(0, {"ran": True, "crashed": False,
                                  "passed_cases": 51, "total_cases": 51, "match_rate": 100.0})
        sig, _r, code = self._sig(True)
        self.assertEqual((sig, code), ("PASS", "precision_passed"))

    def test_passed_full_eval_incomplete_continues(self) -> None:
        # 假阳性②: 轻量过但全量未满 + 可定位 → CONTINUE 带样本继续。
        self._write_full_eval(0, {"ran": True, "crashed": False,
                                  "passed_cases": 48, "total_cases": 51, "match_rate": 94.1})
        sig, _r, code = self._sig(True)
        self.assertEqual((sig, code), ("CONTINUE", "full_eval_regression"))

    def test_nearly_success_full_eval_100_promotes_pass(self) -> None:
        # 推 near-success 过线: 轻量近通过 (mr≥99) + 全量100% → PASS。
        self._write_full_eval(0, {"ran": True, "crashed": False,
                                  "passed_cases": 50, "total_cases": 50, "match_rate": 100.0})
        sig, _r, code = self._sig(False, match_rate=99.5)
        self.assertEqual((sig, code), ("PASS", "precision_passed"))

    def test_nearly_success_no_full_eval_keeps_stop(self) -> None:
        # 向后兼容: 闸关/无全量集 → 维持原 nearly_success STOP。
        sig, _r, code = self._sig(False, match_rate=99.5)
        self.assertEqual((sig, code), ("STOP", "nearly_success"))

    def test_full_eval_crash_continues(self) -> None:
        # 全量 crash (更广输入跑不起来) → CONTINUE 带错误继续。
        self._write_full_eval(0, {"ran": True, "crashed": True,
                                  "passed_cases": 0, "total_cases": 0, "match_rate": 0.0})
        sig, _r, code = self._sig(True)
        self.assertEqual((sig, code), ("CONTINUE", "full_eval_regression"))

    def test_full_eval_stagnant_falls_back_stop(self) -> None:
        # 防爆: 连续 2 轮全量无改善 (state.attempts_done 已 2 轮, best 不被本轮超过) → STOP nearly_success。
        self._write_full_eval(
            0,
            {"ran": True, "crashed": False, "passed_cases": 49, "total_cases": 51,
             "match_rate": 96.0},
            state={"attempts_done": [0, 1], "best_full_match_rate": 96.0})
        sig, _r, code = self._sig(True)
        self.assertEqual((sig, code), ("STOP", "nearly_success"))

    def test_full_eval_improved_continues_despite_rounds(self) -> None:
        # 已做 2 轮但本轮全量有改善 (curr 96.0 > best 90.0 + EPS) → 仍 CONTINUE，不回落。
        self._write_full_eval(
            0,
            {"ran": True, "crashed": False, "passed_cases": 49, "total_cases": 51,
             "match_rate": 96.0},
            state={"attempts_done": [0, 1], "best_full_match_rate": 90.0})
        sig, _r, code = self._sig(True)
        self.assertEqual((sig, code), ("CONTINUE", "full_eval_regression"))

    def _set_loop_guard(self, value) -> None:
        # env 隔离: addCleanup 还原原值，防泄漏污染同文件其他用例。
        prev = os.environ.get("ABLATE_LOOP_GUARD")
        if value is None:
            os.environ.pop("ABLATE_LOOP_GUARD", None)
        else:
            os.environ["ABLATE_LOOP_GUARD"] = value

        def _restore() -> None:
            if prev is None:
                os.environ.pop("ABLATE_LOOP_GUARD", None)
            else:
                os.environ["ABLATE_LOOP_GUARD"] = prev
        self.addCleanup(_restore)

    def test_full_eval_stagnant_no_loopguard_continues(self) -> None:
        # no_loopguard arm: 与 test_full_eval_stagnant_falls_back_stop 同输入，
        # 但 ABLATE_LOOP_GUARD=1 → 抑制 nearly_success 早停，续跑至 max_attempts 硬顶 (与 :349 对齐)。
        self._set_loop_guard("1")
        self._write_full_eval(
            0,
            {"ran": True, "crashed": False, "passed_cases": 49, "total_cases": 51,
             "match_rate": 96.0},
            state={"attempts_done": [0, 1], "best_full_match_rate": 96.0})
        sig, _r, code = self._sig(True)
        self.assertEqual((sig, code), ("CONTINUE", "full_eval_regression"))

    def test_full_eval_no_locatable_no_loopguard_continues(self) -> None:
        # :435 退化边界 (无可定位失败样本): 默认 STOP nearly_success；
        # no_loopguard 时转 CONTINUE。直接调函数锁定两条 nearly_success 出口均受闸约束。
        fe = {"ran": True, "crashed": False, "total_cases": 10,
              "passed_cases": 10, "match_rate": 99.5}
        self.checker._load_full_eval_state = lambda: {
            "attempts_done": [], "best_full_match_rate": None}

        self._set_loop_guard(None)
        sig, _r, code = self.checker._full_eval_continue_or_stop(fe, reason="t")
        self.assertEqual((sig, code), ("STOP", "nearly_success"))

        self._set_loop_guard("1")
        sig, _r, code = self.checker._full_eval_continue_or_stop(fe, reason="t")
        self.assertEqual((sig, code), ("CONTINUE", "full_eval_regression"))


class TestDirectionMetadataFallback(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task = Path(self._tmp.name) / "005_FakeOp"
        self.tuning = self.task / "precision_tuning"
        self.tuning.mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_diagnosis_summary_populates_tuning_direction(self) -> None:
        (self.tuning / "diagnosis_summary_attempt_0.json").write_text(json.dumps({
            "attempt": 0,
            "fix_type": "FIX_PRECISION_ACCUMULATE_FP32",
            "direction_verdict": "否",
            "direction_reason": "改用 FP32 累加",
        }), encoding="utf-8")
        (self.tuning / "validation_result_attempt_0.json").write_text(json.dumps({
            "match_rate": "50.0",
        }), encoding="utf-8")
        checker = _LegacyPrecisionChecker("FakeOp", str(self.task), attempt=0)
        checker._write_round_summary(
            "validation_failed", {"primary_hint": "scattered", "op_type": "fake"})
        checker._write_tuning_directions("validation_failed")
        entry = json.loads(
            (self.tuning / "tuning_directions.json").read_text(encoding="utf-8")
        )["entries"][0]
        self.assertEqual(entry["fix_type"], "FIX_PRECISION_ACCUMULATE_FP32")
        self.assertEqual(entry["direction_verdict"], "否")
        self.assertEqual(entry["direction_reason"], "改用 FP32 累加")


if __name__ == "__main__":
    unittest.main()
