"""test_exit_artifacts.py — 退出产物从 events 重建 (debug_status.json + debug_trace.md)。

验收点 (REWRITE_PLAN §6.5): events → debug_status/trace 重建正确 (含 timeout/crashed 路径)。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine import transition as tr
from engine.exit_artifacts import (
    build_debug_status,
    build_debug_trace,
    write_exit_artifacts,
    _extract_audit_section,
    build_diagnosis_summary,
    write_diagnosis_summary,
    _kernel_file_hashes,
    diff_changed_files,
    _success_category,
)
from engine.types import Abort, Action, Continue, Done


class TestDebugStatus(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_success_status(self) -> None:
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_attempt_started(self.task_dir, Continue(0, "precision_failed"))
        tr.record_decision(self.task_dir, Done(session_outcome="success", reason="ok"))
        s = build_debug_status(self.task_dir)
        self.assertEqual(s["schema_version"], 1)
        self.assertEqual(s["session_outcome"], "success")
        self.assertEqual(s["session_branch"], "1-P")
        self.assertEqual(s["attempts_used"], 1)
        self.assertEqual(s["entry_failure_type"], "precision_failed")
        self.assertEqual(s["final_failure_type"], "precision_failed")
        self.assertIsNotNone(s["started_at"])
        self.assertIsNotNone(s["ended_at"])

    def test_all_ten_keys_present(self) -> None:
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="build_failed")
        tr.record_decision(self.task_dir, Done(session_outcome="failed", reason="x"))
        s = build_debug_status(self.task_dir)
        expected = {"schema_version", "session_outcome", "session_branch",
                    "started_at", "ended_at", "attempts_used", "entry_failure_type",
                    "final_failure_type", "final_verify_status_path", "notes",
                    "objective_success", "anti_cheat_pass", "ast_degrade_pass",
                    "reportable_success", "success_category",
                    "probe_policy_compliant", "probe_policy_violations",
                    "probe_policy_unknown", "probe_policy_records"}
        self.assertEqual(set(s), expected)

    def test_probe_policy_summary_distinguishes_violation_and_unknown(self) -> None:
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tuning = self.task_dir / "precision_tuning"
        tuning.mkdir()
        (tuning / "probe_policy_attempt0.json").write_text(json.dumps({
            "attempt": 0, "policy": "skip", "observed_status": "executed",
            "policy_pass": False,
            "source_audit_pass": False,
            "source_audit_path": "precision_tuning/probe_source_audit_attempt0.json",
            "ablation_violation": True,
        }), encoding="utf-8")
        (tuning / "probe_policy_attempt1.json").write_text(json.dumps({
            "attempt": 1, "policy": "conditional", "observed_status": "unknown",
            "policy_pass": None,
        }), encoding="utf-8")
        tr.record_decision(self.task_dir, Done(session_outcome="failed", reason="x"))
        status = build_debug_status(self.task_dir)
        self.assertFalse(status["probe_policy_compliant"])
        self.assertEqual(status["probe_policy_violations"], 1)
        self.assertEqual(status["probe_policy_unknown"], 1)
        self.assertFalse(
            status["probe_policy_records"][0]["source_audit_pass"])
        self.assertTrue(
            status["probe_policy_records"][0]["ablation_violation"])

    def test_no_anticheat_ignores_stale_prior_run_evidence(self) -> None:
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_attempt_started(self.task_dir, Continue(0, "precision_failed"))
        action = Action(kind="run_skill", name="validate", step="validate")
        action_id = tr.new_action_id()
        tr.record_action_started(self.task_dir, action, action_id)
        tr.record_action_completed(
            self.task_dir, action, action_id,
            {"objective_validation": {"verification_exit_code": 0}})
        tr.record_decision(self.task_dir, Done(session_outcome="success", reason="ok"))

        tuning = self.task_dir / "precision_tuning"
        tuning.mkdir()
        (tuning / "cheat_history.json").write_text(json.dumps({
            "cheating_attempts": [{"severity": "violation"}],
        }), encoding="utf-8")
        (self.task_dir / "_anticheat.json").write_text(json.dumps({
            "verdict": "CHEAT",
            "details": {"ast": {"status": "fail"}},
        }), encoding="utf-8")

        with mock.patch.dict(os.environ, {"ABLATE_ANTICHEAT": "1"}):
            status = build_debug_status(self.task_dir)
        self.assertTrue(status["objective_success"])
        self.assertTrue(status["anti_cheat_pass"])
        self.assertIsNone(status["ast_degrade_pass"])
        self.assertFalse(status["reportable_success"])
        self.assertEqual(
            status["success_category"], "wrapper_assisted_success")

    def test_no_terminal_event_is_crashed(self) -> None:
        # 异常中断 (无终态事件) → crashed。
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        s = build_debug_status(self.task_dir)
        self.assertEqual(s["session_outcome"], "crashed")
        self.assertIsNone(s["ended_at"])

    def test_skipped_no_attempt(self) -> None:
        # skipped_* : 无 attempt → final_verify_status_path 为 None，final_ft=entry。
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="import_failed")
        tr.record_decision(self.task_dir, Done(session_outcome="skipped_env_issue",
                                               reason="env"))
        s = build_debug_status(self.task_dir)
        self.assertEqual(s["session_outcome"], "skipped_env_issue")
        self.assertEqual(s["attempts_used"], 0)
        self.assertIsNone(s["final_verify_status_path"])
        self.assertEqual(s["final_failure_type"], "import_failed")

    def test_abort_session_outcome_from_details(self) -> None:
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_decision(self.task_dir, Abort(category="missing_failure_type",
                                                reason="no ft",
                                                details={"session_outcome": "crashed"}))
        s = build_debug_status(self.task_dir)
        self.assertEqual(s["session_outcome"], "crashed")

    def test_timeout_status(self) -> None:
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_attempt_started(self.task_dir, Continue(0, "precision_failed"))
        tr.record_decision(self.task_dir, Done(session_outcome="timeout", reason="wall-clock"))
        s = build_debug_status(self.task_dir)
        self.assertEqual(s["session_outcome"], "timeout")

    def test_final_failure_type_tracks_drift(self) -> None:
        # 漂移: entry=precision, 后漂到 build → final_failure_type=build, branch 仍 1-P。
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_attempt_started(self.task_dir, Continue(0, "precision_failed"))
        tr.record_attempt_started(self.task_dir, Continue(1, "build_failed"))
        tr.record_decision(self.task_dir, Done(session_outcome="failed", reason="x"))
        s = build_debug_status(self.task_dir)
        self.assertEqual(s["entry_failure_type"], "precision_failed")
        self.assertEqual(s["final_failure_type"], "build_failed")
        self.assertEqual(s["session_branch"], "1-P")  # 入口分支，不随漂移更新


class TestDebugTrace(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_four_sections_present(self) -> None:
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_attempt_started(self.task_dir, Continue(0, "precision_failed"))
        tr.record_decision(self.task_dir, Done(session_outcome="success", reason="ok"))
        trace = build_debug_trace(self.task_dir)
        self.assertIn("## 1. 调用入口快照", trace)
        self.assertIn("## 2. 迭代历史", trace)
        self.assertIn("## 3. 最终 Verdict", trace)
        self.assertIn("## 4. 产物清单", trace)
        self.assertIn("session_outcome: success", trace)

    def test_trace_records_each_attempt(self) -> None:
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_attempt_started(self.task_dir, Continue(0, "precision_failed"))
        tr.record_attempt_started(self.task_dir, Continue(1, "build_failed"))
        tr.record_decision(self.task_dir, Done(session_outcome="failed", reason="x"))
        trace = build_debug_trace(self.task_dir)
        self.assertIn("### Attempt 0", trace)
        self.assertIn("### Attempt 1", trace)

    def test_write_exit_artifacts_creates_files(self) -> None:
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="precision_failed")
        tr.record_decision(self.task_dir, Done(session_outcome="success", reason="ok"))
        status_path, trace_path = write_exit_artifacts(self.task_dir)
        self.assertTrue(status_path.exists())
        self.assertTrue(trace_path.exists())
        # status 是合法 JSON。
        json.loads(status_path.read_text(encoding="utf-8"))


_REAL_AUDIT = """# Precision Audit — Attempt 1

## [FORENSICS_SUMMARY]
- 9/10 cases pass.

## [ROOT_CAUSE]
CANN fp16 cumsum uses stepwise fp16 accumulation.
Residual 0.69% mismatch is platform rounding noise.

## [FIX_PLAN]
FIX_PRECISION_TYPECAST: applied per-step fp16 rounding.

## [TARGET_FILES]
- kernel/cumsum_kernel.h

## [DIRECTION_ASSESSMENT]
本轮是否延续上一轮方向: 是

=== END AUDIT ===
"""

# 裸 marker 格式 (无 ## 前缀) — SKILL.md 模板规定的主格式，真实产物如 010_LayerNorm。
_BARE_AUDIT = """# Precision Audit — Attempt 0

[FORENSICS_SUMMARY]
- baseline mismatch.

[ROOT_CAUSE]
Reduction axis mismatch in LayerNorm mean computation.

[FIX_PLAN]
FIX_PRECISION_REDUCE: fixed reduce axis.

[TARGET_FILES]
- kernel/layernorm_kernel.h

=== END AUDIT ===
"""


class TestExtractAuditSection(unittest.TestCase):
    def test_extract_root_cause(self) -> None:
        sec = _extract_audit_section(_REAL_AUDIT, "ROOT_CAUSE")
        self.assertIsNotNone(sec)
        self.assertIn("stepwise fp16 accumulation", sec)
        # 边界: 不串到下一 section。
        self.assertNotIn("FIX_PRECISION_TYPECAST", sec)
        self.assertNotIn("[FIX_PLAN]", sec)

    def test_extract_fix_plan_stops_at_next_section(self) -> None:
        sec = _extract_audit_section(_REAL_AUDIT, "FIX_PLAN")
        self.assertIsNotNone(sec)
        self.assertIn("FIX_PRECISION_TYPECAST", sec)
        self.assertNotIn("TARGET_FILES", sec)

    def test_last_section_stops_at_end_audit(self) -> None:
        sec = _extract_audit_section(_REAL_AUDIT, "DIRECTION_ASSESSMENT")
        self.assertIsNotNone(sec)
        self.assertIn("延续", sec)
        self.assertNotIn("END AUDIT", sec)

    def test_missing_tag_returns_none(self) -> None:
        self.assertIsNone(_extract_audit_section(_REAL_AUDIT, "NONEXISTENT"))

    def test_empty_body_returns_none(self) -> None:
        self.assertIsNone(_extract_audit_section("## [ROOT_CAUSE]\n## [FIX_PLAN]\nx", "ROOT_CAUSE"))

    def test_bare_marker_root_cause(self) -> None:
        # 裸格式 (无 ##) 是 SKILL 模板主格式，必须能抽取 (旧实现只认 ## 会漏)。
        sec = _extract_audit_section(_BARE_AUDIT, "ROOT_CAUSE")
        self.assertIsNotNone(sec)
        self.assertIn("Reduction axis mismatch", sec)
        self.assertNotIn("FIX_PRECISION_REDUCE", sec)

    def test_bare_marker_fix_plan_stops_at_next(self) -> None:
        sec = _extract_audit_section(_BARE_AUDIT, "FIX_PLAN")
        self.assertIsNotNone(sec)
        self.assertIn("FIX_PRECISION_REDUCE", sec)
        self.assertNotIn("TARGET_FILES", sec)
        self.assertNotIn("kernel/layernorm", sec)


def _record_diagnose(task_dir: Path, attempt: int, result: dict) -> None:
    """落一条 attempt_started + diagnose_and_fix action_completed (12b 数据源)。"""
    tr.record_attempt_started(task_dir, Continue(attempt, "precision_failed"))
    action = Action(kind="spawn_agent", name="debug_worker", step="diagnose_and_fix")
    aid = tr.new_action_id()
    tr.record_action_started(task_dir, action, aid)
    tr.record_action_completed(task_dir, action, aid, result)


def _write_validation(task_dir: Path, attempt: int, payload: dict) -> None:
    tuning = task_dir / "precision_tuning"
    tuning.mkdir(exist_ok=True)
    (tuning / f"validation_result_attempt_{attempt}.json").write_text(
        json.dumps(payload), encoding="utf-8")


class TestSuccessCategory(unittest.TestCase):
    """退出分层枚举: failed / cheat_or_invalid / clean_kernel_success / wrapper_assisted_success。"""

    def test_failed_when_objective_false(self) -> None:
        # 数值没过即 failed，无论是否作弊。
        self.assertEqual(_success_category(False, True, True, False), "failed")
        self.assertEqual(_success_category(False, False, False, False), "failed")

    def test_cheat_or_invalid_on_anticheat_violation(self) -> None:
        # 数值过 + anti_cheat 确证 violation → 假成功。
        self.assertEqual(
            _success_category(True, False, True, False), "cheat_or_invalid")

    def test_cheat_or_invalid_on_ast_degrade(self) -> None:
        # 数值过 + AST 退化确证 (ast_degrade_pass is False) → 假成功。
        self.assertEqual(
            _success_category(True, True, False, False), "cheat_or_invalid")

    def test_clean_kernel_success(self) -> None:
        # 数值过 + 无作弊 + reportable → 真·干净成功。
        self.assertEqual(
            _success_category(True, True, True, True), "clean_kernel_success")

    def test_wrapper_assisted_when_validator_errored(self) -> None:
        # 数值过 + 无确证作弊 (ast None=validator errored) + 非 reportable → 灰色中间态。
        self.assertEqual(
            _success_category(True, True, None, False), "wrapper_assisted_success")

    def test_wrapper_assisted_when_not_reportable(self) -> None:
        # 数值过 + ast clean 但 session 未正常 success 等导致非 reportable → 灰色中间态。
        self.assertEqual(
            _success_category(True, True, True, False), "wrapper_assisted_success")


class TestKernelHashDiff(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)
        (self.task_dir / "kernel").mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_diff_modified_added_deleted(self) -> None:
        k = self.task_dir / "kernel" / "k.h"
        k.write_text("v0", encoding="utf-8")
        before = _kernel_file_hashes(self.task_dir)
        k.write_text("v1", encoding="utf-8")
        (self.task_dir / "kernel" / "n.cpp").write_text("x", encoding="utf-8")
        after = _kernel_file_hashes(self.task_dir)
        self.assertEqual(diff_changed_files(before, after), ["kernel/k.h", "kernel/n.cpp"])
        # 删除也算变化。
        self.assertEqual(diff_changed_files(after, before), ["kernel/k.h", "kernel/n.cpp"])

    def test_posix_paths(self) -> None:
        (self.task_dir / "kernel" / "sub").mkdir()
        (self.task_dir / "kernel" / "sub" / "a.cpp").write_text("y", encoding="utf-8")
        h = _kernel_file_hashes(self.task_dir)
        self.assertIn("kernel/sub/a.cpp", h)

    def test_dotfiles_excluded(self) -> None:
        # 真实产物核实 (hyena 快照): macOS AppleDouble ._foo.cpp 残留须排除，
        # 否则污染 changed_files 文件名列表。
        (self.task_dir / "kernel" / "real.cpp").write_text("v", encoding="utf-8")
        (self.task_dir / "kernel" / "._real.cpp").write_text("junk", encoding="utf-8")
        (self.task_dir / "kernel" / ".hidden.h").write_text("junk", encoding="utf-8")
        h = _kernel_file_hashes(self.task_dir)
        self.assertEqual(list(h), ["kernel/real.cpp"])

    def test_protected_root_files_snapshotted(self) -> None:
        # 反作弊: model.py / 参考 wrapper 纳入快照，篡改后出现在 changed_files。
        (self.task_dir / "model.py").write_text("v0", encoding="utf-8")
        (self.task_dir / "model_new_ascendc.py").write_text("w0", encoding="utf-8")
        before = _kernel_file_hashes(self.task_dir)
        self.assertIn("model.py", before)
        self.assertIn("model_new_ascendc.py", before)
        (self.task_dir / "model.py").write_text("tampered", encoding="utf-8")
        after = _kernel_file_hashes(self.task_dir)
        self.assertEqual(diff_changed_files(before, after), ["model.py"])

    def test_op_json_snapshotted_only_with_op_name(self) -> None:
        # <op>.json / .json.bak 需 op_name 精确定位；不传则不纳入 (避免误纳入引擎产物)。
        (self.task_dir / "cumsum.json").write_text("{}", encoding="utf-8")
        (self.task_dir / "cumsum.json.bak").write_text("{}", encoding="utf-8")
        self.assertNotIn("cumsum.json", _kernel_file_hashes(self.task_dir))
        h = _kernel_file_hashes(self.task_dir, op_name="cumsum")
        self.assertIn("cumsum.json", h)
        self.assertIn("cumsum.json.bak", h)

    def test_engine_products_not_snapshotted(self) -> None:
        # 引擎自身产物 (debug_status.json / _anticheat.json) 即便传 op_name 也不该混入。
        (self.task_dir / "debug_status.json").write_text("{}", encoding="utf-8")
        (self.task_dir / "_anticheat.json").write_text("{}", encoding="utf-8")
        h = _kernel_file_hashes(self.task_dir, op_name="cumsum")
        self.assertNotIn("debug_status.json", h)
        self.assertNotIn("_anticheat.json", h)


class TestDiagnosisSummary(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)
        tr.record_session_started(self.task_dir, op_name="cumsum", agent="constructive",
                                  entry_failure_type="precision_failed")
        self.tuning = self.task_dir / "precision_tuning"
        self.tuning.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_full_summary(self) -> None:
        _record_diagnose(self.task_dir, 0, {
            "success": True, "final_response": "根因: CAST_NONE 应改 CAST_ROUND",
            "changed_files": ["kernel/cumsum_kernel.h"]})
        _write_validation(self.task_dir, 0, {
            "correctness_passed": True, "match_rate": "99.31", "max_diff": "0.0625",
            "first_error_lines": ["x"]})
        (self.tuning / "precision_audit_0.md").write_text(_REAL_AUDIT, encoding="utf-8")
        d = build_diagnosis_summary(self.task_dir, 0)
        self.assertIsNotNone(d)
        self.assertEqual(d["final_response"], "根因: CAST_NONE 应改 CAST_ROUND")
        self.assertEqual(d["changed_files"], ["kernel/cumsum_kernel.h"])
        self.assertEqual(d["validation"]["match_rate"], "99.31")
        self.assertIn("stepwise fp16", d["root_cause"])  # audit enrichment
        self.assertEqual(d["attempt"], 0)

    def test_final_response_only(self) -> None:
        # 无 validation 文件、无 audit: 仅 final_response/changed_files 也成摘要。
        _record_diagnose(self.task_dir, 0, {
            "success": True, "final_response": "诊断文本", "changed_files": []})
        d = build_diagnosis_summary(self.task_dir, 0)
        self.assertIsNotNone(d)
        self.assertEqual(d["final_response"], "诊断文本")
        self.assertEqual(d["changed_files"], [])
        self.assertNotIn("validation", d)
        self.assertNotIn("root_cause", d)

    def test_direction_metadata_extracted_from_final_response(self) -> None:
        response = """[FIX_PLAN]
采用 FIX_PRECISION_ACCUMULATE_FP32
[DIRECTION_ASSESSMENT]
本轮是否延续上一轮方向: 否
换方向理由: 上一轮仅调整 tiling，没有修复累加精度
"""
        _record_diagnose(self.task_dir, 1, {
            "success": True, "final_response": response, "changed_files": ["kernel/k.h"]})
        d = build_diagnosis_summary(self.task_dir, 1)
        self.assertEqual(d["fix_type"], "FIX_PRECISION_ACCUMULATE_FP32")
        self.assertEqual(d["direction_verdict"], "否")
        self.assertIn("累加精度", d["direction_reason"])

    def test_structured_direction_metadata_preferred(self) -> None:
        _record_diagnose(self.task_dir, 1, {
            "success": True,
            "final_response": "普通总结，不包含旧模板",
            "attempt_metadata": {
                "fix_type": "reduce_fp32_accumulation",
                "direction_verdict": "switch",
                "direction_reason": "previous tiling direction regressed",
                "probe_status": "executed",
                "kb_used_ids": ["kb-0123456789ab"],
            },
            "probe_policy": "conditional",
            "probe_policy_pass": True,
            "changed_files": ["kernel/k.h"],
        })
        d = build_diagnosis_summary(self.task_dir, 1)
        self.assertEqual(d["schema_version"], 2)
        self.assertEqual(d["fix_type"], "reduce_fp32_accumulation")
        self.assertEqual(d["direction_verdict"], "switch")
        self.assertEqual(d["probe_status"], "executed")
        self.assertEqual(d["probe_policy"], "conditional")

    def test_validation_only(self) -> None:
        # diagnose result 无 final_response (timeout 早返回路径)，但 validation 已产出。
        _record_diagnose(self.task_dir, 0, {"success": False, "claude_state": "timeout"})
        _write_validation(self.task_dir, 0, {"correctness_passed": False, "match_rate": "0.00"})
        d = build_diagnosis_summary(self.task_dir, 0)
        self.assertIsNotNone(d)
        self.assertNotIn("final_response", d)
        self.assertEqual(d["validation"]["correctness_passed"], False)

    def test_no_content_returns_none(self) -> None:
        # 无 diagnose 事件、无 validation、无 audit → 仅骨架键 → None。
        self.assertIsNone(build_diagnosis_summary(self.task_dir, 0))

    def test_corrupt_validation_best_effort(self) -> None:
        _record_diagnose(self.task_dir, 0, {"final_response": "x", "changed_files": []})
        (self.tuning / "validation_result_attempt_0.json").write_text("{bad", encoding="utf-8")
        d = build_diagnosis_summary(self.task_dir, 0)
        self.assertIsNotNone(d)
        self.assertNotIn("validation", d)  # 损坏静默跳过
        self.assertEqual(d["final_response"], "x")

    def test_invalid_utf8_audit_best_effort(self) -> None:
        _record_diagnose(self.task_dir, 0, {"final_response": "x", "changed_files": []})
        (self.tuning / "precision_audit_0.md").write_bytes(b"[ROOT_CAUSE]\n\xff\xfe bad")
        d = build_diagnosis_summary(self.task_dir, 0)  # 不抛
        self.assertIsNotNone(d)
        self.assertNotIn("root_cause", d)

    def test_audit_root_cause_truncated(self) -> None:
        _record_diagnose(self.task_dir, 0, {"final_response": "x", "changed_files": []})
        (self.tuning / "precision_audit_0.md").write_text(
            "[ROOT_CAUSE]\n" + ("A" * 1000) + "\n[FIX_PLAN]\ny", encoding="utf-8")
        d = build_diagnosis_summary(self.task_dir, 0)
        self.assertIn("…(截断)", d["root_cause"])
        self.assertLess(len(d["root_cause"]), 700)

    def test_multi_attempt_isolation(self) -> None:
        _record_diagnose(self.task_dir, 0, {"final_response": "轮0", "changed_files": ["a"]})
        _record_diagnose(self.task_dir, 1, {"final_response": "轮1", "changed_files": ["b"]})
        self.assertEqual(build_diagnosis_summary(self.task_dir, 0)["final_response"], "轮0")
        self.assertEqual(build_diagnosis_summary(self.task_dir, 1)["final_response"], "轮1")

    def test_write_produces_json(self) -> None:
        _record_diagnose(self.task_dir, 0, {"final_response": "x", "changed_files": []})
        path = write_diagnosis_summary(self.task_dir, 0)
        self.assertIsNotNone(path)
        self.assertEqual(path.name, "diagnosis_summary_attempt_0.json")
        json.loads(path.read_text(encoding="utf-8"))

    def test_write_none_when_empty(self) -> None:
        self.assertIsNone(write_diagnosis_summary(self.task_dir, 0))


class TestDebugTraceDiagnosisInjection(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_trace_injects_diagnosis(self) -> None:
        tr.record_session_started(self.task_dir, op_name="cumsum", agent="constructive",
                                  entry_failure_type="precision_failed")
        _record_diagnose(self.task_dir, 0, {
            "final_response": "根因 CAST_NONE", "changed_files": ["kernel/cumsum_kernel.h"]})
        _write_validation(self.task_dir, 0, {"correctness_passed": False, "match_rate": "99.31"})
        write_diagnosis_summary(self.task_dir, 0)
        tr.record_decision(self.task_dir, Done(session_outcome="failed", reason="x"))
        trace = build_debug_trace(self.task_dir)
        self.assertIn("诊断摘要", trace)
        self.assertIn("kernel/cumsum_kernel.h", trace)
        self.assertIn("99.31", trace)
        self.assertIn("根因 CAST_NONE", trace)

    def test_trace_degrades_without_summary(self) -> None:
        tr.record_session_started(self.task_dir, op_name="add", agent="constructive",
                                  entry_failure_type="build_failed")
        tr.record_attempt_started(self.task_dir, Continue(0, "build_failed"))
        tr.record_decision(self.task_dir, Done(session_outcome="failed", reason="x"))
        trace = build_debug_trace(self.task_dir)
        self.assertIn("诊断摘要: (无)", trace)


if __name__ == "__main__":
    unittest.main()
