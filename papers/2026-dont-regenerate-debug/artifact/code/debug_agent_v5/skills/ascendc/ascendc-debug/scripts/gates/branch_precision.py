"""branch_precision.py — 精度失败分支 Gate.

从原 precision_gate.py 的 `class GateChecker` **原样搬** precision 语义（findings.md §3.3 ③）:
  - check_forensics / _write_baseline_from_forensics
  - check_audit / 7 个 section 检查
  - check_validate + _compute_loop_signal + _count_stagnant + _detect_harmful_regression
    + _compute_improvement_ratio + _write_round_summary + _write_tuning_directions
    + _extract_direction_* + _check_direction_assessment + _extract_section
    + _extract_fix_type + _extract_changed_locations + _write_audit_index
    + _get_baseline_match_rate + _kernel_dir + _check_import_name_match + _result

对外契约:
  - `PrecisionBranch(op_name).run_gate_f(task_dir, attempt)`
  - `PrecisionBranch(op_name).run_gate_a(task_dir, attempt)`
  - `PrecisionBranch(op_name).run_gate_v(task_dir, attempt)`

所有文件产出路径、loop_signal 取值 (PASS/CONTINUE/STOP)、audit 7 section 名等均与搬移前一致。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .common import GateOutcome, MAX_ATTEMPTS, ensure_engine_audit


MAX_STAGNANT_ROUNDS = 2

# L5_PROBE 真探针优先：连续 N 轮 audit 缺真探针 (P1/P2/P3 实测值) 才允许
# INSTRUMENTATION_FINDINGS 兜底并标 degraded。单轮缺失不回退——逼 agent 走真探针。
_L5_PROBE_FALLBACK_THRESHOLD = 2

# 全量复验闸 (§2.1)：连续 N 轮全量未满且无改善 → 回落 nearly_success STOP，
# 避免真·量化噪声 (全量也只到 99%) 无限重试烧预算。EPS 区分真改善 vs 噪声抖动。
_FULL_EVAL_MAX_ROUNDS = 2
_FULL_EVAL_IMPROVE_EPS = 0.5  # 百分点


class _LegacyPrecisionChecker:
    """原 GateChecker 精度相关方法的 1:1 搬移。"""

    def __init__(self, op_name: str, task_dir: str, attempt: int = 0):
        self.op_name = op_name
        self.task_dir = task_dir
        self.attempt = attempt
        self.tuning_dir = os.path.join(task_dir, "precision_tuning")

    # ================================================================
    # Gate-F: 取证报告
    # ================================================================

    def check_forensics(self) -> dict:
        path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        checks = {
            "report_exists": os.path.exists(path),
            "report_parseable": False,
            "status_completed": False,
            "has_primary_hint": False,
            "has_outputs": False,
            "has_basic_stats": False,
            "attempt_matches": False,
        }
        r = None
        if checks["report_exists"]:
            try:
                with open(path) as f:
                    r = json.load(f)
                checks["report_parseable"] = True
                status = r.get("status")
                if status in {"build_failed", "import_failed", "forensics_unavailable"}:
                    checks = {
                        "report_exists": True,
                        "report_parseable": True,
                        "diagnostic_report_available": True,
                        "status_allows_diagnose": True,
                        "has_primary_hint": bool(r.get("primary_hint")),
                        "attempt_matches": r.get("attempt", -1) == self.attempt,
                    }
                    return self._result("GATE-F", checks)
                checks["status_completed"] = status == "completed"
                checks["has_primary_hint"] = bool(r.get("primary_hint"))
                checks["has_outputs"] = len(r.get("outputs", [])) > 0
                if checks["has_outputs"]:
                    checks["has_basic_stats"] = "basic_stats" in r["outputs"][0]
                checks["attempt_matches"] = r.get("attempt", -1) == self.attempt
            except (json.JSONDecodeError, KeyError):
                r = None

        gate_result = self._result("GATE-F", checks)

        # Gate-F 通过且 attempt 0 时：从 forensics 写 baseline_state.json（幂等）
        if gate_result["passed"] and self.attempt == 0 and r is not None:
            self._write_baseline_from_forensics(r)

        return gate_result

    def _write_baseline_from_forensics(self, forensics: dict) -> None:
        baseline_path = os.path.join(self.tuning_dir, "baseline_state.json")
        if os.path.exists(baseline_path):
            return

        try:
            outputs = forensics.get("outputs", [])
            if not outputs:
                return
            stats = outputs[0].get("basic_stats", {})
            raw_match_rate = stats.get("match_rate")
            raw_mismatch_ratio = stats.get("mismatch_ratio")
            if raw_match_rate is None:
                return

            baseline_match_rate = round(float(raw_match_rate) * 100, 4)
            baseline_mismatch_ratio = float(raw_mismatch_ratio) if raw_mismatch_ratio is not None else None

            baseline_state = {
                "match_rate":      baseline_match_rate,
                "mismatch_ratio":  baseline_mismatch_ratio,
                "max_abs_diff":    stats.get("max_abs_diff"),
                "mean_abs_diff":   stats.get("mean_abs_diff"),
                "primary_hint":    forensics.get("primary_hint"),
                "source":          "forensics_report.json/outputs[0]/basic_stats",
                "note":            "Initial precision captured at Gate-F before any code modification"
            }
            os.makedirs(self.tuning_dir, exist_ok=True)
            with open(baseline_path, "w", encoding="utf-8") as f:
                json.dump(baseline_state, f, indent=2, ensure_ascii=False)
        except (OSError, ValueError, KeyError, TypeError):
            pass

    # ================================================================
    # Gate-A: 审计报告
    # ================================================================

    def _ensure_engine_audit(self) -> None:
        """Build the pre-Agent evidence packet consumed by Gate-A and the Agent."""
        ensure_engine_audit(
            Path(self.task_dir),
            self.attempt,
            "precision_failed",
        )

    def check_audit(self) -> dict:
        prereq = self._check_prerequisite_forensics()
        if (
            not prereq["satisfied"]
            and os.environ.get("ABLATE_FORENSICS") == "1"
        ):
            prereq = {
                "satisfied": True,
                "reason": "forensics_disabled_raw_validation_audit",
                "detail": {"forensics_ablation": True},
            }
        if not prereq["satisfied"]:
            checks = {"prerequisite_forensics": False}
            checks.update(prereq["detail"])
            result = self._result("GATE-A", checks)
            result["prerequisite_error"] = prereq["reason"]
            return result
        self._ensure_engine_audit()

        path = os.path.join(self.tuning_dir, f"precision_audit_{self.attempt}.md")
        checks = {
            "prerequisite_forensics": True,
            "report_exists": os.path.exists(path),
            "report_nonempty": False,
            "has_forensics_summary": False,
            "has_computation_decomposition": False,
            "has_reference_impl_spec": False,
            "has_kernel_step_trace": False,
            "has_l5_probe": False,
            "has_root_cause": False,
            "has_causal_chain_analysis": False,
            "has_fix_plan": False,
            "has_target_files": False,
            "has_experiment_results": False,
            "has_direction_assessment": True,
        }
        content = None
        l5_probe_degraded = False
        l5_probe_source = "L5_PROBE"
        if checks["report_exists"]:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            checks["report_nonempty"] = len(content) > 200
            for tag, key in [("FORENSICS_SUMMARY", "has_forensics_summary"),
                             ("COMPUTATION_DECOMPOSITION", "has_computation_decomposition"),
                             ("REFERENCE_IMPL_SPEC", "has_reference_impl_spec"),
                             ("KERNEL_STEP_TRACE", "has_kernel_step_trace"),
                             ("ROOT_CAUSE", "has_root_cause"),
                             ("CAUSAL_CHAIN_ANALYSIS", "has_causal_chain_analysis"),
                             ("FIX_PLAN", "has_fix_plan"),
                             ("TARGET_FILES", "has_target_files"),
                             ("EXPERIMENT_RESULTS", "has_experiment_results")]:
                checks[key] = f"[{tag}]" in content
            # L5_PROBE 真探针优先 (取代统一存在性检查): executed/skipped 算"存在"
            # (保留 SKILL.md 合法跳过通过契约); 仅 missing 才是真探针失败。连续
            # >=_L5_PROBE_FALLBACK_THRESHOLD 轮 missing 才允许 INSTRUMENTATION_FINDINGS
            # 兜底并标 degraded——单轮缺失不回退，逼 agent 走真探针。
            probe_status = self._l5_probe_status(content)
            if probe_status != "missing":
                checks["has_l5_probe"] = True
            else:
                consecutive = self._consecutive_probe_failures()  # 不含本轮
                if (consecutive + 1 >= _L5_PROBE_FALLBACK_THRESHOLD
                        and "[INSTRUMENTATION_FINDINGS]" in content):
                    checks["has_l5_probe"] = True
                    l5_probe_degraded = True
                    l5_probe_source = "INSTRUMENTATION_FINDINGS"
            checks["has_direction_assessment"] = (
                self.attempt == 0 or "[DIRECTION_ASSESSMENT]" in content
            )
            if self.attempt > 0:
                if "[DIRECTION_ASSESSMENT]" in content:
                    checks["direction_assessment_binary"] = self._validate_direction_binary(content)
                else:
                    checks["direction_assessment_binary"] = False

        gate_result = self._result("GATE-A", checks)
        # degraded 标记记 gate_result 顶层 (非 checks——否则进 all(checks.values())
        # 误判)，供统计/消融区分"真探针通过"与"连续失败回退兜底"。
        if l5_probe_degraded:
            gate_result["l5_probe_degraded"] = True
            gate_result["l5_probe_source"] = l5_probe_source

        # 正常通过则写 index；degraded 兜底也写 (即便 passed=False)——回退轮常因别的
        # section 缺失而 passed=False，而 section_sources 留痕恰是最该保住的兜底证据，
        # 不能被 passed 门控吞掉 (§1.7c 必需标记)。
        if content and (gate_result["passed"] or l5_probe_degraded):
            self._write_audit_index(content, l5_probe_source=l5_probe_source)

        return gate_result

    # ================================================================
    # Gate-V: 验证结果 + 循环控制
    # ================================================================

    def check_validate(self) -> dict:
        prereq = self._check_prerequisite_code()
        if not prereq["satisfied"]:
            checks = {"prerequisite_code": False}
            checks.update(prereq["detail"])
            result = self._result("GATE-V", checks)
            result["prerequisite_error"] = prereq["reason"]
            result["loop_signal"] = "STOP"
            result["loop_reason"] = f"前置条件不满足: {prereq['reason']}"
            result["stop_reason_code"] = "prerequisite_failure"
            return result

        result_path = os.path.join(self.tuning_dir,
                                   f"validation_result_attempt_{self.attempt}.json")
        checks = {
            "prerequisite_code": True,
            "result_exists": os.path.exists(result_path),
            "result_parseable": False,
            "precision_passed": False,
        }

        correctness_passed = False
        match_rate = None
        if checks["result_exists"]:
            try:
                with open(result_path) as f:
                    r = json.load(f)
                checks["result_parseable"] = True
                correctness_passed = r.get("correctness_passed", False)
                checks["precision_passed"] = correctness_passed
                mr_str = r.get("match_rate")
                if mr_str is not None:
                    try:
                        match_rate = float(mr_str)
                    except (ValueError, TypeError):
                        pass
            except (json.JSONDecodeError, KeyError):
                pass

        forensics_data = self._load_forensics()
        loop_signal, loop_reason, stop_reason_code = self._compute_loop_signal(
            correctness_passed, match_rate, forensics_data
        )

        gate_result = self._result("GATE-V", checks)
        gate_result["loop_signal"] = loop_signal
        gate_result["loop_reason"] = loop_reason
        gate_result["stop_reason_code"] = stop_reason_code
        gate_result["attempt"] = self.attempt
        gate_result["max_attempts"] = MAX_ATTEMPTS

        self._write_round_summary(stop_reason_code, forensics_data)
        self._write_tuning_directions(stop_reason_code)

        return gate_result

    # ================================================================
    # 前置依赖检查
    # ================================================================

    def _check_prerequisite_forensics(self) -> dict:
        path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        if not os.path.exists(path):
            return {"satisfied": False,
                    "reason": f"forensics_report_{self.attempt}.json 不存在, 必须先运行 precision_forensics.py",
                    "detail": {"forensics_exists": False, "forensics_attempt_match": False}}
        try:
            with open(path) as f:
                r = json.load(f)
            if r.get("status") != "completed":
                return {"satisfied": False,
                        "reason": f"forensics 状态异常: {r.get('status')}",
                        "detail": {"forensics_exists": True, "forensics_attempt_match": False}}
            if r.get("attempt", -1) != self.attempt:
                return {"satisfied": False,
                        "reason": f"forensics attempt={r.get('attempt')} 不匹配当前 attempt={self.attempt}, "
                                  f"必须重新运行 precision_forensics.py",
                        "detail": {"forensics_exists": True, "forensics_attempt_match": False}}
            return {"satisfied": True, "reason": "", "detail": {}}
        except (json.JSONDecodeError, KeyError) as e:
            return {"satisfied": False, "reason": f"forensics 解析失败: {e}",
                    "detail": {"forensics_exists": True, "forensics_attempt_match": False}}

    def _check_prerequisite_code(self) -> dict:
        kdir = self._kernel_dir()
        if not kdir:
            return {"satisfied": False,
                    "reason": f"{self.task_dir}/kernel/ 不存在",
                    "detail": {"kernel_dir_exists": False}}
        pybind = os.path.join(kdir, "pybind11.cpp")
        if not os.path.exists(pybind) or os.path.getsize(pybind) < 100:
            return {"satisfied": False,
                    "reason": f"{pybind} 不存在或内容过少",
                    "detail": {"pybind11_cpp_exists": False}}
        return {"satisfied": True, "reason": "", "detail": {}}

    # ================================================================
    # 循环控制
    # ================================================================

    def _compute_loop_signal(self, passed: bool, match_rate: float = None, forensics_data: dict = None) -> tuple:
        loop_guard_on = os.environ.get("ABLATE_LOOP_GUARD") != "1"
        # 全量复验闸 (§2.1): 轻量结果须经全量 case 集复核才定终态。
        # full_eval 为 None = 闸关 / only_py 算子无全量集 → graceful 回退轻量口径 (原行为)。
        full_eval = self._load_full_eval()

        if passed:
            # 轻量全过。无全量产物 → 维持原 PASS；有则须全量也 100% 才算真过 (消假阳性)。
            if full_eval is None or not full_eval.get("ran"):
                return "PASS", "精度验证通过", "precision_passed"
            if full_eval.get("crashed"):
                return self._full_eval_continue_or_stop(full_eval, reason="轻量过但全量复验crash")
            if full_eval.get("total_cases") and \
                    full_eval.get("passed_cases") == full_eval.get("total_cases"):
                return "PASS", "精度验证通过 (全量复验100%)", "precision_passed"
            return self._full_eval_continue_or_stop(full_eval, reason="轻量过但全量未满")

        # match_rate ≥ 99% 但 evaluate 返回 FAIL：原 nearly_success 触发线。
        # 现在此处不直接 STOP，而是看全量复验：全量也满 → 推过线；否则带样本继续 / 防爆回落。
        if match_rate is not None and match_rate >= 99.0:
            if full_eval is not None and full_eval.get("ran"):
                if not full_eval.get("crashed") and full_eval.get("total_cases") and \
                        full_eval.get("passed_cases") == full_eval.get("total_cases"):
                    return "PASS", (
                        f"轻量近通过 (match_rate={match_rate:.2f}%) 且全量复验100%通过"
                    ), "precision_passed"
                return self._full_eval_continue_or_stop(
                    full_eval, reason=f"轻量近通过 (match_rate={match_rate:.2f}%) 但全量未满")
            # 全量闸关 / 无全量集 → 维持原 nearly_success STOP (向后兼容)；LoopGuard 关时放行续跑。
            if loop_guard_on:
                return (
                    "STOP",
                    f"精度接近通过 (match_rate={match_rate:.2f}%)，疑似量化截断噪声或 float16 精度损失，建议人工确认",
                    "nearly_success",
                )

        # fp16 early exit: 连续两轮仅 fp16 失败且 max_abs_diff ≤ 0.25 且 mismatch_ratio < 2.0%
        if loop_guard_on and self.attempt >= 1 and forensics_data is not None:
            if self._check_fp16_ceiling(forensics_data):
                return (
                    "STOP",
                    "连续两轮仅 fp16 失败且 max_abs_diff ≤ 0.25, mismatch_ratio < 2.0%, 已达 fp16 硬件精度上限",
                    "fp16_precision_ceiling",
                )

        if self.attempt + 1 >= MAX_ATTEMPTS:
            return "STOP", f"已达最大轮次 ({MAX_ATTEMPTS})", "max_attempts_reached"

        fr = forensics_data if loop_guard_on else None
        if fr is not None:
            try:
                trend = fr.get("history_trend")
                if trend:
                    trend_list = trend.get("trend", [])

                    if self._detect_harmful_regression(trend_list):
                        return "STOP", "检测到 A→B→A 振荡型有害回退，需人工分析", "harmful_regression"

                    if not trend.get("mismatch_improving", True):
                        stagnant = self._count_stagnant(trend_list)
                        if stagnant >= MAX_STAGNANT_ROUNDS:
                            direction_ok = self._check_direction_assessment()
                            if direction_ok == "continue":
                                return "CONTINUE", (
                                    f"mismatch 连续 {stagnant} 轮未改善, "
                                    f"但 Agent 已明确换方向，继续探索"
                                ), "stagnant_new_direction"
                            elif direction_ok == "missing":
                                # 12a 兜底: audit 产出缺失，无据判定方向 →
                                # 不据缺失误判 STOP，保守续跑 (实测 audit 缺失率高)。
                                return "CONTINUE", (
                                    f"mismatch 连续 {stagnant} 轮未改善, "
                                    f"但 audit 方向评估缺失，无据判定，保守续跑"
                                ), "stagnant_audit_missing"
                            else:
                                return "STOP", (
                                    f"mismatch 连续 {stagnant} 轮未改善, "
                                    f"Agent 仍沿用同一方向，可能方向错误，需人工分析"
                                ), "stagnant_same_direction"
            except (json.JSONDecodeError, KeyError):
                pass

        return "CONTINUE", f"精度未通过, 进入第 {self.attempt + 2} 轮", None

    def _full_eval_continue_or_stop(self, full_eval: dict, *, reason: str) -> tuple:
        """全量复验未满时的收敛判定 (防爆核心，§2.1)。

        有可定位失败样本 (total>passed 且未 crash) 或 crash → CONTINUE 带样本继续 debug
        (stop_reason_code=full_eval_regression，CONTINUE 标签不进 _STOP_OUTCOME)；
        连续 _FULL_EVAL_MAX_ROUNDS 轮全量无改善 → 回落 nearly_success STOP，避免真·量化
        噪声 (全量也只到 99%) 无限重试。无可定位样本 (异常态) → 保守 STOP nearly_success。
        """
        state = self._load_full_eval_state()
        rounds = len(state.get("attempts_done", []))
        best = state.get("best_full_match_rate")
        curr = float(full_eval.get("match_rate", 0.0))
        improved = (best is None) or (curr > float(best) + _FULL_EVAL_IMPROVE_EPS)
        crashed = bool(full_eval.get("crashed"))
        total = full_eval.get("total_cases", 0) or 0
        passed_cases = full_eval.get("passed_cases", 0) or 0
        locatable = (total > passed_cases) and not crashed
        # LoopGuard 关 (no_loopguard arm): 抑制本函数的 nearly_success 早停，
        # 续跑至 _compute_loop_signal 的 max_attempts_reached 硬顶 (有界)，与 :349 对齐。
        loop_guard_on = os.environ.get("ABLATE_LOOP_GUARD") != "1"

        if loop_guard_on and rounds >= _FULL_EVAL_MAX_ROUNDS and not improved:
            return (
                "STOP",
                f"{reason}；全量复验连续 {rounds} 轮无改善，疑似精度上限，建议人工确认",
                "nearly_success",
            )
        if locatable or crashed:
            return (
                "CONTINUE",
                f"{reason}，带全量失败样本继续 debug "
                f"(全量 {passed_cases}/{total}，match_rate={curr:.2f}%)",
                "full_eval_regression",
            )
        if loop_guard_on:
            return (
                "STOP",
                f"{reason}；全量复验无可定位失败样本，建议人工确认",
                "nearly_success",
            )
        return (
            "CONTINUE",
            f"{reason}；LoopGuard 关，全量无可定位样本仍续跑至最大轮次",
            "full_eval_regression",
        )

    def _count_stagnant(self, trend: list) -> int:
        ratios = [t["mismatch_ratio"] for t in trend if t.get("mismatch_ratio") is not None]
        if len(ratios) < 2:
            return 0
        count = 0
        for i in range(len(ratios) - 1, 0, -1):
            if ratios[i] >= ratios[i - 1]:
                count += 1
            else:
                break
        return count

    def _detect_harmful_regression(self, trend: list) -> bool:
        ratios = [t["mismatch_ratio"] for t in trend if t.get("mismatch_ratio") is not None]
        if len(ratios) < 3:
            return False
        r_prev, r_mid, r_curr = ratios[-3], ratios[-2], ratios[-1]
        mid_improved = (r_prev - r_mid) > 0.01
        curr_regressed = r_curr >= (r_prev - 0.005)
        return mid_improved and curr_regressed

    def _check_fp16_ceiling(self, forensics_data: dict) -> bool:
        """检查连续两轮是否仅 fp16 失败且达到硬件精度上限。

        条件：
        1. 当前轮和上一轮的 forensics 都存在
        2. 两轮都只有 fp16 失败（fp32/bf16 通过或不存在）
        3. 当前轮 max_abs_diff ≤ 0.25 且 mismatch_ratio < 0.02
        """
        if self.attempt < 1:
            return False

        # 读取当前轮 forensics
        curr_forensics = forensics_data
        if curr_forensics is None:
            return False

        # 读取上一轮 forensics
        prev_forensics_path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt - 1}.json")
        if not os.path.exists(prev_forensics_path):
            return False
        try:
            with open(prev_forensics_path) as f:
                prev_forensics = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False

        # 检查两轮是否都只有 fp16 失败
        def _only_fp16_fails(forensics: dict) -> bool:
            outputs = forensics.get("outputs", [])
            if not outputs:
                return False
            per_case = outputs[0].get("per_case", [])
            if not per_case:
                return False

            has_fp16_fail = False
            for case in per_case:
                dtype = case.get("input_dtype", "")
                passed = case.get("passed", False)
                if "float16" in dtype or "fp16" in dtype:
                    if not passed:
                        has_fp16_fail = True
                elif "float32" in dtype or "fp32" in dtype or "bfloat16" in dtype or "bf16" in dtype:
                    if not passed:
                        return False  # fp32/bf16 失败，不是纯 fp16 问题
            return has_fp16_fail

        if not _only_fp16_fails(curr_forensics) or not _only_fp16_fails(prev_forensics):
            return False

        # 检查当前轮精度指标
        curr_outputs = curr_forensics.get("outputs", [])
        if not curr_outputs:
            return False
        stats = curr_outputs[0].get("basic_stats", {})
        max_abs_diff = stats.get("max_abs_diff")
        mismatch_ratio = stats.get("mismatch_ratio")

        if max_abs_diff is None or mismatch_ratio is None:
            return False

        return float(max_abs_diff) <= 0.25 and float(mismatch_ratio) < 0.02

    def _compute_improvement_ratio(self, prev_mismatch: float, curr_mismatch: float):
        prev_match = (1 - prev_mismatch) * 100
        curr_match = (1 - curr_mismatch) * 100
        remaining = 100 - prev_match
        if remaining <= 0:
            return None
        return round((curr_match - prev_match) / remaining, 4)

    def _write_round_summary(self, stop_reason_code, forensics_data: dict = None) -> None:
        summary_path = os.path.join(self.tuning_dir, f"round_summary_{self.attempt}.json")

        if stop_reason_code is None:
            stop_reason_code = "validation_failed"

        existing = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        match_rate = None
        mismatch_ratio = None
        improvement_ratio = None
        absolute_improvement = None
        forensics_hint = None
        op_type = None

        result_path = os.path.join(self.tuning_dir, f"validation_result_attempt_{self.attempt}.json")
        if os.path.exists(result_path):
            try:
                with open(result_path) as f:
                    r = json.load(f)
                mr_str = r.get("match_rate")
                if mr_str is not None:
                    match_rate = round(float(mr_str), 4)
                    mismatch_ratio = round(1 - match_rate / 100, 8)
            except (json.JSONDecodeError, KeyError, OSError, ValueError):
                pass

        if match_rate is not None:
            if self.attempt == 0:
                baseline_match_rate = self._get_baseline_match_rate(forensics_data)
                if baseline_match_rate is not None:
                    baseline_mismatch = 1 - baseline_match_rate / 100
                    curr_mismatch = 1 - match_rate / 100
                    improvement_ratio = self._compute_improvement_ratio(
                        baseline_mismatch, curr_mismatch
                    )
                    absolute_improvement = round(match_rate - baseline_match_rate, 4)
            elif self.attempt > 0:
                prev_result_path = os.path.join(
                    self.tuning_dir, f"validation_result_attempt_{self.attempt - 1}.json"
                )
                if os.path.exists(prev_result_path):
                    try:
                        with open(prev_result_path) as f:
                            prev_r = json.load(f)
                        prev_mr_str = prev_r.get("match_rate")
                        if prev_mr_str is not None:
                            prev_match_rate = float(prev_mr_str)
                            prev_mismatch = 1 - prev_match_rate / 100
                            curr_mismatch = 1 - match_rate / 100
                            improvement_ratio = self._compute_improvement_ratio(
                                prev_mismatch, curr_mismatch
                            )
                            absolute_improvement = round(match_rate - prev_match_rate, 4)
                    except (json.JSONDecodeError, KeyError, OSError, ValueError):
                        pass

        fr = forensics_data
        if fr is not None:
            forensics_hint = fr.get("primary_hint")
            op_type = fr.get("op_type") or fr.get("L8_operator", {}).get("op_type")

        compile_log_abs = os.path.join(
            self.tuning_dir, f"compilation_log_{self.attempt}.json"
        )
        compilation_log_ref = (
            f"precision_tuning/compilation_log_{self.attempt}.json"
            if os.path.exists(compile_log_abs) else None
        )

        summary = dict(existing)
        summary["attempt"] = self.attempt

        metrics = summary.get("metrics", {})
        metrics.update({
            "match_rate":            match_rate,
            "mismatch_ratio":        mismatch_ratio,
            "improvement_ratio":     improvement_ratio,
            "absolute_improvement":  absolute_improvement,
            "stop_reason_code":      stop_reason_code,
        })
        summary["metrics"] = metrics

        diagnostics = summary.get("diagnostics", {})
        diagnostics["forensics_hint"] = forensics_hint
        diagnostics["op_type"] = op_type
        diagnosis_path = os.path.join(
            self.tuning_dir, f"diagnosis_summary_attempt_{self.attempt}.json")
        if os.path.exists(diagnosis_path):
            try:
                with open(diagnosis_path, encoding="utf-8") as f:
                    diagnosis = json.load(f)
                for key in ("fix_type", "direction_verdict", "direction_reason"):
                    if diagnostics.get(key) is None and diagnosis.get(key) is not None:
                        diagnostics[key] = diagnosis[key]
            except (json.JSONDecodeError, OSError):
                pass
        summary["diagnostics"] = diagnostics

        index = summary.get("index", {})
        index["compilation_log"] = compilation_log_ref
        summary["index"] = index

        try:
            os.makedirs(self.tuning_dir, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _write_tuning_directions(self, stop_reason_code) -> None:
        directions_path = os.path.join(self.tuning_dir, "tuning_directions.json")

        data = {"op_name": self.op_name, "final_status": "in_progress", "entries": []}
        if os.path.exists(directions_path):
            try:
                with open(directions_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        fix_type = None
        direction_verdict = None
        direction_reason = None
        forensics_hint = None
        improvement_ratio = None
        absolute_improvement = None
        match_rate = None
        mismatch_ratio = None

        summary_path = os.path.join(self.tuning_dir, f"round_summary_{self.attempt}.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, encoding="utf-8") as f:
                    summary = json.load(f)
                diag = summary.get("diagnostics", {})
                fix_type = diag.get("fix_type")
                direction_verdict = diag.get("direction_verdict")
                direction_reason = diag.get("direction_reason")
                forensics_hint = diag.get("forensics_hint")
                metrics = summary.get("metrics", {})
                improvement_ratio = metrics.get("improvement_ratio")
                absolute_improvement = metrics.get("absolute_improvement")
                match_rate = metrics.get("match_rate")
                mismatch_ratio = metrics.get("mismatch_ratio")
            except (json.JSONDecodeError, OSError):
                pass

        result_path = os.path.join(self.tuning_dir, f"validation_result_attempt_{self.attempt}.json")
        if stop_reason_code == "precision_passed":
            outcome = "passed"
        elif improvement_ratio is None and not os.path.exists(result_path):
            outcome = "unknown"  # validation_result 文件缺失，不应误判为 stagnant
        elif improvement_ratio is None:
            outcome = "stagnant"
        elif improvement_ratio < -0.05:
            outcome = "regressed"
        elif improvement_ratio >= 0.1:
            outcome = "improved"
        else:
            outcome = "stagnant"

        if direction_reason is None:
            direction_reason = self._extract_direction_reason()

        new_entry = {
            "attempt":               self.attempt,
            "fix_type":              fix_type,
            "forensics_hint":        forensics_hint,
            "direction_verdict":     direction_verdict,
            "direction_reason":      direction_reason,
            "improvement_ratio":     improvement_ratio,
            "absolute_improvement":  absolute_improvement,
            "outcome":               outcome,
            "evidence": {
                "forensics_ref":     f"precision_tuning/forensics_report_{self.attempt}.json",
                "audit_ref":         f"precision_tuning/precision_audit_{self.attempt}.md",
                "match_rate":        match_rate,
                "mismatch_ratio":    mismatch_ratio,
            }
        }

        data["entries"] = [e for e in data["entries"] if e.get("attempt") != self.attempt]
        data["entries"].append(new_entry)
        data["entries"].sort(key=lambda e: e.get("attempt", 0))

        terminal_codes = {
            "max_attempts_reached", "stagnant_same_direction",
            "harmful_regression", "prerequisite_failure",
            "nearly_success",
        }
        if stop_reason_code == "precision_passed":
            data["final_status"] = "success"
            for entry in data["entries"]:
                ir = entry.get("improvement_ratio")
                same_fix = entry.get("fix_type") == fix_type
                nonneg = ir is None or ir >= 0
                entry["contributed"] = same_fix and nonneg
            for entry in data["entries"]:
                if entry.get("attempt") == self.attempt:
                    entry["contributed"] = True
        elif stop_reason_code == "nearly_success":
            data["final_status"] = "nearly_success"
        elif stop_reason_code in terminal_codes:
            data["final_status"] = "failed"

        try:
            os.makedirs(self.tuning_dir, exist_ok=True)
            with open(directions_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _extract_direction_reason(self):
        path = os.path.join(self.tuning_dir, f"precision_audit_{self.attempt}.md")
        if not os.path.exists(path):
            return None

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()

            marker = "[DIRECTION_ASSESSMENT]"
            start = content.find(marker)
            if start == -1:
                return None
            start += len(marker)
            next_bracket = content.find("\n[", start)
            section = content[start:next_bracket].strip() if next_bracket != -1 else content[start:].strip()

            if not section:
                return None

            for line in section.split("\n"):
                if "换方向理由" in line:
                    colon_pos = line.find(":")
                    if colon_pos == -1:
                        continue
                    reason = line[colon_pos + 1:].strip()
                    return reason if reason else None

            return None
        except (OSError, UnicodeDecodeError):
            return None

    def _check_direction_assessment(self) -> str:
        """判定本轮方向延续性，返回四态之一 (12a 防御性兜底):
          - "continue"/"stop": audit 明确给出 否/是
          - "unknown": audit 产出存在但答案模糊 (非 是/否) → 保守按同方向 STOP
          - "missing": audit 产出物理缺失 (文件/marker/section 不存在) → CONTINUE 兜底
            (实测 audit 缺失率高，缺失不应误判 STOP 提前掐断诊断)
        """
        path = os.path.join(self.tuning_dir, f"precision_audit_{self.attempt}.md")
        if not os.path.exists(path):
            return "missing"

        try:
            with open(path) as f:
                content = f.read()

            marker = "[DIRECTION_ASSESSMENT]"
            start = content.find(marker)
            if start == -1:
                return "missing"
            start += len(marker)
            next_bracket = content.find("\n[", start)
            section = content[start:next_bracket].strip() if next_bracket != -1 else content[start:].strip()

            if not section:
                return "missing"

            for line in section.split("\n"):
                key = "本轮是否延续上一轮方向"
                if key not in line and "本轮是否延续" not in line:
                    continue
                colon_pos = line.find(":")
                if colon_pos == -1:
                    continue
                answer = line[colon_pos + 1:].strip()
                first_word = self._extract_direction_first_word(answer)
                if first_word == "否":
                    return "continue"
                elif first_word == "是":
                    return "stop"
            return "unknown"
        except (OSError, UnicodeDecodeError):
            return "missing"

    def _extract_direction_first_word(self, text: str) -> str:
        if not text:
            return ""
        first = text.split()[0] if text.split() else text
        return first.rstrip("，。！？、；：…,.")

    def _validate_direction_binary(self, content: str) -> bool:
        marker = "[DIRECTION_ASSESSMENT]"
        start = content.find(marker)
        if start == -1:
            return False
        start += len(marker)
        next_bracket = content.find("\n[", start)
        section = content[start:next_bracket].strip() if next_bracket != -1 else content[start:].strip()
        for line in section.split("\n"):
            if "本轮是否延续上一轮方向" not in line and "本轮是否延续" not in line:
                continue
            colon_pos = line.find(":")
            if colon_pos == -1:
                continue
            first_word = self._extract_direction_first_word(line[colon_pos + 1:].strip())
            return first_word in ("是", "否")
        return False

    # ----------------------------------------------------------------
    # L5_PROBE 真探针优先治理 (优先真探针 / 连续失败才回退 / 回退留痕)
    # ----------------------------------------------------------------
    def _l5_probe_status(self, content: str) -> str:
        """判定 [L5_PROBE] 真探针状态，返回三态之一:
          - "executed": section 存在，P1/P2/P3 至少一阶段有实测数值 (printf 真值)
          - "skipped":  section 存在且状态=跳过 (带理由的合法跳过，SKILL.md 4 条跳过条件)
          - "missing":  section 缺失，或状态=已执行但三阶段全 N/A / 仅模板占位 (无真数据)
        has_l5_probe 据此判: executed/skipped 算"存在"(保留 SKILL.md 合法跳过通过契约);
        仅 missing 才是真探针失败，连续多轮 missing 才允许 INSTRUMENTATION_FINDINGS 兜底。
        """
        section = self._extract_section(content, "L5_PROBE")
        if section is None:
            return "missing"
        status_skipped = False
        for line in section.split("\n"):
            if "状态" in line:
                parts = re.split(r"[:：]", line, maxsplit=1)
                val = parts[1] if len(parts) > 1 else ""
                # SKILL 模板状态行同时含"已执行 / 跳过"两词，子串匹配两头不讨好:
                # 仅当含"跳过"、不含"已执行"、且无 <...> 未填占位，才是真·合法跳过。
                # 模板残留 (两词俱在或带占位) 不算 skipped，落到下方按实测值判 executed/missing。
                if ("跳过" in val and "已执行" not in val
                        and not re.search(r"<[^>]*>", val)):
                    status_skipped = True
                break
        if status_skipped:
            return "skipped"
        return "executed" if self._l5_probe_has_measured_value(section) else "missing"

    def _l5_probe_has_measured_value(self, section: str) -> bool:
        """P1/P2/P3 阶段行是否含真实测值 (排除 N/A 与未填的 <...> 模板占位)。"""
        for line in section.split("\n"):
            if not re.search(r"\bP[123]\b", line):
                continue
            parts = re.split(r"[:：]", line, maxsplit=1)
            val = parts[1].strip() if len(parts) > 1 else ""
            if not val or (val.startswith("<") and val.endswith(">")):
                continue  # 空 / 未填模板占位
            # 先认真值: 行内存在数字即视为有实测值。"N/A" 仅在剥去 N/A token 后
            # 再无数字时才否决——避免混合行 (如 "x[0]=0.5000 / N/A") 被整行作废。
            stripped = re.sub(r"\bN/?A\b", "", val, flags=re.IGNORECASE)
            if re.search(r"\d", stripped):
                return True
        return False

    def _consecutive_probe_failures(self) -> int:
        """回看本轮之前 (attempt-1, attempt-2, …) 连续真探针缺失 (missing) 的轮数。

        历史某轮 audit md 缺失或不可读 → break (无法判定，保守不累计)，避免把
        provider/budget 异常缺产物的轮误计为真探针失败。
        """
        count = 0
        a = self.attempt - 1
        while a >= 0:
            path = os.path.join(self.tuning_dir, f"precision_audit_{a}.md")
            if not os.path.exists(path):
                break
            try:
                with open(path, encoding="utf-8") as f:
                    prev = f.read()
            except (OSError, UnicodeDecodeError):
                break
            if self._l5_probe_status(prev) == "missing":
                count += 1
                a -= 1
            else:
                break
        return count

    # ================================================================
    # Section 提取与 audit index 写入
    # ================================================================

    def _extract_section(self, content: str, section_name: str):
        marker = f"[{section_name}]"
        start = content.find(marker)
        if start == -1:
            return None
        start += len(marker)
        end_marker = content.find("\n[", start)
        end_audit = content.find("=== END AUDIT ===", start)
        candidates = [pos for pos in [end_marker, end_audit] if pos != -1]
        end = min(candidates) if candidates else len(content)
        text = content[start:end].strip()
        return text if text else None

    def _extract_fix_type(self, content: str):
        section = self._extract_section(content, "FIX_PLAN")
        if not section:
            return None
        m = re.search(r"FIX_PRECISION_\w+", section)
        return m.group(0) if m else None

    def _extract_changed_locations(self, content: str) -> list:
        section = self._extract_section(content, "TARGET_FILES")
        if not section:
            return []
        locations = []
        seen = set()
        for line in section.split("\n"):
            line = line.strip().lstrip("-*•·").strip()
            for part in line.split():
                part = part.rstrip(",:;")
                if re.search(r"\.\w{1,5}$", part) and part not in seen:
                    locations.append(part)
                    seen.add(part)
        return locations

    def _extract_direction_verdict_value(self, content: str):
        if self.attempt == 0:
            return None
        section = self._extract_section(content, "DIRECTION_ASSESSMENT")
        if not section:
            return None
        for line in section.split("\n"):
            if "本轮是否延续上一轮方向" not in line and "本轮是否延续" not in line:
                continue
            colon_pos = line.find(":")
            if colon_pos == -1:
                continue
            first_word = self._extract_direction_first_word(line[colon_pos + 1:].strip())
            if first_word in ("是", "否"):
                return first_word
        return None

    def _write_audit_index(self, content: str, l5_probe_source: str = "L5_PROBE") -> None:
        attempt_dir = os.path.join(self.tuning_dir, "history", f"attempt_{self.attempt}")
        sections_dir = os.path.join(attempt_dir, "sections")
        try:
            os.makedirs(sections_dir, exist_ok=True)
        except OSError:
            return
        # 真探针缺失连续超阈时 check_audit 传入 l5_probe_source=INSTRUMENTATION_FINDINGS,
        # 此时 l5_probe 索引取该 section 内容兜底; section_sources 顶层留痕 (§1.7c 必需标记)。
        section_sources = {}

        SECTION_MAP = [
            ("forensics_summary",        "FORENSICS_SUMMARY"),
            ("computation_decomposition","COMPUTATION_DECOMPOSITION"),
            ("reference_impl_spec",      "REFERENCE_IMPL_SPEC"),
            ("kernel_step_trace",        "KERNEL_STEP_TRACE"),
            ("l5_probe",                 "L5_PROBE"),
            ("knowledge_match",          "KNOWLEDGE_MATCH"),
            ("root_cause",               "ROOT_CAUSE"),
            ("causal_chain_analysis",    "CAUSAL_CHAIN_ANALYSIS"),
            ("fix_plan",                 "FIX_PLAN"),
            ("target_files",             "TARGET_FILES"),
            ("direction_assessment",     "DIRECTION_ASSESSMENT"),
            ("experiment_results",       "EXPERIMENT_RESULTS"),
            ("instrumentation_findings", "INSTRUMENTATION_FINDINGS"),
        ]

        sections_index = {}
        base = f"precision_tuning/history/attempt_{self.attempt}/sections"
        for key, tag in SECTION_MAP:
            extract_tag = tag
            # 真探针缺失连续超阈：l5_probe 改取兜底来源 (INSTRUMENTATION_FINDINGS) 内容。
            if key == "l5_probe" and l5_probe_source != "L5_PROBE":
                extract_tag = l5_probe_source
            sec_text = self._extract_section(content, extract_tag)
            rel_path = f"{base}/{key}.md"
            if sec_text is not None:
                abs_path = os.path.join(sections_dir, f"{key}.md")
                try:
                    with open(abs_path, "w", encoding="utf-8") as f:
                        f.write(f"[{extract_tag}]\n\n{sec_text}\n")
                    sections_index[key] = rel_path
                    if extract_tag != tag:
                        section_sources[key] = extract_tag  # 留痕：内容实际来自别名 section
                except OSError:
                    sections_index[key] = None
            else:
                sections_index[key] = None

        diagnostics = {
            "forensics_hint":    None,
            "op_type":           None,
            "fix_type":          self._extract_fix_type(content),
            "changed_locations": self._extract_changed_locations(content),
            "direction_verdict": self._extract_direction_verdict_value(content),
        }

        n = self.attempt
        index = {
            "forensics":          f"precision_tuning/history/attempt_{n}/forensics_report.json",
            "audit_full":         f"precision_tuning/precision_audit_{n}.md",
            "sections":           sections_index,
            "code_snapshot":      f"precision_tuning/history/attempt_{n}/code_snapshot/",
            "validation":         f"precision_tuning/validation_result_attempt_{n}.json",
            "compilation_log":    None,
            "tuning_directions":  "precision_tuning/tuning_directions.json",
            "forensics_used":     f"precision_tuning/forensics_report_{n}.json",
        }
        # 仅 degraded 兜底发生时写 section_sources (§1.7c 必需标记)，正常轮不污染既有 schema。
        if section_sources:
            index["section_sources"] = section_sources

        initial_summary = {
            "attempt": self.attempt,
            "metrics": {
                "match_rate":            None,
                "mismatch_ratio":        None,
                "improvement_ratio":     None,
                "absolute_improvement":  None,
                "stop_reason_code":      None,
            },
            "diagnostics": diagnostics,
            "index": index,
        }
        summary_path = os.path.join(self.tuning_dir, f"round_summary_{self.attempt}.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(initial_summary, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    # ================================================================
    # 工具
    # ================================================================

    def _load_forensics(self) -> dict:
        """读取当前 attempt 的 forensics_report；失败返回 None。"""
        path = os.path.join(self.tuning_dir, f"forensics_report_{self.attempt}.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _load_full_eval(self) -> dict:
        """读取当前 attempt validation_result 的 full_eval 子字段 (§2.1 全量复验产物)。

        无 full_eval / 文件缺失 / 解析失败 / 闸关 → 返回 None (全量闸关或 only_py 算子的正常态)。
        """
        if os.environ.get("ABLATE_FULL_EVAL") == "1":
            return None
        path = os.path.join(self.tuning_dir, f"validation_result_attempt_{self.attempt}.json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                fe = data.get("full_eval")
                return fe if isinstance(fe, dict) else None
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _load_full_eval_state(self) -> dict:
        """读取 .full_eval_state.json 跨轮收敛态；缺失/损坏返回 {}。"""
        path = os.path.join(self.tuning_dir, ".full_eval_state.json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _get_baseline_match_rate(self, forensics_data: dict = None):
        baseline_path = os.path.join(self.tuning_dir, "baseline_state.json")
        if os.path.exists(baseline_path):
            try:
                with open(baseline_path) as f:
                    bs = json.load(f)
                mr = bs.get("match_rate")
                if mr is not None:
                    return float(mr)
            except (json.JSONDecodeError, OSError, ValueError):
                pass

        fr = forensics_data if forensics_data is not None else self._load_forensics()
        if fr is not None:
            try:
                history_trend = fr.get("history_trend")
                if history_trend:
                    trend_list = history_trend.get("trend", [])
                    if len(trend_list) >= 2:
                        baseline_mismatch = trend_list[0].get("mismatch_ratio")
                        if baseline_mismatch is not None:
                            return round((1 - float(baseline_mismatch)) * 100, 4)

                if self.attempt == 0:
                    outputs = fr.get("outputs", [])
                    if outputs:
                        raw_mr = outputs[0].get("basic_stats", {}).get("match_rate")
                        if raw_mr is not None:
                            return round(float(raw_mr) * 100, 4)

            except (json.JSONDecodeError, OSError, ValueError, KeyError):
                pass

        return None

    def _kernel_dir(self):
        kdir = os.path.join(self.task_dir, "kernel")
        return kdir if os.path.isdir(kdir) else None

    def _result(self, gate_name: str, checks: dict) -> dict:
        return {"gate": gate_name, "passed": all(checks.values()), "checks": checks}


def _legacy_to_outcome(raw: dict) -> GateOutcome:
    """把 _LegacyPrecisionChecker 返回的 dict (gate/passed/checks[/loop_*]) 转成 GateOutcome。"""
    checks = dict(raw.get("checks", {}))
    # 附带 prerequisite_error / stop_reason_code / attempt / max_attempts 等保留到 checks。
    # l5_probe_degraded / l5_probe_source 同走 checks 通道 (GateOutcome.to_gate_output
    # 仅输出 checks，这是唯一能穿过投影链落进 events 的载体)；二者在 check_audit 内
    # passed 算完后才挂顶层，故此处搬入 checks 不影响 all(checks.values()) 判定。
    for k in ("prerequisite_error", "stop_reason_code", "attempt", "max_attempts",
              "l5_probe_degraded", "l5_probe_source"):
        if k in raw:
            checks[k] = raw[k]
    return GateOutcome(
        gate=raw.get("gate", ""),
        ok=bool(raw.get("passed", False)),
        checks=checks,
        loop_signal=raw.get("loop_signal"),
        reason=raw.get("loop_reason"),
    )


class PrecisionBranch:
    """精度分支的 Gate 入口。与其它 branch_* 的签名对齐。

    方法签名: run_gate_f(task_dir, attempt) / run_gate_a(...) / run_gate_v(...)
    task_dir: pathlib.Path 或 str 都接受
    """

    def __init__(self, op_name: str):
        self.op_name = op_name

    def _checker(self, task_dir, attempt: int) -> _LegacyPrecisionChecker:
        return _LegacyPrecisionChecker(self.op_name, str(task_dir), attempt)

    def run_gate_f(self, task_dir, attempt: int) -> GateOutcome:
        raw = self._checker(task_dir, attempt).check_forensics()
        return _legacy_to_outcome(raw)

    def run_gate_a(self, task_dir, attempt: int) -> GateOutcome:
        raw = self._checker(task_dir, attempt).check_audit()
        return _legacy_to_outcome(raw)

    def run_gate_v(self, task_dir, attempt: int) -> GateOutcome:
        raw = self._checker(task_dir, attempt).check_validate()
        return _legacy_to_outcome(raw)
