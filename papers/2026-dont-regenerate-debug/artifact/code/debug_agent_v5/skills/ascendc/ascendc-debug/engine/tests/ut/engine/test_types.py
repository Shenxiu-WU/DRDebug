"""test_types.py — 闭集 dataclass 与 Literal 表的自洽性。

验收点 (REWRITE_PLAN §6.1): Action / 终态闭集。
Literal 在运行期不强制，故这里测「数据结构构造 + 闭集表互相一致」，把口径错误
(漏值 / 偏序矛盾) 在 Step 1 就钉死，供后续 state/next_action 安全消费。
"""
from __future__ import annotations

import unittest
from dataclasses import fields

from engine.types import (
    DEBUGGABLE_FAILURE_TYPES,
    PIPELINE_ORDER,
    Abort,
    Action,
    Continue,
    Done,
    Escalate,
)


class TestDecisionConstruction(unittest.TestCase):
    def test_action_spawn_agent(self) -> None:
        a = Action(kind="spawn_agent", name="constructive", step="audit",
                   skill_args={"attempt": 0}, expected_artifacts=["precision_audit_0.md"])
        self.assertEqual(a.kind, "spawn_agent")
        self.assertEqual(a.expected_artifacts, ["precision_audit_0.md"])

    def test_action_py_action_defaults(self) -> None:
        a = Action(kind="py_action", name="forensics")
        self.assertIsNone(a.step)
        self.assertEqual(a.expected_artifacts, [])  # field(default_factory=list)

    def test_action_artifacts_not_shared(self) -> None:
        # default_factory 必须每实例独立，否则跨决策污染。
        a1 = Action(kind="py_action", name="x")
        a2 = Action(kind="py_action", name="y")
        a1.expected_artifacts.append("z")
        self.assertEqual(a2.expected_artifacts, [])

    def test_continue_carries_attempt_boundary(self) -> None:
        c = Continue(next_attempt=1, next_failure_type="build_failed", reason="精度修复后引入编译错")
        self.assertEqual(c.next_attempt, 1)
        self.assertEqual(c.next_failure_type, "build_failed")

    def test_done_requires_outcome_and_reason(self) -> None:
        d = Done(session_outcome="success", reason="验证全过")
        self.assertEqual(d.session_outcome, "success")

    def test_abort_optional_fields(self) -> None:
        ab = Abort(category="state_machine_stuck")
        self.assertIsNone(ab.reason)
        self.assertIsNone(ab.details)

    def test_escalate(self) -> None:
        e = Escalate(reason="need human")
        self.assertEqual(e.reason, "need human")


class TestClosedSetConsistency(unittest.TestCase):
    def test_debuggable_subset_of_pipeline(self) -> None:
        # 五个可调试分支都必须在偏序表里有位置。
        for ft in DEBUGGABLE_FAILURE_TYPES:
            self.assertIn(ft, PIPELINE_ORDER, f"{ft} 缺偏序定义")

    def test_pipeline_keys_are_debuggable_only(self) -> None:
        # 偏序表不该混入 success / execution_aborted (它们不是可调试分支)。
        self.assertEqual(set(PIPELINE_ORDER), set(DEBUGGABLE_FAILURE_TYPES))

    def test_precision_is_last(self) -> None:
        # 不变量地基: precision 必须是偏序最末 → 其计数永不被跨分支重置。
        self.assertEqual(PIPELINE_ORDER["precision_failed"], max(PIPELINE_ORDER.values()))

    def test_build_is_first(self) -> None:
        self.assertEqual(PIPELINE_ORDER["build_failed"], min(PIPELINE_ORDER.values()))

    def test_runtime_timeout_same_stage(self) -> None:
        # runtime 与 timeout 同属执行期，同级。
        self.assertEqual(PIPELINE_ORDER["runtime_error"], PIPELINE_ORDER["timeout"])

    def test_pipeline_partial_order_monotone(self) -> None:
        # 偏序值应覆盖 0..max 连续区间 (无空洞)，便于「比新状态更靠前」判定。
        vals = sorted(set(PIPELINE_ORDER.values()))
        self.assertEqual(vals, list(range(len(vals))))

    def test_action_field_names_stable(self) -> None:
        # 锁定 Action 字段集，防后续 state/next_action 依赖的字段被悄悄改名。
        names = {f.name for f in fields(Action)}
        self.assertEqual(
            names, {"kind", "name", "step", "skill_args", "expected_artifacts"})

    def test_session_outcome_matches_exit_code_map(self) -> None:
        # SessionOutcome 闭集 与 __main__._OUTCOME_EXIT_CODE 必须一一对应。
        # 防回归: provider_api_error 曾有退出码却漏出闭集 (Tier 0-2 审查发现)。
        from typing import get_args
        from engine.types import SessionOutcome
        from engine.__main__ import _OUTCOME_EXIT_CODE
        outcomes = set(get_args(SessionOutcome))
        self.assertEqual(
            outcomes, set(_OUTCOME_EXIT_CODE),
            "SessionOutcome 闭集与退出码映射不一致 (漏值或编外值)")
        # 退出码无重复。
        codes = list(_OUTCOME_EXIT_CODE.values())
        self.assertEqual(len(codes), len(set(codes)), "退出码有重复")


if __name__ == "__main__":
    unittest.main()
