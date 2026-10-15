"""test_validate_runner.py — forensics staleness 缓存 (建议A, Tier 1)。

不真跑 precision_forensics.py 子进程: monkeypatch subprocess.run 计数调用次数，验证:
  1. 首轮无缓存 → 跑子进程 (miss)。
  2. 源码 hash 不变 → 第二轮复用上轮 report，不跑子进程 (hit)，且复制成当前 attempt 名。
  3. kernel 源码改动 → hash 变 → 重跑 (miss)。
  4. 新增/删除源文件 → hash 变 → 重跑 (miss)。
  5. ASCENDC_DEBUG_FORENSICS_NO_CACHE=1 → 永不复用。
  6. 上轮 report 已被删 → 即便 hash 命中也回退重跑 (向正确性倾斜)。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine import validate_runner


def _make_task(tmp: Path) -> Path:
    """造一个含 kernel/ + model_new_ascendc.py 的最小任务目录。"""
    (tmp / "kernel").mkdir(parents=True, exist_ok=True)
    (tmp / "kernel" / "op.cpp").write_text("// v1\n", encoding="utf-8")
    (tmp / "kernel" / "op.h").write_text("#pragma once\n", encoding="utf-8")
    (tmp / "model_new_ascendc.py").write_text("x = 1\n", encoding="utf-8")
    (tmp / "precision_tuning").mkdir(parents=True, exist_ok=True)
    return tmp


class _FakeForensics:
    """fake subprocess.run: 计调用次数，并把 report 写到当前 attempt 名 (模拟脚本产出)。"""

    def __init__(self, task_dir: Path) -> None:
        self.task_dir = task_dir
        self.calls = 0

    def __call__(self, cmd, **kwargs):
        # prebuild step (build_ascendc.py) 不带 --attempt，直接返回成功，不计入 calls。
        if "--attempt" not in cmd:
            class _R:
                returncode = 0
                stderr = ""
            return _R()
        self.calls += 1
        # 从 cmd 里取 --attempt 值，产出对应 report (模拟真实脚本行为)。
        attempt = cmd[cmd.index("--attempt") + 1]
        report = self.task_dir / "precision_tuning" / f"forensics_report_{attempt}.json"
        report.write_text(json.dumps({"attempt": int(attempt)}), encoding="utf-8")

        class _R:
            returncode = 0
            stderr = ""
        return _R()


class TestForensicsCache(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = _make_task(Path(self._tmp.name))
        self._env_backup = os.environ.pop("ASCENDC_DEBUG_FORENSICS_NO_CACHE", None)

    def tearDown(self) -> None:
        if self._env_backup is not None:
            os.environ["ASCENDC_DEBUG_FORENSICS_NO_CACHE"] = self._env_backup
        else:
            os.environ.pop("ASCENDC_DEBUG_FORENSICS_NO_CACHE", None)
        self._tmp.cleanup()

    def _run(self, attempt: int, fake: _FakeForensics) -> dict:
        with mock.patch.object(validate_runner.subprocess, "run", fake):
            return validate_runner.run_forensics(self.task_dir, attempt=attempt)

    def test_miss_then_hit(self) -> None:
        fake = _FakeForensics(self.task_dir)
        r0 = self._run(0, fake)
        self.assertTrue(r0["success"])
        self.assertEqual(fake.calls, 1)
        self.assertFalse(r0.get("cached"))  # 重跑路径显式 cached=False
        self.assertFalse(r0["cache_hit"])
        self.assertTrue(r0["forensics_executed"])

        # 源码未变 → 第二轮复用，不跑子进程。
        r1 = self._run(1, fake)
        self.assertTrue(r1["success"])
        self.assertEqual(fake.calls, 1, "源码不变应复用，不应再跑子进程")
        self.assertTrue(r1.get("cached"))
        self.assertEqual(r1.get("cached_from_attempt"), 0)
        self.assertTrue(r1["cache_hit"])
        self.assertEqual(r1["reuse_kind"], "input_hash")
        self.assertFalse(r1["forensics_executed"])
        self.assertTrue(r1["build_skipped"])
        # 复用应把上轮 report 复制成当前 attempt 名 (Gate-F 精确文件名读取)。
        report = self.task_dir / "precision_tuning" / "forensics_report_1.json"
        self.assertTrue(report.exists())
        payload = json.loads(report.read_text())
        self.assertEqual(payload["attempt"], 1)
        self.assertTrue(payload["cached"])
        self.assertEqual(payload["cached_from_attempt"], 0)

    def test_src_change_triggers_rerun(self) -> None:
        fake = _FakeForensics(self.task_dir)
        self._run(0, fake)
        self.assertEqual(fake.calls, 1)

        # 改 kernel 源码 → hash 变 → 重跑。
        (self.task_dir / "kernel" / "op.cpp").write_text("// v2 changed\n", encoding="utf-8")
        r1 = self._run(1, fake)
        self.assertEqual(fake.calls, 2, "源码改动应重跑")
        self.assertFalse(r1.get("cached"))


    def test_add_file_triggers_rerun(self) -> None:
        fake = _FakeForensics(self.task_dir)
        self._run(0, fake)
        self.assertEqual(fake.calls, 1)

        # 新增源文件 → hash 变 (文件名并入 hash) → 重跑。
        (self.task_dir / "kernel" / "extra.cpp").write_text("// new\n", encoding="utf-8")
        self._run(1, fake)
        self.assertEqual(fake.calls, 2, "新增源文件应重跑")

    def test_no_cache_env_disables(self) -> None:
        os.environ["ASCENDC_DEBUG_FORENSICS_NO_CACHE"] = "1"
        fake = _FakeForensics(self.task_dir)
        self._run(0, fake)
        self._run(1, fake)
        self.assertEqual(fake.calls, 2, "NO_CACHE=1 应禁用复用，每轮都跑")

    def test_missing_prev_report_falls_back(self) -> None:
        fake = _FakeForensics(self.task_dir)
        self._run(0, fake)
        self.assertEqual(fake.calls, 1)

        # 删掉上轮 report → 即便 hash 命中也应回退重跑。
        (self.task_dir / "precision_tuning" / "forensics_report_0.json").unlink()
        r1 = self._run(1, fake)
        self.assertEqual(fake.calls, 2, "上轮 report 缺失应回退重跑")
        self.assertFalse(r1.get("cached"))


class TestBuildFailureForensicsDegrade(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = _make_task(Path(self._tmp.name))
        status_dir = self.task_dir / ".verify_status"
        status_dir.mkdir()
        self.log_path = self.task_dir / ".verify_logs" / "phase8_attempt0.log"
        self.log_path.parent.mkdir()
        self.log_path.write_text(
            "kernel/op.cpp:42:7: error: invalid AscendC API call\n",
            encoding="utf-8",
        )
        (status_dir / "latest.json").write_text(json.dumps({
            "failure_type": "build_failed",
            "failed_step": "compile",
            "log_path": str(self.log_path),
            "verification_exit_code": 1,
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_existing_build_log_skips_rebuild_and_proceeds_to_agent(self) -> None:
        with mock.patch.object(validate_runner, "_run_forensics_prebuild") as prebuild, \
                mock.patch.object(validate_runner.subprocess, "run") as child:
            result = validate_runner.run_forensics(
                self.task_dir, attempt=1, failure_type="build_failed")

        prebuild.assert_not_called()
        child.assert_not_called()
        self.assertTrue(result["success"])
        self.assertTrue(result["forensics_degraded"])
        self.assertTrue(result["proceed_to_agent"])
        self.assertEqual(result["diagnostic_evidence_kind"], "existing_build_log")
        self.assertFalse(result["prebuild_executed"])
        self.assertTrue(result["build_skipped"])
        self.assertEqual(result["build_skip_reason"], "existing_build_failure_evidence")
        report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "build_failed")
        self.assertIn("invalid AscendC API call", "\n".join(report["first_error_lines"]))


class TestRuntimeFailureForensicsDegrade(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = _make_task(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _failed_child(self, *, error: str):
        def run(cmd, **kwargs):
            attempt = int(cmd[cmd.index("--attempt") + 1])
            report = (
                self.task_dir / "precision_tuning"
                / f"forensics_report_{attempt}.json"
            )
            report.write_text(json.dumps({
                "version": "2.0",
                "attempt": attempt,
                "status": "error",
                "error": error,
                "traceback": error,
                "outputs": [],
                "primary_hint": "error",
            }), encoding="utf-8")

            class Result:
                returncode = 1
                stderr = error
            return Result()
        return run

    def test_aicore_runtime_error_proceeds_to_agent_without_retry(self) -> None:
        error = (
            "RuntimeError: ACL stream synchronize failed, error code:507015; "
            "rtDeviceSynchronize execution failed, reason=aicore exception"
        )
        with mock.patch.object(validate_runner, "_kernel_build_ready", return_value=True), \
                mock.patch.object(
                    validate_runner.subprocess, "run",
                    self._failed_child(error=error),
                ):
            result = validate_runner.run_forensics(
                self.task_dir, attempt=1, failure_type="runtime_error")

        self.assertTrue(result["success"])
        self.assertTrue(result["forensics_degraded"])
        self.assertTrue(result["proceed_to_agent"])
        self.assertEqual(result["unavailable_reason"], "runtime_error")
        self.assertEqual(result["diagnostic_evidence_kind"], "runtime_error_log")
        self.assertTrue(result["forensics_executed"])
        self.assertFalse(result["forensics_completed"])
        report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
        self.assertTrue(report["proceed_to_agent"])
        self.assertIn("aicore exception", report["diagnostic_signatures"])

    def test_unclassified_child_failure_remains_retryable(self) -> None:
        with mock.patch.object(validate_runner, "_kernel_build_ready", return_value=True), \
                mock.patch.object(
                    validate_runner.subprocess, "run",
                    self._failed_child(
                        error="temporary rtDeviceSynchronize command timeout"
                    ),
                ):
            result = validate_runner.run_forensics(
                self.task_dir, attempt=1, failure_type="runtime_error")

        self.assertFalse(result["success"])
        self.assertNotIn("forensics_degraded", result)
        self.assertNotIn("proceed_to_agent", result)


class TestRunFullEval(unittest.TestCase):
    """全量复验 (§2.1): 备份→覆盖<op>.json→跑→finally恢复 的隔离正确性。

    mock _run_logged 不真跑 verification: 按需写 case 行到 stdout 控制 match_rate，
    核心断言全量跑前后生效 <op>.json 字节一致 (恢复正确，不污染 agent 后续输入)。
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)
        (self.task_dir / "precision_tuning").mkdir(parents=True)
        # 轻量 5_FakeOp.json (2 行) + 全量 .bak (4 行)，含 inputs 键以过 _looks_like_case_jsonl。
        self.light = self.task_dir / "5_FakeOp.json"
        self.light.write_text(
            '{"inputs": [1]}\n{"inputs": [2]}\n', encoding="utf-8")
        self.full = self.task_dir / "5_FakeOp.json.bak"
        self.full.write_text(
            '{"inputs": [1]}\n{"inputs": [2]}\n{"inputs": [3]}\n{"inputs": [4]}\n',
            encoding="utf-8")
        self._env_backup = {k: os.environ.pop(k, None)
                            for k in ("ABLATE_FULL_EVAL", "ASCENDC_ABLATE_FULL_EVAL")}

    def tearDown(self) -> None:
        for k, v in self._env_backup.items():
            if v is not None:
                os.environ[k] = v
        self._tmp.cleanup()

    def _fake_run_logged(self, *, case_lines: str, rc: int):
        """返回一个 fake _run_logged: 把 case_lines 写进 stdout_path 并返回 rc。"""
        def _fake(cmd, *, cwd, env, stdout_path, stderr_path, title, timeout):
            Path(stdout_path).write_text(case_lines, encoding="utf-8")
            return rc
        return _fake

    def _call(self):
        return validate_runner._run_full_eval(
            self.task_dir, attempt=0, repo_root=Path("/repo"),
            env={}, clean_build=True, timeout=None)

    def test_full_eval_all_matched_restores_json(self) -> None:
        before = self.light.read_bytes()
        cases = "case[0]: output: matched\ncase[1]: output: matched\n" \
                "case[2]: output: matched\ncase[3]: output: matched\n"
        with mock.patch.object(validate_runner, "_run_logged",
                               self._fake_run_logged(case_lines=cases, rc=0)):
            fe = self._call()
        self.assertTrue(fe["ran"])
        self.assertFalse(fe["crashed"])
        self.assertEqual(fe["total_cases"], 4)
        self.assertEqual(fe["passed_cases"], 4)
        self.assertEqual(fe["full_json_source"], "5_FakeOp.json.bak")
        # 关键: 生效 <op>.json 恢复为轻量原文 (字节一致)。
        self.assertEqual(self.light.read_bytes(), before)
        # 备份文件已删。
        self.assertFalse(
            (self.task_dir / "5_FakeOp.json.lightweight_bak_attempt0").exists())
        # 单独落盘 + 跨轮状态。
        self.assertTrue(
            (self.task_dir / "precision_tuning" / "validation_result_attempt_0_full.json").exists())
        self.assertTrue(
            (self.task_dir / "precision_tuning" / ".full_eval_state.json").exists())

    def test_full_eval_swaps_and_restores_model_json_alias(self) -> None:
        model_json = self.task_dir / "model.json"
        model_json.write_bytes(self.light.read_bytes())
        before = model_json.read_bytes()
        seen: dict[str, bytes] = {}

        def fake_run_logged(
            _cmd, *, cwd, env, stdout_path, stderr_path, title, timeout
        ):
            seen["active"] = self.light.read_bytes()
            seen["model"] = model_json.read_bytes()
            Path(stdout_path).write_text(
                "case[0]: output: matched\n", encoding="utf-8")
            return 0

        with mock.patch.object(
            validate_runner, "_run_logged", side_effect=fake_run_logged
        ):
            fe = self._call()

        self.assertEqual(seen["active"], self.full.read_bytes())
        self.assertEqual(seen["model"], self.full.read_bytes())
        self.assertEqual(self.light.read_bytes(), before)
        self.assertEqual(model_json.read_bytes(), before)
        self.assertEqual(fe["active_json_aliases"], ["model.json"])

    def test_full_eval_partial_fail_restores_json(self) -> None:
        before = self.light.read_bytes()
        cases = "case[0]: output: matched\ncase[1]: output: matched\n" \
                "case[2]: output: mismatch_ratio=10.000000%\ncase[3]: output: matched\n"
        with mock.patch.object(validate_runner, "_run_logged",
                               self._fake_run_logged(case_lines=cases, rc=1)):
            fe = self._call()
        self.assertEqual(fe["total_cases"], 4)
        self.assertEqual(fe["passed_cases"], 3)
        self.assertFalse(fe["crashed"])
        self.assertEqual(self.light.read_bytes(), before)

    def test_full_eval_crash_restores_json(self) -> None:
        before = self.light.read_bytes()
        with mock.patch.object(validate_runner, "_run_logged",
                               self._fake_run_logged(case_lines="Traceback...\n", rc=1)):
            fe = self._call()
        self.assertTrue(fe["crashed"])  # rc!=0 且无 case 数据
        self.assertEqual(self.light.read_bytes(), before)

    def test_no_fuller_json_returns_none(self) -> None:
        # 删掉 .bak → 只剩轻量 → 无更全集合 → None (only_py 算子同此态)。
        self.full.unlink()
        with mock.patch.object(validate_runner, "_run_logged",
                               self._fake_run_logged(case_lines="", rc=0)):
            fe = self._call()
        self.assertIsNone(fe)

    def test_equal_count_different_case_set_still_runs(self) -> None:
        self.light.write_text(
            '{"inputs": [1]}\n{"inputs": [2]}\n', encoding="utf-8")
        self.full.write_text(
            '{"inputs": [1]}\n{"inputs": [99]}\n', encoding="utf-8")
        cases = "case[0]: output: matched\ncase[1]: output: matched\n"
        fake = mock.Mock(
            side_effect=self._fake_run_logged(case_lines=cases, rc=0))
        with mock.patch.object(
            validate_runner,
            "_run_logged",
            fake,
        ):
            fe = self._call()
        self.assertTrue(fe["ran"])
        self.assertFalse(fe["coverage_equivalent"])
        fake.assert_called_once()

    def test_equal_semantic_case_set_records_equivalence_without_rerun(self) -> None:
        self.light.write_text(
            '{"inputs":[1],"meta":{"b":2,"a":1}}\n{"inputs":[2]}\n',
            encoding="utf-8",
        )
        self.full.write_text(
            '{"inputs": [2]}\n{"meta":{"a":1,"b":2},"inputs":[1]}\n',
            encoding="utf-8",
        )
        with mock.patch.object(validate_runner, "_run_logged") as run:
            fe = self._call()
        self.assertFalse(fe["ran"])
        self.assertTrue(fe["coverage_equivalent"])
        self.assertEqual(
            fe["active_case_set_sha256"], fe["full_case_set_sha256"])
        run.assert_not_called()
        persisted = json.loads(
            (
                self.task_dir
                / "precision_tuning"
                / "validation_result_attempt_0_full.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(persisted["coverage_equivalent"])


class TestForensicsReuseAfterRollback(unittest.TestCase):
    """问题 7: 回滚后复用 best 轮 forensics_report，不跑取证子进程 (省编译+取证)。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = _make_task(Path(self._tmp.name))
        os.environ.pop("ASCENDC_DEBUG_FORENSICS_NO_CACHE", None)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, attempt: int, fake: _FakeForensics) -> dict:
        with mock.patch.object(validate_runner.subprocess, "run", fake):
            return validate_runner.run_forensics(self.task_dir, attempt=attempt)

    def test_reuse_best_report_no_subprocess(self) -> None:
        from engine import transition
        # report_0 是 attempt0 修复前；report_1 才对应 attempt0 validate 后保存的 best 源码。
        (self.task_dir / "precision_tuning" / "forensics_report_0.json").write_text(
            json.dumps({"attempt": 0, "primary_hint": "stale_before_best"}), encoding="utf-8")
        (self.task_dir / "precision_tuning" / "forensics_report_1.json").write_text(
            json.dumps({"attempt": 1, "primary_hint": "uniform_offset"}), encoding="utf-8")
        # 上一轮(attempt 2)触发回滚到 best(attempt 0)
        transition.record_rollback(
            self.task_dir, from_attempt=2,
            best_metric={"attempt": 0, "case_pass_rate": 50.0, "match_rate": "57.62"})
        # 本轮 attempt=3 紧跟回滚 → 应复用 best 报告，不跑子进程
        fake = _FakeForensics(self.task_dir)
        r = self._run(3, fake)
        self.assertTrue(r["success"])
        self.assertEqual(fake.calls, 0, "回滚后应复用 best 报告，不跑取证子进程")
        self.assertTrue(r.get("reused_after_rollback"))
        self.assertEqual(r.get("reused_from_attempt"), 1)
        self.assertEqual(r.get("reused_best_attempt"), 0)
        self.assertTrue(r["cache_hit"])
        self.assertEqual(r["reuse_kind"], "rollback")
        self.assertTrue(r["forensics_reused"])
        self.assertFalse(r["forensics_executed"])
        self.assertFalse(r["prebuild_executed"])
        self.assertTrue(r["build_skipped"])
        # best 报告应被复制成当前 attempt 名，供 Gate-F/knowledge_search 读取
        cur = self.task_dir / "precision_tuning" / "forensics_report_3.json"
        self.assertTrue(cur.exists())
        payload = json.loads(cur.read_text())
        self.assertEqual(payload["primary_hint"], "uniform_offset")
        self.assertEqual(payload["attempt"], 3)
        self.assertTrue(payload["reused_after_rollback"])

    def test_no_reuse_when_not_after_rollback(self) -> None:
        # 无回滚事件 → 正常跑子进程
        fake = _FakeForensics(self.task_dir)
        r = self._run(1, fake)
        self.assertEqual(fake.calls, 1)
        self.assertFalse(r.get("reused_after_rollback"))

    def test_fallback_when_best_report_missing(self) -> None:
        from engine import transition
        # 回滚事件指向 best attempt 0，但其 report 不存在 → 回退正常执行
        transition.record_rollback(
            self.task_dir, from_attempt=2,
            best_metric={"attempt": 0, "case_pass_rate": 50.0, "match_rate": "57.62"})
        fake = _FakeForensics(self.task_dir)
        r = self._run(3, fake)
        self.assertEqual(fake.calls, 1, "best 报告缺失应回退重跑")
        self.assertFalse(r.get("reused_after_rollback"))

    def test_no_reuse_when_rollback_not_immediately_prior(self) -> None:
        from engine import transition
        (self.task_dir / "precision_tuning" / "forensics_report_1.json").write_text(
            json.dumps({"attempt": 1}), encoding="utf-8")
        # 回滚发生在 from_attempt=2，但本轮是 attempt=5 (非紧邻) → 不复用
        transition.record_rollback(
            self.task_dir, from_attempt=2,
            best_metric={"attempt": 0, "case_pass_rate": 50.0, "match_rate": "57.62"})
        fake = _FakeForensics(self.task_dir)
        r = self._run(5, fake)
        self.assertEqual(fake.calls, 1)
        self.assertFalse(r.get("reused_after_rollback"))


if __name__ == "__main__":
    unittest.main()
