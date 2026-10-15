"""test_agent_backend.py — diagnose_and_fix 的 claude backend (方案 C 唯一 spawn 点)。

不真跑 claude: 用 fake _run 注入，验证三件事:
  1. Action → claude 命令行参数正确 (--agent/--session-id/--add-dir/--allowedTools/prompt)。
  2. claude --output-format json 结果 → result dict 映射 (ok/pause_turn/error/api_error)。
  3. make_agent_callback 闭包注入 runner，跑通 precision→CONTINUE→PASS 2 轮 (claude 被 fake，
     gate 用 mock dispatcher)，确认 backend 与 runner 接口契合。
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine.agent_backend import (
    _build_prompt,
    _finalize_probe_policy,
    make_agent_callback,
    spawn_diagnose_agent,
)
from engine.types import Action


def _diagnose_action(failure_type="precision_failed", attempt=0) -> Action:
    return Action(kind="spawn_agent", name="debug_worker", step="diagnose_and_fix",
                  skill_args={"failure_type": failure_type, "attempt": attempt})


def _fake_run_factory(claude_json: dict, capture: dict):
    """造一个 fake subprocess.run: 记录 cmd，并把 claude_json 写进 stdout 文件句柄。"""
    def _fake_run(cmd, stdout=None, stderr=None, timeout=None, check=False, text=True):
        capture["cmd"] = cmd
        capture["timeout"] = timeout
        if stdout is not None:
            stdout.write(json.dumps(claude_json))
        class _R:
            returncode = 0
        return _R()
    return _fake_run


class TestCommandConstruction(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_cmd_has_required_flags(self) -> None:
        cap = {}
        spawn_diagnose_agent(
            _diagnose_action("build_failed", 2), self.task_dir, "add", 2,
            agent_name="ascendc-debug-agent-constructive", model="claude-x",
            allowed_tools="Bash,Read", npu="3",
            _run=_fake_run_factory({"is_error": False}, cap))
        cmd = cap["cmd"]
        self.assertIn("--agent", cmd)
        self.assertEqual(cmd[cmd.index("--agent") + 1], "ascendc-debug-agent-constructive")
        self.assertIn("--session-id", cmd)
        self.assertIn("--add-dir", cmd)
        self.assertIn("--allowedTools", cmd)
        self.assertEqual(cmd[cmd.index("--allowedTools") + 1], "Bash,Read")
        self.assertIn("--output-format", cmd)
        self.assertEqual(cmd[cmd.index("--output-format") + 1], "json")
        self.assertNotIn("--bare", cmd)
        self.assertIn("--setting-sources", cmd)
        self.assertEqual(cmd[cmd.index("--setting-sources") + 1], "project")
        # prompt 是最后一个位置参数，含本轮 failure_type + attempt + 单轮约束。
        prompt = cmd[-1]
        self.assertIn("build_failed", prompt)
        self.assertIn("attempt: 2", prompt)
        self.assertIn("单个 attempt", prompt)
        self.assertIn("禁止自己跑", prompt)

class TestTurnsWiring(unittest.TestCase):
    """失控治本闸接线: 仅 --max-turns (模型无关)；杜绝任何 usd 量纲闸。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_max_turns_appended_when_set(self) -> None:
        cap = {}
        spawn_diagnose_agent(
            _diagnose_action("precision_failed", 0), self.task_dir, "add", 0,
            max_turns="120",
            _run=_fake_run_factory({"is_error": False}, cap))
        cmd = cap["cmd"]
        self.assertIn("--max-turns", cmd)
        self.assertEqual(cmd[cmd.index("--max-turns") + 1], "120")

    def test_no_turns_flag_when_unset(self) -> None:
        cap = {}
        spawn_diagnose_agent(
            _diagnose_action("precision_failed", 0), self.task_dir, "add", 0,
            _run=_fake_run_factory({"is_error": False}, cap))
        self.assertNotIn("--max-turns", cap["cmd"])

    def test_task_remaining_clamps_session_max_turns(self) -> None:
        cap = {}
        action = _diagnose_action("precision_failed", 2)
        action = Action(
            kind=action.kind,
            name=action.name,
            step=action.step,
            skill_args={**action.skill_args, "task_turns_used": 524,
                        "task_turns_limit": 600, "task_turns_remaining": 76,
                        "task_turn_budget_tier": "hard_extended"},
        )
        result = spawn_diagnose_agent(
            action, self.task_dir, "add", 2, max_turns="240",
            _run=_fake_run_factory({"is_error": False}, cap))
        cmd = cap["cmd"]
        self.assertEqual(cmd[cmd.index("--max-turns") + 1], "76")
        self.assertEqual(result["max_turns_applied"], "76")
        self.assertEqual(result["task_turns_remaining"], 76)

    def test_dynamic_task_boundary_hit_is_observable(self) -> None:
        cap = {}
        action = _diagnose_action("precision_failed", 2)
        action = Action(
            kind=action.kind,
            name=action.name,
            step=action.step,
            skill_args={**action.skill_args, "task_turns_used": 404,
                        "task_turns_limit": 480, "task_turns_remaining": 76,
                        "task_turn_budget_tier": "soft"},
        )
        result = spawn_diagnose_agent(
            action, self.task_dir, "add", 2, max_turns="240",
            _run=_fake_run_factory({
                "is_error": True,
                "subtype": "error_max_turns",
                "num_turns": 76,
            }, cap),
        )
        self.assertEqual(result["claude_state"], "max_turns_exceeded")
        self.assertTrue(result["session_turn_cap_hit"])
        self.assertTrue(result["task_budget_boundary_hit"])

    def test_ordinary_session_cap_is_not_task_boundary(self) -> None:
        cap = {}
        action = _diagnose_action("precision_failed", 0)
        action = Action(
            kind=action.kind,
            name=action.name,
            step=action.step,
            skill_args={**action.skill_args, "task_turns_used": 0,
                        "task_turns_limit": 480, "task_turns_remaining": 480,
                        "task_turn_budget_tier": "soft"},
        )
        result = spawn_diagnose_agent(
            action, self.task_dir, "add", 0, max_turns="240",
            _run=_fake_run_factory({
                "is_error": True,
                "subtype": "error_max_turns",
                "num_turns": 240,
            }, cap),
        )
        self.assertTrue(result["session_turn_cap_hit"])
        self.assertFalse(result["task_budget_boundary_hit"])

    def test_no_usd_budget_flag_ever(self) -> None:
        # 护栏: usd 闸已彻底移除 (换模型即失效)，命令行永不出现 --max-budget-usd。
        cap = {}
        spawn_diagnose_agent(
            _diagnose_action("precision_failed", 0), self.task_dir, "add", 0,
            max_turns="120",
            _run=_fake_run_factory({"is_error": False}, cap))
        self.assertNotIn("--max-budget-usd", cap["cmd"])
        self.assertNotIn("--max-budget", " ".join(cap["cmd"]))


class TestResultClassification(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _spawn(self, claude_json):
        cap = {}
        return spawn_diagnose_agent(
            _diagnose_action(), self.task_dir, "add", 0,
            _run=_fake_run_factory(claude_json, cap))

    def test_ok_result_success(self) -> None:
        r = self._spawn({"is_error": False, "stop_reason": "end_turn"})
        self.assertTrue(r["success"])
        self.assertEqual(r["claude_state"], "ok")
        self.assertIn("session_id", r)
        self.assertIn("result_path", r)
        self.assertIn("latest_result_path", r)

    def test_result_archived_per_session_and_latest_kept(self) -> None:
        r1 = self._spawn({"is_error": False, "stop_reason": "end_turn", "result": "first"})
        r2 = self._spawn({"is_error": False, "stop_reason": "end_turn", "result": "second"})

        p1 = Path(r1["result_path"])
        p2 = Path(r2["result_path"])
        latest = self.task_dir / "_claude_result_attempt0.json"
        self.assertTrue(p1.exists())
        self.assertTrue(p2.exists())
        self.assertNotEqual(p1, p2)
        self.assertTrue(latest.exists())
        self.assertEqual(json.loads(latest.read_text(encoding="utf-8"))["result"], "second")
        self.assertEqual(json.loads(p1.read_text(encoding="utf-8"))["result"], "first")
        self.assertEqual(json.loads(p2.read_text(encoding="utf-8"))["result"], "second")

    def test_pause_turn_not_success(self) -> None:
        # pause_turn = 本轮未自然结束，不算正常完成 (runner 据此不当作干净一轮)。
        r = self._spawn({"is_error": False, "stop_reason": "pause_turn"})
        self.assertFalse(r["success"])
        self.assertEqual(r["claude_state"], "claude_pause_turn")

    def test_is_error_not_success(self) -> None:
        r = self._spawn({"is_error": True, "result": "boom"})
        self.assertFalse(r["success"])
        self.assertEqual(r["claude_state"], "claude_error")

    def test_api_error_not_success(self) -> None:
        r = self._spawn({"is_error": True, "api_error_status": 401})
        self.assertFalse(r["success"])
        self.assertEqual(r["claude_state"], "api_error_401")

    def test_empty_result_file_invalid(self) -> None:
        # 真实流程: spawn 用 open("w") 必先建文件，fake _run 不写内容 → 空文件 →
        # json 解析失败 → invalid_claude_result (而非 missing；missing 仅在 open 都未发生)。
        def _no_write(cmd, stdout=None, stderr=None, timeout=None, check=False, text=True):
            class _R:
                returncode = 0
            return _R()
        r = spawn_diagnose_agent(_diagnose_action(), self.task_dir, "add", 0, _run=_no_write)
        self.assertFalse(r["success"])
        self.assertEqual(r["claude_state"], "invalid_claude_result")

    def test_nonzero_empty_result_persists_stderr(self) -> None:
        def _fail(cmd, stdout=None, stderr=None, timeout=None, check=False, text=True):
            class _R:
                returncode = 1
                stderr = "--agent custom-agent not found"
            return _R()
        r = spawn_diagnose_agent(
            _diagnose_action(), self.task_dir, "add", 0, _run=_fail)
        self.assertFalse(r["success"])
        self.assertEqual(r["claude_return_code"], 1)
        self.assertIn("not found", r["error"])
        self.assertEqual(Path(r["stderr_path"]).read_text(), "--agent custom-agent not found")

    def test_spawn_exception_fatal(self) -> None:
        def _boom(*a, **k):
            raise OSError("claude not found")
        r = spawn_diagnose_agent(_diagnose_action(), self.task_dir, "add", 0, _run=_boom)
        self.assertFalse(r["success"])
        self.assertTrue(r["fatal"])
        self.assertEqual(r["claude_state"], "spawn_failed")

    def test_timeout(self) -> None:
        def _slow(cmd, stdout=None, stderr=None, timeout=None, check=False, text=True):
            raise subprocess.TimeoutExpired(cmd, timeout)
        r = spawn_diagnose_agent(_diagnose_action(), self.task_dir, "add", 0,
                                 timeout_sec=1, _run=_slow)
        self.assertFalse(r["success"])
        self.assertEqual(r["claude_state"], "timeout")

    def test_final_response_extracted(self) -> None:
        # 项 12b: 顶层 `result` 字段透传为 final_response (diagnosis_summary 主数据源)。
        r = self._spawn({"is_error": False, "stop_reason": "end_turn",
                         "result": "## 诊断总结\n根因: CAST_NONE 应改 CAST_ROUND"})
        self.assertEqual(r["final_response"], "## 诊断总结\n根因: CAST_NONE 应改 CAST_ROUND")

    def test_structured_attempt_metadata_extracted_before_truncation(self) -> None:
        response = "x" * 5000 + """
[ENGINE_ATTEMPT_METADATA]
fix_type: fix_reduce_accumulation
direction_verdict: switch
direction_reason: prior vector path regressed
probe_status: skipped
probe_reason: first round fast path
kb_used_ids: kb-0123456789ab, kb-abcdef012345
"""
        r = self._spawn({"is_error": False, "result": response})
        self.assertEqual(r["direction_verdict"], "switch")
        self.assertEqual(r["fix_type"], "fix_reduce_accumulation")
        self.assertEqual(
            r["attempt_metadata"]["kb_used_ids"],
            ["kb-0123456789ab", "kb-abcdef012345"],
        )

    def test_final_response_truncated(self) -> None:
        from engine.agent_backend import _FINAL_RESPONSE_MAXLEN
        r = self._spawn({"is_error": False, "result": "x" * (_FINAL_RESPONSE_MAXLEN + 50)})
        self.assertTrue(r["final_response"].endswith("…(截断)"))
        self.assertLessEqual(len(r["final_response"]), _FINAL_RESPONSE_MAXLEN + 10)

    def test_final_response_none_when_missing_or_blank(self) -> None:
        self.assertIsNone(self._spawn({"is_error": False})["final_response"])
        self.assertIsNone(self._spawn({"is_error": False, "result": "   "})["final_response"])

    def test_final_response_suppressed_on_error(self) -> None:
        # 真实产物核实 (artifacts 14/20 样例): is_error/stop_sequence 态 result 是
        # "API Error: 400 ..." 错误串，非诊断 → final_response 必须 None，不污染摘要。
        r = self._spawn({"is_error": True, "stop_reason": "stop_sequence",
                         "result": "API Error: 400 Invalid request: token limit"})
        self.assertIsNone(r["final_response"])
        # api_error 态同理。
        r2 = self._spawn({"is_error": True, "api_error_status": 400, "result": "API Error: 400"})
        self.assertIsNone(r2["final_response"])
        # pause_turn (本轮未自然结束) 也不取。
        r3 = self._spawn({"is_error": False, "stop_reason": "pause_turn", "result": "半截"})
        self.assertIsNone(r3["final_response"])

    def test_probe_policy_record_written_and_audited(self) -> None:
        response = """done
[ENGINE_ATTEMPT_METADATA]
fix_type: initial_static_fix
direction_verdict: initial
direction_reason: forensics evidence
probe_status: skipped
probe_reason: first round fast path
kb_used_ids: none
"""
        r = self._spawn({"is_error": False, "result": response})
        self.assertEqual(r["probe_policy"], "skip")
        self.assertEqual(r["probe_status"], "skipped")
        self.assertTrue(r["probe_policy_pass"])
        record = json.loads(Path(r["probe_record_path"]).read_text(encoding="utf-8"))
        self.assertEqual(record["reason"], "first_round_fast_path")
        self.assertEqual(record["observation_source"], "structured_metadata")

    def test_probe_policy_marks_partial_footer_incomplete(self) -> None:
        policy = {
            "schema_version": 1,
            "attempt": 0,
            "failure_type": "precision_failed",
            "policy": "skip",
            "reason": "first_round_fast_path",
            "primary_hint": "all_wrong",
            "instruction": "skip",
            "enforcement": "prompt_contract_with_posthoc_audit",
            "created_at": "2026-07-14T00:00:00+00:00",
        }
        metadata = {"probe_status": "skipped"}
        record = _finalize_probe_policy(
            self.task_dir,
            policy,
            "[ENGINE_ATTEMPT_METADATA]\nprobe_status: skipped",
            metadata,
        )
        self.assertTrue(record["policy_pass"])
        self.assertFalse(record["metadata_complete"])

    def test_probe_source_audit_overrides_false_skipped_self_report(self) -> None:
        kernel = self.task_dir / "kernel"
        kernel.mkdir()
        source = kernel / "op.cpp"
        source.write_text("void Run() {}\n", encoding="utf-8")
        response = """done
[ENGINE_ATTEMPT_METADATA]
fix_type: probe_cheat
direction_verdict: initial
direction_reason: claimed static evidence
probe_status: skipped
probe_reason: first round fast path
kb_used_ids: none
"""

        def _fake_run(cmd, stdout=None, stderr=None, timeout=None,
                      check=False, text=True):
            source.write_text(
                'void Run() { printf("probe=%f", 1.0f); }\n',
                encoding="utf-8",
            )
            stdout.write(json.dumps({"is_error": False, "result": response}))

            class _R:
                returncode = 0
                stderr = ""

            return _R()

        result = spawn_diagnose_agent(
            _diagnose_action("precision_failed", 0),
            self.task_dir,
            "op",
            0,
            _run=_fake_run,
        )
        self.assertFalse(result["probe_policy_pass"])
        self.assertTrue(result["ablation_violation"])
        audit = json.loads(
            Path(result["probe_source_audit_path"]).read_text(encoding="utf-8")
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(
            audit["new_probe_occurrences"][0]["probe"], "printf_call")


class TestCallbackIntoRunner(unittest.TestCase):
    """make_agent_callback 注入 runner，跑通 precision→CONTINUE→PASS 2 轮。

    claude 被 fake (不真跑)；gate 用自定义 dispatcher 喂预设 loop_signal。验证 backend
    的 callback 在 runner 真实调用链里被正确触发，且 session 抵达 success。
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_callback_invoked_session_to_success(self) -> None:
        from engine.runner import run_debug_session

        spawn_calls = []

        def _fake_run(cmd, stdout=None, stderr=None, timeout=None, check=False, text=True):
            spawn_calls.append(cmd[cmd.index("--agent") + 1])
            if stdout is not None:
                stdout.write(json.dumps({"is_error": False, "stop_reason": "end_turn"}))
            class _R:
                returncode = 0
            return _R()

        callback = make_agent_callback(
            agent_name="ascendc-debug-agent-constructive", _run=_fake_run)

        # 自定义 dispatcher: spawn_agent 走 backend callback；py_action 喂预设 gate。
        validate_seq = iter([
            {"gate": "GATE-V", "passed": False, "loop_signal": "CONTINUE"},
            {"gate": "GATE-V", "passed": True, "loop_signal": "PASS"},
        ])

        def _dispatcher(action, task_dir, op_name, agent_callback):
            if action.kind == "spawn_agent":
                return agent_callback(action, task_dir, op_name,
                                      (action.skill_args or {}).get("attempt", 0))
            if action.step == "validate":
                return dict(next(validate_seq))
            return {"success": True}

        status = run_debug_session(
            self.task_dir, op_name="add", agent="constructive",
            entry_failure_type="precision_failed",
            agent_callback=callback, dispatcher=_dispatcher)

        self.assertEqual(status["session_outcome"], "success")
        self.assertEqual(status["attempts_used"], 2)
        # backend callback 被触发 2 次 (每轮 diagnose_and_fix 一次)，都用 constructive。
        self.assertEqual(spawn_calls,
                         ["ascendc-debug-agent-constructive"] * 2)


class TestKbUsageTrace(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)
        (self.task_dir / "precision_tuning").mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_kb_log(self, entries: list) -> None:
        p = self.task_dir / "precision_tuning" / "knowledge_search_log.json"
        p.write_text(json.dumps(entries), encoding="utf-8")

    def _read_trace(self) -> list:
        p = self.task_dir / "precision_tuning" / "kb_usage_trace.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

    def _fake_run(self, final_response: str | None):
        body: dict = {"is_error": False, "stop_reason": "end_turn"}
        if final_response is not None:
            body["result"] = final_response
        return _fake_run_factory(body, {})

    def test_cited_titles_detected(self) -> None:
        self._write_kb_log([
            {"attempt": 0, "call_index": 0, "top_titles": ["TileGemm", "VecAdd"]},
        ])
        spawn_diagnose_agent(
            _diagnose_action("precision_failed", 0), self.task_dir, "op", 0,
            _run=self._fake_run("参考 TileGemm 的做法修复了循环边界"))
        trace = self._read_trace()
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["cited_titles"], ["TileGemm"])
        self.assertAlmostEqual(trace[0]["citation_rate"], 0.5)

    def test_no_kb_log_writes_empty_injected(self) -> None:
        spawn_diagnose_agent(
            _diagnose_action("precision_failed", 0), self.task_dir, "op", 0,
            _run=self._fake_run("some response"))
        trace = self._read_trace()
        self.assertEqual(trace[0]["injected_titles"], [])
        self.assertIsNone(trace[0]["citation_rate"])

    def test_none_final_response_no_crash(self) -> None:
        self._write_kb_log([
            {"attempt": 1, "call_index": 0, "top_titles": ["SomeKB"]},
        ])
        spawn_diagnose_agent(
            _diagnose_action("precision_failed", 1), self.task_dir, "op", 1,
            _run=self._fake_run(None))
        trace = self._read_trace()
        self.assertEqual(trace[0]["cited_titles"], [])

    def test_retrieved_injected_and_declared_used_ids_are_separate(self) -> None:
        self._write_kb_log([{
            "attempt": 0,
            "call_index": 0,
            "top_titles": ["TileGemm", "VecAdd"],
            "top_ids": ["kb-0123456789ab", "kb-abcdef012345"],
            "match_reasons": [
                {"knowledge_id": "kb-0123456789ab", "title": "TileGemm"},
                {"knowledge_id": "kb-abcdef012345", "title": "VecAdd"},
            ],
        }])
        response = """采用第一个知识修复。
[ENGINE_ATTEMPT_METADATA]
fix_type: tile_fix
direction_verdict: initial
direction_reason: matched tiling evidence
probe_status: skipped
probe_reason: first round fast path
kb_used_ids: kb-0123456789ab
"""
        spawn_diagnose_agent(
            _diagnose_action("precision_failed", 0), self.task_dir, "op", 0,
            _run=self._fake_run(response))
        trace = self._read_trace()[0]
        self.assertEqual(
            trace["retrieved_ids"], ["kb-0123456789ab", "kb-abcdef012345"])
        self.assertEqual(trace["injected_ids"], trace["retrieved_ids"])
        self.assertEqual(trace["declared_used_ids"], ["kb-0123456789ab"])
        self.assertTrue(trace["usage_trace_complete"])

    def test_no_kb_ablation_ignores_stale_search_log(self) -> None:
        self._write_kb_log([{
            "attempt": 0,
            "top_titles": ["StaleEntry"],
            "top_ids": ["kb-0123456789ab"],
        }])
        response = """No KB was injected.
[ENGINE_ATTEMPT_METADATA]
fix_type: local_fix
direction_verdict: initial
direction_reason: local source evidence
probe_status: skipped
probe_reason: first round fast path
kb_used_ids: kb-0123456789ab
"""
        with mock.patch.dict(os.environ, {"ABLATE_KB": "1"}):
            spawn_diagnose_agent(
                _diagnose_action("precision_failed", 0),
                self.task_dir, "op", 0, _run=self._fake_run(response))
        trace = self._read_trace()[0]
        self.assertTrue(trace["ablation_disabled"])
        self.assertEqual(trace["retrieved_ids"], [])
        self.assertEqual(trace["injected_ids"], [])
        self.assertEqual(trace["declared_used_ids"], [])
        self.assertEqual(
            trace["declared_unknown_ids"], ["kb-0123456789ab"])


class TestNoprobeInjection(unittest.TestCase):
    """ABLATE_PROBE=1 时 _build_prompt 末尾追加 noprobe 约束 (no_probe/baseline arm)。

    env 用 addCleanup 隔离防泄漏污染其他用例。
    """

    def _set_probe(self, value) -> None:
        prev = os.environ.get("ABLATE_PROBE")
        if value is None:
            os.environ.pop("ABLATE_PROBE", None)
        else:
            os.environ["ABLATE_PROBE"] = value

        def _restore() -> None:
            if prev is None:
                os.environ.pop("ABLATE_PROBE", None)
            else:
                os.environ["ABLATE_PROBE"] = prev
        self.addCleanup(_restore)

    def test_probe_ablated_injects_constraint(self) -> None:
        self._set_probe("1")
        with tempfile.TemporaryDirectory() as d:
            prompt = _build_prompt(Path(d), "FakeOp", "precision_failed", 0, "0")
        self.assertIn("ABLATE_PROBE", prompt)
        self.assertIn("禁用插桩", prompt)
        self.assertIn("[L5_PROBE]", prompt)

    def test_probe_default_uses_first_round_fast_path(self) -> None:
        self._set_probe(None)
        with tempfile.TemporaryDirectory() as d:
            prompt = _build_prompt(Path(d), "FakeOp", "precision_failed", 0, "0")
        self.assertNotIn("ABLATE_PROBE: 本次调用禁用插桩", prompt)
        self.assertIn("policy: skip", prompt)
        self.assertIn("first_round_fast_path", prompt)
        self.assertIn("[ENGINE_ATTEMPT_METADATA]", prompt)

    def test_nan_inf_first_round_requires_probe(self) -> None:
        self._set_probe(None)
        with tempfile.TemporaryDirectory() as d:
            task = Path(d)
            (task / "precision_tuning").mkdir()
            (task / "precision_tuning" / "forensics_report_0.json").write_text(
                json.dumps({"attempt": 0, "primary_hint": "nan_inf_contamination"}),
                encoding="utf-8",
            )
            prompt = _build_prompt(task, "FakeOp", "precision_failed", 0, "0")
        self.assertIn("policy: required", prompt)
        self.assertIn("nan_inf_contamination_exception", prompt)

    def test_no_forensics_ignores_stale_hint_and_adds_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            task = Path(d)
            (task / "precision_tuning").mkdir()
            (task / "precision_tuning" / "forensics_report_0.json").write_text(
                json.dumps({"primary_hint": "nan_inf_contamination"}),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"ABLATE_FORENSICS": "1"}):
                prompt = _build_prompt(
                    task, "FakeOp", "precision_failed", 0, "0")
        self.assertIn("ABLATE_FORENSICS", prompt)
        self.assertIn("first_round_fast_path", prompt)
        self.assertNotIn("nan_inf_contamination_exception", prompt)

    def test_later_attempt_receives_direction_history(self) -> None:
        self._set_probe(None)
        with tempfile.TemporaryDirectory() as d:
            task = Path(d)
            (task / "precision_tuning").mkdir()
            (task / "precision_tuning" / "tuning_directions.json").write_text(
                json.dumps({"entries": [{
                    "attempt": 0,
                    "fix_type": "tiling_only",
                    "direction_verdict": "initial",
                    "direction_reason": "suspected tail",
                    "outcome": "regressed",
                    "case_pass_rate": 20.0,
                }]}),
                encoding="utf-8",
            )
            prompt = _build_prompt(task, "FakeOp", "precision_failed", 1, "0")
        self.assertIn("近期修复方向与客观结果", prompt)
        self.assertIn("fix_type=tiling_only", prompt)
        self.assertIn("outcome=regressed", prompt)

    def test_recovery_ablation_suppresses_direction_history(self) -> None:
        previous = os.environ.get("ABLATE_RECOVERY")
        os.environ["ABLATE_RECOVERY"] = "1"
        self.addCleanup(
            lambda: (
                os.environ.pop("ABLATE_RECOVERY", None)
                if previous is None
                else os.environ.__setitem__("ABLATE_RECOVERY", previous)
            )
        )
        with tempfile.TemporaryDirectory() as d:
            task = Path(d)
            (task / "precision_tuning").mkdir()
            (task / "precision_tuning" / "tuning_directions.json").write_text(
                json.dumps({"entries": [{
                    "attempt": 0,
                    "fix_type": "old_direction",
                    "outcome": "regressed",
                }]}),
                encoding="utf-8",
            )
            prompt = _build_prompt(
                task, "FakeOp", "precision_failed", 1, "0")
        self.assertIn("ABLATE_RECOVERY", prompt)
        self.assertNotIn("old_direction", prompt)

    def test_kb_ablation_suppresses_stale_search_log(self) -> None:
        previous = os.environ.get("ABLATE_KB")
        os.environ["ABLATE_KB"] = "1"
        self.addCleanup(
            lambda: (
                os.environ.pop("ABLATE_KB", None)
                if previous is None
                else os.environ.__setitem__("ABLATE_KB", previous)
            )
        )
        with tempfile.TemporaryDirectory() as d:
            task = Path(d)
            (task / "precision_tuning").mkdir()
            (task / "precision_tuning" / "knowledge_search_log.json").write_text(
                json.dumps([{
                    "attempt": 0,
                    "top_titles": ["must_not_leak"],
                }]),
                encoding="utf-8",
            )
            prompt = _build_prompt(
                task, "FakeOp", "precision_failed", 0, "0")
        self.assertIn("ABLATE_KB", prompt)
        self.assertNotIn("must_not_leak", prompt)

    def test_detect_only_does_not_inject_prior_cheat_warning(self) -> None:
        previous = os.environ.get("ANTICHEAT_DETECT_ONLY")
        os.environ["ANTICHEAT_DETECT_ONLY"] = "1"
        self.addCleanup(
            lambda: (
                os.environ.pop("ANTICHEAT_DETECT_ONLY", None)
                if previous is None
                else os.environ.__setitem__("ANTICHEAT_DETECT_ONLY", previous)
            )
        )
        with tempfile.TemporaryDirectory() as d:
            task = Path(d)
            (task / "precision_tuning").mkdir()
            (task / "precision_tuning" / "cheat_history.json").write_text(
                json.dumps({"cheating_attempts": [{
                    "attempt": 0,
                    "severity": "violation",
                    "cheat_type": "must_not_inject",
                    "instruction": "must_not_inject",
                }]}),
                encoding="utf-8",
            )
            prompt = _build_prompt(
                task, "FakeOp", "precision_failed", 1, "0")
        self.assertNotIn("must_not_inject", prompt)

    def test_gate_a_audit_context_is_injected_into_agent_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            task = Path(d)
            tuning = task / "precision_tuning"
            tuning.mkdir()
            (tuning / "audit_context_attempt_0.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "attempt": 0,
                    "generated_by": "engine_pre_agent_audit",
                    "generated_at": "2026-07-25T00:00:00Z",
                    "source_type": "forensics_report",
                    "source_path": "precision_tuning/forensics_report_0.json",
                    "source_parseable": True,
                    "primary_hint": "tail writes are unmasked",
                    "direction_verdict": "initial",
                }),
                encoding="utf-8",
            )

            prompt = _build_prompt(
                task, "FakeOp", "precision_failed", 0, "0")

        self.assertIn("[ENGINE_GATE_A_AUDIT]", prompt)
        self.assertIn("tail writes are unmasked", prompt)
        self.assertIn("engine_pre_agent_audit", prompt)

    def test_gate_a_ablation_suppresses_stale_audit_context(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            task = Path(d)
            tuning = task / "precision_tuning"
            tuning.mkdir()
            (tuning / "audit_context_attempt_0.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "attempt": 0,
                    "generated_by": "engine_pre_agent_audit",
                    "generated_at": "2026-07-25T00:00:00Z",
                    "source_type": "forensics_report",
                    "source_path": "precision_tuning/forensics_report_0.json",
                    "source_parseable": True,
                    "primary_hint": "must_not_leak",
                    "direction_verdict": "initial",
                }),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"ABLATE_GATE_A": "1"}):
                prompt = _build_prompt(
                    task, "FakeOp", "precision_failed", 0, "0")

        self.assertNotIn("[ENGINE_GATE_A_AUDIT]", prompt)
        self.assertNotIn("must_not_leak", prompt)


if __name__ == "__main__":
    unittest.main()
