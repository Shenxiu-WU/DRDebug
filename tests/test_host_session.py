"""Native host protocol tests: real frozen state machine, fake expensive gates.

No user Python, compiler, NPU, provider or nested agent CLI is executed. The
dispatcher fixture explicitly marks its objective results as synthetic.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "skills/drdebug/scripts/host_session.py"
spec = importlib.util.spec_from_file_location("public_host_session_tests", SCRIPT)
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)
FROZEN = REPO / "papers/2026-dont-regenerate-debug/artifact/code/debug_agent_v5"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HostSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="drdebug-host-cpu-")
        self.base = Path(self.temp.name).resolve()
        self.workspace = self.base / "workspace"
        self.task = self.workspace / "task"
        self.runtime = self.workspace / "runtime"
        engine = self.runtime / host.DEBUG_REL / "engine"
        shutil.copytree(FROZEN / host.DEBUG_REL / "engine", engine,
                        ignore=shutil.ignore_patterns("__pycache__", "tests"))
        self.method = self.runtime / "host-instructions.md"
        self.method.write_text("Prepared host method. Keep Phase A → B → C and protected references.\n")
        (self.task / "kernel").mkdir(parents=True)
        (self.task / "kernel/k.cpp").write_text("// original kernel\n")
        (self.task / "kernel/pybind11.cpp").write_text("PYBIND11_MODULE(example, m) {}\n")
        (self.task / "kernel/support.py").write_text("raise RuntimeError('never import')\n")
        (self.task / "model.py").write_text("raise RuntimeError('MUST NOT EXECUTE')\n")
        (self.task / "model_new_ascendc.py").write_text("raise RuntimeError('MUST NOT EXECUTE')\n")
        row = json.dumps({"inputs": [{"shape": [1]}]}) + "\n"
        (self.task / "Add.json").write_text(row)
        (self.task / "Add.json.bak").write_text(row + row)
        self.manifest = self.workspace / host.MANIFEST
        self.manifest.write_text(json.dumps({"schema_version": 3, "execution": {"determinism": "on"}, "task_contract": {
            "compact_file": "Add.json", "full_file": "Add.json.bak",
            "protected_sha256": {name: digest(self.task / name) for name in
                                 ("model.py", "model_new_ascendc.py", "Add.json", "Add.json.bak")}}}))
        self.prior = {name: module for name, module in sys.modules.items()
                      if name == "engine" or name.startswith("engine.")}
        for name in self.prior:
            del sys.modules[name]
        self.paths = list(sys.path)
        self.env = mock.patch.dict(os.environ, {}, clear=True)
        self.env.start()
        rt_spec = importlib.util.spec_from_file_location("host_fixture_runtime", REPO / "skills/drdebug/scripts/runtime_checks.py")
        self.runtime_api = importlib.util.module_from_spec(rt_spec)
        rt_spec.loader.exec_module(self.runtime_api)
        startup, _ = self.runtime_api.configure_runtime_environment({}, self.workspace / "adapter", "on")
        os.environ.update(startup)
        os.environ.update(ASCEND_RT_VISIBLE_DEVICES="4", ASCEND_DEVICE_ID="0", ASCENDC_SOC_VERSION="ActualFixtureSOC")
        self.now = 1000.0
        self.calls = []
        self.signal = "PASS"
        self.options = {"op_name": "Add", "runtime": self.runtime,
                        "budget": {"max_attempts": 5, "deadline_sec": 86400},
                        "kb_path": None, "resource_context": "Public source-only CPU fixture",
                        "_dispatcher": self.dispatch, "_clock": lambda: self.now}

    def tearDown(self):
        sys.path[:] = self.paths
        for name in list(sys.modules):
            if name == "engine" or name.startswith("engine."):
                del sys.modules[name]
        sys.modules.update(self.prior)
        self.env.stop()
        self.temp.cleanup()

    def dispatch(self, action, task, op_name, callback):
        self.assertEqual(action.kind, "py_action")
        self.assertIsNone(callback)
        self.calls.append(action.step)
        if action.step == "baseline_checkpoint":
            return {"success": True, "failure_type": "precision_failed",
                    "baseline_common_gate": {"passed": True}, "synthetic_cpu_fixture": True}
        if action.step == "checkpoint_and_rollback":
            return {"success": True, "updated": True, "synthetic_cpu_fixture": True}
        if action.step == "validate":
            return {"gate": "GATE-V", "passed": self.signal == "PASS", "loop_signal": self.signal,
                    "failure_type": "success" if self.signal == "PASS" else "precision_failed",
                    "checks": {}, "synthetic_cpu_fixture": True}
        return {"passed": True, "success": True, "synthetic_cpu_fixture": True}

    def advance(self, **overrides):
        return host.advance(self.workspace, **{**self.options, **overrides})

    def submit(self, request, result=None):
        return host.submit(self.workspace, request_id=request["request_id"],
                           result=result or self.result(), **self.options)

    def result(self, **changes):
        value = {"outcome": "completed", "final_response": (
            "Inspected public evidence; this is a synthetic CPU test patch.\n"
            "[ENGINE_ATTEMPT_METADATA]\nfix_type: accumulation\n"
            "direction_verdict: initial\ndirection_reason: observed source evidence\n"
            "probe_status: skipped\nprobe_reason: first_round_fast_path\nkb_used_ids: none\n")}
        return {**value, **changes}

    def events(self):
        path = self.task / ".debug_events/events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def journal(self):
        return host._read(self.workspace / "host-session/state.json")

    def test_pending_is_nonterminal_repeatable_and_preserves_active_source(self):
        before = digest(self.task / "kernel/k.cpp")
        request = self.advance()
        self.assertEqual(request["state"], "awaiting_host")
        self.assertFalse(request["terminal"])
        instructions = Path(request["instructions"]).read_text()
        self.assertIn("Prepared host method", instructions)
        self.assertIn("ENGINE_ATTEMPT_METADATA", instructions)
        audit_line = next(line for line in instructions.splitlines() if line.startswith("Allowed audit files"))
        self.assertIn("relative to candidate", audit_line)
        ownership = instructions.split("# Final native-host path and ownership override\n", 1)[1]
        candidate = Path(request["candidate"])
        for name in ("precision_audit_0.md", "candidate_kb_entry.json"):
            self.assertIn("precision_tuning/" + name, audit_line)
            self.assertIn(str(candidate / "precision_tuning" / name), ownership)
            self.assertNotIn(str(candidate / name), instructions)
        self.assertNotIn("Allowed audit files: precision_audit_0.md", instructions)
        events = self.events()
        self.assertEqual(events[-1]["action"]["step"], "diagnose_and_fix")
        self.assertEqual(events[-1]["type"], "action_started")
        self.now += 10
        second = self.advance()
        self.assertEqual(second["request_id"], request["request_id"])
        self.assertEqual(self.events(), events)
        self.assertEqual(digest(self.task / "kernel/k.cpp"), before)
        self.assertIsNone(second["budget"]["agent_turns"])
        self.assertEqual(second["budget"]["consumed_sec"], 10)

    def test_completed_patch_uses_real_state_machine_then_validate_checkpoint_terminal(self):
        request = self.advance()
        (Path(request["candidate"]) / "kernel/k.cpp").write_text("// repaired candidate\n")
        audit = Path(request["candidate"]) / "precision_tuning/precision_audit_0.md"
        audit.write_text("[REFERENCE_IMPL_SPEC]\n[COMPUTATION_DECOMPOSITION]\n")
        with ExitStack() as stack:
            backend = sys.modules["engine.agent_backend"]
            stack.enter_context(mock.patch.object(backend, "spawn_diagnose_agent", side_effect=AssertionError("No nested model")))
            stack.enter_context(mock.patch.object(sys.modules["engine.runner"], "run_debug_session", side_effect=AssertionError("No original dangling reconciliation")))
            result = self.submit(request)
        self.assertTrue(result["terminal"])
        self.assertEqual(result["engine_status"]["session_outcome"], "success")
        self.assertEqual((self.task / "kernel/k.cpp").read_text(), "// repaired candidate\n")
        self.assertEqual((self.task / "precision_tuning/precision_audit_0.md").read_bytes(), audit.read_bytes())
        self.assertLess(self.calls.index("validate"), self.calls.index("checkpoint_and_rollback"))
        done = [e for e in self.events() if e.get("type") == "action_completed" and e["action"]["step"] == "diagnose_and_fix"]
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["result"]["changed_files"], ["kernel/k.cpp"])
        self.assertIsNone(done[0]["result"]["agent_turns"])
        self.assertTrue(done[0]["result"]["probe_policy_pass"])
        self.assertTrue((Path(request["request"]).parent / "previous-kernel/k.cpp").is_file())

    def test_diagnostic_sandbox_is_writable_and_never_imported(self):
        request = self.advance()
        diagnostic = request["diagnostics"]
        task = Path(diagnostic["task"])
        self.assertEqual((task / "model.json").read_bytes(), (self.task / "Add.json").read_bytes())
        for name, expected in diagnostic["protected_sha256"].items():
            self.assertEqual(digest(task / name), expected)
        (task / "kernel/build").mkdir()
        (task / "kernel/build/diagnostic.so").write_bytes(b"not a real native library")
        (task / "kernel/k.cpp").write_text("// temporary diagnostic instrumentation\n")
        (Path(diagnostic["directory"]) / "probe.log").write_text("synthetic fixture observations\n")
        self.submit(request)
        self.assertEqual((self.task / "kernel/k.cpp").read_text(), "// original kernel\n")
        self.assertFalse((self.task / "kernel/build/diagnostic.so").exists())
        self.assertFalse((self.task / "probe.log").exists())
        self.assertFalse(diagnostic["imported_on_submit"])

    def test_diagnostic_recipe_retains_startup_device_and_excludes_provider_secrets(self):
        os.environ["ANTHROPIC_API_KEY"] = "TEST_SECRET_DO_NOT_RECORD"
        os.environ["PRIVATE_UNRELATED_TOKEN"] = "OTHER_TEST_SECRET"
        request = self.advance()
        record = request["diagnostics"]
        path = Path(record["execution"])
        execution = json.loads(path.read_text())
        selected = execution["environment"]
        self.assertEqual(digest(path), record["execution_sha256"])
        self.assertTrue(set(selected) <= host.DIAGNOSTIC_ENVIRONMENT)
        self.assertEqual(selected["ASCEND_RT_VISIBLE_DEVICES"], "4")
        self.assertEqual(selected["ASCEND_DEVICE_ID"], "0")
        self.assertEqual(selected["DRDEBUG_DETERMINISM"], "on")
        self.assertEqual(selected["DRDEBUG_RUNTIME_STARTUP_CONFIG_SHA256"],
                         os.environ["DRDEBUG_RUNTIME_STARTUP_CONFIG_SHA256"])
        self.assertEqual(selected["DRDEBUG_RUNTIME_RECEIPTS"], str(Path(record["directory"]) / "startup-receipts"))
        self.assertIn(str(self.workspace / "adapter"), selected["PYTHONPATH"].split(os.pathsep))
        self.assertNotIn("TEST_SECRET_DO_NOT_RECORD", path.read_text())
        self.assertNotIn("OTHER_TEST_SECRET", path.read_text())
        self.assertEqual(execution["commands"]["clean_build"], [sys.executable, "-B",
            str(self.runtime / "utils/build_ascendc.py"), record["task"], "--soc-version", "ActualFixtureSOC", "--clean"])
        self.assertNotIn("unless separately authorized", Path(request["instructions"]).read_text())

    def test_duplicate_submit_and_terminal_next_never_repeat_worker_or_gates(self):
        request = self.advance()
        first = self.submit(request)
        events, calls = self.events(), list(self.calls)
        duplicate = self.submit(request)
        self.assertTrue(duplicate["already_submitted"])
        self.assertEqual(duplicate["engine_status"], first["engine_status"])
        self.advance()
        self.assertEqual(self.events(), events)
        self.assertEqual(self.calls, calls)
        with self.assertRaisesRegex(host.HostSessionError, "different submitted result"):
            self.submit(request, self.result(final_response="Different claimed diagnosis"))

    def test_gate_continue_creates_a_new_request_only_after_checkpoint(self):
        self.signal = "CONTINUE"
        request = self.advance()
        next_request = self.submit(request)
        self.assertFalse(next_request["terminal"])
        self.assertNotEqual(next_request["request_id"], request["request_id"])
        self.assertEqual(next_request["attempt"], 1)
        self.assertIn("checkpoint_and_rollback", self.calls)

    def test_gate_stop_is_terminal_failure_after_checkpoint_not_a_host_success(self):
        self.signal = "STOP"
        request = self.advance()
        result = self.submit(request)
        self.assertTrue(result["terminal"])
        self.assertEqual(result["returncode"], 1)
        self.assertEqual(result["engine_status"]["session_outcome"], "failed")
        self.assertIn("checkpoint_and_rollback", self.calls)

    def test_clean_baseline_needs_no_host_request_or_agent_invocation(self):
        def baseline(action, *args):
            self.assertEqual(action.step, "baseline_checkpoint")
            return {"success": True, "failure_type": "success", "baseline_common_gate": {"passed": True},
                    "synthetic_cpu_fixture": True}
        result = self.advance(_dispatcher=baseline)
        self.assertTrue(result["terminal"])
        self.assertEqual(result["returncode"], 0)
        self.assertFalse(any(e.get("action", {}).get("kind") == "spawn_agent" for e in self.events()))

    def test_candidate_cannot_edit_reference_or_non_native_source(self):
        for name in ("model.py", "kernel/support.py", "kernel/new.py", "precision_tuning/validation_result_attempt_0.json"):
            with self.subTest(name=name):
                request = self.advance()
                path = Path(request["candidate"]) / name
                original = path.read_bytes() if path.exists() else None
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("unauthorized")
                with self.assertRaises(host.HostSessionError):
                    self.submit(request)
                self.assertEqual((self.task / "kernel/k.cpp").read_text(), "// original kernel\n")
                if original is None:
                    path.unlink()
                else:
                    path.write_bytes(original)

    def test_symlink_and_hardlink_candidates_fail_before_import(self):
        request = self.advance()
        source = Path(request["candidate"]) / "kernel/k.cpp"
        source.unlink()
        source.symlink_to(self.task / "kernel/k.cpp")
        with self.assertRaisesRegex(host.HostSessionError, "Linked"):
            self.submit(request)
        source.unlink()
        os.link(self.task / "kernel/k.cpp", source)
        with self.assertRaisesRegex(host.HostSessionError, "Hard-linked"):
            self.submit(request)

    def test_parent_source_protected_evidence_and_event_tamper_fail_closed(self):
        request = self.advance()
        for path in (self.task / "kernel/k.cpp", self.task / "model.py", self.task / "Add.json"):
            before = path.read_bytes()
            path.write_bytes(before + b"tamper")
            with self.assertRaisesRegex(host.HostSessionError, "active task/evidence changed"):
                self.submit(request)
            path.write_bytes(before)
        stream = self.task / ".debug_events/events.jsonl"
        before = stream.read_bytes()
        stream.write_bytes(before + b'{"type":"session_done"}\n')
        with self.assertRaises(host.HostSessionError):
            self.submit(request)
        with self.assertRaises(host.HostSessionError):
            self.advance()
        stream.write_bytes(before)
        stream.write_bytes(before[:-1])
        with self.assertRaisesRegex(host.HostSessionError, "Truncated"):
            self.advance()

    def test_packet_and_manifest_tamper_or_wrong_request_fail_closed(self):
        request = self.advance()
        path = Path(request["request"])
        before = path.read_bytes()
        path.write_bytes(before + b" ")
        with self.assertRaisesRegex(host.HostSessionError, "request record changed"):
            self.advance()
        path.write_bytes(before)
        with self.assertRaisesRegex(host.HostSessionError, "current pending"):
            self.submit({"request_id": "host_wrong"})
        self.manifest.write_text("{}")
        with self.assertRaisesRegex(host.HostSessionError, "configuration or prepared manifest changed"):
            self.advance()

    def test_failed_host_result_retains_patch_without_import_and_records_truthful_failure(self):
        request = self.advance()
        (Path(request["candidate"]) / "kernel/k.cpp").write_text("// unfinished candidate\n")
        next_request = self.submit(request, self.result(outcome="failed", error="host permission denied"))
        self.assertEqual((self.task / "kernel/k.cpp").read_text(), "// original kernel\n")
        self.assertEqual(next_request["attempt"], 0)
        records = self.journal()["completed_requests"]
        self.assertFalse(records[request["request_id"]]["engine_result"]["success"])
        self.assertEqual(records[request["request_id"]]["engine_result"]["changed_files"], [])

    def test_unknown_or_fabricated_result_fields_are_rejected(self):
        request = self.advance()
        for result in (self.result(success=True), self.result(agent_turns=0), self.result(outcome="success"),
                       self.result(final_response=""), self.result(outcome="failed")):
            with self.subTest(result=result):
                with self.assertRaises(host.HostSessionError):
                    self.submit(request, result)

    def test_interrupted_deterministic_action_is_not_silently_replayed(self):
        calls = []
        def interrupted(action, *args):
            calls.append(action.step)
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.advance(_dispatcher=interrupted)
        result = self.advance()
        self.assertEqual(result["state"], "needs_attention")
        self.assertEqual(result["reason"], "interrupted_deterministic_action")
        self.assertEqual(calls, ["baseline_checkpoint"])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.events()[-1]["type"], "action_started")

    def test_durable_result_before_event_commit_recovers_without_redispatch(self):
        original = host._Session.complete
        def interrupted(session, op):
            if op["action"]["step"] == "baseline_checkpoint":
                raise KeyboardInterrupt
            return original(session, op)
        with mock.patch.object(host._Session, "complete", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.advance()
        request = self.advance()
        self.assertEqual(request["state"], "awaiting_host")
        self.assertEqual(self.calls.count("baseline_checkpoint"), 1)
        completed = [e for e in self.events() if e["type"] == "action_completed" and e["action"]["step"] == "baseline_checkpoint"]
        self.assertEqual(len(completed), 1)

    def test_interrupted_atomic_source_import_is_completed_before_any_validation(self):
        request = self.advance()
        (Path(request["candidate"]) / "kernel/k.cpp").write_text("// repaired import\n")
        original = host._bytes
        fired = []
        def interrupted(path, data):
            original(path, data)
            if path == self.task / "kernel/k.cpp" and not fired:
                fired.append(True)
                raise KeyboardInterrupt
        with mock.patch.object(host, "_bytes", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.submit(request)
        self.assertTrue((self.task / "kernel/pybind11.cpp").is_file())
        self.assertNotIn("validate", self.calls)
        result = self.submit(request)
        self.assertTrue(result["terminal"])
        self.assertEqual(self.calls.count("validate"), 1)
        self.assertEqual((self.task / "kernel/k.cpp").read_text(), "// repaired import\n")

    def test_host_completion_commit_interruption_reuses_fixed_result(self):
        request = self.advance()
        original = host._Session.complete
        def interrupted(session, op):
            if op["kind"] == "host":
                raise KeyboardInterrupt
            return original(session, op)
        with mock.patch.object(host._Session, "complete", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.submit(request)
        result = self.submit(request)
        self.assertTrue(result["terminal"])
        matches = [e for e in self.events() if e["type"] == "action_completed" and e["action"]["step"] == "diagnose_and_fix"]
        self.assertEqual(len(matches), 1)

    def test_immutable_submitted_result_tamper_is_not_replaced_on_recovery(self):
        request = self.advance()
        original = host._Session.complete
        def interrupted(session, op):
            if op["kind"] == "host":
                raise KeyboardInterrupt
            return original(session, op)
        with mock.patch.object(host._Session, "complete", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.submit(request)
        path = Path(request["request"]).parent / "submitted-result.json"
        changed = self.result(final_response="tampered statement")
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(host.HostSessionError, "Immutable submitted host result changed"):
            self.submit(request)
        self.assertEqual(json.loads(path.read_text()), changed)
        self.assertNotIn("validate", self.calls)

    def test_budget_charges_host_and_submit_work_exactly_once(self):
        request = self.advance()
        self.now += 100
        original = host._Session.snapshot_candidate
        def slow_snapshot(session, *args):
            self.now += 5
            return original(session, *args)
        original_dispatch = self.dispatch
        def slow_dispatch(action, *args):
            self.now += 3
            return original_dispatch(action, *args)
        self.options["_dispatcher"] = slow_dispatch
        with mock.patch.object(host._Session, "snapshot_candidate", slow_snapshot):
            self.submit(request)
        # 100 host seconds + 5 snapshot seconds + 3 each validate/checkpoint.
        self.assertEqual(self.journal()["consumed_sec"], 111)

    def test_cumulative_external_time_is_charged_once_without_pending_overlap(self):
        request = self.advance(external_elapsed_sec=3)
        self.assertEqual(request["budget"]["consumed_sec"], 3)
        events = self.events()
        repeated = self.advance(external_elapsed_sec=3)
        self.assertEqual(repeated["budget"]["consumed_sec"], 3)
        self.now += 10
        # Two of those ten pending seconds were another launcher preflight.
        updated = self.advance(external_elapsed_sec=5)
        self.assertEqual(updated["budget"]["consumed_sec"], 13)
        self.assertEqual(self.journal()["accounted_external_sec"], 5)
        self.assertEqual(self.journal()["external_overlap_sec"], 2)
        self.assertEqual(self.advance(external_elapsed_sec=5)["budget"]["consumed_sec"], 13)
        self.assertEqual(self.events(), events)

    def test_external_time_must_be_monotonic_finite_and_obeys_deadline(self):
        self.advance(external_elapsed_sec=5)
        for value in (4, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaises(host.HostSessionError):
                    self.advance(external_elapsed_sec=value)
        result = self.advance(external_elapsed_sec=86401)
        self.assertTrue(result["terminal"])
        self.assertEqual(result["returncode"], 4)
        self.assertEqual(result["engine_status"]["session_outcome"], "timeout")
        self.assertNotIn("validate", self.calls)

    def test_pending_host_wall_time_exhaustion_does_not_import_or_claim_success(self):
        request = self.advance()
        (Path(request["candidate"]) / "kernel/k.cpp").write_text("// unsubmitted\n")
        self.now += 86401
        result = self.submit(request)
        self.assertTrue(result["terminal"])
        self.assertEqual(result["returncode"], 4)
        self.assertEqual(result["engine_status"]["session_outcome"], "timeout")
        self.assertEqual((self.task / "kernel/k.cpp").read_text(), "// original kernel\n")
        self.assertNotIn("validate", self.calls)

    def test_resume_budget_and_context_are_fixed_and_existing_sessions_not_adopted(self):
        self.advance()
        with self.assertRaisesRegex(host.HostSessionError, "configuration"):
            self.advance(budget={"max_attempts": 6, "deadline_sec": 86400})
        with self.assertRaisesRegex(host.HostSessionError, "configuration"):
            self.advance(resource_context="different method")
        (self.workspace / "host-session/state.json").unlink()
        with self.assertRaisesRegex(host.HostSessionError, "Cannot adopt"):
            self.advance()


if __name__ == "__main__":
    unittest.main()
