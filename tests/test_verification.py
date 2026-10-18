"""CPU regression tests of independent replay control flow, never precision claims.

Process mocks test orchestration, byte preservation and failure recording only.
They do not compile a kernel, execute an NPU or certify numerical correctness.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "skills/drdebug/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import drdebug as api
import runtime_checks
import verification


def passing_comparisons(count):
    return {"ok": True, "device": "npu:0",
            "inputs": [f"inputs[{i}][0]: Tensor(shape=(1,), dtype=torch.float32, device=npu:0)" for i in range(count)],
            "comparisons": [f"case[{i}]: output[{i}]: matched" for i in range(count)]}


class ComparisonTests(unittest.TestCase):
    def test_complete_ordered_matched_npu_rows_are_required(self):
        self.assertTrue(verification.checked_comparisons(passing_comparisons(2), 2))

    def test_missing_duplicate_or_out_of_order_case_rows_are_rejected(self):
        mutations = (
            lambda value: value.pop("comparisons"),
            lambda value: value["comparisons"].pop(),
            lambda value: value["comparisons"].__setitem__(1, value["comparisons"][0]),
            lambda value: value["comparisons"].reverse(),
        )
        for change in mutations:
            report = passing_comparisons(2)
            change(report)
            self.assertFalse(verification.checked_comparisons(report, 2), report)

    def test_cpu_report_or_cpu_tensor_input_cannot_be_counted(self):
        for field in ("device", "inputs"):
            report = passing_comparisons(2)
            report[field] = "cpu" if field == "device" else ["Tensor(shape=(1,), dtype=torch.float32, device=cpu)"]
            self.assertFalse(verification.checked_comparisons(report, 2))

    def test_mismatch_false_status_and_no_actual_tensor_inputs_are_rejected(self):
        for field, value in (("ok", False), ("comparisons", ["case[0]: output[0]: numerical mismatch"]), ("inputs", [])):
            report = passing_comparisons(1)
            report[field] = value
            self.assertFalse(verification.checked_comparisons(report, 1))


class IndependentVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="drdebug-verification-test-")
        self.root = Path(self.temp.name).resolve()
        self.input = self.root / "input"
        (self.input / "kernel").mkdir(parents=True)
        (self.input / "model.py").write_text("raise RuntimeError('USER REFERENCE MUST NOT BE IMPORTED BY CPU TEST')\n")
        (self.input / "model_new_ascendc.py").write_text("raise RuntimeError('USER WRAPPER MUST NOT BE IMPORTED BY CPU TEST')\n")
        (self.input / "kernel/pybind11.cpp").write_text("PYBIND11_MODULE(fixture_native, m) {}\n")
        (self.input / "kernel/fixture.cpp").write_text("// no real kernel is compiled by these tests\n")
        row = {"inputs": [{"name": "x", "shape": [1], "dtype": "float32", "type": "tensor"}]}
        compact = json.dumps(row).encode() + b"\n"
        (self.input / "Fixture.json").write_bytes(compact)
        (self.input / "Fixture.json.bak").write_bytes(compact + compact)
        self.workspace = self.root / "workspace"
        self.frozen_before = api.tree_hashes(api.source_root())
        api.prepare(argparse.Namespace(task=str(self.input), output=str(self.workspace), op_name="Fixture",
                                       reference_root=None, determinism="on"))
        self.args = argparse.Namespace(workspace=str(self.workspace), npu="0", soc="FixtureSOC", determinism="on",
                                       stage_timeout_sec=60, overall_timeout_sec=300, casewise=False)
        self.spawned = []
        self.case_observations = []
        self.preflight = mock.patch.object(api, "environment_report", return_value={"ready": True})
        self.preflight_mock = self.preflight.start()

    def tearDown(self):
        self.preflight.stop()
        self.assertEqual(self.frozen_before, api.tree_hashes(api.source_root()))
        self.temp.cleanup()

    def popen(self, command, **kwargs):
        child = argparse.Namespace(command=command, kwargs=kwargs)
        self.spawned.append(child)
        return child

    def wait(self, child, timeout):
        command = child.command
        if command[2].endswith("/utils/build_ascendc.py"):
            task = Path(command[3])
            (task / "kernel/build").mkdir()
            (task / "kernel/build/fixture_native.fixture.so").write_bytes(b"MOCK PROCESS CONTROL FLOW ONLY; NOT A REAL NATIVE LIBRARY")
        elif "--_replay" in command:
            path = Path(command[-2])
            suite = command[-1]
            config = api.read_json(path)
            task = Path(config["task"])
            self.case_observations.append((suite, (task / "Fixture.json").read_bytes()))
            count = 1 if suite.startswith("case-") else config["contract"][suite + "_cases"]
            guard = config["nonfinite_guard"]
            receipt = {"schema_version": 1, "policy": guard["policy"], "helper_sha256": guard["helper_sha256"],
                       "entrypoint": "import", "expected_tensor_leaves": count, "checked_tensor_leaves": count,
                       "unpaired_tensor_leaves": 0, "structural_rejections": 0,
                       "nonfinite_mismatches": 0, "guard_errors": 0, "coverage_complete": True, "passed": True}
            # A synthetic child result exercises parent aggregation only.
            api.write_json(path.parent / (suite + ".json"), {"passed": True, "mock_control_flow_only": True,
                           "native": {"sha256": config["native_sha256"]}, "public_nonfinite_guard": receipt})
        return 0

    def execute(self, wait=None):
        with mock.patch.object(verification.subprocess, "Popen", side_effect=self.popen), \
             mock.patch.object(api, "_wait_engine", side_effect=wait or self.wait):
            return verification.verify(self.args, api)

    def test_mock_pipeline_uses_separate_processes_and_preserves_original_inputs(self):
        before = api.tree_hashes(self.workspace / "task")
        self.args.casewise = True
        summary = self.execute()
        self.assertTrue(summary["passed"], summary)
        self.assertTrue(summary["nonfinite_guard"]["required"])
        self.assertEqual(set(summary["nonfinite_guard"]["suites"]), {"compact", "full", "case-00", "case-01"})
        self.assertEqual(summary["returncode"], 0)
        self.assertEqual(len(self.spawned), 7)  # clean build, compact/full, 2 cases, C++ integrity, AST
        self.assertEqual([row[0] for row in self.case_observations], ["compact", "full", "case-00", "case-01"])
        compact = (self.input / "Fixture.json").read_bytes()
        full = (self.input / "Fixture.json.bak").read_bytes()
        self.assertEqual([row[1] for row in self.case_observations], [compact, full, full, full])
        self.assertEqual(api.tree_hashes(self.workspace / "task"), before)
        self.assertFalse((self.workspace / ".drdebug.lock").exists())
        self.assertIs(self.preflight_mock.call_args.kwargs["require_worker"], False)
        self.assertTrue(self.preflight_mock.call_args.kwargs["runtime_probe"])
        for child in self.spawned:
            self.assertNotIn("claude", child.command)
            self.assertEqual(child.kwargs["env"]["DRDEBUG_DETERMINISM"], "on")
            self.assertEqual(child.kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0], str(self.workspace / "adapter"))

    def test_existing_writer_lock_is_preserved_and_no_child_is_started(self):
        lock = self.workspace / ".drdebug.lock"
        raw = b'{"pid":12345,"purpose":"existing_writer"}\n'
        lock.write_bytes(raw)
        with mock.patch.object(verification.subprocess, "Popen", side_effect=AssertionError("must not start any replay child")):
            with self.assertRaises(api.ContractError):
                verification.verify(self.args, api)
        self.assertEqual(lock.read_bytes(), raw)
        self.preflight_mock.assert_not_called()

    def test_missing_changed_or_incomplete_guarded_child_receipt_is_not_a_pass(self):
        for mode in ("missing", "helper", "entrypoint", "coverage", "errors"):
            with self.subTest(mode=mode):
                def changed(child, timeout):
                    code = self.wait(child, timeout)
                    if "--_replay" in child.command and child.command[-1] == "full":
                        path = Path(child.command[-2]).parent / "full.json"
                        value = api.read_json(path)
                        if mode == "missing": value.pop("public_nonfinite_guard")
                        else:
                            key, replacement = {"helper": ("helper_sha256", "0" * 64),
                                "entrypoint": ("entrypoint", "cli"), "coverage": ("checked_tensor_leaves", 0),
                                "errors": ("guard_errors", 1)}[mode]
                            value["public_nonfinite_guard"][key] = replacement
                        api.write_json(path, value)
                    return code
                summary = self.execute(changed)
                self.assertFalse(summary["passed"])
                self.assertEqual(summary["returncode"], 2)
                self.assertIn("Nonfinite guard", summary["error"])

    def test_mode_conflict_is_rejected_before_preflight_or_any_process(self):
        self.args.determinism = "off"
        with self.assertRaises(api.ContractError):
            verification.verify(self.args, api)
        self.preflight_mock.assert_not_called()
        self.assertFalse((self.workspace / "verification").exists())

    def test_protected_input_drift_is_rejected_without_import_or_process(self):
        (self.workspace / "task/model.py").write_text("# changed protected reference\n")
        with self.assertRaises(api.ContractError):
            verification.verify(self.args, api)
        self.preflight_mock.assert_not_called()

    def test_source_drift_during_independent_copy_is_a_recorded_failure(self):
        original_copy = api.copy_tree
        def drift(source, target, **kwargs):
            original_copy(source, target, **kwargs)
            (Path(target) / "fixture.cpp").write_text("// simulated concurrent source drift\n")
        with mock.patch.object(api, "copy_tree", side_effect=drift):
            summary = self.execute()
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["returncode"], 2)
        self.assertIn("Source changed", summary["error"])
        self.assertFalse(self.spawned)
        saved = next((self.workspace / "verification").glob("*/summary.json"))
        self.assertEqual(api.read_json(saved)["error"], summary["error"])
        self.assertFalse((self.workspace / ".drdebug.lock").exists())

    def test_full_stage_timeout_preserves_failure_and_restores_independent_case_bytes(self):
        original_bytes = (self.workspace / "task/Fixture.json").read_bytes()
        def fail_full(child, timeout):
            if child.command[-1] == "full":
                raise api.EngineStopped(4, "simulated_timeout_control_flow_only")
            return self.wait(child, timeout)
        summary = self.execute(fail_full)
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["returncode"], 4)
        self.assertEqual(summary["error"], "simulated_timeout_control_flow_only")
        independent = Path(summary["output"]) / "task/Fixture.json"
        self.assertEqual(independent.read_bytes(), original_bytes)
        self.assertEqual((self.workspace / "task/Fixture.json").read_bytes(), original_bytes)
        self.assertTrue((Path(summary["output"]) / "compact.json").is_file())
        self.assertTrue((Path(summary["output"]) / "full.stderr.log").is_file())
        self.assertTrue((Path(summary["output"]) / "summary.json").is_file())
        self.assertFalse((self.workspace / ".drdebug.lock").exists())

    def test_child_rejects_stale_source_identity_before_startup_or_user_task_import(self):
        config = self.root / "replay-config.json"
        api.write_json(config, {"task": str(self.workspace / "task"), "source_sha256": {"fixture.cpp": "0" * 64}})
        with mock.patch.object(runtime_checks, "startup_evidence", side_effect=AssertionError("must reject stale source first")) as startup:
            code = verification.replay(config, "compact")
        self.assertEqual(code, 1)
        startup.assert_not_called()
        result = api.read_json(self.root / "compact.json")
        self.assertFalse(result["passed"])
        self.assertIn("source identity", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
