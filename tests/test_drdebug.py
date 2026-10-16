"""CPU behavior tests for the public skill wrapper, never model/NPU calls.

Run: python3 -B -m unittest discover -s tests -p test_drdebug.py -v
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "skills/drdebug/scripts/drdebug.py"
spec = importlib.util.spec_from_file_location("drdebug_public_launcher", LAUNCHER)
drdebug = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drdebug)


class PortableSkillTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="drdebug-test-")
        self.base = Path(self.temp.name).resolve()
        self.task = self.base / "input"
        (self.task / "kernel").mkdir(parents=True)
        # These would fail if prepare imported/executed user code.
        (self.task / "model.py").write_text("raise RuntimeError('MUST NOT EXECUTE')\n", encoding="utf-8")
        (self.task / "model_new_ascendc.py").write_text("raise RuntimeError('MUST NOT EXECUTE')\n", encoding="utf-8")
        (self.task / "kernel/pybind11.cpp").write_text("PYBIND11_MODULE(example, m) {}\n", encoding="utf-8")
        (self.task / "kernel/kernel.cpp").write_text("// deliberately not compiled by CPU tests\n", encoding="utf-8")
        one = {"inputs": [{"name": "x", "shape": [1], "dtype": "float16", "type": "tensor"}]}
        two = {"inputs": [{"name": "x", "shape": [3], "dtype": "float16", "type": "tensor"}]}
        (self.task / "Add.json").write_text(json.dumps(one) + "\n", encoding="utf-8")
        (self.task / "Add.json.bak").write_text(json.dumps(one) + "\n" + json.dumps(two) + "\n", encoding="utf-8")
        self.workspace = self.base / "workspace"

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, **kwargs):
        values = {"task": str(self.task), "output": str(self.workspace), "op_name": "Add", "reference_root": None}
        values.update(kwargs)
        return drdebug.prepare(argparse.Namespace(**values))

    def run_args(self, **kwargs):
        values = {"workspace": str(self.workspace), "npu": "3", "soc": "ActualTestSOC", "model": None,
                  "claude_bin": "claude", "max_attempts": None, "max_turns": None,
                  "deadline_sec": None, "dry_run": False, "allow_worker_shell": False}
        values.update(kwargs)
        return argparse.Namespace(**values)

    def append_events(self, events):
        path = self.workspace / "task/.debug_events/events.jsonl"
        path.parent.mkdir(exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event) + "\n")

    def test_prepare_is_source_only_and_original_artifact_unchanged(self):
        root = drdebug.source_root()
        before = drdebug.tree_hashes(root)
        (self.task / "kernel/build").mkdir()
        (self.task / "kernel/build/stale.so").write_bytes(b"old binary")
        (self.task / ".debug_events").mkdir()
        (self.task / ".debug_events/events.jsonl").write_text('{"type":"session_done"}\n', encoding="utf-8")
        result = self.prepare()
        self.assertTrue(result["prepared"])
        self.assertEqual(before, drdebug.tree_hashes(root))
        self.assertFalse((self.workspace / "task/kernel/build").exists())
        self.assertFalse((self.workspace / "task/.debug_events").exists())
        self.assertEqual((self.task / "model.py").read_bytes(), (self.workspace / "task/model.py").read_bytes())
        _, manifest = drdebug.load_workspace(self.workspace)
        self.assertEqual(manifest["task_contract"]["full_cases"], 2)

    def test_project_registration_and_explicit_portable_adapter(self):
        self.prepare()
        runtime = self.workspace / "runtime"
        worker = (runtime / ".claude/agents/ascendc-debug-agent-constructive.md").read_text(encoding="utf-8")
        original = (drdebug.source_root() / ".claude/agents/ascendc-debug-agent-constructive.md").read_text(encoding="utf-8")
        self.assertTrue(worker.startswith(original))
        self.assertIn("public_portable_reference_fallback", worker)
        self.assertIn("Phase A", worker)
        self.assertTrue((runtime / ".claude/skills/ascendc-debug/SKILL.md").is_file())
        for relative in ("engine/runner.py", "engine/validate_runner.py", "scripts/precision_gate.py"):
            self.assertEqual(drdebug.sha(runtime / drdebug.DEBUG_REL / relative),
                             drdebug.sha(drdebug.source_root() / drdebug.DEBUG_REL / relative))

    def test_reference_root_is_optional_and_user_reference_can_be_staged(self):
        references = self.base / "references"
        (references / "add/kernel").mkdir(parents=True)
        (references / "add/kernel/add.cpp").write_text("// user reference\n", encoding="utf-8")
        self.prepare(reference_root=str(references))
        self.assertTrue((self.workspace / "runtime/archive_tasks/user_references/add/kernel/add.cpp").is_file())
        self.assertTrue(drdebug.read_json(self.workspace / drdebug.MANIFEST)["reference_root_supplied"])

    def test_user_reference_collisions_cannot_overwrite_bundled_history(self):
        archive = drdebug.source_root() / "archive_tasks"
        historical = drdebug.tree_hashes(archive)
        references = self.base / "colliding-references"
        names = ("README.md", "PROVENANCE.json", "LICENSE", "LICENSE-APACHE-2.0.txt",
                 "rms_norm/kernel/rms_norm_single_row.cpp")
        for name in names:
            file = references / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("User reference, separate attribution: " + name + "\n", encoding="utf-8")
        self.prepare(reference_root=str(references))
        copied_archive = self.workspace / "runtime/archive_tasks"
        for name, expected_hash in historical.items():
            self.assertEqual(drdebug.sha(copied_archive / name), expected_hash, name)
        self.assertFalse((copied_archive / "LICENSE").exists())
        for name in names:
            self.assertEqual((references / name).read_bytes(),
                             (copied_archive / "user_references" / name).read_bytes())
        self.assertEqual(drdebug.tree_hashes(archive), historical)
        scope = (copied_archive / "PUBLIC_REFERENCE_SCOPE.md").read_text(encoding="utf-8")
        self.assertIn("archive_tasks/user_references/", scope)
        self.assertIn("never replace bundled sources or metadata", scope)
        worker = (self.workspace / "runtime/.claude/agents/ascendc-debug-agent-constructive.md").read_text(encoding="utf-8")
        self.assertIn("archive_tasks/user_references/", worker)

    def test_optional_attribution_is_copied_and_protected(self):
        for name in drdebug.ATTRIBUTION_FILES:
            (self.task / name).write_text("Preserve attribution: " + name + "\n", encoding="utf-8")
        result = self.prepare()
        contract = result["contract"]
        self.assertEqual(set(contract["optional_protected_sha256"]), set(drdebug.ATTRIBUTION_FILES))
        for name in drdebug.ATTRIBUTION_FILES:
            self.assertEqual((self.task / name).read_bytes(), (self.workspace / "task" / name).read_bytes())
            self.assertEqual(contract["protected_sha256"][name], drdebug.sha(self.task / name))
        (self.workspace / "task/SOURCE_NOTICE.md").write_text("modified attribution\n", encoding="utf-8")
        with self.assertRaisesRegex(drdebug.ContractError, "Protected task inputs or attribution"):
            drdebug.load_workspace(self.workspace)

    def test_bundled_historical_archive_keeps_sources_and_accurate_scope(self):
        self.prepare()
        archive = drdebug.source_root() / "archive_tasks"
        self.assertTrue((archive / "PROVENANCE.json").is_file())
        copied = self.workspace / "runtime/archive_tasks"
        for source in drdebug.files(archive):
            self.assertEqual(drdebug.sha(source), drdebug.sha(copied / source.relative_to(archive)))
        scope = (copied / "PUBLIC_REFERENCE_SCOPE.md").read_text(encoding="utf-8")
        self.assertIn("bundled historical reference", scope)
        self.assertIn("does not certify fresh device execution", scope)
        self.assertNotIn("User-supplied cases only", scope)

    def test_prepare_does_not_overwrite_existing_workspace(self):
        self.prepare()
        with self.assertRaisesRegex(drdebug.ContractError, "already exists"):
            self.prepare()

    def test_missing_full_cases_is_explicit_failure(self):
        (self.task / "Add.json.bak").unlink()
        with self.assertRaisesRegex(drdebug.ContractError, "full-case"):
            self.prepare()
        self.assertFalse(self.workspace.exists())

    def test_compact_and_full_equivalent_case_sets_are_allowed(self):
        shutil.copyfile(self.task / "Add.json", self.task / "Add.json.bak")
        self.assertTrue(self.prepare()["contract"]["equivalent_case_sets"])

    def test_equivalent_duplicate_full_files_allowed_but_ambiguous_sets_rejected(self):
        (self.task / "Add.json.full").write_bytes((self.task / "Add.json").read_bytes())
        with self.assertRaisesRegex(drdebug.ContractError, "Ambiguous full"):
            self.prepare()
        (self.task / "Add.json.full").write_bytes((self.task / "Add.json.bak").read_bytes())
        result = self.prepare()
        self.assertEqual(result["contract"]["full_files"], ["Add.json.bak", "Add.json.full"])
        self.assertTrue((self.workspace / "task/Add.json.full").is_file())

    def test_case_jsonl_and_independent_case_sets(self):
        (self.task / "Add.json").write_text('{"inputs": []}\n', encoding="utf-8")
        with self.assertRaisesRegex(drdebug.ContractError, "nonempty inputs"):
            self.prepare()
        (self.task / "Add.json").write_text('{"inputs": [{"shape": [999]}]}\n', encoding="utf-8")
        compact_before = (self.task / "Add.json").read_bytes()
        full_before = (self.task / "Add.json.bak").read_bytes()
        contract = self.prepare()["contract"]
        self.assertEqual(contract["case_set_relation"], "disjoint")
        self.assertEqual(contract["compact_only_cases"], 1)
        self.assertEqual(contract["full_only_cases"], 2)
        self.assertEqual((self.workspace / "task/Add.json").read_bytes(), compact_before)
        self.assertEqual((self.workspace / "task/Add.json.bak").read_bytes(), full_before)

    def test_real_public_add_cumsum_sum_case_contracts_without_containment(self):
        dataset = REPO / "papers/2026-dont-regenerate-debug/dataset/tasks/level1"
        for folder, stem, compact_only in (("003_Add", "3_Add", 2),
                                          ("005_Cumsum", "5_Cumsum", 2),
                                          ("007_Sum", "7_Sum", 1)):
            with self.subTest(operator=folder):
                original = dataset / folder
                task = self.base / ("actual-" + folder)
                shutil.copytree(original, task)
                # Only candidate inputs are stubs. Reference/cases are the
                # actual released bytes; none are imported or executed.
                shutil.copyfile(self.task / "model_new_ascendc.py", task / "model_new_ascendc.py")
                shutil.copytree(self.task / "kernel", task / "kernel")
                output = self.base / ("prepared-" + folder)
                result = self.prepare(task=str(task), output=str(output), op_name=stem)
                contract = result["contract"]
                self.assertEqual(contract["compact_only_cases"], compact_only)
                self.assertEqual(contract["case_set_relation"], "overlapping_without_containment")
                self.assertGreater(contract["full_only_cases"], 0)
                for name in ("model.py", stem + ".json", stem + ".json.bak"):
                    self.assertEqual((original / name).read_bytes(), (output / "task" / name).read_bytes())
                drdebug.load_workspace(output)

    def test_native_candidate_and_binding_are_required(self):
        (self.task / "kernel/kernel.cpp").unlink()
        with self.assertRaisesRegex(drdebug.ContractError, "native"):
            self.prepare()
        (self.task / "kernel/kernel.cpp").write_text("// native\n", encoding="utf-8")
        (self.task / "kernel/pybind11.cpp").write_text("// missing module\n", encoding="utf-8")
        with self.assertRaisesRegex(drdebug.ContractError, "PYBIND11_MODULE"):
            self.prepare()

    def test_symlink_task_and_reference_inputs_rejected(self):
        (self.task / "kernel/outside.h").symlink_to(self.base / "private")
        with self.assertRaisesRegex(drdebug.ContractError, "Symlink"):
            self.prepare()
        self.assertFalse(self.workspace.exists())

    def test_symlink_parent_cannot_place_workspace_inside_original_task(self):
        alias = self.base / "parent-alias"
        alias.symlink_to(self.task, target_is_directory=True)
        before = drdebug.tree_hashes(self.task)
        with self.assertRaisesRegex(drdebug.ContractError, "must not contain"):
            self.prepare(output=str(alias / "new-repair"))
        self.assertFalse((self.task / "new-repair").exists())
        self.assertEqual(before, drdebug.tree_hashes(self.task))

    def test_symlink_task_root_is_rejected(self):
        alias = self.base / "task-alias"
        alias.symlink_to(self.task, target_is_directory=True)
        with self.assertRaisesRegex(drdebug.ContractError, "Symlink task root"):
            self.prepare(task=str(alias))

    def test_output_cannot_be_created_inside_frozen_source_or_reference_root(self):
        with self.assertRaisesRegex(drdebug.ContractError, "frozen runtime source"):
            self.prepare(output=str(drdebug.source_root() / "must-not-be-created-by-test"))
        references = self.base / "refs"
        references.mkdir()
        with self.assertRaisesRegex(drdebug.ContractError, "Reference root"):
            self.prepare(output=str(references / "must-not-be-created"), reference_root=str(references))
        self.assertFalse((references / "must-not-be-created").exists())

    def test_no_path_traversal_operator_or_permission_glob_expansion(self):
        with self.assertRaisesRegex(drdebug.ContractError, "plain filename"):
            self.prepare(op_name="../Add")
        with self.assertRaisesRegex(drdebug.ContractError, "unsafe"):
            self.prepare(output=str(self.base / "workspace[other]"))

    def test_protected_inputs_and_frozen_runtime_checked_before_resume(self):
        self.prepare()
        protected = self.workspace / "task/model.py"
        protected.write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(drdebug.ContractError, "Protected"):
            drdebug.run(self.run_args(dry_run=True))
        self.assertFalse(drdebug.status(self.workspace)["integrity"]["ok"])
        shutil.copyfile(self.task / "model.py", protected)
        (self.workspace / "runtime" / drdebug.DEBUG_REL / "engine/runner.py").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(drdebug.ContractError, "runtime changed"):
            drdebug.run(self.run_args(dry_run=True))

    def test_dry_run_is_no_exec_scoped_and_cleans_ablation_environment(self):
        self.prepare()
        with mock.patch.dict(os.environ, {"ABLATE_FULL_EVAL": "1", "ASCENDC_DEBUG_CLEAN_BUILD": "0", "PYTHONPATH": "/untrusted"}), \
             mock.patch.object(drdebug, "execute_engine", side_effect=AssertionError("must not execute")):
            args = self.run_args(dry_run=True)
            result = drdebug.run(args)
            _, manifest = drdebug.load_workspace(self.workspace)
            command, env, config = drdebug.engine_command(self.workspace, manifest, args, result["budget"], 1800)
        self.assertTrue(result["dry_run"])
        self.assertFalse(result["worker_shell_authorized"])
        self.assertEqual(result["environment"]["ASCEND_RT_VISIBLE_DEVICES"], "3")
        self.assertEqual(command[command.index("--npu") + 1], "0")
        self.assertEqual(command[command.index("--workdir") + 1], str(self.workspace / "task"))
        self.assertNotIn("ABLATE_FULL_EVAL", env)
        self.assertNotIn("ASCENDC_DEBUG_CLEAN_BUILD", env)
        self.assertNotIn("/untrusted", env["PYTHONPATH"])
        self.assertEqual(env["ASCENDC_DEBUG_KB_READ_ONLY"], "1")
        self.assertEqual(config["permissions"]["defaultMode"], "dontAsk")
        self.assertIn("Bash", config["permissions"]["deny"])
        self.assertNotIn("Bash", config["permissions"]["allow"])
        edits = [rule for rule in config["permissions"]["allow"] if rule.startswith(("Write(", "Edit("))]
        self.assertTrue(all(rule.startswith(("Write(//", "Edit(//")) for rule in edits))
        self.assertTrue(all("kernel/" in rule or "precision_audit_" in rule or "candidate_kb_entry.json" in rule for rule in edits))
        self.assertFalse(any("precision_tuning/**" in rule for rule in edits))
        self.assertFalse((self.workspace / drdebug.RUN_STATE).exists())

    def test_worker_shell_is_explicit_per_invocation_without_bypass(self):
        self.prepare()
        result = drdebug.run(self.run_args(dry_run=True, allow_worker_shell=True))
        self.assertTrue(result["worker_shell_authorized"])
        allowed = result["command"][result["command"].index("--allowed-tools") + 1]
        self.assertTrue(allowed.endswith(",Bash"))
        self.assertNotIn("--dangerously-skip-permissions", result["command"])
        self.assertNotIn("--permission-mode", result["command"])
        self.assertFalse(drdebug.run(self.run_args(dry_run=True))["worker_shell_authorized"])
        with mock.patch.object(drdebug, "environment_report", return_value={"ready": True}), \
             mock.patch.object(drdebug, "execute_engine", return_value=8):
            drdebug.run(self.run_args(allow_worker_shell=True))
        self.assertIn("Bash", drdebug.read_json(self.workspace / "runtime/.claude/settings.json")["permissions"]["deny"])

    def test_doctor_missing_dependencies_is_structured_and_strict(self):
        with mock.patch.object(drdebug.shutil, "which", return_value=None), \
             mock.patch.object(drdebug.importlib.util, "find_spec", return_value=None), \
             mock.patch.object(drdebug.subprocess, "run", side_effect=AssertionError("must not execute")):
            report = drdebug.environment_report()
            self.assertFalse(report["ready"])
            self.assertTrue(any(check["name"] == "python:torch_npu" and not check["ok"] for check in report["checks"]))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(drdebug.main(["doctor", "--strict", "--json"]), 2)
                self.assertFalse(json.loads(output.getvalue())["ready"])

    def test_run_refuses_missing_dependencies_before_model_process(self):
        self.prepare()
        with mock.patch.object(drdebug, "environment_report", return_value={"ready": False, "checks": [{"name": "npu_runtime", "ok": False}]}), \
             mock.patch.object(drdebug, "execute_engine", side_effect=AssertionError("must not execute")):
            result = drdebug.run(self.run_args())
        self.assertEqual(result["launcher_outcome"], "preflight_failed")
        self.assertFalse((self.workspace / drdebug.RUN_STATE).exists())

    def test_argv_forwarding_budget_persistence_and_resume(self):
        self.prepare()
        model = "a-model;touch NEVER_EXECUTED"
        with mock.patch.object(drdebug, "environment_report", return_value={"ready": True}), \
             mock.patch.object(drdebug, "execute_engine", return_value=8) as execute:
            first = drdebug.run(self.run_args(model=model))
            first_command = execute.call_args.args[0]
            first_timeout = execute.call_args.kwargs["timeout"]
            self.assertEqual(first["returncode"], 8)
            self.assertEqual(first_command[first_command.index("--model") + 1], model)
            self.assertEqual(execute.call_args.kwargs["cwd"], self.workspace / "runtime")
            self.assertEqual(first_timeout, 1800)
            self.append_events([{"type": "session_started", "entry_failure_type": "precision_failed"},
                                {"type": "attempt_started", "attempt": 0, "failure_type": "precision_failed"}])
            self.assertEqual(drdebug.run(self.run_args())["returncode"], 8)
            self.assertLess(execute.call_args.kwargs["timeout"], first_timeout)
        history = drdebug.read_json(self.workspace / drdebug.RUN_STATE)
        self.assertEqual(len(history["runs"]), 2)
        self.assertEqual(drdebug.status(self.workspace)["session"]["attempts_used"], 1)
        with self.assertRaisesRegex(drdebug.ContractError, "Resume retains"):
            drdebug.run(self.run_args(max_attempts=6, dry_run=True))

    def test_terminal_session_never_restarts_and_failure_code_is_preserved(self):
        self.prepare()
        self.append_events([{"type": "session_started"}, {"type": "session_done", "session_outcome": "provider_api_error"}])
        before = (self.workspace / "task/.debug_events/events.jsonl").read_bytes()
        with mock.patch.object(drdebug, "execute_engine", side_effect=AssertionError("must not execute")), \
             mock.patch.object(drdebug, "environment_report", side_effect=AssertionError("no dependency check needed")):
            result = drdebug.run(self.run_args())
        self.assertTrue(result["already_terminal"])
        self.assertEqual(result["returncode"], 8)
        self.assertEqual(before, (self.workspace / "task/.debug_events/events.jsonl").read_bytes())

    def test_torn_final_event_requires_explicit_repair_and_is_never_rewritten(self):
        self.prepare()
        self.append_events([{"type": "session_started", "entry_failure_type": "precision_failed"}])
        path = self.workspace / "task/.debug_events/events.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"type":"action_started"')
        before = path.read_bytes()
        self.assertTrue(drdebug.status(self.workspace)["session"]["truncated_last_event"])
        with self.assertRaisesRegex(drdebug.ContractError, "truncated final record"):
            drdebug.run(self.run_args(dry_run=True))
        self.assertEqual(before, path.read_bytes())

    def test_process_launch_uses_argv_not_shell_strings(self):
        cwd = self.base / "runtime"
        cwd.mkdir()
        with mock.patch.object(drdebug.subprocess, "Popen") as spawn:
            spawn.return_value.wait.return_value = 8
            command = [sys.executable, "-m", "engine", "; touch NEVER_EXECUTED"]
            self.assertEqual(drdebug.execute_engine(command, cwd=cwd, env={}, timeout=1), 8)
            self.assertEqual(spawn.call_args.args[0], command)
            self.assertNotIn("shell", spawn.call_args.kwargs)

    def test_installed_bundle_resolves_without_repository(self):
        isolated = self.base / "standalone/skills/drdebug"
        (isolated / "scripts").mkdir(parents=True)
        (isolated / "references").mkdir()
        shutil.copyfile(LAUNCHER, isolated / "scripts/drdebug.py")
        shutil.copyfile(drdebug.SKILL_ROOT / "references/portable-worker.md", isolated / "references/portable-worker.md")
        drdebug.copy_tree(drdebug.source_root(), isolated / "runtime/source/debug_agent_v5")
        self.assertEqual(drdebug.source_root(isolated), isolated / "runtime/source/debug_agent_v5")
        result = subprocess.run([sys.executable, "-B", str(isolated / "scripts/drdebug.py"), "prepare", str(self.task),
                                 "--output", str(self.workspace), "--op-name", "Add", "--json"],
                                cwd=self.base, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                                capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = drdebug.read_json(self.workspace / drdebug.MANIFEST)
        self.assertEqual(manifest["runtime_source"], str(isolated / "runtime/source/debug_agent_v5"))
        self.assertNotIn(str(REPO), manifest["runtime_source"])

    def test_missing_installed_bundle_cannot_resolve_unrelated_runtime(self):
        skill = self.base / "empty/skills/drdebug"
        skill.mkdir(parents=True)
        with self.assertRaisesRegex(drdebug.ContractError, "complete skill bundle"):
            drdebug.source_root(skill)

    def test_original_engine_cpu_mock_closure_and_terminal_resume(self):
        self.prepare()
        runtime = self.workspace / "runtime"
        snippet = r'''
import json, sys
from pathlib import Path
from engine.runner import run_debug_session
from engine.events import read_events
calls=[]
def dispatcher(action, task, op, callback):
    calls.append(action.step)
    if action.step == 'baseline_checkpoint':
        return {'success': True, 'failure_type': 'precision_failed'}
    if action.step == 'validate':
        return {'gate':'GATE-V', 'passed':True, 'loop_signal':'PASS', 'checks':{'correctness_passed':True}}
    return {'success':True}
task=Path(sys.argv[1])
first=run_debug_session(task, op_name='Add', entry_failure_type='precision_failed', dispatcher=dispatcher)
before=len(read_events(task))
second=run_debug_session(task, op_name='Add', dispatcher=lambda *a: (_ for _ in ()).throw(AssertionError('terminal redispatch')))
assert len(read_events(task)) == before
assert calls.count('diagnose_and_fix') == 1, calls
assert first['session_outcome'] == second['session_outcome'] == 'success', first
print(json.dumps({'mock_only':True, 'outcome':first['session_outcome'], 'calls':calls}))
'''
        env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(runtime / drdebug.DEBUG_REL), str(runtime))),
               "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run([sys.executable, "-B", "-c", snippet, str(self.workspace / "task")],
                                cwd=runtime, env=env, capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"mock_only": true', result.stdout)
        self.assertTrue(drdebug.status(self.workspace)["session"]["terminal"])
        self.assertTrue(drdebug.status(self.workspace)["integrity"]["ok"])

    def test_original_engine_cpu_mock_reconciles_interrupted_action(self):
        self.prepare()
        runtime = self.workspace / "runtime"
        snippet = r'''
import json, sys
from pathlib import Path
from engine.runner import run_debug_session
from engine.events import read_events
task=Path(sys.argv[1])
def complete(action, task, op, callback):
    if action.step == 'baseline_checkpoint':
        return {'success': True, 'failure_type': 'precision_failed'}
    if action.step == 'validate':
        return {'gate':'GATE-V', 'passed':True, 'loop_signal':'PASS', 'checks':{'correctness_passed':True}}
    return {'success':True}
def interrupted(action, *rest):
    if action.step == 'diagnose_and_fix':
        raise KeyboardInterrupt('CPU simulated interruption')
    return complete(action, *rest)
try:
    run_debug_session(task, op_name='Add', entry_failure_type='precision_failed', dispatcher=interrupted)
except KeyboardInterrupt:
    pass
result=run_debug_session(task, op_name='Add', dispatcher=complete)
events=read_events(task)
assert sum(e['type']=='session_started' for e in events)==1
assert any((e.get('result') or {}).get('reason','').startswith('crash-recovery:') for e in events)
assert result['session_outcome']=='success', result
print(json.dumps({'mock_only':True, 'resume':'reconciled'}))
'''
        env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(runtime / drdebug.DEBUG_REL), str(runtime))),
               "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run([sys.executable, "-B", "-c", snippet, str(self.workspace / "task")],
                                cwd=runtime, env=env, capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"resume": "reconciled"', result.stdout)


if __name__ == "__main__":
    unittest.main()
