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
import signal
import subprocess
import sys
import tempfile
import time
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
                  "command": "next", "max_attempts": None,
                  "deadline_sec": None, "dry_run": False}
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
        worker = (runtime / "host-instructions.md").read_text(encoding="utf-8")
        original = (drdebug.source_root() / ".claude/agents/ascendc-debug-agent-constructive.md").read_text(encoding="utf-8")
        original_frontmatter, separator, original_body = original.partition("\n---\n")
        self.assertTrue(worker.startswith("## Prepared workspace resource context"))
        self.assertEqual(worker.count(original_body), 1)
        self.assertLess(worker.index("## Prepared workspace resource context"), worker.index(original_body))
        self.assertIn("public_portable_reference_fallback", worker)
        self.assertIn("Phase A", worker)
        self.assertFalse((runtime / ".claude/settings.json").exists())
        for relative in ("engine/runner.py", "scripts/precision_gate.py"):
            self.assertEqual(drdebug.sha(runtime / drdebug.DEBUG_REL / relative),
                             drdebug.sha(drdebug.source_root() / drdebug.DEBUG_REL / relative))

    def test_new_workspace_guard_is_staged_audited_and_integrity_protected(self):
        before = drdebug.tree_hashes(drdebug.source_root())
        self.prepare()
        _, manifest = drdebug.load_workspace(self.workspace)
        guard = manifest["nonfinite_guard"]
        self.assertTrue(guard["hook_before_cli_main"])
        self.assertTrue(guard["import_entrypoint_covered"])
        self.assertFalse(guard["frozen_archive_modified"])
        runtime = self.workspace / "runtime"
        self.assertEqual(drdebug.sha(runtime / guard["helper"]), guard["helper_sha256"])
        self.assertEqual(drdebug.sha(runtime / guard["verifier"]), guard["adapted_sha256"])
        self.assertEqual(drdebug.sha(drdebug.source_root() / guard["verifier"]), guard["original_sha256"])
        self.assertEqual(drdebug.status(self.workspace)["adapter_support"]["nonfinite_guard"], "verified")
        (runtime / guard["helper"]).write_text("# changed guard\n")
        with self.assertRaisesRegex(drdebug.ContractError, "runtime changed"):
            drdebug.load_workspace(self.workspace)
        self.assertEqual(drdebug.tree_hashes(drdebug.source_root()), before)

    def test_guard_staging_failure_removes_only_new_incomplete_output(self):
        before = drdebug.tree_hashes(drdebug.source_root())
        with mock.patch.object(drdebug.nonfinite_guard, "stage_guard", side_effect=ValueError("synthetic guard staging failure")):
            with self.assertRaisesRegex(ValueError, "guard staging failure"):
                self.prepare()
        self.assertFalse(self.workspace.exists())
        self.assertEqual(drdebug.tree_hashes(drdebug.source_root()), before)
        self.assertTrue((self.task / "kernel/kernel.cpp").is_file())

    def test_old_workspace_without_guard_remains_explicit_and_is_not_hotpatched(self):
        self.prepare()
        manifest = drdebug.read_json(self.workspace / drdebug.MANIFEST)
        guard = manifest.pop("nonfinite_guard")
        manifest["schema_version"] = 3
        runtime = self.workspace / "runtime"
        (runtime / guard["helper"]).unlink()
        (runtime / guard["verifier"]).write_bytes((drdebug.source_root() / guard["verifier"]).read_bytes())
        manifest["runtime_sha256"] = drdebug.tree_hashes(runtime, exclude_settings=True)
        drdebug.write_json(self.workspace / drdebug.MANIFEST, manifest)
        before = drdebug.tree_hashes(self.workspace)
        status = drdebug.status(self.workspace)
        self.assertEqual(status["adapter_support"]["nonfinite_guard"], "legacy_unchecked")
        self.assertTrue(status["nonfinite_guard"]["requires_new_workspace"])
        with self.assertRaisesRegex(drdebug.ContractError, "Legacy worker"):
            drdebug.host_command(self.run_args(dry_run=True))
        self.assertEqual(drdebug.tree_hashes(self.workspace), before)

    def test_original_gate_discovers_ast_bridge_and_rejects_pure_torch_wrapper(self):
        (self.task / "model_new_ascendc.py").write_text(
            "import torch\nclass ModelNew(torch.nn.Module):\n"
            "    def forward(self, x):\n        return torch.relu(x)\n", encoding="utf-8")
        self.prepare()
        runtime = self.workspace / "runtime"
        bridge = self.workspace / drdebug.AST_VALIDATOR_REL
        self.assertFalse(bridge.is_symlink())
        self.assertEqual(bridge.read_bytes(), (runtime / drdebug.AST_VALIDATOR_REL).read_bytes())
        snippet = r'''
import json, sys
from pathlib import Path
from scripts.gates.common import _find_ast_validator, check_ast_degrade
task = Path(sys.argv[1])
print(json.dumps({'validator': str(_find_ast_validator(task)), 'checks': check_ast_degrade(task)}))
'''
        env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(runtime / drdebug.DEBUG_REL), str(runtime))),
               "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run([sys.executable, "-B", "-c", snippet, str(self.workspace / "task")],
                                cwd=runtime, env=env, capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        observed = json.loads(result.stdout)
        self.assertEqual(observed["validator"], str(bridge))
        self.assertTrue(observed["checks"]["ast_validator_present"])
        self.assertFalse(observed["checks"]["ast_degrade_pass"])
        self.assertEqual(drdebug.status(self.workspace)["adapter_support"]["ast_validator_bridge"], "verified")

    def test_ast_bridge_tampering_insertion_and_symlinks_are_rejected_before_run(self):
        self.prepare()
        bridge = self.workspace / drdebug.AST_VALIDATOR_REL
        original = bridge.read_bytes()
        bridge.write_text("raise SystemExit(0)\n", encoding="utf-8")
        with mock.patch.object(drdebug, "environment_report", side_effect=AssertionError("must not preflight")), \
             mock.patch.object(drdebug.subprocess, "Popen", side_effect=AssertionError("must not execute")):
            with self.assertRaisesRegex(drdebug.ContractError, "support changed"):
                drdebug.host_command(self.run_args())
        self.assertFalse(drdebug.status(self.workspace)["integrity"]["ok"])
        bridge.write_bytes(original)
        injected = bridge.parent / "json.py"
        injected.write_text("# untracked module\n", encoding="utf-8")
        with self.assertRaisesRegex(drdebug.ContractError, "support changed"):
            drdebug.load_workspace(self.workspace)
        injected.unlink()
        unexpected_directory = self.workspace / "skills/untracked"
        unexpected_directory.mkdir()
        with self.assertRaisesRegex(drdebug.ContractError, "Unexpected workspace support directory"):
            drdebug.load_workspace(self.workspace)
        unexpected_directory.rmdir()
        bridge.unlink()
        bridge.symlink_to(self.workspace / "runtime" / drdebug.AST_VALIDATOR_REL)
        with self.assertRaisesRegex(drdebug.ContractError, "Symlink workspace support"):
            drdebug.load_workspace(self.workspace)
        bridge.unlink()
        bridge.write_bytes(original)
        drdebug.load_workspace(self.workspace)


    def test_task_local_exit_zero_shadow_cannot_replace_the_discovered_ast_bridge(self):
        (self.task / "model_new_ascendc.py").write_text(
            "import torch\nclass ModelNew(torch.nn.Module):\n"
            "    def forward(self, x):\n        return torch.relu(x)\n", encoding="utf-8")
        self.prepare()
        shadow = self.workspace / "task" / drdebug.AST_VALIDATOR_REL
        shadow.parent.mkdir(parents=True)
        shadow.write_text("raise SystemExit(0)\n", encoding="utf-8")
        runtime = self.workspace / "runtime"
        snippet = r'''
import json, sys
from pathlib import Path
from scripts.gates.common import _find_ast_validator, check_ast_degrade
task = Path(sys.argv[1])
print(json.dumps({'validator': str(_find_ast_validator(task)), 'checks': check_ast_degrade(task)}))
'''
        env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(runtime / drdebug.DEBUG_REL), str(runtime))),
               "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run([sys.executable, "-B", "-c", snippet, str(self.workspace / "task")],
                                cwd=runtime, env=env, capture_output=True, text=True, timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        observed = json.loads(result.stdout)
        # This is a real bypass if the adapter lets the frozen gate run.
        self.assertEqual(observed["validator"], str(shadow))
        self.assertTrue(observed["checks"]["ast_degrade_pass"])
        with mock.patch.object(drdebug, "environment_report", side_effect=AssertionError("must not preflight")), \
             mock.patch.object(drdebug.subprocess, "Popen", side_effect=AssertionError("must not execute")):
            with self.assertRaisesRegex(drdebug.ContractError, "shadow support"):
                drdebug.host_command(self.run_args())
        status = drdebug.status(self.workspace)
        self.assertFalse(status["integrity"]["ok"])
        self.assertEqual(status["adapter_support"]["ast_validator_bridge"], "unverified")
        shutil.rmtree(self.workspace / "task/skills")
        drdebug.load_workspace(self.workspace)
        (self.workspace / "task/skills").mkdir()
        with self.assertRaisesRegex(drdebug.ContractError, "shadow support"):
            drdebug.load_workspace(self.workspace)

    def test_legacy_workspace_keeps_original_status_without_claiming_ast_bridge(self):
        self.prepare()
        shutil.rmtree(self.workspace / "skills")
        manifest_path = self.workspace / drdebug.MANIFEST
        manifest = drdebug.read_json(manifest_path)
        manifest["schema_version"] = 1
        manifest.pop("support_sha256")
        drdebug.write_json(manifest_path, manifest)
        old_status = {"session_outcome": "provider_api_error", "ast_degrade_pass": True}
        drdebug.write_json(self.workspace / "task/debug_status.json", old_status)
        self.append_events([{"type": "session_started"},
                            {"type": "session_aborted", "details": {"session_outcome": "provider_api_error"}}])
        before = drdebug.tree_hashes(self.workspace)
        status = drdebug.status(self.workspace)
        self.assertTrue(status["integrity"]["ok"])
        self.assertEqual(status["adapter_support"]["ast_validator_bridge"], "legacy_unmanaged")
        self.assertEqual(status["engine_status"], old_status)
        with self.assertRaisesRegex(drdebug.ContractError, "Legacy worker"):
            drdebug.host_command(self.run_args())
        self.assertEqual(drdebug.tree_hashes(self.workspace), before)
        injected = self.workspace / drdebug.AST_VALIDATOR_REL
        injected.parent.mkdir(parents=True)
        injected.write_text("# untracked legacy support\n", encoding="utf-8")
        with self.assertRaisesRegex(drdebug.ContractError, "Legacy workspace has untracked support"):
            drdebug.load_workspace(self.workspace)

    def test_reference_root_is_optional_and_user_reference_can_be_staged(self):
        references = self.base / "references"
        (references / "add/kernel").mkdir(parents=True)
        (references / "add/kernel/add.cpp").write_text("// user reference\n", encoding="utf-8")
        self.prepare(reference_root=str(references))
        staged = self.workspace / "runtime/archive_tasks/user_references"
        self.assertEqual((staged / "add/kernel/add.cpp").read_bytes(),
                         (references / "add/kernel/add.cpp").read_bytes())
        manifest = drdebug.read_json(self.workspace / drdebug.MANIFEST)
        self.assertTrue(manifest["reference_root_supplied"])
        worker = (self.workspace / "runtime/host-instructions.md").read_text(encoding="utf-8")
        context = worker[:worker.index("# System Prompt")]
        self.assertIn("User references supplied: `true`", context)
        self.assertIn(f"Copied user-reference directory (read-only): `{staged}`", context)
        self.assertIn("Phase A", context)
        self.assertIn("selected paths", context)
        self.assertIn("Only the isolated request candidate may be edited", context)
        self.assertNotIn("{{", worker)
        self.assertIn("workspace-specific read-only reference discovery", manifest["portability_changes"])
        drdebug.load_workspace(self.workspace)

    def test_no_user_reference_is_explicit_in_early_resource_context(self):
        self.prepare()
        staged = self.workspace / "runtime/archive_tasks/user_references"
        self.assertFalse(staged.exists())
        manifest = drdebug.read_json(self.workspace / drdebug.MANIFEST)
        self.assertFalse(manifest["reference_root_supplied"])
        worker = (self.workspace / "runtime/host-instructions.md").read_text(encoding="utf-8")
        context = worker[:worker.index("# System Prompt")]
        self.assertIn("User references supplied: `false`", context)
        self.assertIn(f"No user-reference directory was staged; do not search `{staged}`", context)
        self.assertIn("bundled historical references", context)
        self.assertNotIn("Copied user-reference directory (read-only)", context)
        self.assertNotIn("{{", worker)

    def test_empty_supplied_reference_has_real_directory_and_no_relevance_claim(self):
        references = self.base / "empty-references"
        references.mkdir()
        self.prepare(reference_root=str(references))
        staged = self.workspace / "runtime/archive_tasks/user_references"
        self.assertTrue(staged.is_dir())
        self.assertEqual(list(staged.iterdir()), [])
        worker = (self.workspace / "runtime/host-instructions.md").read_text(encoding="utf-8")
        context = worker[:worker.index("# System Prompt")]
        self.assertIn("User references supplied: `true`", context)
        self.assertIn("Empty or irrelevant references", context)
        self.assertIn("do not assume a matching example exists", context)

    def test_reference_context_preserves_paths_with_spaces_and_original_worker_bytes(self):
        references = self.base / "user examples with spaces"
        (references / "Add example/kernel").mkdir(parents=True)
        source = references / "Add example/kernel/add.cpp"
        source.write_bytes(b"// supplied independent reference\n")
        self.workspace = self.base / "workspace with spaces"
        before = drdebug.tree_hashes(drdebug.source_root())
        self.prepare(reference_root=str(references))
        staged = self.workspace / "runtime/archive_tasks/user_references"
        self.assertEqual((staged / "Add example/kernel/add.cpp").read_bytes(), source.read_bytes())
        original = (drdebug.source_root() / ".claude/agents/ascendc-debug-agent-constructive.md").read_bytes()
        worker = (self.workspace / "runtime/host-instructions.md").read_bytes()
        frontmatter, separator, body = original.partition(b"\n---\n")
        self.assertTrue(worker.startswith(b"## Prepared workspace resource context"))
        self.assertEqual(worker.count(body), 1)
        self.assertIn(f"`{staged}`".encode("utf-8"), worker[:worker.index(body)])
        self.assertEqual(drdebug.tree_hashes(drdebug.source_root()), before)
        drdebug.load_workspace(self.workspace)

    def test_reference_and_early_resource_context_remain_integrity_protected(self):
        references = self.base / "references"
        references.mkdir()
        (references / "example.cpp").write_text("// source example\n", encoding="utf-8")
        self.prepare(reference_root=str(references))
        runtime = self.workspace / "runtime"
        for path in (runtime / "archive_tasks/user_references/example.cpp",
                     runtime / "host-instructions.md"):
            with self.subTest(path=path.relative_to(runtime)):
                original = path.read_bytes()
                path.write_bytes(original + b"\n// changed after prepare\n")
                with self.assertRaises(drdebug.ContractError):
                    drdebug.load_workspace(self.workspace)
                path.write_bytes(original)
                drdebug.load_workspace(self.workspace)

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
        worker = (self.workspace / "runtime/host-instructions.md").read_text(encoding="utf-8")
        self.assertIn(str(copied_archive / "user_references"), worker)

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
        with self.assertRaisesRegex(drdebug.ContractError, "control characters"):
            self.prepare(output=str(self.base / "workspace\nother"))

    def test_protected_inputs_and_frozen_runtime_checked_before_resume(self):
        self.prepare()
        protected = self.workspace / "task/model.py"
        protected.write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(drdebug.ContractError, "Protected"):
            drdebug.host_command(self.run_args(dry_run=True))
        self.assertFalse(drdebug.status(self.workspace)["integrity"]["ok"])
        shutil.copyfile(self.task / "model.py", protected)
        (self.workspace / "runtime" / drdebug.DEBUG_REL / "engine/runner.py").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(drdebug.ContractError, "runtime changed"):
            drdebug.host_command(self.run_args(dry_run=True))

    def test_dry_run_has_no_model_or_npu_execution_and_cleans_environment(self):
        self.prepare()
        with mock.patch.dict(os.environ, {"ABLATE_FULL_EVAL": "1", "ASCENDC_DEBUG_CLEAN_BUILD": "0", "PYTHONPATH": "/untrusted"}), \
             mock.patch.object(drdebug.subprocess, "Popen", side_effect=AssertionError("must not execute")):
            result = drdebug.host_command(self.run_args(dry_run=True))
            _, manifest = drdebug.load_workspace(self.workspace)
            env = drdebug.base_execution_environment(self.workspace, manifest, npu="3", soc="ActualTestSOC")
        self.assertTrue(result["dry_run"])
        self.assertFalse(result["external_worker_required"])
        self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "3")
        self.assertEqual(env["ASCEND_DEVICE_ID"], "0")
        self.assertNotIn("ABLATE_FULL_EVAL", env)
        self.assertNotIn("ASCENDC_DEBUG_CLEAN_BUILD", env)
        self.assertNotIn("/untrusted", env["PYTHONPATH"])
        self.assertEqual(env["ASCENDC_DEBUG_KB_READ_ONLY"], "1")
        self.assertFalse((self.workspace / drdebug.RUN_STATE).exists())

    def test_worker_command_cannot_launch_and_guides_fresh_migration(self):
        with self.assertRaisesRegex(drdebug.ContractError, "nested Claude worker was removed"):
            drdebug.run(self.run_args())

    def test_doctor_checks_runtime_without_calling_any_agent_cli(self):
        with mock.patch.object(drdebug.runtime_checks, "dependency_report", return_value={"checks": []}), \
             mock.patch.object(drdebug.subprocess, "run", side_effect=AssertionError("no CLI probes")):
            result = drdebug.environment_report()
        self.assertTrue(result["ready"])
        self.assertFalse(result["external_worker_required"])

    def test_explicit_host_budget_is_checked_without_execution(self):
        self.prepare()
        for bad in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(bad=bad), self.assertRaises(drdebug.ContractError):
                drdebug.host_command(self.run_args(dry_run=True, deadline_sec=bad))
        self.assertEqual(drdebug.host_command(self.run_args(dry_run=True, max_attempts=7))["configuration"]["budget"]["max_attempts"], 7)

    def test_missing_saved_budget_never_gets_fresh_defaults(self):
        self.prepare()
        for history in ({}, {"device": {}, "runs": []},
                        {"budget": {"max_attempts": 1}, "device": {}, "runs": []}):
            with self.subTest(history=history):
                drdebug.write_json(self.workspace / drdebug.RUN_STATE, history)
                with self.assertRaisesRegex(drdebug.ContractError, "missing or invalid budget"):
                    drdebug.host_command(self.run_args(dry_run=True))
                self.assertEqual(drdebug.read_json(self.workspace / drdebug.RUN_STATE), history)
                self.assertFalse((self.workspace / ".drdebug.lock").exists())

    def test_missing_history_cannot_adopt_existing_engine_evidence(self):
        self.prepare()
        self.append_events([{"type": "session_started"}])
        with self.assertRaisesRegex(drdebug.ContractError, "no saved host budget"):
            drdebug.host_command(self.run_args(dry_run=True))
        self.assertFalse((self.workspace / drdebug.RUN_STATE).exists())

    def test_saved_budget_and_device_cannot_change_on_resume(self):
        self.prepare()
        history = {"budget": {"max_attempts": 2, "deadline_sec": 5000},
                   "device": {"physical_npu": "3", "soc": "ActualTestSOC"}, "runs": [], "external_elapsed_sec": 0}
        drdebug.write_json(self.workspace / drdebug.RUN_STATE, history)
        self.assertEqual(drdebug.host_command(self.run_args(dry_run=True))["configuration"]["budget"], history["budget"])
        for changes in ({"max_attempts": 5}, {"deadline_sec": 86400}, {"npu": "0"}):
            with self.subTest(changes=changes), self.assertRaises(drdebug.ContractError):
                drdebug.host_command(self.run_args(dry_run=True, **changes))
        self.assertEqual(drdebug.read_json(self.workspace / drdebug.RUN_STATE), history)

    def test_failed_preflight_time_is_preserved_and_not_a_fresh_budget(self):
        self.prepare()
        with mock.patch.object(drdebug, "environment_report", return_value={"ready": False}), \
             mock.patch.object(drdebug.time, "monotonic", side_effect=[100, 108]), \
             mock.patch.object(drdebug.subprocess, "Popen", side_effect=AssertionError("no engine or CLI launch")):
            result = drdebug.host_command(self.run_args(deadline_sec=1))
        self.assertEqual(result["state"], "preflight_failed")
        history = drdebug.read_json(self.workspace / drdebug.RUN_STATE)
        self.assertEqual(history["budget"]["deadline_sec"], 1)
        self.assertEqual(history["external_elapsed_sec"], 8)
        self.assertEqual(history["runs"][0]["preflight_elapsed_sec"], 8)
        self.assertFalse((self.workspace / ".drdebug.lock").exists())

    def test_numerical_children_do_not_inherit_host_provider_secrets(self):
        self.prepare()
        _, manifest = drdebug.load_workspace(self.workspace)
        names = ["ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "KIMI_API_KEY", "KIMI_MODEL_API_KEY", "KIMI_CODE_HOME", "MOONSHOT_API_KEY", "CODEX_API_KEY", "OPENAI_API_KEY"]
        with mock.patch.dict(os.environ, dict.fromkeys(names, "synthetic-test-value")):
            env = drdebug.base_execution_environment(self.workspace, manifest, npu="3", soc="ActualTestSOC")
        self.assertFalse(set(names) & set(env))
        self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "3")
        self.assertEqual(env["ASCENDC_SOC_VERSION"], "ActualTestSOC")

    def test_unprepared_doctor_does_not_forward_new_kimi_host_credentials(self):
        with mock.patch.dict(os.environ, {"KIMI_MODEL_API_KEY": "synthetic-test-key", "OPENAI_API_KEY": "synthetic-test-key"}), \
             mock.patch.object(drdebug.runtime_checks, "dependency_report", return_value={"checks": []}) as dependency:
            self.assertTrue(drdebug.environment_report()["ready"])
        self.assertNotIn("KIMI_MODEL_API_KEY", dependency.call_args.kwargs["env"])
        self.assertNotIn("OPENAI_API_KEY", dependency.call_args.kwargs["env"])

    def test_timeout_stops_term_ignoring_child_even_when_parent_exits_promptly(self):
        ready, survived = self.base / "child.ready", self.base / "child.survived"
        child = ("import pathlib, signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                 "pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(2); "
                 "pathlib.Path(sys.argv[2]).write_text('survived')")
        parent = ("import subprocess, sys, time; "
                  "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2], sys.argv[3]]); "
                  "time.sleep(60)")
        proc = subprocess.Popen([sys.executable, "-c", parent, child, str(ready), str(survived)],
                                start_new_session=True)
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(ready.exists(), "test child did not initialize")
            with self.assertRaises(drdebug.EngineStopped) as stopped:
                drdebug._wait_engine(proc, 0.1)
            self.assertEqual(stopped.exception.outcome, "active_wall_clock_budget_exhausted")
            time.sleep(2.1)
            self.assertFalse(survived.exists(), "child outlived its engine's wall-clock cap")
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    def test_wait_cleanup_handles_grace_timeout_interruption_and_missing_group(self):
        for first, grace, vanished in ((subprocess.TimeoutExpired("engine", 1),
                                       subprocess.TimeoutExpired("engine", 5), False),
                                      (KeyboardInterrupt(), 0, False),
                                      (subprocess.TimeoutExpired("engine", 1), 0, True)):
            with self.subTest(first=type(first).__name__, vanished=vanished):
                proc = mock.Mock(pid=12345)
                proc.wait.side_effect = [first, grace, 0]
                with mock.patch.object(drdebug.os, "name", "posix"), \
                     mock.patch.object(drdebug.os, "killpg", side_effect=ProcessLookupError() if vanished else None) as kill:
                    with self.assertRaises(drdebug.EngineStopped) as stopped:
                        drdebug._wait_engine(proc, 1)
                self.assertEqual(stopped.exception.returncode, 130 if isinstance(first, KeyboardInterrupt) else 4)
                self.assertEqual(kill.call_args_list, [mock.call(12345, signal.SIGTERM), mock.call(12345, signal.SIGKILL)])
        proc = mock.Mock()
        proc.wait.return_value = 4
        with mock.patch.object(drdebug.os, "killpg") as kill:
            self.assertEqual(drdebug._wait_engine(proc, 1), 4)
        kill.assert_not_called()


    def test_torn_final_event_requires_explicit_repair_and_is_never_rewritten(self):
        self.prepare()
        self.append_events([{"type": "session_started", "entry_failure_type": "precision_failed"}])
        path = self.workspace / "task/.debug_events/events.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"type":"action_started"')
        before = path.read_bytes()
        self.assertTrue(drdebug.status(self.workspace)["session"]["truncated_last_event"])
        with self.assertRaisesRegex(drdebug.ContractError, "Truncated event tail"):
            drdebug.host_command(self.run_args(dry_run=True))
        self.assertEqual(before, path.read_bytes())

    def test_new_execution_mode_is_frozen_and_startup_adapter_is_protected(self):
        self.prepare(determinism="on")
        _, manifest = drdebug.load_workspace(self.workspace)
        self.assertEqual(manifest["schema_version"], 4)
        self.assertEqual(manifest["execution"]["determinism"], "on")
        command = drdebug.host_command(self.run_args(dry_run=True))
        env = drdebug.base_execution_environment(self.workspace, manifest, npu="3", soc="ActualTestSOC")
        self.assertIn(str(self.workspace / "adapter"), env["PYTHONPATH"])
        with self.assertRaises(drdebug.ContractError):
            drdebug.host_command(self.run_args(dry_run=True, determinism="off"))
        (self.workspace / "adapter/sitecustomize.py").write_text("# changed startup policy\n")
        with self.assertRaises(drdebug.ContractError):
            drdebug.load_workspace(self.workspace)

    def test_saved_replay_preflight_does_not_require_model_cli(self):
        with mock.patch.object(drdebug.runtime_checks, "dependency_report", return_value={"ready": True, "checks": []}), \
             mock.patch.object(drdebug.subprocess, "run", side_effect=AssertionError("must not call model CLI")):
            self.assertTrue(drdebug.environment_report()["ready"])


    def test_installed_bundle_resolves_without_repository(self):
        isolated = self.base / "standalone/skills/drdebug"
        (isolated / "scripts").mkdir(parents=True)
        (isolated / "references").mkdir()
        shutil.copytree(LAUNCHER.parent, isolated / "scripts", dirs_exist_ok=True)
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
