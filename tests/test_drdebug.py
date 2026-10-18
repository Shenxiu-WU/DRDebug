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
                  "claude_bin": "claude", "max_attempts": None, "max_turns": None,
                  "deadline_sec": None, "worker_timeout_sec": None, "dry_run": False, "allow_worker_shell": False}
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
        original_frontmatter, separator, original_body = original.partition("\n---\n")
        self.assertTrue(worker.startswith(original_frontmatter + separator))
        self.assertEqual(worker.count(original_body), 1)
        self.assertLess(worker.index("## Prepared workspace resource context"), worker.index(original_body))
        self.assertIn("public_portable_reference_fallback", worker)
        self.assertIn("Phase A", worker)
        self.assertTrue((runtime / ".claude/skills/ascendc-debug/SKILL.md").is_file())
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
        runtime = self.workspace / "runtime"
        (runtime / guard["helper"]).unlink()
        (runtime / guard["verifier"]).write_bytes((drdebug.source_root() / guard["verifier"]).read_bytes())
        manifest["runtime_sha256"] = drdebug.tree_hashes(runtime, exclude_settings=True)
        drdebug.write_json(self.workspace / drdebug.MANIFEST, manifest)
        before = drdebug.tree_hashes(self.workspace)
        status = drdebug.status(self.workspace)
        self.assertEqual(status["adapter_support"]["nonfinite_guard"], "legacy_unchecked")
        self.assertTrue(status["nonfinite_guard"]["requires_new_workspace"])
        self.assertEqual(drdebug.run(self.run_args(dry_run=True))["returncode"], 0)
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
             mock.patch.object(drdebug, "execute_engine", side_effect=AssertionError("must not execute")):
            with self.assertRaisesRegex(drdebug.ContractError, "support changed"):
                drdebug.run(self.run_args())
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

    def test_ast_bridge_is_denied_to_worker_edit_and_write_tools(self):
        self.prepare()
        support = drdebug.absolute_rule(self.workspace / "skills")
        shadow = drdebug.absolute_rule(self.workspace / "task/skills")
        for shell in (False, True):
            config = drdebug.permission_config(self.workspace, allow_shell=shell)["permissions"]
            for tool in ("Edit", "Write"):
                self.assertIn(f"{tool}({support}/**)", config["deny"])
                self.assertIn(f"{tool}({shadow}/**)", config["deny"])
                self.assertFalse(any(rule.startswith(f"{tool}({support}") for rule in config["allow"]))

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
             mock.patch.object(drdebug, "execute_engine", side_effect=AssertionError("must not execute")):
            with self.assertRaisesRegex(drdebug.ContractError, "shadow support"):
                drdebug.run(self.run_args())
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
        self.assertEqual(drdebug.run(self.run_args())["returncode"], 8)
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
        worker = (self.workspace / "runtime/.claude/agents/ascendc-debug-agent-constructive.md").read_text(encoding="utf-8")
        context = worker[:worker.index("# System Prompt")]
        self.assertIn("User references supplied: `true`", context)
        self.assertIn(f"Copied user-reference directory (read-only): `{staged}`", context)
        self.assertIn("At the start of Phase A, use Glob", context)
        self.assertIn("source paths and their applicability", context)
        self.assertIn("do not establish that a repair passes", context)
        self.assertNotIn("{{", worker)
        self.assertIn("workspace-specific read-only reference discovery", manifest["portability_changes"])
        drdebug.load_workspace(self.workspace)

    def test_no_user_reference_is_explicit_in_early_resource_context(self):
        self.prepare()
        staged = self.workspace / "runtime/archive_tasks/user_references"
        self.assertFalse(staged.exists())
        manifest = drdebug.read_json(self.workspace / drdebug.MANIFEST)
        self.assertFalse(manifest["reference_root_supplied"])
        worker = (self.workspace / "runtime/.claude/agents/ascendc-debug-agent-constructive.md").read_text(encoding="utf-8")
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
        worker = (self.workspace / "runtime/.claude/agents/ascendc-debug-agent-constructive.md").read_text(encoding="utf-8")
        context = worker[:worker.index("# System Prompt")]
        self.assertIn("User references supplied: `true`", context)
        self.assertIn("if the directory is empty or no example is relevant", context)
        self.assertIn("Do not assume a matching example exists", context)

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
        worker = (self.workspace / "runtime/.claude/agents/ascendc-debug-agent-constructive.md").read_bytes()
        frontmatter, separator, body = original.partition(b"\n---\n")
        self.assertTrue(worker.startswith(frontmatter + separator))
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
                     runtime / ".claude/agents/ascendc-debug-agent-constructive.md"):
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
        worker = (self.workspace / "runtime/.claude/agents/ascendc-debug-agent-constructive.md").read_text(encoding="utf-8")
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
        with mock.patch.object(drdebug.runtime_checks, "dependency_report", return_value={"ready": False, "checks": [{"name": "python:torch_npu", "ok": False, "detail": "missing"}]}), \
             mock.patch.object(drdebug.shutil, "which", return_value=None), \
             mock.patch.object(drdebug.importlib.util, "find_spec", return_value=None), \
             mock.patch.object(drdebug.subprocess, "run", side_effect=AssertionError("must not execute")):
            report = drdebug.environment_report()
            self.assertFalse(report["ready"])
            self.assertTrue(any(check["name"] == "python:torch_npu" and not check["ok"] for check in report["checks"]))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(drdebug.main(["doctor", "--strict", "--json"]), 2)
                self.assertFalse(json.loads(output.getvalue())["ready"])

    def cli_contract_report(self, cli_run):
        with mock.patch.object(drdebug.runtime_checks, "dependency_report", return_value={"ready": True, "checks": []}), \
             mock.patch.object(drdebug.shutil, "which", side_effect=lambda name: "/test/claude" if name == "claude" else None), \
             mock.patch.object(drdebug.importlib.util, "find_spec", return_value=None), \
             mock.patch.object(drdebug.subprocess, "run", side_effect=cli_run):
            report = drdebug.environment_report()
        return next(check for check in report["checks"] if check["name"] == "claude_cli_contract")

    def test_doctor_accepts_hidden_supported_cli_flags_with_missing_argument_parser_errors(self):
        required = {"--agent", "--setting-sources", "--allowedTools", "--add-dir", "--session-id", "--max-turns"}
        for hidden in (set(), {"--max-turns"}, required):
            with self.subTest(hidden=hidden):
                calls = []

                def cli_run(command, **kwargs):
                    calls.append(command)
                    self.assertNotIn("-p", command)
                    self.assertNotIn("--model", command)
                    self.assertNotIn("--version", command)
                    self.assertFalse(kwargs.get("shell", False))
                    self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
                    if command == ["/test/claude", "--help"]:
                        return subprocess.CompletedProcess(command, 0, stdout=" ".join(required - hidden))
                    self.assertEqual(len(command), 2)
                    self.assertIn(command[1], hidden)
                    return subprocess.CompletedProcess(command, 1, stdout="",
                                                       stderr=f"error: option '{command[1]} <value>' argument missing\n")

                check = self.cli_contract_report(cli_run)
                self.assertTrue(check["ok"])
                self.assertEqual(set(check["detail"]["help_missing_flags"]), hidden)
                self.assertEqual(check["detail"]["missing_flags"], [])
                self.assertEqual({probe["flag"] for probe in check["detail"]["parser_probes"]}, hidden)
                self.assertTrue(all(probe["ok"] for probe in check["detail"]["parser_probes"]))
                self.assertEqual(len(calls), 1 + len(hidden))

    def test_doctor_rejects_unknown_hidden_cli_flag(self):
        help_text = "--agent --setting-sources --allowedTools --add-dir --session-id"
        for returncode, error in ((1, "error: unknown option '--max-turns'\n"),
                                  (1, "error: option '--different <value>' argument missing\n"),
                                  (0, "")):
            with self.subTest(returncode=returncode, error=error):
                calls = []

                def cli_run(command, **kwargs):
                    calls.append(command)
                    if command == ["/test/claude", "--help"]:
                        return subprocess.CompletedProcess(command, 0, stdout=help_text)
                    if "--version" in command:
                        # Real CLI behavior: even an unknown option can be
                        # hidden by the version action's early successful exit.
                        return subprocess.CompletedProcess(command, 0, stdout="2.1.204\n")
                    self.assertEqual(command, ["/test/claude", "--max-turns"])
                    self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
                    return subprocess.CompletedProcess(command, returncode, stdout="", stderr=error)

                check = self.cli_contract_report(cli_run)
                self.assertFalse(check["ok"])
                self.assertEqual(check["detail"]["help_missing_flags"], ["--max-turns"])
                self.assertEqual(check["detail"]["missing_flags"], ["--max-turns"])
                self.assertEqual(check["detail"]["parser_probes"][0]["returncode"], returncode)
                self.assertEqual(len(calls), 2)

    def test_doctor_cli_probe_timeout_is_failure_without_model_execution(self):
        help_text = "--agent --setting-sources --allowedTools --add-dir --session-id"
        for timeout_stage in ("help", "parser"):
            with self.subTest(timeout_stage=timeout_stage):
                calls = []

                def cli_run(command, **kwargs):
                    calls.append(command)
                    if command == ["/test/claude", "--help"] and timeout_stage != "help":
                        return subprocess.CompletedProcess(command, 0, stdout=help_text)
                    self.assertIn(command, (["/test/claude", "--help"],
                                            ["/test/claude", "--max-turns"]))
                    self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
                    raise subprocess.TimeoutExpired(command, kwargs["timeout"])

                check = self.cli_contract_report(cli_run)
                self.assertFalse(check["ok"])
                if timeout_stage == "help":
                    self.assertEqual(check["detail"]["help_error"], "TimeoutExpired")
                    self.assertEqual(check["detail"]["parser_probes"], [])
                    self.assertEqual(len(calls), 1)
                else:
                    self.assertEqual(check["detail"]["missing_flags"], ["--max-turns"])
                    self.assertEqual(check["detail"]["parser_probes"][0]["error"], "TimeoutExpired")
                    self.assertEqual(len(calls), 2)

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
            self.assertAlmostEqual(first_timeout, 86400, delta=1)
            for flag, expected in (("--max-attempts", 5), ("--max-turns", 240),
                                   ("--soft-task-turns", 960), ("--max-task-turns", 1200),
                                   ("--deadline-sec", 86400), ("--agent-timeout-sec", 21600)):
                self.assertAlmostEqual(float(first_command[first_command.index(flag) + 1]), expected, delta=1)
            self.assertEqual(first["status"]["runs"]["budget"], {
                "max_attempts": 5, "max_turns": 240, "deadline_sec": 86400, "worker_timeout_sec": 21600})
            self.append_events([{"type": "session_started", "entry_failure_type": "precision_failed"},
                                {"type": "attempt_started", "attempt": 0, "failure_type": "precision_failed"}])
            self.assertEqual(drdebug.run(self.run_args())["returncode"], 8)
            self.assertLess(execute.call_args.kwargs["timeout"], first_timeout)
        history = drdebug.read_json(self.workspace / drdebug.RUN_STATE)
        self.assertEqual(len(history["runs"]), 2)
        self.assertEqual(drdebug.status(self.workspace)["session"]["attempts_used"], 1)
        state = self.workspace / drdebug.RUN_STATE
        before = state.read_bytes()
        with mock.patch.object(drdebug, "environment_report", side_effect=AssertionError("must not preflight")), \
             mock.patch.object(drdebug, "execute_engine", side_effect=AssertionError("must not execute")):
            for override in ({"max_attempts": 6}, {"max_turns": 241}, {"deadline_sec": 86401},
                             {"worker_timeout_sec": 21601}):
                with self.subTest(override=override), self.assertRaisesRegex(drdebug.ContractError, "Resume retains"):
                    drdebug.run(self.run_args(**override))
        self.assertEqual(state.read_bytes(), before)

    def test_explicit_profile_overrides_reach_engine_and_run_record(self):
        self.prepare()
        argv = ["run", str(self.workspace), "--npu", "3", "--soc", "ActualTestSOC",
                "--max-attempts", "2", "--max-turns", "160", "--deadline-sec", "28800",
                "--worker-timeout-sec", "7200.5"]
        args = drdebug.parser().parse_args(argv + ["--dry-run"])
        preview = drdebug.run(args)
        self.assertEqual(preview["budget"], {"max_attempts": 2, "max_turns": 160,
                                           "deadline_sec": 28800, "worker_timeout_sec": 7200.5})
        self.assertEqual(preview["effective_worker_timeout_sec"], 7200.5)
        self.assertEqual(float(preview["command"][preview["command"].index("--agent-timeout-sec") + 1]), 7200.5)
        self.assertFalse((self.workspace / drdebug.RUN_STATE).exists())
        with mock.patch.object(drdebug, "environment_report", return_value={"ready": True}), \
             mock.patch.object(drdebug, "execute_engine", return_value=8) as execute:
            result = drdebug.run(drdebug.parser().parse_args(argv))
        command = execute.call_args.args[0]
        for flag, expected in (("--max-attempts", 2), ("--max-turns", 160),
                               ("--soft-task-turns", 160), ("--max-task-turns", 320),
                               ("--deadline-sec", 28800), ("--agent-timeout-sec", 7200.5)):
            self.assertAlmostEqual(float(command[command.index(flag) + 1]), expected, delta=1)
        self.assertAlmostEqual(execute.call_args.kwargs["timeout"], 28800, delta=1)
        history = result["status"]["runs"]
        self.assertEqual(history["budget"]["worker_timeout_sec"], 7200.5)
        self.assertEqual(history["runs"][0]["worker_timeout_sec"], 7200.5)
        self.assertEqual(history["runs"][0]["effective_worker_timeout_sec"], 7200.5)

    def test_worker_timeout_is_clamped_to_remaining_total_wall_budget(self):
        self.prepare()
        drdebug.write_json(self.workspace / drdebug.RUN_STATE, {
            "budget": {"max_attempts": 5, "deadline_sec": 30000, "max_turns": 80,
                       "worker_timeout_sec": 21600},
            "runs": [{"started_unix": 0, "elapsed_sec": 28500}]})
        preview = drdebug.run(self.run_args(dry_run=True))
        self.assertEqual(preview["budget"]["worker_timeout_sec"], 21600)
        self.assertEqual(preview["remaining_active_seconds"], 1500)
        self.assertEqual(preview["effective_worker_timeout_sec"], 1500)
        with mock.patch.object(drdebug, "environment_report", return_value={"ready": True}), \
             mock.patch.object(drdebug, "execute_engine", return_value=8) as execute:
            result = drdebug.run(self.run_args())
        command = execute.call_args.args[0]
        self.assertAlmostEqual(float(command[command.index("--agent-timeout-sec") + 1]), 1500, delta=1)
        self.assertAlmostEqual(float(command[command.index("--deadline-sec") + 1]), 1500, delta=1)
        self.assertAlmostEqual(execute.call_args.kwargs["timeout"], 1500, delta=1)
        record = result["status"]["runs"]["runs"][-1]
        self.assertEqual(record["worker_timeout_sec"], 21600)
        self.assertAlmostEqual(record["effective_worker_timeout_sec"], 1500, delta=1)

    def test_worker_timeout_resume_retains_first_budget_and_rejects_changes(self):
        self.prepare()
        with mock.patch.object(drdebug, "environment_report", return_value={"ready": True}), \
             mock.patch.object(drdebug, "execute_engine", return_value=8) as execute:
            drdebug.run(self.run_args(deadline_sec=50000, worker_timeout_sec=21600))
            self.append_events([{"type": "session_started", "entry_failure_type": "precision_failed"},
                                {"type": "attempt_started", "attempt": 0, "failure_type": "precision_failed"}])
            resumed = drdebug.run(self.run_args())
            command = execute.call_args.args[0]
            self.assertEqual(float(command[command.index("--agent-timeout-sec") + 1]), 21600)
            self.assertEqual(execute.call_count, 2)
        self.assertEqual(resumed["status"]["runs"]["budget"]["worker_timeout_sec"], 21600)
        self.assertEqual(resumed["status"]["session"]["attempts_used"], 1)
        self.assertEqual(drdebug.run(self.run_args(dry_run=True, worker_timeout_sec=21600))["budget"]["worker_timeout_sec"], 21600)
        state = self.workspace / drdebug.RUN_STATE
        before = state.read_bytes()
        for changed in (600, 21601):
            with self.subTest(changed=changed), self.assertRaisesRegex(drdebug.ContractError, "Resume retains"):
                drdebug.run(self.run_args(dry_run=True, worker_timeout_sec=changed))
        self.assertEqual(state.read_bytes(), before)

    def test_legacy_budget_keeps_implicit_worker_timeout_of_600(self):
        self.prepare()
        state = self.workspace / drdebug.RUN_STATE
        drdebug.write_json(state, {"budget": {"max_attempts": 5, "deadline_sec": 1800, "max_turns": 80},
                                  "runs": [{"started_unix": 0, "elapsed_sec": 100}]})
        before = state.read_bytes()
        preview = drdebug.run(self.run_args(dry_run=True))
        self.assertEqual(preview["budget"], {"max_attempts": 5, "deadline_sec": 1800,
                                           "max_turns": 80, "worker_timeout_sec": 600})
        for flag, expected in (("--max-turns", 80), ("--deadline-sec", 1700), ("--agent-timeout-sec", 600)):
            self.assertEqual(float(preview["command"][preview["command"].index(flag) + 1]), expected)
        for override in ({"worker_timeout_sec": 21600}, {"max_turns": 240}, {"deadline_sec": 86400}):
            with self.subTest(override=override), self.assertRaisesRegex(drdebug.ContractError, "Resume retains"):
                drdebug.run(self.run_args(dry_run=True, **override))
        self.assertEqual(state.read_bytes(), before)
        with mock.patch.object(drdebug, "environment_report", return_value={"ready": True}), \
             mock.patch.object(drdebug, "execute_engine", return_value=8) as execute:
            drdebug.run(self.run_args(worker_timeout_sec=600))
        command = execute.call_args.args[0]
        self.assertEqual(float(command[command.index("--agent-timeout-sec") + 1]), 600)
        self.assertAlmostEqual(execute.call_args.kwargs["timeout"], 1700, delta=1)
        self.assertEqual(drdebug.read_json(state)["budget"]["worker_timeout_sec"], 600)

    def test_incomplete_legacy_budget_cannot_gain_fresh_profile_defaults(self):
        self.prepare()
        state = self.workspace / drdebug.RUN_STATE
        drdebug.write_json(state, {"budget": {"max_attempts": 3},
                                  "runs": [{"started_unix": 0, "elapsed_sec": 100}]})
        before = state.read_bytes()
        preview = drdebug.run(self.run_args(dry_run=True))
        expected = {"max_attempts": 3, "deadline_sec": 1800, "max_turns": 80, "worker_timeout_sec": 600}
        self.assertEqual(preview["budget"], expected)
        self.assertEqual(preview["remaining_active_seconds"], 1700)
        for override in ({"max_turns": 240}, {"deadline_sec": 86400}, {"worker_timeout_sec": 21600}):
            with self.subTest(override=override), self.assertRaisesRegex(drdebug.ContractError, "Resume retains"):
                drdebug.run(self.run_args(dry_run=True, **override))
        self.assertEqual(state.read_bytes(), before)
        with mock.patch.object(drdebug, "environment_report", return_value={"ready": True}), \
             mock.patch.object(drdebug, "execute_engine", return_value=8) as execute:
            resumed = drdebug.run(self.run_args())
        command = execute.call_args.args[0]
        self.assertEqual(float(command[command.index("--max-turns") + 1]), 80)
        self.assertEqual(float(command[command.index("--agent-timeout-sec") + 1]), 600)
        self.assertAlmostEqual(execute.call_args.kwargs["timeout"], 1700, delta=1)
        self.assertEqual(resumed["status"]["runs"]["budget"], expected)

    def test_worker_timeout_rejects_nonpositive_and_nonfinite_before_spawn(self):
        self.prepare()
        with mock.patch.object(drdebug, "environment_report", side_effect=AssertionError("must not preflight")), \
             mock.patch.object(drdebug, "execute_engine", side_effect=AssertionError("must not execute")):
            for invalid in (0, -1, float("nan"), float("inf"), float("-inf")):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(drdebug.ContractError, "worker_timeout_sec must be finite and positive"):
                    drdebug.run(self.run_args(worker_timeout_sec=invalid))
        self.assertFalse((self.workspace / drdebug.RUN_STATE).exists())

    def test_existing_history_without_budget_never_becomes_a_fresh_task(self):
        self.prepare()
        state = self.workspace / drdebug.RUN_STATE
        for invalid in ({}, {"runs": [{"started_unix": 0, "elapsed_sec": 100}]},
                        {"budget": {}}, {"budget": None}, {"budget": []}):
            with self.subTest(invalid=invalid):
                drdebug.write_json(state, invalid)
                before = state.read_bytes()
                with self.assertRaisesRegex(drdebug.ContractError, "no saved budget"):
                    drdebug.run(self.run_args(dry_run=True))
                self.assertEqual(state.read_bytes(), before)
        state.unlink()
        self.append_events([{"type": "session_started"}])
        with self.assertRaisesRegex(drdebug.ContractError, "no saved budget history"):
            drdebug.run(self.run_args(dry_run=True))

    def test_preflight_cannot_overwrite_another_completed_runs_budget_history(self):
        self.prepare()
        state = self.workspace / drdebug.RUN_STATE
        history = {"budget": {"max_attempts": 5, "deadline_sec": 1800, "max_turns": 80},
                   "runs": [{"started_unix": 0, "elapsed_sec": 100}]}
        drdebug.write_json(state, history)
        settings = self.workspace / "runtime/.claude/settings.json"
        original_settings = settings.read_bytes()

        def another_run_completed(**kwargs):
            history["runs"].append({"started_unix": 100, "elapsed_sec": 1000})
            drdebug.write_json(state, history)
            return {"ready": True}

        with mock.patch.object(drdebug, "environment_report", side_effect=another_run_completed), \
             mock.patch.object(drdebug, "execute_engine") as execute:
            with self.assertRaisesRegex(drdebug.ContractError, "history changed during preflight"):
                drdebug.run(self.run_args(allow_worker_shell=True))
        execute.assert_not_called()
        self.assertEqual(drdebug.read_json(state), history)
        self.assertEqual(settings.read_bytes(), original_settings)
        self.assertFalse((self.workspace / ".drdebug.lock").exists())
        self.assertEqual(drdebug.run(self.run_args(dry_run=True))["remaining_active_seconds"], 700)

    def test_preflight_cannot_start_an_engine_after_session_events_changed(self):
        self.prepare()

        def another_session_started(**kwargs):
            self.append_events([{"type": "session_started"}])
            return {"ready": True}

        with mock.patch.object(drdebug, "environment_report", side_effect=another_session_started), \
             mock.patch.object(drdebug, "execute_engine") as execute:
            with self.assertRaisesRegex(drdebug.ContractError, "history changed during preflight"):
                drdebug.run(self.run_args())
        execute.assert_not_called()
        self.assertEqual(drdebug.event_status(self.workspace / "task")["events"], 1)
        self.assertFalse((self.workspace / drdebug.RUN_STATE).exists())
        self.assertFalse((self.workspace / ".drdebug.lock").exists())

    def test_outer_timeout_and_interruption_remain_distinct_from_engine_exit_code_4(self):
        self.prepare()
        with mock.patch.object(drdebug, "environment_report", return_value={"ready": True}), \
             mock.patch.object(drdebug, "execute_engine", side_effect=[
                 4, drdebug.EngineStopped(4, "active_wall_clock_budget_exhausted"),
                 drdebug.EngineStopped(130, "interrupted")]):
            results = [drdebug.run(self.run_args()) for _ in range(3)]
        self.assertEqual([r["returncode"] for r in results], [4, 4, 130])
        outcomes = ["engine_returned", "active_wall_clock_budget_exhausted", "interrupted"]
        self.assertEqual([r["launcher_outcome"] for r in results], outcomes)
        records = drdebug.read_json(self.workspace / drdebug.RUN_STATE)["runs"]
        self.assertEqual([r["launcher_outcome"] for r in records], outcomes)
        self.assertTrue(all(r["elapsed_sec"] >= 0 and r["finished_at"] for r in records))
        self.assertFalse((self.workspace / ".drdebug.lock").exists())

    @unittest.skipUnless(os.name == "posix", "requires an isolated POSIX process group")
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

    def test_new_execution_mode_is_frozen_and_startup_adapter_is_protected(self):
        self.prepare(determinism="on")
        _, manifest = drdebug.load_workspace(self.workspace)
        self.assertEqual(manifest["schema_version"], 3)
        self.assertEqual(manifest["execution"]["determinism"], "on")
        command = drdebug.run(self.run_args(dry_run=True))
        self.assertIn(str(self.workspace / "adapter"), command["environment"]["PYTHONPATH"])
        with self.assertRaises(drdebug.ContractError):
            drdebug.run(self.run_args(dry_run=True, determinism="off"))
        (self.workspace / "adapter/sitecustomize.py").write_text("# changed startup policy\n")
        with self.assertRaises(drdebug.ContractError):
            drdebug.load_workspace(self.workspace)

    def test_saved_replay_preflight_does_not_require_model_cli(self):
        with mock.patch.object(drdebug.runtime_checks, "dependency_report", return_value={"ready": True, "checks": []}), \
             mock.patch.object(drdebug.subprocess, "run", side_effect=AssertionError("must not call model CLI")):
            self.assertTrue(drdebug.environment_report(require_worker=False)["ready"])

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
