"""CPU regression tests for full-aware checkpoint selection and source binding.

The real frozen best/rollback transaction is imported from an adapted temporary
runtime. Validation subprocesses are replaced by explicit synthetic verifier
outputs; no NPU, compiler or model is invoked.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "skills/drdebug/scripts/checkpoint_adapter.py"
spec = importlib.util.spec_from_file_location("public_checkpoint_adapter_test", SCRIPT)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
guard_spec = importlib.util.spec_from_file_location("public_guard_checkpoint_tests", REPO / "skills/drdebug/scripts/nonfinite_guard.py")
guard_api = importlib.util.module_from_spec(guard_spec)
guard_spec.loader.exec_module(guard_api)
FROZEN = REPO / "papers/2026-dont-regenerate-debug/artifact/code/debug_agent_v5"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class CheckpointAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="public-checkpoint-")
        self.base = Path(self.temp.name).resolve()
        self.workspace = self.base / "workspace"
        self.runtime = self.workspace / "runtime"
        self.task = self.workspace / "task"
        self.engine = self.runtime / adapter.ENGINE_REL
        shutil.copytree(FROZEN / adapter.ENGINE_REL, self.engine)
        self.provenance = adapter.stage_checkpoint_adapter(self.runtime)
        self.prior_modules = {name: module for name, module in sys.modules.items()
                              if name == "engine" or name.startswith("engine.")}
        for name in self.prior_modules:
            del sys.modules[name]
        self.import_path = str(self.engine.parent)
        sys.path.insert(0, self.import_path)
        from engine import best_rollback, validate_runner
        self.br, self.vr = best_rollback, validate_runner
        (self.task / "kernel").mkdir(parents=True)
        (self.task / "kernel/k.cpp").write_text("candidate_0")
        (self.task / "model.py").write_text("raise RuntimeError('never imported')")
        (self.task / "model_new_ascendc.py").write_text("raise RuntimeError('never imported')")
        self.compact_rows = [{"inputs": [{"shape": [index + 1]}]} for index in range(10)]
        self.full_rows = [{"inputs": [{"shape": [index + 100]}]} for index in range(51)]
        self.guard = None
        self.write_contract()

    def tearDown(self):
        sys.path.remove(self.import_path)
        for name in list(sys.modules):
            if name == "engine" or name.startswith("engine."):
                del sys.modules[name]
        sys.modules.update(self.prior_modules)
        self.temp.cleanup()

    def write_contract(self):
        (self.task / "Cumsum.json").write_text("".join(json.dumps(row) + "\n" for row in self.compact_rows))
        (self.task / "Cumsum.json.bak").write_text("".join(json.dumps(row) + "\n" for row in self.full_rows))
        protected = {name: adapter._sha(self.task / name) for name in
                     ("model.py", "model_new_ascendc.py", "Cumsum.json", "Cumsum.json.bak")}
        manifest = {
            "schema_version": 3, "workspace_path": str(self.workspace),
            "task_contract": {"compact_file": "Cumsum.json", "full_file": "Cumsum.json.bak",
                              "compact_cases": len(self.compact_rows), "full_cases": len(self.full_rows),
                              "protected_sha256": protected}}
        if self.guard is not None:
            manifest["nonfinite_guard"] = self.guard
        dump(self.workspace / "drdebug-workspace.json", manifest)

    def enable_guard(self):
        (self.runtime / "utils").mkdir()
        shutil.copy2(FROZEN / "utils/verification_ascendc.py", self.runtime / "utils/verification_ascendc.py")
        self.guard = guard_api.stage_guard(self.runtime)
        self.write_contract()

    def guard_output(self, path, success, total, mode):
        if self.guard is None or mode == "missing":
            return
        receipt = {"schema_version": 1, "policy": self.guard["policy"], "helper_sha256": self.guard["helper_sha256"],
                   "entrypoint": "cli", "expected_tensor_leaves": total, "checked_tensor_leaves": total,
                   "unpaired_tensor_leaves": 0, "structural_rejections": 0,
                   "nonfinite_mismatches": 0, "guard_errors": 0, "coverage_complete": True, "passed": success}
        changes = {"wrong_helper": ("helper_sha256", "0" * 64), "wrong_entry": ("entrypoint", "import"),
                   "incomplete": ("checked_tensor_leaves", max(0, total - 1)),
                   "guard_error": ("guard_errors", 1), "wrong_outcome": ("passed", not success)}
        if mode in changes:
            key, value = changes[mode]; receipt[key] = value
        if mode == "structural":
            receipt.update(checked_tensor_leaves=total - 1, unpaired_tensor_leaves=1,
                           structural_rejections=1)
        line = guard_api.MARKER + json.dumps(receipt) + "\n"
        with path.open("a") as stream:
            stream.write(line * (2 if mode == "duplicate" else 1))

    def objective(self, attempt, *, compact=10, full=50, compact_total=None,
                  full_total=None, full_crashed=False, include_full=True,
                  duplicate_index=False, source_mutation=False, full_digest=None,
                  returncode=0, compact_guard="valid", full_guard="valid"):
        expected_compact, expected_full = len(self.compact_rows), len(self.full_rows)
        compact_total = expected_compact if compact_total is None else compact_total
        full_total = expected_full if full_total is None else full_total

        def original(task_dir, *, attempt, **kwargs):
            task = Path(task_dir)
            logs = task / ".verify_logs"
            logs.mkdir(exist_ok=True)
            out = logs / f"phase8_attempt{attempt}.stdout"
            err = logs / f"phase8_attempt{attempt}.stderr"
            out.write_text("".join(f"case[{index}]: output[{index}]: " +
                                   ("matched" if index < compact else "value mismatch") + "\n"
                                   for index in range(compact_total)))
            self.guard_output(out, compact == expected_compact, compact_total, compact_guard)
            err.write_text("")
            result = {"attempt": attempt, "source": "engine_objective_validate",
                      "case_pass_rate": 100 * compact / expected_compact,
                      "passed_cases": compact, "total_cases": compact_total,
                      "match_rate": "100.00" if compact == expected_compact else "90.00",
                      "correctness_passed": compact == expected_compact,
                      "stdout_path": str(out), "stderr_path": str(err)}
            if include_full:
                full_out = logs / f"phase8_attempt{attempt}_full.stdout"
                full_out.write_text("".join(f"case[{0 if duplicate_index else index}]: output[{index}]: " +
                                            ("matched" if index < full else "value mismatch") + "\n"
                                            for index in range(full_total)))
                self.guard_output(full_out, full == expected_full, full_total, full_guard)
                full_result = {"ran": True, "coverage_equivalent": False,
                               "active_json": "Cumsum.json", "full_json_source": "Cumsum.json.bak",
                               "active_json_cases": expected_compact, "full_json_cases": expected_full,
                               "active_case_set_sha256": adapter._case_digest(task / "Cumsum.json"),
                               "full_case_set_sha256": full_digest or adapter._case_digest(task / "Cumsum.json.bak"),
                               "passed_cases": full, "total_cases": full_total,
                               "correctness_passed": full == expected_full,
                               "match_rate": 100 * full / expected_full,
                               "crashed": full_crashed, "stdout_path": str(full_out)}
                result["full_eval"] = full_result
                dump(task / "precision_tuning" / f"validation_result_attempt_{attempt}_full.json", full_result)
            dump(task / "precision_tuning" / f"validation_result_attempt_{attempt}.json", result)
            if source_mutation:
                (task / "kernel/k.cpp").write_text("mutated_during_validation")
            return {"build_exit_code": returncode, "verification_ran": returncode == 0,
                    "verification_exit_code": 0 if compact == expected_compact else 1,
                    "classify_exit_code": 0}

        adapter.install_validation_adapter(original)(self.task, attempt=attempt)

    def promote(self, attempt, **kwargs):
        self.objective(attempt, **kwargs)
        return self.br.ensure_current_best(self.task, attempt)

    def staged_origin(self):
        # Like the public example, this second top-level JSON prevents the
        # frozen legacy model.json alias helper from creating tuning early.
        origin = self.task / "example-origin.json"
        dump(origin, {"mode": "replay", "test": "CPU control flow only"})
        manifest = json.loads((self.workspace / "drdebug-workspace.json").read_text())
        manifest["task_contract"]["protected_sha256"][origin.name] = adapter._sha(origin)
        dump(self.workspace / "drdebug-workspace.json", manifest)

    def frozen_objective(self, *, adapted=True, full_passed=None, delete_full=False):
        """Real frozen objective/full writers, only external commands mocked."""
        original = self.vr.run_objective_validation.__wrapped__
        target = self.vr.run_objective_validation if adapted else original
        if delete_full:
            def deleted_sidecar(task_dir, **kwargs):
                result = original(task_dir, **kwargs)
                (Path(task_dir) / "precision_tuning/validation_result_attempt_-1_full.json").unlink()
                return result
            target = adapter.install_validation_adapter(deleted_sidecar)

        def logged(command, *, stdout_path, stderr_path, title, **kwargs):
            stderr_path.touch(exist_ok=True)
            if title == "build_ascendc":
                stdout_path.touch(exist_ok=True)
                return 0
            count = len(self.full_rows) if title == "verification_ascendc_full" else len(self.compact_rows)
            passed = (full_passed if full_passed is not None else count) if title == "verification_ascendc_full" else count
            with stdout_path.open("a") as stream:
                stream.write("".join(f"case[{i}]: output[{i}]: " + ("matched" if i < passed else "value mismatch") + "\n"
                                     for i in range(count)))
            self.guard_output(stdout_path, passed == count, count, "valid")
            return 0 if passed == count else 1

        classifier = subprocess.CompletedProcess([], 0, stdout='{"failure_type":"precision_passed"}', stderr="")
        with mock.patch.object(self.vr, "_run_logged", side_effect=logged), \
             mock.patch.object(self.vr.subprocess, "run", return_value=classifier), \
             mock.patch.dict(self.vr.os.environ, {"ABLATE_FULL_EVAL": "0", "ASCENDC_ABLATE_FULL_EVAL": "0"}):
            return target(self.task, attempt=-1, repo_root=self.runtime)

    def test_frozen_fresh_baseline_reproduces_missing_full_sidecar_without_adapter(self):
        self.enable_guard(); self.staged_origin()
        self.assertFalse((self.task / "precision_tuning").exists())
        self.frozen_objective(adapted=False)
        data = json.loads((self.task / "precision_tuning/validation_result_attempt_-1.json").read_text())
        self.assertTrue(data["full_eval"]["ran"])
        self.assertTrue(data["full_eval"]["correctness_passed"])
        self.assertTrue((self.task / "precision_tuning/.full_eval_state.json").is_file())
        self.assertFalse((self.task / "precision_tuning/validation_result_attempt_-1_full.json").exists())
        with self.assertRaises(FileNotFoundError):
            adapter._result_files(self.task, -1, data)

    def test_fresh_baseline_preserves_original_full_sidecar_and_bound_acceptance(self):
        self.enable_guard(); self.staged_origin()
        self.assertFalse((self.task / "precision_tuning").exists())
        self.frozen_objective()
        data = json.loads((self.task / "precision_tuning/validation_result_attempt_-1.json").read_text())
        full = json.loads((self.task / "precision_tuning/validation_result_attempt_-1_full.json").read_text())
        self.assertEqual(full, data["full_eval"])
        self.assertTrue(full["ran"]); self.assertEqual(full["passed_cases"], 51)
        pointer_dir = self.task / adapter._IDENTITY_DIR / "attempt_-1"
        pointer = json.loads((pointer_dir / "latest.json").read_text())
        receipt = json.loads((pointer_dir / pointer["path"]).read_text())
        self.assertEqual(len(receipt["result_files_sha256"]), 5)
        self.assertEqual(receipt["before"], receipt["after"])
        self.assertTrue(self.br.ensure_current_best(self.task, -1, force=True)["updated"])
        accepted = adapter.acceptance_status(self.task)
        self.assertTrue(accepted["ready"], accepted)
        self.assertEqual(accepted["best_attempt"], -1)
        self.assertEqual(accepted["full"], {"passed": 51, "total": 51})

    def test_fresh_equivalent_full_coverage_keeps_original_sidecar_and_no_extra_run(self):
        self.enable_guard(); self.full_rows = list(self.compact_rows); self.write_contract(); self.staged_origin()
        self.frozen_objective()
        full = json.loads((self.task / "precision_tuning/validation_result_attempt_-1_full.json").read_text())
        self.assertTrue(full["coverage_equivalent"]); self.assertFalse(full["ran"])
        self.assertFalse((self.task / ".verify_logs/phase8_attempt-1_full.stdout").exists())
        self.assertTrue(self.br.ensure_current_best(self.task, -1, force=True)["updated"])
        accepted = adapter.acceptance_status(self.task)
        self.assertTrue(accepted["ready"], accepted)
        self.assertTrue(accepted["nonfinite_guard"]["full_coverage_equivalent"])

    def test_fresh_full_numerical_failure_remains_complete_but_unaccepted(self):
        self.enable_guard(); self.staged_origin(); self.frozen_objective(full_passed=50)
        self.assertTrue(self.br.ensure_current_best(self.task, -1, force=True)["updated"])
        metric = self.br.read_best_metric(self.task)["public_checkpoint"]
        self.assertTrue(metric["full_complete"]); self.assertEqual(metric["full_passed"], 50)
        self.assertFalse(metric["accepted"]); self.assertFalse(adapter.acceptance_status(self.task)["ready"])

    def test_missing_original_full_sidecar_still_fails_closed_after_directory_fix(self):
        self.enable_guard(); self.staged_origin()
        with self.assertRaises(FileNotFoundError):
            self.frozen_objective(delete_full=True)
        self.assertIsNone(self.br.read_attempt_metric(self.task, -1))
        self.assertFalse(adapter.acceptance_status(self.task)["ready"])

    def test_archive_is_unchanged_and_staging_is_audited(self):
        self.assertEqual(self.provenance["policy"], "completed_full_coverage_then_compact")
        self.assertFalse(self.provenance["frozen_archive_modified"])
        for item in self.provenance["modules"]:
            name = Path(item["path"]).name
            self.assertEqual(adapter._sha(FROZEN / item["path"]), adapter.ORIGINAL_SHA256[name])
            self.assertEqual(adapter._sha(self.runtime / item["path"]), item["adapted_sha256"])
            self.assertNotEqual(item["original_sha256"], item["adapted_sha256"])
        with self.assertRaisesRegex(adapter.CheckpointAdapterError, "already staged"):
            adapter.stage_checkpoint_adapter(self.runtime)

    def test_unsupported_frozen_module_fails_without_staging(self):
        second = self.base / "unsupported"
        shutil.copytree(FROZEN / adapter.ENGINE_REL, second / adapter.ENGINE_REL)
        (second / adapter.ENGINE_REL / "best_rollback.py").write_text("# unsupported")
        with self.assertRaisesRegex(adapter.CheckpointAdapterError, "Unsupported"):
            adapter.stage_checkpoint_adapter(second)
        self.assertFalse((second / adapter.ENGINE_REL / "_public_checkpoint_adapter.py").exists())

    def test_compact_tie_full_50_to_51_promotes_actual_source(self):
        self.assertTrue(self.promote(0, full=50)["updated"])
        self.assertFalse(self.br.read_best_metric(self.task)["public_checkpoint"]["accepted"])
        (self.task / "kernel/k.cpp").write_text("candidate_full_pass")
        self.assertTrue(self.promote(1, full=51)["updated"])
        best = self.br.read_best_metric(self.task)
        self.assertEqual(best["attempt"], 1)
        self.assertTrue(best["public_checkpoint"]["accepted"])
        source = self.task / self.br._BEST_DIR / "src/k.cpp"
        self.assertEqual(source.read_text(), "candidate_full_pass")
        self.assertEqual(best["public_checkpoint"]["sources"], adapter._sources(source.parent))

    def test_guarded_compact_tie_full_improvement_and_final_acceptance(self):
        self.enable_guard()
        self.assertTrue(self.promote(0, full=50)["updated"])
        metric = self.br.read_best_metric(self.task)
        self.assertFalse(metric["public_checkpoint"]["nonfinite_guard"]["full"]["passed"])
        self.assertFalse(adapter.acceptance_status(self.task)["ready"])
        (self.task / "kernel/k.cpp").write_text("new_guarded_candidate")
        self.assertTrue(self.promote(1, full=51)["updated"])
        accepted = adapter.acceptance_status(self.task)
        self.assertTrue(accepted["ready"])
        self.assertTrue(accepted["nonfinite_guard"]["required"])
        self.assertEqual(accepted["nonfinite_guard"]["helper_sha256"], self.guard["helper_sha256"])
        self.assertTrue(accepted["nonfinite_guard"]["compact"]["passed"])
        self.assertTrue(accepted["nonfinite_guard"]["full"]["passed"])

    def test_guarded_objective_requires_unique_complete_matching_cli_receipts(self):
        self.enable_guard()
        for number, mode in enumerate(("missing", "duplicate", "wrong_helper", "wrong_entry", "incomplete", "guard_error", "wrong_outcome")):
            with self.subTest(mode=mode):
                self.objective(number, full=51, compact_guard=mode)
                self.assertIsNone(self.br.read_attempt_metric(self.task, number))
                self.assertFalse(self.br.ensure_current_best(self.task, number)["success"])

    def test_structural_failure_is_recorded_as_failed_objective_and_can_be_repaired(self):
        self.enable_guard()
        self.objective(0, compact=9, full=50, compact_guard="structural", full_guard="structural")
        metric = self.br.read_attempt_metric(self.task, 0)
        self.assertIsNotNone(metric)
        self.assertTrue(metric["public_checkpoint"]["compact_complete"])
        self.assertTrue(metric["public_checkpoint"]["full_complete"])
        self.assertFalse(metric["public_checkpoint"]["accepted"])
        self.assertEqual(metric["public_checkpoint"]["nonfinite_guard"]["compact"]["structural_rejections"], 1)
        self.assertTrue(self.br.ensure_current_best(self.task, 0)["updated"])
        self.assertFalse(adapter.acceptance_status(self.task)["ready"])
        (self.task / "kernel/k.cpp").write_text("repaired_structural_failure")
        self.assertTrue(self.promote(1, full=51)["updated"])
        self.assertTrue(adapter.acceptance_status(self.task)["ready"])
        # Structural-rejection metadata can never accompany an all-pass result.
        self.objective(2, full=51, compact_guard="structural")
        self.assertIsNone(self.br.read_attempt_metric(self.task, 2))

    def test_full_guard_cannot_be_replaced_by_valid_compact_guard(self):
        self.enable_guard()
        for number, mode in enumerate(("missing", "duplicate", "wrong_helper", "wrong_entry", "incomplete", "guard_error")):
            with self.subTest(mode=mode):
                self.objective(number, full=51, full_guard=mode)
                self.assertIsNone(self.br.read_attempt_metric(self.task, number))
                self.assertFalse(adapter.acceptance_status(self.task)["ready"])

    def test_old_receipt_cannot_become_guarded_by_adding_only_manifest_metadata(self):
        self.assertTrue(self.promote(0, full=51)["updated"])
        self.enable_guard()
        self.assertIsNone(self.br.read_attempt_metric(self.task, 0))
        self.assertFalse(adapter.acceptance_status(self.task)["ready"])

    def test_full_regression_cannot_replace_best_despite_compact_improvement(self):
        self.assertTrue(self.promote(0, compact=9, full=50)["updated"])
        (self.task / "kernel/k.cpp").write_text("compact_improved_full_regressed")
        self.assertFalse(self.promote(1, compact=10, full=49)["updated"])
        self.assertEqual(self.br.read_best_metric(self.task)["attempt"], 0)

    def test_full_missing_or_partial_cannot_displace_complete_full_best(self):
        self.assertTrue(self.promote(0)["updated"])
        for attempt, kwargs in enumerate(({"include_full": False}, {"full_total": 20, "full": 20},
                                          {"full_crashed": True}, {"duplicate_index": True},
                                          {"full_digest": "0" * 64}), 1):
            self.assertFalse(self.promote(attempt, **kwargs)["updated"], kwargs)
            self.assertEqual(self.br.read_best_metric(self.task)["attempt"], 0)

    def test_compact_only_fallback_still_keeps_near_miss_progress(self):
        self.assertTrue(self.promote(0, compact=7, include_full=False)["updated"])
        self.assertTrue(self.promote(1, compact=9, include_full=False)["updated"])
        metric = self.br.read_best_metric(self.task)
        self.assertEqual(metric["case_pass_rate"], 90)
        self.assertFalse(metric["public_checkpoint"]["accepted"])

    def test_incomplete_compact_is_not_checkpointed(self):
        result = self.promote(0, compact=5, compact_total=5)
        self.assertTrue(result["success"])
        self.assertFalse(result["updated"])
        self.assertFalse((self.task / self.br._BEST_DIR).exists())

    def test_build_failure_is_not_checkpointed(self):
        result = self.promote(0, returncode=1)
        self.assertTrue(result["success"])
        self.assertFalse(result["updated"])

    def test_changed_sources_during_or_after_validation_cannot_promote(self):
        self.objective(0, source_mutation=True)
        self.assertFalse(self.br.ensure_current_best(self.task, 0)["success"])
        self.objective(1)
        (self.task / "kernel/k.cpp").write_text("changed_after_validation")
        self.assertEqual(self.br.ensure_current_best(self.task, 1)["error"], "public_objective_source_changed")

    def test_missing_binding_cannot_accept_caller_metric(self):
        result = self.br.ensure_current_best(self.task, 0, {"case_pass_rate": 100, "match_rate": 100}, force=True)
        self.assertFalse(result["success"])
        self.assertFalse((self.task / self.br._BEST_DIR).exists())

    def test_immutable_evidence_not_mutable_raw_log_is_the_resume_authority(self):
        self.objective(0)
        log = self.task / ".verify_logs/phase8_attempt0.stdout"
        log.write_text(log.read_text() + "extra text")
        self.assertIsNotNone(self.br.read_attempt_metric(self.task, 0))
        directory = self.task / adapter._IDENTITY_DIR / "attempt_0"
        pointer = json.loads((directory / "latest.json").read_text())
        evidence = directory / pointer["path"][:-5] / ".verify_logs/phase8_attempt0.stdout"
        evidence.write_text(evidence.read_text() + "changed frozen evidence")
        self.assertIsNone(self.br.read_attempt_metric(self.task, 0))
        self.objective(1)
        (self.task / "model.py").write_text("changed")
        self.assertIsNone(self.br.read_attempt_metric(self.task, 1))

    def test_retries_keep_previous_identity_receipts(self):
        self.objective(0)
        first = list((self.task / adapter._IDENTITY_DIR / "attempt_0").glob("*.json"))
        self.objective(0)
        second = list((self.task / adapter._IDENTITY_DIR / "attempt_0").glob("*.json"))
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 3)

    def test_same_attempt_retry_preserves_previous_best_and_its_bound_artifacts(self):
        self.assertTrue(self.promote(0, full=51)["updated"])
        old_metric = self.br.read_best_metric(self.task)
        (self.task / "kernel/k.cpp").write_text("worse_retry")
        self.assertFalse(self.promote(0, full=49)["updated"])
        self.assertEqual(self.br.read_best_metric(self.task), old_metric)
        self.assertEqual(self.br.do_rollback(self.task), old_metric)
        self.assertEqual((self.task / "kernel/k.cpp").read_text(), "candidate_0")

    def test_full_file_alias_in_contract_is_ranked_by_actual_equivalent_coverage(self):
        alias = self.task / "Cumsum.json.full"
        alias.write_bytes((self.task / "Cumsum.json.bak").read_bytes())
        manifest_path = self.workspace / "drdebug-workspace.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["task_contract"]["full_files"] = ["Cumsum.json.bak", "Cumsum.json.full"]
        manifest["task_contract"]["protected_sha256"][alias.name] = adapter._sha(alias)
        dump(manifest_path, manifest)
        self.objective(0, full=51)
        # Recreate a genuine identity receipt whose real full output selected
        # the other protected alias, as the frozen filename tie-break does.
        def alias_objective(task_dir, *, attempt):
            raw = self.task / "precision_tuning/validation_result_attempt_0.json"
            data = json.loads(raw.read_text())
            data["attempt"] = attempt
            data["full_eval"]["full_json_source"] = alias.name
            dump(self.task / "precision_tuning" / f"validation_result_attempt_{attempt}.json", data)
            dump(self.task / "precision_tuning" / f"validation_result_attempt_{attempt}_full.json", data["full_eval"])
            return {"build_exit_code": 0, "verification_ran": True,
                    "verification_exit_code": 0, "classify_exit_code": 0}
        adapter.install_validation_adapter(alias_objective)(self.task, attempt=1)
        self.assertTrue(self.br.ensure_current_best(self.task, 1)["updated"])
        self.assertTrue(self.br.read_best_metric(self.task)["public_checkpoint"]["full_complete"])

    def test_changed_checkpoint_metric_cannot_be_recovered_as_valid(self):
        self.assertTrue(self.promote(0, full=50)["updated"])
        metric_path = self.task / self.br._BEST_DIR / "metric.json"
        metric = json.loads(metric_path.read_text())
        metric["public_checkpoint"]["full_passed"] = 51
        metric["public_checkpoint"]["accepted"] = True
        dump(metric_path, metric)
        self.assertFalse(self.br.checkpoint_is_valid(self.task))
        self.assertIsNone(self.br.read_best_metric(self.task))

    def test_two_full_regressions_trigger_original_rollback_policy(self):
        self.assertTrue(self.promote(0, full=50)["updated"])
        (self.task / "kernel/k.cpp").write_text("bad_1")
        self.promote(1, full=49)
        self.assertFalse(self.br.should_rollback(self.task, 1))
        (self.task / "kernel/k.cpp").write_text("bad_2")
        self.promote(2, full=48)
        self.assertTrue(self.br.should_rollback(self.task, 2))
        self.assertEqual(self.br.do_rollback(self.task)["attempt"], 0)
        self.assertEqual((self.task / "kernel/k.cpp").read_text(), "candidate_0")

    def test_checkpoint_source_race_is_caught_by_original_atomic_verification(self):
        self.assertTrue(self.promote(0, full=50)["updated"])
        (self.task / "kernel/k.cpp").write_text("new_candidate")
        self.objective(1, full=51)
        original_copy = self.br._copy_source_tree

        def racing_copy(source, target):
            (self.task / "kernel/k.cpp").write_text("race_mutation")
            return original_copy(source, target)

        with mock.patch.object(self.br, "_copy_source_tree", side_effect=racing_copy):
            result = self.br.ensure_current_best(self.task, 1)
        self.assertFalse(result["success"])
        self.assertEqual(self.br.read_best_metric(self.task)["attempt"], 0)

    def test_checkpoint_directory_switch_crash_recovers_previous_source(self):
        self.assertTrue(self.promote(0, full=50)["updated"])
        (self.task / "kernel/k.cpp").write_text("new_candidate")
        self.objective(1, full=51)
        original_replace = self.br.os.replace

        def fail_activation(source, target):
            if Path(source).name.startswith(self.br._BEST_TMP_PREFIX) and Path(target).name == "current_best":
                raise OSError("injected switch failure")
            return original_replace(source, target)

        with mock.patch.object(self.br.os, "replace", side_effect=fail_activation):
            result = self.br.ensure_current_best(self.task, 1)
        self.assertFalse(result["success"])
        self.assertEqual(self.br.read_best_metric(self.task)["attempt"], 0)
        self.assertEqual((self.task / self.br._BEST_DIR / "src/k.cpp").read_text(), "candidate_0")

    def test_rollback_journal_replays_without_rebinding_old_metrics(self):
        self.assertTrue(self.promote(0)["updated"])
        (self.task / "kernel/k.cpp").write_text("bad")
        with mock.patch.object(self.br, "_write_file_atomically", side_effect=OSError("interrupt")):
            self.assertIsNone(self.br.do_rollback(self.task))
        self.assertTrue((self.task / self.br._ROLLBACK_TXN / "journal.json").exists())
        self.assertEqual(self.br.do_rollback(self.task)["attempt"], 0)
        self.assertEqual((self.task / "kernel/k.cpp").read_text(), "candidate_0")

    def test_failed_validation_keeps_an_unaccepted_identity_receipt(self):
        def failing(task_dir, *, attempt):
            raise RuntimeError("validation interrupted")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            adapter.install_validation_adapter(failing)(self.task, attempt=0)
        receipts = list((self.task / adapter._IDENTITY_DIR / "attempt_0").glob("*.json"))
        self.assertEqual(len(receipts), 2)
        self.assertIsNone(self.br.read_attempt_metric(self.task, 0))

    def test_read_only_acceptance_requires_all_cases_and_same_active_source(self):
        self.assertFalse(adapter.acceptance_status(self.task)["ready"])
        self.assertTrue(self.promote(0, full=50)["updated"])
        self.assertFalse(adapter.acceptance_status(self.task)["ready"])
        (self.task / "kernel/k.cpp").write_text("all_pass_candidate")
        self.assertTrue(self.promote(1, full=51)["updated"])
        status = adapter.acceptance_status(self.task)
        self.assertTrue(status["ready"], status)
        self.assertEqual(status["full"], {"passed": 51, "total": 51})
        self.assertEqual(status["best_attempt"], 1)
        (self.task / "kernel/k.cpp").write_text("changed_after_success")
        status = adapter.acceptance_status(self.task)
        self.assertFalse(status["ready"])
        self.assertIn("active_source_differs", status["error"])

    def test_read_only_acceptance_rejects_legacy_stale_or_forged_best(self):
        best = self.task / self.br._BEST_DIR
        (best / "src").mkdir(parents=True)
        shutil.copy2(self.task / "kernel/k.cpp", best / "src/k.cpp")
        sources = adapter._sources(best / "src")
        dump(best / "manifest.json", {"schema_version": 1, "complete": True, "attempt": 0, "files": sources})
        dump(best / "metric.json", {"attempt": 0, "case_pass_rate": 100, "match_rate": 100,
                                    "correctness_passed": True})
        status = adapter.acceptance_status(self.task)
        self.assertFalse(status["ready"])
        self.assertIn("missing_source_bound", status["error"])
        shutil.rmtree(best)
        self.assertTrue(self.promote(0, full=51)["updated"])
        metric = json.loads((best / "metric.json").read_text())
        metric["public_checkpoint"]["full_passed"] = 50
        dump(best / "metric.json", metric)
        self.assertFalse(adapter.acceptance_status(self.task)["ready"])

    def test_read_only_acceptance_preserves_pending_recovery_evidence(self):
        self.assertTrue(self.promote(0, full=51)["updated"])
        best = self.task / self.br._BEST_DIR
        previous = self.task / self.br._BEST_PREVIOUS
        best.rename(previous)
        before = {path.relative_to(self.task).as_posix(): adapter._sha(path)
                  for path in self.task.rglob("*") if path.is_file()}
        status = adapter.acceptance_status(self.task)
        after = {path.relative_to(self.task).as_posix(): adapter._sha(path)
                 for path in self.task.rglob("*") if path.is_file()}
        self.assertFalse(status["ready"])
        self.assertTrue(previous.exists())
        self.assertFalse(best.exists())
        self.assertEqual(before, after)

    def test_symlinked_evidence_directory_is_rejected_before_execution(self):
        outside = self.base / "outside"
        outside.mkdir()
        (self.task / "precision_tuning").symlink_to(outside, target_is_directory=True)
        def must_not_execute(task_dir, *, attempt):
            raise AssertionError("must not execute through a linked output directory")
        with self.assertRaisesRegex(adapter.CheckpointAdapterError, "Symlinked"):
            adapter.install_validation_adapter(must_not_execute)(self.task, attempt=0)
        self.assertEqual(list(outside.iterdir()), [])

    def test_symlinked_immutable_artifact_is_not_a_valid_checkpoint(self):
        self.assertTrue(self.promote(0, full=50)["updated"])
        directory = self.task / adapter._IDENTITY_DIR / "attempt_0"
        pointer = json.loads((directory / "latest.json").read_text())
        evidence = directory / pointer["path"][:-5] / ".verify_logs/phase8_attempt0.stdout"
        external = self.base / "external.stdout"
        external.write_bytes(evidence.read_bytes())
        evidence.unlink()
        evidence.symlink_to(external)
        self.assertFalse(self.br.checkpoint_is_valid(self.task))

    def test_unknown_ast_status_cannot_checkpoint(self):
        from engine.gate_adapter import GateResult
        for checks in ({}, {"anticheat_pass": True, "cpp_regression_pass": True,
                            "ast_degrade_pass": False, "ast_validator_errored": True},
                       {"anticheat_pass": True, "cpp_regression_pass": True, "ast_degrade_pass": True,
                        "cpp_checker_present": True, "ast_validator_present": True, "cpp_checker_errored": True},
                       {"anticheat_pass": True, "cpp_regression_pass": True, "ast_degrade_pass": True,
                        "cpp_checker_present": True, "ast_validator_present": True, "ast_validator_errored": True}):
            self.assertFalse(self.br.checkpoint_allowed(GateResult(gate="GATE-V", passed=False, checks=checks)))
        self.assertTrue(self.br.checkpoint_allowed(GateResult(gate="GATE-V", passed=False, checks={
            "anticheat_pass": True, "cpp_regression_pass": True, "ast_degrade_pass": True,
            "cpp_checker_present": True, "ast_validator_present": True})))

    def test_equivalent_sets_use_actual_compact_coverage(self):
        self.full_rows = self.compact_rows
        self.write_contract()
        original = adapter.install_validation_adapter

        def equivalent(task_dir, *, attempt):
            # Emit the same original engine schema for an equivalence no-op.
            self.objective(99, full=10, full_total=10)
            source = self.task / "precision_tuning/validation_result_attempt_99.json"
            data = json.loads(source.read_text())
            data["attempt"] = attempt
            full = data["full_eval"]
            for key in ("passed_cases", "total_cases", "correctness_passed", "match_rate", "crashed", "stdout_path"):
                full.pop(key)
            full.update({"ran": False, "coverage_equivalent": True})
            dump(self.task / "precision_tuning" / f"validation_result_attempt_{attempt}_full.json", full)
            dump(self.task / "precision_tuning" / f"validation_result_attempt_{attempt}.json", data)
            return {"build_exit_code": 0, "verification_ran": True,
                    "verification_exit_code": 0, "classify_exit_code": 0}

        original(equivalent)(self.task, attempt=0)
        self.assertTrue(self.br.ensure_current_best(self.task, 0)["updated"])
        meta = self.br.read_best_metric(self.task)["public_checkpoint"]
        self.assertTrue(meta["full_complete"])
        self.assertTrue(meta["accepted"])


if __name__ == "__main__":
    unittest.main()
