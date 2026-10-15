from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "analyze_v5_ablation.py"


def _load():
    spec = importlib.util.spec_from_file_location("v5_analysis", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V5AnalysisTests(unittest.TestCase):
    @staticmethod
    def _write_terminal_artifacts(task: Path) -> None:
        (task / "debug_status.json").write_text(json.dumps({
            "session_outcome": "success",
            "ended_at": "2026-07-25T00:00:00Z",
            "objective_success": True,
            "reportable_success": True,
            "anti_cheat_pass": True,
            "ast_degrade_pass": True,
        }))
        (task / "run_summary.json").write_text(json.dumps({
            "schema_version": 1,
            "session_outcome": "success",
            "attempts_used": 1,
            "turns": 4,
            "gate": {},
            "forensics": {},
        }))

    @staticmethod
    def _write_claude_result(tuning: Path) -> None:
        archive = tuning / "claude_results"
        archive.mkdir(parents=True, exist_ok=True)
        (archive / "attempt0.json").write_text(json.dumps({
            "num_turns": 4,
            "total_cost_usd": 0.25,
            "modelUsage": {
                "qwen3.8-max-preview": {
                    "inputTokens": 100,
                    "outputTokens": 10,
                    "cacheCreationInputTokens": 0,
                    "cacheReadInputTokens": 0,
                    "contextWindow": 1000000,
                    "maxOutputTokens": 65536,
                },
            },
        }))

    @staticmethod
    def _write_events(task: Path, rows: list[dict]) -> None:
        path = task / ".debug_events" / "events.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_append_only_results_avoid_latest_file_double_count(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp)
            archive = task / "precision_tuning" / "claude_results"
            archive.mkdir(parents=True)
            payload = {
                "num_turns": 12,
                "total_cost_usd": 1.25,
                "modelUsage": {
                    "qwen3.8-max-preview": {
                        "inputTokens": 100,
                        "outputTokens": 20,
                        "cacheCreationInputTokens": 30,
                        "cacheReadInputTokens": 40,
                        "costUSD": 99,
                        "contextWindow": 1000000,
                        "maxOutputTokens": 65536,
                    }
                },
            }
            (archive / "attempt0_session.json").write_text(json.dumps(payload))
            (task / "_claude_result_attempt0.json").write_text(json.dumps(payload))

            paths = module.claude_result_paths(task)
            result = module.summarize_claude_cost(paths)

            self.assertEqual(len(paths), 1)
            self.assertEqual(result["turns"], 12)
            self.assertEqual(result["total_tokens"], 190)
            self.assertEqual(result["cost_usd"], 1.25)
            self.assertEqual(result["models"], ["qwen3.8-max-preview"])
            self.assertEqual(result["context_windows"], [1000000])
            self.assertEqual(result["max_output_tokens"], [65536])

    def test_mcnemar_and_bootstrap_are_paired(self) -> None:
        module = _load()
        full = [True, True, False, False]
        treatment = [True, False, True, False]
        exact = module.mcnemar_exact(full, treatment)
        bootstrap = module.paired_bootstrap_ci(
            full, treatment, samples=1000, seed=7)
        self.assertEqual(exact["full_only"], 1)
        self.assertEqual(exact["treatment_only"], 1)
        self.assertEqual(exact["two_sided_exact_p"], 1.0)
        self.assertEqual(bootstrap["difference"], 0.0)

    def test_long_failure_requires_two_burden_dimensions(self) -> None:
        module = _load()
        rows = []
        for arm in module.ARMS:
            rows.extend([
                {
                    "arm": arm,
                    "posthoc_clean_success": False,
                    "turns": 10,
                    "total_tokens": 100,
                    "cost_usd": 1.0,
                },
                {
                    "arm": arm,
                    "posthoc_clean_success": False,
                    "turns": 100,
                    "total_tokens": 1000,
                    "cost_usd": 1.0,
                },
            ])
        module.classify_long_failures(rows)
        for arm in module.ARMS:
            arm_rows = [row for row in rows if row["arm"] == arm]
            self.assertFalse(arm_rows[0]["long_failure"])
            self.assertTrue(arm_rows[1]["long_failure"])

    def test_long_failure_uses_full_arm_thresholds_for_every_arm(self) -> None:
        module = _load()
        rows = []
        for arm in module.ARMS:
            scale = 100 if arm == "no_loopguard" else 1
            for value in range(1, 28):
                rows.append({
                    "arm": arm,
                    "posthoc_clean_success": False,
                    "turns": value * scale,
                    "total_tokens": value * 1000 * scale,
                    "cost_usd": value * 0.1 * scale,
                })

        thresholds = module.classify_long_failures(rows)

        self.assertEqual(thresholds["turns"], 20.5)
        self.assertEqual(sum(
            row["long_failure"] for row in rows if row["arm"] == "full"
        ), 7)
        self.assertEqual(sum(
            row["long_failure"]
            for row in rows if row["arm"] == "no_loopguard"
        ), 27)
        self.assertTrue(all(
            row["long_failure_threshold_source"] == "full_arm_p75"
            for row in rows
        ))

    def test_paired_cost_uses_each_full_treatment_intersection(self) -> None:
        module = _load()
        rows = []
        for arm in module.ARMS:
            for task in ("a", "b", "c"):
                rows.append({
                    "arm": arm,
                    "task": task,
                    "posthoc_clean_success": (
                        task != "c" if arm == "full"
                        else task != "b" if arm == "no_kb"
                        else task == "a"
                    ),
                    "evidence_backed_success": task == "a",
                    "turns": 1,
                    "total_tokens": 10,
                    "cost_usd": 0.5,
                })

        paired, intersections = module._paired_rows(rows)

        scope = "full_vs_no_kb_common_posthoc"
        selected = [row for row in paired if row["scope"] == scope]
        self.assertEqual(
            intersections["full_pairwise"]["no_kb"]["common_posthoc_tasks"],
            ["a"],
        )
        self.assertEqual([row["arm"] for row in selected], ["full", "no_kb"])
        self.assertTrue(all(row["task_count"] == 1 for row in selected))

    def test_arm_summary_reports_risk_and_cost_per_ebs(self) -> None:
        module = _load()
        rows = []
        for arm in module.ARMS:
            rows.append({
                "arm": arm,
                "objective_success": True,
                "reportable_success": True,
                "evidence_backed_success": arm == "full",
                "posthoc_clean_success": True,
                "long_failure": False,
                "turns": 4,
                "total_tokens": 100,
                "cost_usd": 2.0,
            })

        summaries = {
            row["arm"]: row for row in module._arm_summary(rows)
        }

        self.assertEqual(summaries["full"]["rsir"], 0.0)
        self.assertEqual(summaries["full"]["osre"], 0.0)
        self.assertEqual(summaries["full"]["tokens_per_ebs"], 100.0)
        self.assertEqual(summaries["no_anticheat"]["rsir"], 1.0)

    def test_observability_uses_real_direction_and_rollback_records(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp)
            tuning = task / "precision_tuning"
            tuning.mkdir()
            self._write_terminal_artifacts(task)
            self._write_claude_result(tuning)
            (tuning / "tuning_directions.json").write_text(json.dumps({
                "entries": [{
                    "attempt": 0,
                    "fix_type": "tail_mask",
                    "direction_verdict": "initial",
                    "direction_reason": "first repair direction",
                    "outcome": "improved",
                    "evidence": {},
                }],
            }))
            (tuning / "probe_policy_attempt0.json").write_text(json.dumps({
                "attempt": 0,
                "failure_type": "precision_failed",
                "policy": "required",
                "policy_pass": True,
                "metadata_complete": True,
                "observed_status": "executed",
                "ablation_violation": False,
            }))
            (tuning / "kb_usage_trace.json").write_text(json.dumps([{
                "attempt": 0,
                "retrieved_ids": ["kb-1"],
                "injected_ids": ["kb-1"],
                "declared_used_ids": ["kb-1"],
                "declared_unknown_ids": [],
                "usage_trace_complete": True,
            }]))
            (tuning / "forensics_report_0.json").write_text(json.dumps({
                "attempt": 0,
                "status": "completed",
                "primary_hint": "tail writes are unmasked",
                "version": "2.0",
            }))
            current_best = tuning / "history" / "current_best"
            current_best.mkdir(parents=True)
            (current_best / "src").mkdir()
            (current_best / "metric.json").write_text(json.dumps({
                "attempt": 0,
                "case_pass_rate": 0.7,
                "match_rate": 0.9,
                "correctness_passed": False,
            }))
            (current_best / "manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "complete": True,
                "attempt": 0,
                "files": [{
                    "path": "foo.cpp",
                    "size": 12,
                    "sha256": "sha256",
                }],
            }))
            self._write_events(task, [
                {
                    "type": "action_started",
                    "action": {"step": step},
                }
                for step in (
                    "baseline_checkpoint", "forensics", "knowledge_search",
                    "diagnose_and_fix", "validate", "checkpoint_and_rollback",
                )
            ] + [
                {
                    "type": "action_completed",
                    "action": {"step": "baseline_checkpoint"},
                    "result": {
                        "success": True,
                        "checkpoint": {"success": True, "updated": True},
                    },
                },
                {
                    "type": "action_completed",
                    "action": {"step": "checkpoint_and_rollback"},
                    "result": {
                        "success": True,
                        "checkpoint": {"success": True, "updated": False},
                        "rolled_back": True,
                        "from_attempt": 1,
                        "best_attempt": 0,
                    },
                },
            ])

            result = module._observability(
                task, {"attempts_used": 1}, "full")

            self.assertTrue(result["complete"])
            self.assertTrue(result["checks"]["direction"]["observed"])
            self.assertTrue(result["checks"]["checkpoint"]["observed"])
            self.assertTrue(result["checks"]["rollback"]["required"])
            self.assertTrue(result["checks"]["rollback"]["observed"])

    def test_observability_marks_disabled_or_untriggered_evidence_complete(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp)
            tuning = task / "precision_tuning"
            self._write_terminal_artifacts(task)
            self._write_claude_result(tuning)
            (tuning / "diagnosis_summary_attempt_0.json").write_text(
                json.dumps({
                    "attempt": 0,
                    "fix_type": "raw_log_diagnosis",
                    "direction_verdict": "initial",
                    "direction_reason": "structured diagnostics disabled",
                })
            )
            self._write_events(task, [
                {
                    "type": "action_started",
                    "action": {"step": "diagnose_and_fix"},
                },
                {
                    "type": "action_started",
                    "action": {"step": "validate"},
                },
            ])

            result = module._observability(
                task, {"attempts_used": 1}, "no_diagnostic_evidence")

            self.assertTrue(result["complete"])
            for name in ("probe", "kb", "forensics"):
                self.assertFalse(result["checks"][name]["required"])
                self.assertEqual(
                    result["checks"][name]["reason"], "disabled_by_arm")
            self.assertFalse(result["checks"]["rollback"]["observed"])
            self.assertTrue(result["checks"]["rollback"]["complete"])

    def test_observability_rejects_empty_semantic_artifacts(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp)
            tuning = task / "precision_tuning"
            archive = tuning / "claude_results"
            archive.mkdir(parents=True)
            (task / "debug_status.json").write_text("{}")
            (task / "run_summary.json").write_text("{}")
            (archive / "attempt0.json").write_text("{}")
            (tuning / "tuning_directions.json").write_text("{}")
            (tuning / "probe_policy_attempt0.json").write_text("{}")
            (tuning / "kb_usage_trace.json").write_text("{}")
            (tuning / "forensics_report_0.json").write_text("{}")
            self._write_events(task, [
                {
                    "type": "action_started",
                    "action": {"step": step},
                }
                for step in (
                    "forensics", "knowledge_search",
                    "diagnose_and_fix", "validate",
                )
            ])

            result = module._observability(task, {}, "full")

            self.assertFalse(result["complete"])
            for name in (
                "status", "run_summary", "claude_results",
                "probe", "direction", "kb", "forensics",
            ):
                self.assertFalse(result["checks"][name]["observed"], name)

    def test_observability_flags_missing_direction_after_validate_started(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp)
            (task / "debug_status.json").write_text("{}")
            (task / "run_summary.json").write_text("{}")
            self._write_events(task, [{
                "type": "action_started",
                "action": {"step": "validate"},
            }])

            result = module._observability(
                task, {"attempts_used": 1}, "full")

            self.assertFalse(result["complete"])
            self.assertTrue(result["checks"]["direction"]["required"])
            self.assertFalse(result["checks"]["direction"]["observed"])

    def test_observability_recognizes_current_kb_usage_trace_name(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp)
            tuning = task / "precision_tuning"
            tuning.mkdir()
            (task / "debug_status.json").write_text("{}")
            (task / "run_summary.json").write_text("{}")
            (tuning / "kb_usage_trace.json").write_text(json.dumps([{
                "attempt": 0,
                "retrieved_ids": ["kb-1"],
                "injected_ids": ["kb-1"],
                "declared_used_ids": ["kb-1"],
                "declared_unknown_ids": [],
                "usage_trace_complete": True,
            }]))
            self._write_events(task, [{
                "type": "action_started",
                "action": {"step": "diagnose_and_fix"},
            }])

            result = module._observability(task, {}, "full")

            self.assertTrue(result["checks"]["kb"]["required"])
            self.assertTrue(result["checks"]["kb"]["observed"])
            self.assertTrue(result["checks"]["kb"]["complete"])

    def test_evidence_backed_success_requires_complete_observability(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "arm_full" / "tasks" / "level1" / "op"
            (task / "precision_tuning").mkdir(parents=True)
            (task / "debug_status.json").write_text(json.dumps({
                "session_outcome": "success",
                "ended_at": "2026-07-25T00:00:00Z",
                "objective_success": True,
                "reportable_success": True,
                "anti_cheat_pass": True,
                "ast_degrade_pass": True,
            }))

            row = module.task_row(
                root,
                "full",
                task,
                coverage={
                    "level1/op": {
                        "full_eval_applicable": False,
                        "coverage_equivalent": True,
                    },
                },
                posthoc={
                    "level1/op": {
                        "run_state": "completed",
                        "posthoc_clean_success": True,
                    },
                },
            )

            self.assertFalse(row["observability_complete"])
            self.assertFalse(row["evidence_backed_success"])

    def test_manifest_compliance_requires_terminal_posthoc_and_model_usage(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "arm_full" / "tasks" / "level1" / "op").mkdir(
                parents=True)
            manifests = root / "experiment_control" / "arm_manifests"
            manifests.mkdir(parents=True)
            runtime = {
                "targets": [{}],
                "containers": "v5_cann",
                "npus": "3",
                "max_attempts": 5,
                "max_turns": "240",
                "soft_task_turns": 480,
                "max_task_turns": 600,
                "timeout_sec": 43200,
                "agent": "constructive",
                "entry_failure_type": "precision_failed",
                "ablate_profile": "full",
                "kb_path": "/kb",
                "model": "qwen3.8-max-preview",
                "provider_assignment_mode": "fixed_single_provider",
                "providers": ["provider"],
                "provider_env_storage": "ephemeral_secret_dir",
                "usage_query_enabled": False,
                "mixed_provider_enabled": False,
                "kb_read_only": True,
            }
            (root / "arm_full" / "experiment_manifest.json").write_text(
                json.dumps(runtime))
            (root / "arm_full" / "quota_batch_cc_state.json").write_text(
                json.dumps({
                    "event": "completed",
                    "done": 1,
                    "pending": 0,
                }))
            frozen = {
                "task_count": 1,
                "containers": ["v5_cann"],
                "npus": [3],
                "max_attempts": 5,
                "max_turns": 240,
                "soft_task_turns": 480,
                "max_task_turns": 600,
                "timeout": 43200,
                "agent": "constructive",
                "entry_failure_type": "precision_failed",
                "arm": "full",
                "kb_path": "/kb",
                "model": "qwen3.8-max-preview",
                "model_context_window": 1000000,
                "provider_assignment_mode": "fixed_single_provider",
                "provider_names": ["provider"],
                "provider_env_storage": "ephemeral_secret_dir",
                "usage_query_enabled": False,
                "mixed_provider_enabled": False,
                "kb_read_only": True,
            }
            (manifests / "arm_full.json").write_text(json.dumps(frozen))
            contracts = (
                root / "experiment_control" / "container_contracts"
            )
            contracts.mkdir()
            (contracts / "full_before.json").write_text(json.dumps({
                "passed": True,
                "arm": "full",
                "container_id": "container-full",
                "created_at": "2026-07-25T00:00:00Z",
                "image_id": "sha256:image",
                "fresh_label": True,
                "privileged": True,
                "mount_modes": {
                    "root_rw": False,
                    "outputs_rw": True,
                    "dataset_rw": False,
                    "control_rw": False,
                },
                "kb_masked": False,
                "forensics_masked": False,
                "cap_drop": ["SYS_ADMIN"],
            }))
            task_row = {
                "task": "level1/op",
                "terminal_complete": True,
                "result_count": 1,
                "models": ["qwen3.8-max-preview"],
                "context_windows": [1000000],
                "observability_complete": True,
            }
            posthoc = {"level1/op": {"run_state": "completed"}}

            passed = module._manifest_compliance(
                root, "full", [task_row], posthoc)
            self.assertTrue(passed["passed"])
            self.assertTrue(passed["supervisor_completed"])

            task_row["terminal_complete"] = False
            missing_terminal = module._manifest_compliance(
                root, "full", [task_row], posthoc)
            self.assertFalse(missing_terminal["passed"])
            self.assertEqual(missing_terminal["terminal_task_count"], 0)

            task_row["terminal_complete"] = True
            task_row["context_windows"] = [200000]
            wrong_context = module._manifest_compliance(
                root, "full", [task_row], posthoc)
            self.assertFalse(wrong_context["passed"])
            self.assertEqual(len(wrong_context["model_usage_violations"]), 1)

            task_row["context_windows"] = [1000000]
            task_row["observability_complete"] = False
            missing_observability = module._manifest_compliance(
                root, "full", [task_row], posthoc)
            self.assertFalse(missing_observability["passed"])
            self.assertEqual(
                missing_observability["observability_complete_count"], 0)


if __name__ == "__main__":
    unittest.main()
