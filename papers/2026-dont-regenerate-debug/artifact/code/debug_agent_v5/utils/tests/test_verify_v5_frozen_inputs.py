from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "verify_v5_frozen_inputs.py"


def _load():
    spec = importlib.util.spec_from_file_location("v5_frozen_inputs", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V5FrozenInputTests(unittest.TestCase):
    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_code_manifest_excludes_secret_output_and_git(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".secrets").mkdir()
            (root / "outputs").mkdir()
            (root / ".git").mkdir()
            (root / "utils").mkdir()
            (root / ".secrets" / "key.json").write_text("secret")
            (root / "outputs" / "result.json").write_text("result")
            (root / ".git" / "config").write_text("git")
            (root / "utils" / "tool.py").write_text("print('ok')\n")

            manifest = module.file_manifest(
                root, skip_runtime=True, code_root=True)

            self.assertEqual(list(manifest), ["utils/tool.py"])

    def test_compare_manifest_reports_changed_missing_and_extra(self) -> None:
        module = _load()
        result = module.compare_manifest(
            {"a": "1", "b": "2"},
            {"a": "9", "c": "3"},
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["changed"], ["a"])
        self.assertEqual(result["missing"], ["b"])
        self.assertEqual(result["extra"], ["c"])

    def test_control_plane_manifest_detects_deployed_drift(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            deployed = root / "deployed"
            source.mkdir()
            deployed.mkdir()
            (source / "launch.sh").write_text("echo frozen\n")
            (deployed / "launch.sh").write_text("echo changed\n")

            source_manifest = module.selected_file_manifest(
                source, ["launch.sh"])
            deployed_manifest = module.selected_file_manifest(
                deployed, ["launch.sh"])
            result = module.compare_manifest(
                source_manifest, deployed_manifest)

            self.assertFalse(result["passed"])
            self.assertEqual(result["changed"], ["launch.sh"])

    def test_preflight_checks_require_passed_content_not_just_files(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            control = Path(tmp)
            (control / "dataset_preflight").mkdir()
            (control / "qwen38max.smoke.meta.json").write_text(json.dumps({
                "passed": True,
                "models": ["qwen3.8-max-preview"],
                "context_windows": [1000000],
            }))
            (control / "ablation_profile_smoke.json").write_text(json.dumps({
                "passed": True,
                "profiles": [
                    {"profile": arm} for arm in module.ARMS
                ],
            }))
            (control / "arm_contract_verification.json").write_text(json.dumps({
                "passed": True,
                "arms": [
                    {"arm": arm, "passed": True} for arm in module.ARMS
                ],
            }))
            preflight_path = (
                control / "dataset_preflight"
                / "dataset_preflight_results.json"
            )
            preflight_path.write_text(json.dumps({
                "passed": True,
                "task_count": 27,
                "passed_count": 27,
                "used_npus": [3, 4, 5, 6, 7],
                "required_npus": [3, 4, 5, 6, 7],
                "all_required_npus_exercised": True,
            }))
            (control / "dataset_audit.json").write_text(json.dumps({
                "task_count": 27,
                "invalid_tasks": [],
            }))

            checks = module.preflight_artifact_checks(
                control,
                expected_tasks=27,
                required_npus=[3, 4, 5, 6, 7],
                expected_model="qwen3.8-max-preview",
                expected_context=1000000,
            )
            self.assertTrue(all(row["passed"] for row in checks.values()))

            preflight_path.write_text(json.dumps({
                "passed": False,
                "task_count": 27,
                "passed_count": 26,
                "used_npus": [3, 4, 5, 6, 7],
                "required_npus": [3, 4, 5, 6, 7],
                "all_required_npus_exercised": True,
            }))
            checks = module.preflight_artifact_checks(
                control,
                expected_tasks=27,
                required_npus=[3, 4, 5, 6, 7],
                expected_model="qwen3.8-max-preview",
                expected_context=1000000,
            )
            self.assertFalse(checks["dataset_preflight_pass"]["passed"])

    def test_source_list_check_requires_exact_frozen_dataset_entries(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = root / "control"
            dataset = root / "dataset"
            (dataset / "tasks" / "level1" / "001_Foo").mkdir(parents=True)
            (dataset / "tasks" / "level2" / "002_Bar").mkdir(parents=True)
            control.mkdir()
            source_list = control / "source_dirs_n27.txt"
            entries = [
                str(dataset / "tasks" / "level1" / "001_Foo"),
                str(dataset / "tasks" / "level2" / "002_Bar"),
            ]
            source_list.write_text("\n".join(entries) + "\n")
            metadata = control / "source_dirs_n27.sha256.json"
            metadata.write_text(json.dumps({
                "schema_version": 1,
                "path": str(source_list),
                "sha256": self._sha256(source_list),
                "task_count": 2,
                "entries": entries,
            }))

            passed = module.source_list_check(
                control, dataset, expected_tasks=2)

            self.assertTrue(passed["passed"])
            self.assertEqual(passed["task_count"], 2)

            source_list.write_text(entries[0] + "\n" + entries[0] + "\n")
            tampered = module.source_list_check(
                control, dataset, expected_tasks=2)
            self.assertFalse(tampered["passed"])
            self.assertFalse(tampered["sha256_matches"])
            self.assertFalse(tampered["unique_entries"])


if __name__ == "__main__":
    unittest.main()
