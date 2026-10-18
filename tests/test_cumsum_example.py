"""CPU source/provenance tests; no task imports, model calls, builds or NPU use."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples/cumsum"
SPEC = importlib.util.spec_from_file_location("cumsum_example", EXAMPLE / "prepare.py")
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


class CumsumExampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="drdebug-cumsum-example-")
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.assets = self.repo / "examples/cumsum"
        self.initial = self.repo / example.PAPER_TASK
        shutil.copytree(REPO / example.PAPER_TASK, self.initial)
        shutil.copytree(EXAMPLE, self.assets, ignore=shutil.ignore_patterns("__pycache__"))
        self.output = self.root / "new-task"

    def tearDown(self):
        self.temp.cleanup()

    def stage(self, mode="replay", output=None):
        return example.stage(mode, output or self.output, example=self.assets, repo=self.repo)

    def update_manifest(self, update):
        path = self.assets / "example.json"
        value = json.loads(path.read_text())
        update(value)
        path.write_text(json.dumps(value))

    def test_published_bytes_are_pinned_without_execution(self):
        manifest, _, initial, repaired = example.verified_assets()
        self.assertEqual(len(initial), 15)
        self.assertEqual(len(repaired), 8)
        self.assertEqual(manifest["historical_validation"]["full_cases"], 51)
        self.assertNotEqual(initial["kernel/cumsum_kernel.h"], repaired["cumsum_kernel.h"])
        self.assertTrue(all(Path(name).suffix in {".cpp", ".h"} for name in repaired))

    def test_live_and_replay_preserve_original_contract_and_have_distinct_origins(self):
        _, manifest_bytes, initial, repaired = example.verified_assets(self.assets, self.repo)
        live = self.root / "live"
        self.stage("live", live)
        self.stage("replay")
        for name, data in initial.items():
            self.assertEqual((live / name).read_bytes(), data)
            expected = repaired[Path(name).name] if name.startswith("kernel/") else data
            self.assertEqual((self.output / name).read_bytes(), expected)
            self.assertEqual((self.initial / name).read_bytes(), data)
        live_origin = json.loads((live / "example-origin.json").read_text())
        replay_origin = json.loads((self.output / "example-origin.json").read_text())
        self.assertEqual(live_origin["kernel_origin"], "original paper near-miss input")
        self.assertEqual(replay_origin["kernel_origin"], "saved precision-repaired candidate")
        self.assertEqual(replay_origin["example_manifest_sha256"], example.digest(manifest_bytes))
        for name, sha in replay_origin["files_sha256"].items():
            self.assertEqual(example.digest((self.output / name).read_bytes()), sha)

    def test_staging_does_not_import_reference_or_wrapper(self):
        payload = b"raise RuntimeError('TASK CODE MUST NEVER RUN DURING STAGING')\n"
        (self.initial / "model.py").write_bytes(payload)
        self.update_manifest(lambda m: m["original_task_files"].update({
            "model.py": {"sha256": example.digest(payload), "size": len(payload)}}))
        self.stage()
        self.assertEqual((self.output / "model.py").read_bytes(), payload)

    def test_existing_task_is_not_overwritten(self):
        self.output.mkdir()
        marker = self.output / "existing-log.txt"
        marker.write_text("preserve me")
        with self.assertRaisesRegex(example.ExampleError, "already exists"):
            self.stage()
        self.assertEqual(marker.read_text(), "preserve me")
        self.assertEqual(list(self.output.iterdir()), [marker])

    def test_broken_output_symlink_is_preserved(self):
        self.output.symlink_to(self.root / "missing", target_is_directory=True)
        with self.assertRaisesRegex(example.ExampleError, "already exists"):
            self.stage()
        self.assertTrue(self.output.is_symlink())

    def test_corrupt_original_fails_before_output_creation(self):
        (self.initial / "5_Cumsum.json.bak").write_text("changed cases\n")
        with self.assertRaisesRegex(example.ExampleError, "hash or size mismatch"):
            self.stage()
        self.assertFalse(self.output.exists())

    def test_corrupt_repaired_source_fails_before_output_creation(self):
        (self.assets / "repaired/kernel/cumsum_kernel.h").write_text("changed candidate\n")
        with self.assertRaisesRegex(example.ExampleError, "hash or size mismatch"):
            self.stage()
        self.assertFalse(self.output.exists())

    def test_binary_extra_is_rejected(self):
        (self.assets / "repaired/kernel/old.so").write_bytes(b"stale native extension")
        with self.assertRaisesRegex(example.ExampleError, "file set differs"):
            self.stage()
        self.assertFalse(self.output.exists())

    def test_source_symlink_is_rejected_even_if_bytes_would_match(self):
        source = self.assets / "repaired/kernel/cumsum_fp32.cpp"
        outside = self.root / "outside.cpp"
        shutil.move(source, outside)
        source.symlink_to(outside)
        with self.assertRaisesRegex(example.ExampleError, "Symlink source"):
            self.stage()
        self.assertFalse(self.output.exists())

    def test_manifest_path_injection_is_rejected_before_source_reads(self):
        self.update_manifest(lambda m: m.update(original_task="../private-task"))
        with self.assertRaisesRegex(example.ExampleError, "Unknown or incomplete"):
            self.stage()
        self.assertFalse(self.output.exists())

    def test_manifest_extra_native_path_is_rejected(self):
        self.update_manifest(lambda m: m["repaired_files"].update({
            "../../outside": {"sha256": "0" * 64, "size": 1}}))
        with self.assertRaisesRegex(example.ExampleError, "Unknown or incomplete"):
            self.stage()
        self.assertFalse(self.output.exists())

    def test_output_inside_sources_is_rejected(self):
        with self.assertRaisesRegex(example.ExampleError, "must not contain"):
            self.stage(output=self.initial / "new-task")
        with self.assertRaisesRegex(example.ExampleError, "must not contain"):
            self.stage(output=self.assets / "new-task")
        self.assertFalse((self.initial / "new-task").exists())
        self.assertFalse((self.assets / "new-task").exists())


if __name__ == "__main__":
    unittest.main()
