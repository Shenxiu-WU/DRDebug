#!/usr/bin/env python3
"""Regression tests for Contract-v2 shared-library materialization/staging."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_regenerate import materialize_candidate
from validate_manifest import prepare_eval_dir


class CandidateLibraryStagingTest(unittest.TestCase):
    def test_materializer_copies_declared_library_and_eval_can_see_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cwd = root / "run"
            source = root / "source"
            (cwd / "spec").mkdir(parents=True)
            (cwd / "candidate_adapter").mkdir()
            (cwd / "task" / "kernel" / "build").mkdir(parents=True)
            (cwd / "task" / "kernel" / "demo.cpp").write_text("// native source\n")
            library = cwd / "task" / "kernel" / "build" / "_demo_ext.so"
            library.write_bytes(b"ELF-test-payload")
            (cwd / "spec" / "model.py").write_text("class Model: pass\n")
            (cwd / "spec" / "model.json").write_text('{"inputs": []}\n')
            (cwd / "candidate_adapter" / "operator_dir.txt").write_text("task\n")
            (cwd / "candidate_adapter" / "model_new_ascendc.py").write_text(
                "from pathlib import Path\n"
                "_BUILD = Path(__file__).resolve().parent / 'kernel' / 'build'\n"
            )

            materialized, note = materialize_candidate(cwd)
            self.assertTrue(materialized, note)
            copied = cwd / "candidate_task" / "kernel" / "build" / "_demo_ext.so"
            self.assertEqual(copied.read_bytes(), library.read_bytes())

            manifest = json.loads(
                (cwd / "candidate_task" / "materialization_manifest.json").read_text()
            )
            self.assertEqual(manifest["libraries"], ["kernel/build/_demo_ext.so"])

            source.mkdir()
            (source / "model.py").write_text("class Model: pass\n")
            (source / "model.json").write_text('{"inputs": []}\n')
            evaluation = cwd / "validation" / "work"
            _, _, library_paths, build_alias = prepare_eval_dir(
                source, cwd / "candidate_task", evaluation
            )
            self.assertTrue((evaluation / "model_new_ascendc.py").is_file())
            self.assertFalse((evaluation / "model_new_ascendc.py").is_symlink())
            self.assertTrue((evaluation / "kernel" / "build" / "_demo_ext.so").is_file())
            self.assertTrue(library_paths)
            self.assertIsNotNone(build_alias)


if __name__ == "__main__":
    unittest.main()
