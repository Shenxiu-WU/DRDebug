from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "experiments" / "v5_ablation" / "prepare_v5_manifests.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("v5_prepare_manifests", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V5PrepareManifestTests(unittest.TestCase):
    def test_equal_count_different_cases_require_full_eval(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp)
            (task / "Foo.json").write_text(
                '{"inputs":[1]}\n{"inputs":[2]}\n')
            (task / "Foo.json.bak").write_text(
                '{"inputs":[1]}\n{"inputs":[99]}\n')

            row = module._coverage_row(task)

            self.assertEqual(row["active_cases"], 2)
            self.assertEqual(row["full_cases"], 2)
            self.assertFalse(row["coverage_equivalent"])
            self.assertTrue(row["full_eval_applicable"])

    def test_semantically_identical_cases_are_equivalent(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp)
            (task / "Foo.json").write_text(
                '{"inputs":[1],"meta":{"a":1,"b":2}}\n{"inputs":[2]}\n')
            (task / "Foo.json.bak").write_text(
                '{"inputs": [2]}\n{"meta":{"b":2,"a":1},"inputs":[1]}\n')

            row = module._coverage_row(task)

            self.assertTrue(row["coverage_equivalent"])
            self.assertFalse(row["full_eval_applicable"])

    def test_code_manifest_excludes_secrets_and_outputs(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".secrets").mkdir()
            (root / "outputs").mkdir()
            (root / "utils").mkdir()
            (root / ".secrets" / "provider.json").write_text("secret")
            (root / "outputs" / "run.json").write_text("runtime")
            (root / "utils" / "tool.py").write_text("pass\n")

            result = module.file_manifest(
                root, skip_runtime=True, code_root=True)

            self.assertEqual(list(result), ["utils/tool.py"])


if __name__ == "__main__":
    unittest.main()
