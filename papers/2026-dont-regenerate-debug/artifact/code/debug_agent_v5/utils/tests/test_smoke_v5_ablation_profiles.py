from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "smoke_v5_ablation_profiles.py"
REPO_ROOT = SCRIPT.parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("v5_profile_smoke", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V5ProfileSmokeTests(unittest.TestCase):
    def test_no_diagnostic_profile_is_joint_and_keeps_other_gates(self) -> None:
        module = _load()
        payload = module.describe(
            REPO_ROOT / "utils" / "run_ascendc_debug_batch_cc.sh",
            "no_diagnostic_evidence",
            "/frozen/kb.json",
        )
        enabled = {
            key for key, value in payload["ablated"].items() if value
        }
        self.assertEqual(
            enabled,
            {"diagnostic_evidence", "forensics", "probe", "kb"},
        )
        self.assertIsNone(payload["kb_path"])

    def test_smoke_writes_passing_artifact(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "profiles.json"
            old_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT),
                    "--repo-root", str(REPO_ROOT),
                    "--output", str(output),
                ]
                self.assertEqual(module.main(), 0)
            finally:
                sys.argv = old_argv
            self.assertIn('"passed": true', output.read_text())


if __name__ == "__main__":
    unittest.main()
