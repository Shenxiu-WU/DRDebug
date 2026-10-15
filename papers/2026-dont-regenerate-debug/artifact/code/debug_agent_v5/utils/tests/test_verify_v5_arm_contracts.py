from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "verify_v5_arm_contracts.py"


def _load():
    spec = importlib.util.spec_from_file_location("v5_arm_contracts", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V5ArmContractTests(unittest.TestCase):
    def test_contract_accepts_executable_profile(self) -> None:
        module = _load()
        manifest = {
            "arm": "no_kb",
            "ablated_capabilities": ["kb"],
            "kb_path": None,
            "kb_read_only": False,
            "kb_filesystem_masked": True,
            "forensics_script_masked": False,
            **module.COMMON,
        }
        row = module.verify_arm(
            manifest,
            {
                "profile": "no_kb",
                "ablated": {"kb": True, "probe": False},
                "kb_path": None,
            },
            arm="no_kb",
            kb=Path("/frozen/kb.json"),
        )
        self.assertTrue(row["passed"])

    def test_contract_rejects_kb_and_ablation_drift(self) -> None:
        module = _load()
        manifest = {
            "arm": "baseline",
            "ablated_capabilities": ["kb"],
            "kb_path": None,
            "kb_read_only": True,
            "kb_filesystem_masked": True,
            "forensics_script_masked": True,
            **module.COMMON,
        }
        row = module.verify_arm(
            manifest,
            {
                "profile": "baseline",
                "ablated": {"kb": True, "anticheat_detect_only": True},
                "kb_path": None,
            },
            arm="baseline",
            kb=Path("/frozen/kb.json"),
        )
        self.assertFalse(row["passed"])
        self.assertIn("ablated_capabilities", row["mismatches"])
        self.assertIn("kb_read_only", row["mismatches"])


if __name__ == "__main__":
    unittest.main()
