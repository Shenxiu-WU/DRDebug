#!/usr/bin/env python3
"""Offline tests for the narrow direct-invoke anti-cheat compatibility check."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from validate_manifest import adjudicate_direct_invoke_anticheat


OFFICIAL_BLIND_SPOT = {
    "verdict": "CHEAT",
    "reasons": ["AST_FAIL:type1", "NO_KERNEL_LAUNCH:x"],
    "details": {
        "ast": {"status": "fail", "regression_type": 1},
        "cpp": {
            "status": "fail",
            "violations": [{"type": "NO_KERNEL_LAUNCH", "file": "kernel/", "line": 0}],
        },
    },
}


WRAPPER = '''
import torch
torch.ops.load_library("operator_project/build/libcustom.so")
class ModelNew:
    def forward(self, x, y):
        op = torch.ops.npu.custom
        return op(x.npu(), y.npu()).to(x.device)
'''


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        task = Path(directory)
        (task / "kernel").mkdir()
        (task / "operator_project" / "build").mkdir(parents=True)
        (task / "model_new_ascendc.py").write_text(WRAPPER)
        (task / "kernel" / "custom.asc").write_text(
            'custom_kernel<<<block_num, nullptr, stream>>>(x, y, z);\n'
        )
        (task / "operator_project" / "build" / "libcustom.so").write_bytes(b"ELF-test")
        result = adjudicate_direct_invoke_anticheat(task, OFFICIAL_BLIND_SPOT)
        assert result["applicable"] and result["passed"], json.dumps(result, indent=2)

        bad = json.loads(json.dumps(OFFICIAL_BLIND_SPOT))
        bad["details"]["cpp"]["violations"].append({"type": "CPP_ATEN_CALL", "detail": "at::sum"})
        rejected = adjudicate_direct_invoke_anticheat(task, bad)
        assert rejected["applicable"] is False and rejected["passed"] is False

        (task / "model_new_ascendc.py").write_text(WRAPPER.replace("return op(", "return torch.add("))
        rejected = adjudicate_direct_invoke_anticheat(task, OFFICIAL_BLIND_SPOT)
        assert rejected["passed"] is False

    print("PASS: narrow override accepts real direct invoke and rejects other violations/fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
