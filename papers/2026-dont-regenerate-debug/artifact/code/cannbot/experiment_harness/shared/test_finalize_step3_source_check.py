#!/usr/bin/env python3
"""Regression test for the final source-isolation byte comparison."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import finalize_step3


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        evidence = root / "evidence"
        evidence.mkdir()
        evidence.joinpath("ascendopgenagent_head_before_step3.txt").write_text("abc123\n")
        evidence.joinpath("ascendopgenagent_status_before_step3.txt").write_bytes(b" M owned.txt\n")
        responses = [
            SimpleNamespace(stdout="abc123\n"),
            SimpleNamespace(stdout=b" M owned.txt\n"),
        ]
        with mock.patch.object(subprocess, "run", side_effect=responses):
            result = finalize_step3.source_is_unchanged(root)
        assert result["unchanged"] is True, result
        assert result["status_bytes_unchanged"] is True, result
    print("PASS: finalizer compares git status as bytes without encode failure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
