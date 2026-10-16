#!/usr/bin/env python3
"""Reproduce the frozen paper tables in a disposable copy, using only Python."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "papers/2026-dont-regenerate-debug/artifact"
STEPS = (
    "verify_manifest.py",
    "build_final_integrity_results.py",
    "reproduce_tables.py",
    "derive_no_full_evaluation_shadow.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced")
    output = parser.parse_args().output.absolute()
    if output.exists() or output.is_symlink():
        parser.error("Output already exists; choose a new --output directory.")
    if output.resolve().is_relative_to(ARTIFACT.resolve()):
        parser.error("Output must be outside the frozen paper artifact.")
    if any(p.is_symlink() for p in ARTIFACT.rglob("*")):
        parser.error("The paper artifact must not contain symbolic links.")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    with tempfile.TemporaryDirectory(prefix="debug-agent-paper-") as directory:
        work = Path(directory) / "artifact"
        shutil.copytree(ARTIFACT, work,
                        ignore=shutil.ignore_patterns(".DS_Store", "__pycache__", "*.pyc", "*.pyo"))
        for step in STEPS:
            subprocess.run([sys.executable, str(work / "scripts" / step)],
                           cwd=work, env=env, check=True)
        # Only publish derived outputs after every check has succeeded.
        shutil.copytree(work / "reproduced", output)
        (output / "integrity").mkdir()
        for name in ("final_integrity_results.csv", "integrity_manifest.json"):
            shutil.copyfile(work / "integrity" / name, output / "integrity" / name)
    summary = {
        "paper": "arXiv:2608.02712",
        "artifact_manifest_sha256": hashlib.sha256(
            (ARTIFACT / "MANIFEST.sha256").read_bytes()).hexdigest(),
        "completed_steps": list(STEPS),
        "scope": "Recompute published tables from frozen records; no NPU or model execution.",
    }
    (output / "REPRODUCTION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"All paper table checks passed. Outputs: {output}")


if __name__ == "__main__":
    main()
