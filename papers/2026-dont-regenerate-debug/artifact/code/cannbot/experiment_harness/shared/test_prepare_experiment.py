#!/usr/bin/env python3
"""Focused regression tests for experiment manifest path generation."""

from pathlib import Path

from prepare_experiment import escaped_project_dir


def main() -> int:
    cwd = Path("/workspace/regenerate_experiment/runs/level1_003_Add/regenerate/trial1")
    expected = (
        "-workspace-regenerate-experiment-runs-"
        "level1-003-Add-regenerate-trial1"
    )
    actual = escaped_project_dir(cwd)
    assert actual == expected, (actual, expected)
    print("PASS: Claude Code project journal slug matches runtime layout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
