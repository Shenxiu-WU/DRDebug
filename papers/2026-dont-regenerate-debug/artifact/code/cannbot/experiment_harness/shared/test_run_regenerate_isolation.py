#!/usr/bin/env python3
"""Deterministic checks for the Regenerate filesystem isolation boundary."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import run_regenerate


def main() -> int:
    record = {
        "run_id": "regen:level1/005_Cumsum:t2",
        "expected_session_id": "a10d08bf-61cc-5b81-b224-c65188e76275",
    }
    root = Path("/workspace/regenerate_experiment")
    cwd = root / "runs/level1_005_Cumsum/regenerate/trial2"
    home = cwd / ".home"
    plugin = root / "runtime/ops-direct-invoke-step3-sanitized"
    devkit = root / "vendor/asc-devkit"
    with mock.patch.object(run_regenerate.subprocess, "run") as mocked_run:
        name = run_regenerate.start_trial_container(
            record=record,
            cwd=cwd,
            home=home,
            plugin=plugin,
            devkit=devkit,
            npu="1",
            image="cannbot-regenerate-runtime:release",
        )
    assert name.startswith("cannbot-regen-")
    docker_run = mocked_run.call_args_list[1].args[0]
    joined = " ".join(docker_run)
    assert str(cwd) in joined
    assert f"src={plugin},dst={plugin},readonly" in joined
    assert f"src={devkit},dst={devkit},readonly" in joined
    assert f"src={root},dst={root}" not in joined
    assert "debug_fresh_sources" not in joined
    assert "/workspace/debug_agent" not in joined

    defaults = run_regenerate.parse_args([])
    assert defaults.timeout == 21600
    assert defaults.concurrency == 5
    assert defaults.npus == "1,2,4,5,3"

    with tempfile.TemporaryDirectory() as temporary:
        test_root = Path(temporary)
        test_cwd = test_root / "runs/op/regenerate/trial1"
        journal = test_cwd / ".home/.claude/projects/project/session.jsonl"
        journal.parent.mkdir(parents=True)
        event = {
            "timestamp": "2026-07-18T00:00:00Z",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Read",
                        "input": {"file_path": str(test_root / "debug_fresh_sources/op/model.py")},
                    }
                ]
            },
        }
        journal.write_text(json.dumps(event) + "\n")
        attempts = run_regenerate.audit_isolation_access_attempts(test_cwd / ".home", test_cwd, test_root)
        assert len(attempts) == 1
        assert "debug_fresh_sources" in attempts[0]["markers"]
    print("PASS: 6h default, five-card default, allowlisted mounts, and postflight audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
