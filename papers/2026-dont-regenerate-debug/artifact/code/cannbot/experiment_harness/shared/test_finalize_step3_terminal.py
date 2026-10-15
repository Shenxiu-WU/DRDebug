#!/usr/bin/env python3
"""Regression checks for Regenerate terminal-state reconciliation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import finalize_step3


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        result = Path(temporary) / "final_stdout.json"
        result.write_text("")

        timeout = {
            "status": "failed",
            "provider_aborted": False,
            "timed_out": True,
            "launcher_returncode": 124,
            "ended_at": "2026-07-24T00:00:00+00:00",
        }
        assert finalize_step3.regenerate_record_terminal(timeout, result)

        provider_abort = dict(timeout, provider_aborted=True)
        assert not finalize_step3.regenerate_record_terminal(provider_abort, result)

        unexplained_failure = {
            "status": "failed",
            "provider_aborted": False,
            "ended_at": "2026-07-24T00:00:00+00:00",
        }
        assert not finalize_step3.regenerate_record_terminal(unexplained_failure, result)

        completed = {"status": "completed", "provider_aborted": False}
        assert not finalize_step3.regenerate_record_terminal(completed, result)
        result.write_text("{}\n")
        assert finalize_step3.regenerate_record_terminal(completed, result)

    print("PASS: completed results and auditable failed timeouts are terminal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
