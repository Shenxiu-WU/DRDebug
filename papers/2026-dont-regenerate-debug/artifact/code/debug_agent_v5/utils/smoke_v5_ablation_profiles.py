#!/usr/bin/env python3
"""Prove the runtime contract of every formal V5 ablation profile."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


FORMAL_PROFILES = {
    "full": set(),
    "no_kb": {"kb"},
    "no_diagnostic_evidence": {
        "diagnostic_evidence", "forensics", "probe", "kb",
    },
    "no_loopguard": {"loop_guard"},
    "no_anticheat": {"anticheat"},
    "no_fulleval": {"full_eval"},
    "baseline": {
        "forensics", "probe", "kb", "loop_guard", "gate_a",
        "full_eval", "recovery", "anticheat_detect_only",
    },
}


@contextmanager
def _temporary_env(values: dict[str, str]) -> Iterator[None]:
    old = os.environ.copy()
    os.environ.update(values)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def describe(wrapper: Path, profile: str, kb_path: str) -> dict:
    proc = subprocess.run(
        [
            "bash", str(wrapper),
            "--describe-ablate-profile", profile,
            "--kb-path", kb_path,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"profile={profile} describe failed rc={proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
    return json.loads(proc.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kb-path", default="/frozen/v5_kb.json")
    args = parser.parse_args()

    wrapper = args.repo_root / "utils" / "run_ascendc_debug_batch_cc.sh"
    engine_dir = args.repo_root / "skills" / "ascendc" / "ascendc-debug"
    sys.path.insert(0, str(engine_dir))
    from engine.next_action import _round_sequence  # noqa: PLC0415

    rows = []
    for profile, expected in FORMAL_PROFILES.items():
        payload = describe(wrapper, profile, args.kb_path)
        actual = {
            key for key, value in payload["ablated"].items() if value
        }
        if actual != expected:
            raise SystemExit(
                f"{profile}: expected ablated={sorted(expected)}, "
                f"got={sorted(actual)}"
            )
        expected_kb = None if "kb" in expected else args.kb_path
        if payload.get("kb_path") != expected_kb:
            raise SystemExit(
                f"{profile}: expected kb_path={expected_kb!r}, "
                f"got={payload.get('kb_path')!r}"
            )
        env = {
            f"ABLATE_{key.upper()}": "1"
            for key in expected
            if key != "anticheat_detect_only"
        }
        with _temporary_env(env):
            precision_sequence = list(_round_sequence("precision_failed"))
            runtime_sequence = list(_round_sequence("runtime_error"))
        if profile == "no_diagnostic_evidence":
            if precision_sequence != ["audit", "diagnose_and_fix", "validate"]:
                raise SystemExit(
                    "no_diagnostic_evidence precision route is not strict: "
                    f"{precision_sequence}"
                )
            if runtime_sequence != ["diagnose_and_fix", "validate"]:
                raise SystemExit(
                    "no_diagnostic_evidence runtime route is not strict: "
                    f"{runtime_sequence}"
                )
        rows.append({
            **payload,
            "precision_round_sequence": precision_sequence,
            "runtime_round_sequence": runtime_sequence,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "passed": True,
                "profiles": rows,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
