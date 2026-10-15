#!/usr/bin/env python3
"""Verify frozen V5 arm manifests against the executable launch contract."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARMS = (
    "full",
    "no_kb",
    "no_diagnostic_evidence",
    "no_loopguard",
    "no_anticheat",
    "no_fulleval",
    "baseline",
)
COMMON = {
    "task_count": 27,
    "containers": ["v5_cann"] * 5,
    "npus": [3, 4, 5, 6, 7],
    "model": "qwen3.8-max-preview",
    "model_context_window": 1000000,
    "provider_assignment_mode": "fixed_single_provider",
    "provider_names": ["provider-slot-a"],
    "declared_parallel_limit": 5,
    "usage_query_enabled": False,
    "mixed_provider_enabled": False,
    "provider_env_storage": "ephemeral_secret_dir",
    "max_attempts": 5,
    "max_turns": 240,
    "soft_task_turns": 480,
    "max_task_turns": 600,
    "timeout": 43200,
    "agent": "constructive",
    "entry_failure_type": "precision_failed",
    "posthoc_observer_isolated": True,
    "primary_cost_scope": "final_valid_cycle_only",
    "failed_cycles_excluded_from_primary_cost": True,
    "fresh_container_per_arm": True,
    "container_privileged": True,
    "container_cap_drop": ["SYS_ADMIN"],
}


def _describe(wrapper: Path, arm: str, kb: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "bash",
            str(wrapper),
            "--describe-ablate-profile",
            arm,
            "--kb-path",
            str(kb),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"profile={arm} describe failed rc={proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
    return json.loads(proc.stdout)


def verify_arm(
    manifest: dict[str, Any],
    described: dict[str, Any],
    *,
    arm: str,
    kb: Path,
) -> dict[str, Any]:
    actual_ablations = sorted(
        key for key, value in described.get("ablated", {}).items() if value
    )
    kb_enabled = described.get("kb_path") is not None
    expected: dict[str, Any] = {
        "arm": arm,
        "ablated_capabilities": actual_ablations,
        "kb_path": str(kb) if kb_enabled else None,
        "kb_read_only": kb_enabled,
        "kb_filesystem_masked": not kb_enabled,
        "forensics_script_masked": (
            arm in {"no_diagnostic_evidence", "baseline"}
        ),
        **COMMON,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    return {
        "arm": arm,
        "passed": not mismatches,
        "mismatches": mismatches,
        "described_profile": described,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    wrapper = args.repo_root / "utils" / "run_ascendc_debug_batch_cc.sh"
    rows = []
    for arm in ARMS:
        manifest_path = args.control / "arm_manifests" / f"arm_{arm}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.append(verify_arm(
            manifest,
            _describe(wrapper, arm, args.kb),
            arm=arm,
            kb=args.kb,
        ))
    passed = all(row["passed"] for row in rows)
    payload = {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "arms": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
