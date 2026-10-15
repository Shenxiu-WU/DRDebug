#!/usr/bin/env python3
"""Verify V5 code, dataset, KB and credential invariants before launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNTIME_NAMES = {
    ".bench_baseline",
    ".debug_events",
    ".verify_logs",
    ".verify_status",
    "__pycache__",
    "build",
    "debug_status.json",
    "debug_trace.md",
    "precision_tuning",
}
CODE_EXCLUDED_TOP_LEVEL = {".git", ".secrets", "outputs"}
ARMS = (
    "full",
    "no_kb",
    "no_diagnostic_evidence",
    "no_loopguard",
    "no_anticheat",
    "no_fulleval",
    "baseline",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(
    root: Path,
    *,
    skip_runtime: bool,
    code_root: bool = False,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if (
            ".git" in rel.parts
            or path.name == ".DS_Store"
            or path.name.startswith("._")
            or (code_root and rel.parts[0] in CODE_EXCLUDED_TOP_LEVEL)
        ):
            continue
        if skip_runtime and (
            any(part in RUNTIME_NAMES for part in rel.parts)
            or path.name.startswith("_claude_result")
            or path.name.startswith("_provider")
            or path.name == "_anticheat.json"
            or path.suffix == ".pyc"
        ):
            continue
        result[str(rel)] = sha256(path)
    return result


def compare_manifest(expected: dict[str, str], actual: dict[str, str]) -> dict[str, Any]:
    expected_keys = set(expected)
    actual_keys = set(actual)
    changed = sorted(
        key for key in expected_keys & actual_keys
        if expected[key] != actual[key]
    )
    return {
        "passed": not changed and expected_keys == actual_keys,
        "expected_files": len(expected),
        "actual_files": len(actual),
        "missing": sorted(expected_keys - actual_keys),
        "extra": sorted(actual_keys - expected_keys),
        "changed": changed,
    }


def _load_files(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"manifest has no files mapping: {path}")
    return {str(key): str(value) for key, value in files.items()}


def selected_file_manifest(root: Path, paths: Any) -> dict[str, str]:
    return {
        str(rel): sha256(root / str(rel))
        for rel in paths
        if (root / str(rel)).is_file()
    }


def _write_bits(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode) & (
        stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def preflight_artifact_checks(
    control: Path,
    *,
    expected_tasks: int,
    required_npus: list[int],
    expected_model: str,
    expected_context: int,
) -> dict[str, dict[str, Any]]:
    provider = _load_json(control / "qwen38max.smoke.meta.json")
    profiles = _load_json(control / "ablation_profile_smoke.json")
    contracts = _load_json(control / "arm_contract_verification.json")
    preflight = _load_json(
        control / "dataset_preflight" / "dataset_preflight_results.json")
    audit = _load_json(control / "dataset_audit.json")
    profile_names = [
        row.get("profile")
        for row in profiles.get("profiles", [])
        if isinstance(row, dict)
    ]
    contract_names = [
        row.get("arm")
        for row in contracts.get("arms", [])
        if isinstance(row, dict)
    ]
    return {
        "provider_smoke_pass": {
            "passed": (
                provider.get("passed") is True
                and provider.get("models") == [expected_model]
                and provider.get("context_windows") == [expected_context]
            ),
            "models": provider.get("models"),
            "context_windows": provider.get("context_windows"),
        },
        "profile_smoke_pass": {
            "passed": (
                profiles.get("passed") is True
                and profile_names == list(ARMS)
            ),
            "profiles": profile_names,
        },
        "arm_contract_pass": {
            "passed": (
                contracts.get("passed") is True
                and contract_names == list(ARMS)
                and all(
                    row.get("passed") is True
                    for row in contracts.get("arms", [])
                    if isinstance(row, dict)
                )
            ),
            "arms": contract_names,
        },
        "dataset_preflight_pass": {
            "passed": (
                preflight.get("passed") is True
                and preflight.get("task_count") == expected_tasks
                and preflight.get("passed_count") == expected_tasks
                and preflight.get("used_npus") == required_npus
                and preflight.get("required_npus") == required_npus
                and preflight.get("all_required_npus_exercised") is True
            ),
            "task_count": preflight.get("task_count"),
            "passed_count": preflight.get("passed_count"),
            "used_npus": preflight.get("used_npus"),
        },
        "dataset_audit_pass": {
            "passed": (
                audit.get("task_count") == expected_tasks
                and audit.get("invalid_tasks") == []
            ),
            "task_count": audit.get("task_count"),
            "invalid_tasks": audit.get("invalid_tasks"),
        },
    }


def source_list_check(
    control: Path,
    dataset: Path,
    *,
    expected_tasks: int,
) -> dict[str, Any]:
    list_path = control / "source_dirs_n27.txt"
    meta = _load_json(control / "source_dirs_n27.sha256.json")
    try:
        entries = [
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError:
        entries = []
    expected_entries = meta.get("entries")
    if not isinstance(expected_entries, list):
        expected_entries = []
    source_root = (dataset / "tasks").resolve()
    invalid_entries = []
    for raw in entries:
        path = Path(raw)
        try:
            resolved = path.resolve()
            resolved.relative_to(source_root)
        except (OSError, ValueError):
            invalid_entries.append(raw)
            continue
        if not resolved.is_dir():
            invalid_entries.append(raw)
    actual_sha = sha256(list_path) if list_path.is_file() else None
    sha256_matches = actual_sha == meta.get("sha256")
    unique_entries = (
        len(entries) == expected_tasks
        and len(set(entries)) == expected_tasks
    )
    return {
        "passed": (
            list_path.is_file()
            and sha256_matches
            and len(entries) == expected_tasks
            and unique_entries
            and entries == expected_entries
            and not invalid_entries
        ),
        "expected_sha256": meta.get("sha256"),
        "actual_sha256": actual_sha,
        "sha256_matches": sha256_matches,
        "task_count": len(entries),
        "unique_task_count": len(set(entries)),
        "unique_entries": unique_entries,
        "invalid_entries": invalid_entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--secret", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--formal-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=27)
    parser.add_argument("--required-npus", default="3,4,5,6,7")
    parser.add_argument("--expected-model", default="qwen3.8-max-preview")
    parser.add_argument("--expected-context", type=int, default=1000000)
    parser.add_argument(
        "--allow-existing-formal-output",
        action="store_true",
        help="Allow an existing output when checking between formal arms.",
    )
    args = parser.parse_args()
    required_npus = [
        int(value) for value in args.required_npus.split(",") if value
    ]

    code_expected = _load_files(args.control / "code_snapshot.sha256.json")
    source_expected = _load_files(
        args.control / "source_snapshot_effective.sha256.json")
    code_actual = file_manifest(
        args.root, skip_runtime=True, code_root=True)
    source_actual = file_manifest(
        args.dataset, skip_runtime=True, code_root=False)
    control_expected = _load_files(
        args.control / "control_plane.sha256.json")
    control_actual = selected_file_manifest(
        args.control, control_expected)
    control_source = selected_file_manifest(
        args.root / "experiments" / "v5_ablation", control_expected)
    checks: dict[str, Any] = {
        "code_snapshot": compare_manifest(code_expected, code_actual),
        "source_snapshot": compare_manifest(source_expected, source_actual),
        "control_plane_snapshot": compare_manifest(
            control_expected, control_actual),
        "control_plane_matches_code": compare_manifest(
            control_source, control_actual),
    }

    kb_meta = json.loads(
        (args.control / "kb_snapshot.sha256.json").read_text(encoding="utf-8"))
    checks["kb_snapshot"] = {
        "passed": args.kb.is_file() and sha256(args.kb) == kb_meta.get("sha256"),
        "expected_sha256": kb_meta.get("sha256"),
        "actual_sha256": sha256(args.kb) if args.kb.is_file() else None,
    }
    source_writable = [
        str(path.relative_to(args.dataset))
        for path in [args.dataset, *sorted(args.dataset.rglob("*"))]
        if _write_bits(path)
    ]
    checks["source_read_only"] = {
        "passed": not source_writable,
        "writable_paths": source_writable[:100],
        "writable_path_count": len(source_writable),
    }
    checks["kb_read_only"] = {
        "passed": args.kb.is_file() and _write_bits(args.kb) == 0,
        "mode": oct(stat.S_IMODE(args.kb.stat().st_mode))
        if args.kb.exists() else None,
    }
    secret_mode = (
        stat.S_IMODE(args.secret.stat().st_mode)
        if args.secret.is_file() else None
    )
    checks["secret_mode"] = {
        "passed": secret_mode == 0o600,
        "mode": oct(secret_mode) if secret_mode is not None else None,
    }
    checks.update(preflight_artifact_checks(
        args.control,
        expected_tasks=args.expected_tasks,
        required_npus=required_npus,
        expected_model=args.expected_model,
        expected_context=args.expected_context,
    ))
    checks["source_list"] = source_list_check(
        args.control,
        args.dataset,
        expected_tasks=args.expected_tasks,
    )
    checks["formal_output_absent"] = {
        "passed": args.allow_existing_formal_output or not args.formal_output.exists(),
        "path": str(args.formal_output),
        "exists": args.formal_output.exists(),
        "required_absent": not args.allow_existing_formal_output,
    }
    passed = all(check["passed"] for check in checks.values())
    payload = {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
