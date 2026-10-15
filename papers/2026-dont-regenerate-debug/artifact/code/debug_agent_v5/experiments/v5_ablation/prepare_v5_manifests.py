#!/usr/bin/env python3
"""Freeze V5 code, dataset, KB, and redacted provider metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
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
CASE_EXCLUDES = {
    "debug_status.json",
    "experiment_manifest.json",
    "model.json",
    "round_summary.json",
    "run_summary.json",
}
ARMS = (
    "full",
    "no_kb",
    "no_diagnostic_evidence",
    "no_loopguard",
    "no_anticheat",
    "no_fulleval",
    "baseline",
)
PROFILE_ABLATIONS = {
    "full": [],
    "no_kb": ["kb"],
    "no_diagnostic_evidence": [
        "diagnostic_evidence", "forensics", "probe", "kb",
    ],
    "no_loopguard": ["loop_guard"],
    "no_anticheat": ["anticheat"],
    "no_fulleval": ["full_eval"],
    "baseline": [
        "forensics", "probe", "kb", "loop_guard", "gate_a",
        "full_eval", "recovery", "anticheat_detect_only",
    ],
}
CONTROL_PLANE_FILES = (
    "Dockerfile.v5-agent",
    "claude_qwen38max.sh",
    "launch_v5_ablation.sh",
    "manage_v5_container.sh",
    "blocked_kb.json",
    "blocked_forensics.py",
    "prepare_v5_manifests.py",
    "prepare_v5_runtime.sh",
    "qwen38max.settings.json",
    "smoke_qwen38max_provider.sh",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any, mode: int = 0o644) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(mode)


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


def command_output(*args: str) -> str:
    proc = subprocess.run(args, text=True, capture_output=True, check=False)
    return (proc.stdout or proc.stderr).strip()


def _case_count(path: Path) -> int:
    return sum(1 for line in path.read_text(errors="replace").splitlines()
               if line.strip())


def _case_digest(path: Path) -> str:
    rows: list[str] = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            rows.append(line)
        else:
            rows.append(json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    digest = hashlib.sha256()
    for row in sorted(rows):
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _looks_like_case_jsonl(path: Path) -> bool:
    if path.name.startswith(("_", ".")) or path.name in CASE_EXCLUDES:
        return False
    if not path.name.endswith((".json", ".json.bak", ".json.full")):
        return False
    lines = [line for line in path.read_text(errors="replace").splitlines()
             if line.strip()]
    if not lines:
        return True
    try:
        value = json.loads(lines[0])
    except (TypeError, ValueError):
        return False
    return isinstance(value, dict) and "inputs" in value


def _active_name(path: Path) -> str:
    if path.name.endswith(".json.bak"):
        return path.name[:-4]
    if path.name.endswith(".json.full"):
        return path.name[:-5]
    return path.name


def _coverage_row(task: Path) -> dict[str, Any]:
    candidates = sorted(
        path for path in task.iterdir()
        if path.is_file() and _looks_like_case_jsonl(path)
    )
    active_candidates = [
        path for path in candidates
        if path.name.endswith(".json")
        and not path.name.endswith((".json.bak", ".json.full"))
    ]
    if not candidates or not active_candidates:
        return {
            "active_json": None,
            "full_json": None,
            "full_eval_applicable": False,
            "coverage_equivalent": None,
        }
    active = max(active_candidates, key=lambda path: (_case_count(path), path.name))
    full = max(
        candidates,
        key=lambda path: (
            _case_count(path),
            path.name.endswith((".json.bak", ".json.full")),
            path.name,
        ),
    )
    active_count = _case_count(active)
    full_count = _case_count(full)
    active_digest = _case_digest(active)
    full_digest = _case_digest(full)
    equivalent = active_count == full_count and active_digest == full_digest
    return {
        "active_json": active.name,
        "active_cases": active_count,
        "active_case_set_sha256": active_digest,
        "full_json": full.name,
        "full_cases": full_count,
        "full_case_set_sha256": full_digest,
        "coverage_equivalent": equivalent,
        "full_eval_applicable": (
            full != active and (
                full_count > active_count
                or (full_count == active_count and not equivalent)
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()

    args.control.mkdir(parents=True, exist_ok=True)
    task_root = args.dataset / "tasks"
    tasks = sorted(path for path in task_root.glob("level*/*") if path.is_dir())
    if len(tasks) != 27:
        raise SystemExit(f"expected 27 tasks, found {len(tasks)}")

    required = ("model.py", "model_new_ascendc.py", "kernel")
    rows = []
    invalid = []
    missing_optional_metadata = []
    legacy_runtime = []
    source_lines = []
    for task in tasks:
        rel = str(task.relative_to(task_root))
        missing = [name for name in required if not (task / name).exists()]
        case_files = sorted(
            path.name for path in task.iterdir()
            if path.is_file() and _looks_like_case_jsonl(path)
        )
        json_files = sorted(
            name for name in case_files
            if name.endswith(".json")
            and not name.endswith((".json.bak", ".json.full"))
        )
        bak_files = sorted(
            name for name in case_files
            if name.endswith((".json.bak", ".json.full"))
        )
        runtime = sorted(
            path.name
            for path in task.iterdir()
            if path.name in RUNTIME_NAMES
            or path.name.startswith("_claude_result")
            or path.name == "_anticheat.json"
        )
        row = {
            "task": rel,
            "source": str(task),
            "missing_required": missing,
            "operator_json": json_files,
            "operator_json_bak": bak_files,
            "full_eval_coverage": _coverage_row(task),
            "legacy_runtime_artifacts_filtered_by_reset_target": runtime,
        }
        rows.append(row)
        source_lines.append(str(task))
        if missing or not json_files or not bak_files:
            if missing:
                invalid.append(rel)
            if not json_files or not bak_files:
                missing_optional_metadata.append(rel)
        if runtime:
            legacy_runtime.append(rel)

    if invalid:
        raise SystemExit(f"dataset audit failed for: {', '.join(invalid)}")

    source_list_path = args.control / "source_dirs_n27.txt"
    source_list_path.write_text(
        "\n".join(source_lines) + "\n", encoding="utf-8"
    )
    write_json(
        args.control / "source_dirs_n27.sha256.json",
        {
            "schema_version": 1,
            "path": str(source_list_path),
            "sha256": sha256(source_list_path),
            "task_count": len(source_lines),
            "entries": source_lines,
        },
    )
    write_json(
        args.control / "dataset_audit.json",
        {
            "dataset_root": str(args.dataset),
            "task_count": len(tasks),
            "valid_task_count": len(tasks) - len(invalid),
            "invalid_tasks": invalid,
            "missing_optional_operator_metadata": missing_optional_metadata,
            "legacy_runtime_task_count": len(legacy_runtime),
            "legacy_runtime_note": (
                "These source artifacts are excluded by supervisor reset_target and are not "
                "copied into arm task workspaces."
            ),
            "tasks": rows,
        },
    )
    write_json(
        args.control / "dataset_manifest.sha256.json",
        {
            "root": str(args.dataset),
            "files": file_manifest(args.dataset, skip_runtime=False),
        },
    )
    write_json(
        args.control / "source_snapshot_effective.sha256.json",
        {
            "root": str(args.dataset),
            "files": file_manifest(args.dataset, skip_runtime=True),
        },
    )
    write_json(
        args.control / "code_snapshot.sha256.json",
        {
            "root": str(args.root),
            "git_commit": args.git_commit,
            "git_branch": "v5-ablation",
            "excluded_top_level": sorted(CODE_EXCLUDED_TOP_LEVEL),
            "files": file_manifest(
                args.root, skip_runtime=True, code_root=True),
        },
    )
    control_files: dict[str, str] = {}
    for name in CONTROL_PLANE_FILES:
        source = args.root / "experiments" / "v5_ablation" / name
        deployed = args.control / name
        if not source.is_file() or not deployed.is_file():
            raise SystemExit(
                f"missing control-plane file source={source} deployed={deployed}"
            )
        source_sha = sha256(source)
        deployed_sha = sha256(deployed)
        if source_sha != deployed_sha:
            raise SystemExit(
                f"control-plane drift before freeze: {name} "
                f"repo={source_sha} deployed={deployed_sha}"
            )
        control_files[name] = deployed_sha
    write_json(
        args.control / "control_plane.sha256.json",
        {
            "root": str(args.control),
            "source_root": str(args.root / "experiments" / "v5_ablation"),
            "files": control_files,
        },
    )

    kb_data = json.loads(args.kb.read_text(encoding="utf-8"))
    kb_entries = kb_data if isinstance(kb_data, list) else kb_data.get("entries", [])
    write_json(
        args.control / "kb_snapshot.sha256.json",
        {
            "path": str(args.kb),
            "entries": len(kb_entries),
            "sha256": sha256(args.kb),
            "read_only": True,
        },
    )
    write_json(
        args.control / "provider_config_redacted.json",
        {
            "model": "qwen3.8-max-preview",
            "model_context_window": 1000000,
            "base_url": "https://provider.example.invalid/anthropic",
            "providers": [
                {
                    "name": "provider-slot-a",
                    "model": "qwen3.8-max-preview",
                    "base_url": (
                        "https://provider.example.invalid/anthropic"
                    ),
                }
            ],
            "credential_file": "/workspace/debug_agent/.secrets/provider_config.json",
            "credential_material_in_manifest": False,
            "assignment_mode": "fixed_single_provider",
            "quota_query_supported": False,
            "usage_query_enabled": False,
            "declared_parallel_limit": 5,
        },
    )
    write_json(
        args.control / "environment_snapshot.json",
        {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "docker": command_output("docker", "--version"),
            "npu_smi": command_output("npu-smi", "info"),
            "container_image": "ascendc-v5-agent:20260724",
            "container_name": "v5_cann",
            "containers": ["v5_cann"] * 5,
            "npus": [3, 4, 5, 6, 7],
        },
    )
    arm_dir = args.control / "arm_manifests"
    arm_dir.mkdir(exist_ok=True)
    common = {
        "task_count": 27,
        "source_dirs_file": str(args.control / "source_dirs_n27.txt"),
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
    for arm in ARMS:
        kb_enabled = arm not in {
            "no_kb", "no_diagnostic_evidence", "baseline",
        }
        write_json(
            arm_dir / f"arm_{arm}.json",
            {
                "schema_version": 1,
                "arm": arm,
                "ablated_capabilities": sorted(PROFILE_ABLATIONS[arm]),
                "kb_path": str(args.kb) if kb_enabled else None,
                "kb_read_only": kb_enabled,
                "kb_filesystem_masked": not kb_enabled,
                "forensics_script_masked": (
                    arm in {"no_diagnostic_evidence", "baseline"}
                ),
                **common,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
