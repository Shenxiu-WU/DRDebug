#!/usr/bin/env python3
"""Prepare the isolated 27x1 CANNBot precision-debug C2/C2a arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from prepare_experiment import (
    NPU_DEVICE_GID,
    chown_tree,
    escaped_project_dir,
    looks_like_cases,
    make_read_only,
    secure_parent_traversal,
)


SESSION_NAMESPACE = uuid.UUID("1b7c9a52-99aa-4996-9e94-93ae5e0d4bdd")
UID_BASE = 36000
ARM = "cannbot-debug"
MODEL = "qwen3.8-max-preview"
CONTEXT_WINDOW = 1_048_576
FORBIDDEN_INPUT_NAMES = {
    ".phase4_history.json",
    ".verify_logs",
    ".verify_status",
    "_anticheat.json",
    "_claude_result.json",
    "_claude_result_0.json",
    "debug_kernel.py",
    "debug_status.json",
    "model_new_tilelang.py",
    "precision_tuning",
    "trace.md",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_frozen_near_miss(source: Path, destination: Path) -> None:
    """Copy only the pre-debug task contract and failed implementation.

    Prior Claude transcripts, Debug status, anti-cheat outcomes, verification
    logs, TileLang alternatives, and precision_tuning outputs are deliberately
    excluded so the baseline sees no historical outcome leakage.
    """

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in ("model.py", "model.json", "model_new_ascendc.py"):
        path = source / name
        if path.is_file():
            copy_file(path, destination / name)
    for path in sorted(source.iterdir()):
        if path.name in FORBIDDEN_INPUT_NAMES:
            continue
        if looks_like_cases(path):
            copy_file(path, destination / path.name)
    for name in ("design", "kernel"):
        path = source / name
        if path.is_dir():
            shutil.copytree(
                path,
                destination / name,
                symlinks=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "._*"),
            )
    if not (destination / "model.py").is_file():
        raise FileNotFoundError(f"missing model.py: {source}")
    if not (destination / "model_new_ascendc.py").is_file():
        raise FileNotFoundError(f"missing failed candidate wrapper: {source}")
    native = [
        path
        for path in destination.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".asc"}
    ]
    if not native:
        raise FileNotFoundError(f"missing failed native implementation: {source}")
    leaked = [path for path in destination.rglob("*") if path.name in FORBIDDEN_INPUT_NAMES]
    if leaked:
        raise RuntimeError(f"historical outcome leakage: {leaked}")


def copy_spec(frozen: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(frozen.iterdir()):
        if path.is_file() and (
            path.name in {"model.py", "model.json"} or looks_like_cases(path)
        ):
            shutil.copy2(path, destination / path.name)
    if not (destination / "model.py").is_file():
        raise FileNotFoundError(f"spec/model.py missing in {frozen}")


def install_skill(source: Path, destination: Path) -> dict[str, str]:
    if not source.is_dir():
        raise FileNotFoundError(f"missing CANNBot skill: {source}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "._*"),
    )
    return tree_manifest(destination)


def secure_trial(cwd: Path, uid: int) -> None:
    chown_tree(cwd, uid, NPU_DEVICE_GID)
    cwd.chmod(0o700)
    for path in (
        cwd / ".home",
        cwd / "task",
        cwd / "candidate_adapter",
        cwd / "evidence",
        cwd / "validation",
    ):
        path.mkdir(parents=True, exist_ok=True)
        chown_tree(path, uid, NPU_DEVICE_GID)
        path.chmod(0o700)
    task = cwd / "task"
    for path in [task, *task.rglob("*")]:
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod(0o700)
        else:
            path.chmod(0o700 if os.access(path, os.X_OK) else 0o600)
    for path in (cwd / ".claude", cwd / "spec", cwd / "input_evidence"):
        make_read_only(path)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("/workspace/cannbot_debug_experiment"),
    )
    parser.add_argument(
        "--source-pool",
        type=Path,
        default=Path(
            "/workspace/debug_agent/outputs/"
            "precision_failed_clean33_compare_input_rebuilt_20260522_161422/cases"
        ),
    )
    parser.add_argument(
        "--forensics-root",
        type=Path,
        default=Path(
            "/workspace/debug_agent/outputs/"
            "precision_debug_v3_33_kimi_20260622_200425_V3-33/tasks"
        ),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(
            "/workspace/regenerate_experiment/"
            "experiment_harness/run_manifest.json"
        ),
    )
    parser.add_argument(
        "--cannbot-vendor",
        type=Path,
        default=Path(
            "/workspace/regenerate_experiment/vendor/cannbot-skills"
        ),
    )
    parser.add_argument(
        "--devkit-runtime",
        type=Path,
        default=Path(
            "/workspace/regenerate_experiment/vendor/asc-devkit"
        ),
    )
    parser.add_argument("--uid-base", type=int, default=UID_BASE)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    root = args.experiment_root
    if root.exists() and any(root.iterdir()) and not args.force:
        raise SystemExit(f"refusing to overwrite non-empty experiment root: {root}")
    if args.force and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads(args.source_manifest.read_text())
    operators = sorted(
        {
            str(row["operator"])
            for row in source_manifest["runs"]
            if row.get("arm") == "regenerate"
        }
    )
    if len(operators) != 27:
        raise RuntimeError(f"expected 27 frozen operators, found {len(operators)}")

    plugin = root / "runtime" / "minimal-cannbot-debug-plugin"
    (plugin / ".claude-plugin").mkdir(parents=True)
    atomic_json(
        plugin / ".claude-plugin" / "plugin.json",
        {
            "name": "cannbot-debug-c2-minimal",
            "version": "1.0.0",
            "description": "Config-only host for the frozen CANNBot precision-debug skill.",
        },
    )
    make_read_only(plugin)

    skill_sources = {
        "ascendc-precision-debug": args.cannbot_vendor / "ops" / "ascendc-precision-debug",
        "ascendc-docs-search": args.cannbot_vendor / "ops" / "ascendc-docs-search",
    }
    shared_skill_hashes = {
        name: tree_manifest(path)
        for name, path in skill_sources.items()
        if path.is_dir()
    }
    if set(shared_skill_hashes) != set(skill_sources):
        missing = sorted(set(skill_sources) - set(shared_skill_hashes))
        raise FileNotFoundError(f"missing required CANNBot skills: {missing}")

    records = []
    eligibility = []
    frozen_root = root / "frozen_inputs"
    runs_root = root / "runs"
    for index, operator in enumerate(operators):
        source = args.source_pool / operator
        forensic_source = (
            args.forensics_root / operator / "precision_tuning" / "forensics_report_0.json"
        )
        if not source.is_dir() or not forensic_source.is_file():
            raise FileNotFoundError(f"missing frozen input/evidence for {operator}")
        forensic = json.loads(forensic_source.read_text())
        eligible = (
            forensic.get("status") == "completed"
            and forensic.get("L0_pass") is False
            and forensic.get("all_passed") is False
        )
        if not eligible:
            raise RuntimeError(f"pre-debug eligibility failed for {operator}: {forensic}")

        operator_key = operator.replace("/", "_")
        frozen = frozen_root / operator_key
        copy_frozen_near_miss(source, frozen)
        frozen_hashes = tree_manifest(frozen)
        atomic_json(
            frozen / "FROZEN_INPUT_MANIFEST.json",
            {
                "schema_version": 1,
                "operator": operator,
                "source": str(source),
                "source_pool": str(args.source_pool),
                "files": frozen_hashes,
                "forbidden_historical_artifacts_excluded": sorted(FORBIDDEN_INPUT_NAMES),
                "created_at": now(),
            },
        )
        make_read_only(frozen)

        uid = args.uid_base + index
        cwd = runs_root / operator_key / ARM / "trial1"
        task = cwd / "task"
        shutil.copytree(frozen, task, symlinks=True)
        (task / "FROZEN_INPUT_MANIFEST.json").unlink(missing_ok=True)
        copy_spec(frozen, cwd / "spec")
        evidence_dir = cwd / "input_evidence"
        evidence_dir.mkdir(parents=True)
        shutil.copy2(forensic_source, evidence_dir / "forensics_report_0.json")
        atomic_json(
            evidence_dir / "input_evidence_manifest.json",
            {
                "operator": operator,
                "source": str(forensic_source),
                "sha256": sha256(evidence_dir / "forensics_report_0.json"),
                "status": forensic.get("status"),
                "L0_pass": forensic.get("L0_pass"),
                "all_passed": forensic.get("all_passed"),
            },
        )

        skill_root = cwd / ".claude" / "skills"
        installed_hashes = {
            name: install_skill(source_path, skill_root / name)
            for name, source_path in skill_sources.items()
        }
        if installed_hashes != shared_skill_hashes:
            raise RuntimeError(f"CANNBot skill copy hash mismatch for {operator}")
        settings = cwd / ".claude" / "settings.json"
        atomic_json(settings, {"env": {"HOME": str(cwd / ".home")}})

        adapter = cwd / "candidate_adapter"
        adapter.mkdir(parents=True)
        shutil.copy2(task / "model_new_ascendc.py", adapter / "model_new_ascendc.py")
        (adapter / "operator_dir.txt").write_text("task\n")
        for path in (cwd / ".home", cwd / "evidence", cwd / "validation"):
            path.mkdir(parents=True, exist_ok=True)

        session_id = str(
            uuid.uuid5(SESSION_NAMESPACE, f"c2-cannbot-debug:{operator}:trial1")
        )
        project_dir = (
            cwd / ".home" / ".claude" / "projects" / escaped_project_dir(cwd)
        )
        record_path = cwd / "run_record.json"
        record = {
            "run_id": f"cannbot-debug:{operator}:t1",
            "operator": operator,
            "arm": ARM,
            "trial": 1,
            "status": "prepared",
            "cwd": str(cwd),
            "task_dir": str(task),
            "expected_session_id": session_id,
            "model_expected": MODEL,
            "provider": "anthropic-compatible-qwen",
            "credential_name_expected": "provider-slot-b",
            "context_window_tokens": CONTEXT_WINDOW,
            "max_turns": 240,
            "soft_task_turns": 480,
            "max_task_turns": 600,
            "max_internal_attempts": 5,
            "timeout_seconds": 43200,
            "repair_policy": "debug_existing_candidate_only",
            "input_source_path": str(source),
            "frozen_input_path": str(frozen),
            "frozen_input_manifest_path": str(
                frozen / "FROZEN_INPUT_MANIFEST.json"
            ),
            "frozen_input_tree_sha256": hashlib.sha256(
                json.dumps(frozen_hashes, sort_keys=True).encode()
            ).hexdigest(),
            "input_evidence_path": str(
                evidence_dir / "forensics_report_0.json"
            ),
            "plugin_runtime": str(plugin),
            "devkit_runtime": str(args.devkit_runtime),
            "final_result_path": str(cwd / "evidence" / "final_stdout.json"),
            "journal_main_path": str(project_dir / f"{session_id}.jsonl"),
            "subagent_dir": str(project_dir / session_id / "subagents"),
            "full_eval_path": str(cwd / "validation" / "full_eval.json"),
            "anticheat_path": str(cwd / "validation" / "anticheat.json"),
            "run_record_path": str(record_path),
            "candidate_task_path": str(cwd / "candidate_task"),
            "source_type": "frozen_initial_near_miss",
            "same_model_eligible": True,
            "container_uid": uid,
            "container_gid": NPU_DEVICE_GID,
            "cannbot_commit": "279417465ecfbf56de6b7a6dc9f3814db600adb8",
            "cannbot_skill_sha256": sha256(
                skill_root / "ascendc-precision-debug" / "SKILL.md"
            ),
        }
        atomic_json(record_path, record)
        secure_trial(cwd, uid)
        records.append(record)
        eligibility.append(
            {
                "operator": operator,
                "status": forensic.get("status"),
                "L0_pass": forensic.get("L0_pass"),
                "all_passed": forensic.get("all_passed"),
                "forensics_sha256": sha256(forensic_source),
                "frozen_file_count": len(frozen_hashes),
                "eligible": True,
            }
        )

    make_read_only(frozen_root)
    secure_parent_traversal(runs_root)
    root.chmod(0o755)
    manifest = {
        "schema_version": 2,
        "manifest_id": "cannbot_c2_cannbot_debug_qwen3_8_max_preview_n27_20260726",
        "cohort_id": "v3_33_predebug_eligible_n27_frozen_20260726",
        "generated_at": now(),
        "arm": ARM,
        "trials_per_operator": 1,
        "operators": operators,
        "cannbot_commit": "279417465ecfbf56de6b7a6dc9f3814db600adb8",
        "model": MODEL,
        "context_window_tokens": CONTEXT_WINDOW,
        "max_turns": 240,
        "soft_task_turns": 480,
        "max_task_turns": 600,
        "max_internal_attempts": 5,
        "timeout_seconds": 43200,
        "runs": records,
    }
    harness = root / "experiment_harness"
    atomic_json(harness / "run_manifest.json", manifest)
    atomic_json(
        harness / "eligibility_preflight.json",
        {
            "generated_at": now(),
            "requested": len(operators),
            "eligible": sum(row["eligible"] for row in eligibility),
            "rows": eligibility,
        },
    )
    atomic_json(
        harness / "frozen_protocol.json",
        {
            "arm": ARM,
            "runs": 27,
            "model": MODEL,
            "provider": "anthropic-compatible-qwen",
            "credential_name": "provider-slot-b",
            "context_window_tokens": CONTEXT_WINDOW,
            "max_turns": 240,
            "soft_task_turns": 480,
            "max_task_turns": 600,
            "max_internal_attempts": 5,
            "timeout_seconds": 43200,
            "concurrency": 5,
            "slots_per_key": 5,
            "queue_order": "reverse_trial_then_operator",
            "npu_model": "Ascend 910B4",
            "debug_only": True,
            "full_regeneration_forbidden": True,
            "external_full_eval_required": True,
            "external_anticheat_required": True,
            "cost_views": ["end_to_end", "core_debug_worker"],
        },
    )
    print(
        json.dumps(
            {
                "experiment_root": str(root),
                "operators": len(operators),
                "runs": len(records),
                "eligible": sum(row["eligible"] for row in eligibility),
                "manifest": str(harness / "run_manifest.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
