#!/usr/bin/env python3
"""Prepare an isolated Kimi K3 copy of the 27x1 CANNBot Debug C2 arm.

The input source is the immutable ``frozen_inputs`` tree of the Qwen C2 arm.
No Qwen task workspace, journal, candidate, validation result, or provider
state is copied into this experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from prepare_cannbot_debug_c2 import (
    FORBIDDEN_INPUT_NAMES,
    atomic_json,
    copy_spec,
    install_skill,
    secure_trial,
    sha256,
    tree_manifest,
)
from prepare_experiment import make_read_only, secure_parent_traversal


ARM = "cannbot-debug"
MODEL = "k3"
CONTEXT_WINDOW = 1_048_576
SESSION_NAMESPACE = uuid.UUID("5ed05283-96fa-4857-b4ce-44ff1a3b535e")
UID_BASE = 39000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def copy_reference_frozen(source: Path, destination: Path) -> dict[str, str]:
    """Copy and verify the exact pre-debug input tree from the reference arm."""

    manifest_path = source / "FROZEN_INPUT_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"reference frozen manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError(f"reference frozen manifest has no files: {manifest_path}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "._*"),
    )
    for relative, digest in expected.items():
        path = destination / relative
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"frozen input hash mismatch: {path}")
    actual_payload = {
        str(path.relative_to(destination)): sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.name != "FROZEN_INPUT_MANIFEST.json"
    }
    if actual_payload != expected:
        raise RuntimeError(
            f"reference frozen payload differs from manifest: {source}"
        )
    return expected


def copy_task_from_frozen(frozen: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(frozen, destination, symlinks=True)
    (destination / "FROZEN_INPUT_MANIFEST.json").unlink(missing_ok=True)
    leaked = [
        path
        for path in destination.rglob("*")
        if path.name in FORBIDDEN_INPUT_NAMES
    ]
    if leaked:
        raise RuntimeError(f"historical outcome leakage in frozen input: {leaked}")


def reference_rows(reference_root: Path) -> list[dict[str, Any]]:
    manifest_path = reference_root / "experiment_harness" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = [
        row
        for row in manifest.get("runs", [])
        if isinstance(row, dict) and row.get("arm") == ARM
    ]
    if len(rows) != 27:
        raise RuntimeError(f"expected 27 reference rows, found {len(rows)}")
    return rows


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("/workspace/cannbot_debug_experiment"),
    )
    parser.add_argument(
        "--reference-experiment",
        type=Path,
        default=Path("/workspace/cannbot_debug_experiment"),
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
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    root = args.experiment_root.resolve()
    reference_root = args.reference_experiment.resolve()
    if root == reference_root:
        raise SystemExit("experiment root must differ from the Qwen reference")
    if root.exists() and any(root.iterdir()) and not args.force:
        raise SystemExit(f"refusing to overwrite non-empty experiment root: {root}")
    if args.force and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    rows = reference_rows(reference_root)
    plugin = root / "runtime" / "minimal-cannbot-debug-plugin"
    (plugin / ".claude-plugin").mkdir(parents=True)
    atomic_json(
        plugin / ".claude-plugin" / "plugin.json",
        {
            "name": "cannbot-debug-c2-kimi-minimal",
            "version": "1.0.0",
            "description": "Config-only host for the frozen CANNBot precision-debug skill.",
        },
    )
    make_read_only(plugin)

    skill_sources = {
        "ascendc-precision-debug": (
            args.cannbot_vendor / "ops" / "ascendc-precision-debug"
        ),
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

    records: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    frozen_root = root / "frozen_inputs"
    runs_root = root / "runs"
    for index, reference in enumerate(rows):
        operator = str(reference["operator"])
        operator_key = operator.replace("/", "_")
        reference_frozen = Path(reference["frozen_input_path"])
        reference_evidence = Path(reference["input_evidence_path"])
        frozen = frozen_root / operator_key
        frozen_hashes = copy_reference_frozen(reference_frozen, frozen)
        reference_tree_digest = str(reference["frozen_input_tree_sha256"])
        copied_tree_digest = hashlib.sha256(
            json.dumps(frozen_hashes, sort_keys=True).encode()
        ).hexdigest()
        if copied_tree_digest != reference_tree_digest:
            raise RuntimeError(
                f"reference tree digest mismatch for {operator}: "
                f"{copied_tree_digest} != {reference_tree_digest}"
            )
        make_read_only(frozen)

        uid = args.uid_base + index
        cwd = runs_root / operator_key / ARM / "trial1"
        task = cwd / "task"
        copy_task_from_frozen(frozen, task)
        copy_spec(frozen, cwd / "spec")
        evidence_dir = cwd / "input_evidence"
        evidence_dir.mkdir(parents=True)
        shutil.copy2(reference_evidence, evidence_dir / "forensics_report_0.json")
        atomic_json(
            evidence_dir / "input_evidence_manifest.json",
            {
                "operator": operator,
                "source": str(reference_evidence),
                "sha256": sha256(evidence_dir / "forensics_report_0.json"),
                "reference_qwen_run_id": reference["run_id"],
            },
        )

        skill_root = cwd / ".claude" / "skills"
        installed_hashes = {
            name: install_skill(source_path, skill_root / name)
            for name, source_path in skill_sources.items()
        }
        if installed_hashes != shared_skill_hashes:
            raise RuntimeError(f"CANNBot skill copy hash mismatch for {operator}")
        atomic_json(
            cwd / ".claude" / "settings.json",
            {"env": {"HOME": str(cwd / ".home")}},
        )

        adapter = cwd / "candidate_adapter"
        adapter.mkdir(parents=True)
        shutil.copy2(task / "model_new_ascendc.py", adapter / "model_new_ascendc.py")
        (adapter / "operator_dir.txt").write_text("task\n")
        for path in (cwd / ".home", cwd / "evidence", cwd / "validation"):
            path.mkdir(parents=True, exist_ok=True)

        session_id = str(
            uuid.uuid5(
                SESSION_NAMESPACE,
                f"c2-cannbot-debug-kimi-k3:{operator}:trial1",
            )
        )
        from prepare_experiment import escaped_project_dir

        project_dir = (
            cwd / ".home" / ".claude" / "projects" / escaped_project_dir(cwd)
        )
        record_path = cwd / "run_record.json"
        record = {
            "run_id": f"cannbot-debug-kimi:{operator}:t1",
            "operator": operator,
            "arm": ARM,
            "trial": 1,
            "status": "prepared",
            "cwd": str(cwd),
            "task_dir": str(task),
            "expected_session_id": session_id,
            "model_expected": MODEL,
            "provider": "kimi-code-anthropic-compatible",
            "credential_name_expected": "dynamic:kimi_keys_agent2.json",
            "credential_exclusion": ["providera-kimi-key"],
            "context_window_tokens": CONTEXT_WINDOW,
            "max_turns": 240,
            "soft_task_turns": 480,
            "max_task_turns": 600,
            "max_internal_attempts": 5,
            "timeout_seconds": 43200,
            "repair_policy": "debug_existing_candidate_only",
            "input_source_path": str(reference_frozen),
            "frozen_input_path": str(frozen),
            "frozen_input_manifest_path": str(
                frozen / "FROZEN_INPUT_MANIFEST.json"
            ),
            "frozen_input_tree_sha256": copied_tree_digest,
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
            "source_type": "exact_qwen_c2_frozen_input_copy",
            "reference_experiment": str(reference_root),
            "reference_qwen_run_id": reference["run_id"],
            "reference_qwen_frozen_input_tree_sha256": reference_tree_digest,
            "same_model_eligible": True,
            "comparison_model": MODEL,
            "container_uid": uid,
            "container_gid": int(reference.get("container_gid", 1001)),
            "cannbot_commit": "279417465ecfbf56de6b7a6dc9f3814db600adb8",
            "cannbot_skill_sha256": sha256(
                skill_root / "ascendc-precision-debug" / "SKILL.md"
            ),
        }
        atomic_json(record_path, record)
        secure_trial(cwd, uid)
        records.append(record)
        parity_rows.append(
            {
                "operator": operator,
                "reference_qwen_run_id": reference["run_id"],
                "reference_tree_sha256": reference_tree_digest,
                "kimi_tree_sha256": copied_tree_digest,
                "input_equal": copied_tree_digest == reference_tree_digest,
                "forensics_sha256": sha256(reference_evidence),
            }
        )

    make_read_only(frozen_root)
    secure_parent_traversal(runs_root)
    root.chmod(0o755)
    manifest = {
        "schema_version": 2,
        "manifest_id": "cannbot_c2_cannbot_debug_kimi_k3_n27_20260726",
        "cohort_id": "v3_33_predebug_eligible_n27_frozen_20260726",
        "generated_at": now(),
        "arm": ARM,
        "trials_per_operator": 1,
        "operators": [str(row["operator"]) for row in rows],
        "cannbot_commit": "279417465ecfbf56de6b7a6dc9f3814db600adb8",
        "model": MODEL,
        "context_window_tokens": CONTEXT_WINDOW,
        "max_turns": 240,
        "soft_task_turns": 480,
        "max_task_turns": 600,
        "max_internal_attempts": 5,
        "timeout_seconds": 43200,
        "reference_qwen_experiment": str(reference_root),
        "runs": records,
    }
    harness = root / "experiment_harness"
    atomic_json(harness / "run_manifest.json", manifest)
    atomic_json(
        harness / "input_parity_with_qwen.json",
        {
            "generated_at": now(),
            "requested": len(parity_rows),
            "equal": sum(row["input_equal"] for row in parity_rows),
            "rows": parity_rows,
        },
    )
    atomic_json(
        harness / "frozen_protocol.json",
        {
            "arm": ARM,
            "runs": 27,
            "model": MODEL,
            "provider": "kimi-code-anthropic-compatible",
            "key_config_name": "kimi_keys_agent2.json",
            "excluded_provider": "providera-kimi-key",
            "context_window_tokens": CONTEXT_WINDOW,
            "effort": "max",
            "max_turns": 240,
            "soft_task_turns": 480,
            "max_task_turns": 600,
            "max_internal_attempts": 5,
            "timeout_seconds": 43200,
            "concurrency": args.concurrency,
            "configured_npus": list(range(8)),
            "npu_policy": "dynamic_free_910B4_no_preemption",
            "slots_per_key": 5,
            "provider_policy": "live_remaining_capacity_diverse_assignment",
            "queue_order": "reverse_trial_then_operator",
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
                "operators": len(records),
                "runs": len(records),
                "input_parity": sum(row["input_equal"] for row in parity_rows),
                "manifest": str(harness / "run_manifest.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
