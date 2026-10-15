#!/usr/bin/env python3
"""Prepare an isolated Qwen rerun of the original K3 Regenerate subset.

The supplementary arm copies only the frozen operator specifications from the
original experiment. It does not copy candidates, journals, validation output,
or any other generated state, so every rerun starts from a fresh session while
the original 81-run experiment remains immutable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prepare_experiment import (
    NPU_DEVICE_GID,
    SESSION_NAMESPACE,
    copy_spec,
    escaped_project_dir,
    install_allowed_regenerate_skills,
    secure_parent_traversal,
    secure_regenerate_cwd,
)


DEFAULT_ORIGINAL_ROOT = Path("/workspace/regenerate_experiment")
DEFAULT_SUPPLEMENT_ROOT = Path("/workspace/debug_agent")
DEFAULT_MODEL = "qwen3.8-max-preview"
DEFAULT_CREDENTIAL_NAME = "paper-provider"
DEFAULT_UID_BASE = 32000


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: Any, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.chmod(mode)
    temporary.replace(path)


def symlink_exact(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        raise RuntimeError(f"refusing to replace mismatched symlink: {link}")
    if link.exists():
        raise RuntimeError(f"refusing to replace existing path: {link}")
    link.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def snapshot_source(source_repo: Path, evidence_dir: Path) -> None:
    head = subprocess.run(
        ["git", "-C", str(source_repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    status = subprocess.run(
        ["git", "-C", str(source_repo), "status", "--porcelain=v1", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "ascendopgenagent_head_before_step3.txt").write_text(head)
    (evidence_dir / "ascendopgenagent_status_before_step3.txt").write_bytes(status)


def copy_harness_snapshot(source: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".json"}:
            continue
        target = destination / path.name
        shutil.copy2(path, target)
        copied.append(path.name)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--supplement-root", type=Path, default=DEFAULT_SUPPLEMENT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--credential-name", default=DEFAULT_CREDENTIAL_NAME)
    parser.add_argument("--uid-base", type=int, default=DEFAULT_UID_BASE)
    parser.add_argument("--expected-runs", type=int, default=14)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    original_root = args.original_root.resolve()
    supplement_root = args.supplement_root.resolve()
    original_harness = original_root / "experiment_harness"
    supplement_harness = supplement_root / "experiment_harness"
    manifest_path = supplement_harness / "run_manifest.json"

    if manifest_path.exists():
        raise RuntimeError(
            f"supplement already prepared; refusing to overwrite: {manifest_path}"
        )
    if supplement_root.exists() and any(supplement_root.iterdir()):
        raise RuntimeError(
            f"supplement root is non-empty; refusing to reuse: {supplement_root}"
        )

    original_manifest = load(original_harness / "run_manifest.json")
    selected = [
        row
        for row in original_manifest["runs"]
        if row.get("arm") == "regenerate"
        and row.get("same_model_eligible") is True
        and str(row.get("model_expected", "")).lower() in {"k3", "k3[1m]", "kimi-k3"}
    ]
    if len(selected) != args.expected_runs:
        raise RuntimeError(
            f"expected {args.expected_runs} original K3 runs, found {len(selected)}"
        )

    supplement_root.mkdir(parents=True, exist_ok=True)
    copied_harness = copy_harness_snapshot(original_harness, supplement_harness)
    symlink_exact(
        original_root / "vendor" / "asc-devkit",
        supplement_root / "vendor" / "asc-devkit",
    )
    symlink_exact(
        original_root / "vendor" / "cannbot-skills",
        supplement_root / "vendor" / "cannbot-skills",
    )
    symlink_exact(
        original_root / ".secrets" / "qwen3_8_max_preview.json",
        supplement_root / ".secrets" / "qwen3_8_max_preview.json",
    )
    plugin_runtime = (
        original_root / "runtime" / "ops-direct-invoke-step3-sanitized"
    ).resolve()
    if not plugin_runtime.is_dir():
        raise FileNotFoundError(f"sanitized plugin missing: {plugin_runtime}")

    records: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    cohort_id = "qwen_rerun_original_k3_regenerate_20260725"
    runs_root = supplement_root / "runs"
    for index, original in enumerate(selected):
        operator = str(original["operator"])
        trial = int(original["trial"])
        operator_key = operator.replace("/", "_")
        cwd = runs_root / operator_key / "regenerate" / f"trial{trial}"
        source_spec = Path(str(original["cwd"])) / "spec"
        if not source_spec.is_dir():
            raise FileNotFoundError(f"original frozen spec missing: {source_spec}")

        copy_spec(source_spec, cwd / "spec")
        install_allowed_regenerate_skills(supplement_root, cwd)
        for path in (cwd / "evidence", cwd / "validation", cwd / ".home"):
            path.mkdir(parents=True, exist_ok=True)

        session_id = str(
            uuid.uuid5(
                SESSION_NAMESPACE,
                f"{cohort_id}:{original['run_id']}:{args.model}",
            )
        )
        container_uid = args.uid_base + index
        project_dir = (
            cwd
            / ".home"
            / ".claude"
            / "projects"
            / escaped_project_dir(cwd)
        )
        run_id = f"regen-qwen-k3-control:{operator}:t{trial}"
        record_path = cwd / "run_record.json"
        record = {
            "run_id": run_id,
            "operator": operator,
            "arm": "regenerate",
            "trial": trial,
            "cwd": str(cwd),
            "expected_session_id": session_id,
            "model_expected": args.model,
            "context_window_tokens": 1048576,
            "provider": "anthropic-compatible-qwen",
            "credential_name": args.credential_name,
            "repair_policy": "repair_allowed",
            "final_result_path": str(cwd / "evidence" / "final_stdout.json"),
            "journal_main_path": str(project_dir / f"{session_id}.jsonl"),
            "subagent_dir": str(project_dir / session_id / "subagents"),
            "full_eval_path": str(cwd / "validation" / "full_eval.json"),
            "anticheat_path": str(cwd / "validation" / "anticheat.json"),
            "run_record_path": str(record_path),
            "source_type": "fresh_spec_qwen_control_for_original_k3",
            "plugin_runtime": str(plugin_runtime),
            "same_model_eligible": False,
            "supplementary_model_control": True,
            "source_k3_run_id": original["run_id"],
            "source_k3_model": original.get("model_expected"),
            "source_k3_trial_cwd": original.get("cwd"),
            "container_uid": container_uid,
            "container_gid": NPU_DEVICE_GID,
        }
        atomic_json(
            record_path,
            {
                **record,
                "status": "prepared",
                "session_id": session_id,
                "fresh_start": True,
                "resume_session": False,
                "historical_llm_cost_included": False,
            },
        )
        secure_regenerate_cwd(cwd, container_uid, NPU_DEVICE_GID)
        records.append(record)
        comparison_rows.append(
            {
                "original_k3_run_id": original["run_id"],
                "original_k3_operator": operator,
                "original_k3_trial": trial,
                "qwen_control_run_id": run_id,
                "qwen_control_model": args.model,
                "same_operator": True,
                "same_trial_index": True,
                "fresh_session": True,
            }
        )

    secure_parent_traversal(runs_root)
    try:
        os.chown(supplement_root, 0, 0)
        supplement_root.chmod(0o755)
    except PermissionError:
        pass

    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "manifest_id": "cannbot_regenerate_qwen_rerun_k3_20260725",
        "cohort_id": cohort_id,
        "generated_at": generated_at,
        "purpose": (
            "rerun every original K3 Regenerate run with Qwen under the same "
            "harness configuration to isolate model-choice effects"
        ),
        "original_experiment_root": str(original_root),
        "original_manifest": str(original_harness / "run_manifest.json"),
        "selection_rule": (
            "arm=regenerate AND same_model_eligible=true AND model_expected=K3"
        ),
        "original_runs_preserved": True,
        "fresh_qwen_sessions": True,
        "runs": records,
    }
    atomic_json(manifest_path, manifest)
    atomic_json(
        supplement_harness / "qwen_k3_control_pairs.json",
        {
            "schema_version": 1,
            "generated_at": generated_at,
            "pairs": comparison_rows,
        },
    )
    snapshot_source(
        Path("/workspace/debug_agent"),
        supplement_root / "evidence",
    )

    intended_runner = [
        "python3",
        str(supplement_harness / "run_regenerate.py"),
        "--manifest",
        str(manifest_path),
        "--experiment-root",
        str(supplement_root),
        "--key-config",
        str(supplement_root / ".secrets" / "qwen3_8_max_preview.json"),
        "--model",
        args.model,
        "--npus",
        "0,1,2,3,4",
        "--concurrency",
        "5",
        "--slots-per-key",
        "5",
        "--provider-retries",
        "0",
        "--deliverable-retries",
        "3",
        "--repair-invalid-completed",
        "--max-turns",
        "240",
        "--timeout",
        "21600",
        "--preferred-provider",
        args.credential_name,
        "--provider-abort-cooldown-seconds",
        "300",
        "--provider-quota-cooldown-seconds",
        "21600",
        "--quota-retry-wait",
        "60",
        "--provider-refresh-seconds",
        "60",
        "--quota-timeout",
        "20",
        "--queue-order",
        "reverse_trial_then_operator",
    ]
    atomic_json(
        supplement_harness / "regenerate_isolated_restart.json",
        {
            "schema_version": 1,
            "updated_at": generated_at,
            "generation_service_state": "prepared_qwen_rerun_of_original_k3",
            "generation_resume_requires_user_confirmation": False,
            "runner_pid": None,
            "finalizer_pid": None,
            "runner_command_argv": intended_runner,
            "runner_log": str(supplement_harness / "regenerate_runner.log"),
            "finalizer_log": str(supplement_harness / "step3_finalizer.log"),
            "model": args.model,
            "credential_name": args.credential_name,
            "key_config": str(
                supplement_root / ".secrets" / "qwen3_8_max_preview.json"
            ),
            "queue_order_scope": "trial3_to_trial1_manifest_reverse",
            "original_k3_run_count": len(records),
            "original_experiment_immutable": True,
            "fresh_sessions": True,
            "same_configuration_except_model_and_isolated_output_root": True,
        },
        mode=0o600,
    )
    print(
        json.dumps(
            {
                "supplement_root": str(supplement_root),
                "prepared_runs": len(records),
                "operators": sorted({row["operator"] for row in records}),
                "trials": {
                    str(trial): sum(row["trial"] == trial for row in records)
                    for trial in (3, 2, 1)
                },
                "copied_harness_files": len(copied_harness),
                "manifest": str(manifest_path),
                "comparison_map": str(
                    supplement_harness / "qwen_k3_control_pairs.json"
                ),
                "original_experiment_preserved": True,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
