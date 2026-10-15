#!/usr/bin/env python3
"""Prepare isolated N=27 experiment directories without mutating source projects."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SESSION_NAMESPACE = uuid.UUID("4ef3d799-f235-4eac-bd7f-e35a4316e805")
TRIAL_UID_BASE = 30000
NPU_DEVICE_GID = 1000
COPY_IGNORE = shutil.ignore_patterns("__pycache__", ".DS_Store", "._*", "*.pyc")
ALLOWED_REGENERATE_SKILLS = (
    "ascendc-api-best-practices",
    "ascendc-blaze-best-practice",
    "ascendc-code-review",
    "ascendc-direct-invoke-template",
    "ascendc-docs-search",
    "ascendc-env-check",
    "ascendc-regbase-best-practice",
    "ascendc-tiling-design",
    "gitcode-issue-gen",
    "gitcode-issue-handler",
    "gitcode-pr-handler",
    "gitcode-toolkit",
    "npu-arch",
    "ops-precision-standard",
    "ops-profiling",
    "torch-ascendc-op-extension",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def looks_like_cases(path: Path) -> bool:
    if not path.is_file() or not any(path.name.endswith(suffix) for suffix in (".json", ".json.bak", ".json.full")):
        return False
    if path.name in {"model.json", "debug_status.json", "run_summary.json"} or path.name.startswith("_"):
        return False
    try:
        first = next(line for line in path.read_text(errors="replace").splitlines() if line.strip())
        return isinstance(json.loads(first), dict) and "inputs" in json.loads(first)
    except (StopIteration, json.JSONDecodeError):
        return False


def copy_spec(source: Path, destination: Path) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in ("model.py", "model.json"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)
            copied.append(destination / name)
    for path in sorted(source.iterdir()):
        if looks_like_cases(path):
            shutil.copy2(path, destination / path.name)
            copied.append(destination / path.name)
    if not (destination / "model.py").is_file():
        raise FileNotFoundError(f"missing model.py in source task: {source}")
    case_files = [path for path in destination.iterdir() if looks_like_cases(path)]
    embedded_case_count = 0
    if not case_files:
        # A small subset of the frozen cohort embeds deterministic INPUT_CASES
        # directly in model.py.  That is still original spec, not a near-miss.
        tree = ast.parse((destination / "model.py").read_text())
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                target = node.targets[0] if isinstance(node, ast.Assign) else node.target
                if isinstance(target, ast.Name) and target.id == "INPUT_CASES":
                    try:
                        value = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        value = None
                    if isinstance(value, list):
                        embedded_case_count = len(value)
        if embedded_case_count == 0 or "def get_input_groups" not in (destination / "model.py").read_text():
            raise FileNotFoundError(f"missing JSONL or embedded INPUT_CASES in source task: {source}")
    forbidden = [
        path
        for path in destination.rglob("*")
        if path.name in {"model_new_ascendc.py", "model_new_tilelang.py"} or "kernel" in path.parts
    ]
    if forbidden:
        raise RuntimeError(f"near-miss leaked into regenerate spec: {forbidden}")
    hashes = {str(path.relative_to(destination)): sha256(path) for path in copied}
    (destination / "spec_manifest.json").write_text(
        json.dumps(
            {
                "source_task": str(source),
                "copied_files": hashes,
                "near_miss_present": False,
                "case_source": "jsonl" if case_files else "model.py:INPUT_CASES",
                "embedded_case_count": embedded_case_count,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    return hashes


def replace_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=True, ignore=COPY_IGNORE)


def install_allowed_regenerate_skills(experiment_root: Path, cwd: Path) -> None:
    skill_root = cwd / ".claude" / "skills"
    skill_root.mkdir(parents=True, exist_ok=True)
    vendor = experiment_root / "vendor" / "cannbot-skills"
    for name in ALLOWED_REGENERATE_SKILLS:
        candidates = (vendor / "ops" / name, vendor / "infra" / name)
        source = next((path for path in candidates if path.is_dir()), None)
        if source is None:
            raise FileNotFoundError(f"allowed CANNBot skill missing: {name}")
        target = skill_root / name
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == source.resolve():
                continue
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.symlink_to(source.resolve(), target_is_directory=True)
    installed = {path.name for path in skill_root.iterdir()}
    forbidden = installed.intersection(
        {"ascendc-precision-debug", "ascendc-crash-debug", "ascendc-runtime-debug"}
    )
    if forbidden:
        raise RuntimeError(f"forbidden debug skills installed in {cwd}: {sorted(forbidden)}")


def prepare_sanitized_plugin_runtime(experiment_root: Path) -> Path:
    """Create an isolated config-only copy whose dependency bundle is not needed.

    The upstream dependency plugin includes the three forbidden debug skills.
    The run instead discovers the allowlisted project skills installed above.
    Agent/workflow content is copied byte-for-byte; only the copied plugin
    manifest's dependency declaration is removed.
    """
    source = experiment_root / "vendor" / "cannbot-skills" / "plugins-official" / "ops-direct-invoke"
    target = experiment_root / "runtime" / "ops-direct-invoke-step3-sanitized"
    if not source.is_dir():
        raise FileNotFoundError(f"missing upstream ops-direct-invoke plugin: {source}")
    replace_tree(source, target)
    manifest_path = target / ".claude-plugin" / "plugin.json"
    manifest = load(manifest_path)
    original_dependencies = list(manifest.get("dependencies", []))
    manifest["dependencies"] = []
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (target / "STEP3_CONFIG_ADAPTATION.json").write_text(
        json.dumps(
            {
                "source_plugin": str(source),
                "source_manifest_sha256": sha256(source / ".claude-plugin" / "plugin.json"),
                "adaptation": "dependency declaration removed in isolated runtime copy",
                "reason": "upstream ops-direct-invoke-skills bundle includes three forbidden debug skills",
                "original_dependencies": original_dependencies,
                "skills_source": "per-trial allowlisted .claude/skills symlinks",
                "agent_workflow_content_modified": False,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return target


def escaped_project_dir(cwd: Path) -> str:
    # Claude Code replaces every non-alphanumeric path character (including
    # '/' and '_') with '-'.  Match the runtime directory name exactly so the
    # manifest points at the journals that token/purity audits must consume.
    return re.sub(r"[^A-Za-z0-9]", "-", str(cwd))


def chown_tree(root: Path, uid: int, gid: int) -> None:
    if uid < 0 or gid < 0:
        return
    for path in [root, *root.rglob("*")]:
        try:
            os.chown(path, uid, gid, follow_symlinks=False)
        except PermissionError:
            return


def make_read_only(root: Path) -> None:
    """Make a copied runtime/spec tree root-owned and immutable to trial UIDs."""
    if not root.exists():
        return
    for path in [root, *root.rglob("*")]:
        try:
            os.chown(path, 0, 0, follow_symlinks=False)
        except PermissionError:
            pass
        if path.is_symlink():
            continue
        mode = 0o555 if path.is_dir() or os.access(path, os.X_OK) else 0o444
        path.chmod(mode)


def secure_regenerate_cwd(cwd: Path, uid: int, gid: int) -> None:
    """Give one numeric UID its cwd while locking inputs and project config."""
    settings = cwd / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(
        json.dumps({"env": {"HOME": str(cwd / ".home")}}, indent=2) + "\n"
    )
    chown_tree(cwd, uid, gid)
    cwd.chmod(0o700)
    for path in (cwd / "evidence", cwd / "validation", cwd / ".home"):
        path.mkdir(parents=True, exist_ok=True)
        chown_tree(path, uid, gid)
        path.chmod(0o700)
    make_read_only(cwd / "spec")
    make_read_only(cwd / ".claude")


def secure_parent_traversal(root: Path) -> None:
    """Permit Claude path checks while trial directories remain UID-private."""
    if not root.exists():
        return
    for path in [root, *[item for item in root.rglob("*") if item.is_dir()]]:
        relative_parts = path.relative_to(root).parts
        if any(part.startswith("trial") for part in relative_parts):
            continue
        try:
            os.chown(path, 0, 0, follow_symlinks=False)
            # Claude's tool safety layer enumerates/stat-checks path ancestors.
            # Sibling trial contents remain protected by their own uid:0700.
            path.chmod(0o755)
        except PermissionError:
            pass


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=here / "cohort.json")
    parser.add_argument("--manifest", type=Path, default=here / "run_manifest.json")
    parser.add_argument("--uid-base", type=int, default=TRIAL_UID_BASE)
    parser.add_argument("--force", action="store_true", help="replace already prepared task/spec copies")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    cohort = load(args.cohort)
    experiment_root = Path(cohort["experiment_root"])
    source_pool = Path(cohort["source_pool"])
    historical_root = Path(cohort["historical_debug_k3_root"])
    runs_root = experiment_root / "runs"
    records: list[dict] = []
    fresh_debug_rows: list[dict[str, str]] = []
    sanitized_plugin = prepare_sanitized_plugin_runtime(experiment_root)

    for operator_index, item in enumerate(cohort["operators"]):
        operator = item["operator"]
        level, task_name = operator.split("/", 1)
        operator_key = f"{level}_{task_name}"
        source_task = source_pool / operator
        if not source_task.is_dir():
            raise FileNotFoundError(f"missing frozen source task: {source_task}")
        operator_root = runs_root / operator_key

        for trial in range(1, int(cohort["regen_trials"]) + 1):
            cwd = operator_root / "regenerate" / f"trial{trial}"
            container_uid = args.uid_base + operator_index * int(cohort["regen_trials"]) + trial
            if args.force and cwd.exists():
                shutil.rmtree(cwd)
            spec_dir = cwd / "spec"
            if args.force or not (spec_dir / "spec_manifest.json").is_file():
                if args.force and spec_dir.exists():
                    shutil.rmtree(spec_dir)
                copy_spec(source_task, spec_dir)
            install_allowed_regenerate_skills(experiment_root, cwd)
            for path in (cwd / "evidence", cwd / "validation", cwd / ".home"):
                path.mkdir(parents=True, exist_ok=True)
            session_id = str(uuid.uuid5(SESSION_NAMESPACE, f"{cohort['cohort_id']}:{operator}:regenerate:{trial}"))
            project_dir = cwd / ".home" / ".claude" / "projects" / escaped_project_dir(cwd)
            record_path = cwd / "run_record.json"
            if not record_path.exists():
                record_path.write_text(
                    json.dumps(
                        {
                            "status": "prepared",
                            "operator": operator,
                            "arm": "regenerate",
                            "trial": trial,
                            "session_id": session_id,
                            "repair_policy": "repair_allowed",
                        },
                        indent=2,
                    )
                    + "\n"
                )
            secure_regenerate_cwd(cwd, container_uid, NPU_DEVICE_GID)
            records.append(
                {
                    "run_id": f"regen:{operator}:t{trial}",
                    "operator": operator,
                    "arm": "regenerate",
                    "trial": trial,
                    "cwd": str(cwd),
                    "expected_session_id": session_id,
                    "model_expected": cohort["model"],
                    "context_window_tokens": 1048576,
                    "provider": "kimi-code",
                    "repair_policy": "repair_allowed",
                    "final_result_path": str(cwd / "evidence" / "final_stdout.json"),
                    "journal_main_path": str(project_dir / f"{session_id}.jsonl"),
                    "subagent_dir": str(project_dir / session_id / "subagents"),
                    "full_eval_path": str(cwd / "validation" / "full_eval.json"),
                    "anticheat_path": str(cwd / "validation" / "anticheat.json"),
                    "run_record_path": str(record_path),
                    "source_type": "fresh_spec",
                    "plugin_runtime": str(sanitized_plugin),
                    "same_model_eligible": True,
                    "container_uid": container_uid,
                    "container_gid": NPU_DEVICE_GID,
                }
            )

        debug_root = operator_root / "debug"
        historical = bool(item["historical_debug_k3"])
        selected_source = historical_root / operator if historical else source_task
        if not selected_source.is_dir():
            raise FileNotFoundError(f"missing debug source: {selected_source}")
        if historical:
            debug_task = debug_root / "task"
            if args.force or not debug_task.is_dir():
                replace_tree(selected_source, debug_task)
            # A zero-byte per-session result has no token or terminal evidence
            # and represents an interrupted operational attempt.  Keep the
            # source project untouched, but do not archive such invalid runs in
            # the isolated experiment copy.
            results_dir = debug_task / "precision_tuning" / "claude_results"
            if results_dir.is_dir():
                for result_path in results_dir.glob("*.json"):
                    if result_path.stat().st_size == 0:
                        result_path.unlink()
            debug_source_copy = None
        else:
            # The historical supervisor derives the destination as
            # tasks/<source.parent.name>/<source.name>; preserve level/op here.
            debug_source_copy = experiment_root / "debug_fresh_sources" / operator
            if args.force or not debug_source_copy.is_dir():
                replace_tree(selected_source, debug_source_copy)
            debug_task = experiment_root / "debug_fresh_k3" / "agent2" / "tasks" / operator
            fresh_debug_rows.append(
                {
                    "level": level,
                    "case_id": str(int(task_name.split("_", 1)[0])),
                    "op_name": task_name.split("_", 1)[1],
                    "category": "predebug_precision_failure",
                    "verify_failure_type": "precision_failed",
                    "anticheat_verdict": "CLEAN",
                    "selected_case_path": str(debug_source_copy),
                }
            )
        (debug_root / "validation").mkdir(parents=True, exist_ok=True)
        records.append(
            {
                "run_id": f"debug:{operator}",
                "operator": operator,
                "arm": "debug",
                "trial": None,
                "cwd": str(debug_root),
                "task_dir": str(debug_task),
                "model_expected": cohort["model"],
                "context_window_tokens": 1048576,
                "provider": "kimi-code",
                "claude_results_glob": "precision_tuning/claude_results/*.json",
                "full_eval_path": str(debug_root / "validation" / "full_eval.json"),
                "anticheat_path": str(debug_root / "validation" / "anticheat.json"),
                "source_type": "historical_k3_copy" if historical else "fresh_initial_near_miss",
                "same_model_eligible": True,
                "status": "copied_historical" if historical else "prepared_for_fresh_k3",
                "source_task_path": str(debug_source_copy) if debug_source_copy else str(selected_source),
            }
        )

    manifest = {
        "schema_version": 1,
        "manifest_id": "cannbot_debug_vs_regenerate_k3_20260717",
        "cohort_id": cohort["cohort_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.manifest)
    source_map = args.manifest.parent / "fresh_debug_source_map.csv"
    with source_map.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "level",
            "case_id",
            "op_name",
            "category",
            "verify_failure_type",
            "anticheat_verdict",
            "selected_case_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fresh_debug_rows)
    # Shared generation/runtime inputs are visible but not writable by trial UIDs.
    for shared in (
        experiment_root / "vendor",
        experiment_root / "runtime",
        experiment_root / "frozen_debug_engine",
        experiment_root / "debug_fresh_sources",
    ):
        make_read_only(shared)
    try:
        os.chown(experiment_root, 0, 0)
        experiment_root.chmod(0o755)
    except PermissionError:
        pass
    secure_parent_traversal(runs_root)
    print(
        json.dumps(
            {
                "operators": len(cohort["operators"]),
                "regenerate_runs": sum(record["arm"] == "regenerate" for record in records),
                "debug_runs": sum(record["arm"] == "debug" for record in records),
                "historical_debug_k3": sum(record.get("source_type") == "historical_k3_copy" for record in records),
                "fresh_debug_k3_required": sum(record.get("source_type") == "fresh_initial_near_miss" for record in records),
                "manifest": str(args.manifest),
                "fresh_debug_source_map": str(source_map),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
