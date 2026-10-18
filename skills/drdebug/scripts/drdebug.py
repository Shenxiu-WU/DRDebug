#!/usr/bin/env python3
"""Portable launcher around the public frozen engine; Python 3.10+ stdlib only.

Task code is trusted executable input when run on an NPU. prepare/doctor never
execute it. No command is interpolated into a shell. The paper archive remains
byte-identical; prepared runtimes declare their portability adapters.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import checkpoint_adapter
import nonfinite_guard
import runtime_checks

SKILL_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REL = Path("papers/2026-dont-regenerate-debug/artifact/code/debug_agent_v5")
DEBUG_REL = Path("skills/ascendc/ascendc-debug")
AST_VALIDATOR_REL = Path("skills/ascendc/ascendc-translator/scripts/validate_ascendc_impl.py")
SEED_REL = Path("experiments/v5_ablation/fixtures/initial_kb_102_cleaned_20260715.json")
MANIFEST = "drdebug-workspace.json"
RUN_STATE = "drdebug-runs.json"
TERMINAL = {"session_done", "session_aborted", "session_escalated"}
IGNORED = {".git", "__pycache__", ".DS_Store"}
WORKERS = ("ascendc-debug-agent-constructive", "ascendc-debug-agent-discovery")
ATTRIBUTION_FILES = ("LICENSE", "LICENSE-APACHE-2.0.txt", "SOURCE_NOTICE.md", "example-origin.json")
OUTCOME_CODES = {"success": 0, "failed": 1, "stopped_by_gate": 2, "stopped_by_loop_limit": 3,
                 "stopped_by_attempt_limit": 12, "stopped_by_branch_limit": 13, "timeout": 4,
                 "skipped_env_issue": 5, "skipped_unsupported_type": 6, "crashed": 7,
                 "provider_api_error": 8, "stopped_by_budget": 9, "degenerate_no_progress": 10,
                 "ablation_violation": 11}


class ContractError(ValueError):
    pass


class EngineStopped(Exception):
    def __init__(self, returncode, outcome):
        super().__init__(outcome)
        self.returncode = returncode
        self.outcome = outcome


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def source_root(skill_root=None):
    """Installed bundle first; repository fallback is for an unpackaged checkout."""
    skill = Path(skill_root or SKILL_ROOT).resolve()
    choices = [skill / "runtime/source/debug_agent_v5", skill.parents[1] / SOURCE_REL]
    for root in choices:
        if (root / DEBUG_REL / "engine/__main__.py").is_file() and (root / SEED_REL).is_file():
            return root
    raise ContractError("Frozen public runtime not found. Install the complete skill bundle (runtime/source/debug_agent_v5).")


def files(root):
    """Reject links, including hidden ones, instead of dereferencing outside input."""
    root = Path(root)
    if root.is_symlink():
        raise ContractError(f"Symlink input is not supported: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"Symlink input is not supported: {path}")
        rel = path.relative_to(root)
        if any(part in IGNORED or part.startswith("._") for part in rel.parts):
            continue
        if path.is_file():
            yield path


def tree_hashes(root, *, exclude_settings=False):
    return {p.relative_to(root).as_posix(): sha(p) for p in files(root)
            if not (exclude_settings and p.relative_to(root).as_posix() == ".claude/settings.json")}


def support_hashes(workspace):
    """Inventory the exact ancestor-discoverable support tree, without links."""
    workspace = Path(workspace)
    root = workspace / "skills"
    if root.is_symlink() or not root.is_dir():
        raise ContractError("Workspace support tree is missing or symlinked")
    directories = set(AST_VALIDATOR_REL.parents)
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"Symlink workspace support is not allowed: {path}")
        relative = path.relative_to(workspace)
        if path.is_dir():
            if relative not in directories:
                raise ContractError(f"Unexpected workspace support directory: {relative}")
        elif path.is_file():
            hashes[relative.as_posix()] = sha(path)
        else:
            raise ContractError(f"Non-regular workspace support file: {relative}")
    return hashes


def copy_tree(source, target, *, kernel=False):
    for path in files(source):
        rel = path.relative_to(source)
        if kernel and ("build" in rel.parts or path.suffix in {".so", ".o", ".a", ".pyc"}):
            continue
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def case_rows(path):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ContractError(f"Invalid JSONL case {path.name}:{number}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("inputs"), list) or not row["inputs"]:
            raise ContractError(f"Case {path.name}:{number} must have a nonempty inputs list")
        rows.append(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    if not rows:
        raise ContractError(f"Empty case file: {path.name}")
    return rows


def task_contract(task, op_name):
    task = Path(task)
    if task.is_symlink():
        raise ContractError(f"Symlink task root is not supported: {task}")
    task = task.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", op_name):
        raise ContractError("op-name must be a plain filename stem (letters, digits, _, -, .)")
    if not task.is_dir():
        raise ContractError(f"Task directory does not exist: {task}")
    list(files(task))  # reject all links without importing any task code
    required = [task / "model.py", task / "model_new_ascendc.py", task / "kernel/pybind11.cpp", task / f"{op_name}.json"]
    for path in required:
        if not path.is_file() or not path.stat().st_size:
            raise ContractError(f"Missing or empty task input: {path}")
    if not [p for p in (task / "kernel").glob("*.cpp") if p.name != "pybind11.cpp"]:
        raise ContractError("At least one native kernel/*.cpp source besides pybind11.cpp is required")
    if not re.search(r"PYBIND11_MODULE\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*,", required[2].read_text(encoding="utf-8")):
        raise ContractError("kernel/pybind11.cpp must declare its PYBIND11_MODULE binding")
    full = [task / f"{op_name}.json.bak", task / f"{op_name}.json.full"]
    full = [p for p in full if p.is_file()]
    if not full:
        raise ContractError("Supply an explicit full-case file: <op>.json.bak or <op>.json.full (equivalent compact/full sets are allowed).")
    compact_rows, full_rows = case_rows(required[-1]), case_rows(full[0])
    if len(full) > 1 and Counter(full_rows) != Counter(case_rows(full[1])):
        raise ContractError("Ambiguous full cases: .json.bak and .json.full must contain equivalent sets when both are supplied")
    # The frozen release supplies two protocol roles, not a containment
    # guarantee: Add/Cumsum/Sum have compact rows absent from their full file.
    # Keep the original bytes and let the unchanged engine evaluate each set.
    compact_set, full_set = Counter(compact_rows), Counter(full_rows)
    compact_only, full_only = compact_set - full_set, full_set - compact_set
    shared = compact_set & full_set
    if not compact_only and not full_only:
        relation = "equivalent"
    elif not compact_only:
        relation = "compact_multiset_subset_of_full"
    elif not full_only:
        relation = "full_multiset_subset_of_compact"
    elif shared:
        relation = "overlapping_without_containment"
    else:
        relation = "disjoint"
    protected = list(task.glob("*.py")) + [required[-1]] + full
    optional_protected = [task / name for name in ATTRIBUTION_FILES if (task / name).is_file()]
    protected += optional_protected
    if (task / "trace.md").is_file():
        protected.append(task / "trace.md")
    protected = sorted(set(protected))
    return {"compact_file": required[-1].name, "full_file": full[0].name,
            "full_files": [p.name for p in full],
            "compact_cases": len(compact_rows), "full_cases": len(full_rows),
            "equivalent_case_sets": compact_set == full_set,
            "case_set_relation": relation, "compact_only_cases": sum(compact_only.values()),
            "full_only_cases": sum(full_only.values()), "shared_cases": sum(shared.values()),
            "optional_protected_sha256": {p.name: sha(p) for p in optional_protected},
            "protected_sha256": {p.name: sha(p) for p in protected}}


def absolute_rule(path):
    # Claude's documented absolute-path form is //path, not /path.
    value = Path(path).resolve().as_posix()
    if any(c in value for c in "\n\r,*?[]!()"):
        raise ContractError("Workspace path contains characters unsafe for scoped Claude permission rules; choose a plain path.")
    return "/" + value


def permission_config(workspace, allow_shell=False):
    workspace = Path(workspace)
    task = workspace / "task"
    root = absolute_rule(task)
    runtime = absolute_rule(workspace / "runtime")
    support = absolute_rule(workspace / "skills")
    shadow_support = absolute_rule(task / "skills")
    writable = [f"{root}/kernel/**/*.{ext}" for ext in ("cpp", "h", "hpp")]
    writable += [f"{root}/precision_tuning/precision_audit_*.md", f"{root}/precision_tuning/candidate_kb_entry.json"]
    allow = ["Read", "Glob", "Grep", "Skill"]
    allow += [f"{tool}({path})" for tool in ("Edit", "Write") for path in writable]
    if allow_shell:
        allow.append("Bash")
    deny = ["PowerShell", "NotebookEdit", "Agent", "mcp__*"]
    if not allow_shell:
        deny.append("Bash")
    for tool in ("Edit", "Write"):
        deny += [f"{tool}({runtime}/**)", f"{tool}({support}/**)", f"{tool}({shadow_support}/**)",
                 f"{tool}({root}/*.py)", f"{tool}({root}/*.json)",
                 f"{tool}({root}/*.json.bak)", f"{tool}({root}/*.json.full)",
                 f"{tool}({root}/.debug_events/**)", f"{tool}({root}/.verify_status/**)"]
        deny += [f"{tool}({root}/{name})" for name in ATTRIBUTION_FILES]
    return {"permissions": {"defaultMode": "dontAsk", "allow": allow, "deny": deny,
                             "additionalDirectories": [str(task.resolve())],
                             "disableBypassPermissionsMode": "disable"}}


def prepare(args):
    source = source_root()
    task_input = Path(args.task)
    if task_input.is_symlink():
        raise ContractError(f"Symlink task root is not supported: {task_input}")
    task = task_input.resolve()
    output_input = Path(args.output).absolute()
    # Resolve existing parent aliases before containment tests: lexical paths
    # alone would permit alias-to-task/new-workspace to overwrite the input.
    output = output_input.resolve()
    if output.exists() or output_input.is_symlink():
        raise ContractError("Output already exists; prepare never overwrites a workspace. Use run to resume.")
    if task == output or task in output.parents or output in task.parents:
        raise ContractError("Task and output workspace must not contain each other")
    if source == output or source in output.parents:
        raise ContractError("Output workspace must not be inside the frozen runtime source")
    absolute_rule(output)
    contract = task_contract(task, args.op_name)
    source_hashes = tree_hashes(source)
    template = SKILL_ROOT / "references/portable-worker.md"
    if not template.is_file():
        raise ContractError("Missing public portable-worker template in the skill installation")
    reference = Path(args.reference_root).resolve() if args.reference_root else None
    if reference and not reference.is_dir():
        raise ContractError("reference-root must be a directory of reference cases")
    if reference:
        if reference == output or reference in output.parents or output in reference.parents:
            raise ContractError("Reference root and output workspace must not contain each other")
        list(files(reference))
    output.mkdir(parents=True)
    runtime, copied_task = output / "runtime", output / "task"
    try:
        runtime.mkdir()
        copied_task.mkdir()
        copy_tree(source, runtime)
        # The frozen gate searches task ancestors for skills/, while runtime/
        # is a sibling. Bridge only its original self-contained AST validator.
        validator = output / AST_VALIDATOR_REL
        validator.parent.mkdir(parents=True)
        shutil.copy2(runtime / AST_VALIDATOR_REL, validator)
        # Copy source inputs only. A fresh prepare does not inherit stale events,
        # compiled libraries or validation claims from an unrelated location.
        for name in contract["protected_sha256"]:
            shutil.copy2(task / name, copied_task / name)
        copy_tree(task / "kernel", copied_task / "kernel", kernel=True)
        if reference:
            # User references must never replace the recovered historical
            # sources, licenses or provenance, even when names collide.
            user_references = runtime / "archive_tasks/user_references"
            if user_references.exists():
                raise ContractError("Bundled runtime already contains the reserved user_references directory")
            user_references.mkdir(parents=True)
            copy_tree(reference, user_references)
        adapter = template.read_text(encoding="utf-8").replace("{{TASK}}", str(copied_task)).replace("{{RUNTIME}}", str(runtime))
        worker_path = runtime / ".claude/agents/ascendc-debug-agent-constructive.md"
        original_worker = worker_path.read_bytes()
        frontmatter, separator, body = original_worker.partition(b"\n---\n")
        if not original_worker.startswith(b"---\n") or not separator:
            raise ContractError("Constructive worker has no supported YAML frontmatter")
        reference_path = runtime / "archive_tasks/user_references"
        context = (
            "## Prepared workspace resource context\n\n"
            f"- Task operator: `{args.op_name}`.\n"
            f"- User references supplied: `{str(bool(reference)).lower()}`.\n"
        )
        if reference:
            context += (
                f"- Copied user-reference directory (read-only): `{reference_path}`.\n\n"
                "At the start of Phase A, use Glob to list that exact directory, then Read existing "
                "implementations relevant to this task. Record the selected source paths and their "
                "applicability in `[REFERENCE_IMPL_SPEC]`; if the directory is empty or no example "
                "is relevant, record that fact and use the bundled historical references or public "
                "fallback described below. Do not assume a matching example exists.\n"
            )
        else:
            context += (
                f"- No user-reference directory was staged; do not search `{reference_path}`.\n\n"
                "In Phase A, select relevant bundled historical references or use the explicit "
                "public fallback described below.\n"
            )
        context += (
            "Keep the original Phase A → B → C order and required API/audit fields. References "
            "inform the analysis; they do not replace the protected task's mathematical contract, "
            "cases or tolerances and do not establish that a repair passes. Diagnose changes from "
            "this task's failure evidence and retain all original acceptance checks.\n"
        )
        # Preserve the frozen header and body bytes; only the workspace copy
        # gains an early, factual resource entry point and the public adapter.
        worker_path.write_bytes(frontmatter + separator + b"\n" + context.encode("utf-8")
                                + b"\n" + body + b"\n\n" + adapter.encode("utf-8"))
        for name, method in (("ascendc-debug", DEBUG_REL), ("ascendc-translator", Path("skills/ascendc/ascendc-translator"))):
            registration = runtime / ".claude/skills" / name / "SKILL.md"
            registration.parent.mkdir(parents=True, exist_ok=True)
            registration.write_text(f"---\nname: {name}\ndescription: Public frozen AscendC {name} method and references.\n---\n\n"
                                    f"Read `{runtime / method / 'SKILL.md'}` when present and use the original resources under `{runtime / method}`. "
                                    "Do not relocate scripts. The public portable adapter appended to the constructive worker takes precedence for resource locations and availability, including historical examples, optional user references, API documents and the public KB seed. Original algorithm, audit and acceptance rules still apply.\n", encoding="utf-8")
        (runtime / "archive_tasks").mkdir(exist_ok=True)
        (runtime / "archive_tasks/PUBLIC_REFERENCE_SCOPE.md").write_text(
            "This workspace includes the public artifact's bundled historical reference examples when present. Explicit --reference-root files are kept separately under archive_tasks/user_references/ and never replace bundled sources or metadata. "
            "Preserve their README, provenance, licenses and source notices. Historical source identity does not certify fresh device execution or establish which example every paper run used. "
            "If no relevant case exists, use the explicitly recorded public portable Phase A fallback.\n", encoding="utf-8")
        checkpoint_provenance = checkpoint_adapter.stage_checkpoint_adapter(runtime)
        nonfinite_provenance = nonfinite_guard.stage_guard(runtime)
        mode = getattr(args, "determinism", "on")
        _, execution = runtime_checks.configure_runtime_environment(
            os.environ, output / "adapter", mode, output / "adapter-receipts")
        write_json(runtime / ".claude/settings.json", permission_config(output))
        manifest = {"schema_version": 3, "created_at": stamp(), "op_name": args.op_name,
                    "workspace_path": str(output.resolve()), "task_contract": contract,
                    "task_source": str(task), "runtime_source": str(source),
                    "source_sha256": source_hashes, "runtime_sha256": tree_hashes(runtime, exclude_settings=True),
                    "support_sha256": support_hashes(output),
                    "worker_template_sha256": sha(template), "kb": {"path": SEED_REL.as_posix(), "mode": "public_102_seed_read_only"},
                    "portability_changes": ["constructive Phase A explicit public-reference fallback", "workspace-specific read-only reference discovery", "project skill resource registration", "scoped worker permissions; shell opt-in", "byte-identical ancestor-discoverable AST validator bridge", "source-bound completed-full checkpoint ordering", "explicit deterministic startup contract", "original-dtype nonfinite category identity with coverage receipts"],
                    "reference_root_supplied": bool(reference),
                    "execution": {"determinism": mode, "startup": execution},
                    "adapter_sha256": tree_hashes(output / "adapter"),
                    "checkpoint_adapter": checkpoint_provenance,
                    "nonfinite_guard": nonfinite_provenance,
                    "copy_policy": "fresh task source inputs + native kernel; no previous events, binaries, or validation claims"}
        write_json(output / MANIFEST, manifest)
    except Exception:
        shutil.rmtree(output)
        raise
    return {"workspace": str(output.resolve()), "prepared": True, "contract": contract,
            "next": "doctor --workspace WORKSPACE --strict; run WORKSPACE --npu ID --soc SOC"}


def load_workspace(workspace, *, verify=True):
    workspace = Path(workspace).resolve()
    manifest = read_json(workspace / MANIFEST)
    version = manifest.get("schema_version")
    if version not in {1, 2, 3} or manifest.get("workspace_path") != str(workspace):
        raise ContractError("Unknown or moved workspace; prepare a new workspace at its final execution path")
    if verify:
        # The frozen finder checks task/skills before workspace/skills. This
        # nearer tree must not shadow the hashed bridge in schema-2 workspaces.
        shadow = workspace / "task/skills"
        if version >= 2 and (shadow.exists() or shadow.is_symlink()):
            raise ContractError("Task contains shadow support; AST validation must use the protected workspace bridge")
        current = task_contract(workspace / "task", manifest["op_name"])
        if current != manifest["task_contract"]:
            raise ContractError("Protected task inputs or attribution changed; refusing to continue")
        if tree_hashes(workspace / "runtime", exclude_settings=True) != manifest["runtime_sha256"]:
            raise ContractError("Prepared runtime changed; original engine and adapter must remain intact")
        if version >= 2:
            expected = {AST_VALIDATOR_REL.as_posix(): sha(workspace / "runtime" / AST_VALIDATOR_REL)}
            if manifest.get("support_sha256") != expected or support_hashes(workspace) != expected:
                raise ContractError("Workspace support changed; the original AST validator bridge must remain intact")
        elif (workspace / "skills").exists() or (workspace / "skills").is_symlink():
            raise ContractError("Legacy workspace has untracked support files; prepare a new workspace instead of inserting a bridge")
        if version == 3:
            if tree_hashes(workspace / "adapter") != manifest.get("adapter_sha256"):
                raise ContractError("Runtime startup adapter changed; preserve the workspace")
            runtime_checks.execution_environment(os.environ, workspace / "adapter",
                manifest["execution"]["determinism"], workspace / "adapter-receipts")
    return workspace, manifest


def environment_report(*, workspace=None, npu=None, soc=None, claude_bin="claude",
                       require_worker=True, runtime_probe=False, determinism=None):
    checks = []
    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
    env = os.environ.copy()
    try:
        if workspace is None:
            root = source_root()
            mode = determinism or "on"
        else:
            work, manifest = load_workspace(workspace)
            root = work / "runtime"
            mode = manifest.get("execution", {}).get("determinism", "off")
            if determinism not in (None, mode):
                raise ContractError("Preflight must retain the workspace deterministic mode")
            env = base_execution_environment(work, manifest, npu=npu if npu is not None else "0", soc=soc)
        add("frozen_runtime", True, str(root))
    except (ContractError, OSError, ValueError) as exc:
        add("frozen_runtime", False, str(exc))
        mode = determinism or "on"
    runtime_report = runtime_checks.dependency_report(npu=npu, soc=soc, deterministic=mode,
                                                    runtime_probe=runtime_probe, env=env)
    checks.extend(runtime_report["checks"])
    resolved_claude = shutil.which(claude_bin) if require_worker else None
    if require_worker:
        add("binary:" + claude_bin, bool(resolved_claude), resolved_claude or "not found on PATH")
    if resolved_claude:
        try:
            proc = subprocess.run([resolved_claude, "--help"], stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, timeout=15, check=False)
            needed = ("--agent", "--setting-sources", "--allowedTools", "--add-dir", "--session-id", "--max-turns")
            help_missing = [flag for flag in needed if flag not in proc.stdout]
            missing, probes = [], []
            if proc.returncode == 0:
                for flag in help_missing:
                    # --version can exit before unknown-option validation.
                    # A required-value option alone must instead produce its
                    # own missing-argument parser error, before model startup.
                    command = [resolved_claude, flag]
                    try:
                        probe = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                               text=True, timeout=15, check=False)
                        diagnostic = (probe.stdout or "") + "\n" + (probe.stderr or "")
                        argument_missing = re.search(
                            r"(?m)^error:\s+option ['\"]" + re.escape(flag)
                            + r"(?:[ ,][^'\"\r\n]*)?['\"] argument missing\s*$", diagnostic)
                        result = {"flag": flag, "ok": probe.returncode != 0 and bool(argument_missing),
                                  "method": "required_argument_parser_error", "returncode": probe.returncode}
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        result = {"flag": flag, "ok": False, "method": "required_argument_parser_error",
                                  "error": type(exc).__name__}
                    probes.append(result)
                    if not result["ok"]:
                        missing.append(flag)
            else:
                missing = help_missing
            add("claude_cli_contract", proc.returncode == 0 and not missing,
                {"help_returncode": proc.returncode, "help_missing_flags": help_missing,
                 "parser_probes": probes, "missing_flags": missing})
        except (OSError, subprocess.TimeoutExpired) as exc:
            add("claude_cli_contract", False, {"help_error": type(exc).__name__, "parser_probes": []})
    return {"ready": all(c["ok"] for c in checks), "checks": checks,
            "determinism": mode, "runtime_probe": runtime_probe,
            "scope": "Runtime dependency preflight; built-in NPU probe only when requested; no user task or paid model executed."}


def event_status(task):
    path = task / ".debug_events/events.jsonl"
    events, truncated = [], False
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError as exc:
            if index == len(lines) - 1:
                truncated = True
                break
            raise ContractError(f"Corrupt event stream at line {index + 1}") from exc
        if not isinstance(event, dict):
            raise ContractError("Invalid event object")
        events.append(event)
    last = events[-1] if events else {}
    terminal = last.get("type") in TERMINAL
    return {"events": len(events), "attempts_used": sum(e.get("type") == "attempt_started" for e in events),
            "terminal": terminal, "terminal_event": last if terminal else None,
            "truncated_last_event": truncated}


def status(workspace):
    workspace, manifest = load_workspace(workspace, verify=False)
    try:
        load_workspace(workspace)
        integrity = {"ok": True}
    except (ContractError, OSError, ValueError) as exc:
        integrity = {"ok": False, "error": str(exc)}
    state = read_json(workspace / RUN_STATE) if (workspace / RUN_STATE).exists() else {}
    return {"workspace": str(workspace), "op_name": manifest["op_name"],
            "integrity": integrity, "session": event_status(workspace / "task"),
            "adapter_support": {"ast_validator_bridge": "legacy_unmanaged" if manifest["schema_version"] == 1
                                else "verified" if integrity["ok"] else "unverified",
                                "nonfinite_guard": ("verified" if integrity["ok"] else "unverified")
                                if "nonfinite_guard" in manifest else "legacy_unchecked"},
            "nonfinite_guard": {"configured": "nonfinite_guard" in manifest,
                                "requires_new_workspace": "nonfinite_guard" not in manifest,
                                "provenance": manifest.get("nonfinite_guard")},
            "locked": (workspace / ".drdebug.lock").exists(), "runs": state,
            "accepted_checkpoint": checkpoint_adapter.acceptance_status(workspace / "task") if manifest["schema_version"] == 3 else None,
            "engine_status": read_json(workspace / "task/debug_status.json") if (workspace / "task/debug_status.json").is_file() else None}


def base_execution_environment(workspace, manifest, *, npu, soc):
    runtime = workspace / "runtime"
    env = os.environ.copy()
    # Neither inherited experiment ablations nor an alternate Python path may
    # quietly disable acceptance or substitute a different engine.
    for key in list(env):
        if key.startswith(("ABLATE_", "ASCENDC_ABLATE_", "ASCENDC_DEBUG_")) or key in {"PYTHONHOME", "PYTHONPATH", "PYTHONOPTIMIZE"}:
            env.pop(key)
    env.update({"PYTHONPATH": os.pathsep.join((str(runtime / DEBUG_REL), str(runtime))),
                "PYTHONDONTWRITEBYTECODE": "1", "ASCENDC_DEBUG_KB_READ_ONLY": "1",
                "ASCENDC_SOC_VERSION": soc or os.environ.get("ASCENDC_SOC_VERSION", ""),
                "ASCEND_RT_VISIBLE_DEVICES": str(npu), "ASCEND_DEVICE_ID": "0"})
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    if manifest.get("schema_version") == 3:
        env = runtime_checks.execution_environment(env, workspace / "adapter",
            manifest["execution"]["determinism"], workspace / "adapter-receipts")
    return env


def engine_command(workspace, manifest, args, budget, remaining):
    runtime, task = workspace / "runtime", workspace / "task"
    config = permission_config(workspace, getattr(args, "allow_worker_shell", False))
    allowed = ",".join(config["permissions"]["allow"])
    command = [sys.executable, "-B", "-m", "engine", str(task), "--op-name", manifest["op_name"],
               "--agent", "constructive", "--workdir", str(task), "--npu", "0",
               "--entry-failure-type", "precision_failed", "--kb-path", str(runtime / SEED_REL),
               "--allowed-tools", allowed, "--claude-bin", getattr(args, "claude_bin", "claude"),
               "--max-attempts", str(budget["max_attempts"]), "--max-turns", str(budget["max_turns"]),
               "--soft-task-turns", str(budget["max_turns"] * max(1, budget["max_attempts"] - 1)),
               "--max-task-turns", str(budget["max_turns"] * budget["max_attempts"]),
               "--deadline-sec", str(remaining), "--agent-timeout-sec", str(min(remaining, budget.get("worker_timeout_sec", 600)))]
    if getattr(args, "model", None):
        command += ["--model", args.model]
    env = base_execution_environment(workspace, manifest, npu=args.npu, soc=args.soc)
    return command, env, config


def execute_engine(command, *, cwd, env, timeout):
    """Argument-vector launch only; stop the child process group at the wall cap."""
    # Keep CLI JSON machine-readable; all child output is preserved locally.
    with (Path(cwd).parent / "drdebug-engine.stdout.log").open("a", encoding="utf-8") as stdout, \
         (Path(cwd).parent / "drdebug-engine.stderr.log").open("a", encoding="utf-8") as stderr:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                start_new_session=(os.name == "posix"))
        return _wait_engine(proc, timeout)


def _wait_engine(proc, timeout):
    try:
        return proc.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name != "posix":
                proc.kill()
        finally:
            # A promptly exiting parent does not prove its workers exited.
            # This group was created exclusively for this engine invocation.
            if os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        proc.wait()
        if isinstance(exc, subprocess.TimeoutExpired):
            raise EngineStopped(4, "active_wall_clock_budget_exhausted") from exc
        raise EngineStopped(130, "interrupted") from exc


def run(args):
    workspace, manifest = load_workspace(args.workspace)
    mode = manifest.get("execution", {}).get("determinism", "off")
    if getattr(args, "determinism", None) not in (None, mode):
        raise ContractError("Resume retains the workspace deterministic mode; prepare a new workspace for another mode")
    session = event_status(workspace / "task")
    if session["truncated_last_event"]:
        raise ContractError("Event stream has a truncated final record. Preserve the evidence and repair that tail explicitly before resuming; the launcher will not rewrite events.")
    if session["terminal"]:
        snapshot = status(workspace)
        event = session["terminal_event"]
        outcome = (snapshot.get("engine_status") or {}).get("session_outcome") or event.get("session_outcome") or (event.get("details") or {}).get("session_outcome") or "crashed"
        code = OUTCOME_CODES.get(outcome, 7)
        binding = checkpoint_adapter.acceptance_status(workspace / "task") if manifest["schema_version"] == 3 and code == 0 else None
        if binding is not None and not binding["ready"]:
            code = 2
        return {"returncode": code, "already_terminal": True, "accepted_checkpoint": binding, "status": snapshot}
    state_path = workspace / RUN_STATE
    prior_bytes = state_path.read_bytes() if state_path.exists() else None
    prior = json.loads(prior_bytes) if prior_bytes is not None else {}
    if not isinstance(prior, dict):
        raise ContractError("Invalid run history; preserve it for inspection before resuming.")
    saved = prior.get("budget", {})
    if prior_bytes is not None and (not isinstance(saved, dict) or not saved):
        raise ContractError("Existing run history has no saved budget; preserve it instead of applying fresh defaults.")
    if prior_bytes is None and session["events"]:
        raise ContractError("Existing engine events have no saved budget history; preserve them before resuming.")
    budget = {}
    legacy_defaults = {"max_attempts": 5, "deadline_sec": 1800, "max_turns": 80,
                       "worker_timeout_sec": 600}
    fresh_defaults = {"max_attempts": 5, "deadline_sec": 86400, "max_turns": 240,
                      "worker_timeout_sec": 21600}
    for name, fresh_default in fresh_defaults.items():
        default = legacy_defaults[name] if saved else fresh_default
        supplied = getattr(args, name)
        value = supplied if supplied is not None else saved.get(name, default)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ContractError(f"{name} must be finite and positive")
        # Missing fields in an older budget retain their historical defaults;
        # the larger fresh profile must never increase a resumed experiment.
        if saved and saved.get(name, default) != value:
            raise ContractError("Resume retains the original attempt/turn/time/worker-timeout budget. Prepare a new workspace for a new experiment.")
        budget[name] = value
    remaining = budget["deadline_sec"] - sum(item.get("elapsed_sec", max(0, time.time() - item["started_unix"])) for item in prior.get("runs", []))
    if remaining <= 0:
        return {"returncode": 4, "launcher_outcome": "active_wall_clock_budget_exhausted", "status": status(workspace)}
    command, env, config = engine_command(workspace, manifest, args, budget, remaining)
    effective_worker_timeout = min(remaining, budget["worker_timeout_sec"])
    preview = {"returncode": 0, "dry_run": True, "command": command, "cwd": str(workspace / "runtime"),
               "environment": {key: env[key] for key in ("PYTHONPATH", "ASCENDC_SOC_VERSION", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_DEVICE_ID", "ASCENDC_DEBUG_KB_READ_ONLY")},
               "worker_shell_authorized": args.allow_worker_shell, "budget": budget,
               "remaining_active_seconds": remaining, "effective_worker_timeout_sec": effective_worker_timeout,
               "session": session,
               "scope": "Command preview only; not NPU validation or provider readiness."}
    if args.dry_run:
        return preview
    lock = workspace / ".drdebug.lock"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ContractError("Workspace is locked. Check status and the recorded PID; never run two engines on one workspace.") from exc
    started = time.monotonic()
    record = {"started_at": stamp(), "started_unix": time.time(), "pid": os.getpid(),
              "worker_shell_authorized": args.allow_worker_shell, "physical_npu": args.npu,
              "logical_npu": "0", "soc": env["ASCENDC_SOC_VERSION"], "model": args.model,
              "worker_timeout_sec": budget["worker_timeout_sec"],
              "effective_worker_timeout_sec": effective_worker_timeout,
              "command": command, "stdout_log": str(workspace / "drdebug-engine.stdout.log"),
              "stderr_log": str(workspace / "drdebug-engine.stderr.log")}
    settings_active = False
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump({"pid": os.getpid(), "started_at": stamp()}, handle)
        report = environment_report(workspace=workspace, npu=args.npu, soc=args.soc,
                                    claude_bin=args.claude_bin, runtime_probe=True, determinism=mode)
        if not report["ready"]:
            return {"returncode": 2, "launcher_outcome": "preflight_failed", "preflight": report}
        remaining -= time.monotonic() - started
        if remaining <= 0:
            return {"returncode": 4, "launcher_outcome": "active_wall_clock_budget_exhausted", "preflight": report}
        command, env, config = engine_command(workspace, manifest, args, budget, remaining)
        record["command"] = command
        record["effective_worker_timeout_sec"] = min(remaining, budget["worker_timeout_sec"])
        latest_bytes = state_path.read_bytes() if state_path.exists() else None
        if latest_bytes != prior_bytes or event_status(workspace / "task") != session:
            raise ContractError("Run history changed during preflight. Inspect status and retry; no stale budget was used.")
        load_workspace(workspace)  # recheck after acquiring the exclusive run lock
        prior.setdefault("runs", []).append(record)
        prior["budget"] = budget
        write_json(workspace / "runtime/.claude/settings.json", config)
        settings_active = True
        write_json(state_path, prior)
        outcome = "engine_returned"
        try:
            code = execute_engine(command, cwd=workspace / "runtime", env=env, timeout=remaining)
        except EngineStopped as exc:
            code, outcome = exc.returncode, exc.outcome
        record.update({"elapsed_sec": time.monotonic() - started, "finished_at": stamp(),
                       "returncode": code, "launcher_outcome": outcome})
        try:
            load_workspace(workspace)
        except (ContractError, OSError, ValueError) as exc:
            record["integrity_error"] = str(exc)
            code = 2
        record["engine_returncode"] = record["returncode"]
        binding = checkpoint_adapter.acceptance_status(workspace / "task") if manifest["schema_version"] == 3 and code == 0 else None
        if binding is not None:
            record["accepted_checkpoint"] = binding
            if not binding["ready"]:
                code, outcome = 2, "accepted_checkpoint_identity_failed"
        record["returncode"] = code
        record["launcher_outcome"] = outcome
        write_json(state_path, prior)
        return {"returncode": code, "launcher_outcome": outcome, "accepted_checkpoint": binding, "status": status(workspace)}
    finally:
        try:
            # Shell permission is an explicit choice for this invocation, not
            # a new persisted default for later manually launched workers.
            if settings_active:
                write_json(workspace / "runtime/.claude/settings.json", permission_config(workspace))
        finally:
            lock.unlink(missing_ok=True)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Read-only dependency and optional workspace preflight")
    doctor.add_argument("--workspace")
    doctor.add_argument("--strict", action="store_true")
    doctor.add_argument("--npu")
    doctor.add_argument("--soc")
    doctor.add_argument("--claude-bin", default="claude")
    doctor.add_argument("--determinism", choices=("on", "off"))
    doctor.add_argument("--runtime-probe", action="store_true", help="Execute a tiny built-in NPU probe; no user task or model")
    doctor.add_argument("--no-worker", action="store_true", help="Check only build/NPU dependencies for saved-candidate replay")
    prep = commands.add_parser("prepare", help="Copy task inputs and frozen public runtime into a fresh workspace")
    prep.add_argument("task")
    prep.add_argument("--output", required=True)
    prep.add_argument("--op-name", required=True)
    prep.add_argument("--reference-root")
    prep.add_argument("--determinism", choices=("on", "off"), default="on", help="Fixed startup mode for this new workspace (default: on)")
    launch = commands.add_parser("run", help="Run/resume the paper engine with public runtime adapters; no permission bypass")
    launch.add_argument("workspace")
    launch.add_argument("--npu", required=True)
    launch.add_argument("--soc", help="Actual CANN SOC name; alternatively ASCENDC_SOC_VERSION")
    launch.add_argument("--model")
    launch.add_argument("--determinism", choices=("on", "off"), help="Must match prepared workspace; never changes it on resume")
    launch.add_argument("--claude-bin", default="claude")
    launch.add_argument("--max-attempts", type=int, help="Attempt budget for a fresh workspace (default: 5); retained on resume")
    launch.add_argument("--deadline-sec", type=float, help="Total active wall-clock budget in seconds for a fresh workspace (default: 86400); retained on resume")
    launch.add_argument("--worker-timeout-sec", type=float, help="Per worker CLI invocation timeout in seconds for a fresh workspace (default: 21600); retained on resume")
    launch.add_argument("--max-turns", type=int, help="Per worker turn budget for a fresh workspace (default: 240); retained on resume")
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--allow-worker-shell", action="store_true", help="Explicitly authorize original worker Bash capability in a trusted isolated NPU workspace")
    verify_command = commands.add_parser("verify", help="Independently clean-build and replay original compact/full cases; no model")
    verify_command.add_argument("workspace")
    verify_command.add_argument("--npu", required=True)
    verify_command.add_argument("--soc")
    verify_command.add_argument("--determinism", choices=("on", "off"))
    verify_command.add_argument("--casewise", action="store_true", help="Also start a fresh process per original full case")
    verify_command.add_argument("--stage-timeout-sec", type=float, default=1800)
    verify_command.add_argument("--overall-timeout-sec", type=float, default=10800)
    inspect = commands.add_parser("status", help="Read events, integrity and budget; never restart a terminal session")
    inspect.add_argument("workspace")
    for command in (doctor, prep, launch, inspect, verify_command):
        command.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if getattr(args, "npu", None) is not None and not re.fullmatch(r"[0-9]+", args.npu):
            raise ContractError("npu must be one nonnegative physical device ID")
        if args.command == "doctor":
            result = environment_report(workspace=args.workspace, npu=args.npu, soc=args.soc, claude_bin=args.claude_bin, require_worker=not args.no_worker, runtime_probe=args.runtime_probe, determinism=args.determinism)
            code = 2 if args.strict and not result["ready"] else 0
        elif args.command == "prepare":
            result, code = prepare(args), 0
        elif args.command == "verify":
            import verification
            result = verification.verify(args, sys.modules[__name__])
            code = result["returncode"]
        elif args.command == "status":
            result, code = status(args.workspace), 0
        else:
            result = run(args)
            code = result["returncode"]
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return code
    except (ContractError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"error": str(exc), "returncode": 2}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
