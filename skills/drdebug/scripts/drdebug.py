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
PROVIDER_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_", "KIMI_", "MOONSHOT_", "CODEX_API_", "OPENAI_API_")
TERMINAL = {"session_done", "session_aborted", "session_escalated"}
IGNORED = {".git", "__pycache__", ".DS_Store"}
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


def validate_workspace_path(path):
    if any(ord(c) < 32 for c in str(Path(path).absolute())):
        raise ContractError("Workspace paths must not contain control characters")


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
        raise ContractError("Output already exists; prepare never overwrites a workspace. Use next to resume.")
    if task == output or task in output.parents or output in task.parents:
        raise ContractError("Task and output workspace must not contain each other")
    if source == output or source in output.parents:
        raise ContractError("Output workspace must not be inside the frozen runtime source")
    validate_workspace_path(output)
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
        original_worker = (runtime / ".claude/agents/ascendc-debug-agent-constructive.md").read_bytes()
        _, separator, body = original_worker.partition(b"\n---\n")
        if not original_worker.startswith(b"---\n") or not separator:
            raise ContractError("Frozen constructive method has no supported YAML frontmatter")
        reference_path = runtime / "archive_tasks/user_references"
        context = (
            "## Prepared workspace resource context\n\n"
            f"- Task operator: `{args.op_name}`.\n"
            f"- User references supplied: `{str(bool(reference)).lower()}`.\n"
            f"- Bundled historical references (read-only): `{runtime / 'archive_tasks'}`.\n"
            f"- Public host reference documents (read-only): `{SKILL_ROOT / 'references'}`.\n"
        )
        if reference:
            context += (f"- Copied user-reference directory (read-only): `{reference_path}`.\n"
                        "Inspect that exact directory for relevant implementations in Phase A; record selected paths "
                        "and applicability in [REFERENCE_IMPL_SPEC]. Empty or irrelevant references require the "
                        "explicit public fallback; do not assume a matching example exists.\n")
        else:
            context += (f"- No user-reference directory was staged; do not search `{reference_path}`.\n"
                        "Select relevant bundled historical references or the explicit public fallback in Phase A.\n")
        context += ("Keep Phase A -> B -> C and required audit fields. The historical method below describes "
                    "the repair reasoning. The public host adapter at the end overrides historical initialization, "
                    "worker launch, engine writes and validation ownership. Do not execute those historical "
                    "orchestration steps. Only the isolated request candidate may be edited.\n\n")
        (runtime / "host-instructions.md").write_bytes(
            context.encode() + body + b"\n\n" + adapter.encode())
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
        manifest = {"schema_version": 4, "execution_mode": "native_host", "created_at": stamp(), "op_name": args.op_name,
                    "workspace_path": str(output.resolve()), "task_contract": contract,
                    "task_source": str(task), "runtime_source": str(source),
                    "source_sha256": source_hashes, "runtime_sha256": tree_hashes(runtime, exclude_settings=True),
                    "support_sha256": support_hashes(output),
                    "host_template_sha256": sha(template),
                    "public_adapter_sha256": adapter_hashes(), "kb": {"path": SEED_REL.as_posix(), "mode": "public_102_seed_read_only"},
                    "portability_changes": ["constructive Phase A explicit public-reference fallback", "workspace-specific read-only reference discovery", "native host instructions; no nested worker or provider dependency", "isolated source candidate and durable engine-owned handoff", "byte-identical ancestor-discoverable AST validator bridge", "source-bound completed-full checkpoint ordering", "explicit deterministic startup contract", "original-dtype nonfinite category identity with coverage receipts"],
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
            "next": "doctor --workspace WORKSPACE --strict; next WORKSPACE --npu ID --soc SOC --json"}


def load_workspace(workspace, *, verify=True):
    workspace = Path(workspace).resolve()
    manifest = read_json(workspace / MANIFEST)
    version = manifest.get("schema_version")
    if version not in {1, 2, 3, 4} or manifest.get("workspace_path") != str(workspace):
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
        if tree_hashes(workspace / "runtime", exclude_settings=version < 4) != manifest["runtime_sha256"]:
            raise ContractError("Prepared runtime changed; original engine and adapter must remain intact")
        if version >= 2:
            expected = {AST_VALIDATOR_REL.as_posix(): sha(workspace / "runtime" / AST_VALIDATOR_REL)}
            if manifest.get("support_sha256") != expected or support_hashes(workspace) != expected:
                raise ContractError("Workspace support changed; the original AST validator bridge must remain intact")
        elif (workspace / "skills").exists() or (workspace / "skills").is_symlink():
            raise ContractError("Legacy workspace has untracked support files; prepare a new workspace instead of inserting a bridge")
        if version >= 3:
            if tree_hashes(workspace / "adapter") != manifest.get("adapter_sha256"):
                raise ContractError("Runtime startup adapter changed; preserve the workspace")
            runtime_checks.execution_environment(os.environ, workspace / "adapter",
                manifest["execution"]["determinism"], workspace / "adapter-receipts")
        if version >= 4 and adapter_hashes() != manifest.get("public_adapter_sha256"):
            raise ContractError("Public adapter changed; preserve this workspace and prepare a fresh one")
    return workspace, manifest


def adapter_hashes():
    return {p.name: sha(p) for p in sorted(SCRIPT_DIR.glob("*.py"))}


def environment_report(*, workspace=None, npu=None, soc=None,
                       runtime_probe=False, determinism=None):
    checks = []
    env = {k: v for k, v in os.environ.items() if not k.startswith(PROVIDER_PREFIXES)}
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
        checks.append({"name": "frozen_runtime", "ok": True, "detail": str(root)})
    except (ContractError, OSError, ValueError) as exc:
        checks.append({"name": "frozen_runtime", "ok": False, "detail": str(exc)})
        mode = determinism or "on"
    checks.extend(runtime_checks.dependency_report(npu=npu, soc=soc, deterministic=mode,
        runtime_probe=runtime_probe, env=env)["checks"])
    return {"ready": all(c["ok"] for c in checks), "checks": checks,
            "execution_mode": "native_host", "external_worker_required": False,
            "determinism": mode, "runtime_probe": runtime_probe,
            "scope": "Build/NPU dependency preflight; no agent CLI, model, or user task executed."}


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
            "execution_mode": manifest.get("execution_mode", "legacy_worker"),
            "host_session": read_json(workspace / "host-session/state.json") if (workspace / "host-session/state.json").is_file() else None,
            "accepted_checkpoint": checkpoint_adapter.acceptance_status(workspace / "task") if manifest["schema_version"] >= 3 else None,
            "engine_status": read_json(workspace / "task/debug_status.json") if (workspace / "task/debug_status.json").is_file() else None}


def base_execution_environment(workspace, manifest, *, npu, soc):
    runtime = workspace / "runtime"
    env = os.environ.copy()
    # Neither inherited experiment ablations nor an alternate Python path may
    # quietly disable acceptance or substitute a different engine.
    for key in list(env):
        if key.startswith(("ABLATE_", "ASCENDC_ABLATE_", "ASCENDC_DEBUG_") + PROVIDER_PREFIXES) or key in {"PYTHONHOME", "PYTHONPATH", "PYTHONOPTIMIZE"}:
            env.pop(key)
    env.update({"PYTHONPATH": os.pathsep.join((str(runtime / DEBUG_REL), str(runtime))),
                "PYTHONDONTWRITEBYTECODE": "1", "ASCENDC_DEBUG_KB_READ_ONLY": "1",
                "ASCENDC_SOC_VERSION": soc or os.environ.get("ASCENDC_SOC_VERSION", ""),
                "ASCEND_RT_VISIBLE_DEVICES": str(npu), "ASCEND_DEVICE_ID": "0"})
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    if manifest.get("schema_version", 0) >= 3:
        env = runtime_checks.execution_environment(env, workspace / "adapter",
            manifest["execution"]["determinism"], workspace / "adapter-receipts")
    return env


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
    raise ContractError("The nested Claude worker was removed. Preserve old workspaces; prepare a fresh native-host workspace and use next / submit / verify.")


def host_command(args):
    workspace, manifest = load_workspace(args.workspace)
    if manifest.get("schema_version") != 4 or manifest.get("execution_mode") != "native_host":
        raise ContractError("Legacy worker workspaces are read-only for status/verify; prepare a fresh native-host workspace")
    mode = manifest["execution"]["determinism"]
    if getattr(args, "determinism", None) not in (None, mode):
        raise ContractError("Resume retains deterministic mode")
    if event_status(workspace / "task")["truncated_last_event"]:
        raise ContractError("Truncated event tail; preserve evidence for explicit recovery")
    lock = workspace / ".drdebug.lock"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ContractError("Workspace is locked; inspect its recorded PID before recovery") from exc
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"pid": os.getpid(), "purpose": args.command, "started_at": stamp()}, stream)
        load_workspace(workspace)
        history_path = workspace / RUN_STATE
        history = read_json(history_path) if history_path.exists() else {}
        if history_path.exists():
            if (not isinstance(history, dict) or not isinstance(history.get("budget"), dict)
                    or set(history["budget"]) != {"max_attempts", "deadline_sec"}
                    or not isinstance(history.get("runs"), list)
                    or not isinstance(history.get("device"), dict)
                    or not isinstance(history.get("external_elapsed_sec"), (int, float))
                    or isinstance(history.get("external_elapsed_sec"), bool)
                    or not math.isfinite(history.get("external_elapsed_sec", -1))
                    or history.get("external_elapsed_sec", -1) < 0):
                raise ContractError("Existing host history has missing or invalid budget/device records; preserve it rather than applying fresh defaults")
        elif (workspace / "host-session/state.json").exists() or event_status(workspace / "task")["events"]:
            raise ContractError("Existing engine evidence has no saved host budget; preserve it instead of starting a fresh run")
        saved = history.get("budget", {})
        budget = {}
        for name, default in (("max_attempts", 5), ("deadline_sec", 86400)):
            supplied = getattr(args, name, None)
            value = supplied if supplied is not None else saved.get(name, default)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0
                    or name == "max_attempts" and type(value) is not int):
                raise ContractError(name + " must be finite and positive (attempts an integer)")
            if saved and value != saved[name]:
                raise ContractError("Resume retains the original host attempt/time budget")
            budget[name] = value
        env = base_execution_environment(workspace, manifest, npu=args.npu, soc=args.soc)
        device = {"physical_npu": args.npu, "soc": env["ASCENDC_SOC_VERSION"]}
        if history and history.get("device") != device:
            raise ContractError("Resume retains the recorded NPU and SOC")
        result = None
        if args.command == "submit":
            result = read_json(args.result)
            if not isinstance(result, dict):
                raise ContractError("Host result must be a JSON object")
        config = {"workspace": str(workspace), "runtime": str(workspace / "runtime"),
                  "op_name": manifest["op_name"], "kb_path": str(workspace / "runtime" / SEED_REL),
                  "budget": budget, "resource_context": str(workspace / "runtime/host-instructions.md"),
                  "operation": args.command}
        if args.command == "submit":
            config.update(request_id=args.request_id, result=result)
        if getattr(args, "dry_run", False):
            return {"returncode": 0, "dry_run": True, "configuration": config,
                    "external_worker_required": False}
        # An existing pending packet is read-only; its identity is checked by the
        # child even when no NPU work will be dispatched in this invocation.
        host_state_path = workspace / "host-session/state.json"
        host_state = read_json(host_state_path) if host_state_path.exists() else {}
        operation = host_state.get("operation") or {}
        pending = operation.get("kind") == "host" and operation.get("phase") in {"intent", "pending"}
        consumed = host_state.get("consumed_sec", 0)
        pending_age = max(0, time.time() - operation["issued_at"]) if pending and operation.get("issued_at") is not None else 0
        consumed += pending_age
        external = history.get("external_elapsed_sec", 0.0)
        external_delta = max(0, external - host_state.get("accounted_external_sec", 0))
        consumed += external_delta - min(external_delta, pending_age)
        remaining = budget["deadline_sec"] - consumed
        history.setdefault("budget", budget)
        history.setdefault("device", device)
        history.setdefault("runs", [])
        history.setdefault("external_elapsed_sec", external)
        tag = str(time.time_ns())
        invocation = workspace / "host-invocations" / tag
        invocation.mkdir(parents=True)
        record = {"operation": args.command, "started_at": stamp(), "directory": str(invocation)}
        history["runs"].append(record)
        write_json(history_path, history)
        # Merely re-reading a pending or terminal request needs no device use.
        if remaining > 0 and not (args.command == "next" and (pending or event_status(workspace / "task")["terminal"])):
            started = time.monotonic()
            try:
                preflight = environment_report(workspace=workspace, npu=args.npu, soc=args.soc,
                                               runtime_probe=True, determinism=mode)
            finally:
                elapsed = max(0, time.monotonic() - started)
                history["external_elapsed_sec"] += elapsed
                record["preflight_elapsed_sec"] = elapsed
                remaining -= elapsed
                write_json(history_path, history)
            write_json(invocation / "preflight.json", preflight)
            if not preflight["ready"]:
                record.update(finished_at=stamp(), returncode=2, state="preflight_failed")
                write_json(history_path, history)
                return {"returncode": 2, "state": "preflight_failed", "preflight": preflight,
                        "invocation": str(invocation)}
        load_workspace(workspace)
        config["external_elapsed_sec"] = history["external_elapsed_sec"]
        write_json(invocation / "config.json", config)
        command = [sys.executable, "-B", str(SCRIPT_DIR / "host_session.py"),
                   "--config", str(invocation / "config.json")]
        # Every numerical subprocess receives the pinned startup/environment;
        # provider credentials are unnecessary for deterministic engine tools.
        env = {k: v for k, v in env.items() if not k.startswith(PROVIDER_PREFIXES)}
        try:
            with (invocation / "stdout.log").open("x") as out, (invocation / "stderr.log").open("x") as err:
                proc = subprocess.Popen(command, cwd=workspace / "runtime", env=env, stdout=out,
                                        stderr=err, start_new_session=os.name == "posix")
                # Expired sessions get a short bookkeeping grace to record their
                # terminal timeout; the driver checks the budget before dispatch.
                code = _wait_engine(proc, max(1, remaining) if remaining > 0 else 30)
            value = json.loads((invocation / "stdout.log").read_text())
            if not isinstance(value, dict) or value.get("returncode") != code:
                raise ContractError("Host driver exit/status disagree; preserve invocation logs")
        except EngineStopped as exc:
            value = {"returncode": exc.returncode, "state": "needs_attention", "error": exc.outcome,
                     "evidence_preserved": True}
        except (ValueError, OSError) as exc:
            value = {"returncode": 2, "state": "needs_attention", "error": str(exc),
                     "evidence_preserved": True}
        load_workspace(workspace)
        binding = checkpoint_adapter.acceptance_status(workspace / "task") if value.get("terminal") and value.get("returncode") == 0 else None
        if binding is not None and not binding["ready"]:
            value.update(returncode=2, state="needs_attention", error="Accepted checkpoint identity failed")
        if binding is not None:
            value["accepted_checkpoint"] = binding
        if value.get("state") == "awaiting_host":
            value["candidate_dir"] = value["candidate"]
            value["result_file"] = str(Path(value["request"]).parent / "host-result.json")
        record.update(finished_at=stamp(), returncode=value["returncode"], state=value.get("state"))
        write_json(history_path, history)
        value["invocation"] = str(invocation)
        return value
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
    doctor.add_argument("--determinism", choices=("on", "off"))
    doctor.add_argument("--runtime-probe", action="store_true", help="Execute a tiny built-in NPU probe; no user task or model")
    prep = commands.add_parser("prepare", help="Copy task inputs and frozen public runtime into a fresh workspace")
    prep.add_argument("task")
    prep.add_argument("--output", required=True)
    prep.add_argument("--op-name", required=True)
    prep.add_argument("--reference-root")
    prep.add_argument("--determinism", choices=("on", "off"), default="on", help="Fixed startup mode for this new workspace (default: on)")
    for name in ("next", "submit"):
        launch = commands.add_parser(name, help="Advance deterministic engine or submit an isolated host candidate")
        launch.add_argument("workspace")
        launch.add_argument("--npu", required=True)
        launch.add_argument("--soc")
        launch.add_argument("--determinism", choices=("on", "off"))
        launch.add_argument("--max-attempts", type=int, help="Fresh host attempt budget (default 5); retained on resume")
        launch.add_argument("--deadline-sec", type=float, help="Fresh command + host-round elapsed budget (default 86400); retained on resume")
        launch.add_argument("--dry-run", action="store_true")
        launch.add_argument("--json", action="store_true")
        if name == "submit":
            launch.add_argument("--request-id", required=True)
            launch.add_argument("--result", required=True, help="Host completion JSON; never a precision verdict")
    legacy = commands.add_parser("run", help="Removed nested worker command; prints migration guidance")
    legacy.add_argument("workspace", nargs="?")
    legacy.add_argument("--json", action="store_true")
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
    for command in (doctor, prep, inspect, verify_command):
        command.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if getattr(args, "npu", None) is not None and not re.fullmatch(r"[0-9]+", args.npu):
            raise ContractError("npu must be one nonnegative device ID exposed to the execution environment")
        if args.command == "doctor":
            result = environment_report(workspace=args.workspace, npu=args.npu, soc=args.soc, runtime_probe=args.runtime_probe, determinism=args.determinism)
            code = 2 if args.strict and not result["ready"] else 0
        elif args.command == "prepare":
            result, code = prepare(args), 0
        elif args.command == "verify":
            import verification
            result = verification.verify(args, sys.modules[__name__])
            code = result["returncode"]
        elif args.command == "status":
            result, code = status(args.workspace), 0
        elif args.command == "run":
            result = run(args)
            code = result["returncode"]
        else:
            result = host_command(args)
            code = result["returncode"]
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return code
    except (ContractError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"error": str(exc), "returncode": 2}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
