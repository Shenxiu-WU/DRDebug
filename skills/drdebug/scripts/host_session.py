#!/usr/bin/env python3
"""Durable host-owned repair turns around the unchanged paper state machine.

The public launcher owns the workspace lock, runtime identity and numerical
environment. This driver never starts a model or a second agent CLI. Only the
host's isolated candidate is editable; original deterministic dispatch, gates,
checkpoint transactions and terminal decisions remain engine-owned.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

DEBUG_REL = Path("skills/ascendc/ascendc-debug")
MANIFEST = "drdebug-workspace.json"
EDITABLE = {".cpp", ".h", ".hpp"}
IGNORED_BINARY = {".so", ".o", ".a", ".pyc"}
DIAGNOSTIC_ENVIRONMENT = {
    "PATH", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED",
    "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH", "CPLUS_INCLUDE_PATH",
    "ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH",
    "ASCENDC_SOC_VERSION", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_DEVICE_ID",
    "ASCEND_LAUNCH_BLOCKING", "HCCL_DETERMINISTIC", "CANN_ENABLE_DETERMINISTIC",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "DRDEBUG_DETERMINISM",
    "DRDEBUG_RUNTIME_STARTUP_CONFIG", "DRDEBUG_RUNTIME_STARTUP_CONFIG_SHA256",
    "DRDEBUG_RUNTIME_RECEIPTS",
}
OUTCOME_CODES = {"success": 0, "failed": 1, "stopped_by_gate": 2, "stopped_by_loop_limit": 3,
                 "stopped_by_attempt_limit": 12, "stopped_by_branch_limit": 13, "timeout": 4,
                 "skipped_env_issue": 5, "skipped_unsupported_type": 6, "crashed": 7,
                 "provider_api_error": 8, "stopped_by_budget": 9, "degenerate_no_progress": 10,
                 "ablation_violation": 11}


class HostSessionError(ValueError):
    pass


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def _read(path):
    if path.is_symlink() or not path.is_file():
        raise HostSessionError(f"Missing or linked host record: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HostSessionError("Host records must be JSON objects")
    return value


def _write(path, value):
    _bytes(path, _encode(value))


def _plain(path, root):
    try:
        parts = path.relative_to(root).parts
    except ValueError as exc:
        raise HostSessionError("Host evidence path escaped the workspace") from exc
    current = root
    for part in (None, *parts):
        if part is not None:
            current /= part
        if current.is_symlink():
            raise HostSessionError(f"Linked evidence ancestor: {current}")


def _bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise HostSessionError(f"Linked output: {path}")
    temp = path.with_name("." + path.name + "." + uuid4().hex)
    try:
        with temp.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _inventory(root, *, source_only=False, task=False):
    """Hash regular files, rejecting links even in otherwise ignored trees."""
    if root.is_symlink() or not root.is_dir():
        raise HostSessionError(f"Missing or linked directory: {root}")
    values = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise HostSessionError(f"Linked candidate or evidence: {path}")
        rel = path.relative_to(root)
        if path.is_dir():
            continue
        if not path.is_file():
            raise HostSessionError(f"Non-regular candidate or evidence: {path}")
        if task and (rel.as_posix() == ".debug_events/events.jsonl" or rel.parts[:2] == ("kernel", "build")):
            continue
        if source_only and ("build" in rel.parts or path.suffix in IGNORED_BINARY):
            continue
        values[rel.as_posix()] = _sha(path.read_bytes())
    return values


def _copy_files(source, target, inventory):
    target.mkdir(parents=True, exist_ok=False)
    for name, digest in inventory.items():
        path = source / name
        data = path.read_bytes()
        if _sha(data) != digest:
            raise HostSessionError("Source changed while taking a candidate snapshot")
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)


def _engine(runtime):
    expected = runtime / DEBUG_REL / "engine"
    loaded = sys.modules.get("engine")
    if loaded is not None and Path(loaded.__file__).resolve().parent != expected:
        raise HostSessionError("Another workspace engine is already imported; use a fresh driver process")
    for path in (runtime, runtime / DEBUG_REL):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return {name: importlib.import_module("engine." + name) for name in
            ("runner", "state", "transition", "types", "next_action", "agent_backend", "exit_artifacts")}


def _action_payload(action):
    return {"kind": action.kind, "name": action.name, "step": action.step,
            "skill_args": action.skill_args, "expected_artifacts": action.expected_artifacts}


def _result(value):
    if (not isinstance(value, dict) or set(value) - {"outcome", "final_response", "error"}
            or value.get("outcome") not in {"completed", "failed"}
            or not isinstance(value.get("final_response"), str)
            or len(value["final_response"].encode()) > 131072
            or not value["final_response"].strip()
            or ("error" in value and not isinstance(value["error"], str))
            or (value["outcome"] == "failed" and not value.get("error"))):
        raise HostSessionError("Result requires outcome completed|failed and actual final_response; failed also requires error")
    return dict(value)


class _Session:
    def __init__(self, workspace, *, op_name, runtime, budget, kb_path=None,
                 resource_context=None, external_elapsed_sec=0, _dispatcher=None, _clock=time.time):
        self.workspace = Path(workspace).resolve()
        self.task, self.runtime = self.workspace / "task", Path(runtime).resolve()
        if self.runtime != self.workspace / "runtime":
            raise HostSessionError("Host engine must use this workspace's prepared runtime")
        self.clock = _clock
        self.started = self.clock()
        self.accounted_command_until = self.started
        self.root = self.workspace / "host-session"
        _plain(self.root, self.workspace)
        self.root.mkdir(exist_ok=True)
        self.state_path = self.root / "state.json"
        if (isinstance(external_elapsed_sec, bool) or not isinstance(external_elapsed_sec, (int, float))
                or not math.isfinite(external_elapsed_sec) or external_elapsed_sec < 0):
            raise HostSessionError("external_elapsed_sec must be finite and nonnegative")
        if (not isinstance(budget, dict) or set(budget) != {"max_attempts", "deadline_sec"}
                or type(budget["max_attempts"]) is not int or budget["max_attempts"] < 1
                or not isinstance(budget["deadline_sec"], (int, float))
                or isinstance(budget["deadline_sec"], bool)
                or not math.isfinite(budget["deadline_sec"]) or budget["deadline_sec"] <= 0):
            raise HostSessionError("Host budget requires positive max_attempts and deadline_sec; no invented host turns")
        manifest = self.workspace / MANIFEST
        _read(manifest)
        config = {"workspace": str(self.workspace), "runtime": str(self.runtime), "op_name": op_name,
                  "kb_path": str(kb_path) if kb_path else None, "budget": budget,
                  "manifest_sha256": _sha(manifest.read_bytes()), "resource_context": resource_context}
        if self.state_path.exists():
            self.data = _read(self.state_path)
            if self.data.get("schema_version") != 1 or self.data.get("config") != config:
                raise HostSessionError("Host session configuration or prepared manifest changed")
            cursor = self.data.get("event_checkpoint", {})
            raw = self._event_bytes()
            size = cursor.get("size")
            if (type(size) is not int or size < 0 or len(raw) < size
                    or _sha(raw[:size]) != cursor.get("sha256")
                    or (not self.data.get("operation") and len(raw) != size)):
                raise HostSessionError("Engine history changed outside the durable host journal")
        else:
            if self._event_bytes():
                raise HostSessionError("Cannot adopt an existing non-host engine session")
            self.data = {"schema_version": 1, "config": config, "consumed_sec": 0.0,
                         "operation": None, "completed_requests": {}, "accounted_external_sec": 0.0,
                         "external_overlap_sec": 0.0}
            self.save()
        accounted = self.data.get("accounted_external_sec", 0.0)
        if (isinstance(accounted, bool) or not isinstance(accounted, (int, float))
                or not math.isfinite(accounted) or accounted < 0 or external_elapsed_sec < accounted):
            raise HostSessionError("Cumulative external elapsed time must not decrease")
        delta = external_elapsed_sec - accounted
        if delta:
            # Launcher preflight occurs outside the child. While awaiting a host
            # its elapsed time also lies inside issued_at..now; classify that
            # overlap as external instead of charging those seconds twice.
            overlap = 0.0
            if self._pending():
                op = self.data["operation"]
                overlap = min(delta, max(0.0, self.clock() - op["issued_at"]))
                op["issued_at"] += overlap
            self.data["consumed_sec"] += delta
            self.data["external_overlap_sec"] = self.data.get("external_overlap_sec", 0.0) + overlap
            self.data["accounted_external_sec"] = external_elapsed_sec
            self.save(bind_events=False)
        self.pending_at_entry = self._pending()
        os.environ["ASCENDC_DEBUG_MAX_ATTEMPTS"] = str(budget["max_attempts"])
        # Host conversation turns are not Claude turns. Enforce real attempt and
        # elapsed budgets instead of supplying fabricated zero-turn observations.
        for name in ("ASCENDC_DEBUG_MAX_TASK_TURNS", "ASCENDC_DEBUG_SOFT_TASK_TURNS"):
            os.environ.pop(name, None)
        self.e = _engine(self.runtime)
        self.dispatch = _dispatcher or self.e["runner"]._default_dispatcher

    def save(self, *, bind_events=True):
        if bind_events:
            raw = self._event_bytes()
            self.data["event_checkpoint"] = {"size": len(raw), "sha256": _sha(raw)}
        _write(self.state_path, self.data)

    def _event_bytes(self):
        path = self.task / ".debug_events/events.jsonl"
        if path.is_symlink():
            raise HostSessionError("Linked event stream")
        data = path.read_bytes() if path.exists() else b""
        if data and not data.endswith(b"\n"):
            raise HostSessionError("Truncated engine event tail; preserve evidence for explicit recovery")
        return data

    def _pending(self):
        op = self.data.get("operation")
        return bool(op and op.get("kind") == "host" and op.get("phase") in {"intent", "pending"})

    def consumed(self):
        value = self.data["consumed_sec"]
        if self._pending() and self.data["operation"].get("issued_at") is not None:
            value += max(0.0, self.clock() - self.data["operation"]["issued_at"])
        return value

    def finish_accounting(self, *, bind_events):
        # A pending host round already accounts for commands used to inspect it.
        if not self.pending_at_entry or not self._pending():
            self.data["consumed_sec"] += max(0.0, self.clock() - self.accounted_command_until)
        if self._pending() and not self.pending_at_entry:
            self.data["operation"]["issued_at"] = self.clock()
        self.save(bind_events=bind_events)

    def action(self, op):
        return self.e["types"].Action(**op["action"])

    def suffix(self, op):
        data = self._event_bytes()
        count = op["before_size"]
        if len(data) < count or _sha(data[:count]) != op["before_sha256"]:
            raise HostSessionError("Engine event parent changed")
        return [json.loads(line) for line in data[count:].splitlines()]

    def ensure_started(self, op):
        suffix = self.suffix(op)
        if not suffix:
            self.e["transition"].record_action_started(self.task, self.action(op), op["action_id"])
            suffix = self.suffix(op)
        first = suffix[0]
        if (first.get("type") != "action_started" or first.get("action_id") != op["action_id"]
                or first.get("action") != op["action"]):
            raise HostSessionError("Host journal does not match the engine action")
        return suffix

    def new_operation(self, action, kind):
        data = self._event_bytes()
        return {"kind": kind, "phase": "intent", "action": _action_payload(action),
                "action_id": self.e["transition"].new_action_id(),
                "before_size": len(data), "before_sha256": _sha(data)}

    def diagnostics(self, packet, candidate, sources, attempt):
        """Prepare an independent, mutable probe sandbox; never an acceptance input."""
        root = packet / "diagnostics"
        task = root / "task"
        task.mkdir(parents=True)
        manifest = _read(self.workspace / MANIFEST)
        contract = manifest["task_contract"]
        for name, digest in contract["protected_sha256"].items():
            if Path(name).name != name:
                raise HostSessionError("Protected diagnostic input must be a task basename")
            source = self.task / name
            if source.is_symlink() or _sha(source.read_bytes()) != digest:
                raise HostSessionError("Protected diagnostic input differs from the prepared task")
            shutil.copy2(source, task / name)
        _copy_files(candidate / "kernel", task / "kernel", sources)
        # The frozen verifier reads this legacy alias. Never inherit a leftover
        # alias from another phase: begin with the declared compact-case bytes.
        (task / "model.json").write_bytes((self.task / contract["compact_file"]).read_bytes())
        receipts = root / "startup-receipts"
        receipts.mkdir()
        scripts = str(Path(__file__).resolve().parent)
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        runtime_checks = importlib.import_module("runtime_checks")
        base = dict(os.environ)
        adapter = self.workspace / "adapter"
        base["PYTHONPATH"] = os.pathsep.join(path for path in base.get("PYTHONPATH", "").split(os.pathsep)
                                            if path and path != str(adapter))
        mode = manifest["execution"]["determinism"]
        environment = runtime_checks.execution_environment(base, adapter, mode, receipts)
        selected = {key: value for key, value in environment.items() if key in DIAGNOSTIC_ENVIRONMENT}
        python = sys.executable
        execution = {"schema_version": 1, "python": python, "cwd": str(self.runtime),
            "environment": selected, "clear_environment": ["PYTHONHOME", "PYTHONOPTIMIZE"],
            "clear_prefixes": ["ABLATE_", "ASCENDC_ABLATE_", "ANTHROPIC_", "CLAUDE_CODE_", "OPENAI_API_", "KIMI_", "MOONSHOT_", "CODEX_API_"],
            "commands": {
                "clean_build": [python, "-B", str(self.runtime / "utils/build_ascendc.py"), str(task),
                                "--soc-version", environment.get("ASCENDC_SOC_VERSION", ""), "--clean"],
                "compact_diagnostic": [python, "-B", str(self.runtime / "utils/verification_ascendc.py"), str(task)],
                "forensics": [python, "-B", str(self.runtime / DEBUG_REL / "scripts/precision_forensics.py"),
                              self.data["config"]["op_name"], "--task-dir", str(task), "--attempt", str(attempt)]},
            "scope": "Diagnostic evidence only; no engine checkpoint, formal validation or source promotion"}
        path = root / "execution.json"
        _write(path, execution)
        return {"directory": str(root), "task": str(task), "execution": str(path),
                "execution_sha256": _sha(path.read_bytes()), "initial_source_sha256": sources,
                "protected_sha256": contract["protected_sha256"],
                "compact_alias_sha256": _sha((task / "model.json").read_bytes()),
                "imported_on_submit": False}

    def issue(self, action):
        backend = self.e["agent_backend"]
        attempt = int((action.skill_args or {}).get("attempt", 0))
        failure = (action.skill_args or {}).get("failure_type", "")
        policy = backend._probe_policy(self.task, failure, attempt)
        policy_path = backend._write_probe_policy(self.task, policy)
        request_id = "host_" + uuid4().hex
        packet = self.root / "packets" / request_id
        packet.mkdir(parents=True)
        candidate = packet / "candidate"
        candidate.mkdir()
        before = _inventory(self.task / "kernel", source_only=True)
        _copy_files(self.task / "kernel", candidate / "kernel", before)
        diagnostics = self.diagnostics(packet, candidate, before, attempt)
        audit_names = [f"precision_audit_{attempt}.md", "candidate_kb_entry.json"]
        (candidate / "precision_tuning").mkdir()
        for name in audit_names:
            path = self.task / "precision_tuning" / name
            if path.is_file():
                if path.is_symlink():
                    raise HostSessionError("Linked audit input")
                shutil.copy2(path, candidate / "precision_tuning" / name)
        method = self.runtime / "host-instructions.md"
        if not method.exists():
            method = self.runtime / ".claude/agents/ascendc-debug-agent-constructive.md"
        text = method.read_text(encoding="utf-8")
        body = text
        if method.name != "host-instructions.md":
            _, separator, body = text.partition("\n---\n")
            if not text.startswith("---\n") or not separator:
                raise HostSessionError("Prepared constructive method lacks supported frontmatter")
        instructions = (
            "# Native host repair request\n\n"
            "You are the repair worker; do not launch another model or agent CLI.\n"
            f"Request: {request_id}\nRead-only active task and engine evidence: {self.task}\n"
            f"Editable isolated candidate: {candidate}\n"
            f"Independent diagnostic sandbox: {diagnostics['task']}\n"
            "Read the task's evidence and the method below. Wherever the method refers to editing "
            "task_dir/kernel or its audit, write only the corresponding candidate path above. "
            "Only candidate/kernel/**/*.cpp, .h, .hpp and the two named candidate audits are editable. "
            "Do not edit the active task, runtime, requests, events, checkpoints, references or cases. "
            "Do not build or evaluate inside this source-only candidate; submit the actual diagnosis "
            "and patch so the deterministic engine performs objective validation. Necessary local "
            "compilation, numerical diagnostics and L5 probes are part of the authorized repair. "
            "Run them in the independent diagnostic sandbox above, following the original probe policy "
            "(including a skip policy when required). Refresh its kernel source from the current "
            "candidate before each experiment. Use the exact Python, cwd, environment overlay and "
            f"argument vectors recorded in {diagnostics['execution']}; clear its listed environment "
            "keys/prefixes, retain the installed CANN environment, and do not use -I, -S or -O, which "
            "would bypass the managed deterministic startup. Keep protected diagnostic reference and "
            "compact/full input copies unchanged; create any explicitly labeled reduced probes as "
            "separate diagnostic files. Record actual commands, source changes and observed outputs. "
            "Diagnostic logs, builds and temporary instrumentation stay in that sandbox and are never "
            "imported or accepted as a final result. Remove temporary probes from the source-only "
            "candidate before submission unless the original method explicitly requires them; never "
            "invent probe evidence or change the active task/runtime.\n\n"
            f"Allowed audit files (relative to candidate): {', '.join('precision_tuning/' + name for name in audit_names)}\n\n"
            "Return a result JSON with outcome completed|failed and final_response containing your "
            "actual diagnosis and the required ENGINE_ATTEMPT_METADATA footer. A completed turn "
            "is not an accepted repair. Failed also requires error. Do not report invented turns.\n\n"
            + body + "\n\n" + backend._build_prompt(self.task, self.data["config"]["op_name"], failure, attempt, "0")
            + "\n\n# Final native-host path and ownership override\n"
            + "This section overrides ALL historical editing, initialization, scheduling and exit instructions above. "
            + f"The active task `{self.task}` and its evidence are READ-ONLY to the host. "
            + f"Every instruction to change kernel or audit means the allowed files under `{candidate}`, never the active task. "
            + f"Write audits only to `{candidate / 'precision_tuning' / audit_names[0]}` and `{candidate / 'precision_tuning' / audit_names[1]}`. "
            + f"Necessary diagnostic commands run only in `{diagnostics['task']}` with `{diagnostics['execution']}`. "
            + f"Write the actual completion JSON to `{packet / 'host-result.json'}`; never claim an engine verdict. "
            + "Then use the public Skill's submit/next protocol to return control to the deterministic engine. "
            + "Do not execute historical Initialization Protocol, gates, checkpoints, rollback, termination or agent launch commands yourself. "
            + "Do not create a separate MAX_ATTEMPTS loop. Each submitted request is one engine-owned repair attempt.\n")
        instruction_path = packet / "instructions.md"
        instruction_path.write_text(instructions, encoding="utf-8")
        request = {"schema_version": 1, "request_id": request_id, "attempt": attempt,
                   "failure_type": failure, "candidate": str(candidate), "read_only_task": str(self.task),
                   "instructions": str(instruction_path), "instructions_sha256": _sha(instruction_path.read_bytes()),
                   "manifest_sha256": self.data["config"]["manifest_sha256"],
                   "parent_task_sha256": _inventory(self.task, task=True), "parent_kernel_sha256": before,
                   "resource_context": self.data["config"]["resource_context"], "audit_names": audit_names,
                   "probe_policy": policy, "probe_policy_path": str(policy_path),
                   "probe_source_before": backend._snapshot_probe_sources(self.task),
                   "diagnostics": diagnostics,
                   "budget": {**self.data["config"]["budget"], "accounting": "command_wall_plus_pending_host_round_wall",
                              "agent_turns": None, "agent_turns_available": False}}
        request_path = packet / "request.json"
        _write(request_path, request)
        op = self.new_operation(action, "host")
        op.update(request_id=request_id, request_sha256=_sha(request_path.read_bytes()), issued_at=self.clock())
        self.data["operation"] = op
        self.save()
        self.ensure_started(op)
        op["phase"] = "pending"
        self.save()
        return self.pending()

    def request(self, op):
        packet = self.root / "packets" / op["request_id"]
        _plain(packet, self.workspace)
        path = packet / "request.json"
        request = _read(path)
        if _sha(path.read_bytes()) != op["request_sha256"]:
            raise HostSessionError("Host request record changed")
        instruction_path = packet / "instructions.md"
        if instruction_path.is_symlink() or _sha(instruction_path.read_bytes()) != request["instructions_sha256"]:
            raise HostSessionError("Host instructions changed")
        if request["candidate"] != str(packet / "candidate") or request["request_id"] != op["request_id"]:
            raise HostSessionError("Request path identity changed")
        return packet, request

    def check_parent(self, request):
        if (_sha((self.workspace / MANIFEST).read_bytes()) != request["manifest_sha256"]
                or _inventory(self.task, task=True) != request["parent_task_sha256"]):
            raise HostSessionError("Read-only active task/evidence changed while awaiting the host")

    def pending(self):
        op = self.data["operation"]
        suffix = self.ensure_started(op)
        if len(suffix) != 1:
            raise HostSessionError("Pending host action has unexpected engine events")
        packet, request = self.request(op)
        self.check_parent(request)
        return {"returncode": 0, "state": "awaiting_host", "terminal": False,
                "request_id": op["request_id"], "request": str(packet / "request.json"),
                "candidate": request["candidate"], "instructions": request["instructions"],
                "diagnostics": request["diagnostics"],
                "attempt": request["attempt"], "failure_type": request["failure_type"],
                "budget": {**request["budget"], "consumed_sec": self.consumed()}}

    def snapshot_candidate(self, packet, request):
        candidate = packet / "candidate"
        all_files = _inventory(candidate)
        before = request["parent_kernel_sha256"]
        allowed_audits = {"precision_tuning/" + name for name in request["audit_names"]}
        for name in all_files:
            path = candidate / name
            if path.stat().st_nlink != 1:
                raise HostSessionError("Hard-linked candidate files are not supported")
            rel = Path(name)
            if rel.parts[0] == "kernel":
                source_name = Path(*rel.parts[1:]).as_posix()
                if (len(rel.parts) < 2 or "build" in rel.parts or any(part.startswith(".") for part in rel.parts)
                        or (path.suffix not in EDITABLE and (source_name not in before or all_files[name] != before[source_name]))):
                    raise HostSessionError("Candidate changed a file outside the native-source allowlist")
            elif name not in allowed_audits:
                raise HostSessionError("Candidate contains a file outside the native-source/audit allowlist")
        for name, digest in before.items():
            if Path(name).suffix not in EDITABLE and all_files.get("kernel/" + name) != digest:
                raise HostSessionError("Candidate removed or changed a non-editable source input")
        if not any(name.startswith("kernel/") and Path(name).suffix in EDITABLE for name in all_files):
            raise HostSessionError("Candidate has no native kernel source")
        if "kernel/pybind11.cpp" not in all_files or not (candidate / "kernel/pybind11.cpp").stat().st_size:
            raise HostSessionError("Candidate must retain a nonempty native pybind11.cpp")
        if not any(name.startswith("kernel/") and Path(name).suffix == ".cpp" and Path(name).name != "pybind11.cpp" for name in all_files):
            raise HostSessionError("Candidate must retain a native implementation besides pybind11.cpp")
        submission = packet / "submission"
        if submission.exists():
            if _inventory(submission) != all_files:
                raise HostSessionError("Existing submission snapshot differs; preserve it for inspection")
        else:
            staging = packet / (".submission-" + uuid4().hex)
            _copy_files(candidate, staging, all_files)
            staging.rename(submission)
        if _inventory(candidate) != all_files or _inventory(submission) != all_files:
            raise HostSessionError("Candidate changed during submission snapshot")
        return all_files

    def complete(self, op):
        suffix = self.ensure_started(op)
        result = op["engine_result"]
        expected = {"type": "action_completed", "action_id": op["action_id"],
                    "action": {k: v for k, v in op["action"].items() if k != "expected_artifacts"}, "result": result}
        if len(suffix) == 1:
            self.e["transition"].record_action_completed(self.task, self.action(op), op["action_id"], result)
            suffix = self.suffix(op)
        actual = {k: v for k, v in suffix[1].items() if k != "seq"}
        if actual != expected:
            raise HostSessionError("Completed action does not match durable driver result")
        rollback = None
        if op["action"]["step"] == "checkpoint_and_rollback" and result.get("rolled_back"):
            rollback = {"type": "rollback", "from_attempt": int(result["from_attempt"]),
                        "best_attempt": result.get("best_attempt"), "best_case_pass_rate": result.get("best_case_pass_rate"),
                        "best_match_rate": result.get("best_match_rate")}
            if len(suffix) == 2:
                self.e["transition"].record_rollback(self.task, from_attempt=rollback["from_attempt"],
                    best_metric={"attempt": rollback["best_attempt"], "case_pass_rate": rollback["best_case_pass_rate"],
                                 "match_rate": rollback["best_match_rate"]})
                suffix = self.suffix(op)
        if len(suffix) != (3 if rollback else 2) or (rollback and {k: v for k, v in suffix[2].items() if k != "seq"} != rollback):
            raise HostSessionError("Unexpected events after driver completion")
        if op["action"]["step"] == "validate":
            self.e["exit_artifacts"].write_diagnosis_summary(self.task, int((op["action"]["skill_args"] or {}).get("attempt", 0)))
        if op["kind"] == "host":
            self.data["completed_requests"][op["request_id"]] = {
                "result_sha256": op["result_sha256"], "action_id": op["action_id"],
                "source_sha256": op.get("applied_sources"), "engine_result": result}
        terminal = op.get("terminal_after")
        self.data["operation"] = None
        if terminal:
            self.data["terminal_after"] = terminal
        self.save()

    def apply_submission(self, op):
        packet, request = self.request(op)
        result = op["host_result"]
        if (_sha(_encode(result)) != op["result_sha256"]
                or _read(packet / "submitted-result.json") != result):
            raise HostSessionError("Immutable submitted host result changed")
        if "engine_result" in op:
            if _inventory(self.task / "kernel", source_only=True) != op["applied_sources"]:
                raise HostSessionError("Completed import source changed before event commit")
            self.complete(op)
            return
        if result["outcome"] == "completed":
            sub = packet / "submission"
            if _inventory(sub) != op["submitted_files"]:
                raise HostSessionError("Immutable submitted candidate changed")
            old, active, staged = packet / "previous-kernel", self.task / "kernel", sub / "kernel"
            staged_sources = _inventory(staged)
            # Keep the active directory and required pybind entry present even
            # across process interruption. Each file is an atomic replacement;
            # the journal permits only original/submitted bytes during recovery.
            # No numerical action can run until the complete import is bound.
            if not old.exists():
                self.check_parent(request)
                previous = packet / (".previous-" + uuid4().hex)
                _copy_files(active, previous, request["parent_kernel_sha256"])
                previous.rename(old)
            if _inventory(old) != request["parent_kernel_sha256"]:
                raise HostSessionError("Interrupted kernel import lost its original source backup")
            current = _inventory(active, source_only=True)
            for name, digest in current.items():
                if digest not in {request["parent_kernel_sha256"].get(name), staged_sources.get(name)}:
                    raise HostSessionError("Active source changed outside the journaled import")
            allowed = {"kernel/" + name for name in set(request["parent_kernel_sha256"]) | set(staged_sources)}
            allowed |= {"precision_tuning/" + name for name in request["audit_names"]}
            actual = _inventory(self.task, task=True)
            if ({k: v for k, v in actual.items() if k not in allowed}
                    != {k: v for k, v in request["parent_task_sha256"].items() if k not in allowed}):
                raise HostSessionError("Read-only evidence changed during journaled import")
            for name in staged_sources:
                _bytes(active / name, (staged / name).read_bytes())
            for name in current.keys() - staged_sources.keys():
                if Path(name).suffix not in EDITABLE:
                    raise HostSessionError("Journal tried to remove non-editable source")
                removed = packet / "removed-kernel" / name
                removed.parent.mkdir(parents=True, exist_ok=True)
                if removed.exists():
                    raise HostSessionError("Removed source backup already exists")
                (active / name).rename(removed)
            if _inventory(active, source_only=True) != staged_sources:
                raise HostSessionError("Active kernel differs from immutable submission")
            for name in request["audit_names"]:
                source = sub / "precision_tuning" / name
                if not source.exists():
                    continue
                dest = self.task / "precision_tuning" / name
                backup = packet / "previous-audits" / name
                if dest.exists() and not backup.exists():
                    backup.parent.mkdir(exist_ok=True)
                    shutil.copy2(dest, backup)
                _bytes(dest, source.read_bytes())
        if "engine_result" not in op:
            backend = self.e["agent_backend"]
            final = result["final_response"].strip()
            metadata = backend._parse_attempt_metadata(final)
            op["applied_sources"] = _inventory(self.task / "kernel", source_only=True)
            engine_result = {"success": result["outcome"] == "completed", "backend": "host",
                             "host_outcome": result["outcome"], "request_id": op["request_id"],
                             "agent_turns": None, "agent_turns_available": False,
                             "budget_accounting": "host_attempts_and_wall_time",
                             "final_response": final[:backend._FINAL_RESPONSE_MAXLEN],
                             "attempt_metadata": metadata, "error": result.get("error"),
                             "changed_files": ["kernel/" + name for name in sorted(set(request["parent_kernel_sha256"]) | set(op["applied_sources"]))
                                               if request["parent_kernel_sha256"].get(name) != op["applied_sources"].get(name)]}
            for key in ("fix_type", "direction_verdict", "direction_reason"):
                if key in metadata:
                    engine_result[key] = metadata[key]
            backend._attach_probe_observability(engine_result, self.task, request["attempt"],
                request["probe_policy"], Path(request["probe_policy_path"]), request["probe_source_before"])
            backend._write_kb_usage_trace(self.task, request["attempt"], final, metadata)
            op["engine_result"] = engine_result
            if engine_result.get("ablation_violation"):
                op["terminal_after"] = {"category": "ablation_violation", "reason": "Host patch violated engine probe policy",
                                        "details": {"session_outcome": "ablation_violation"}}
            op["phase"] = "result_ready"
            self.save()
        self.complete(op)

    def terminate(self, decision):
        status = self.e["runner"]._terminate(self.task, decision, op_name=self.data["config"]["op_name"],
                                             kb_path=self.data["config"]["kb_path"])
        return {"returncode": OUTCOME_CODES.get(status.get("session_outcome"), 7),
                "state": "terminal", "terminal": True, "engine_status": status}

    def advance(self):
        state = self.e["state"].DebugState.load(self.task)
        if state.is_terminal():
            if self.data.get("operation"):
                raise HostSessionError("Unexpected terminal event while a journaled action is pending")
            status = self.e["exit_artifacts"].build_debug_status(self.task)
            return {"returncode": OUTCOME_CODES.get(status.get("session_outcome"), 7),
                    "state": "terminal", "terminal": True, "engine_status": status}
        if not state.events:
            self.e["transition"].record_session_started(self.task, op_name=self.data["config"]["op_name"],
                agent="constructive_host", entry_failure_type="precision_failed")
        for _ in range(200):
            op = self.data.get("operation")
            if op:
                if op["kind"] == "host":
                    if op["phase"] in {"applying", "result_ready"}:
                        self.apply_submission(op)
                        continue
                    self.ensure_started(op)
                    op["phase"] = "pending"
                    self.save()
                    if self.consumed() < self.data["config"]["budget"]["deadline_sec"]:
                        return self.pending()
                    self.pending()  # bind the parent before recording a timeout
                    self.data["consumed_sec"] = self.consumed()
                    self.accounted_command_until = self.clock()
                    op.update(phase="result_ready", result_sha256=None,
                        engine_result={"success": False, "backend": "host", "request_id": op["request_id"],
                                       "error": "host_round_wall_budget_exhausted", "agent_turns": None})
                    self.save()
                    self.complete(op)
                    return self.terminate(self.e["types"].Done("timeout", "Host round wall-clock budget exhausted"))
                self.ensure_started(op)
                if op["phase"] == "dispatching":
                    return {"returncode": 2, "state": "needs_attention", "terminal": False,
                            "reason": "interrupted_deterministic_action", "action_id": op["action_id"],
                            "step": op["action"]["step"], "evidence_preserved": True}
                if op["phase"] == "result_ready":
                    self.complete(op)
                    continue
                op["phase"] = "dispatching"
                self.save()
                try:
                    result = self.dispatch(self.action(op), self.task, self.data["config"]["op_name"], None)
                except Exception as exc:
                    result = {"success": False, "error": str(exc), "fatal": True}
                    op["terminal_after"] = {"category": "dispatch_error", "reason": str(exc),
                                            "details": {"session_outcome": "crashed"}}
                if not isinstance(result, dict):
                    raise HostSessionError("Deterministic dispatcher returned no result object")
                op.update(phase="result_ready", engine_result=result)
                self.save()
                self.complete(op)
                continue
            if self.data.get("terminal_after"):
                return self.terminate(self.e["types"].Abort(**self.data["terminal_after"]))
            if self.consumed() + max(0.0, self.clock() - self.accounted_command_until) >= self.data["config"]["budget"]["deadline_sec"]:
                return self.terminate(self.e["types"].Done("timeout", "Host session wall-clock budget exhausted"))
            state = self.e["state"].DebugState.load(self.task)
            if state.dangling_started():
                raise HostSessionError("Unjournaled engine action exists; refusing implicit crash reconciliation")
            decision = self.e["next_action"].debug_next_action(state)
            types = self.e["types"]
            if isinstance(decision, (types.Done, types.Abort, types.Escalate)):
                return self.terminate(decision)
            if isinstance(decision, types.Continue):
                self.e["transition"].record_attempt_started(self.task, decision)
                continue
            self.e["runner"].validate_action(decision)
            if decision.kind == "spawn_agent":
                return self.issue(decision)
            if decision.step == "knowledge_search" and self.data["config"]["kb_path"]:
                decision.skill_args = dict(decision.skill_args or {}, kb_path=self.data["config"]["kb_path"])
            self.data["operation"] = self.new_operation(decision, "deterministic")
            self.save()
        raise HostSessionError("Host state-machine tick limit exceeded; no terminal success inferred")

    def submit(self, request_id, result):
        result = _result(result)
        digest = _sha(_encode(result))
        previous = self.data["completed_requests"].get(request_id)
        if previous is not None:
            if previous["result_sha256"] != digest:
                raise HostSessionError("This request already has a different submitted result")
            value = self.advance()
            value.update(already_submitted=True, submitted_request_id=request_id)
            return value
        op = self.data.get("operation")
        if not op or op.get("kind") != "host" or op.get("request_id") != request_id:
            raise HostSessionError("Request is not the current pending host action")
        if op["phase"] in {"applying", "result_ready"}:
            if op["result_sha256"] != digest:
                raise HostSessionError("An interrupted submission has a different durable result")
            self.apply_submission(op)
            return self.advance()
        self.pending()
        if self.consumed() >= self.data["config"]["budget"]["deadline_sec"]:
            return self.advance()
        packet, request = self.request(op)
        files = self.snapshot_candidate(packet, request)
        result_path = packet / "submitted-result.json"
        if result_path.exists():
            if _read(result_path) != result:
                raise HostSessionError("An existing immutable host result differs; preserve the earlier evidence")
        else:
            _write(result_path, result)
        self.data["consumed_sec"] = self.consumed()
        self.accounted_command_until = self.clock()
        op.update(phase="applying", result_sha256=digest, host_result=result, submitted_files=files,
                  submitted_sources={name[len("kernel/"):]: h for name, h in files.items() if name.startswith("kernel/")})
        self.save()
        self.apply_submission(op)
        return self.advance()


def advance(workspace, **kwargs):
    session = _Session(workspace, **kwargs)
    completed = False
    try:
        result = session.advance()
        completed = True
        return result
    finally:
        session.finish_accounting(bind_events=completed)
        if completed and "budget" in result:
            result["budget"]["consumed_sec"] = session.consumed()


def submit(workspace, *, packet_id=None, request_id=None, result, **kwargs):
    if packet_id and request_id and packet_id != request_id:
        raise HostSessionError("Conflicting packet/request identifiers")
    session = _Session(workspace, **kwargs)
    completed = False
    try:
        value = session.submit(request_id or packet_id, result)
        completed = True
        return value
    finally:
        session.finish_accounting(bind_events=completed)
        if completed and "budget" in value:
            value["budget"]["consumed_sec"] = session.consumed()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--operation", choices=("next", "submit"))
    args = parser.parse_args(argv)
    try:
        config = _read(Path(args.config))
        operation = args.operation or config.pop("operation", "next")
        config.pop("operation", None)
        if not (Path(config["workspace"]) / ".drdebug.lock").is_file():
            raise HostSessionError("The public launcher must hold the exclusive workspace lock")
        if operation == "next":
            config.pop("request_id", None)
            config.pop("result", None)
            function = advance
        else:
            function = submit
        with redirect_stdout(sys.stderr):
            value = function(**config)
    except (Exception, KeyboardInterrupt) as exc:
        value = {"returncode": 130 if isinstance(exc, KeyboardInterrupt) else 2,
                 "state": "needs_attention", "terminal": False,
                 "error": type(exc).__name__ + ": " + str(exc), "evidence_preserved": True}
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return value["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
