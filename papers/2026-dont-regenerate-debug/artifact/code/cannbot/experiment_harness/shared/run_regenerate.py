#!/usr/bin/env python3
"""Run isolated CANNBot Regenerate trials inside the dedicated container."""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing.connection import wait as wait_for_connections
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_SKILLS = (
    "ascendc-precision-debug",
    "ascendc-crash-debug",
    "ascendc-runtime-debug",
)
PRINT_LOCK = threading.Lock()
TRIAL_CONTAINER_PREFIX = "cannbot-regen-"
DELIVERABLE_CONTRACT_VERSION = 2
NATIVE_SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".asc"})
STANDARD_SOURCE_DIRS = frozenset({"op_extension", "op_host", "op_kernel"})
IGNORED_SOURCE_DIRS = frozenset(
    {
        ".git",
        ".home",
        "build",
        "cmake-build-debug",
        "cmake-build-release",
        "candidate_task",
        "evidence",
        "test",
        "tests",
        "third_party",
        "validation",
        "vendor",
    }
)
DELIVERABLE_RESUMABLE_STATUSES = frozenset({"deliverable_interrupted"})


@dataclass(frozen=True)
class ProviderSlot:
    """One live Kimi key slot and its private environment file."""

    name: str
    path: Path


class TrialIsolationPreflightError(RuntimeError):
    """A harness-control isolation failure, not a model/provider failure."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def host_busy_npus() -> set[str] | None:
    """Return physical NPU ids with host processes, conservatively.

    Formal C2 workers must never share an NPU with another user's process.
    Failure to inspect the process table is therefore treated as "unknown"
    by callers, which retain the configured queue but do not claim a free-card
    guarantee in their evidence.
    """

    try:
        completed = subprocess.run(
            ["npu-smi", "info"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    busy: set[str] = set()
    process_row = re.compile(r"^\|\s*(\d+)\s+\d+\s+\|\s+\d+\s+\|")
    for line in completed.stdout.splitlines():
        match = process_row.match(line)
        if match:
            busy.add(match.group(1))
    return busy


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_as_trial_user(
    *, container: str, cwd: Path, uid: int, gid: int, path: Path, content: str, mode: int
) -> None:
    """Create an immediately-consumed file from inside the bind-mounted container.

    This avoids a short SELinux/bind-mount visibility window observed for new
    host-created inodes while retaining the per-trial numeric UID boundary.
    File content is sent over stdin and never appears in the process argv.
    """
    relative = path.relative_to(cwd)
    command = [
        "docker",
        "exec",
        "-i",
        "-u",
        f"{uid}:{gid}",
        "-w",
        str(cwd),
        container,
        "/bin/bash",
        "-c",
        f"umask 077; cat > {shlex.quote(str(relative))}; chmod {mode:o} {shlex.quote(str(relative))}",
    ]
    subprocess.run(command, input=content, text=True, check=True, stdout=subprocess.DEVNULL)


def trial_container_name(record: dict[str, Any]) -> str:
    """Return a deterministic Docker name scoped to one formal trial."""

    run_id = str(record["run_id"])
    safe = "".join(character.lower() if character.isalnum() else "-" for character in run_id)
    safe = "-".join(part for part in safe.split("-") if part)
    session = str(record.get("expected_session_id") or "session").replace("-", "")[:8]
    return f"{TRIAL_CONTAINER_PREFIX}{safe[:36]}-{session}"


def start_trial_container(
    *,
    record: dict[str, Any],
    cwd: Path,
    home: Path,
    plugin: Path,
    devkit: Path,
    npu: str,
    image: str,
) -> str:
    """Start an ephemeral container that cannot see any sibling/Debug run.

    The seed image contains CANN and Claude Code.  Host visibility is an
    explicit allowlist: this trial cwd (rw), the sanitized plugin (ro), and the
    public Ascend devkit (ro).  In particular, the experiment root itself is
    never mounted.
    """

    name = trial_container_name(record)
    if not name.startswith(TRIAL_CONTAINER_PREFIX):
        raise RuntimeError(f"unsafe trial container name: {name}")
    # A launcher crash may leave a same-trial container behind.  It has no
    # durable state outside cwd and must not be reused by a fresh session.
    subprocess.run(
        ["docker", "rm", "--force", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    command = [
        "docker",
        "run",
        "--detach",
        "--rm",
        "--name",
        name,
        "--privileged",
        "--security-opt",
        "label=disable",
        "--shm-size",
        "64m",
        "-e",
        f"HOME={home}",
        "-e",
        f"ASCEND_RT_VISIBLE_DEVICES={npu}",
        "-e",
        "USER=cannbot",
        "-e",
        "LOGNAME=cannbot",
        "-e",
        "TORCHINDUCTOR_CACHE_DIR=/tmp/cannbot_torchinductor",
        "-e",
        f"ASCEND_PROCESS_LOG_PATH={home}/ascend_log",
        "--mount",
        "type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver",
        "--mount",
        "type=bind,src=/usr/local/Ascend/firmware,dst=/usr/local/Ascend/firmware",
        "--mount",
        "type=bind,src=/usr/local/bin/npu-smi,dst=/usr/local/bin/npu-smi",
        "--mount",
        f"type=bind,src={cwd},dst={cwd}",
        "--mount",
        f"type=bind,src={plugin},dst={plugin},readonly",
        "--mount",
        f"type=bind,src={devkit},dst={devkit},readonly",
        image,
        "sleep",
        "infinity",
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    return name


def stop_trial_container(name: str) -> None:
    if not name.startswith(TRIAL_CONTAINER_PREFIX):
        raise RuntimeError(f"refusing to remove non-trial container: {name}")
    subprocess.run(
        ["docker", "rm", "--force", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def verify_trial_container_isolation(
    *, container: str, cwd: Path, experiment_root: Path, plugin: Path, devkit: Path, uid: int, gid: int
) -> dict[str, Any]:
    """Prove the allowlisted container view before any model call is made."""

    probe = r'''
import json
import os
from pathlib import Path

cwd = Path(os.environ["CANNBOT_TRIAL_CWD"]).resolve()
root = Path(os.environ["CANNBOT_EXPERIMENT_ROOT"])
plugin = Path(os.environ["CANNBOT_PLUGIN"])
devkit = Path(os.environ["CANNBOT_DEVKIT"])
forbidden = [
    root / "debug_fresh_sources",
    root / "debug_initial_sources",
    root / "frozen_debug_engine",
    root / "experiment_harness",
    Path("/workspace/debug_agent"),
]
visible_forbidden = [str(path) for path in forbidden if path.exists()]
other_run_files = []
runs_root = root / "runs"
if runs_root.exists():
    for path in runs_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            # Check the directory entry's lexical location.  Resolving the
            # final component follows legitimate build symlinks (for example
            # work_tmp/cmodel_shim/libruntime.so) outside the workspace and
            # incorrectly reports the in-workspace entry as another run.
            Path(os.path.abspath(path)).relative_to(cwd)
        except ValueError:
            other_run_files.append(str(path))
            if len(other_run_files) >= 20:
                break
report = {
    "own_spec_readable": (cwd / "spec/model.py").is_file() and os.access(cwd / "spec/model.py", os.R_OK),
    "plugin_readable": plugin.is_dir() and os.access(plugin, os.R_OK),
    "devkit_readable": devkit.is_dir() and os.access(devkit, os.R_OK),
    "visible_forbidden": visible_forbidden,
    "other_run_files": other_run_files,
}
report["passed"] = bool(
    report["own_spec_readable"]
    and report["plugin_readable"]
    and report["devkit_readable"]
    and not visible_forbidden
    and not other_run_files
)
print(json.dumps(report))
'''
    command = [
        "docker",
        "exec",
        "-i",
        "-u",
        f"{uid}:{gid}",
        "-e",
        f"CANNBOT_TRIAL_CWD={cwd}",
        "-e",
        f"CANNBOT_EXPERIMENT_ROOT={experiment_root}",
        "-e",
        f"CANNBOT_PLUGIN={plugin}",
        "-e",
        f"CANNBOT_DEVKIT={devkit}",
        container,
        "python3",
        "-",
    ]
    completed = subprocess.run(command, input=probe, text=True, check=True, stdout=subprocess.PIPE)
    report = json.loads(completed.stdout)
    if not report.get("passed"):
        raise TrialIsolationPreflightError(
            f"trial container isolation failed: {report}"
        )
    return report


def audit_isolation_access_attempts(home: Path, cwd: Path, experiment_root: Path) -> list[dict[str, str]]:
    """Record model tool calls that tried to reach a non-allowlisted input.

    The OS-level mount allowlist prevents those paths from resolving.  This
    journal audit is a second, diagnostic layer so an attempted cross-arm read
    is visible in the formal run record instead of being silently ignored.
    """

    fixed_markers = (
        "debug_fresh_sources",
        "debug_initial_sources",
        "frozen_debug_engine",
        "/workspace/debug_agent",
        "/debug/validation",
    )
    own_prefix = str(cwd.resolve())
    runs_prefix = str((experiment_root / "runs").resolve())
    attempts: list[dict[str, str]] = []
    project_root = home / ".claude" / "projects"
    for journal in project_root.rglob("*.jsonl") if project_root.is_dir() else []:
        for line in journal.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = event.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                serialized = json.dumps(block.get("input"), ensure_ascii=False)
                residual = serialized.replace(own_prefix, "")
                matched = [marker for marker in fixed_markers if marker in serialized]
                if runs_prefix in residual:
                    matched.append("other_run_path")
                if not matched:
                    continue
                attempts.append(
                    {
                        "timestamp": str(event.get("timestamp") or ""),
                        "tool": str(block.get("name") or ""),
                        "markers": ",".join(sorted(set(matched))),
                        "input_excerpt": " ".join(serialized.split())[:500],
                    }
                )
                if len(attempts) >= 100:
                    return attempts
    return attempts


def _quota_number(detail: Any, field: str) -> float | None:
    if not isinstance(detail, dict):
        return None
    value = detail.get(field)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def query_kimi_capacity(item: dict[str, Any], base_url: str, timeout: int) -> dict[str, Any]:
    """Return a token-free live quota snapshot for one Kimi provider."""
    state: dict[str, Any] = {"name": item.get("name", "unnamed"), "queried_at": now()}
    token = item.get("api_key")
    if not isinstance(token, str) or not token:
        return {**state, "success": False, "error": "missing_api_key"}
    if "api.kimi.com/coding" not in base_url.lower():
        return {**state, "success": False, "error": "unsupported_base_url"}
    request = urllib.request.Request(
        "https://api.kimi.com/coding/v1/usages",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json", "User-Agent": "claude-code"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            state["http_status"] = response.status
            body = json.loads(response.read(50_000).decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return {**state, "success": False, "http_status": exc.code, "error": "http_error"}
    except Exception as exc:
        return {**state, "success": False, "error": type(exc).__name__}

    windows: list[tuple[str, Any]] = [("weekly", body.get("usage"))]
    limits = body.get("limits")
    if isinstance(limits, list):
        for index, limit_window in enumerate(limits):
            if isinstance(limit_window, dict):
                windows.append(("five_hour" if index == 0 else f"limit_{index}", limit_window.get("detail")))
    windows.append(("total_quota", body.get("totalQuota")))
    normalized: dict[str, Any] = {}
    remaining_values: list[float] = []
    for name, detail in windows:
        limit = _quota_number(detail, "limit")
        used = _quota_number(detail, "used")
        remaining = _quota_number(detail, "remaining")
        if remaining is None and limit is not None and used is not None:
            remaining = max(0.0, limit - used)
        remaining_percent = (
            max(0.0, remaining) * 100.0 / limit
            if remaining is not None and limit not in {None, 0.0}
            else None
        )
        normalized[name] = {
            "limit": limit,
            "used": used,
            "remaining": remaining,
            "remaining_percent": remaining_percent,
            "reset_time": detail.get("resetTime") if isinstance(detail, dict) else None,
        }
        if remaining_percent is not None:
            remaining_values.append(remaining_percent)
    return {
        **state,
        "success": True,
        "http_status": state.get("http_status"),
        "windows": normalized,
        "capacity_percent": min(remaining_values) if remaining_values else None,
    }


def _capacity_probe_child(
    connection: Any,
    item: dict[str, Any],
    base_url: str,
    timeout: int,
) -> None:
    """Run one quota request outside the scheduler process.

    urllib's timeout does not cover every possible kernel/TLS stall.  Keeping
    the request in a disposable child lets the parent enforce a real wall-clock
    deadline without leaving a stuck thread behind.
    """

    try:
        connection.send(query_kimi_capacity(item, base_url, timeout))
    except BaseException as exc:  # pragma: no cover - defensive child boundary
        connection.send(
            {
                "name": item.get("name", "unnamed"),
                "queried_at": now(),
                "success": False,
                "error": f"probe_child_{type(exc).__name__}",
            }
        )
    finally:
        connection.close()


def query_kimi_capacities_bounded(
    items: list[dict[str, Any]],
    base_url: str,
    timeout: int,
) -> list[dict[str, Any]]:
    """Probe all providers concurrently with one hard overall deadline."""

    if not items:
        return []
    context = multiprocessing.get_context("spawn")
    results: list[dict[str, Any] | None] = [None] * len(items)
    pending: dict[Any, tuple[int, multiprocessing.Process]] = {}
    processes: list[multiprocessing.Process] = []
    for index, item in enumerate(items):
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_capacity_probe_child,
            args=(sender, item, base_url, timeout),
            daemon=True,
        )
        process.start()
        sender.close()
        pending[receiver] = (index, process)
        processes.append(process)

    deadline = time.monotonic() + max(1, timeout) + 5
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        ready = wait_for_connections(list(pending), timeout=remaining)
        if not ready:
            break
        for receiver in ready:
            index, _ = pending.pop(receiver)
            try:
                results[index] = receiver.recv()
            except (EOFError, OSError):
                results[index] = {
                    "name": items[index].get("name", "unnamed"),
                    "queried_at": now(),
                    "success": False,
                    "error": "probe_child_eof",
                }
            finally:
                receiver.close()

    for receiver, (index, process) in pending.items():
        results[index] = {
            "name": items[index].get("name", "unnamed"),
            "queried_at": now(),
            "success": False,
            "error": "hard_timeout",
        }
        receiver.close()
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
    return [result for result in results if result is not None]


def explicit_quota_unavailable_until(
    live_state: dict[str, Any],
    wall_now: float,
    fallback_seconds: int,
    diagnostic_text: str = "",
) -> tuple[float, str, list[str]]:
    """Map an explicit quota error to the window that is actually exhausted.

    A provider can exhaust its five-hour window while retaining weekly quota.
    Every exhausted window must reset before the key is usable, hence the
    latest reset among exhausted windows is required.  If the read-only usage
    endpoint shows no exhausted window despite a real request returning a
    quota error, use the bounded fallback instead of guessing "weekly".
    """

    # Some Anthropic-compatible gateways put the authoritative reset only in
    # the failed generation response, e.g. "5-hour quota ... reset at
    # 07-24 04:09:00 UTC". Prefer that real-request evidence over an absent or
    # non-authoritative usage probe. The gateway omits the year, so infer it
    # from the request time.
    reset_match = re.search(
        r"(?P<window>\d+(?:-\w+)?|weekly)?\s*quota\b.*?"
        r"reset\s+at\s+(?P<month>\d{2})-(?P<day>\d{2})\s+"
        r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})\s+UTC",
        diagnostic_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if reset_match:
        request_time = datetime.fromtimestamp(wall_now, timezone.utc)
        parsed = datetime(
            request_time.year,
            int(reset_match.group("month")),
            int(reset_match.group("day")),
            int(reset_match.group("hour")),
            int(reset_match.group("minute")),
            int(reset_match.group("second")),
            tzinfo=timezone.utc,
        )
        if parsed.timestamp() < wall_now - 180 * 24 * 60 * 60:
            parsed = parsed.replace(year=parsed.year + 1)
        window = (reset_match.group("window") or "request_reported").lower()
        return parsed.timestamp(), "actual_request_explicit_reset", [window]

    exhausted: list[str] = []
    reset_epochs: list[float] = []
    missing_reset = False
    for name, window in (live_state.get("windows") or {}).items():
        if not isinstance(window, dict):
            continue
        remaining_percent = _quota_number(window, "remaining_percent")
        remaining = _quota_number(window, "remaining")
        limit = _quota_number(window, "limit")
        used = _quota_number(window, "used")
        is_exhausted = remaining_percent is not None and remaining_percent <= 0
        if remaining_percent is None:
            is_exhausted = (
                remaining is not None and remaining <= 0
            ) or (
                remaining is None
                and limit is not None
                and used is not None
                and used >= limit
            )
        if not is_exhausted:
            continue
        exhausted.append(str(name))
        reset_time = window.get("reset_time")
        if not reset_time:
            missing_reset = True
            continue
        try:
            parsed = datetime.fromisoformat(str(reset_time).replace("Z", "+00:00"))
        except ValueError:
            missing_reset = True
            continue
        if parsed.timestamp() > wall_now:
            reset_epochs.append(parsed.timestamp())

    fallback_until = wall_now + max(0, fallback_seconds)
    if exhausted and not missing_reset and len(reset_epochs) == len(exhausted):
        return max(reset_epochs), "exhausted_usage_window_reset", exhausted
    if exhausted:
        return fallback_until, "exhausted_window_missing_future_reset", exhausted
    return fallback_until, "actual_request_quota_error_without_exhausted_usage_window", []


def make_provider_envs(
    key_config: Path,
    output_dir: Path,
    model: str,
    slots_per_key: int,
    min_provider_remaining: float,
    quota_timeout: int,
) -> list[ProviderSlot]:
    data = load(key_config)
    keys = data.get("keys", [])
    if not keys:
        raise RuntimeError(f"no API keys in {key_config}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    os.chown(output_dir, 0, 0)
    for stale in output_dir.glob("worker_*.env"):
        stale.unlink()
    slots: list[ProviderSlot] = []
    base_url = str(data.get("base_url") or "https://api.kimi.com/coding/")
    quota_probe = str(data.get("quota_probe") or "kimi_usage")
    if quota_probe == "disabled":
        quota = [
            {
                "name": str(item.get("name", "unnamed")),
                "queried_at": now(),
                "success": True,
                "capacity_percent": 100.0,
                "probe_mode": "disabled_static_anthropic_compatible_provider",
            }
            for item in keys
        ]
    elif quota_probe == "kimi_usage":
        quota = query_kimi_capacities_bounded(keys, base_url, quota_timeout)
    else:
        raise RuntimeError(f"unsupported quota_probe mode: {quota_probe}")
    quota_by_name = {str(item.get("name")): item for item in quota}
    eligible = []
    for item in keys:
        state = quota_by_name.get(str(item.get("name")), {})
        capacity = state.get("capacity_percent")
        if state.get("success") and isinstance(capacity, (int, float)) and capacity >= min_provider_remaining:
            eligible.append((float(capacity), item))
    if eligible:
        keys = [item for _, item in sorted(eligible, key=lambda pair: pair[0], reverse=True)]
    else:
        raise RuntimeError(f"no provider has at least {min_provider_remaining:g}% live capacity")
    atomic_json(
        output_dir / "provider_selection.json",
        {
            "generated_at": now(),
            "quota_probe": quota_probe,
            "minimum_remaining_percent": min_provider_remaining,
            "selected": [item.get("name", "unnamed") for item in keys],
            "quota": quota,
        },
    )
    slot_index = 0
    for item in keys:
        token = item.get("api_key")
        if not isinstance(token, str) or not token:
            continue
        values = {
            "ANTHROPIC_API_KEY": token,
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_FABLE_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "CLAUDE_CODE_SUBAGENT_MODEL": model,
            "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "1048576",
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "1048576",
            "CLAUDE_CODE_EFFORT_LEVEL": "max",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "IS_SANDBOX": "1",
        }
        for replica in range(slots_per_key):
            safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(item.get("name", "provider")))
            path = output_dir / f"worker_{slot_index:02d}_{safe_name}_replica{replica + 1}.env"
            lines = [f"export {name}={shlex.quote(value)}" for name, value in values.items()]
            if str(data.get("auth_mode") or "api_key") == "auth_token":
                lines.append(f"export ANTHROPIC_AUTH_TOKEN={shlex.quote(token)}")
                lines.append("unset ANTHROPIC_API_KEY")
            else:
                lines.append("unset ANTHROPIC_AUTH_TOKEN")
            path.write_text("\n".join(lines) + "\n")
            path.chmod(0o600)
            os.chown(path, 0, 0)
            slots.append(ProviderSlot(name=str(item.get("name", "unnamed")), path=path))
            slot_index += 1
    if not slots:
        raise RuntimeError(f"no usable API keys in {key_config}")
    return slots


def deliverable_contract() -> str:
    """Return the single source of truth shared by fresh and resumed prompts.

    CANNBot's historical bootstrap script describes a flat project layout,
    while the experiment adapter originally accepted only the newer direct-
    invoke layout.  Keeping the exact evaluator contract in this harness avoids
    silently asking the model for one layout and validating another.
    """

    return f"""External evaluator deliverable contract (version {DELIVERABLE_CONTRACT_VERSION}, mandatory):
- The directory named by `candidate_adapter/operator_dir.txt` is the generated operator project.
- New work must use the standard direct-invoke source layout inside that project: `op_extension/`, `op_host/`, and `op_kernel/`. Put registration/extension, host/tiling, and AscendC kernel sources in their appropriate directories; do not leave the implementation only in a legacy flat `kernel/`, `torch_library/`, or project-root layout.
- `candidate_adapter/model_new_ascendc.py` and `candidate_adapter/operator_dir.txt` must both exist. The pointer must be one relative path inside this trial cwd.
- The project must contain at least one generated shared library (`*.so`) and native implementation sources (`.c/.cc/.cpp/.cxx/.h/.hpp/.asc`).
- Do not report the trial complete until all of the above exist. The harness independently enforces this contract before accepting `completed`.
"""


def cannbot_debug_deliverable_contract() -> str:
    return f"""External evaluator deliverable contract (version {DELIVERABLE_CONTRACT_VERSION}, mandatory):
- Continue editing the existing failed implementation under `./task/`; do not create a replacement operator project.
- Keep `./candidate_adapter/model_new_ascendc.py` compatible with `./spec/model.py` and keep `./candidate_adapter/operator_dir.txt` equal to `task`.
- The existing operator project must contain the repaired native AscendC/host/extension sources and at least one real built `*.so`.
- The wrapper may call only the registered custom NPU operator for computation. It must not compute with PyTorch, NumPy, or the reference implementation.
- Do not report success until the candidate builds and its available local NPU tests pass. The harness independently materializes it and later runs the frozen external full-eval and anti-cheat.
"""


def contract_repair_directive(note: str | None = None, *, arm: str = "regenerate") -> str:
    observed = note or "the external evaluator deliverable gate was not satisfied"
    if arm == "cannbot-debug":
        return f"""Repair only the packaging/build deliverable for the existing debugged implementation; do not restart diagnosis or generate a replacement operator. The previous gate result was: `{observed}`.

Preserve `./task/` and all valid debugging evidence. Rebuild its real shared library if needed, create or correct `./candidate_adapter/model_new_ascendc.py`, keep `./candidate_adapter/operator_dir.txt` equal to `task`, and verify the CANNBot Debug deliverable contract before finishing."""
    return f"""Repair only the external evaluator deliverable for the existing implementation; do not regenerate working code from scratch. The previous gate result was: `{observed}`.

Inspect the existing project and preserve all usable source, build, and NPU evidence. If the implementation is in a legacy flat layout, organize the owned implementation into `op_extension/`, `op_host/`, and `op_kernel/` as appropriate and keep the build working. Create or correct both candidate_adapter files, ensure the pointer stays inside this trial cwd, rebuild the real shared library if necessary, and verify the mandatory contract before finishing."""


def prompt_for(record: dict[str, Any], *, resume_session: bool = False) -> str:
    operator = record["operator"]
    if record.get("arm") == "cannbot-debug":
        contract = cannbot_debug_deliverable_contract()
        soft_turns = int(record.get("soft_task_turns") or 480)
        hard_turns = int(record.get("max_task_turns") or 600)
        budget = (
            f"- The task-level turn budget has a soft boundary at {soft_turns} "
            f"turns and a hard boundary at {hard_turns} turns. After the soft "
            "boundary, prioritize converging, rebuilding, testing, and packaging "
            "the best existing repair. The harness permits at most "
            f"{int(record.get('max_internal_attempts') or 5)} valid same-session "
            "Agent invocations; provider/infrastructure-invalid invocations do "
            "not consume this allowance.\n"
        )
        if resume_session:
            resume_directive = ""
            if record.get("resume_directive"):
                resume_directive = f"- Trial-specific resume directive: {record['resume_directive']}\n"
            return f"""Continue the same formal CANNBot `ascendc-precision-debug` run for `{operator}` after an infrastructure/provider/deliverable interruption.

- Resume the same Claude session, `./task/` implementation, build state, and debugging evidence.
- Invoke and continue to follow `/ascendc-precision-debug`; do not switch to a generation workflow or restart from the reference specification.
- Preserve the exact frozen input lineage. Do not read any sibling run, prior experiment result, Debug Agent knowledge base, or Regenerate candidate.
{budget}{resume_directive}
{contract}
"""
        return f"""Execute one formal CANNBot Debug baseline run for `{operator}`.

Required method:
1. Invoke `/ascendc-precision-debug` and follow that public CANNBot skill as the debugging method.
2. Start from the existing failed implementation under `./task/` and the frozen initial evidence in `./input_evidence/forensics_report_0.json`.
3. Use `./spec/` only for the operator contract and test cases. Diagnose and repair the existing implementation in place.
4. Run real build and NPU tests, iterating within this single session until success or the fixed budget is exhausted.

Fairness and purity constraints:
- This is Debug, not Regenerate. Do not delete the failed implementation and create a new operator from scratch; do not invoke an operator-generation workflow, agent, or skill.
- Work only inside this trial cwd. Do not read sibling trials, historical Debug outcomes, Regenerate outputs, Debug Agent code/knowledge bases, or `/workspace/debug_agent`.
- Do not substitute the reference model, PyTorch, NumPy, CPU fallback, or wrapper-side computation for the AscendC kernel.
- Keep a concise `./debug_summary.md` recording diagnosis, edits, tests, and the final honest verdict.

{budget}
{contract}
"""
    contract = deliverable_contract()
    if resume_session:
        model_switch = ""
        resume_directive = ""
        if record.get("model_switch_from"):
            model_switch = (
                f"- The user explicitly switched the execution model from "
                f"`{record['model_switch_from']}` to `{record.get('model_expected')}`. "
                "Continue the same trial and preserve usable implementation evidence already in the workspace.\n"
            )
        if record.get("resume_directive"):
            resume_directive = f"- Trial-specific resume directive: {record['resume_directive']}\n"
        return f"""Continue the same formal CANNBot Regenerate trial for `{operator}` after an infrastructure/provider/deliverable interruption.

- Resume from the implementation, build state, test evidence, and conversation already present in this trial workspace.
- Do not restart generation from scratch when usable work already exists.
- Continue to obey the original trial prompt, isolation boundary, direct AscendC requirement, and the exact deliverable contract below.
{model_switch}{resume_directive}- A provider or model switch does not authorize reading any path outside this trial cwd or using independent debug skills.
- Inspect the current files, continue build/test/self-correction, and finish the same candidate honestly.

{contract}
"""
    return f"""You are executing one formal CANNBot Regenerate trial for `{operator}`.

Authorized experiment scope:
- Start from the original specification in `./spec/model.py`, `./spec/model.json` when present, and the case JSONL files under `./spec/`.
- Work only with files inside this trial cwd. It is a fresh workspace containing no prior candidate implementation or prior trial context.
- Use the ops-direct-invoke CANNBot workflow and the installed generation skills. The project skill allowlist intentionally omits the three independent diagnostic skill packages.
- Natural self-correction inside this session is allowed: build, test, edit your own freshly generated code, and retry within the session budget.

Implementation requirements:
1. Implement the complete operator semantics and every input group in `./spec/`, including dtypes, shapes, broadcasting, attributes, constructor arguments, and multi-output behavior.
2. Build a real AscendC direct-invoke operator under `./operators/<operator_slug>/`; run its tests on the NPU and fix your own implementation until your final verdict or budget exhaustion.
3. The wrapper's only computation call must be the registered custom NPU operator; the operator semantics execute in the AscendC kernel.
4. In addition to the normal CANNBot deliverables, create `./candidate_adapter/model_new_ascendc.py`. It must define `ModelNew` with constructor and forward signatures compatible with `spec/model.py`, load the generated shared library from a sibling directory named `operator_project` relative to its eventual file location, and call only the registered custom NPU op for computation.
5. Write the generated operator directory relative path (for example `operators/add_custom`) as the sole line of `./candidate_adapter/operator_dir.txt`.
6. The adapter must be self-contained for the external evaluator: after it is copied to `candidate_task/model_new_ascendc.py` and the generated operator is exposed at `candidate_task/operator_project`, importing it must load the library and run all original cases.

{contract}
Before finishing, verify that the complete external evaluator contract and normal review evidence exist. Report failures honestly; do not substitute the reference implementation.
"""


def snapshot_provider_abort(
    record: dict[str, Any],
    result: dict[str, Any],
    retry_count: int,
    experiment_root: Path,
) -> Path:
    """Preserve an immutable, container-invisible checkpoint before resuming.

    The live cwd remains in place for continuation.  The snapshot is stored
    outside that cwd, so the model cannot browse previous provider metadata or
    mistake checkpoint copies for additional trial inputs.
    """

    safe_run_id = "-".join(
        part for part in "".join(
            character.lower() if character.isalnum() else "-"
            for character in str(record["run_id"])
        ).split("-") if part
    )
    destination = experiment_root / "provider_abort_snapshots" / safe_run_id / f"attempt_{retry_count:03d}"
    if destination.is_dir():
        return destination
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, mode=0o700)

    cwd = Path(record["cwd"])
    selected = (
        "evidence",
        "validation",
        "task",
        "input_evidence",
        "frozen_input",
        "candidate_adapter",
        "candidate_task",
        "operators",
        "npu_watch",
        "regenerate_prompt.md",
        "run_record.json",
    )
    copied: list[str] = []
    for name in selected:
        source = cwd / name
        if not source.exists() and not source.is_symlink():
            continue
        target = temporary / "workspace" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir() and not source.is_symlink():
            shutil.copytree(source, target, symlinks=True)
        else:
            shutil.copy2(source, target, follow_symlinks=False)
        copied.append(name)

    project_journals = cwd / ".home" / ".claude" / "projects"
    if project_journals.is_dir():
        shutil.copytree(project_journals, temporary / "journal_projects", symlinks=True)
        copied.append(".home/.claude/projects")

    atomic_json(
        temporary / "checkpoint.json",
        {
            "schema_version": 1,
            "created_at": now(),
            "run_id": record["run_id"],
            "operator": record.get("operator"),
            "trial": record.get("trial"),
            "expected_session_id": record.get("expected_session_id"),
            "provider_slot": result.get("provider_slot"),
            "retry_count": retry_count,
            "transport_status": result.get("transport_status"),
            "provider_aborted": result.get("provider_aborted"),
            "copied": copied,
            "resume_policy": "same_session_same_workspace_next_provider",
        },
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(destination)
    for path in sorted(destination.rglob("*"), reverse=True):
        if path.is_symlink():
            continue
        path.chmod(0o500 if path.is_dir() else 0o400)
    destination.chmod(0o500)
    return destination


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _discover_native_sources(operator_project: Path) -> tuple[list[Path], str]:
    """Find generated sources without confusing build/test copies for code.

    Version 1 of the materializer only searched the three standard directories.
    Version 2 preserves that as the required layout for new work, but accepts
    legacy layouts for already-generated workspaces so a harness-only interface
    mismatch does not become an artificial model failure.
    """

    sources: list[Path] = []
    standard_count = 0
    legacy_count = 0
    for source in operator_project.rglob("*"):
        if not source.is_file() or source.suffix.lower() not in NATIVE_SOURCE_SUFFIXES:
            continue
        relative = source.relative_to(operator_project)
        if any(part in IGNORED_SOURCE_DIRS for part in relative.parts[:-1]):
            continue
        if not _is_within(source, operator_project):
            continue
        sources.append(source)
        if relative.parts and relative.parts[0] in STANDARD_SOURCE_DIRS:
            standard_count += 1
        else:
            legacy_count += 1
    sources.sort(key=lambda path: str(path.relative_to(operator_project)))
    if standard_count and legacy_count:
        layout = "mixed_compat"
    elif standard_count:
        layout = "standard"
    elif legacy_count:
        layout = "legacy_compat"
    else:
        layout = "missing"
    return sources, layout


def materialize_candidate(cwd: Path) -> tuple[bool, str]:
    adapter = cwd / "candidate_adapter" / "model_new_ascendc.py"
    pointer = cwd / "candidate_adapter" / "operator_dir.txt"
    target = cwd / "candidate_task"
    target.mkdir(parents=True, exist_ok=True)
    for path in (cwd / "spec").iterdir():
        if path.is_file() and (path.name in {"model.py", "model.json"} or path.name.endswith((".json", ".json.bak", ".json.full"))):
            if path.name != "spec_manifest.json":
                shutil.copy2(path, target / path.name)
    if not adapter.is_file() or not pointer.is_file():
        return False, "candidate_adapter missing"
    relative = pointer.read_text().strip()
    operator_project = (cwd / relative).resolve()
    try:
        operator_project.relative_to(cwd.resolve())
    except ValueError:
        return False, f"operator_dir escapes cwd: {relative}"
    if not operator_project.is_dir():
        return False, f"operator project missing: {relative}"
    shutil.copy2(adapter, target / "model_new_ascendc.py")
    link = target / "operator_project"
    if link.exists() or link.is_symlink():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.symlink_to(operator_project, target_is_directory=True)
    kernel_dir = target / "kernel"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir()
    sources, source_layout = _discover_native_sources(operator_project)
    for source in sources:
        relative_source = source.relative_to(operator_project)
        destination = kernel_dir / relative_source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    libraries = sorted(
        (
            library
            for library in operator_project.rglob("*.so")
            if library.is_file() and _is_within(library, operator_project)
        ),
        key=lambda path: str(path.relative_to(operator_project)),
    )
    if not libraries:
        return False, "generated shared library missing"
    if not sources:
        return False, "generated kernel/extension source missing"
    # Contract-v2 candidates must be self-contained for the external evaluator.
    # Keep the operator_project link for provenance/debugging, but also copy each
    # declared shared library to its operator-relative path under candidate_task.
    # Adapters commonly resolve ``kernel/build`` relative to
    # candidate_task/model_new_ascendc.py; a manifest-only library entry leaves
    # those imports broken after isolated staging.
    for library in libraries:
        relative_library = library.relative_to(operator_project)
        destination = target / relative_library
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(library, destination)
    atomic_json(
        target / "materialization_manifest.json",
        {
            "schema_version": DELIVERABLE_CONTRACT_VERSION,
            "materialized_at": now(),
            "operator_dir": relative,
            "source_layout": source_layout,
            "sources": [str(source.relative_to(operator_project)) for source in sources],
            "libraries": [str(library.relative_to(operator_project)) for library in libraries],
            "legacy_layout_accepted_for_existing_workspace": source_layout != "standard",
        },
    )
    return True, (
        f"contract_v={DELIVERABLE_CONTRACT_VERSION} operator={relative} "
        f"source_layout={source_layout} libraries={len(libraries)} sources={len(sources)}"
    )


def clear_run_artifacts(cwd: Path) -> None:
    """Permanently remove every artifact from an invalid operational attempt."""

    for name in (
        "evidence",
        "validation",
        ".home",
        "npu_watch",
        "candidate_adapter",
        "candidate_task",
        "operators",
        "regenerate_prompt.md",
        "run_record.json",
    ):
        path = cwd / name
        if path.is_dir() and not path.is_symlink():
            if name in {"evidence", "validation", ".home"}:
                # Preserve bind-mounted directory inodes while clearing their
                # contents.  This also avoids transient SELinux mount issues.
                for child in path.iterdir():
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            else:
                shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()


def classify_provider_abort(final_data: Any, diagnostic_text: str = "") -> str | None:
    final_is_error = isinstance(final_data, dict) and final_data.get("is_error") is True
    text = diagnostic_text.lower()
    if isinstance(final_data, dict):
        text += "\n" + json.dumps(final_data, ensure_ascii=False).lower()
    if not final_is_error and not diagnostic_text:
        return None
    # A real generation response is stronger evidence than the read-only
    # usage endpoint.  Kimi can report remaining weekly percentage while the
    # actual coding request is blocked by a billing-cycle limit.
    if any(
        marker in text
        for marker in (
            "reached your usage limit",
            "usage limit for this billing cycle",
            "purchase extra usage",
            "insufficient quota",
            "quota exhausted",
            "quota has been exhausted",
        )
    ):
        return "explicit_quota_exhausted"
    if any(marker in text for marker in ("authentication_error", "invalid api key", "http 401")):
        return "authentication_failed"
    if any(marker in text for marker in ("rate limit", "http 429")):
        return "rate_limited"
    if any(marker in text for marker in ("overloaded", "usage policy")):
        return "transient_provider_abort"
    if (isinstance(final_data, dict) and final_data.get("api_error_status") is not None) or any(
        marker in text for marker in ("quota", "api error", "http 403")
    ):
        return "provider_api_error"
    return None


def is_provider_abort(final_data: Any, diagnostic_text: str = "") -> bool:
    return classify_provider_abort(final_data, diagnostic_text) is not None


def classify_trial_terminal(
    final_data: Any, *, provider_aborted: bool, candidate_ok: bool
) -> tuple[str, bool]:
    """Apply the completed gate without conflating transport and output quality."""

    model_reported_success = isinstance(final_data, dict) and final_data.get("is_error") is not True
    deliverable_repair_required = model_reported_success and not candidate_ok
    if provider_aborted:
        return "provider_interrupted", deliverable_repair_required
    if deliverable_repair_required:
        return "deliverable_interrupted", True
    return ("completed" if model_reported_success else "failed"), False


def provider_retry_allowed(retry_count: int, retry_limit: int) -> bool:
    """Return whether a provider-only interruption remains resumable.

    A non-positive limit is intentionally unlimited: quota windows may require
    more resumptions than any fixed trial-local count, and each continuation is
    protected by a readonly checkpoint plus the original session id.
    """
    return retry_limit <= 0 or retry_count <= retry_limit


def run_one(
    record: dict[str, Any],
    worker_index: int,
    provider_env: Path,
    npu: str,
    args: argparse.Namespace,
    *,
    clean_start: bool = False,
    resume_session: bool = False,
    attempt_index: int = 1,
) -> dict[str, Any]:
    cwd = Path(record["cwd"])
    trial_uid = int(record["container_uid"])
    trial_gid = int(record.get("container_gid", trial_uid))
    if args.force or clean_start:
        # An infrastructure retry must start with no journal, implementation,
        # build log, or context from the superseded run.
        clear_run_artifacts(cwd)
    os.chown(cwd, trial_uid, trial_gid)
    cwd.chmod(0o700)
    evidence = cwd / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    os.chown(evidence, trial_uid, trial_gid)
    evidence.chmod(0o700)
    run_record_path = Path(record["run_record_path"])
    existing = load(run_record_path) if run_record_path.is_file() else {}
    task_turns_before = int(existing.get("task_turns_consumed", 0) or 0)
    task_active_seconds_before = float(
        existing.get("task_active_seconds", 0.0) or 0.0
    )
    model_invocations_before = int(
        existing.get("model_invocation_count", 0) or 0
    )
    invocation_max_turns = args.max_turns
    invocation_timeout = args.timeout
    if args.arm == "cannbot-debug":
        if args.max_task_turns is not None:
            remaining_turns = int(args.max_task_turns) - task_turns_before
            if remaining_turns <= 0:
                exhausted = {
                    **record,
                    **existing,
                    "status": "failed",
                    "ended_at": now(),
                    "error_type": "task_turn_budget_exhausted",
                    "error_scope": "model_budget",
                    "task_turns_consumed": task_turns_before,
                    "soft_task_turns": args.soft_task_turns,
                    "max_task_turns": args.max_task_turns,
                }
                atomic_json(run_record_path, exhausted)
                return {"run_id": record["run_id"], **exhausted}
            invocation_max_turns = min(invocation_max_turns, remaining_turns)
        remaining_seconds = int(args.timeout - task_active_seconds_before)
        if remaining_seconds <= 0:
            exhausted = {
                **record,
                **existing,
                "status": "failed",
                "ended_at": now(),
                "error_type": "task_wall_timeout",
                "error_scope": "model_budget",
                "task_active_seconds": task_active_seconds_before,
                "timeout_seconds": args.timeout,
            }
            atomic_json(run_record_path, exhausted)
            return {"run_id": record["run_id"], **exhausted}
        invocation_timeout = remaining_seconds
    final_path = Path(record["final_result_path"])
    if final_path.is_file() and final_path.stat().st_size > 0 and not args.force and not resume_session:
        return {"run_id": record["run_id"], "status": "skipped_existing"}

    home = cwd / ".home"
    home.mkdir(parents=True, exist_ok=True)
    os.chown(home, trial_uid, trial_gid)
    home.chmod(0o700)
    plugin = Path(record.get("plugin_runtime") or args.experiment_root / "runtime" / "ops-direct-invoke-step3-sanitized")
    devkit = Path(
        record.get("devkit_runtime")
        or args.experiment_root / "vendor" / "asc-devkit"
    )
    if not plugin.is_dir() or not devkit.is_dir():
        raise RuntimeError(f"missing isolation input: plugin={plugin.is_dir()} devkit={devkit.is_dir()}")
    debug_log = evidence / "claude_debug.log"
    stderr_path = evidence / "claude_stderr.log"
    private_provider_env = home / ".provider.env"
    prompt_path = cwd / "regenerate_prompt.md"
    if resume_session:
        main_journal = Path(record["journal_main_path"])
        if not main_journal.is_file():
            raise RuntimeError(f"resume journal missing: {main_journal}")
        # The immutable checkpoint retains the previous invocation result and
        # diagnostics. Clear only per-invocation sinks; implementation and
        # conversation state must remain untouched.
        for transient in (final_path, debug_log, stderr_path, prompt_path):
            transient.unlink(missing_ok=True)
    container_name = start_trial_container(
        record=record,
        cwd=cwd,
        home=home,
        plugin=plugin,
        devkit=devkit,
        npu=npu,
        image=args.container_image,
    )
    proc: subprocess.CompletedProcess[str] | None = None
    isolation_report: dict[str, Any] = {}
    try:
        isolation_report = verify_trial_container_isolation(
            container=container_name,
            cwd=cwd,
            experiment_root=args.experiment_root,
            plugin=plugin,
            devkit=devkit,
            uid=trial_uid,
            gid=trial_gid,
        )
        isolation_path = evidence / "isolation_preflight.json"
        atomic_json(isolation_path, isolation_report)
        os.chown(isolation_path, trial_uid, trial_gid)
        isolation_path.chmod(0o600)

        # Minimal onboarding state; no project history or conversation context is copied.
        claude_state = home / ".claude.json"
        if not claude_state.exists():
            write_as_trial_user(
                container=container_name,
                cwd=cwd,
                uid=trial_uid,
                gid=trial_gid,
                path=claude_state,
                content='{"hasCompletedOnboarding":true}\n',
                mode=0o600,
            )
        write_as_trial_user(
            container=container_name,
            cwd=cwd,
            uid=trial_uid,
            gid=trial_gid,
            path=private_provider_env,
            content=provider_env.read_text(),
            mode=0o400,
        )
        write_as_trial_user(
            container=container_name,
            cwd=cwd,
            uid=trial_uid,
            gid=trial_gid,
            path=prompt_path,
            content=prompt_for(record, resume_session=resume_session),
            mode=0o400,
        )
        attempt_started_at = now()
        status = {
            # Keep the complete pre-registered manifest row in the live record.
            # This matters after --force, which intentionally removes the prior
            # run_record.json together with every other superseded-run artifact.
            **record,
            **existing,
            "status": "running",
            "started_at": existing.get("started_at") or attempt_started_at,
            "attempt_started_at": attempt_started_at,
            "attempt_index": attempt_index,
            "resumed_session": resume_session,
            "resume_count": max(int(existing.get("resume_count", 0) or 0), attempt_index - 1),
            "worker": worker_index,
            "npu": npu,
            "provider_slot": provider_env.stem,
            "credential_name": record.get("credential_name") or existing.get("credential_name"),
            "timeout_seconds": args.timeout,
            "invocation_timeout_seconds": invocation_timeout,
            "invocation_max_turns": invocation_max_turns,
            "task_turns_before": task_turns_before,
            "task_active_seconds_before": task_active_seconds_before,
            "model_invocations_before": model_invocations_before,
            "soft_task_turns": args.soft_task_turns,
            "max_task_turns": args.max_task_turns,
            "max_internal_attempts": args.max_internal_attempts,
            "isolation_mode": "ephemeral_trial_container_allowlist",
            "isolation_preflight_passed": True,
            "trial_container_name": container_name,
            "trial_container_image": args.container_image,
        }
        if existing.get("status") in {
            "provider_interrupted",
            "deliverable_interrupted",
            "turn_budget_continuation",
        }:
            history = list(existing.get("interruption_history") or [])
            history.append(
                {
                    "status": existing.get("status"),
                    "error_type": existing.get("error_type"),
                    "error_scope": existing.get("error_scope"),
                    "attempt_index": existing.get("attempt_index"),
                    "recorded_at": existing.get("attempt_ended_at")
                    or existing.get("ended_at")
                    or now(),
                    "checkpoint": existing.get("provider_checkpoints"),
                    "invalid_attempt_excluded": existing.get(
                        "invalid_attempt_excluded", False
                    ),
                }
            )
            status["interruption_history"] = history
        for stale_field in (
            "ended_at",
            "error_type",
            "error_scope",
            "include_in_success_denominator",
            "resume_available",
            "resume_policy",
            "provider_abort_reason",
            "provider_error_reset_at",
            "provider_error_quota_windows",
        ):
            status.pop(stale_field, None)
        atomic_json(run_record_path, status)

        session_argument = (
            f"--resume {shlex.quote(record['expected_session_id'])}"
            if resume_session
            else f"--session-id {shlex.quote(record['expected_session_id'])}"
        )
        inner = f"""set -o pipefail
source .home/.provider.env
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then source /usr/local/Ascend/ascend-toolkit/set_env.sh; fi
export ASC_DEVKIT_DIR={shlex.quote(str(devkit))}
export HOME={shlex.quote(str(home))}
prompt=$(cat regenerate_prompt.md)
claude -p --dangerously-skip-permissions \\
  --plugin-dir {shlex.quote(str(plugin))} \\
  --setting-sources project \\
  --model {shlex.quote(args.model)} \\
  --effort max \\
  --max-turns {invocation_max_turns} \\
  {session_argument} \\
  --output-format json \\
  --debug-file evidence/claude_debug.log \\
  "$prompt"
"""
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=60",
            str(invocation_timeout),
            "docker",
            "exec",
            "-u",
            f"{trial_uid}:{trial_gid}",
            "-e",
            f"HOME={home}",
            "-e",
            f"ASCEND_RT_VISIBLE_DEVICES={npu}",
            "-e",
            "USER=cannbot",
            "-e",
            "LOGNAME=cannbot",
            "-e",
            "TORCHINDUCTOR_CACHE_DIR=/tmp/cannbot_torchinductor",
            "-w",
            str(cwd),
            container_name,
            "/bin/bash",
            "-c",
            inner,
        ]
        invocation_started = time.monotonic()
        with final_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(command, stdout=stdout, stderr=stderr, text=True)
        invocation_active_seconds = max(0.0, time.monotonic() - invocation_started)
    finally:
        private_provider_env.unlink(missing_ok=True)
        stop_trial_container(container_name)
    if proc is None:
        raise RuntimeError("trial container exited before Claude Code launch")
    final_data = None
    if final_path.stat().st_size:
        try:
            final_data = json.loads(final_path.read_text())
        except json.JSONDecodeError:
            pass
    candidate_ok, candidate_note = materialize_candidate(cwd)
    diagnostics = ""
    if proc.returncode != 0:
        for path in (stderr_path, debug_log):
            if path.is_file():
                diagnostics += path.read_text(errors="replace")[-200_000:]
    access_attempts = audit_isolation_access_attempts(home, cwd, args.experiment_root)
    postflight = {
        "audited_at": now(),
        "os_allowlist_preflight_passed": True,
        "cross_arm_access_attempted": bool(access_attempts),
        "cross_arm_access_succeeded": False,
        "attempts": access_attempts,
    }
    postflight_path = evidence / "isolation_postflight.json"
    atomic_json(postflight_path, postflight)
    os.chown(postflight_path, trial_uid, trial_gid)
    postflight_path.chmod(0o600)
    timed_out = proc.returncode in {124, 137}
    provider_abort_reason = classify_provider_abort(final_data, diagnostics)
    # The wall-clock guard applies to one attempt, not to the trial as a
    # whole.  A hard attempt timeout must preserve the same session/workspace
    # and re-enter the infrastructure retry path instead of becoming a model
    # generation failure merely because final stdout was not emitted yet.
    task_wall_timeout = bool(args.arm == "cannbot-debug" and timed_out)
    if timed_out and provider_abort_reason is None and not task_wall_timeout:
        provider_abort_reason = "attempt_timeout"
    provider_aborted = provider_abort_reason is not None
    model_invocation_count = model_invocations_before + (
        0 if provider_aborted else 1
    )
    terminal_status, deliverable_repair_required = classify_trial_terminal(
        final_data,
        provider_aborted=provider_aborted,
        candidate_ok=candidate_ok,
    )
    result_subtype = final_data.get("subtype") if isinstance(final_data, dict) else None
    reported_turns = (
        int(final_data.get("num_turns", 0) or 0)
        if isinstance(final_data, dict)
        else 0
    )
    turns_this_invocation = (
        min(invocation_max_turns, reported_turns)
        if result_subtype == "error_max_turns"
        else reported_turns
    )
    task_turns_consumed = task_turns_before + turns_this_invocation
    task_active_seconds = task_active_seconds_before + invocation_active_seconds
    status.update(
        {
            "status": terminal_status,
            "attempt_ended_at": now(),
            "launcher_returncode": proc.returncode,
            "result_subtype": result_subtype,
            "result_is_error": final_data.get("is_error") if isinstance(final_data, dict) else None,
            "candidate_materialized": candidate_ok,
            "candidate_note": candidate_note,
            "deliverable_contract_version": DELIVERABLE_CONTRACT_VERSION,
            "deliverable_repair_required": deliverable_repair_required,
            "transport_status": "ok" if proc.returncode == 0 else ("disconnected_after_result" if final_data else "disconnected_before_result"),
            "provider_aborted": provider_aborted,
            "provider_abort_reason": provider_abort_reason,
            "timed_out": timed_out,
            "task_wall_timeout": task_wall_timeout,
            "turns_this_invocation": turns_this_invocation,
            "task_turns_consumed": task_turns_consumed,
            "task_active_seconds": round(task_active_seconds, 6),
            "model_invocation_count": model_invocation_count,
            "isolation_preflight": isolation_report,
            "isolation_postflight": postflight,
            "contamination_detected": False,
            "trial_container_removed": True,
        }
    )
    if task_wall_timeout:
        status.update(
            {
                "status": "failed",
                "error_type": "task_wall_timeout",
                "error_scope": "model_budget",
                "provider_aborted": False,
                "provider_abort_reason": None,
            }
        )
        terminal_status = "failed"
    elif (
        args.arm == "cannbot-debug"
        and result_subtype == "error_max_turns"
        and args.max_task_turns is not None
        and task_turns_consumed >= args.max_task_turns
    ):
        status.update(
            {
                "status": "failed",
                "error_type": "task_turn_budget_exhausted",
                "error_scope": "model_budget",
            }
        )
        terminal_status = "failed"
    if provider_abort_reason == "explicit_quota_exhausted":
        reset_at, reset_basis, reset_windows = explicit_quota_unavailable_until(
            {},
            time.time(),
            args.provider_quota_cooldown_seconds,
            diagnostics,
        )
        if reset_basis == "actual_request_explicit_reset":
            status["provider_error_reset_at"] = datetime.fromtimestamp(
                reset_at, timezone.utc
            ).isoformat()
            status["provider_error_quota_windows"] = reset_windows
    if terminal_status in {"completed", "failed"}:
        status["ended_at"] = status["attempt_ended_at"]
        status["include_in_success_denominator"] = True
    else:
        status.pop("ended_at", None)
    atomic_json(run_record_path, status)
    return {"run_id": record["run_id"], **status}


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=here / "run_manifest.json")
    parser.add_argument("--experiment-root", type=Path, default=Path("/workspace/regenerate_experiment"))
    parser.add_argument("--key-config", type=Path, default=Path("/workspace/regenerate_experiment/.secrets/provider_config.json"))
    parser.add_argument("--container-image", default="cannbot-regenerate-runtime:release")
    parser.add_argument("--model", default="k3")
    parser.add_argument(
        "--arm",
        choices=("regenerate", "cannbot-debug"),
        default="regenerate",
        help="manifest arm to execute; cannbot-debug uses the frozen failed implementation and precision-debug skill",
    )
    parser.add_argument("--npus", default="1,2,4,5,3")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=240)
    parser.add_argument(
        "--soft-task-turns",
        type=int,
        default=None,
        help="task-level soft turn boundary; used by cannbot-debug continuations",
    )
    parser.add_argument(
        "--max-task-turns",
        type=int,
        default=None,
        help="task-level hard turn budget; cannbot-debug resumes the same session in max-turns chunks",
    )
    parser.add_argument(
        "--max-internal-attempts",
        type=int,
        default=5,
        help="maximum valid same-session Agent invocations for cannbot-debug; provider/infrastructure-invalid calls do not consume it",
    )
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--only", action="append", default=[], help="run-id or operator substring filter")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--provider-offset", type=int, default=0)
    parser.add_argument("--preferred-provider", default="provider-slot-a")
    parser.add_argument(
        "--fallback-provider",
        action="append",
        default=[],
        help=(
            "provider admitted only when no non-fallback provider slot is "
            "available; may be repeated"
        ),
    )
    parser.add_argument("--slots-per-key", type=int, default=1)
    parser.add_argument(
        "--exclude-provider",
        action="append",
        default=[],
        help="provider name explicitly excluded by the operator; may be repeated",
    )
    parser.add_argument("--min-provider-remaining", type=float, default=20.0)
    parser.add_argument("--quota-timeout", type=int, default=20)
    parser.add_argument(
        "--provider-retries",
        type=int,
        default=0,
        help="maximum resumable provider interruptions per trial; 0 means unlimited",
    )
    parser.add_argument(
        "--deliverable-retries",
        type=int,
        default=3,
        help="maximum targeted same-session repairs after a nominal success misses the evaluator contract",
    )
    parser.add_argument(
        "--repair-invalid-completed",
        action="store_true",
        help="reconcile or resume completed rows rejected by the historical materializer contract",
    )
    parser.add_argument(
        "--provider-abort-cooldown-seconds",
        type=int,
        default=300,
        help=(
            "temporary cooldown after a provider abort; the provider is "
            "re-probed and automatically re-admitted when live quota/auth is healthy"
        ),
    )
    parser.add_argument(
        "--provider-quota-cooldown-seconds",
        type=int,
        default=21600,
        help="fallback unavailability period for an explicit quota/auth failure when no reset time is known",
    )
    parser.add_argument("--quota-retry-wait", type=int, default=300)
    parser.add_argument("--provider-refresh-seconds", type=int, default=300)
    parser.add_argument("--provider-env-name", default="regenerate")
    parser.add_argument(
        "--queue-order",
        choices=("operator_then_trial", "trial_then_operator", "reverse_trial_then_operator"),
        default="operator_then_trial",
        help=(
            "pending-run ordering; trial_then_operator runs every trial1 before trial2 and trial3, "
            "while reverse_trial_then_operator is the strict reverse manifest order (trial3 to trial1)"
        ),
    )
    parser.add_argument(
        "--max-quota-waits",
        type=int,
        default=0,
        help="maximum empty-pool waits; 0 means keep waiting for a quota reset",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    manifest = load(args.manifest)
    records = [record for record in manifest["runs"] if record.get("arm") == args.arm]
    if args.only:
        records = [record for record in records if any(term in record["run_id"] or term in record["operator"] for term in args.only)]
    if args.queue_order in {"trial_then_operator", "reverse_trial_then_operator"}:
        reverse = args.queue_order == "reverse_trial_then_operator"
        records = [
            record
            for _, record in sorted(
                enumerate(records),
                key=lambda pair: (
                    (-1 if reverse else 1)
                    * (int(pair[1].get("trial")) if isinstance(pair[1].get("trial"), int) else 10_000),
                    (-1 if reverse else 1) * pair[0],
                ),
            )
        ]
    if args.limit:
        records = records[: args.limit]
    npus = [item.strip() for item in args.npus.split(",") if item.strip()]
    if not npus or args.concurrency < 1:
        raise SystemExit("no worker slots available")

    # Existing terminal sessions are never re-opened unless --force is used,
    # except when the operator explicitly enables the one-time repair of rows
    # rejected by the historical materializer mismatch. Provider and
    # deliverable interruptions resume the same session/workspace. An
    # unclassified stale running record still receives the conservative clean-
    # retry treatment.
    pending: collections.deque[dict[str, Any]] = collections.deque()
    final_by_run: dict[str, dict[str, Any]] = {}
    stale_incomplete: set[str] = set()
    continuation_runs: set[str] = set()
    clean_retry_runs: set[str] = set()
    retries: collections.Counter[str] = collections.Counter()
    contract_repairs: collections.Counter[str] = collections.Counter()
    attempt_numbers: collections.Counter[str] = collections.Counter()
    for record in records:
        final_path = Path(record["final_result_path"])
        run_record_path = Path(record["run_record_path"])
        run_record = load(run_record_path) if run_record_path.is_file() else {}
        run_status = run_record.get("status") if isinstance(run_record, dict) else None
        attempt_numbers[record["run_id"]] = int(run_record.get("attempt_index", 0) or 0)
        if (
            args.repair_invalid_completed
            and not args.force
            and run_status == "completed"
            and run_record.get("candidate_materialized") is False
        ):
            candidate_ok, candidate_note = materialize_candidate(Path(record["cwd"]))
            if candidate_ok:
                reconciled = dict(run_record)
                reconciled.update(
                    {
                        "candidate_materialized": True,
                        "candidate_note": candidate_note,
                        "deliverable_contract_version": DELIVERABLE_CONTRACT_VERSION,
                        "deliverable_reconciled_at": now(),
                        "deliverable_repair_required": False,
                    }
                )
                atomic_json(run_record_path, reconciled)
                final_by_run[record["run_id"]] = {
                    "run_id": record["run_id"],
                    "status": "reconciled_existing_candidate",
                    "candidate_note": candidate_note,
                }
            elif Path(record["journal_main_path"]).is_file():
                repair_record = {
                    **record,
                    "resume_directive": contract_repair_directive(
                        candidate_note, arm=str(record.get("arm") or args.arm)
                    ),
                }
                interrupted = dict(run_record)
                interrupted.update(
                    {
                        "status": "deliverable_interrupted",
                        "candidate_materialized": False,
                        "candidate_note": candidate_note,
                        "deliverable_contract_version": DELIVERABLE_CONTRACT_VERSION,
                        "deliverable_repair_required": True,
                        "deliverable_original_terminal_status": "completed",
                    }
                )
                interrupted.pop("ended_at", None)
                atomic_json(run_record_path, interrupted)
                pending.append(repair_record)
                continuation_runs.add(record["run_id"])
                contract_repairs[record["run_id"]] = int(
                    run_record.get("deliverable_repair_count", 0) or 0
                )
            else:
                final_by_run[record["run_id"]] = {
                    "run_id": record["run_id"],
                    "status": "contract_repair_blocked_missing_journal",
                    "candidate_note": candidate_note,
                }
        elif not args.force and run_status == "provider_interrupted":
            pending.append(record)
            continuation_runs.add(record["run_id"])
            retries[record["run_id"]] = int(run_record.get("provider_retry_count", 0) or 0)
        elif not args.force and run_status in DELIVERABLE_RESUMABLE_STATUSES:
            pending.append(
                {
                    **record,
                    "resume_directive": contract_repair_directive(
                        run_record.get("candidate_note"),
                        arm=str(record.get("arm") or args.arm),
                    ),
                }
            )
            continuation_runs.add(record["run_id"])
            contract_repairs[record["run_id"]] = int(
                run_record.get("deliverable_repair_count", 0) or 0
            )
        elif final_path.is_file() and final_path.stat().st_size > 0 and not args.force:
            final_by_run[record["run_id"]] = {"run_id": record["run_id"], "status": "skipped_existing"}
        elif (
            not args.force
            and isinstance(run_record, dict)
            and run_record.get("status") in {"completed", "failed"}
        ):
            final_by_run[record["run_id"]] = {
                "run_id": record["run_id"],
                "status": "skipped_terminal_run_record",
                "terminal_status": run_record["status"],
            }
        else:
            pending.append(record)
            if run_status == "running":
                stale_incomplete.add(record["run_id"])
                clean_retry_runs.add(record["run_id"])

    empty_pool_waits = 0
    refresh_index = 0
    next_provider_refresh = 0.0
    idle_slots: list[ProviderSlot] = []
    # Explicit command-line exclusions are distinct from transient provider
    # aborts.  An occasional abort must not become a permanent blacklist: it
    # only starts a bounded cooldown, after which the live quota/auth probe is
    # authoritative again.
    excluded_providers: set[str] = set(args.exclude_provider)
    provider_runtime_state_path = args.experiment_root / "experiment_harness" / "provider_runtime_state.json"
    provider_cooldowns: dict[str, float] = {}
    provider_unavailable_reasons: dict[str, str] = {}
    provider_unavailable_bases: dict[str, dict[str, Any]] = {}
    provider_live_states: dict[str, dict[str, Any]] = {}
    if provider_runtime_state_path.is_file():
        try:
            stored_state = load(provider_runtime_state_path)
            for name, state in (stored_state.get("providers") or {}).items():
                until = float(state.get("unavailable_until_epoch", 0.0) or 0.0)
                if until > time.time():
                    provider_cooldowns[str(name)] = until
                    provider_unavailable_reasons[str(name)] = str(state.get("reason") or "unknown")
                    provider_unavailable_bases[str(name)] = dict(state.get("basis") or {})
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            # A damaged operational cache must not become an implicit key
            # blacklist.  The live probes below rebuild availability.
            provider_cooldowns.clear()
            provider_unavailable_reasons.clear()
            provider_unavailable_bases.clear()
    provider_abort_counts: collections.Counter[str] = collections.Counter()
    # Keep every configured physical card in the candidate queue while the
    # executor separately caps simultaneous workers at ``concurrency``.  This
    # lets a five-worker C2 run choose any currently-free card from an eight-
    # card host instead of silently restricting placement to the first five.
    available_npus: collections.deque[str] = collections.deque(npus)
    active: dict[Future[dict[str, Any]], tuple[dict[str, Any], ProviderSlot, str]] = {}
    max_workers = min(args.concurrency, len(npus))
    provider_root = args.experiment_root / ".provider_env" / args.provider_env_name

    def persist_provider_runtime_state() -> None:
        atomic_json(
            provider_runtime_state_path,
            {
                "schema_version": 1,
                "updated_at": now(),
                "providers": {
                    name: {
                        "reason": provider_unavailable_reasons.get(name, "unknown"),
                        "basis": provider_unavailable_bases.get(name, {}),
                        "unavailable_until_epoch": expires_at,
                        "unavailable_until": datetime.fromtimestamp(expires_at, timezone.utc).isoformat(),
                    }
                    for name, expires_at in sorted(provider_cooldowns.items())
                    if expires_at > time.time()
                },
            },
        )

    def refresh_provider_pool() -> tuple[bool, str | None]:
        nonlocal refresh_index, next_provider_refresh, idle_slots, provider_live_states
        refresh_dir = provider_root / f"refresh_{refresh_index:05d}"
        refresh_index += 1
        try:
            fresh = make_provider_envs(
                args.key_config,
                refresh_dir,
                args.model,
                args.slots_per_key,
                args.min_provider_remaining,
                args.quota_timeout,
            )
        except (RuntimeError, json.JSONDecodeError, OSError) as exc:
            next_provider_refresh = time.monotonic() + args.quota_retry_wait
            return False, str(exc)
        selection_path = refresh_dir / "provider_selection.json"
        if selection_path.is_file():
            selection = load(selection_path)
            provider_live_states = {
                str(row.get("name")): row
                for row in selection.get("quota", [])
                if isinstance(row, dict) and row.get("name")
            }
        active_counts = collections.Counter(slot.name for _, slot, _ in active.values())
        wall_now = time.time()
        state_changed = False
        for name, expires_at in list(provider_cooldowns.items()):
            if expires_at <= wall_now:
                del provider_cooldowns[name]
                provider_unavailable_reasons.pop(name, None)
                provider_unavailable_bases.pop(name, None)
                state_changed = True
        if state_changed:
            persist_provider_runtime_state()
        offset = (args.provider_offset + refresh_index - 1) % len(fresh)
        fresh = fresh[offset:] + fresh[:offset]
        admitted = collections.Counter()
        idle_slots = []
        for slot in fresh:
            if slot.name in excluded_providers:
                continue
            if provider_cooldowns.get(slot.name, 0.0) > wall_now:
                continue
            if active_counts[slot.name] + admitted[slot.name] >= args.slots_per_key:
                continue
            idle_slots.append(slot)
            admitted[slot.name] += 1
        if not idle_slots:
            next_provider_refresh = time.monotonic() + args.quota_retry_wait
            cooling = sorted(
                name
                for name, expires_at in provider_cooldowns.items()
                if expires_at > wall_now
            )
            return False, (
                "no live-healthy provider slot currently available"
                + (f"; cooling_down={cooling}" if cooling else "")
            )
        next_provider_refresh = time.monotonic() + max(10, args.provider_refresh_seconds)
        return True, None

    def take_idle_slot() -> ProviderSlot:
        """Use primary providers first; admit explicit fallbacks only as a tier."""

        active_names = {slot.name for _, slot, _ in active.values()}
        fallback_names = set(args.fallback_provider)
        primary_indices = [
            index
            for index, slot in enumerate(idle_slots)
            if slot.name not in fallback_names
        ]
        eligible_indices = primary_indices or list(range(len(idle_slots)))
        preferred_index = next(
            (
                index
                for index in eligible_indices
                if idle_slots[index].name == args.preferred_provider
            ),
            None,
        )
        if args.preferred_provider not in active_names and preferred_index is not None:
            return idle_slots.pop(preferred_index)
        diverse_index = next(
            (
                index
                for index in eligible_indices
                if idle_slots[index].name not in active_names
            ),
            None,
        )
        if diverse_index is not None:
            return idle_slots.pop(diverse_index)
        if preferred_index is not None:
            return idle_slots.pop(preferred_index)
        return idle_slots.pop(eligible_indices[0])

    def provider_unavailable_until(
        provider: str,
        reason: str,
        result: dict[str, Any] | None = None,
    ) -> float:
        """Choose a bounded retry time from current evidence, never history."""

        wall_now = time.time()
        if reason == "explicit_quota_exhausted":
            explicit_reset = (result or {}).get("provider_error_reset_at")
            if explicit_reset:
                try:
                    expires_at = datetime.fromisoformat(
                        str(explicit_reset).replace("Z", "+00:00")
                    ).timestamp()
                except ValueError:
                    expires_at = 0.0
                if expires_at > 0:
                    windows = list((result or {}).get("provider_error_quota_windows") or [])
                    provider_unavailable_bases[provider] = {
                        "policy": "actual_request_explicit_reset",
                        "exhausted_windows": windows,
                    }
                    return max(wall_now, expires_at)
            expires_at, basis, windows = explicit_quota_unavailable_until(
                provider_live_states.get(provider, {}),
                wall_now,
                args.provider_quota_cooldown_seconds,
            )
            provider_unavailable_bases[provider] = {
                "policy": basis,
                "exhausted_windows": windows,
            }
            return expires_at
        if reason == "authentication_failed":
            provider_unavailable_bases[provider] = {"policy": "authentication_failure_fallback"}
            return wall_now + max(0, args.provider_quota_cooldown_seconds)
        provider_unavailable_bases[provider] = {"policy": "transient_abort_cooldown"}
        return wall_now + max(0, args.provider_abort_cooldown_seconds)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while pending or active:
            spare_capacity = bool(pending and available_npus and len(active) < max_workers)
            if spare_capacity and (not idle_slots or time.monotonic() >= next_provider_refresh):
                refreshed, error = refresh_provider_pool()
                if refreshed:
                    empty_pool_waits = 0
                elif not active:
                    empty_pool_waits += 1
                    if args.max_quota_waits and empty_pool_waits > args.max_quota_waits:
                        while pending:
                            record = pending.popleft()
                            final_by_run[record["run_id"]] = {
                                "run_id": record["run_id"],
                                "status": "provider_pool_exhausted",
                                "error": error,
                            }
                        break
                    print(
                        json.dumps(
                            {
                                "status": "waiting_for_provider_quota",
                                "wait_seconds": args.quota_retry_wait,
                                "wait_count": empty_pool_waits,
                                "error": error,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    time.sleep(args.quota_retry_wait)
                    continue

            npu_wait_required = False
            while pending and available_npus and idle_slots and len(active) < max_workers:
                if args.arm == "cannbot-debug":
                    busy_npus = host_busy_npus()
                    if busy_npus is None:
                        npu_wait_required = True
                        print(
                            json.dumps(
                                {
                                    "status": "waiting_for_npu_probe",
                                    "configured_npus": list(available_npus),
                                    "reason": "npu-smi process inspection failed",
                                    "wait_seconds": 60,
                                }
                            ),
                            flush=True,
                        )
                        break
                    chosen_npu: str | None = None
                    for _ in range(len(available_npus)):
                        candidate_npu = available_npus.popleft()
                        if candidate_npu not in busy_npus:
                            chosen_npu = candidate_npu
                            break
                        available_npus.append(candidate_npu)
                    if chosen_npu is None:
                        npu_wait_required = True
                        print(
                            json.dumps(
                                {
                                    "status": "waiting_for_free_npu",
                                    "configured_npus": list(available_npus),
                                    "busy_npus": sorted(busy_npus),
                                    "wait_seconds": 60,
                                }
                            ),
                            flush=True,
                        )
                        break
                    npu = chosen_npu
                else:
                    npu = available_npus.popleft()
                record = pending.popleft()
                slot = take_idle_slot()
                run_id = record["run_id"]
                clean_start = run_id in clean_retry_runs or run_id in stale_incomplete
                resume_session = run_id in continuation_runs and not clean_start
                stale_incomplete.discard(run_id)
                clean_retry_runs.discard(run_id)
                continuation_runs.discard(run_id)
                record_for_attempt = {**record, "credential_name": slot.name}
                attempt_numbers[run_id] += 1
                attempt_index = attempt_numbers[run_id]
                future = pool.submit(
                    run_one,
                    record_for_attempt,
                    npus.index(npu),
                    slot.path,
                    npu,
                    args,
                    clean_start=clean_start,
                    resume_session=resume_session,
                    attempt_index=attempt_index,
                )
                active[future] = (record, slot, npu)
                print(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "status": "rolling_worker_started",
                            "provider": slot.name,
                            "npu": npu,
                            "resume_session": resume_session,
                            "attempt_index": attempt_index,
                            "active_workers": len(active),
                            "pending": len(pending),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            if not active:
                # Pending work with no active/eligible slot will refresh on the
                # next loop after the bounded quota wait above.
                if npu_wait_required:
                    time.sleep(60)
                continue

            timeout: float | None = None
            if pending and available_npus and len(active) < max_workers:
                timeout = max(0.1, next_provider_refresh - time.monotonic())
            done, _ = wait(active, timeout=timeout, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                record, slot, npu = active.pop(future)
                available_npus.append(npu)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "run_id": record["run_id"],
                        "status": "runner_exception",
                        "error": repr(exc),
                        "harness_error_type": (
                            "isolation_preflight"
                            if isinstance(exc, TrialIsolationPreflightError)
                            else "unknown_runner_exception"
                        ),
                    }
                with PRINT_LOCK:
                    print(json.dumps(result, ensure_ascii=False), flush=True)

                run_id = record["run_id"]
                provider_failure = result.get("provider_aborted") is True
                runner_failure = result.get("status") == "runner_exception"
                contract_failure = result.get("deliverable_repair_required") is True
                turn_budget_continuation = (
                    args.arm == "cannbot-debug"
                    and result.get("result_subtype") == "error_max_turns"
                    and args.max_task_turns is not None
                    and int(result.get("task_turns_consumed", 0) or 0)
                    < int(args.max_task_turns)
                )
                if provider_failure:
                    retries[run_id] += 1
                    checkpoint = snapshot_provider_abort(record, result, retries[run_id], args.experiment_root)
                    checkpoints = list(result.get("provider_checkpoints") or [])
                    if str(checkpoint) not in checkpoints:
                        checkpoints.append(str(checkpoint))
                    result.update(
                        {
                            "status": "provider_interrupted",
                            "provider_retry_count": retries[run_id],
                            "provider_checkpoints": checkpoints,
                            "resume_available": Path(record["journal_main_path"]).is_file(),
                            "resume_policy": "same_session_same_workspace_next_provider",
                        }
                    )
                    atomic_json(Path(record["run_record_path"]), result)
                    continuation_runs.add(run_id)
                    provider_abort_counts[slot.name] += 1
                    abort_reason = str(result.get("provider_abort_reason") or "transient_provider_abort")
                    provider_cooldowns[slot.name] = provider_unavailable_until(
                        slot.name, abort_reason, result
                    )
                    provider_unavailable_reasons[slot.name] = abort_reason
                    persist_provider_runtime_state()
                    idle_slots = [candidate for candidate in idle_slots if candidate.name != slot.name]
                    unavailable_seconds = max(0, round(provider_cooldowns[slot.name] - time.time()))
                    print(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "status": "provider_temporarily_unavailable",
                                "provider": slot.name,
                                "reason": abort_reason,
                                "basis": provider_unavailable_bases.get(slot.name),
                                "cooldown_seconds": unavailable_seconds,
                                "unavailable_until": datetime.fromtimestamp(
                                    provider_cooldowns[slot.name], timezone.utc
                                ).isoformat(),
                                "permanent_quarantine": False,
                            }
                        ),
                        flush=True,
                    )
                    if provider_retry_allowed(retries[run_id], args.provider_retries):
                        pending.appendleft(record)
                        print(
                            json.dumps(
                                {
                                    "run_id": run_id,
                                    "status": "provider_resume_scheduled",
                                    "retry": retries[run_id],
                                    "checkpoint": str(checkpoint),
                                }
                            ),
                            flush=True,
                        )
                    else:
                        exhausted = dict(result)
                        exhausted.update(
                            {
                                "status": "failed",
                                "ended_at": now(),
                                "error_type": "provider_retries_exhausted",
                                "error_scope": "provider",
                                "include_in_success_denominator": False,
                            }
                        )
                        atomic_json(Path(record["run_record_path"]), exhausted)
                        final_by_run[run_id] = {
                            "run_id": run_id,
                            "status": "provider_retries_exhausted",
                            "provider_retries": retries[run_id],
                        }
                elif turn_budget_continuation:
                    resume_available = Path(record["journal_main_path"]).is_file()
                    model_invocations = int(
                        result.get("model_invocation_count", 0) or 0
                    )
                    if (
                        resume_available
                        and model_invocations < args.max_internal_attempts
                    ):
                        task_turns = int(
                            result.get("task_turns_consumed", 0) or 0
                        )
                        result.update(
                            {
                                "status": "turn_budget_continuation",
                                "resume_available": True,
                                "resume_policy": (
                                    "same_session_same_workspace_remaining_task_turn_budget"
                                ),
                            }
                        )
                        result.pop("ended_at", None)
                        atomic_json(Path(record["run_record_path"]), result)
                        if (
                            args.soft_task_turns is not None
                            and task_turns >= args.soft_task_turns
                        ):
                            directive = (
                                "The task soft budget has been reached. Continue the "
                                "same debugging session using only the remaining hard "
                                "turn budget. Converge quickly: run the decisive NPU "
                                "checks, fix the existing candidate, and package the "
                                "required deliverable without regenerating the operator."
                            )
                        else:
                            directive = (
                                "Continue the same debugging session from its current "
                                "workspace and evidence. Do not restart or regenerate "
                                "the operator; keep diagnosing and repairing the existing "
                                "candidate within the remaining task turn budget."
                            )
                        continuation_runs.add(run_id)
                        pending.appendleft(
                            {**record, "resume_directive": directive}
                        )
                        print(
                            json.dumps(
                                {
                                    "run_id": run_id,
                                    "status": "turn_budget_resume_scheduled",
                                    "task_turns_consumed": task_turns,
                                    "remaining_task_turns": (
                                        int(args.max_task_turns) - task_turns
                                    ),
                                    "model_invocation_count": model_invocations,
                                }
                            ),
                            flush=True,
                        )
                    else:
                        failed = dict(result)
                        failed.update(
                            {
                                "status": "failed",
                                "ended_at": now(),
                                "error_type": (
                                    "max_internal_attempts_exhausted"
                                    if resume_available
                                    else "turn_budget_resume_missing_journal"
                                ),
                                "error_scope": "model_budget",
                                "include_in_success_denominator": True,
                            }
                        )
                        atomic_json(Path(record["run_record_path"]), failed)
                        final_by_run[run_id] = {
                            "run_id": run_id,
                            "status": failed["error_type"],
                        }
                elif contract_failure:
                    contract_repairs[run_id] += 1
                    resume_available = Path(record["journal_main_path"]).is_file()
                    model_invocations = int(
                        result.get("model_invocation_count", 0) or 0
                    )
                    result.update(
                        {
                            "status": "deliverable_interrupted",
                            "deliverable_repair_count": contract_repairs[run_id],
                            "resume_available": resume_available,
                            "resume_policy": "same_session_same_workspace_targeted_contract_repair",
                        }
                    )
                    atomic_json(Path(record["run_record_path"]), result)
                    if (
                        resume_available
                        and contract_repairs[run_id] <= args.deliverable_retries
                        and model_invocations < args.max_internal_attempts
                    ):
                        repair_record = {
                            **record,
                            "resume_directive": contract_repair_directive(
                                result.get("candidate_note"),
                                arm=str(record.get("arm") or args.arm),
                            ),
                        }
                        continuation_runs.add(run_id)
                        pending.appendleft(repair_record)
                        print(
                            json.dumps(
                                {
                                    "run_id": run_id,
                                    "status": "deliverable_repair_scheduled",
                                    "repair": contract_repairs[run_id],
                                    "candidate_note": result.get("candidate_note"),
                                }
                            ),
                            flush=True,
                        )
                    else:
                        failed = dict(result)
                        failed.update(
                            {
                                "status": "failed",
                                "ended_at": now(),
                                "error_type": "deliverable_contract_not_satisfied"
                                if (
                                    resume_available
                                    and model_invocations
                                    < args.max_internal_attempts
                                )
                                else (
                                    "max_internal_attempts_exhausted"
                                    if resume_available
                                    else "deliverable_repair_missing_journal"
                                ),
                                "error_scope": "model_deliverable",
                                "include_in_success_denominator": True,
                            }
                        )
                        atomic_json(Path(record["run_record_path"]), failed)
                        final_by_run[run_id] = {
                            "run_id": run_id,
                            "status": "deliverable_repair_exhausted",
                            "repairs": contract_repairs[run_id],
                        }
                elif runner_failure:
                    retries[run_id] += 1
                    if result.get("harness_error_type") == "isolation_preflight":
                        # Isolation failures are harness-control failures.  Do
                        # not erase the workspace and do not feed an unbounded
                        # clean-retry loop through --provider-retries 0.
                        run_record_path = Path(record["run_record_path"])
                        existing = (
                            load(run_record_path)
                            if run_record_path.is_file()
                            else {}
                        )
                        interrupted = {
                            **record,
                            **existing,
                            "status": "provider_interrupted",
                            "attempt_ended_at": now(),
                            "error_type": "harness_isolation_preflight_failed",
                            "error_scope": "harness_control",
                            "harness_error": result.get("error"),
                            "resume_available": Path(
                                record["journal_main_path"]
                            ).is_file(),
                            "resume_policy": "same_session_same_workspace_after_harness_repair",
                            "same_session_same_workspace": True,
                            "count_as_model_failure": False,
                            "count_as_valid_internal_attempt": False,
                            "include_in_success_denominator": False,
                        }
                        atomic_json(run_record_path, interrupted)
                        final_by_run[run_id] = {
                            "run_id": run_id,
                            "status": "harness_isolation_interrupted",
                        }
                        print(
                            json.dumps(
                                {
                                    "run_id": run_id,
                                    "status": "harness_isolation_interrupted",
                                    "workspace_preserved": True,
                                    "retry_scheduled": False,
                                }
                            ),
                            flush=True,
                        )
                        continue
                    # Unknown harness exceptions are not safe to resume as
                    # conversation checkpoints. Preserve the clean-retry
                    # behavior for this separate failure class.
                    clear_run_artifacts(Path(record["cwd"]))
                    clean_retry_runs.add(run_id)
                    if provider_retry_allowed(retries[run_id], args.provider_retries):
                        pending.appendleft(record)
                        print(
                            json.dumps(
                                {
                                    "run_id": run_id,
                                    "status": "clean_retry_scheduled",
                                    "retry": retries[run_id],
                                }
                            ),
                            flush=True,
                        )
                    else:
                        final_by_run[run_id] = result
                else:
                    final_by_run[run_id] = result
                # Re-query before reusing a just-freed key.  This both avoids
                # quota-overrun retries and admits a recovered fifth provider.
                idle_slots = []
                next_provider_refresh = 0.0

    results = [final_by_run[record["run_id"]] for record in records]
    accepted_result_statuses = {"completed", "skipped_existing", "reconciled_existing_candidate"}
    completed_results = sum(
        result.get("status") in accepted_result_statuses
        or (
            result.get("status") == "skipped_terminal_run_record"
            and result.get("terminal_status") == "completed"
        )
        for result in results
    )
    summary = {
        "generated_at": now(),
        "arm": args.arm,
        "scheduling_policy": "rolling_worker_pool",
        "queue_order": args.queue_order,
        "configured_concurrency": args.concurrency,
        "npu_order": npus,
        "preferred_provider": args.preferred_provider,
        "fallback_providers": sorted(set(args.fallback_provider)),
        "excluded_providers": sorted(excluded_providers),
        "provider_availability_policy": "live_quota_auth_plus_abort_cooldown",
        "provider_abort_cooldown_seconds": args.provider_abort_cooldown_seconds,
        "provider_quota_cooldown_seconds": args.provider_quota_cooldown_seconds,
        "provider_runtime_state_path": str(provider_runtime_state_path),
        "temporarily_unavailable_providers": {
            name: {
                "reason": provider_unavailable_reasons.get(name),
                "basis": provider_unavailable_bases.get(name),
                "unavailable_until": datetime.fromtimestamp(expires_at, timezone.utc).isoformat(),
            }
            for name, expires_at in sorted(provider_cooldowns.items())
            if expires_at > time.time()
        },
        "provider_abort_counts": dict(provider_abort_counts),
        "permanent_abort_quarantine": False,
        "deliverable_contract_version": DELIVERABLE_CONTRACT_VERSION,
        "repair_invalid_completed": args.repair_invalid_completed,
        "deliverable_retry_limit": args.deliverable_retries,
        "deliverable_repairs": dict(contract_repairs),
        "timeout_seconds": args.timeout,
        "per_invocation_max_turns": args.max_turns,
        "soft_task_turns": args.soft_task_turns,
        "max_task_turns": args.max_task_turns,
        "max_internal_attempts": args.max_internal_attempts,
        "isolation_mode": "ephemeral_trial_container_allowlist",
        "trial_container_image": args.container_image,
        "requested": len(records),
        "provider_retries": dict(retries),
        "completed": completed_results,
        "failed": len(results) - completed_results,
        "results": results,
    }
    summary_name = (
        "regenerate_run_summary.json"
        if args.arm == "regenerate"
        else "cannbot_debug_run_summary.json"
    )
    atomic_json(args.experiment_root / "experiment_harness" / summary_name, summary)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
