#!/usr/bin/env python3
"""Quota-aware supervisor for run_ascendc_debug_batch_cc.sh.

This wrapper keeps experiments inside Docker via the existing batch_cc runner.
It runs on the host only to prepare task copies, generate per-provider env files,
invoke the Docker batch runner, and rotate Kimi keys when provider-level API
errors are detected.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return None


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{now()}] {message}\n")


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def unique_containers(value: str) -> list[str]:
    seen: set[str] = set()
    containers: list[str] = []
    for raw in value.split(","):
        name = raw.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        containers.append(name)
    return containers


def cleanup_leftover_processes(containers: str, target_path: Path, log_path: Path) -> None:
    """Kill stale container processes whose cwd/cmdline references target_path.

    Provider rotation resets task directories between cycles. If a detached
    engine/Claude/build subprocess survives the previous cycle, it can write
    stale events into the freshly copied task directory. Clean by output root
    before/after each cycle to prevent cross-cycle path collisions.
    """
    script = r'''
        set +e
        target="$1"
        kill_matching() {
            sig="$1"
            for proc in /proc/[0-9]*; do
                pid="${proc##*/}"
                [ "$pid" = "1" ] && continue
                [ "$pid" = "$$" ] && continue
                [ "$pid" = "$BASHPID" ] && continue
                [ "$pid" = "$PPID" ] && continue
                cmd="$(tr "\0" " " < "$proc/cmdline" 2>/dev/null || true)"
                cwd="$(readlink "$proc/cwd" 2>/dev/null || true)"
                case "$cmd $cwd" in
                    *"$target"*) kill "-$sig" "$pid" 2>/dev/null || true ;;
                esac
            done
        }
        kill_matching TERM
        sleep 1
        kill_matching KILL
    '''
    for container in unique_containers(containers):
        try:
            proc = subprocess.run(
                ["docker", "exec", container, "bash", "-lc", script, "_", str(target_path)],
                cwd=target_path.parent if target_path.parent.exists() else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
            )
        except Exception as exc:  # noqa: BLE001
            append_log(log_path, f"cleanup_leftovers container={container} path={target_path} error={exc}")
            continue
        if proc.returncode != 0:
            tail = (proc.stdout or "").strip().replace("\n", " | ")[:300]
            append_log(
                log_path,
                f"cleanup_leftovers container={container} path={target_path} rc={proc.returncode} {tail}",
            )


@dataclass(frozen=True)
class Provider:
    name: str
    api_key: str
    model: str
    base_url: str


@dataclass(frozen=True)
class Target:
    source: Path
    task: Path

    @property
    def rel(self) -> str:
        return f"{self.task.parent.name}/{self.task.name}"


def load_providers(path: Path) -> list[Provider]:
    data = read_json(path)
    if not data:
        raise SystemExit(f"missing or invalid key config: {path}")
    default_model = data.get("model", "kimi-for-coding/k2p7")
    default_base_url = data.get("base_url", "https://api.kimi.com/coding/")
    providers: list[Provider] = []
    for item in data.get("keys", []):
        name = item.get("name")
        api_key = item.get("api_key")
        if not name or not api_key:
            continue
        providers.append(
            Provider(
                name=name,
                api_key=api_key,
                model=item.get("model", default_model),
                base_url=item.get("base_url", default_base_url),
            )
        )
    if not providers:
        raise SystemExit(f"no usable providers in {path}")
    return providers


def query_kimi_usage(provider: Provider, timeout: int) -> dict[str, Any]:
    state: dict[str, Any] = {
        "name": provider.name,
        "model": provider.model,
        "base_url": provider.base_url,
        "queried_at": now(),
        "supported": "api.kimi.com/coding" in provider.base_url.lower(),
    }
    if not state["supported"]:
        return state
    req = urllib.request.Request(
        "https://api.kimi.com/coding/v1/usages",
        headers={
            "Authorization": f"Bearer {provider.api_key}",
            "Accept": "application/json",
            "User-Agent": "claude-code",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            state["http_status"] = resp.status
            raw = resp.read(50_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        state["http_status"] = exc.code
        raw = exc.read(10_000).decode("utf-8", "replace")
    except Exception as exc:
        state["success"] = False
        state["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return state
    try:
        body = json.loads(raw)
    except Exception:
        state["success"] = False
        state["error"] = raw[:500]
        return state
    state["success"] = 200 <= int(state.get("http_status", 0)) < 300
    if not state["success"]:
        state["error"] = body.get("error", body) if isinstance(body, dict) else body
        return state

    def window(detail: Any) -> dict[str, Any] | None:
        if not isinstance(detail, dict):
            return None
        limit = detail.get("limit")
        remaining = detail.get("remaining")
        used = detail.get("used")
        return {
            "limit": limit,
            "used": used,
            "remaining": remaining,
            "reset_time": detail.get("resetTime"),
        }

    if isinstance(body, dict):
        state["weekly_limit"] = window(body.get("usage"))
        limits = body.get("limits")
        if isinstance(limits, list) and limits:
            first = limits[0] if isinstance(limits[0], dict) else {}
            state["five_hour"] = window(first.get("detail"))
        state["total_quota"] = window(body.get("totalQuota"))
        parallel = body.get("parallel")
        if isinstance(parallel, dict):
            state["parallel_limit"] = parallel.get("limit")
    return state


def write_usage_snapshot(
    output: Path,
    providers: list[Provider],
    timeout: int,
    log_path: Path,
    reason: str,
) -> None:
    usage = [query_kimi_usage(provider, timeout) for provider in providers]
    payload = {
        "updated_at": now(),
        "reason": reason,
        "providers": usage,
    }
    atomic_write_json(output / "kimi_usage_latest.json", payload)
    # Backward-compatible snapshot name used by earlier runs.
    atomic_write_json(output / "kimi_usage_snapshot.json", usage)
    with (output / "kimi_usage_history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    summaries = []
    for item in usage:
        weekly = item.get("weekly_limit") or {}
        five_hour = item.get("five_hour") or {}
        summaries.append(
            f"{item.get('name')}:5h={five_hour.get('remaining', '?')}/{five_hour.get('limit', '?')}"
            f",weekly={weekly.get('remaining', '?')}/{weekly.get('limit', '?')}"
        )
    append_log(log_path, f"usage_poll reason={reason} {'; '.join(summaries)}")


def start_usage_poller(
    *,
    output: Path,
    providers: list[Provider],
    timeout: int,
    interval: int,
    log_path: Path,
) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()

    def loop() -> None:
        count = 0
        while not stop.is_set():
            count += 1
            try:
                write_usage_snapshot(
                    output,
                    providers,
                    timeout,
                    log_path,
                    "initial" if count == 1 else f"periodic_{count}",
                )
            except Exception as exc:  # noqa: BLE001
                append_log(log_path, f"usage_poll error={type(exc).__name__}: {str(exc)[:300]}")
            if interval <= 0:
                break
            stop.wait(interval)

    thread = threading.Thread(target=loop, name="kimi-usage-poller", daemon=True)
    thread.start()
    return stop, thread


def build_targets(source_dirs_file: Path, tasks_root: Path) -> list[Target]:
    targets: list[Target] = []
    for raw in source_dirs_file.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        source = Path(line)
        if not source.exists():
            raise SystemExit(f"missing source task dir: {source}")
        level = source.parent.name
        task = tasks_root / level / source.name
        targets.append(Target(source=source, task=task))
    if not targets:
        raise SystemExit(f"no targets in {source_dirs_file}")
    return targets


def reset_target(target: Target) -> None:
    remove_tree(target.task)
    target.task.parent.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns(
        "build",
        "__pycache__",
        "*.pyc",
        ".debug_events",
        ".verify_logs",
        ".verify_status",
        ".bench_baseline",
        "precision_tuning",
        "debug_status.json",
        "debug_trace.md",
        "_anticheat.json",
        "_claude_result*.json",
    )
    shutil.copytree(target.source, target.task, ignore=ignore)


def archive_retry_evidence(target: Target, reason: str) -> Path | None:
    """Preserve abnormal-run evidence before a retry reset rewrites task files."""

    if not target.task.exists():
        return None
    try:
        output_root = target.task.parents[2]
    except IndexError:
        output_root = target.task.parent
    archive = (
        output_root
        / "retry_evidence"
        / target.rel.replace("/", "__")
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    archive.mkdir(parents=True, exist_ok=True)
    evidence_rels = [
        Path("precision_tuning"),
        Path(".debug_events/events.jsonl"),
        Path(".verify_logs"),
        Path(".verify_status"),
        Path("debug_status.json"),
        Path("debug_trace.md"),
        Path("_anticheat.json"),
    ]
    copied: list[str] = []
    for rel in evidence_rels:
        src = target.task / rel
        if not src.exists():
            continue
        dst = archive / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        copied.append(str(rel))
    for pattern in (
        "_claude_result*.json",
        "_provider*.json",
        "_provider*.jsonl",
    ):
        for src in sorted(target.task.glob(pattern)):
            if not src.is_file():
                continue
            shutil.copy2(src, archive / src.name)
            copied.append(src.name)
    if not copied:
        shutil.rmtree(archive, ignore_errors=True)
        return None
    atomic_write_json(
        archive / "archive_manifest.json",
        {
            "archived_at": now(),
            "reason": reason,
            "target": target.rel,
            "source": str(target.source),
            "task": str(target.task),
            "copied": copied,
            "cost_accounting": {
                "excluded_from_primary_cost": True,
                "primary_cost_scope": "final_valid_cycle_only",
                "retention_purpose": (
                    "provider stability and failed-cycle root-cause diagnostics"
                ),
            },
        },
    )
    return archive


def task_outcome(task: Path) -> str | None:
    data = read_json(task / "debug_status.json")
    if not data:
        return None
    value = data.get("session_outcome")
    return str(value) if value else None


def should_retry(target: Target) -> bool:
    data = read_json(target.task / "debug_status.json")
    if not data:
        return True
    outcome = data.get("session_outcome")
    if outcome == "provider_api_error":
        return True
    notes = str(data.get("notes") or "")
    if outcome == "crashed" and (
        data.get("ended_at") is None
        or "无终态事件" in notes
        or "异常中断" in notes
    ):
        return True
    if (
        outcome == "crashed"
        and data.get("entry_failure_type") is None
        and "failure_type" in notes
    ):
        return True
    return False


def is_done(target: Target) -> bool:
    outcome = task_outcome(target.task)
    return outcome is not None and not should_retry(target)


def pending_targets(targets: list[Target]) -> list[Target]:
    return [target for target in targets if not is_done(target)]


def write_provider_env(path: Path, provider: Provider) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "#!/usr/bin/env bash",
            f"export ANTHROPIC_MODEL={json.dumps(provider.model)}",
            f"export ANTHROPIC_BASE_URL={json.dumps(provider.base_url)}",
            f"export ANTHROPIC_API_KEY={json.dumps(provider.api_key)}",
            "unset ANTHROPIC_AUTH_TOKEN",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


def write_task_file(path: Path, targets: list[Target]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(target.task) for target in targets) + "\n", encoding="utf-8")


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _elapsed_sec(started_at: Any, ended_at: Any) -> int | str:
    started = _parse_utc(started_at)
    ended = _parse_utc(ended_at)
    if started is None or ended is None:
        return ""
    return max(0, int((ended - started).total_seconds()))


def _tsv(value: Any) -> str:
    return str(value or "").replace("\t", " ").replace("\n", " ")


def write_task_elapsed_summary(path: Path, targets: list[Target]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "\t".join(
            [
                "rel",
                "task",
                "source",
                "session_outcome",
                "started_at",
                "ended_at",
                "elapsed_sec",
                "attempts_used",
                "entry_failure_type",
                "final_failure_type",
            ]
        )
    ]
    for target in targets:
        status = read_json(target.task / "debug_status.json") or {}
        rows.append(
            "\t".join(
                _tsv(value)
                for value in [
                    target.rel,
                    target.task,
                    target.source,
                    status.get("session_outcome"),
                    status.get("started_at"),
                    status.get("ended_at"),
                    _elapsed_sec(status.get("started_at"), status.get("ended_at")),
                    status.get("attempts_used"),
                    status.get("entry_failure_type"),
                    status.get("final_failure_type"),
                ]
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def snapshot_state(path: Path, targets: list[Target], cycle: int, provider: str, event: str) -> None:
    rows = []
    for target in targets:
        rows.append({
            "task": str(target.task),
            "source": str(target.source),
            "rel": target.rel,
            "outcome": task_outcome(target.task),
            "done": is_done(target),
        })
    atomic_write_json(
        path,
        {
            "updated_at": now(),
            "cycle": cycle,
            "provider": provider,
            "event": event,
            "total": len(targets),
            "done": sum(1 for target in targets if is_done(target)),
            "pending": sum(1 for target in targets if not is_done(target)),
            "targets": rows,
        },
    )


def write_manifest(path: Path, args: argparse.Namespace, providers: list[Provider], targets: list[Target]) -> None:
    atomic_write_json(
        path,
        {
            "started_at": now(),
            "source_dirs_file": str(args.source_dirs_file),
            "key_config": str(args.key_config),
            "output": str(args.output),
            "workdir": str(args.workdir),
            "containers": args.containers,
            "npus": args.npus,
            "timeout_sec": args.timeout,
            "max_attempts": args.max_attempts,
            "agent_timeout_sec": args.agent_timeout,
            "max_turns": args.max_turns,
            "soft_task_turns": args.soft_task_turns,
            "max_task_turns": args.max_task_turns,
            "ablate_profile": args.ablate_profile,
            "kb_path": str(args.kb_path) if args.kb_path else None,
            "kb_read_only": args.kb_read_only,
            "max_cycles": args.max_cycles,
            "agent": args.agent,
            "entry_failure_type": args.entry_failure_type,
            "usage_poll_interval_sec": args.usage_poll_interval,
            "usage_query_enabled": not args.disable_usage_query,
            "mixed_provider_enabled": not args.disable_mixed_provider,
            "mixed_provider_min_remaining": args.mixed_provider_min_remaining,
            "model": (
                providers[0].model
                if len({provider.model for provider in providers}) == 1
                else sorted({provider.model for provider in providers})
            ),
            "provider_assignment_mode": (
                "mixed_provider"
                if not args.disable_mixed_provider
                else "fixed_single_provider"
            ),
            "provider_env_storage": (
                "ephemeral_secret_dir"
                if args.provider_env_base
                else "experiment_output"
            ),
            "providers": [provider.name for provider in providers],
            "targets": [{"rel": target.rel, "source": str(target.source), "task": str(target.task)}
                        for target in targets],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dirs-file", type=Path, required=True)
    parser.add_argument("--key-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path("/workspace/debug_agent"))
    parser.add_argument("--containers", default="cannbot-runtime,cannbot-runtime")
    parser.add_argument("--npus", default="2,3")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--agent-timeout", type=int, default=None)
    parser.add_argument("--max-turns", default="240")
    parser.add_argument("--soft-task-turns", type=int, default=480)
    parser.add_argument("--max-task-turns", type=int, default=600)
    parser.add_argument("--ablate-profile", default="full")
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--agent", default="constructive")
    parser.add_argument("--entry-failure-type", default="precision_failed")
    parser.add_argument("--kb-path", type=Path, default=None)
    parser.add_argument("--kb-read-only", action="store_true")
    parser.add_argument("--tilelang-env", default="/workspace/debug_agent/set_env.sh")
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--usage-timeout", type=int, default=10)
    parser.add_argument("--usage-poll-interval", type=int, default=300)
    parser.add_argument("--mixed-provider-min-remaining", type=float, default=10.0)
    parser.add_argument("--disable-mixed-provider", action="store_true")
    parser.add_argument("--disable-usage-query", action="store_true")
    parser.add_argument(
        "--provider-env-base",
        type=Path,
        default=None,
        help="Create provider env files in a temporary 0700 directory here.",
    )
    parser.add_argument("--reset-all", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "quota_batch_cc_supervisor.log"
    state_path = args.output / "quota_batch_cc_state.json"
    tasks_root = args.output / "tasks"
    cycles_root = args.output / "cycles"
    ephemeral_env_root = args.provider_env_base is not None
    if args.provider_env_base is not None:
        args.provider_env_base.mkdir(parents=True, exist_ok=True)
        args.provider_env_base.chmod(0o700)
        env_root = Path(tempfile.mkdtemp(
            prefix="ascendc-provider-",
            dir=args.provider_env_base,
        ))
        env_root.chmod(0o700)
    else:
        env_root = args.output / ".provider_env"
    elapsed_path = args.output / "task_final_elapsed.tsv"

    providers = load_providers(args.key_config)
    max_cycles = args.max_cycles or max(1, len(providers) * 4)
    targets = build_targets(args.source_dirs_file, tasks_root)
    write_manifest(args.output / "experiment_manifest.json", args, providers, targets)

    if args.reset_all:
        append_log(log_path, "reset_all requested")
        cleanup_leftover_processes(args.containers, tasks_root, log_path)
        remove_tree(tasks_root)
    for target in targets:
        if not target.task.exists():
            reset_target(target)

    usage_stop: threading.Event | None = None
    usage_thread: threading.Thread | None = None
    if not args.disable_usage_query:
        usage_stop, usage_thread = start_usage_poller(
            output=args.output,
            providers=providers,
            timeout=args.usage_timeout,
            interval=args.usage_poll_interval,
            log_path=log_path,
        )

    provider_idx = 0
    try:
        append_log(log_path, f"start total={len(targets)} providers={','.join(p.name for p in providers)}")
        snapshot_state(state_path, targets, 0, "", "prepared")
        write_task_elapsed_summary(elapsed_path, targets)

        for cycle in range(1, max_cycles + 1):
            pending = pending_targets(targets)
            if not pending:
                append_log(log_path, "all targets completed")
                snapshot_state(state_path, targets, cycle, "", "completed")
                write_task_elapsed_summary(elapsed_path, targets)
                return 0

            provider = providers[provider_idx]
            mixed_provider = not args.disable_mixed_provider
            cleanup_leftover_processes(args.containers, tasks_root, log_path)
            for target in pending:
                if should_retry(target):
                    archived = archive_retry_evidence(target, "retry_reset")
                    if archived:
                        append_log(log_path, f"archived_retry_evidence target={target.rel} path={archived}")
                    reset_target(target)

            env_file = env_root / f"{cycle:03d}_{provider.name}.env"
            write_provider_env(env_file, provider)
            cycle_label = "mixed" if mixed_provider else provider.name
            cycle_dir = cycles_root / f"cycle_{cycle:03d}_{cycle_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            pending_file = cycle_dir / "pending_task_dirs.txt"
            write_task_file(pending_file, pending)

            provider_state = "mixed" if mixed_provider else provider.name
            append_log(
                log_path,
                f"cycle={cycle} provider={provider_state} fallback_provider={provider.name} "
                f"pending={len(pending)}",
            )
            snapshot_state(state_path, targets, cycle, provider_state, "cycle_start")

            cmd = [
                "bash",
                "utils/run_ascendc_debug_batch_cc.sh",
                "--task-dirs-file",
                str(pending_file),
                "--containers",
                args.containers,
                "--npus",
                args.npus,
                "--output",
                str(cycle_dir),
                "--workdir",
                str(args.workdir),
                "--tilelang-env",
                args.tilelang_env,
                "--claude-env",
                str(env_file),
                "--claude-bin",
                args.claude_bin,
                "--agent",
                args.agent,
                "--entry-failure-type",
                args.entry_failure_type,
                "--timeout",
                str(args.timeout),
                "--max-attempts",
                str(args.max_attempts),
            ]
            if args.max_turns:
                cmd.extend(["--max-turns", str(args.max_turns)])
            if args.soft_task_turns is not None:
                cmd.extend(["--soft-task-turns", str(args.soft_task_turns)])
            if args.max_task_turns is not None:
                cmd.extend(["--max-task-turns", str(args.max_task_turns)])
            if args.agent_timeout is not None:
                cmd.extend(["--agent-timeout", str(args.agent_timeout)])
            if args.ablate_profile:
                cmd.extend(["--ablate-profile", str(args.ablate_profile)])
            if args.kb_path:
                cmd.extend(["--kb-path", str(args.kb_path)])
            if args.kb_read_only:
                cmd.append("--kb-read-only")
            if mixed_provider:
                cmd.extend(
                    [
                        "--provider-pool-config",
                        str(args.key_config),
                        "--mixed-provider-min-remaining",
                        str(args.mixed_provider_min_remaining),
                    ]
                )
            started = time.time()
            rc = subprocess.run(cmd, cwd=args.workdir).returncode
            elapsed = int(time.time() - started)
            cleanup_leftover_processes(args.containers, tasks_root, log_path)
            fatal = (cycle_dir / ".fatal").read_text(errors="replace").strip() if (cycle_dir / ".fatal").exists() else ""
            append_log(
                log_path,
                f"cycle={cycle} provider={provider_state} fallback_provider={provider.name} "
                f"rc={rc} elapsed={elapsed}s fatal={fatal or '<none>'}",
            )
            if fatal.startswith("provider_api_error"):
                old_provider = provider.name
                provider_idx = (provider_idx + 1) % len(providers)
                append_log(
                    log_path,
                    f"provider_switch old={old_provider} new={providers[provider_idx].name} reason={fatal}",
                )
            snapshot_state(state_path, targets, cycle, provider_state, "cycle_done")
            write_task_elapsed_summary(elapsed_path, targets)

        append_log(log_path, f"stopped: reached max_cycles={max_cycles}")
        snapshot_state(state_path, targets, max_cycles, "", "max_cycles_reached")
        write_task_elapsed_summary(elapsed_path, targets)
        return 2
    finally:
        if usage_stop is not None:
            usage_stop.set()
        if usage_thread is not None:
            usage_thread.join(timeout=max(1, min(args.usage_timeout + 1, 15)))
        if ephemeral_env_root:
            remove_tree(env_root)


if __name__ == "__main__":
    raise SystemExit(main())
