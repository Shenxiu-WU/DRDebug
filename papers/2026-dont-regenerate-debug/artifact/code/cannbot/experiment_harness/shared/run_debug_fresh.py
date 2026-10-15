#!/usr/bin/env python3
"""Launch only the N=27 Debug operators not covered by the frozen K3 run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=here / "run_manifest.json")
    parser.add_argument("--experiment-root", type=Path, default=Path("/workspace/regenerate_experiment"))
    parser.add_argument("--key-config", type=Path, default=Path("/workspace/regenerate_experiment/.secrets/provider_config.json"))
    parser.add_argument("--containers", default="cannbot-runtimebot_debug,cannbot-runtimebot_debug")
    parser.add_argument("--npus", default="4,5")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reset-all", action="store_true")
    args = parser.parse_args()

    root = args.experiment_root
    engine = root / "frozen_debug_engine"
    manifest = json.loads(args.manifest.read_text())
    fresh = [
        record
        for record in manifest["runs"]
        if record.get("arm") == "debug" and record.get("source_type") == "fresh_initial_near_miss"
    ]
    if args.only:
        fresh = [
            record
            for record in fresh
            if any(term in record.get("run_id", "") or term in record.get("operator", "") for term in args.only)
        ]
    if args.limit:
        fresh = fresh[: args.limit]
    if not fresh:
        raise SystemExit("no fresh Debug tasks matched")
    source_file = here / "fresh_debug_source_dirs.txt"
    source_file.write_text("".join(f"{record['source_task_path']}\n" for record in fresh))

    raw_keys = json.loads(args.key_config.read_text())
    raw_keys["model"] = "k3"
    raw_keys["base_url"] = raw_keys.get("base_url") or "https://api.kimi.com/coding/"
    for item in raw_keys.get("keys", []):
        item["model"] = "k3"
        item["base_url"] = item.get("base_url") or raw_keys["base_url"]
    step3_keys = root / ".secrets" / "kimi_keys_step3_k3.json"
    step3_keys.write_text(json.dumps(raw_keys, indent=2) + "\n")
    step3_keys.chmod(0o600)

    output = root / "debug_fresh_k3" / "agent2"
    kb = root / "debug_fresh_k3" / "precision_knowledge_base.runtime.json"
    if not kb.exists():
        source_kb = engine / "skills" / "ascendc" / "ascendc-debug" / "references" / "precision_knowledge_base.json"
        if not source_kb.is_file():
            raise SystemExit(f"missing frozen initial KB: {source_kb}")
        kb.parent.mkdir(parents=True, exist_ok=True)
        kb.write_bytes(source_kb.read_bytes())

    command = [
        "python3",
        str(engine / "utils" / "supervise_ascendc_debug_batch_cc_quota.py"),
        "--source-dirs-file",
        str(source_file),
        "--key-config",
        str(step3_keys),
        "--output",
        str(output),
        "--workdir",
        str(engine),
        "--containers",
        args.containers,
        "--npus",
        args.npus,
        "--timeout",
        "43200",
        "--max-attempts",
        "5",
        "--max-turns",
        "240",
        "--soft-task-turns",
        "480",
        "--max-task-turns",
        "600",
        "--agent",
        "constructive",
        "--entry-failure-type",
        "precision_failed",
        "--kb-path",
        str(kb),
        "--tilelang-env",
        "/workspace/debug_agent/set_env.sh",
        "--mixed-provider-min-remaining",
        "10",
    ]
    if args.reset_all:
        command.append("--reset-all")
    print(json.dumps({"fresh_debug_tasks": len(fresh), "command": command}, indent=2), flush=True)
    return subprocess.run(command, cwd=engine).returncode


if __name__ == "__main__":
    raise SystemExit(main())
