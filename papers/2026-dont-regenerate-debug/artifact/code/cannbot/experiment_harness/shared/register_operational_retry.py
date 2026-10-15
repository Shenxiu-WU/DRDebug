#!/usr/bin/env python3
"""Register a provider/transport retry as a new isolated session row."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from prepare_experiment import NPU_DEVICE_GID, escaped_project_dir, secure_regenerate_cwd


NAMESPACE = uuid.UUID("be8fb85f-d529-4899-8317-c66011df550d")


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run_id")
    parser.add_argument("--retry", type=int, required=True)
    parser.add_argument("--manifest", type=Path, default=here / "run_manifest.json")
    args = parser.parse_args()

    data = json.loads(args.manifest.read_text())
    source = next((record for record in data["runs"] if record.get("run_id") == args.source_run_id), None)
    if source is None or source.get("arm") != "regenerate":
        raise SystemExit(f"missing Regenerate source run: {args.source_run_id}")
    run_id = f"{args.source_run_id}:retry{args.retry}"
    existing = next((record for record in data["runs"] if record.get("run_id") == run_id), None)
    if existing:
        print(json.dumps(existing, indent=2))
        return 0

    source_cwd = Path(source["cwd"])
    cwd = source_cwd.parent / f"trial{source['trial']}_retry{args.retry}"
    if cwd.exists():
        raise SystemExit(f"retry cwd already exists but is not registered: {cwd}")
    cwd.mkdir(parents=True)
    shutil.copytree(source_cwd / "spec", cwd / "spec", symlinks=True)
    # Copy only the allowlisted skill links.  Per-trial settings are generated
    # below so HOME can never point back to the source/superseded session.
    shutil.copytree(source_cwd / ".claude" / "skills", cwd / ".claude" / "skills", symlinks=True)
    for path in (cwd / "evidence", cwd / "validation", cwd / ".home"):
        path.mkdir(parents=True)
    session_id = str(uuid.uuid5(NAMESPACE, run_id))
    escaped = escaped_project_dir(cwd)
    project = cwd / ".home" / ".claude" / "projects" / escaped
    run_record = cwd / "run_record.json"
    run_record.write_text(
        json.dumps(
            {
                "status": "prepared",
                "operator": source["operator"],
                "arm": "regenerate",
                "trial": source["trial"],
                "session_id": session_id,
                "repair_policy": "repair_allowed",
                "operational_retry_of": args.source_run_id,
            },
            indent=2,
        )
        + "\n"
    )
    record = dict(source)
    container_uid = max(int(row.get("container_uid", 30000)) for row in data["runs"]) + 1
    record.update(
        {
            "run_id": run_id,
            "cwd": str(cwd),
            "expected_session_id": session_id,
            "final_result_path": str(cwd / "evidence" / "final_stdout.json"),
            "journal_main_path": str(project / f"{session_id}.jsonl"),
            "subagent_dir": str(project / session_id / "subagents"),
            "full_eval_path": str(cwd / "validation" / "full_eval.json"),
            "anticheat_path": str(cwd / "validation" / "anticheat.json"),
            "run_record_path": str(run_record),
            "operational_retry_of": args.source_run_id,
            "is_operational_retry": True,
            "container_uid": container_uid,
            "container_gid": NPU_DEVICE_GID,
        }
    )
    secure_regenerate_cwd(cwd, container_uid, NPU_DEVICE_GID)
    data["runs"].append(record)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = args.manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.manifest)
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
