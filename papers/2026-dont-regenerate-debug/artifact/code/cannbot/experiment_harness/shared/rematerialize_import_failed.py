#!/usr/bin/env python3
"""Re-materialize candidates invalidated only by evaluator import staging.

This is deliberately a zero-LLM repair: it copies already-built shared
libraries from each preserved operator workspace into ``candidate_task`` using
the contract-v2 materializer, then records an auditable repair report.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_regenerate import materialize_candidate


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--selection",
        choices=("import-failed", "non-import-failed", "all"),
        default="import-failed",
    )
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    results: list[dict[str, Any]] = []
    for run in manifest.get("runs", []):
        full_eval_path = Path(run["full_eval_path"])
        if not full_eval_path.is_file():
            continue
        full_eval = load_json(full_eval_path)
        is_import_failed = full_eval.get("final_failure_type") == "import_failed"
        if args.selection == "import-failed" and not is_import_failed:
            continue
        if args.selection == "non-import-failed" and is_import_failed:
            continue

        cwd = Path(run["cwd"])
        before_manifest = load_json(cwd / "candidate_task" / "materialization_manifest.json")
        declared = [Path(item) for item in before_manifest.get("libraries", [])]
        before_candidate_exists = [
            (cwd / "candidate_task" / relative).is_file() for relative in declared
        ]
        operator_project = (cwd / "candidate_adapter" / "operator_dir.txt").read_text().strip()
        source_root = (cwd / operator_project).resolve()
        source_exists = [(source_root / relative).is_file() for relative in declared]

        ok, note = materialize_candidate(cwd)
        after_manifest = load_json(cwd / "candidate_task" / "materialization_manifest.json")
        after_declared = [Path(item) for item in after_manifest.get("libraries", [])]
        after_candidate_exists = [
            (cwd / "candidate_task" / relative).is_file() for relative in after_declared
        ]
        valid = bool(
            ok
            and declared
            and all(source_exists)
            and after_declared
            and all(after_candidate_exists)
        )
        results.append(
            {
                "run_id": run["run_id"],
                "operator": run["operator"],
                "cwd": str(cwd),
                "prior_failure_type": full_eval.get("final_failure_type"),
                "declared_libraries_before": [str(item) for item in declared],
                "source_libraries_exist": source_exists,
                "candidate_libraries_existed_before": before_candidate_exists,
                "materialize_ok": ok,
                "materialize_note": note,
                "declared_libraries_after": [str(item) for item in after_declared],
                "candidate_libraries_exist_after": after_candidate_exists,
                "repair_valid": valid,
            }
        )

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repair_type": "zero_llm_candidate_shared_library_rematerialization",
        "selection": args.selection,
        "manifest": str(args.manifest),
        "requested": len(results),
        "repaired": sum(bool(row["repair_valid"]) for row in results),
        "errors": sum(not bool(row["repair_valid"]) for row in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: report[key] for key in ("requested", "repaired", "errors")}))
    return 0 if report["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
