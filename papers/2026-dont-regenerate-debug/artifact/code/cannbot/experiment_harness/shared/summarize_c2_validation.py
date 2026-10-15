#!/usr/bin/env python3
"""Create one canonical 27-row validation summary after corrective reruns."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INVALID_FAILURES = {
    "import_failed",
    "validator_preflight_error",
    "harness_staging_invalid",
    "validation_infrastructure_error",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def verdict(value: Any) -> str:
    return str(value or "").strip().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for run in load(args.manifest).get("runs", []):
        cwd = Path(run["cwd"])
        full_path = Path(run["full_eval_path"])
        anti_path = Path(run["anticheat_path"])
        full = load(full_path) if full_path.is_file() else {}
        anti = load(anti_path) if anti_path.is_file() else {}
        external_path = cwd / "validation" / "work" / "_external_eval_manifest.json"
        external = load(external_path) if external_path.is_file() else {}
        material_path = cwd / "candidate_task" / "materialization_manifest.json"
        material = load(material_path) if material_path.is_file() else {}
        frozen_model_path = Path(run["frozen_input_path"]) / "model.py"
        candidate_model_path = cwd / "candidate_task" / "model.py"
        reference_model_byte_identical = bool(
            frozen_model_path.is_file()
            and candidate_model_path.is_file()
            and frozen_model_path.read_bytes() == candidate_model_path.read_bytes()
        )
        libraries = [Path(item) for item in material.get("libraries", [])]
        library_presence = [
            (cwd / "candidate_task" / relative).is_file() for relative in libraries
        ]

        full_result = verdict(full.get("full_eval_result") or full.get("verdict"))
        failure = str(full.get("final_failure_type") or full.get("failure_class") or "")
        harness = verdict(anti.get("verdict"))
        official = verdict(anti.get("official_verdict") or anti.get("raw_official_verdict"))
        if not official:
            official = harness
        staging_valid = bool(
            external
            and material
            and libraries
            and all(library_presence)
            and external.get("staged_library_paths")
        )
        outcome_valid = bool(
            staging_valid
            and full_result in {"PASS", "FAIL"}
            and failure not in INVALID_FAILURES
        )
        rows.append(
            {
                "run_id": run["run_id"],
                "operator": run["operator"],
                "trial": run["trial"],
                "candidate_materialized": bool(libraries and all(library_presence)),
                "reference_model_byte_identical": reference_model_byte_identical,
                "frozen_case_count": external.get("full_case_lines"),
                "declared_library_count": len(libraries),
                "staging_valid": staging_valid,
                "staged_kernel_build_alias": external.get("staged_kernel_build_alias"),
                "staged_library_paths": ";".join(external.get("staged_library_paths") or []),
                "full_eval_npu": full.get("npu"),
                "full_eval_result": full_result,
                "failure_class": failure,
                "matched_count": full.get("matched_count"),
                "total_outputs": full.get("total_outputs"),
                "match_rate_pct": full.get("match_rate_pct"),
                "official_anticheat": official,
                "harness_anticheat": harness,
                "compatibility_override": bool(anti.get("compatibility_override")),
                "outcome_evidence_valid": outcome_valid,
                "strict_auto_success": bool(
                    outcome_valid and full_result == "PASS" and harness == "CLEAN"
                ),
            }
        )

    full_counts = Counter(row["full_eval_result"] for row in rows)
    harness_counts = Counter(row["harness_anticheat"] for row in rows)
    official_counts = Counter(row["official_anticheat"] for row in rows)
    failure_counts = Counter(
        row["failure_class"] or "pass" for row in rows if row["full_eval_result"] != "PASS"
    )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "outcome_evidence_valid": sum(row["outcome_evidence_valid"] for row in rows),
        "candidate_materialized": sum(row["candidate_materialized"] for row in rows),
        "reference_model_byte_identical": sum(
            row["reference_model_byte_identical"] for row in rows
        ),
        "full_eval": dict(sorted(full_counts.items())),
        "failure_classes": dict(sorted(failure_counts.items())),
        "harness_anticheat": dict(sorted(harness_counts.items())),
        "official_anticheat": dict(sorted(official_counts.items())),
        "strict_auto_success": sum(row["strict_auto_success"] for row in rows),
        "results": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "rows",
                    "outcome_evidence_valid",
                    "full_eval",
                    "harness_anticheat",
                    "official_anticheat",
                    "strict_auto_success",
                )
            }
        )
    )
    return (
        0
        if len(rows) == 27
        and payload["outcome_evidence_valid"] == 27
        and payload["reference_model_byte_identical"] == 27
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
