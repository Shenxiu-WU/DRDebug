#!/usr/bin/env python3
"""Join automatic anti-cheat verdicts with final integrity metadata."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTOMATIC = ROOT / "integrity/automatic_anticheat_verdicts.csv"
REASONS = ROOT / "integrity/integrity_resolution_reasons.csv"
OUTPUT = ROOT / "integrity/final_integrity_results.csv"
MANIFEST = ROOT / "integrity/integrity_manifest.json"


def read_index(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    index = {row["candidate_id"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Duplicate candidate_id in {path}")
    return index


def truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "pass"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    automatic = read_index(AUTOMATIC)
    reasons = read_index(REASONS)
    if automatic.keys() != reasons.keys():
        missing_resolution = sorted(automatic.keys() - reasons.keys())
        missing_automatic = sorted(reasons.keys() - automatic.keys())
        raise ValueError(
            f"Integrity layers disagree: missing_resolution={missing_resolution}, "
            f"missing_automatic={missing_automatic}"
        )

    fields = [
        "candidate_id",
        "method",
        "model",
        "operator_id",
        "trial",
        "full_eval_pass",
        "automatic_anticheat_verdict",
        "final_integrity_verdict",
        "pass_at_1",
        "resolution_code",
        "evidence_summary",
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in sorted(automatic):
            base = automatic[candidate]
            resolution = reasons[candidate]
            numeric = truth(base["full_eval_pass"])
            verdict = resolution["final_integrity_verdict"]
            if numeric and verdict not in {"CLEAN", "PURITY_VIOLATION"}:
                raise ValueError(f"Missing final verdict for full-eval pass: {candidate}")
            if not numeric and verdict != "NOT_APPLICABLE":
                raise ValueError(f"Non-passing candidate has an active integrity verdict: {candidate}")
            writer.writerow(
                {
                    **{field: base.get(field, "") for field in fields},
                    "final_integrity_verdict": verdict,
                    "pass_at_1": numeric and verdict == "CLEAN",
                    "resolution_code": resolution["resolution_code"],
                    "evidence_summary": resolution["evidence_summary"],
                }
            )

    manifest = {
        "schema_version": 1,
        "derivation": "automatic_anticheat_verdicts + final_integrity_metadata -> final_integrity_results",
        "inputs": {
            "automatic_anticheat_verdicts.csv": digest(AUTOMATIC),
            "integrity_resolution_reasons.csv": digest(REASONS),
        },
        "output": {"final_integrity_results.csv": digest(OUTPUT)},
        "script": {"build_final_integrity_results.py": digest(Path(__file__))},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(automatic)} candidates")


if __name__ == "__main__":
    main()
