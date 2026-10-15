#!/usr/bin/env python3
"""Summarize the matched first-compact-pass shadow of removing full evaluation."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/rq4_no_full_evaluation_shadow.csv"
OUTPUT = ROOT / "reproduced/no_full_evaluation_shadow.json"


def truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "pass"}


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    categories = Counter(row["shadow_outcome"] for row in rows)
    summary = {
        "operators": len(rows),
        "arm_local_compact_successes": sum(truth(row["compact_objective_pass"]) for row in rows),
        "final_pass_at_1": sum(truth(row["pass_at_1"]) for row in rows),
        "shadow_outcomes": dict(sorted(categories.items())),
    }
    expected = {
        "operators": 27,
        "arm_local_compact_successes": 18,
        "final_pass_at_1": 12,
    }
    for key, value in expected.items():
        if summary[key] != value:
            raise ValueError(f"Unexpected {key}: {summary[key]} != {value}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
