#!/usr/bin/env python3
"""Verify the frozen task-data companion without importing any task code."""

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


ROOT = Path(__file__).resolve().parent
COHORT = ROOT.parent / "artifact/manifests/frozen_27_operators.csv"
SUPPORT_FILES = {"README.md", "PROVENANCE.json", "NOTICE", "LICENSE-APACHE-2.0.txt",
                 "verify_dataset.py"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def safe_file(relative):
    path = PurePosixPath(relative)
    require(not path.is_absolute() and str(path) == relative and ".." not in path.parts
            and "\\" not in relative, "Unsafe manifest path")
    target = ROOT / relative
    require(target.is_file() and not target.is_symlink(), "Missing or symlinked dataset file")
    require(target.resolve().is_relative_to(ROOT), "Dataset file escapes its root")
    return target


def read_cases(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        require(isinstance(case, dict) and "inputs" in case, "Invalid input-case row")
        rows.append(case)
    normalized = [json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                  for row in rows]
    # The original V5 normalizer sorts canonical JSONL rows, retains duplicates,
    # and terminates each row with a newline. Task code is never imported/evaluated.
    content = "".join(row + "\n" for row in sorted(normalized)).encode("utf-8")
    return len(rows), digest(content)


def verify():
    expected = {}
    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sha, relative = line.split("  ", 1)
        require(re.fullmatch(r"[0-9a-f]{64}", sha) is not None, "Malformed checksum")
        require(relative not in expected, "Duplicate checksum entry")
        require(digest(safe_file(relative).read_bytes()) == sha, "File checksum mismatch")
        expected[relative] = sha
    actual = set()
    ignored_local_metadata = []
    for path in ROOT.rglob("*"):
        require(not path.is_symlink(), "Dataset contains a symlink")
        if path.is_file():
            relative = path.relative_to(ROOT)
            if (path.name == ".DS_Store" or "__pycache__" in relative.parts[:-1]
                    or path.suffix == ".pyc"):
                ignored_local_metadata.append(relative.as_posix())
                continue
            actual.add(relative.as_posix())
    require(actual == set(expected) | {"MANIFEST.sha256"}, "Unexpected or missing dataset files")
    provenance = json.loads(safe_file("PROVENANCE.json").read_text(encoding="utf-8"))
    require(provenance["schema_version"] == 1, "Unsupported provenance schema")
    require(COHORT.is_file() and not COHORT.is_symlink(), "Frozen artifact cohort CSV is required")
    require(digest(COHORT.read_bytes()) == provenance["frozen_cohort"]["sha256"],
            "Frozen artifact cohort CSV changed")
    with COHORT.open(newline="", encoding="utf-8") as stream:
        cohort_rows = list(csv.DictReader(stream))
    cohort = {row["operator_id"]: row for row in cohort_rows}
    operators = provenance["operators"]
    require(len(cohort_rows) == len(cohort) == len(operators) == 27, "Expected exactly 27 operators")
    require({row["operator_id"] for row in operators} == set(cohort), "Cohort identities differ")
    tasks = set()
    summary = {"operators": 27, "reference_byte_exact": 0, "compact_byte_exact": 0,
               "full_byte_exact": 0, "compact_case_set_exact": 0, "full_case_set_exact": 0,
               "compact_cases": 0, "full_cases": 0, "raw_byte_exceptions": []}
    for operator in operators:
        identifier = operator["operator_id"]
        frozen = cohort[identifier]
        level, name = identifier.split("/", 1)
        index, suffix = name.split("_", 1)
        basename = f"{int(index)}_{suffix}.json"
        reference = operator["reference"]
        require(reference["path"] == f"tasks/{identifier}/model.py", "Invalid reference path")
        tasks.add(reference["path"])
        require(digest(safe_file(reference["path"]).read_bytes())
                == reference["sha256"] == reference["frozen_byte_sha256"],
                "Reference differs from the frozen input-file hash")
        summary["reference_byte_exact"] += 1
        for role in ("compact", "full"):
            entry = operator[role]
            filename = basename + (".bak" if role == "full" else "")
            require(entry["path"] == f"tasks/{identifier}/{filename}", "Invalid case-file path")
            tasks.add(entry["path"])
            path = safe_file(entry["path"])
            actual_sha = digest(path.read_bytes())
            require(actual_sha == entry["sha256"], "Case-file checksum differs from provenance")
            count, normalized = read_cases(path)
            require(count == entry["cases"] == int(frozen[f"{role}_cases"]), "Case count differs")
            require(normalized == entry["normalized_case_set_sha256"]
                    == frozen[f"{role}_case_set_sha256"], "Frozen normalized case-set mismatch")
            raw_exact = actual_sha == entry["frozen_byte_sha256"]
            require(raw_exact == entry["byte_matches_frozen"], "Incorrect byte-identity declaration")
            require(raw_exact, "Case file differs from the original frozen byte hash")
            summary[f"{role}_byte_exact"] += 1
            summary[f"{role}_case_set_exact"] += 1
            summary[f"{role}_cases"] += count
    require(set(expected) == tasks | SUPPORT_FILES, "Companion contains unapproved files")
    require(len(tasks) == 81, "Expected exactly 81 reference/case files")
    summary["checksum_files"] = len(expected)
    summary["ignored_local_metadata"] = sorted(ignored_local_metadata)
    summary["task_execution"] = "none"
    return summary


if __name__ == "__main__":
    try:
        print(json.dumps(verify(), indent=2, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Dataset verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
