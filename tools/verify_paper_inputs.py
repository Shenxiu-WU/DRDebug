#!/usr/bin/env python3
"""Read-only verification of the restored 27-task paper inputs (Python 3.10+)."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "papers/2026-dont-regenerate-debug/artifact"
DEFAULT_INPUTS = ARTIFACT / "inputs"
DATASET = ARTIFACT.parent / "dataset"
COHORT = ARTIFACT / "manifests/frozen_27_operators.csv"
HISTORICAL_MANIFEST_SHA256 = "d905ab31d47b0074cc5bb3718eea92ebb4fc20a8ea0de76a451d55524dde1f45"
ROOT_METADATA = {"README.md", "PROVENANCE.json", "FROZEN_INPUT_SHA256.json", "NOTICE",
                 "LICENSE", "LICENSE-APACHE-2.0.txt"}
TASK_METADATA = {"LICENSE", "LICENSE-APACHE-2.0.txt", "SOURCE_NOTICE.md"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_file(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    require(not path.is_absolute() and path.as_posix() == relative
            and ".." not in path.parts and "\\" not in relative,
            "Unsafe input-package path")
    target = root / relative
    for parent in [target, *target.parents]:
        require(not parent.is_symlink(), f"Symlinked input-package path: {relative}")
        if parent == root:
            break
    require(target.is_file() and stat.S_ISREG(target.lstat().st_mode),
            f"Missing or non-regular input file: {relative}")
    require(target.resolve().is_relative_to(root.resolve()), "Input file escapes its package")
    return target


def read_cases(path: Path) -> tuple[int, str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    require(all(isinstance(row, dict) and "inputs" in row for row in rows),
            "Invalid input-case row")
    normalized = sorted(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")) for row in rows)
    return len(rows), digest("".join(row + "\n" for row in normalized).encode("utf-8"))


def verify(inputs: Path = DEFAULT_INPUTS, *, dataset: Path = DATASET,
           cohort_path: Path = COHORT) -> dict:
    inputs = inputs.absolute()
    require(inputs.is_dir() and not inputs.is_symlink(), "Input package directory is required")
    checksums = {}
    for number, line in enumerate(safe_file(inputs, "MANIFEST.sha256").read_text().splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"Malformed input manifest entry on line {number}")
        sha, relative = match.groups()
        require(relative not in checksums, "Duplicate input manifest entry")
        require(digest(safe_file(inputs, relative).read_bytes()) == sha,
                f"Input checksum mismatch: {relative}")
        checksums[relative] = sha
    actual = set()
    ignored = []
    for path in inputs.rglob("*"):
        require(not path.is_symlink(), "Input package contains a symlink")
        mode = path.lstat().st_mode
        require(stat.S_ISREG(mode) or stat.S_ISDIR(mode), "Input package contains a special file")
        if path.is_file():
            relative = path.relative_to(inputs)
            if (path.name == ".DS_Store" or "__pycache__" in relative.parts[:-1]
                    or path.suffix == ".pyc"):
                ignored.append(relative.as_posix())
                continue
            actual.add(relative.as_posix())
    require(actual == set(checksums) | {"MANIFEST.sha256"},
            "Unexpected or missing input-package files")
    provenance = json.loads(safe_file(inputs, "PROVENANCE.json").read_text())
    frozen = json.loads(safe_file(inputs, "FROZEN_INPUT_SHA256.json").read_text())
    require(provenance["schema_version"] == frozen["schema_version"] == 1,
            "Unsupported input provenance schema")
    require(provenance["historical_frozen_input_manifest"]["sha256"]
            == frozen["source_manifest_sha256"] == HISTORICAL_MANIFEST_SHA256,
            "Historical frozen input-manifest identity differs")
    require(cohort_path.is_file() and not cohort_path.is_symlink(), "Frozen cohort CSV is required")
    require(digest(cohort_path.read_bytes()) == provenance["frozen_cohort"]["sha256"],
            "Frozen cohort CSV checksum differs")
    with cohort_path.open(newline="", encoding="utf-8") as stream:
        cohort_rows = list(csv.DictReader(stream))
    cohort = {row["operator_id"]: row for row in cohort_rows}
    operators = provenance["operators"]
    require(len(cohort_rows) == len(cohort) == len(operators) == 27,
            "Expected exactly 27 input tasks")
    require({row["operator_id"] for row in operators} == set(cohort),
            "Input tasks differ from the paper cohort")
    require(digest(safe_file(dataset, "MANIFEST.sha256").read_bytes())
            == provenance["dataset_companion"]["manifest_sha256"],
            "Dataset companion manifest differs from the recorded source")
    require(digest(safe_file(dataset, "PROVENANCE.json").read_bytes())
            == provenance["dataset_companion"]["provenance_sha256"],
            "Dataset companion provenance differs from the recorded source")
    dataset_provenance = json.loads(safe_file(dataset, "PROVENANCE.json").read_text())
    dataset_operators = {row["operator_id"]: row for row in dataset_provenance["operators"]}
    source_hashes = frozen["files"]
    require(len(source_hashes) == 289, "Expected 289 original task files")
    task_files = set()
    metadata = set(ROOT_METADATA)
    counts = {"operators": 27, "wrapper_files": 0, "kernel_files": 0,
              "reference_files": 0, "compact_files": 0, "full_files": 0,
              "compact_cases": 0, "full_cases": 0}
    for operator in operators:
        identifier = operator["operator_id"]
        require(re.fullmatch(r"level[123567]/[0-9]{3}_[A-Za-z0-9_]+", identifier) is not None,
                "Invalid operator identifier")
        prefix = f"tasks/{identifier}/"
        origin = operator["original_generation"]
        require(origin["verify_failure_type"] == "precision_failed"
                and origin["anticheat_verdict"] == "CLEAN",
                "Historical input eligibility differs")
        records = operator["files"]
        record_paths = [record["path"] for record in records]
        require(len(record_paths) == len(set(record_paths)), "Duplicate task source record")
        require(set(record_paths) == {name for name in source_hashes if name.startswith(prefix)},
                f"Incomplete original source expectations: {identifier}")
        kernels = 0
        for record in records:
            relative = record["path"]
            role = record["role"]
            require(relative.startswith(prefix) and relative == record["historical_frozen_relative_path"],
                    "Original source task identity differs")
            content = safe_file(inputs, relative).read_bytes()
            require(digest(content) == record["sha256"] == source_hashes[relative],
                    f"Original frozen source checksum differs: {relative}")
            task_files.add(relative)
            if role == "kernel":
                require(relative.startswith(prefix + "kernel/"), "Invalid kernel source path")
                kernels += 1
            elif role == "wrapper":
                require(relative == prefix + "model_new_ascendc.py", "Invalid wrapper path")
                ast.parse(content, filename=relative)
            elif role in {"reference", "compact", "full"}:
                source = dataset_operators[identifier][role]
                require(relative == source["path"] and record["sha256"]
                        == source["sha256"] == source["frozen_byte_sha256"],
                        "Task data differs from the original dataset expectation")
                require(content == safe_file(dataset, source["path"]).read_bytes(),
                        "Task data differs from the dataset companion")
                if role == "reference":
                    ast.parse(content, filename=relative)
                else:
                    case_count, case_hash = read_cases(inputs / relative)
                    expected = cohort[identifier]
                    require(case_count == int(expected[f"{role}_cases"])
                            and case_hash == expected[f"{role}_case_set_sha256"],
                            f"Original case-set identity differs: {identifier}/{role}")
                    counts[f"{role}_cases"] += case_count
            else:
                raise ValueError("Unapproved original task file role")
            counts[f"{role}_files"] += 1
        require(kernels > 0, f"Task has no original kernel sources: {identifier}")
        require({record["role"] for record in records} == {"wrapper", "kernel", "reference", "compact", "full"},
                f"Task does not contain the complete prepare input: {identifier}")
        require(set(operator["supplemental_task_metadata"]) == TASK_METADATA,
                "Incomplete task redistribution metadata")
        for name in TASK_METADATA:
            relative = prefix + name
            metadata.add(relative)
            if name in {"LICENSE", "LICENSE-APACHE-2.0.txt"}:
                require(safe_file(inputs, relative).read_bytes() == safe_file(inputs, name).read_bytes(),
                        "Task license differs from the supplied license text")
    require(task_files == set(source_hashes), "Unapproved selected frozen source expectation")
    require(set(checksums) == task_files | metadata, "Unapproved input-package file")
    require(counts == {"operators": 27, "wrapper_files": 27, "kernel_files": 181,
                       "reference_files": 27, "compact_files": 27, "full_files": 27,
                       "compact_cases": 347, "full_cases": 1134},
            "Restored input counts differ from the release scope")
    return {**counts, "original_task_files_byte_exact": len(task_files),
            "supplemental_task_metadata_files": len(TASK_METADATA) * 27,
            "checksum_files": len(checksums), "ignored_local_metadata": sorted(ignored),
            "task_execution": "none"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS,
                        help="Restored input-package directory (default: the paper artifact inputs).")
    arguments = parser.parse_args()
    try:
        print(json.dumps(verify(arguments.inputs), indent=2, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as error:
        print(f"Paper input verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
