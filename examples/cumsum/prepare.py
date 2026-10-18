#!/usr/bin/env python3
"""Check and stage public Cumsum example sources; never import task code.

Python 3.10+, standard library only. This command does not compile, execute an
NPU kernel, contact a model, or change the frozen paper artifact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

EXAMPLE = Path(__file__).resolve().parent
REPO = EXAMPLE.parents[1]
PAPER_TASK = "papers/2026-dont-regenerate-debug/artifact/inputs/tasks/level1/005_Cumsum"
NATIVE_FILES = {
    "cumsum_bf16.cpp", "cumsum_fp16.cpp", "cumsum_fp32.cpp",
    "cumsum_kernel.h", "cumsum_tiling.h", "kernel_common.h",
    "pybind11.cpp", "vector_tile.h",
}
TASK_FILES = {f"kernel/{name}" for name in NATIVE_FILES} | {
    "model.py", "model_new_ascendc.py", "5_Cumsum.json", "5_Cumsum.json.bak",
    "LICENSE", "LICENSE-APACHE-2.0.txt", "SOURCE_NOTICE.md",
}


class ExampleError(ValueError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_files(root, expected):
    """Read the exact pinned regular-file tree once, rejecting links/extras."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ExampleError(f"Missing or symlinked source directory: {root}")
    paths = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ExampleError(f"Symlink source is not supported: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ExampleError(f"Non-regular source is not supported: {path}")
        paths[path.relative_to(root).as_posix()] = path
    if set(paths) != set(expected):
        raise ExampleError(f"Source file set differs from example.json: {root}")
    result = {}
    for name, path in paths.items():
        record = expected[name]
        if (not isinstance(record, dict) or set(record) != {"sha256", "size"}
                or not isinstance(record["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
                or type(record["size"]) is not int or record["size"] <= 0):
            raise ExampleError(f"Invalid hash record: {name}")
        data = path.read_bytes()
        if digest(data) != record["sha256"] or len(data) != record["size"]:
            raise ExampleError(f"Source hash or size mismatch: {path}")
        result[name] = data
    return result


def verified_assets(example=EXAMPLE, repo=REPO):
    example, repo = Path(example), Path(repo)
    manifest_path = example / "example.json"
    if manifest_path.is_symlink():
        raise ExampleError("example.json must not be a symlink")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("schema_version") != 1 or manifest.get("op_name") != "5_Cumsum"
            or manifest.get("original_task") != PAPER_TASK
            or set(manifest.get("original_task_files", {})) != TASK_FILES
            or set(manifest.get("repaired_files", {})) != NATIVE_FILES):
        raise ExampleError("Unknown or incomplete Cumsum example manifest")
    # Fixed path sets are checked before using the manifest's keys as paths.
    original = checked_files(repo / PAPER_TASK, manifest["original_task_files"])
    repaired = checked_files(example / "repaired/kernel", manifest["repaired_files"])
    return manifest, manifest_bytes, original, repaired


def stage(mode, output, *, example=EXAMPLE, repo=REPO):
    if mode not in {"live", "replay"}:
        raise ExampleError("mode must be live or replay")
    example, repo = Path(example).resolve(), Path(repo).resolve()
    manifest, manifest_bytes, original, repaired = verified_assets(example, repo)
    output_input = Path(output).absolute()
    output = output_input.resolve()
    if output_input.is_symlink() or output.exists():
        raise ExampleError("Output already exists; choose a new task directory")
    for source in (example, repo / PAPER_TASK):
        if output == source or source in output.parents or output in source.parents:
            raise ExampleError("Output and example source directories must not contain each other")
    payload = dict(original)
    if mode == "replay":
        payload.update({f"kernel/{name}": data for name, data in repaired.items()})
    origin = {
        "schema_version": 1,
        "example": "cumsum-precision-repair",
        "mode": mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "example_manifest_sha256": digest(manifest_bytes),
        "original_task": PAPER_TASK,
        "kernel_origin": "original paper near-miss input" if mode == "live" else "saved precision-repaired candidate",
        "files_sha256": {name: digest(data) for name, data in sorted(payload.items())},
        "original_notice_scope": "SOURCE_NOTICE.md is preserved byte-for-byte and describes the original paper input; kernel_origin above identifies whether this staged task uses the repaired example instead.",
        "validation_claim": "None from staging. Run the public adapter's verify command on your execution machine.",
    }
    # Source bytes have all been verified before creating any output. mkdir is
    # exclusive, so neither an old run nor an existing symlink is overwritten.
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, data in payload.items():
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(data)
        with (output / "example-origin.json").open("x", encoding="utf-8") as handle:
            json.dump(origin, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except BaseException:
        # Only this invocation's newly created directory is removed; an
        # existing task fails before this point and is never touched.
        shutil.rmtree(output)
        raise
    return {"staged": True, "task": str(output), "mode": mode,
            "op_name": "5_Cumsum", "example_manifest_sha256": digest(manifest_bytes),
            "provenance": str(output / "example-origin.json"),
            "compact_cases": manifest["historical_validation"]["compact_cases"],
            "full_cases": manifest["historical_validation"]["full_cases"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check pinned public source bytes without writing")
    parser.add_argument("--mode", choices=("live", "replay"), help="original failing input or saved repaired kernel")
    parser.add_argument("--output", help="new task directory; an existing path is never overwritten")
    args = parser.parse_args(argv)
    if args.check and (args.mode or args.output):
        parser.error("--check cannot be combined with --mode or --output")
    if not args.check and (not args.mode or not args.output):
        parser.error("provide --mode and --output, or use --check")
    try:
        if args.check:
            _, manifest_bytes, original, repaired = verified_assets()
            result = {"verified": True, "original_files": len(original), "repaired_files": len(repaired),
                      "example_manifest_sha256": digest(manifest_bytes), "execution_performed": False}
        else:
            result = stage(args.mode, args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (ExampleError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Cumsum example: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
