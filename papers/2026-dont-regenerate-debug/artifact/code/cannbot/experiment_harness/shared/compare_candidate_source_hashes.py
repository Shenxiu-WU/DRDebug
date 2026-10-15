#!/usr/bin/env python3
"""Compare final candidate source trees between two CANNBot-Debug manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SUFFIXES = {".asc", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".py"}
EXCLUDED = {"build", "build_out", "__pycache__"}


def source_digest(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED for part in relative.parts):
            continue
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        count += 1
    return digest.hexdigest(), count


def rows(path: Path) -> dict[str, dict[str, object]]:
    manifest = json.loads(path.read_text())
    result = {}
    for run in manifest.get("runs", []):
        digest, files = source_digest(Path(run["cwd"]) / "candidate_task")
        result[run["operator"]] = {"sha256": digest, "files": files}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    left, right = rows(args.left), rows(args.right)
    operators = sorted(set(left) | set(right))
    results = [
        {
            "operator": operator,
            "left": left.get(operator),
            "right": right.get(operator),
            "source_byte_identical": left.get(operator) == right.get(operator),
        }
        for operator in operators
    ]
    payload = {
        "left_manifest": str(args.left),
        "right_manifest": str(args.right),
        "operators": len(results),
        "source_byte_identical": sum(row["source_byte_identical"] for row in results),
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("operators", "source_byte_identical")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
