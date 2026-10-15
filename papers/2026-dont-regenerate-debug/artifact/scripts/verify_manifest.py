#!/usr/bin/env python3
"""Verify every packaged file against the release SHA-256 manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    if not MANIFEST.is_file():
        raise SystemExit("MANIFEST.sha256 is missing")

    expected: dict[str, str] = {}
    for number, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            checksum, relative = line.split("  ", 1)
        except ValueError as exc:
            raise SystemExit(f"Malformed manifest line {number}") from exc
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"Unsafe manifest path on line {number}: {relative}")
        if relative in expected:
            raise SystemExit(f"Duplicate manifest entry: {relative}")
        expected[relative] = checksum

    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path != MANIFEST
    }
    missing = sorted(set(expected) - actual)
    unlisted = sorted(actual - set(expected))
    if missing or unlisted:
        raise SystemExit(f"Manifest file-set mismatch: missing={missing}, unlisted={unlisted}")

    failures = []
    for relative, checksum in sorted(expected.items()):
        observed = digest(ROOT / relative)
        if observed != checksum:
            failures.append(relative)
    if failures:
        raise SystemExit(f"SHA-256 mismatch: {failures}")

    print(f"Verified {len(expected)} packaged files")


if __name__ == "__main__":
    main()
