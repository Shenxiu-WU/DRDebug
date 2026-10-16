#!/usr/bin/env python3
"""Install the public DRDebug skill and its frozen paper source, without executing it.

Python 3.10+, standard library only. Installation never configures a model, account,
shell, or device. All source files are read before any destination is changed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from uuid import uuid4


SKILL_NAME = "drdebug"
ARTIFACT_REL = Path("papers/2026-dont-regenerate-debug/artifact")
ENGINE_REL = ARTIFACT_REL / "code/debug_agent_v5"
REPOSITORY_URL = "https://github.com/Shenxiu-WU/DRDebug"
PAPER_URL = "https://arxiv.org/abs/2608.02712"


class InstallError(Exception):
    """An unsafe, incomplete, or conflicting installation request."""


@dataclass(frozen=True)
class FilePayload:
    data: bytes
    mode: int
    source: str | None


@dataclass(frozen=True)
class InstallResult:
    status: str
    target: Path
    file_count: int
    backup: Path | None = None


def checked_path(value: str | Path) -> Path:
    """Reject symlinks before normalizing '..', including existing ancestors."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise InstallError(f"Symbolic links are not allowed: {current}")
    return Path(os.path.abspath(path))


def source_files(root: Path) -> dict[str, FilePayload]:
    """Read a tree without following links or accepting special files."""
    root = checked_path(root)
    if not root.is_dir():
        raise InstallError(f"Required source directory is missing: {root}")
    files: dict[str, FilePayload] = {}

    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir()):
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise InstallError(f"Symbolic links are not allowed: {path}")
            if stat.S_ISDIR(mode):
                visit(path)
            elif stat.S_ISREG(mode):
                # Executability is portable; avoid copying writable/set-ID modes.
                files[path.relative_to(root).as_posix()] = FilePayload(
                    path.read_bytes(), 0o755 if mode & 0o111 else 0o644, None)
            else:
                raise InstallError(f"Only regular files and directories are allowed: {path}")

    visit(root)
    return files


def read_source(root: Path, relative: Path) -> FilePayload:
    path = checked_path(root / relative)
    if not path.is_file() or not stat.S_ISREG(path.lstat().st_mode):
        raise InstallError(f"Required source file is missing: {path}")
    mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
    return FilePayload(path.read_bytes(), mode, relative.as_posix())


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact_hashes(data: bytes, label: str = "artifact") -> dict[str, str]:
    hashes: dict[str, str] = {}
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise InstallError(f"The {label} manifest is not UTF-8.") from error
    for number, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise InstallError(f"Invalid {label} manifest entry on line {number}.")
        digest, name = match.groups()
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
            raise InstallError(f"Unsafe {label} manifest path on line {number}.")
        if name in hashes:
            raise InstallError(f"Duplicate {label} manifest entry on line {number}.")
        hashes[name] = digest
    return hashes


def build_plan(repo_root: Path) -> dict[str, FilePayload]:
    """Build a deterministic, self-contained payload from public sources only."""
    root = checked_path(repo_root)
    skill = source_files(root / "skills" / SKILL_NAME)
    if "SKILL.md" not in skill:
        raise InstallError("skills/drdebug/SKILL.md is required.")
    if any(name == "runtime" or name.startswith("runtime/") for name in skill):
        raise InstallError("skills/drdebug/runtime is reserved for the frozen runtime payload.")
    plan = {
        name: FilePayload(file.data, file.mode, f"skills/{SKILL_NAME}/{name}")
        for name, file in skill.items()
    }
    manifest = read_source(root, ARTIFACT_REL / "MANIFEST.sha256")
    hashes = artifact_hashes(manifest.data)
    engine = source_files(root / ENGINE_REL)
    prefix = "code/debug_agent_v5/"
    expected = {name[len(prefix):] for name in hashes if name.startswith(prefix)}
    if set(engine) != expected:
        raise InstallError("Frozen Debug-Agent source files differ from the artifact manifest "
                           "(missing or extra files); use a clean public release.")
    if not expected:
        raise InstallError("The artifact manifest contains no Debug-Agent source files.")
    for name, file in engine.items():
        if sha256(file.data) != hashes[prefix + name]:
            raise InstallError(f"Frozen source hash mismatch: {prefix}{name}")
        plan[f"runtime/source/debug_agent_v5/{name}"] = FilePayload(
            file.data, file.mode, (ENGINE_REL / name).as_posix())

    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        plan[f"runtime/{name}"] = read_source(root, Path(name))
    plan["runtime/provenance/MANIFEST.sha256"] = manifest
    for name in ("PUBLIC_RELEASE_CHANGES.json", "KNOWN_LIMITATIONS.md"):
        file = read_source(root, ARTIFACT_REL / name)
        if hashes.get(name) != sha256(file.data):
            raise InstallError(f"Frozen provenance hash mismatch: {name}")
        plan[f"runtime/provenance/{name}"] = file
    try:
        changes = json.loads(plan["runtime/provenance/PUBLIC_RELEASE_CHANGES.json"].data)
    except (ValueError, UnicodeDecodeError) as error:
        raise InstallError("PUBLIC_RELEASE_CHANGES.json is not valid JSON.") from error
    if not isinstance(changes, dict):
        raise InstallError("PUBLIC_RELEASE_CHANGES.json must contain an object.")

    provenance = (
        "# Installed DRDebug runtime\n\n"
        f"Source: {REPOSITORY_URL}/tree/main/{ENGINE_REL.as_posix()}\n"
        f"Paper: {PAPER_URL}\n\n"
        "source/debug_agent_v5 is copied byte-for-byte from the public frozen paper "
        "artifact, including its documented historical archive_tasks additions and "
        "their license/provenance files. No CANNBot harness, benchmark dataset, "
        "model credentials, or post-paper private code is installed.\n\n"
        "provenance/MANIFEST.sha256 is the public artifact's complete manifest, including "
        "files not installed here. PUBLIC_RELEASE_CHANGES.json records the public edits "
        "and the hash of the earlier 618-entry source manifest; that earlier manifest "
        "is not supplied by this installer. manifest.json lists the installed source "
        "paths, hashes and executable modes.\n\n"
        "The historical experiment launchers retain dependencies on omitted experiment "
        "assets. Installing a skill does not validate the NPU environment or reproduce "
        "the paper experiments. Read provenance/KNOWN_LIMITATIONS.md and the skill's "
        "input/evaluation contract before execution.\n\n"
        "LICENSE and THIRD_PARTY_NOTICES.md are retained from the repository. The latter "
        "also describes third-party material elsewhere in the full repository; its "
        "repository-relative links refer to that original repository, not this installed "
        "subset. File-level notices remain authoritative.\n"
    )
    plan["runtime/PROVENANCE.md"] = FilePayload(provenance.encode(), 0o644, None)
    entries = [
        {"path": name, "source": file.source, "sha256": sha256(file.data),
         "mode": f"{file.mode:04o}", "size": len(file.data)}
        for name, file in sorted(plan.items())
    ]
    identity = sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode())
    receipt = {
        "schema_version": 1,
        "skill": SKILL_NAME,
        "repository": REPOSITORY_URL,
        "paper": PAPER_URL,
        "bundle_sha256": identity,
        "artifact_manifest_sha256": sha256(manifest.data),
        "historical_source_manifest_sha256": changes.get("source_manifest_sha256"),
        "scope": "Public skill, frozen paper Debug-Agent V5 source and historical archive_tasks references; no CANNBot harness.",
        "files": entries,
    }
    plan["runtime/manifest.json"] = FilePayload(
        (json.dumps(receipt, indent=2, ensure_ascii=False) + "\n").encode(), 0o644, None)
    return plan


def protect_sources(root: Path, target: Path) -> None:
    # Project-scoped installs inside the checkout are valid, but never inside or
    # over the source skill, frozen artifact, or provenance/license inputs.
    protected = [root / "skills" / SKILL_NAME, root / ARTIFACT_REL,
                 root / "LICENSE", root / "THIRD_PARTY_NOTICES.md"]
    for source in protected:
        if target.is_relative_to(source) or source.is_relative_to(target):
            raise InstallError(f"Destination overlaps a protected source: {source}")


def same_install(target: Path, plan: dict[str, FilePayload]) -> bool:
    actual = source_files(target)
    if actual.keys() != plan.keys():
        return False
    if any(actual[name].data != file.data or actual[name].mode != file.mode
           for name, file in plan.items()):
        return False
    expected_dirs = {parent.as_posix() for name in plan
                     for parent in PurePosixPath(name).parents if str(parent) != "."}
    actual_dirs = {path.relative_to(target).as_posix()
                   for path in target.rglob("*") if path.is_dir()}
    return expected_dirs == actual_dirs


def write_plan(stage: Path, plan: dict[str, FilePayload]) -> None:
    for name, file in sorted(plan.items()):
        path = stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file.data)
        path.chmod(file.mode)


def install(repo_root: Path, destination: Path, *, dry_run: bool = False,
            force: bool = False) -> InstallResult:
    root = checked_path(repo_root)
    parent = checked_path(destination)
    target = checked_path(parent / SKILL_NAME)
    protect_sources(root, target)
    plan = build_plan(root)
    exists = target.exists()
    if exists and not target.is_dir():
        raise InstallError(f"The destination is not a directory: {target}")
    if exists and same_install(target, plan):
        return InstallResult("unchanged", target, len(plan))
    if exists and not force:
        raise InstallError(f"An existing installation differs: {target}. "
                           "Use --force to retain a backup and replace it.")
    if dry_run:
        return InstallResult("would_replace" if exists else "would_install", target, len(plan))

    parent.mkdir(parents=True, exist_ok=True)
    checked_path(parent)
    lock = parent / f".{SKILL_NAME}.install.lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise InstallError(f"Another install may be running; lock exists: {lock}") from error
    stage: Path | None = None
    backup: Path | None = None
    displaced: Path | None = None
    promoted = False
    try:
        # Recheck under the lock so two installers cannot overwrite each other.
        checked_path(target)
        if target.exists():
            if not target.is_dir():
                raise InstallError(f"The destination is not a directory: {target}")
            if same_install(target, plan):
                return InstallResult("unchanged", target, len(plan))
            if not force:
                raise InstallError(f"An installation appeared at {target}; refusing to overwrite it.")
        stage = Path(tempfile.mkdtemp(prefix=f".{SKILL_NAME}.install-", dir=parent))
        write_plan(stage, plan)
        if not same_install(stage, plan):
            raise InstallError("Staged installation failed its content check.")
        if target.exists():
            backup = parent / f"{SKILL_NAME}.backup-{uuid4().hex}"
            target.rename(backup)
        stage.rename(target)
        promoted = True
        if not same_install(target, plan):
            raise InstallError("Installed content failed its integrity check.")
        return InstallResult("replaced" if backup else "installed", target, len(plan), backup)
    except BaseException as error:
        try:
            if promoted:
                displaced = parent / f".{SKILL_NAME}.failed-{uuid4().hex}"
                target.rename(displaced)
            if backup is not None:
                if target.exists() or target.is_symlink():
                    raise InstallError(f"Cannot restore automatically; backup retained at {backup}")
                backup.rename(target)
                backup = None
        except BaseException as restore_error:
            raise InstallError(f"Installation failed and recovery needs attention: {restore_error}") from error
        raise
    finally:
        for temporary in (stage, displaced):
            if temporary is not None and temporary.exists():
                shutil.rmtree(temporary)
        lock.rmdir()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    host = result.add_mutually_exclusive_group()
    host.add_argument("--codex", dest="host", action="store_const", const="codex",
                      help="Install for Codex (default user scope: ~/.agents/skills).")
    host.add_argument("--claude", dest="host", action="store_const", const="claude",
                      help="Install for Claude Code (default: ~/.claude/skills).")
    result.set_defaults(host="claude")
    location = result.add_mutually_exclusive_group()
    location.add_argument("--project", nargs="?", const=".", metavar="PATH",
                          help="Install in PATH's host-specific project skill directory (default: cwd).")
    location.add_argument("--dest", type=Path, metavar="DIR",
                          help="Use DIR as the skill parent; the skill is installed as DIR/drdebug.")
    result.add_argument("--dry-run", action="store_true", help="Validate and show the plan without writing.")
    result.add_argument("--force", action="store_true", help="Back up a differing installation before replacing it.")
    return result


def destination_for(arguments: argparse.Namespace) -> Path:
    if arguments.dest is not None:
        return arguments.dest
    base = Path(arguments.project) if arguments.project is not None else Path.home()
    return base / (".agents" if arguments.host == "codex" else ".claude") / "skills"


def main(argv: list[str] | None = None) -> int:
    if sys.version_info < (3, 10):
        print("DRDebug installation requires Python 3.10 or newer.", file=sys.stderr)
        return 1
    arguments = parser().parse_args(argv)
    try:
        outcome = install(Path(__file__).absolute().parents[1], destination_for(arguments),
                          dry_run=arguments.dry_run, force=arguments.force)
    except (InstallError, OSError) as error:
        print(f"DRDebug installation failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Installation interrupted; the previous installation was restored if needed.", file=sys.stderr)
        return 130
    print(f"DRDebug: {outcome.status.replace('_', ' ')} ({outcome.file_count} files)")
    print(f"Target: {outcome.target}")
    if arguments.dry_run:
        print("Dry run: no files were written.")
    elif outcome.status != "unchanged":
        if outcome.backup:
            print(f"Previous installation retained at: {outcome.backup}")
        print("Installed only public skill files and the frozen paper source; no model or device was run.")
    print("Next: ask your agent to use drdebug for a near-miss Ascend C kernel, starting with preflight.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
