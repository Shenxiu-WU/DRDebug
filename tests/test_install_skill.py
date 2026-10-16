"""Installer safety tests. Every destination is inside a disposable directory."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock


sys.dont_write_bytecode = True
INSTALLER = Path(__file__).absolute().parents[1] / "tools/install_skill.py"
SPEC = importlib.util.spec_from_file_location("drdebug_installer", INSTALLER)
assert SPEC is not None and SPEC.loader is not None
installer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = installer
SPEC.loader.exec_module(installer)


class InstallSkillTests(unittest.TestCase):
    def setUp(self) -> None:
        # Resolve the OS temporary directory itself (e.g. macOS /var -> /private/var).
        self.temporary = tempfile.TemporaryDirectory(
            prefix="drdebug-install-test-", dir=Path(tempfile.gettempdir()).resolve())
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "public repo with spaces"
        self.destination = self.directory / "skill parent with spaces"
        self.skill = self.root / "skills/drdebug"
        self.artifact = self.root / installer.ARTIFACT_REL
        self.engine = self.root / installer.ENGINE_REL
        self.write(self.skill / "SKILL.md", "---\nname: drdebug\n---\nPublic repair skill.\n")
        self.write(self.skill / "scripts/helper.py", "print('preflight only')\n", mode=0o755)
        self.write(self.engine / ".claude/agents/repair.md", "Frozen paper agent.\n")
        self.write(self.engine / "utils/check.py", "print('frozen check')\n", mode=0o755)
        self.write(self.engine / "skills/ascendc/ascendc-debug/engine/data.json", "{}\n")
        self.write(self.root / "LICENSE", "Test project license\n")
        self.write(self.root / "THIRD_PARTY_NOTICES.md", "Test source notices\n")
        self.write(self.artifact / "KNOWN_LIMITATIONS.md", "Original experiment assets are omitted.\n")
        self.write(self.artifact / "PUBLIC_RELEASE_CHANGES.json", json.dumps({
            "source_manifest_sha256": "0" * 64,
            "source_manifest_entries": 618,
            "changes": [],
        }) + "\n")
        # This source exists in the artifact, but must never be installed.
        self.write(self.artifact / "code/cannbot/private-looking-unused.txt", "not selected\n")
        self.write(self.engine / "archive_tasks/example/kernel/example.cpp", "// historical reference\n")
        self.write(self.engine / "archive_tasks/README.md", "Historical reference examples and supplemental metadata.\n")
        self.write(self.engine / "archive_tasks/PROVENANCE.json", '{"historical_commit": "example"}\n')
        self.write(self.engine / "archive_tasks/LICENSE-APACHE-2.0.txt", "Test Apache license text\n")
        self.refresh_manifest()

    @staticmethod
    def write(path: Path, text: str, mode: int = 0o644) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        path.chmod(mode)

    def refresh_manifest(self) -> None:
        entries = []
        for path in sorted(self.artifact.rglob("*")):
            if path.is_file() and path.name != "MANIFEST.sha256":
                entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                               f"{path.relative_to(self.artifact).as_posix()}\n")
        self.write(self.artifact / "MANIFEST.sha256", "".join(entries))

    def run_install(self, **kwargs):
        return installer.install(self.root, self.destination, **kwargs)

    def test_host_and_scope_destinations(self) -> None:
        parser = installer.parser()
        home = self.directory / "isolated home"
        with mock.patch.object(Path, "home", return_value=home):
            self.assertEqual(installer.destination_for(parser.parse_args([])), home / ".claude/skills")
            self.assertEqual(installer.destination_for(parser.parse_args(["--claude"])),
                             home / ".claude/skills")
            self.assertEqual(installer.destination_for(parser.parse_args(["--codex"])),
                             home / ".agents/skills")
        self.assertEqual(installer.destination_for(parser.parse_args(["--codex", "--project", str(home)])),
                         home / ".agents/skills")
        self.assertEqual(installer.destination_for(parser.parse_args(["--project"])),
                         Path(".claude/skills"))
        self.assertEqual(installer.destination_for(parser.parse_args(["--dest", str(home)])), home)

    def test_conflicting_cli_flags_are_rejected(self) -> None:
        for arguments in (["--codex", "--claude"], ["--project", ".", "--dest", "."]):
            with self.subTest(arguments=arguments), mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit) as caught:
                    installer.parser().parse_args(arguments)
                self.assertEqual(caught.exception.code, 2)

    def test_install_is_self_contained_and_preserves_relative_tree(self) -> None:
        outcome = self.run_install()
        self.assertEqual(outcome.status, "installed")
        target = self.destination / "drdebug"
        self.assertEqual((target / "SKILL.md").read_bytes(), (self.skill / "SKILL.md").read_bytes())
        frozen = target / "runtime/source/debug_agent_v5"
        for source in self.engine.rglob("*"):
            if source.is_file():
                self.assertEqual((frozen / source.relative_to(self.engine)).read_bytes(), source.read_bytes())
        self.assertFalse((target / "runtime/source/cannbot").exists())
        self.assertEqual((frozen / "archive_tasks/example/kernel/example.cpp").read_bytes(),
                         (self.engine / "archive_tasks/example/kernel/example.cpp").read_bytes())
        self.assertEqual((frozen / "archive_tasks/LICENSE-APACHE-2.0.txt").read_bytes(),
                         (self.engine / "archive_tasks/LICENSE-APACHE-2.0.txt").read_bytes())
        self.assertEqual((frozen / "archive_tasks/PROVENANCE.json").read_bytes(),
                         (self.engine / "archive_tasks/PROVENANCE.json").read_bytes())
        self.assertEqual(stat.S_IMODE((frozen / "utils/check.py").stat().st_mode), 0o755)
        self.assertTrue((target / "runtime/LICENSE").is_file())
        self.assertTrue((target / "runtime/THIRD_PARTY_NOTICES.md").is_file())
        self.assertEqual((target / "runtime/provenance/MANIFEST.sha256").read_bytes(),
                         (self.artifact / "MANIFEST.sha256").read_bytes())
        receipt = json.loads((target / "runtime/manifest.json").read_text())
        self.assertEqual(receipt["historical_source_manifest_sha256"], "0" * 64)
        for entry in receipt["files"]:
            self.assertEqual(hashlib.sha256((target / entry["path"]).read_bytes()).hexdigest(), entry["sha256"])
            self.assertFalse(entry["source"] and Path(entry["source"]).is_absolute())
        self.assertNotIn(str(self.directory), (target / "runtime/manifest.json").read_text())

    def test_dry_run_performs_no_writes(self) -> None:
        before = {str(path.relative_to(self.directory)) for path in self.directory.rglob("*")}
        outcome = self.run_install(dry_run=True)
        self.assertEqual(outcome.status, "would_install")
        self.assertFalse(self.destination.exists())
        self.assertEqual(before, {str(path.relative_to(self.directory)) for path in self.directory.rglob("*")})

    def test_identical_install_is_noop_even_with_force(self) -> None:
        self.run_install()
        marker = self.destination / "drdebug/SKILL.md"
        before = marker.stat().st_mtime_ns
        with mock.patch.object(installer, "write_plan", side_effect=AssertionError("must not write")):
            self.assertEqual(self.run_install().status, "unchanged")
            self.assertEqual(self.run_install(force=True).status, "unchanged")
        self.assertEqual(marker.stat().st_mtime_ns, before)
        self.assertEqual(list(self.destination.iterdir()), [self.destination / "drdebug"])

    def test_different_existing_install_is_preserved(self) -> None:
        marker = self.destination / "drdebug/local.txt"
        self.write(marker, "keep me\n")
        with self.assertRaises(installer.InstallError):
            self.run_install()
        self.assertEqual(marker.read_text(), "keep me\n")
        with self.assertRaises(installer.InstallError):
            self.run_install(dry_run=True)

    def test_force_retains_backup(self) -> None:
        marker = self.destination / "drdebug/local.txt"
        self.write(marker, "keep me\n")
        outcome = self.run_install(force=True)
        self.assertEqual(outcome.status, "replaced")
        self.assertIsNotNone(outcome.backup)
        self.assertEqual((outcome.backup / "local.txt").read_text(), "keep me\n")
        self.assertTrue((outcome.target / "SKILL.md").is_file())

    def test_dry_run_force_does_not_backup_or_replace(self) -> None:
        marker = self.destination / "drdebug/local.txt"
        self.write(marker, "keep me\n")
        self.assertEqual(self.run_install(force=True, dry_run=True).status, "would_replace")
        self.assertEqual(list(self.destination.iterdir()), [self.destination / "drdebug"])
        self.assertEqual(marker.read_text(), "keep me\n")

    def test_staging_failure_preserves_existing_install(self) -> None:
        marker = self.destination / "drdebug/local.txt"
        self.write(marker, "keep me\n")
        with mock.patch.object(installer, "write_plan", side_effect=OSError("injected disk failure")):
            with self.assertRaises(OSError):
                self.run_install(force=True)
        self.assertEqual(marker.read_text(), "keep me\n")
        self.assertEqual(list(self.destination.iterdir()), [self.destination / "drdebug"])

    def test_promotion_failure_restores_backup(self) -> None:
        marker = self.destination / "drdebug/local.txt"
        self.write(marker, "keep me\n")
        original = Path.rename

        def rename(path, target):
            if path.name.startswith(".drdebug.install-"):
                raise OSError("injected promotion failure")
            return original(path, target)

        with mock.patch.object(Path, "rename", new=rename):
            with self.assertRaises(OSError):
                self.run_install(force=True)
        self.assertEqual(marker.read_text(), "keep me\n")
        self.assertEqual(list(self.destination.iterdir()), [self.destination / "drdebug"])

    def test_post_promotion_failure_restores_backup(self) -> None:
        marker = self.destination / "drdebug/local.txt"
        self.write(marker, "keep me\n")
        original = installer.same_install

        def same(target, plan):
            if target == self.destination / "drdebug" and (target / "SKILL.md").exists():
                return False
            return original(target, plan)

        with mock.patch.object(installer, "same_install", side_effect=same):
            with self.assertRaises(installer.InstallError):
                self.run_install(force=True)
        self.assertEqual(marker.read_text(), "keep me\n")
        self.assertEqual(list(self.destination.iterdir()), [self.destination / "drdebug"])

    def test_source_symlinks_are_rejected(self) -> None:
        for path in (self.skill / "linked", self.engine / "linked"):
            with self.subTest(path=path):
                path.symlink_to(self.root / "LICENSE")
                try:
                    with self.assertRaises(installer.InstallError):
                        self.run_install()
                    self.assertFalse(self.destination.exists())
                finally:
                    path.unlink()

    def test_source_ancestor_symlink_is_rejected(self) -> None:
        alias = self.directory / "repo alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(installer.InstallError):
            installer.install(alias, self.destination)

    def test_destination_symlink_and_ancestor_symlink_are_rejected(self) -> None:
        self.destination.mkdir()
        other = self.directory / "other directory"
        other.mkdir()
        target = self.destination / "drdebug"
        target.symlink_to(other, target_is_directory=True)
        with self.assertRaises(installer.InstallError):
            self.run_install(force=True)
        self.assertTrue(target.is_symlink())
        target.unlink()
        ancestor = self.directory / "destination alias"
        ancestor.symlink_to(self.destination, target_is_directory=True)
        with self.assertRaises(installer.InstallError):
            installer.install(self.root, ancestor / "nested", force=True)

    def test_existing_nested_or_broken_symlink_is_rejected_even_with_force(self) -> None:
        self.destination.mkdir()
        target = self.destination / "drdebug"
        target.mkdir()
        link = target / "link"
        link.symlink_to(self.directory / "does-not-exist")
        with self.assertRaises(installer.InstallError):
            self.run_install(force=True)
        self.assertTrue(link.is_symlink())

    def test_recursive_or_frozen_artifact_destinations_are_rejected(self) -> None:
        for destination in (self.skill, self.skill.parent, self.engine, self.artifact / "new"):
            with self.subTest(destination=destination):
                with self.assertRaises(installer.InstallError):
                    installer.install(self.root, destination)

    def test_project_install_inside_checkout_is_allowed(self) -> None:
        destination = self.root / ".agents/skills"
        self.assertEqual(installer.install(self.root, destination).status, "installed")
        self.assertTrue((destination / "drdebug/SKILL.md").is_file())

    def test_changed_or_extra_frozen_source_is_rejected(self) -> None:
        changed = self.engine / "utils/check.py"
        self.write(changed, "changed after publication\n")
        with self.assertRaises(installer.InstallError):
            self.run_install()
        self.refresh_manifest()
        self.write(self.engine / "unlisted.py", "new private code must not leak\n")
        with self.assertRaises(installer.InstallError):
            self.run_install()
        self.assertFalse(self.destination.exists())

    def test_missing_source_and_changed_provenance_are_rejected(self) -> None:
        source = self.engine / "utils/check.py"
        source.unlink()
        with self.assertRaises(installer.InstallError):
            self.run_install()
        self.refresh_manifest()
        self.write(self.artifact / "KNOWN_LIMITATIONS.md", "modified after publication\n")
        with self.assertRaises(installer.InstallError):
            self.run_install()

    def test_reserved_runtime_and_special_files_are_rejected(self) -> None:
        self.write(self.skill / "runtime/prebuilt.txt", "unexpected runtime\n")
        with self.assertRaises(installer.InstallError):
            self.run_install()
        (self.skill / "runtime/prebuilt.txt").unlink()
        (self.skill / "runtime").rmdir()
        if hasattr(os, "mkfifo"):
            os.mkfifo(self.skill / "fifo")
            with self.assertRaises(installer.InstallError):
                self.run_install()

    def test_existing_regular_file_and_lock_are_preserved(self) -> None:
        self.write(self.destination / "drdebug", "not a skill directory\n")
        with self.assertRaises(installer.InstallError):
            self.run_install(force=True)
        (self.destination / "drdebug").unlink()
        lock = self.destination / ".drdebug.install.lock"
        lock.mkdir()
        with self.assertRaises(installer.InstallError):
            self.run_install()
        self.assertTrue(lock.is_dir())

    def test_manifest_paths_duplicates_and_hashes_are_validated(self) -> None:
        for content in ("a" * 64 + "  ../escape\n", "a" * 64 + "  /absolute\n", "invalid\n",
                        ("a" * 64 + "  same\n") * 2):
            with self.subTest(content=content):
                with self.assertRaises(installer.InstallError):
                    installer.artifact_hashes(content.encode())

    def test_empty_extra_directory_is_a_conflict(self) -> None:
        self.run_install()
        (self.destination / "drdebug/user-empty-directory").mkdir()
        with self.assertRaises(installer.InstallError):
            self.run_install()


if __name__ == "__main__":
    unittest.main()
