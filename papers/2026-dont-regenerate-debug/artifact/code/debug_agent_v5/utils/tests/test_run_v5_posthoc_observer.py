from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "run_v5_posthoc_observer.py"


def _load():
    spec = importlib.util.spec_from_file_location("v5_posthoc", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V5PosthocIsolationTests(unittest.TestCase):
    def test_prepare_uses_full_cases_without_mutating_treatment(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source" / "level1" / "001_Foo"
            treatment = root / "arm" / "tasks" / "level1" / "001_Foo"
            work = root / "posthoc" / "work" / "level1" / "001_Foo"
            (source / "kernel").mkdir(parents=True)
            (treatment / "kernel" / "build").mkdir(parents=True)
            (treatment / ".verify_logs").mkdir()
            (treatment / ".bench_baseline").mkdir()
            (source / "model.py").write_text("# frozen\n", encoding="utf-8")
            (source / "model_new_ascendc.py").write_text(
                "# original candidate\n", encoding="utf-8")
            (source / "1_Foo.json").write_text(
                '{"inputs":[1]}\n{"inputs":[2]}\n', encoding="utf-8")
            (source / "model.json").write_text(
                '{"inputs":[1]}\n{"inputs":[2]}\n', encoding="utf-8")
            (source / "1_Foo.json.bak").write_text(
                '{"inputs":[1]}\n{"inputs":[99]}\n', encoding="utf-8")
            (treatment / "model.py").write_text("# treatment\n", encoding="utf-8")
            (treatment / "model_new_ascendc.py").write_text(
                "# candidate\n", encoding="utf-8")
            (treatment / "1_Foo.json").write_text(
                '{"inputs":[1]}\n', encoding="utf-8")
            (treatment / "kernel" / "foo.cpp").write_text(
                "// candidate\n", encoding="utf-8")
            (treatment / "kernel" / "build" / "stale.o").write_text(
                "stale\n", encoding="utf-8")
            (treatment / ".verify_logs" / "old.stdout").write_text(
                "old\n", encoding="utf-8")
            (treatment / ".bench_baseline" / "stale.txt").write_text(
                "stale treatment baseline\n", encoding="utf-8")

            target = module.Target(
                rel="level1/001_Foo", treatment=treatment, source=source)
            result = module.prepare_isolated_task(target, work)

            self.assertTrue(result["available"])
            self.assertFalse(result["coverage_equivalent"])
            self.assertEqual(
                (work / "1_Foo.json").read_text(encoding="utf-8"),
                (source / "1_Foo.json.bak").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (work / "model.json").read_text(encoding="utf-8"),
                (source / "1_Foo.json.bak").read_text(encoding="utf-8"),
            )
            self.assertEqual(result["active_json_aliases"], ["model.json"])
            self.assertEqual(
                (treatment / "1_Foo.json").read_text(encoding="utf-8"),
                '{"inputs":[1]}\n',
            )
            self.assertEqual(
                (work / "model.py").read_text(encoding="utf-8"), "# frozen\n")
            self.assertFalse((work / "kernel" / "build").exists())
            self.assertFalse((work / ".verify_logs").exists())
            self.assertFalse((work / ".bench_baseline" / "stale.txt").exists())
            self.assertEqual(
                (work / ".bench_baseline" / "model_new_ascendc.py").read_text(),
                "# original candidate\n",
            )
            self.assertNotEqual(
                (work / ".bench_baseline" / "model_new_ascendc.py").read_text(),
                (work / "model_new_ascendc.py").read_text(),
            )

    def test_case_digest_ignores_formatting_and_order(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.json"
            second = root / "b.json"
            first.write_text(
                '{"inputs":[1],"meta":{"b":2,"a":1}}\n{"inputs":[2]}\n',
                encoding="utf-8",
            )
            second.write_text(
                '{"inputs": [2]}\n{"meta":{"a":1,"b":2},"inputs":[1]}\n',
                encoding="utf-8",
            )
            self.assertEqual(
                module._case_set_digest(first),
                module._case_set_digest(second),
            )

    def test_evaluator_clean_builds_before_verify(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = module.Target(
                rel="level1/001_Foo",
                treatment=root / "treatment",
                source=root / "source",
            )
            commands: list[str] = []

            def fake_docker_run(**kwargs):
                commands.append(kwargs["command"])
                if "build_ascendc.py" in kwargs["command"]:
                    return subprocess.CompletedProcess([], 1, "build out", "build err")
                return subprocess.CompletedProcess(
                    [], 0, '{"verdict":"CLEAN"}', "")

            with (
                mock.patch.object(
                    module,
                    "prepare_isolated_task",
                    return_value={"available": True},
                ),
                mock.patch.object(
                    module, "_docker_run", side_effect=fake_docker_run),
            ):
                row = module._evaluate_one(
                    target=target,
                    output=root / "posthoc",
                    repo_root=root,
                    container="v5_cann",
                    npu="3",
                    tilelang_env="/env.sh",
                    timeout=30,
                )

            self.assertEqual(commands[0], "python3 utils/build_ascendc.py {task} --clean")
            self.assertFalse(any(
                command == "python3 utils/verification_ascendc.py"
                for command in commands
            ))
            self.assertFalse(row["posthoc_clean_success"])
            self.assertEqual(
                row["verification"]["skipped_reason"], "clean_build_failed")

    def test_507015_is_retried_and_not_counted_as_completed(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = module.Target(
                rel="level1/001_Foo",
                treatment=root / "treatment",
                source=root / "source",
            )
            infrastructure = {
                "task": target.rel,
                "run_state": "infrastructure_error",
                "infrastructure_error": "507015",
                "posthoc_clean_success": False,
            }
            stable = {
                "task": target.rel,
                "run_state": "completed",
                "infrastructure_error": None,
                "posthoc_clean_success": False,
            }
            with (
                mock.patch.object(
                    module,
                    "_evaluate_one",
                    side_effect=[infrastructure, stable],
                ) as evaluate,
                mock.patch.object(module, "_archive_transient_attempt") as archive,
            ):
                row = module.evaluate_with_transient_rechecks(
                    target=target,
                    output=root / "posthoc",
                    repo_root=root,
                    container="v5_cann",
                    npu="3",
                    tilelang_env="/env.sh",
                    timeout=30,
                    transient_rechecks=2,
                )

            self.assertEqual(evaluate.call_count, 2)
            archive.assert_called_once()
            self.assertEqual(row["run_state"], "completed")
            self.assertTrue(row["stable_result_after_transient_recheck"])
            self.assertEqual(len(row["transient_infrastructure_rechecks"]), 1)

    def test_persistent_507015_blocks_posthoc_closure(self) -> None:
        module = _load()
        proc = subprocess.CompletedProcess(
            [],
            1,
            "RuntimeError: ACL stream synchronize failed, error code:507015\n"
            "Status : FAIL\n",
            "",
        )

        result = module._classify_verify(proc)

        self.assertFalse(result["objective_passed"])
        self.assertEqual(result["infrastructure_error"], "ACL stream synchronize failed")

    def test_persistent_507015_becomes_task_runtime_failure_after_rechecks(
        self,
    ) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = module.Target(
                rel="level1/001_Foo",
                treatment=root / "treatment",
                source=root / "source",
            )
            infrastructure = {
                "task": target.rel,
                "run_state": "infrastructure_error",
                "infrastructure_error": "ACL stream synchronize failed",
                "verification": {
                    "objective_passed": False,
                    "failure_type": "runtime_error",
                    "infrastructure_error": "ACL stream synchronize failed",
                    "signal": (
                        "ACL stream synchronize failed, error code:507015"
                    ),
                },
                "posthoc_clean_success": False,
            }
            with (
                mock.patch.object(
                    module,
                    "_evaluate_one",
                    side_effect=[
                        dict(infrastructure),
                        dict(infrastructure),
                        dict(infrastructure),
                    ],
                ) as evaluate,
                mock.patch.object(module, "_archive_transient_attempt"),
            ):
                row = module.evaluate_with_transient_rechecks(
                    target=target,
                    output=root / "posthoc",
                    repo_root=root,
                    container="v5_cann",
                    npu="3",
                    tilelang_env="/env.sh",
                    timeout=30,
                    transient_rechecks=2,
                )

            self.assertEqual(evaluate.call_count, 3)
            self.assertEqual(row["run_state"], "completed")
            self.assertIsNone(row["infrastructure_error"])
            self.assertTrue(row["persistent_runtime_failure"])
            self.assertEqual(
                row["persistent_runtime_failure_type"], "runtime_error")
            self.assertEqual(
                row["verification"]["failure_type"], "runtime_error")
            self.assertIsNone(
                row["verification"]["infrastructure_error"])


if __name__ == "__main__":
    unittest.main()
