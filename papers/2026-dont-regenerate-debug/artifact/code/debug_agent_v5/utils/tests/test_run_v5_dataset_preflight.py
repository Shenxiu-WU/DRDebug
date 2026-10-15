from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "run_v5_dataset_preflight.py"


def _load():
    spec = importlib.util.spec_from_file_location("v5_dataset_preflight", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V5DatasetPreflightTests(unittest.TestCase):
    def test_empty_mountpoint_is_allowed_but_nonempty_output_is_refused(
            self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "preflight"
            output.mkdir()
            module.prepare_output_dir(output)
            (output / "existing.json").write_text("{}\n")
            with self.assertRaises(SystemExit):
                module.prepare_output_dir(output)

    def test_prepare_is_deep_copy_and_removes_runtime(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source" / "level1" / "001_Foo"
            work = root / "preflight" / "work" / "level1" / "001_Foo"
            (source / "kernel" / "build").mkdir(parents=True)
            (source / ".verify_logs").mkdir()
            (source / "model.py").write_text("# source\n")
            (source / "kernel" / "foo.cpp").write_text("// kernel\n")
            (source / "kernel" / "build" / "stale.o").write_text("stale\n")
            (source / ".verify_logs" / "old.log").write_text("old\n")
            target = module.Target(rel="level1/001_Foo", source=source)

            module.prepare_isolated_task(target, work)

            self.assertTrue((work / "model.py").is_file())
            self.assertTrue((work / "kernel" / "foo.cpp").is_file())
            self.assertFalse((work / "kernel" / "build").exists())
            self.assertFalse((work / ".verify_logs").exists())
            (work / "model.py").write_text("# changed\n")
            self.assertEqual((source / "model.py").read_text(), "# source\n")

    def test_debuggable_failure_types_are_closed(self) -> None:
        module = _load()
        self.assertEqual(
            module.DEBUGGABLE_FAILURE_TYPES,
            {
                "build_failed", "import_failed", "runtime_error",
                "timeout", "precision_failed",
            },
        )

    def test_507015_is_rechecked_and_stable_precision_result_wins(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = module.Target(
                rel="level3/004_MatmulTransA",
                source=root / "source",
            )
            transient = {
                "task": target.rel,
                "failure_type": "runtime_error",
                "status": {
                    "exit_signal": "NPU_AICORE_EXCEPTION",
                    "stderr_tail": "runtime result = 507015",
                },
            }
            stable = {
                "task": target.rel,
                "failure_type": "precision_failed",
                "preflight_passed": True,
                "status": {"exit_signal": None},
            }
            with mock.patch.object(
                module, "evaluate_one", side_effect=[transient, stable]
            ) as evaluate:
                row = module.evaluate_with_transient_rechecks(
                    target=target,
                    output=root / "preflight",
                    repo_root=root,
                    container="v5_cann",
                    npu="6",
                    tilelang_env="/env.sh",
                    timeout=30,
                    transient_rechecks=2,
                )

            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(row["failure_type"], "precision_failed")
            self.assertTrue(row["stable_result_after_transient_recheck"])
            self.assertEqual(
                len(row["transient_infrastructure_rechecks"]), 1)
            self.assertTrue(
                (root / "preflight" / "transient_rechecks" / target.rel
                 / "attempt_0" / "result.json").is_file()
            )

    def test_repeated_507015_remains_runtime_error(self) -> None:
        module = _load()
        transient = {
            "task": "level3/004_MatmulTransA",
            "failure_type": "runtime_error",
            "status": {"stderr_tail": "507015"},
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            module, "evaluate_one", side_effect=[transient] * 3
        ):
            row = module.evaluate_with_transient_rechecks(
                target=module.Target(
                    rel="level3/004_MatmulTransA",
                    source=Path(tmp) / "source",
                ),
                output=Path(tmp) / "preflight",
                repo_root=Path(tmp),
                container="v5_cann",
                npu="6",
                tilelang_env="/env.sh",
                timeout=30,
                transient_rechecks=2,
            )
        self.assertEqual(row["failure_type"], "runtime_error")
        self.assertFalse(row["stable_result_after_transient_recheck"])
        self.assertEqual(len(row["transient_infrastructure_rechecks"]), 3)


if __name__ == "__main__":
    unittest.main()
