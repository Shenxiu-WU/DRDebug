from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "verify_v5_no_workers.py"


def _load():
    spec = importlib.util.spec_from_file_location("v5_no_workers", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class V5NoWorkersTests(unittest.TestCase):
    @unittest.skipUnless(Path("/proc").is_dir(), "requires Linux /proc")
    def test_detects_process_with_task_cwd(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as tmp:
            tasks = Path(tmp) / "tasks"
            tasks.mkdir()
            proc = subprocess.Popen(["sleep", "10"], cwd=tasks)
            try:
                for _ in range(20):
                    matches = module.matching_processes(tasks)
                    if any(row["pid"] == proc.pid for row in matches):
                        break
                    time.sleep(0.05)
                self.assertTrue(any(
                    row["pid"] == proc.pid for row in matches
                ))
            finally:
                proc.terminate()
                proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
