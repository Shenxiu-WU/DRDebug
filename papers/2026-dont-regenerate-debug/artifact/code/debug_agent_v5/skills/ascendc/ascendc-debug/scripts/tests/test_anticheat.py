from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "anticheat.py"


class TestReferenceHash(unittest.TestCase):
    def test_snapshot_and_verify_catches_model_py_change(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            task = Path(d)
            (task / "model.py").write_text("def ref(x):\n    return x\n", encoding="utf-8")

            snap = subprocess.run(
                [sys.executable, str(_SCRIPT), "snapshot", str(task), "--json"],
                capture_output=True,
                text=True,
                check=True,
            )
            snap_data = json.loads(snap.stdout)
            self.assertIn("model.py", snap_data["saved"])

            (task / "model.py").write_text(
                "def ref(x):\n    return x.cpu()\n", encoding="utf-8")
            verify = subprocess.run(
                [sys.executable, str(_SCRIPT), "verify", str(task), "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(verify.returncode, 1)
            data = json.loads(verify.stdout)
            self.assertEqual(data["verdict"], "CHEAT")
            self.assertTrue(any(r == "HASH_DIFF:model.py" for r in data["reasons"]))

    def test_snapshot_keeps_existing_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            task = Path(d)
            (task / "model.py").write_text("v1\n", encoding="utf-8")
            subprocess.run(
                [sys.executable, str(_SCRIPT), "snapshot", str(task), "--json"],
                capture_output=True,
                text=True,
                check=True,
            )

            (task / "model.py").write_text("v2\n", encoding="utf-8")
            snap = subprocess.run(
                [sys.executable, str(_SCRIPT), "snapshot", str(task), "--json"],
                capture_output=True,
                text=True,
                check=True,
            )
            snap_data = json.loads(snap.stdout)
            self.assertIn("model.py", snap_data["skipped"])

            verify = subprocess.run(
                [sys.executable, str(_SCRIPT), "verify", str(task), "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(verify.returncode, 1)


class TestCppEmbeddedPython(unittest.TestCase):
    def _verify(self, source: str) -> dict:
        with tempfile.TemporaryDirectory() as d:
            task = Path(d)
            (task / "kernel").mkdir()
            (task / "kernel" / "pybind11.cpp").write_text(source, encoding="utf-8")
            verify = subprocess.run(
                [sys.executable, str(_SCRIPT), "verify", str(task), "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            data = json.loads(verify.stdout)
            data["returncode"] = verify.returncode
            return data

    def test_detects_nms_pyrun_monkey_patch(self) -> None:
        data = self._verify(r'''
void launch() { aclrtLaunchKernel(nullptr, 0, nullptr, 0, nullptr); }
void patch() {
  const char *code =
      "import torch_npu as _tnpu\n"
      "_tnpu.npu_nms_v4 = _nms_v4_pad0\n";
  PyRun_SimpleString(code);
}
''')
        self.assertEqual(data["returncode"], 1)
        self.assertTrue(any("CPP_PYTHON_EXEC:PyRun_SimpleString" in reason
                            for reason in data["reasons"]))
        self.assertTrue(any("CPP_GLOBAL_MONKEY_PATCH:torch_npu.npu_nms_v4 assignment"
                            in reason for reason in data["reasons"]))

    def test_detects_batchmatmul_tensor_half_patch(self) -> None:
        data = self._verify(r'''
void launch() { aclrtLaunchKernel(nullptr, 0, nullptr, 0, nullptr); }
void patch() {
  pybind11::exec(R"PY(
import torch
torch.Tensor.half = _bm_patched_half
)PY");
}
''')
        self.assertEqual(data["returncode"], 1)
        self.assertTrue(any("CPP_PYTHON_EXEC:pybind11::exec" in reason
                            for reason in data["reasons"]))
        self.assertTrue(any("CPP_GLOBAL_MONKEY_PATCH:torch.Tensor.half assignment"
                            in reason for reason in data["reasons"]))

    def test_detects_framework_attr_assignment(self) -> None:
        data = self._verify(r'''
void launch() { aclrtLaunchKernel(nullptr, 0, nullptr, 0, nullptr); }
void patch() {
  pybind11::module_ torchNpu = pybind11::module_::import("torch_npu");
  torchNpu.attr("npu_nms_v4") = wrapper;
}
''')
        self.assertEqual(data["returncode"], 1)
        self.assertTrue(any("CPP_GLOBAL_MONKEY_PATCH" in reason
                            for reason in data["reasons"]))

    def test_normal_pybind_binding_remains_clean(self) -> None:
        data = self._verify(r'''
void launch() { aclrtLaunchKernel(nullptr, 0, nullptr, 0, nullptr); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_op", &launch, "AscendC operator");
}
''')
        self.assertEqual(data["returncode"], 0)
        self.assertEqual(data["verdict"], "CLEAN")


if __name__ == "__main__":
    unittest.main()
