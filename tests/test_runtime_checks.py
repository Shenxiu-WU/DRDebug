"""CPU contract tests; fake packages exercise startup without NPU/model calls."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import types
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "skills/drdebug/scripts/runtime_checks.py"
SPEC = importlib.util.spec_from_file_location("drdebug_runtime_checks_tests", SCRIPT)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="drdebug-runtime-test-")
        self.root = Path(self.temp.name).resolve()
        self.packages = self.root / "packages"
        self.packages.mkdir()
        self.fake_torch()
        (self.packages / "torch_npu.py").write_text("__version__='2.9.0.post6'\n")
        self.adapter = self.root / "adapter"
        self.receipts = self.root / "adapter-receipts"
        self.base_env = dict(os.environ)
        for key in (runtime.MODE_VARIABLE, runtime.CONFIG_VARIABLE, runtime.CONFIG_HASH_VARIABLE, runtime.RECEIPT_VARIABLE):
            self.base_env.pop(key, None)
        self.base_env["PYTHONPATH"] = os.pathsep.join((str(self.packages), str(SCRIPT.parent)))

    def tearDown(self):
        self.temp.cleanup()

    def fake_torch(self, *, refuses=False):
        source = """__version__='2.9.0'
_enabled=False
_warn=True
def use_deterministic_algorithms(value,*,warn_only=False):
 global _enabled,_warn
 _enabled=bool(value)
 _warn=bool(warn_only)
def are_deterministic_algorithms_enabled(): return _enabled
def is_deterministic_algorithms_warn_only_enabled(): return _warn
"""
        if refuses:
            source += "\ndef are_deterministic_algorithms_enabled(): return False\n"
        (self.packages / "torch.py").write_text(source)

    def stage(self, mode="on"):
        return runtime.configure_runtime_environment(self.base_env, self.adapter, mode, self.receipts)

    def child(self, env, code):
        return subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, timeout=15, check=False)

    def test_on_and_off_are_explicit_and_applied_before_user_entrypoint(self):
        for mode in ("on", "off"):
            directory = self.root / ("adapter-" + mode)
            receipts = self.root / ("receipts-" + mode)
            env, contract = runtime.configure_runtime_environment(self.base_env, directory, mode, receipts)
            proc = self.child(env, "import runtime_checks,json;print(json.dumps(runtime_checks.startup_evidence()))")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(proc.stdout)
            self.assertEqual(report["mode"], mode)
            self.assertIs(report["deterministic_enabled"], mode == "on")
            self.assertIs(report["warn_only"], False)
            self.assertTrue(report["verified_current_flags"])
            self.assertEqual(report["hook_sha256"], contract["hook_sha256"])
            self.assertEqual(len(list(receipts.glob("*.json"))), 1)
            self.assertEqual(set(path.name for path in directory.iterdir()), {"startup.json", "sitecustomize.py"})

    def test_stage_is_idempotent_and_execution_does_not_rewrite_source(self):
        env, contract = self.stage()
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.adapter.iterdir()}
        again, same_contract = self.stage()
        rebuilt = runtime.execution_environment(self.base_env, self.adapter, "on", self.receipts)
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.adapter.iterdir()}
        self.assertEqual(before, after)
        self.assertEqual(env, again)
        self.assertEqual(env, rebuilt)
        self.assertEqual(contract, same_contract)
        self.assertEqual(env["PATH"].split(os.pathsep)[0], str(Path(sys.executable).absolute().parent))
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(self.adapter))

    def test_wrong_flags_fail_closed_before_any_user_code(self):
        self.fake_torch(refuses=True)
        env, _ = self.stage()
        sentinel = self.root / "user-entrypoint-ran"
        proc = self.child(env, f"from pathlib import Path;Path({str(sentinel)!r}).write_text('bad')")
        self.assertEqual(proc.returncode, 78)
        self.assertIn("refusing interpreter continuation", proc.stderr)
        self.assertFalse(sentinel.exists())
        self.assertFalse(list(self.receipts.glob("*.json")))

    def test_contract_or_hook_changes_cannot_silently_disable_startup(self):
        env, _ = self.stage()
        for path in (self.adapter / "startup.json", self.adapter / "sitecustomize.py"):
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            proc = self.child(env, "print('USER-CODE')")
            self.assertEqual(proc.returncode, 78, (str(path), proc.stdout, proc.stderr))
            self.assertNotIn("USER-CODE", proc.stdout)
            with self.assertRaises(runtime.RuntimeConfigurationError):
                runtime.execution_environment(self.base_env, self.adapter, "on", self.receipts)
            path.write_bytes(original)

    def test_cached_bytecode_or_unexpected_adapter_members_are_refused(self):
        self.stage()
        (self.adapter / "__pycache__").mkdir()
        with self.assertRaises(runtime.RuntimeConfigurationError):
            runtime.execution_environment(self.base_env, self.adapter, "on", self.receipts)

    def test_mode_conflict_fails_before_entrypoint(self):
        env, _ = self.stage()
        env[runtime.MODE_VARIABLE] = "off"
        proc = self.child(env, "print('USER-CODE')")
        self.assertEqual(proc.returncode, 78)
        self.assertNotIn("USER-CODE", proc.stdout)
        with self.assertRaises(runtime.RuntimeConfigurationError):
            runtime.execution_environment(self.base_env, self.adapter, "off", self.receipts)

    def test_receipt_content_and_current_flags_are_independently_checked(self):
        env, _ = self.stage()
        for code in (
            "import torch;torch.use_deterministic_algorithms(False,warn_only=False)",
            "import sys,json;from pathlib import Path;p=Path(sys._drdebug_runtime_startup['receipt_path']);r=json.loads(p.read_text());r['pid']=0;p.write_text(json.dumps(r))",
        ):
            proc = self.child(env, code + ";import runtime_checks;runtime_checks.startup_evidence()")
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("RuntimeConfigurationError", proc.stderr)

    def test_missing_hook_evidence_is_not_filled_in_later(self):
        self.stage()
        with self.assertRaises(runtime.RuntimeConfigurationError):
            runtime.startup_evidence("on", env=self.base_env)

    def test_receipts_cannot_be_inside_inventory_and_symlinks_are_refused(self):
        with self.assertRaises(runtime.RuntimeConfigurationError):
            runtime.configure_runtime_environment(self.base_env, self.adapter, "on", self.adapter / "receipts")
        target = self.root / "actual"
        target.mkdir()
        self.adapter.symlink_to(target, target_is_directory=True)
        with self.assertRaises(runtime.RuntimeConfigurationError):
            self.stage()

    def test_unexpected_existing_adapter_files_are_not_overwritten(self):
        self.adapter.mkdir()
        path = self.adapter / "sitecustomize.py"
        path.write_text("# existing startup customization\n")
        with self.assertRaises(runtime.RuntimeConfigurationError):
            self.stage()
        self.assertEqual(path.read_text(), "# existing startup customization\n")

    def test_an_existing_external_startup_hook_is_not_silently_shadowed(self):
        self.stage()
        fake_hook = types.SimpleNamespace(__file__="/fixture/external/sitecustomize.py")
        with mock.patch.dict(sys.modules, {"sitecustomize": fake_hook}):
            env = runtime.execution_environment(self.base_env, self.adapter, "on", self.receipts)
        self.assertEqual(env["DRDEBUG_RUNTIME_SHADOWED_STARTUP"], fake_hook.__file__)


class DependencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="drdebug-dependency-test-")
        self.root = Path(self.temp.name).resolve()
        (self.root / "compiler/tikcpp/ascendc_kernel_cmake").mkdir(parents=True)
        (self.root / "compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake").write_text("# fixture\n")
        (self.root / "compiler/tikcpp/tikcfw").mkdir()
        (self.root / "compiler/tikcpp/tikcfw/kernel_operator.h").write_text("// fixture\n")
        (self.root / "compiler/ccec_compiler/bin").mkdir(parents=True)
        ccec = self.root / "compiler/ccec_compiler/bin/ccec"
        ccec.write_text("fixture")
        ccec.chmod(0o700)
        (self.root / "version.info").write_text("version=9.1.0\n")
        self.env = {"PATH": str(Path(sys.executable).absolute().parent), "ASCEND_HOME_PATH": str(self.root)}
        self.metadata = {"imports": {name: {"ok": True, "version": version} for name, version in
                         (("numpy", "1.26.4"), ("torch", "2.9.0+cpu"), ("torch_npu", "2.9.0.post6"))},
                         "torch_cxx11_abi": True, "numpy_bridge": True,
                         "pybind11_headers": {"ok": True}, "python_headers": {"ok": True},
                         "npu_available": True, "device_count": 1, "device_name": "fixture-NPU",
                         "deterministic_enabled": False, "warn_only": False, "sitecustomize": None}
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def probe(self, command, **kwargs):
        self.calls.append(command)
        source = command[3]
        if source == runtime.PYTHON_IDENTITY:
            return {"ok": True, "data": {"executable": sys.executable, "prefix": sys.prefix,
                    "base_prefix": sys.base_prefix, "version": list(sys.version_info[:3]), "soabi": sysconfig.get_config_var("SOABI")}}
        if source == runtime.RUNTIME_METADATA:
            return {"ok": True, "data": self.metadata}
        if source == runtime.NPU_NUMERICAL_PROBE:
            return {"ok": True, "data": {"ok": True, "bitwise_stable": True, "repeats": 5}}
        raise AssertionError("Unexpected subprocess: " + repr(command))

    def report(self, **kwargs):
        with mock.patch.object(runtime, "_json_probe", side_effect=self.probe), \
             mock.patch.object(runtime.shutil, "which", side_effect=lambda binary, path=None: "/fixture/bin/" + binary), \
             mock.patch.object(runtime, "_version", return_value={"ok": True, "version": "fixture version 3.24.1"}):
            return runtime.dependency_report(npu="3", soc="ActualFixtureSOC", env=self.env, **kwargs)

    def test_metadata_checks_real_abi_bridge_headers_version_without_numerical_probe(self):
        report = self.report()
        self.assertTrue(report["ready"], [item for item in report["checks"] if not item["ok"]])
        self.assertFalse(report["runtime_probe_requested"])
        self.assertEqual(len(self.calls), 2)
        self.assertNotIn(runtime.NPU_NUMERICAL_PROBE, [command[3] for command in self.calls])

    def test_import_success_cannot_conceal_wrong_abi_or_broken_numpy_bridge(self):
        for field in ("torch_cxx11_abi", "numpy_bridge"):
            self.metadata[field] = False
            report = self.report()
            self.assertFalse(report["ready"])
            self.assertFalse(next(item for item in report["checks"] if item["name"] == field)["ok"])
            self.metadata[field] = True

    def test_unknown_or_incompatible_package_family_is_reported(self):
        for version in ("unknown", "2.8.0"):
            self.metadata["imports"]["torch_npu"]["version"] = version
            report = self.report()
            self.assertFalse(report["ready"])
            self.assertFalse(next(item for item in report["checks"] if item["name"] == "torch_npu_version_family")["ok"])

    def test_runtime_probe_is_opt_in_and_no_device_prevents_it(self):
        report = self.report(runtime_probe=True)
        self.assertTrue(report["ready"])
        self.assertEqual(sum(command[3] == runtime.NPU_NUMERICAL_PROBE for command in self.calls), 1)
        self.calls.clear()
        self.metadata["npu_available"] = False
        report = self.report(runtime_probe=True)
        self.assertFalse(report["ready"])
        self.assertFalse(any(command[3] == runtime.NPU_NUMERICAL_PROBE for command in self.calls))

    def test_unrelated_startup_hook_is_a_conflict_not_silently_shadowed(self):
        self.metadata["sitecustomize"] = "/fixture/external-sitecustomize.py"
        report = self.report()
        self.assertFalse(report["ready"])
        self.assertFalse(next(item for item in report["checks"] if item["name"] == "python_startup_hook")["ok"])

    def test_path_python_with_different_venv_prefix_is_rejected(self):
        original = self.probe
        def wrong_identity(command, **kwargs):
            result = original(command, **kwargs)
            if command[3] == runtime.PYTHON_IDENTITY:
                result["data"]["prefix"] = "/fixture/another-venv"
            return result
        with mock.patch.object(runtime, "_json_probe", side_effect=wrong_identity), \
             mock.patch.object(runtime.shutil, "which", side_effect=lambda binary, path=None: "/fixture/bin/" + binary), \
             mock.patch.object(runtime, "_version", return_value={"ok": True, "version": "fixture version 3.24.1"}):
            report = runtime.dependency_report(soc="ActualFixtureSOC", env=self.env)
        self.assertFalse(report["ready"])
        self.assertFalse(next(item for item in report["checks"] if item["name"] == "python_path_identity")["ok"])

    def test_cmake_module_without_ascendc_compiler_or_headers_is_insufficient(self):
        (self.root / "compiler/tikcpp/tikcfw/kernel_operator.h").unlink()
        report = self.report()
        self.assertFalse(report["ready"])
        self.assertFalse(next(item for item in report["checks"] if item["name"] == "ascendc_headers")["ok"])

    def test_probe_failures_are_structured_and_do_not_echo_stderr(self):
        proc = subprocess.CompletedProcess(["python"], 1, "", "TOKEN=must-not-log")
        with mock.patch.object(runtime.subprocess, "run", return_value=proc):
            result = runtime._json_probe(["python", "-c", ""], env={}, timeout=1)
        self.assertFalse(result["ok"])
        self.assertNotIn("must-not-log", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
