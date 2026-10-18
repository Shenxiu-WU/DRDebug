"""Public nonfinite guard CPU contracts. --require-torch forbids skipping real Torch tests.

The fake backend gives dependency-free structural checks, not numerical proof.
Real tests import only Torch CPU and the frozen comparator function; they never
import torch_npu, allocate an NPU tensor, build native code or call a model.
"""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "papers/2026-dont-regenerate-debug/artifact/code/debug_agent_v5/utils/verification_ascendc.py"
SCRIPT = REPO / "skills/drdebug/scripts/nonfinite_guard.py"
spec = importlib.util.spec_from_file_location("public_nonfinite_guard_tests", SCRIPT)
guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
integration = guard


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)
    def to(self, dtype):
        return self


class FakeTorch(types.ModuleType):
    Tensor = FakeTensor
    float32 = "float32"
    def __init__(self):
        super().__init__("torch")
        self.nn = types.ModuleType("torch.nn")
        self.nn.Module = object
    @staticmethod
    def isnan(value):
        return FakeTensor(math.isnan(item) for item in value.values)
    @staticmethod
    def isposinf(value):
        return FakeTensor(item == float("inf") for item in value.values)
    @staticmethod
    def isneginf(value):
        return FakeTensor(item == -float("inf") for item in value.values)
    @staticmethod
    def equal(lhs, rhs):
        return lhs.values == rhs.values
    @staticmethod
    def nan_to_num(value):
        limit = 3.4028234663852886e38
        return FakeTensor(0.0 if math.isnan(item) else limit if item == float("inf")
                          else -limit if item == -float("inf") else item for item in value.values)
    @staticmethod
    def allclose(lhs, rhs, atol, rtol):
        return all(abs(a - b) <= atol + rtol * abs(b) for a, b in zip(lhs.values, rhs.values))


def comparator_namespace(torch, reference, candidate, name="test_import"):
    tree = ast.parse(FROZEN.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "_compare_values")
    namespace = {"torch": torch, "__name__": name,
                 "_tensor_diff_summary": lambda *args, **kwargs: "finite numerical mismatch"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(FROZEN), "exec"), namespace)
    def original_run(_):
        ok, message = namespace["_compare_values"](reference, candidate, 1e-4, 1e-4, "output[0]")
        return {"ok": ok, "comparisons": ["case[0]: " + message]}
    namespace["_run_verification"] = original_run
    return namespace


class ComparatorContracts:
    def tensor(self, values):
        return self.torch.tensor(values, dtype=self.torch.float32) if hasattr(self.torch, "tensor") else FakeTensor(values)

    def check_pair(self, reference, candidate, expected):
        namespace = comparator_namespace(self.torch, self.tensor(reference), self.tensor(candidate))
        guard.install(namespace)
        report = namespace["_run_verification"]("CPU_ONLY")
        self.assertEqual(report["ok"], expected, report)
        self.assertEqual(report["public_nonfinite_guard"]["passed"], expected, report)
        return report

    def test_legacy_zero_nan_false_pass_is_rejected(self):
        namespace = comparator_namespace(self.torch, self.tensor([0.0]), self.tensor([float("nan")]))
        self.assertTrue(namespace["_run_verification"]("CPU_ONLY")["ok"])
        guard.install(namespace)
        value = namespace["_run_verification"]("CPU_ONLY")
        self.assertFalse(value["ok"])
        self.assertIn("nonfinite_category_mismatch", value["comparisons"][0])
        self.assertEqual(value["public_nonfinite_guard"]["nonfinite_mismatches"], 1)

    def test_finite_inf_and_reverse_nan_are_rejected(self):
        limit = 3.4028234663852886e38
        for reference, candidate in (([limit], [float("inf")]), ([-limit], [-float("inf")]),
                                     ([float("nan")], [0.0]), ([float("inf")], [limit])):
            with self.subTest(reference=reference, candidate=candidate):
                self.check_pair(reference, candidate, False)

    def test_matching_reference_nonfinite_categories_are_allowed(self):
        values = [float("nan"), float("inf"), -float("inf"), 1.0]
        value = self.check_pair(values, values, True)
        self.assertEqual(value["public_nonfinite_guard"]["expected_tensor_leaves"], 1)
        self.assertEqual(value["public_nonfinite_guard"]["checked_tensor_leaves"], 1)

    def test_changed_nonfinite_sign_or_position_is_rejected(self):
        for reference, candidate in (([float("inf")], [-float("inf")]),
                                     ([float("nan"), 0.0], [0.0, float("nan")]),
                                     ([float("inf"), 0.0], [0.0, float("inf")])):
            with self.subTest(reference=reference, candidate=candidate):
                self.check_pair(reference, candidate, False)

    def test_original_finite_tolerances_are_preserved(self):
        self.check_pair([1.0, 2.0], [1.00015, 2.00015], True)
        self.check_pair([1.0, 2.0], [1.01, 2.0], False)
        self.check_pair([float("nan"), float("inf"), 1.0],
                        [float("nan"), float("inf"), 1.01], False)

    def test_nested_output_leaf_coverage_is_complete(self):
        reference = {"a": [self.tensor([float("nan")]), self.tensor([1.0])],
                     "b": (self.tensor([float("inf")]), self.tensor([-float("inf")]))}
        candidate = {"a": [self.tensor([float("nan")]), self.tensor([1.0])],
                     "b": (self.tensor([float("inf")]), self.tensor([-float("inf")]))}
        namespace = comparator_namespace(self.torch, reference, candidate)
        guard.install(namespace)
        value = namespace["_run_verification"]("CPU_ONLY")
        self.assertTrue(value["ok"])
        self.assertEqual(value["public_nonfinite_guard"]["checked_tensor_leaves"], 4)
        self.assertTrue(value["public_nonfinite_guard"]["coverage_complete"])

    def test_early_finite_mismatch_still_checks_all_nested_tensor_leaves(self):
        reference = [self.tensor([1.0]), self.tensor([float("inf")])]
        candidate = [self.tensor([2.0]), self.tensor([float("inf")])]
        namespace = comparator_namespace(self.torch, reference, candidate)
        guard.install(namespace)
        value = namespace["_run_verification"]("CPU_ONLY")
        receipt = value["public_nonfinite_guard"]
        self.assertFalse(value["ok"])
        self.assertEqual(receipt["expected_tensor_leaves"], 2)
        self.assertEqual(receipt["checked_tensor_leaves"], 2)
        self.assertTrue(receipt["coverage_complete"])
        self.assertEqual(receipt["guard_errors"], 0)

    def test_original_structure_failures_are_accounted_for_without_guard_errors(self):
        tensor = self.tensor
        pairs = ((tensor([1.0, 2.0]), tensor([1.0])),
                 (tensor([1.0]), [tensor([1.0])]),
                 ([tensor([1.0]), tensor([2.0])], [tensor([1.0])]),
                 ({"a": tensor([1.0])}, {"b": tensor([1.0])}),
                 ([tensor([1.0]), tensor([2.0])], [tensor([1.0]), [tensor([2.0])]]))
        for reference, candidate in pairs:
            with self.subTest(reference_type=type(reference), candidate_type=type(candidate)):
                namespace = comparator_namespace(self.torch, reference, candidate)
                original = namespace["_run_verification"]("CPU_ONLY")
                self.assertFalse(original["ok"])
                guard.install(namespace)
                value = namespace["_run_verification"]("CPU_ONLY")
                self.assertFalse(value["ok"])
                self.assertEqual(value["comparisons"], original["comparisons"])
                receipt = value["public_nonfinite_guard"]
                self.assertTrue(receipt["coverage_complete"])
                self.assertGreater(receipt["structural_rejections"], 0)
                self.assertEqual(receipt["guard_errors"], 0)
                self.assertEqual(receipt["expected_tensor_leaves"],
                                 receipt["checked_tensor_leaves"] + receipt["unpaired_tensor_leaves"])
                provenance = {"schema_version": guard.VERSION, "policy": guard.POLICY,
                              "helper_sha256": receipt["helper_sha256"]}
                self.assertEqual(guard.validate_receipt(receipt, provenance, entrypoint="import",
                                                       expected_success=False), receipt)
                with self.assertRaises(guard.NonfiniteGuardError):
                    guard.validate_receipt(receipt, provenance, entrypoint="import")

    def test_guard_errors_fail_closed(self):
        namespace = comparator_namespace(self.torch, self.tensor([1.0]), self.tensor([1.0]))
        guard.install(namespace)
        with mock.patch.object(self.torch, "isnan", side_effect=RuntimeError("CPU synthetic guard failure")):
            value = namespace["_run_verification"]("CPU_ONLY")
        self.assertFalse(value["ok"])
        self.assertEqual(value["public_nonfinite_guard"]["guard_errors"], 1)
        self.assertFalse(value["public_nonfinite_guard"]["coverage_complete"])

    def test_cli_marker_and_import_receipt_are_consistent(self):
        for entry in ("__main__", "test_import"):
            namespace = comparator_namespace(self.torch, self.tensor([1.0]), self.tensor([1.0]), entry)
            identity = guard.install(namespace)
            self.assertEqual(guard.install(namespace), identity)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                value = namespace["_run_verification"]("CPU_ONLY")
            receipt = value["public_nonfinite_guard"]
            self.assertEqual(receipt["entrypoint"], "cli" if entry == "__main__" else "import")
            self.assertEqual(bool(output.getvalue()), entry == "__main__")
            if entry == "__main__":
                self.assertEqual(json.loads(output.getvalue().split("=", 1)[1]), receipt)


class FakeComparatorTests(ComparatorContracts, unittest.TestCase):
    def setUp(self):
        self.torch = FakeTorch()


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "real Torch CPU unavailable; fake checks are not numerical proof")
class RealTorchCPUComparatorTests(ComparatorContracts, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_supported_float_and_integer_dtypes_use_original_dtype_masks(self):
        for dtype in (self.torch.float16, self.torch.bfloat16, self.torch.float32):
            with self.subTest(dtype=dtype):
                left = self.torch.tensor([0.0, float("inf")], dtype=dtype)
                right = self.torch.tensor([float("nan"), float("inf")], dtype=dtype)
                namespace = comparator_namespace(self.torch, left, right)
                guard.install(namespace)
                self.assertFalse(namespace["_run_verification"]("CPU_ONLY")["ok"])
        for dtype in (self.torch.int8, self.torch.int32, self.torch.int64, self.torch.bool):
            with self.subTest(dtype=dtype):
                left = self.torch.tensor([0, 1], dtype=dtype)
                namespace = comparator_namespace(self.torch, left, left.clone())
                guard.install(namespace)
                self.assertTrue(namespace["_run_verification"]("CPU_ONLY")["ok"])


class IntegrationTests(unittest.TestCase):
    def test_real_frozen_identity_and_fresh_staging_only(self):
        original = FROZEN.read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(), integration.ORIGINAL_SHA)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); (root / "utils").mkdir()
            verifier = root / integration.VERIFIER_REL; verifier.write_bytes(original)
            receipt = integration.stage_guard(root)
            self.assertEqual(FROZEN.read_bytes(), original)
            self.assertEqual(receipt["helper_sha256"], hashlib.sha256((root / integration.HELPER_REL).read_bytes()).hexdigest())
            self.assertLess(verifier.read_text().index("_public_nf_module.install(globals())"),
                            verifier.read_text().index('if __name__ == "__main__":'))
            with self.assertRaises(ValueError):
                integration.stage_guard(root)

    def test_real_staged_source_installs_for_import_and_before_cli_usage_exit(self):
        fake = FakeTorch()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); (root / "utils").mkdir()
            verifier = root / integration.VERIFIER_REL; verifier.write_bytes(FROZEN.read_bytes())
            integration.stage_guard(root)
            for entry in ("test_import", "__main__"):
                namespace = {"__file__": str(verifier), "__name__": entry}
                with mock.patch.dict(sys.modules, {"torch": fake, "torch.nn": fake.nn}), \
                     mock.patch.object(sys, "argv", [str(verifier)]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    if entry == "__main__":
                        with self.assertRaises(SystemExit) as stopped:
                            exec(compile(verifier.read_bytes(), str(verifier), "exec"), namespace)
                        self.assertEqual(stopped.exception.code, 1)  # Usage only; no task/device execution.
                    else:
                        exec(compile(verifier.read_bytes(), str(verifier), "exec"), namespace)
                self.assertTrue(namespace["_public_nonfinite_guard_installed"])
                self.assertFalse(namespace["_compare_values"](FakeTensor([0.0]), FakeTensor([float("nan")]), 1e-4, 1e-4)[0])

    def test_unknown_source_is_rejected_without_creating_helper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); (root / "utils").mkdir()
            (root / integration.VERIFIER_REL).write_text("print('unknown verifier')\n")
            with self.assertRaisesRegex(ValueError, "Unknown frozen verifier"):
                integration.stage_guard(root)
            self.assertFalse((root / integration.HELPER_REL).exists())


class ReceiptTests(unittest.TestCase):
    def receipt(self):
        return {"schema_version": 1, "policy": guard.POLICY, "helper_sha256": "a" * 64,
                "entrypoint": "import", "expected_tensor_leaves": 2, "checked_tensor_leaves": 2,
                "unpaired_tensor_leaves": 0, "structural_rejections": 0,
                "nonfinite_mismatches": 0, "guard_errors": 0, "coverage_complete": True, "passed": True}

    def provenance(self):
        return {"schema_version": 1, "policy": guard.POLICY, "helper_sha256": "a" * 64}

    def test_missing_changed_wrong_entrypoint_or_incomplete_receipts_are_rejected(self):
        for name, value in (("helper_sha256", "b" * 64), ("policy", "unknown"), ("entrypoint", "cli"),
                            ("checked_tensor_leaves", 1), ("coverage_complete", False),
                            ("guard_errors", 1), ("nonfinite_mismatches", 1), ("passed", False),
                            ("structural_rejections", 1), ("unpaired_tensor_leaves", 1),
                            ("expected_tensor_leaves", True)):
            with self.subTest(field=name):
                receipt = self.receipt(); receipt[name] = value
                with self.assertRaises(guard.NonfiniteGuardError):
                    guard.validate_receipt(receipt, self.provenance(), entrypoint="import")
        with self.assertRaises(guard.NonfiniteGuardError):
            guard.validate_receipt(None, self.provenance(), entrypoint="import")

    def test_guarded_numerical_failure_remains_distinct_from_missing_coverage(self):
        receipt = self.receipt(); receipt.update(passed=False, nonfinite_mismatches=1)
        self.assertEqual(guard.validate_receipt(receipt, self.provenance(), entrypoint="import", expected_success=False), receipt)
        with self.assertRaises(guard.NonfiniteGuardError):
            guard.validate_receipt(receipt, self.provenance(), entrypoint="import", expected_success=True)


if __name__ == "__main__":
    if "--require-torch" in sys.argv:
        sys.argv.remove("--require-torch")
        if not HAS_TORCH:
            raise SystemExit("Real Torch is required; no numerical proof claimed")
    unittest.main()
