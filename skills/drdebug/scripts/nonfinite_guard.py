"""Public runtime-only exceptional-value identity and coverage guard.

The frozen paper verifier stays unchanged. New runtime copies check original
NaN/+Inf/-Inf masks, then delegate to the original finite-value comparator with
its original tolerances. Importing this helper never imports Torch or task code.
"""
from functools import wraps
import hashlib
import json
from pathlib import Path
import re

VERSION = 1
POLICY = "exact_nan_posinf_neginf_masks_then_frozen_comparator"
MARKER = "DRDEBUG_NONFINITE_GUARD="
VERIFIER_REL = Path("utils/verification_ascendc.py")
HELPER_REL = Path("utils/_public_nonfinite_guard.py")
ORIGINAL_SHA = "227408745d07f39d60e2c40ac1fc298330b1585c4b786aab92a2ce3738c3fd45"
ANCHOR = '\nif __name__ == "__main__":\n    main()\n'
HOOK = '''
# Public runtime-only nonfinite identity guard; frozen paper bytes stay intact.
import importlib.util as _public_nf_import
_public_nf_spec = _public_nf_import.spec_from_file_location(
    "_drdebug_public_nonfinite_guard", Path(__file__).with_name("_public_nonfinite_guard.py"))
if _public_nf_spec is None or _public_nf_spec.loader is None:
    raise RuntimeError("Public nonfinite guard loader is unavailable")
_public_nf_module = _public_nf_import.module_from_spec(_public_nf_spec)
_public_nf_spec.loader.exec_module(_public_nf_module)
_public_nf_module.install(globals())
'''


class NonfiniteGuardError(ValueError):
    pass


def stage_guard(runtime):
    """Adapt only a fresh runtime copy; return audited source/helper identities."""
    runtime = Path(runtime)
    verifier, helper = runtime / VERIFIER_REL, runtime / HELPER_REL
    if (any(path.is_symlink() for path in (runtime, *runtime.parents))
            or verifier.is_symlink() or not verifier.is_file()
            or helper.exists() or helper.is_symlink()):
        raise NonfiniteGuardError("Fresh regular verifier and absent helper required")
    original = verifier.read_bytes()
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA:
        raise NonfiniteGuardError("Unknown frozen verifier source")
    text = original.decode("utf-8")
    if text.count(ANCHOR) != 1:
        raise NonfiniteGuardError("Unknown verifier CLI anchor")
    payload = Path(__file__).read_bytes()
    adapted = text.replace(ANCHOR, "\n" + HOOK + ANCHOR).encode("utf-8")
    compile(adapted, str(verifier), "exec")
    with helper.open("xb") as stream:
        stream.write(payload)
    verifier.write_bytes(adapted)
    return {"schema_version": VERSION, "policy": POLICY, "verifier": VERIFIER_REL.as_posix(),
            "original_sha256": ORIGINAL_SHA, "adapted_sha256": hashlib.sha256(adapted).hexdigest(),
            "helper": HELPER_REL.as_posix(), "helper_sha256": hashlib.sha256(payload).hexdigest(),
            "hook_before_cli_main": True, "import_entrypoint_covered": True,
            "frozen_archive_modified": False}


def validate_receipt(value, provenance, *, entrypoint, expected_success=True):
    """Validate guard identity/coverage without importing or executing a task."""
    if (not isinstance(provenance, dict) or provenance.get("schema_version") != VERSION
            or provenance.get("policy") != POLICY
            or not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("helper_sha256", "")))):
        raise NonfiniteGuardError("Invalid nonfinite guard provenance")
    if (not isinstance(value, dict) or value.get("schema_version") != VERSION
            or value.get("policy") != POLICY or value.get("entrypoint") != entrypoint
            or value.get("helper_sha256") != provenance["helper_sha256"]
            or any(type(value.get(name)) is not int or value[name] < 0 for name in
                   ("expected_tensor_leaves", "checked_tensor_leaves", "unpaired_tensor_leaves",
                    "structural_rejections", "nonfinite_mismatches", "guard_errors"))
            or value.get("coverage_complete") is not True
            or value["expected_tensor_leaves"] != value["checked_tensor_leaves"] + value["unpaired_tensor_leaves"]
            or value["guard_errors"] != 0 or type(value.get("passed")) is not bool
            or value["nonfinite_mismatches"] > value["checked_tensor_leaves"]
            or (value["unpaired_tensor_leaves"] and not value["structural_rejections"])
            or (value["passed"] and any(value[name] for name in
                ("nonfinite_mismatches", "unpaired_tensor_leaves", "structural_rejections")))
            or (expected_success is not None and value["passed"] is not expected_success)):
        raise NonfiniteGuardError("Nonfinite guard identity, coverage or outcome mismatch")
    return dict(value)


def install(namespace):
    if namespace.get("_public_nonfinite_guard_installed") is True:
        return namespace["_public_nonfinite_guard_identity"]
    original, original_run, torch = namespace.get("_compare_values"), namespace.get("_run_verification"), namespace.get("torch")
    if (not callable(original) or not callable(original_run) or torch is None
            or not all(callable(getattr(torch, name, None)) for name in ("isnan", "isposinf", "isneginf", "equal"))):
        raise NonfiniteGuardError("Missing frozen comparator/runtime contract")
    identity = {"schema_version": VERSION, "policy": POLICY,
                "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "entrypoint": "cli" if namespace.get("__name__") == "__main__" else "import"}
    state = {"depth": 0, "expected_tensor_leaves": 0, "checked_tensor_leaves": 0,
             "unpaired_tensor_leaves": 0, "structural_rejections": 0,
             "nonfinite_mismatches": 0, "guard_errors": 0}

    def leaves(value):
        if isinstance(value, torch.Tensor):
            return 1
        if isinstance(value, (tuple, list)):
            return sum(leaves(item) for item in value)
        if isinstance(value, dict):
            return sum(leaves(item) for item in value.values())
        return 0

    def inspect_values(lhs, rhs, path):
        structural = (type(lhs) is not type(rhs)
            or (isinstance(lhs, torch.Tensor) and lhs.shape != rhs.shape)
            or (isinstance(lhs, (tuple, list)) and len(lhs) != len(rhs))
            or (isinstance(lhs, dict) and lhs.keys() != rhs.keys()))
        if structural:
            # These reference leaves cannot be paired. They are accounted for
            # explicitly; the original comparator still rejects the structure.
            state["structural_rejections"] += 1
            state["unpaired_tensor_leaves"] += leaves(lhs)
            return None  # The original comparator rejects this structure.
        if isinstance(lhs, torch.Tensor):
            try:
                different = [name for name, mask in (("nan", torch.isnan), ("posinf", torch.isposinf),
                                                     ("neginf", torch.isneginf))
                             if not torch.equal(mask(lhs), mask(rhs))]
                state["checked_tensor_leaves"] += 1
                if different:
                    state["nonfinite_mismatches"] += 1
                    return f"{path}: nonfinite_category_mismatch(category={different[0]})"
            except Exception as exc:
                state["guard_errors"] += 1
                return f"{path}: nonfinite_guard_error({type(exc).__name__})"
        elif isinstance(lhs, (tuple, list)) and len(lhs) == len(rhs):
            errors = [inspect_values(a, b, f"{path}[{index}]") for index, (a, b) in enumerate(zip(lhs, rhs))]
            return next((error for error in errors if error), None)
        elif isinstance(lhs, dict) and lhs.keys() == rhs.keys():
            errors = [inspect_values(lhs[key], rhs[key], f"{path}.{key}") for key in lhs]
            return next((error for error in errors if error), None)
        return None

    @wraps(original)
    def compare(lhs, rhs, atol, rtol, path="output"):
        top = state["depth"] == 0
        before = state["checked_tensor_leaves"]
        expected = leaves(lhs) if top else 0
        if top:
            state["expected_tensor_leaves"] += expected
        state["depth"] += 1
        try:
            # Inspect all paired leaves first, even when the original numerical
            # comparison subsequently stops at an early finite mismatch.
            error = inspect_values(lhs, rhs, path) if top else None
            if error:
                return False, error
            ok, message = original(lhs, rhs, atol, rtol, path)
            if ok and top and state["checked_tensor_leaves"] - before != expected:
                state["guard_errors"] += 1
                return False, f"{path}: nonfinite_guard_incomplete"
            return ok, message
        finally:
            state["depth"] -= 1

    @wraps(original_run)
    def run(*args, **kwargs):
        state.update(depth=0, expected_tensor_leaves=0, checked_tensor_leaves=0,
                     unpaired_tensor_leaves=0, structural_rejections=0,
                     nonfinite_mismatches=0, guard_errors=0)
        report = original_run(*args, **kwargs)
        if not isinstance(report, dict):
            raise NonfiniteGuardError("Frozen verifier returned no report object")
        receipt = {**identity, **{name: value for name, value in state.items() if name != "depth"}}
        receipt["coverage_complete"] = (state["depth"] == 0
            and state["expected_tensor_leaves"] == state["checked_tensor_leaves"] + state["unpaired_tensor_leaves"])
        receipt["passed"] = (report.get("ok") is True and receipt["coverage_complete"]
            and all(state[name] == 0 for name in
                    ("nonfinite_mismatches", "guard_errors", "structural_rejections", "unpaired_tensor_leaves")))
        if report.get("ok") is True and not receipt["passed"]:
            report["ok"], report["error"] = False, "nonfinite_guard_incomplete"
        report["public_nonfinite_guard"] = receipt
        if identity["entrypoint"] == "cli":
            print(MARKER + json.dumps(receipt, sort_keys=True))
        return report

    namespace.update(_compare_values=compare, _run_verification=run,
                     _public_nonfinite_guard_installed=True, _public_nonfinite_guard_identity=identity)
    return identity
