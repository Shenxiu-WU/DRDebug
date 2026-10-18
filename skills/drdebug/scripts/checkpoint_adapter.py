"""Public portability correction for the frozen engine's checkpoint boundary.

Only a newly prepared runtime copy is adapted. The paper archive stays byte
identical. The original crash-consistent directory switch, rollback journal,
validation commands, comparator and acceptance gates remain in use. This module
binds objective results to source/input bytes and includes completed full-set
results in the existing best/rollback ordering. A best checkpoint may be a
partially correct candidate; it is not an acceptance certificate.
"""
from __future__ import annotations

from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

VERSION = 1
ENGINE_REL = Path("skills/ascendc/ascendc-debug/engine")
ORIGINAL_SHA256 = {
    "best_rollback.py": "ec4fa7f4dcb4feb37531392ac73824edf4ac279fd9466e8f6788cadca795e1f2",
    "validate_runner.py": "de0c0e5d839270b304cf43d0dcb77b9a6eaf4cf9c0687c4e9021753af76b11d2",
}
_IDENTITY_DIR = "precision_tuning/public_checkpoint_identity"
_SOURCE_SUFFIXES = {".cpp", ".h", ".hpp", ".py"}
_CASE_LINE = re.compile(r"^case\[(\d+)\]:\s*(.+)$", re.MULTILINE)
_HASH = re.compile(r"[0-9a-f]{64}")
_NONFINITE_POLICY = "exact_nan_posinf_neginf_masks_then_frozen_comparator"
_NONFINITE_MARKER = "DRDEBUG_NONFINITE_GUARD="


class CheckpointAdapterError(ValueError):
    pass


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CheckpointAdapterError("Expected JSON object")
    return value


def _no_links(root, path):
    """Reject links in both existing files and directory ancestors."""
    root, path = Path(root), Path(path)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise CheckpointAdapterError("Checkpoint evidence path escapes task") from exc
    current = root
    if current.is_symlink():
        raise CheckpointAdapterError("Symlinked checkpoint evidence root")
    for part in relative.parts:
        if part in {"..", "."}:
            raise CheckpointAdapterError("Unsafe checkpoint evidence path")
        current /= part
        if current.is_symlink():
            raise CheckpointAdapterError("Symlinked checkpoint evidence path")
    return path


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stage_checkpoint_adapter(runtime):
    """Adapt a fresh runtime copy and return original/adapted SHA provenance.

    Call after copying frozen sources, before hashing runtime_sha256. This is
    intentionally not an upgrade/migration operation for existing workspaces.
    """
    runtime = Path(runtime)
    engine = runtime / ENGINE_REL
    helper = engine / "_public_checkpoint_adapter.py"
    if helper.exists() or helper.is_symlink():
        raise CheckpointAdapterError("Checkpoint adapter is already staged; prepare a fresh workspace")
    originals = {}
    for name, expected in ORIGINAL_SHA256.items():
        path = engine / name
        if path.is_symlink() or not path.is_file() or _sha(path) != expected:
            raise CheckpointAdapterError(f"Unsupported frozen engine source: {name}")
        originals[name] = path.read_bytes()
    payload = Path(__file__).read_bytes()
    additions = {
        "best_rollback.py": (
            "\n\n# Public workspace portability correction; frozen archive remains unchanged.\n"
            "from ._public_checkpoint_adapter import install_checkpoint_adapter as _public_install_checkpoint\n"
            "_public_install_checkpoint(globals())\n"
        ),
        "validate_runner.py": (
            "\n\n# Bind objective results to the exact source/input bytes for checkpoint recovery.\n"
            "from ._public_checkpoint_adapter import install_validation_adapter as _public_install_validation\n"
            "run_objective_validation = _public_install_validation(run_objective_validation)\n"
        ),
    }
    helper.write_bytes(payload)
    records = []
    for name, data in originals.items():
        path = engine / name
        path.write_bytes(data + additions[name].encode("utf-8"))
        records.append({"path": (ENGINE_REL / name).as_posix(),
                        "original_sha256": ORIGINAL_SHA256[name], "adapted_sha256": _sha(path)})
    return {"schema_version": VERSION, "policy": "completed_full_coverage_then_compact",
            "implementation": "public_runtime_only_portability_correction",
            "helper_path": helper.relative_to(runtime).as_posix(),
            "helper_sha256": _sha(helper), "modules": records,
            "frozen_archive_modified": False}


def _sources(kernel):
    kernel = Path(kernel)
    if kernel.is_symlink() or not kernel.is_dir():
        raise CheckpointAdapterError("Missing or symlinked kernel source root")
    entries = []
    for path in sorted(kernel.rglob("*")):
        relative = path.relative_to(kernel)
        if "build" in relative.parts or path.name.startswith("."):
            continue
        if path.is_symlink():
            raise CheckpointAdapterError("Symlinked kernel input")
        if path.is_file() and path.suffix in _SOURCE_SUFFIXES:
            entries.append({"path": relative.as_posix(), "size": path.stat().st_size,
                            "sha256": _sha(path)})
    if not entries:
        raise CheckpointAdapterError("No native kernel sources")
    return entries


def _contract(task):
    task = Path(task).resolve()
    manifest_path = task.parent / "drdebug-workspace.json"
    if task.name != "task" or manifest_path.is_symlink():
        raise CheckpointAdapterError("Checkpoint correction requires a prepared public workspace")
    workspace = _json(manifest_path)
    if workspace.get("workspace_path") != str(task.parent):
        raise CheckpointAdapterError("Workspace location changed")
    contract = workspace.get("task_contract")
    if not isinstance(contract, dict):
        raise CheckpointAdapterError("Missing protected task contract")
    hashes = contract.get("protected_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise CheckpointAdapterError("Missing protected task inputs")
    for name, digest in hashes.items():
        if (not isinstance(name, str) or Path(name).name != name
                or not isinstance(digest, str) or not _HASH.fullmatch(digest)):
            raise CheckpointAdapterError("Invalid protected task input identity")
        path = task / name
        if path.is_symlink() or not path.is_file() or _sha(path) != digest:
            raise CheckpointAdapterError(f"Protected task input changed: {name}")
    for field in ("compact_cases", "full_cases"):
        value = contract.get(field)
        if type(value) is not int or value < 1:
            raise CheckpointAdapterError("Invalid expected case coverage")
    for field in ("compact_file", "full_file"):
        if contract.get(field) not in hashes:
            raise CheckpointAdapterError("Case file is not protected")
    full_files = contract.get("full_files", [contract["full_file"]])
    if not isinstance(full_files, list) or not full_files or any(name not in hashes for name in full_files):
        raise CheckpointAdapterError("Full case inputs are not protected")
    return contract


def _case_digest(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.dumps(json.loads(line), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return hashlib.sha256("".join(row + "\n" for row in sorted(rows)).encode()).hexdigest()


def _nonfinite_contract(task):
    workspace = _json(Path(task).parent / "drdebug-workspace.json")
    if "nonfinite_guard" not in workspace:
        return None  # Historical workspaces are not silently adapted.
    value = workspace["nonfinite_guard"]
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or value.get("policy") != _NONFINITE_POLICY
            or value.get("helper") != "utils/_public_nonfinite_guard.py"
            or value.get("verifier") != "utils/verification_ascendc.py"
            or value.get("hook_before_cli_main") is not True
            or value.get("import_entrypoint_covered") is not True
            or any(not _HASH.fullmatch(str(value.get(key, "")))
                   for key in ("helper_sha256", "original_sha256", "adapted_sha256"))):
        raise CheckpointAdapterError("Invalid workspace nonfinite guard provenance")
    return value


def _nonfinite_receipt(path, provenance, success):
    # Stdlib-only metadata validation: acceptance_status never imports a task,
    # engine, Torch or an executable helper from the workspace.
    lines = [line[len(_NONFINITE_MARKER):] for line in Path(path).read_text(encoding="utf-8").splitlines()
             if line.startswith(_NONFINITE_MARKER)]
    if len(lines) != 1:
        raise CheckpointAdapterError("Expected exactly one objective nonfinite guard receipt")
    value = json.loads(lines[0])
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or value.get("policy") != _NONFINITE_POLICY or value.get("entrypoint") != "cli"
            or value.get("helper_sha256") != provenance["helper_sha256"]
            or any(type(value.get(name)) is not int or value[name] < 0 for name in
                   ("expected_tensor_leaves", "checked_tensor_leaves", "unpaired_tensor_leaves",
                    "structural_rejections", "nonfinite_mismatches", "guard_errors"))
            or value.get("coverage_complete") is not True
            or value["expected_tensor_leaves"] != value["checked_tensor_leaves"] + value["unpaired_tensor_leaves"]
            or value["guard_errors"] != 0 or value.get("passed") is not success
            or value["nonfinite_mismatches"] > value["checked_tensor_leaves"]
            or (value["unpaired_tensor_leaves"] and not value["structural_rejections"])
            or (success and any(value[name] for name in
                    ("nonfinite_mismatches", "unpaired_tensor_leaves", "structural_rejections")))):
        raise CheckpointAdapterError("Objective nonfinite guard identity, coverage or outcome mismatch")
    return value


def _snapshot(task):
    contract = _contract(task)
    # Bind the complete contracted input set, not just the numeric case count.
    value = {"sources": _sources(Path(task) / "kernel"),
            "protected_sha256": dict(contract["protected_sha256"]),
            "compact_file": contract["compact_file"], "full_file": contract["full_file"],
            "full_files": contract.get("full_files", [contract["full_file"]]),
            "compact_cases": contract["compact_cases"], "full_cases": contract["full_cases"],
            "compact_case_set_sha256": _case_digest(Path(task) / contract["compact_file"]),
            "full_case_set_sha256": _case_digest(Path(task) / contract["full_file"])}
    guard = _nonfinite_contract(task)
    if guard is not None:
        value["nonfinite_guard"] = guard
    return value


def _result_files(task, attempt, data):
    """Record objective raw result/log identities without reading task code."""
    task = Path(task).resolve()
    tuning = task / "precision_tuning"
    compact = tuning / f"validation_result_attempt_{attempt}.json"
    _no_links(task, compact)
    output = {compact.relative_to(task).as_posix(): _sha(compact)}
    full = tuning / f"validation_result_attempt_{attempt}_full.json"
    if data.get("full_eval") is not None:
        _no_links(task, full)
        output[full.relative_to(task).as_posix()] = _sha(full)
    paths = [data.get("stdout_path"), data.get("stderr_path")]
    if isinstance(data.get("full_eval"), dict) and data["full_eval"].get("ran") is True:
        paths.append(data["full_eval"].get("stdout_path"))
    for value in paths:
        if not isinstance(value, str):
            raise CheckpointAdapterError("Objective output path is missing")
        path = _no_links(task, Path(value))
        if path.is_symlink() or not path.is_file() or task not in path.resolve().parents:
            raise CheckpointAdapterError("Objective result path escapes task or is missing")
        output[path.relative_to(task).as_posix()] = _sha(path)
    return output


def install_validation_adapter(original):
    """Wrap the real engine objective call; never synthesize a pass or replay."""
    @wraps(original)
    def bound_validation(task_dir, *, attempt, **kwargs):
        task = Path(task_dir).resolve()
        before = _snapshot(task)
        _no_links(task, task / "precision_tuning")
        _no_links(task, task / ".verify_logs")
        # The frozen full-eval writer runs before its compact-result writer.
        # On a fresh baseline it otherwise silently loses the full sidecar
        # because that compact writer creates precision_tuning only later.
        # Create its regular parent first; keep the original writer/results.
        (task / "precision_tuning").mkdir(exist_ok=True)
        record_id = uuid.uuid4().hex
        directory = task / _IDENTITY_DIR / f"attempt_{attempt}"
        evidence = directory / record_id
        _no_links(task, evidence)
        receipt = {"schema_version": VERSION, "attempt": attempt, "before": before,
                   "identity_valid": False, "objective_completed": False}
        try:
            result = original(task, attempt=attempt, **kwargs)
            after = _snapshot(task)
            data = _json(task / "precision_tuning" / f"validation_result_attempt_{attempt}.json")
            records = _result_files(task, attempt, data)
            for relative, expected in records.items():
                source = task / relative
                content = source.read_bytes()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise CheckpointAdapterError("Objective output changed while preserving evidence")
                target = evidence / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
            receipt.update({"after": after, "identity_valid": before == after,
                            "objective_completed": True,
                            "result_files_sha256": records,
                            "evidence_directory": evidence.relative_to(task).as_posix(),
                            "build_exit_code": result.get("build_exit_code"),
                            "verification_ran": result.get("verification_ran"),
                            "verification_exit_code": result.get("verification_exit_code"),
                            "classify_exit_code": result.get("classify_exit_code")})
            return result
        finally:
            # Append immutable receipts; retries of one attempt retain all evidence.
            path = directory / (record_id + ".json")
            _atomic_json(path, receipt)
            _atomic_json(directory / "latest.json", {"path": path.name, "sha256": _sha(path)})
    return bound_validation


def _integer(value):
    return value if type(value) is int and value >= 0 else None


def _finite(value, *, default=-1.0):
    if isinstance(value, bool):
        return default
    try:
        value = float(value)
        return value if math.isfinite(value) and 0 <= value <= 100 else default
    except (ValueError, TypeError):
        return default


def _coverage(task, path, expected):
    text = Path(path).read_text(encoding="utf-8")
    rows = [(int(match.group(1)), match.group(2).strip()) for match in _CASE_LINE.finditer(text)]
    if len(rows) != expected or {index for index, _ in rows} != set(range(expected)):
        return None
    # Exact verifier suffix; "not matched" and arbitrary contains("matched")
    # strings must never count as matches.
    passed = sum(bool(re.fullmatch(r"output\[\d+\]: matched", detail)) for _, detail in rows)
    return passed


def _read_bound_metric(task_dir, attempt, *, receipt_path=None, receipt_sha256=None):
    task = Path(task_dir).resolve()
    directory = task / _IDENTITY_DIR / f"attempt_{attempt}"
    if receipt_path is None:
        latest = _json(_no_links(task, directory / "latest.json"))
        name, expected_receipt_sha = latest.get("path"), latest.get("sha256")
    else:
        relative = Path(receipt_path)
        if relative.is_absolute() or ".." in relative.parts or relative.parent != Path(_IDENTITY_DIR) / f"attempt_{attempt}":
            raise CheckpointAdapterError("Checkpoint receipt escapes its attempt")
        name, expected_receipt_sha = relative.name, receipt_sha256
    if not isinstance(name, str) or not re.fullmatch(r"[0-9a-f]{32}\.json", name):
        raise CheckpointAdapterError("Invalid checkpoint identity receipt pointer")
    path = _no_links(task, directory / name)
    if path.is_symlink() or _sha(path) != expected_receipt_sha:
        raise CheckpointAdapterError("Checkpoint identity receipt changed")
    receipt = _json(path)
    if (receipt.get("schema_version") != VERSION or receipt.get("attempt") != attempt
            or receipt.get("objective_completed") is not True
            or receipt.get("identity_valid") is not True
            or receipt.get("before") != receipt.get("after")):
        raise CheckpointAdapterError("Objective source/input identity was not stable")
    snapshot = receipt["after"]
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("sources"), list) or not snapshot["sources"]:
        raise CheckpointAdapterError("Missing objective source inventory")
    contract = _contract(task)
    # Recheck protected inputs on resume; historical source bytes are deliberately
    # checked against current_best/src only when activating or recovering it.
    expected_snapshot = {
        "protected_sha256": contract["protected_sha256"],
        "compact_file": contract["compact_file"], "full_file": contract["full_file"],
        "full_files": contract.get("full_files", [contract["full_file"]]),
        "compact_cases": contract["compact_cases"], "full_cases": contract["full_cases"],
        "compact_case_set_sha256": _case_digest(task / contract["compact_file"]),
        "full_case_set_sha256": _case_digest(task / contract["full_file"]),
    }
    guard = _nonfinite_contract(task)
    if guard is not None:
        expected_snapshot["nonfinite_guard"] = guard
    if any(snapshot.get(name) != value for name, value in expected_snapshot.items()):
        raise CheckpointAdapterError("Objective inputs do not match workspace")
    records = receipt.get("result_files_sha256")
    if not isinstance(records, dict) or not records:
        raise CheckpointAdapterError("Missing objective result identities")
    evidence_path = Path(receipt.get("evidence_directory", ""))
    expected_evidence = Path(_IDENTITY_DIR) / f"attempt_{attempt}" / name[:-5]
    if evidence_path != expected_evidence:
        raise CheckpointAdapterError("Objective evidence directory is not bound to its receipt")
    evidence = task / evidence_path
    for relative, digest in records.items():
        relative_path = Path(relative)
        target = _no_links(task, evidence / relative_path)
        if (relative_path.is_absolute() or ".." in relative_path.parts or target.is_symlink()
                or task not in target.resolve().parents or _sha(target) != digest):
            raise CheckpointAdapterError("Objective result or log changed")
    data_path = task / "precision_tuning" / f"validation_result_attempt_{attempt}.json"
    if data_path.relative_to(task).as_posix() not in records:
        raise CheckpointAdapterError("Objective metric is not bound")
    data = _json(evidence / data_path.relative_to(task))
    if data.get("attempt") != attempt or data.get("source") != "engine_objective_validate":
        raise CheckpointAdapterError("Wrong objective attempt or producer")
    metric = {"attempt": attempt, "case_pass_rate": data.get("case_pass_rate"),
              "match_rate": data.get("match_rate"), "correctness_passed": data.get("correctness_passed")}
    meta = {"schema_version": VERSION, "receipt": path.relative_to(task).as_posix(),
            "receipt_sha256": _sha(path), "sources": snapshot["sources"],
            "protected_sha256": snapshot["protected_sha256"],
            "compact_complete": False, "full_complete": False, "accepted": False}
    metric["public_checkpoint"] = meta
    if (receipt.get("build_exit_code") != 0 or receipt.get("verification_ran") is not True
            or receipt.get("classify_exit_code") != 0):
        return metric
    compact_path = Path(data.get("stdout_path", ""))
    if compact_path.relative_to(task).as_posix() not in records:
        raise CheckpointAdapterError("Compact case output is not bound")
    compact_passed = _coverage(task, evidence / compact_path.relative_to(task), snapshot["compact_cases"])
    if (compact_passed is None or _integer(data.get("total_cases")) != snapshot["compact_cases"]
            or _integer(data.get("passed_cases")) != compact_passed):
        return metric
    compact_all = compact_passed == snapshot["compact_cases"]
    if data.get("correctness_passed") is not compact_all:
        return metric
    if guard is not None:
        compact_guard = _nonfinite_receipt(evidence / compact_path.relative_to(task), guard, compact_all)
        meta["nonfinite_guard"] = {"required": True, "policy": _NONFINITE_POLICY,
                                  "helper_sha256": guard["helper_sha256"], "compact": compact_guard}
    meta.update({"compact_complete": True, "compact_passed": compact_passed,
                 "compact_total": snapshot["compact_cases"],
                 "compact_match_rate": _finite(data.get("match_rate"))})
    # Derive ratios from verified counts, not rounded/free-form metric strings.
    metric["case_pass_rate"] = 100.0 * compact_passed / snapshot["compact_cases"]
    full = data.get("full_eval")
    if not isinstance(full, dict):
        return metric
    full_path = task / "precision_tuning" / f"validation_result_attempt_{attempt}_full.json"
    if full_path.relative_to(task).as_posix() not in records or _json(evidence / full_path.relative_to(task)) != full:
        raise CheckpointAdapterError("Full result is missing, changed, or from another attempt")
    identity = (full.get("active_json") == snapshot["compact_file"]
                and full.get("full_json_source") in snapshot["full_files"]
                and full.get("active_json_cases") == snapshot["compact_cases"]
                and full.get("full_json_cases") == snapshot["full_cases"]
                and full.get("active_case_set_sha256") == snapshot["compact_case_set_sha256"]
                and full.get("full_case_set_sha256") == snapshot["full_case_set_sha256"])
    if not identity:
        return metric
    if full.get("coverage_equivalent") is True and full.get("ran") is False:
        equivalent = (snapshot["compact_cases"] == snapshot["full_cases"]
                      and snapshot["compact_case_set_sha256"] == snapshot["full_case_set_sha256"])
        full_passed = compact_passed if equivalent else None
        full_match = meta["compact_match_rate"]
        if guard is not None:
            meta["nonfinite_guard"].update(full=compact_guard, full_coverage_equivalent=True)
    elif full.get("ran") is True and full.get("coverage_equivalent") is False:
        full_stdout = Path(full.get("stdout_path", ""))
        if full_stdout.relative_to(task).as_posix() not in records:
            raise CheckpointAdapterError("Full case output is not bound")
        full_passed = _coverage(task, evidence / full_stdout.relative_to(task), snapshot["full_cases"])
        if (full.get("crashed") is not False or full_passed is None
                or _integer(full.get("total_cases")) != snapshot["full_cases"]
                or _integer(full.get("passed_cases")) != full_passed
                or full.get("correctness_passed") is not (full_passed == snapshot["full_cases"])):
            return metric
        full_match = _finite(full.get("match_rate"))
        if guard is not None:
            meta["nonfinite_guard"]["full"] = _nonfinite_receipt(
                evidence / full_stdout.relative_to(task), guard, full_passed == snapshot["full_cases"])
    else:
        return metric
    if full_passed is not None:
        meta.update({"full_complete": True, "full_passed": full_passed,
                     "full_total": snapshot["full_cases"], "full_match_rate": full_match,
                     "accepted": compact_all and full_passed == snapshot["full_cases"]})
    return metric


def _metric_key(metric):
    meta = metric.get("public_checkpoint") or {}
    if not isinstance(meta, dict) or not meta.get("compact_complete"):
        return (-1.0,) * 5
    compact = 100.0 * meta["compact_passed"] / meta["compact_total"]
    if meta.get("full_complete"):
        return (1.0, 100.0 * meta["full_passed"] / meta["full_total"],
                _finite(meta.get("full_match_rate")), compact, _finite(meta.get("compact_match_rate")))
    return (0.0, compact, _finite(meta.get("compact_match_rate")), -1.0, -1.0)


def install_checkpoint_adapter(namespace):
    """Use the real checkpoint transaction and its existing rollback policy."""
    original_ensure = namespace["ensure_current_best"]
    original_valid = namespace["_checkpoint_valid"]
    original_allowed = namespace["checkpoint_allowed"]

    def read_attempt_metric(task_dir, attempt):
        try:
            return _read_bound_metric(task_dir, attempt)
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def checkpoint_allowed(gate_result):
        checks = getattr(gate_result, "checks", None) or {}
        return (original_allowed(gate_result)
                and all(checks.get(name) is True for name in
                        ("anticheat_pass", "cpp_regression_pass", "ast_degrade_pass",
                         "cpp_checker_present", "ast_validator_present"))
                and checks.get("cpp_checker_errored") is not True
                and checks.get("ast_validator_errored") is not True)

    def checkpoint_valid(path, *, allow_legacy=True):
        if not original_valid(path, allow_legacy=False):
            return False
        try:
            data = _json(Path(path) / "metric.json")
            meta = data.get("public_checkpoint") or {}
            if (not isinstance(meta, dict) or meta.get("schema_version") != VERSION or not meta.get("compact_complete")
                    or meta.get("sources") != _sources(Path(path) / "src")):
                return False
            task = Path(path).parents[2]
            verified = _read_bound_metric(task, data.get("attempt"),
                                          receipt_path=meta.get("receipt"),
                                          receipt_sha256=meta.get("receipt_sha256"))
            return data == verified
        except (OSError, ValueError, TypeError, KeyError):
            return False

    def ensure_current_best(task_dir, attempt, metric=None, *, force=False):
        verified = read_attempt_metric(task_dir, attempt)
        if verified is None:
            return {"success": False, "updated": False, "error": "public_objective_identity_missing_or_changed"}
        if not verified["public_checkpoint"]["compact_complete"]:
            return {"success": True, "updated": False, "reason": "objective_case_coverage_incomplete"}
        try:
            if _sources(Path(task_dir) / "kernel") != verified["public_checkpoint"]["sources"]:
                raise CheckpointAdapterError("Kernel changed after objective validation")
        except (OSError, ValueError):
            return {"success": False, "updated": False, "error": "public_objective_source_changed"}
        # Ignore caller-supplied metrics: only the bound real engine artifacts
        # may choose a checkpoint. Keep original atomic activation/recovery.
        return original_ensure(task_dir, attempt, verified, force=force)

    namespace.update({"read_attempt_metric": read_attempt_metric,
                      "_metric_key": _metric_key, "checkpoint_allowed": checkpoint_allowed,
                      "_checkpoint_valid": checkpoint_valid, "ensure_current_best": ensure_current_best})


def acceptance_status(task_dir):
    """Read-only public success gate for the active source and current_best.

    No task/engine import, model call, process, validation or checkpoint recovery
    is performed. This checks the real checkpoint's original manifest and its
    fixed objective receipt. It does not replace independent numerical replay or
    the C++/AST static checks required by the launcher.
    """
    detail = {"ready": False, "checkpoint": "precision_tuning/history/current_best",
              "source": "active_task_equals_source_bound_checkpoint",
              "read_only": True}
    try:
        task = Path(task_dir).resolve()
        best = task / "precision_tuning/history/current_best"
        _no_links(task, best)
        if not best.is_dir():
            raise CheckpointAdapterError("missing_current_best")
        manifest = _json(_no_links(task, best / "manifest.json"))
        metric = _json(_no_links(task, best / "metric.json"))
        if (manifest.get("complete") is not True or manifest.get("schema_version") != 1
                or type(metric.get("attempt")) is not int
                or manifest.get("attempt") != metric.get("attempt")):
            raise CheckpointAdapterError("invalid_checkpoint_manifest")
        sources = _sources(best / "src")
        if manifest.get("files") != sources:
            raise CheckpointAdapterError("checkpoint_source_manifest_mismatch")
        meta = metric.get("public_checkpoint")
        if (not isinstance(meta, dict) or meta.get("schema_version") != VERSION
                or not isinstance(meta.get("receipt"), str)
                or not isinstance(meta.get("receipt_sha256"), str)):
            raise CheckpointAdapterError("missing_source_bound_checkpoint_metric")
        verified = _read_bound_metric(task, metric["attempt"],
                                      receipt_path=meta["receipt"],
                                      receipt_sha256=meta["receipt_sha256"])
        if metric != verified or meta.get("sources") != sources:
            raise CheckpointAdapterError("checkpoint_metric_or_source_receipt_mismatch")
        if not all(meta.get(name) is True for name in ("compact_complete", "full_complete", "accepted")):
            raise CheckpointAdapterError("checkpoint_not_all_cases_matched")
        if _sources(task / "kernel") != sources:
            raise CheckpointAdapterError("active_source_differs_from_current_best")
        detail.update(ready=True, best_attempt=metric["attempt"],
                      compact={"passed": meta["compact_passed"], "total": meta["compact_total"]},
                      full={"passed": meta["full_passed"], "total": meta["full_total"]},
                      source_sha256={entry["path"]: entry["sha256"] for entry in sources},
                      receipt=meta["receipt"], receipt_sha256=meta["receipt_sha256"],
                      nonfinite_guard=meta.get("nonfinite_guard", {
                          "required": False, "requires_new_workspace": True, "mode": "legacy_unchecked"}))
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        detail["error"] = type(exc).__name__ + ": " + str(exc)
    return detail
