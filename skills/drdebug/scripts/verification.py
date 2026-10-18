"""Independent clean-build precision replay for a prepared public workspace.

This portability tool never promotes a checkpoint or edits engine history. The
frozen verifier owns input generation, comparisons and dtype-specific limits.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from uuid import uuid4
import nonfinite_guard


def kernel_hashes(api, task):
    return {p.relative_to(task / "kernel").as_posix(): api.sha(p)
            for p in api.files(task / "kernel")
            if "build" not in p.relative_to(task / "kernel").parts
            and p.suffix not in {".so", ".o", ".a", ".pyc"}}


def checked_comparisons(report, count):
    rows = report.get("comparisons", [])
    indices = []
    matched = 0
    for row in rows:
        match = re.fullmatch(r"case\[(\d+)\]: (.*)", row)
        if not match:
            return False
        index = int(match[1])
        indices.append(index)
        matched += match[2] == f"output[{index}]: matched"
    tensors = [row for row in report.get("inputs", []) if "Tensor(" in row]
    return (report.get("ok") is True and report.get("device") in {"npu", "npu:0"}
            and indices == list(range(count)) and matched == count and bool(tensors)
            and all("device=npu:0" in row for row in tensors))


def replay(config_path, suite):
    """Private child entrypoint, one fresh interpreter for each requested suite."""
    import drdebug as api
    import runtime_checks
    config_path = Path(config_path)
    config = api.read_json(config_path)
    task = Path(config["task"])
    result = {"suite": suite, "passed": False, "errors": [], "pid": os.getpid(),
              "source_sha256": kernel_hashes(api, task), "python_optimize": sys.flags.optimize}
    try:
        if sys.flags.optimize or result["source_sha256"] != config["source_sha256"]:
            raise ValueError("Interpreter/source identity differs from independent replay configuration")
        result["startup"] = runtime_checks.startup_evidence()
        active = task / config["contract"]["compact_file"]
        expected_hash = config["contract"]["protected_sha256"][
            config["contract"]["compact_file"] if suite == "compact" else config["contract"]["full_file"]]
        if api.sha(active) != expected_hash:
            raise ValueError("Active independent case collection differs from original bytes")
        if api.sha(task / "model.json") != expected_hash:
            raise ValueError("Legacy model.json input alias differs from the selected original collection")
        binding = (task / "kernel/pybind11.cpp").read_text()
        match = re.search(r"PYBIND11_MODULE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,", binding)
        if not match or match[1] in sys.modules:
            raise ValueError("Missing binding or unexpected preloaded native module")
        native_name = match[1]
        candidates = list((task / "kernel/build").glob(native_name + ".*.so"))
        candidates += list((task / "kernel/build").glob(native_name + ".so"))
        if len(candidates) != 1:
            raise ValueError("Expected exactly one independently built native extension")
        native_path = candidates[0].resolve()
        native_before = api.sha(native_path)
        if native_before != config["native_sha256"] or str(native_path) != config["native_path"]:
            raise ValueError("Native extension differs from the pinned clean-build artifact")
        verifier_path = Path(config["runtime"]) / "utils/verification_ascendc.py"
        if api.sha(verifier_path) != config["verifier_sha256"]:
            raise ValueError("Original verifier changed")
        spec = importlib.util.spec_from_file_location("drdebug_original_verifier", verifier_path)
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        selected = None
        if suite.startswith("case-"):
            selected = int(suite[5:])
            original_groups = verifier._get_input_groups

            def select_after_original_generation(module):
                groups = original_groups(module)
                result["generated_full_case_count"] = len(groups)
                if len(groups) != config["contract"]["full_cases"] or not 0 <= selected < len(groups):
                    raise ValueError("Original full input generation did not match declared coverage")
                return [groups[selected]]
            verifier._get_input_groups = select_after_original_generation
        original = verifier._run_verification(str(task))
        result["public_verifier_report"] = original
        if config.get("nonfinite_guard") is not None:
            result["public_nonfinite_guard"] = nonfinite_guard.validate_receipt(
                original.get("public_nonfinite_guard"), config["nonfinite_guard"],
                entrypoint="import", expected_success=original.get("ok") is True)
        count = 1 if selected is not None else config["contract"][suite + "_cases"]
        result.update(expected_cases=count, full_case_index=selected,
                      case_collection_sha256=api.sha(active),
                      executed_cases=len(original.get("comparisons", [])))
        wrapper = sys.modules.get(str(task) + "_ascendc_model")
        extension = sys.modules.get(native_name)
        actual = Path(getattr(extension, "__file__", "/missing")).resolve()
        bound = bool(wrapper and extension and any(v is extension for v in vars(wrapper).values()))
        native = isinstance(getattr(extension, "__loader__", None), importlib.machinery.ExtensionFileLoader)
        result["native"] = {"module": native_name, "path": str(actual), "sha256": api.sha(actual) if actual.is_file() else None,
                            "extension_loader": native, "wrapper_bound": bound,
                            "clean_build_module_unchanged": actual == native_path and api.sha(native_path) == native_before}
        after = runtime_checks.startup_evidence()
        if after != result["startup"]:
            raise ValueError("Runtime flags/environment changed during numerical replay")
        if not (checked_comparisons(original, count) and bound and native
                and Path(getattr(wrapper, "__file__", "/missing")).resolve() == (task / "model_new_ascendc.py").resolve()
                and result["native"]["clean_build_module_unchanged"]):
            raise ValueError("Original precision verification or actual native module binding failed")
        if kernel_hashes(api, task) != config["source_sha256"]:
            raise ValueError("Source changed during replay")
        result["passed"] = True
    except (Exception, SystemExit) as exc:
        result["errors"].append(type(exc).__name__ + ": " + str(exc))
    api.write_json(config_path.parent / (suite + ".json"), result)
    return 0 if result["passed"] else 1


def verify(args, api):
    """Copy active source, rebuild, run original compact/full, optionally fresh cases."""
    for name in ("stage_timeout_sec", "overall_timeout_sec"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise api.ContractError(name + " must be finite and positive")
    workspace, manifest = api.load_workspace(args.workspace)
    if manifest.get("schema_version", 1) < 3:
        raise api.ContractError("Independent replay requires a newly prepared schema-3 workspace; preserve the old workspace")
    mode = manifest["execution"]["determinism"]
    if getattr(args, "determinism", None) not in (None, mode):
        raise api.ContractError("Verification retains the workspace deterministic mode")
    lock = workspace / ".drdebug.lock"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise api.ContractError("Workspace is locked; do not audit while a worker can write it") from exc
    output = workspace / "verification" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid4().hex[:8])
    summary = {"passed": False, "returncode": 2, "source": "active_task", "engine_history_modified": False,
               "checkpoint_promoted": False, "started_at": api.stamp(), "stages": {}, "casewise_requested": args.casewise}
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"pid": os.getpid(), "purpose": "independent_verify", "started_at": api.stamp()}, f)
        report = api.environment_report(workspace=workspace, npu=args.npu, soc=args.soc,
                                        require_worker=False, runtime_probe=True, determinism=mode)
        if not report["ready"]:
            return {"returncode": 2, "outcome": "preflight_failed", "preflight": report}
        api.load_workspace(workspace)
        output.mkdir(parents=True, exist_ok=False)
        original = workspace / "task"
        source = kernel_hashes(api, original)
        history_before = {p.relative_to(original).as_posix(): api.sha(p) for p in api.files(original)
                          if p.relative_to(original).parts[0] != "kernel"}
        task = output / "task"
        task.mkdir()
        contract = manifest["task_contract"]
        for name in contract["protected_sha256"]:
            shutil.copy2(original / name, task / name)
        api.copy_tree(original / "kernel", task / "kernel", kernel=True)
        if kernel_hashes(api, task) != source:
            raise api.ContractError("Source changed while creating independent task")
        summary.update(workspace=str(workspace), output=str(output), source_sha256=source,
                       protected_sha256=contract["protected_sha256"], determinism=mode)
        runtime = workspace / "runtime"
        config = {"task": str(task), "runtime": str(runtime), "source_sha256": source,
                  "contract": contract, "verifier_sha256": api.sha(runtime / "utils/verification_ascendc.py"),
                  "nonfinite_guard": manifest.get("nonfinite_guard")}
        summary["nonfinite_guard"] = {"required": "nonfinite_guard" in manifest,
            "policy": (manifest.get("nonfinite_guard") or {}).get("policy"),
            "helper_sha256": (manifest.get("nonfinite_guard") or {}).get("helper_sha256"),
            "suites": {}, "requires_new_workspace": "nonfinite_guard" not in manifest}
        api.write_json(output / "config.json", config)
        _, env, _ = api.engine_command(workspace, manifest, args,
            {"max_attempts": 1, "max_turns": 1, "worker_timeout_sec": 1}, 1)
        # Verification never needs provider credentials, even when the launching shell has them.
        env = {k: v for k, v in env.items() if not k.startswith(("ANTHROPIC_", "CLAUDE_CODE_"))}
        deadline = time.monotonic() + args.overall_timeout_sec
        def stage(name, command):
            cap = min(args.stage_timeout_sec, deadline - time.monotonic())
            if cap <= 0:
                raise api.EngineStopped(4, "verification_wall_clock_budget_exhausted")
            started = time.monotonic()
            with (output / (name + ".stdout.log")).open("x") as out, (output / (name + ".stderr.log")).open("x") as err:
                child = subprocess.Popen(command, cwd=runtime, env=env, stdout=out, stderr=err,
                                         start_new_session=os.name == "posix")
                code = api._wait_engine(child, cap)
            summary["stages"][name] = {"returncode": code, "elapsed_sec": time.monotonic() - started}
            api.write_json(output / "progress.json", summary)
            return code
        if stage("clean-build", [sys.executable, "-B", str(runtime / "utils/build_ascendc.py"), str(task),
                                 "--soc-version", args.soc or os.environ.get("ASCENDC_SOC_VERSION", ""), "--clean"]):
            raise ValueError("Independent clean build failed")
        match = re.search(r"PYBIND11_MODULE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,", (task / "kernel/pybind11.cpp").read_text())
        candidates = list((task / "kernel/build").glob(match[1] + ".*.so")) + list((task / "kernel/build").glob(match[1] + ".so")) if match else []
        if len(candidates) != 1:
            raise ValueError("Clean build did not produce exactly one native extension")
        native_path = candidates[0].resolve()
        config.update(native_path=str(native_path), native_sha256=api.sha(native_path))
        summary["native_clean_build_sha256"] = config["native_sha256"]
        api.write_json(output / "config.json", config)
        compact_path = task / contract["compact_file"]
        compact = compact_path.read_bytes()
        alias = task / "model.json"
        # Mirror the frozen validator's legacy input alias, only in this new
        # independent copy. Never edit model.py or any original case file.
        alias.write_bytes(compact)
        summary["model_json_alias"] = {"path": str(alias), "source": contract["compact_file"], "sha256": api.sha(alias)}
        helper = str(Path(__file__).resolve())
        modes = ["compact", "full"]
        if args.casewise:
            modes += ["case-" + str(i).zfill(2) for i in range(contract["full_cases"])]
        try:
            for suite in modes:
                selected_bytes = compact if suite == "compact" else (task / contract["full_file"]).read_bytes()
                compact_path.write_bytes(selected_bytes)
                alias.write_bytes(selected_bytes)
                code = stage(suite, [sys.executable, "-B", helper, "--_replay", str(output / "config.json"), suite])
                if code not in (0, 1):
                    raise ValueError("Unexpected verifier child exit; no further replay launched")
        finally:
            compact_path.write_bytes(compact)
            alias.write_bytes(compact)
        stage("cpp-integrity", [sys.executable, "-B", helper, "--_cpp", str(output / "config.json")])
        stage("ast", [sys.executable, "-B", str(runtime / api.AST_VALIDATOR_REL), str(task / "model_new_ascendc.py"), "--json"])
        api.load_workspace(workspace)
        summary["integrity"] = {
            "original_source_unchanged": kernel_hashes(api, original) == source,
            "independent_source_unchanged": kernel_hashes(api, task) == source,
            "native_artifact_unchanged": api.sha(native_path) == config["native_sha256"],
            "protected_inputs_unchanged": all(api.sha(task / n) == h for n, h in contract["protected_sha256"].items()),
            "original_history_unchanged": history_before == {p.relative_to(original).as_posix(): api.sha(p) for p in api.files(original)
                                                            if p.relative_to(original).parts[0] != "kernel"}}
        summary["suite_results"] = {s: api.read_json(output / (s + ".json")) for s in modes}
        if "nonfinite_guard" in manifest:
            summary["nonfinite_guard"]["suites"] = {suite: nonfinite_guard.validate_receipt(
                row.get("public_nonfinite_guard"), manifest["nonfinite_guard"], entrypoint="import")
                for suite, row in summary["suite_results"].items()}
        summary["passed"] = (all(v["returncode"] == 0 for v in summary["stages"].values())
                              and all(summary["integrity"].values())
                              and all(r["passed"] and r["native"]["sha256"] == config["native_sha256"] for r in summary["suite_results"].values()))
        summary["returncode"] = 0 if summary["passed"] else 1
    except api.EngineStopped as exc:
        summary.update(returncode=exc.returncode, error=exc.outcome)
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(returncode=130 if isinstance(exc, KeyboardInterrupt) else 2, error=type(exc).__name__ + ": " + str(exc))
    finally:
        summary["finished_at"] = api.stamp()
        if output.is_dir():
            api.write_json(output / "summary.json", summary)
        lock.unlink(missing_ok=True)
    return summary


def cpp_integrity(config_path):
    import drdebug as api
    config_path = Path(config_path)
    config = api.read_json(config_path)
    checker = Path(config["runtime"]) / api.DEBUG_REL / "scripts/anticheat.py"
    result = {"passed": False, "checker_sha256": api.sha(checker)}
    try:
        spec = importlib.util.spec_from_file_location("drdebug_original_anticheat", checker)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        value = module._check_cpp_regression(Path(config["task"]))
        result["result"] = value
        result["passed"] = value.get("status") == "pass" and value.get("launch_found") is True and bool(value.get("files_scanned")) and not value.get("violations")
    except (Exception, SystemExit) as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    api.write_json(config_path.parent / "cpp-integrity.json", result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--_cpp":
        raise SystemExit(cpp_integrity(sys.argv[2]))
    if len(sys.argv) == 4 and sys.argv[1] == "--_replay":
        raise SystemExit(replay(sys.argv[2], sys.argv[3]))
    raise SystemExit("Use drdebug.py verify WORKSPACE; this helper has no standalone public command")
