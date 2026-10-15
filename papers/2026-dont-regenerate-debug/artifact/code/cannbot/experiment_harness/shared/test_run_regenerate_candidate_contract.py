#!/usr/bin/env python3
"""Regression tests for the external evaluator deliverable contract."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import run_regenerate


def write_candidate(cwd: Path, *, layout: str, with_adapter: bool = True) -> Path:
    (cwd / "spec").mkdir(parents=True)
    (cwd / "spec/model.py").write_text("class Model: pass\n")
    project = cwd / "operators/example"
    project.mkdir(parents=True)
    if layout == "standard":
        source = project / "op_kernel/example.asc"
    elif layout == "legacy":
        source = project / "kernel/example.asc"
    else:
        raise AssertionError(layout)
    source.parent.mkdir(parents=True)
    source.write_text("extern \"C\" __global__ __aicore__ void example() {}\n")
    library = project / "build/libexample.so"
    library.parent.mkdir()
    library.write_bytes(b"not-an-elf-but-nonempty-test-fixture")
    if with_adapter:
        adapter = cwd / "candidate_adapter"
        adapter.mkdir()
        (adapter / "model_new_ascendc.py").write_text("class ModelNew: pass\n")
        (adapter / "operator_dir.txt").write_text("operators/example\n")
    return project


def materializer_layout_tests() -> None:
    for layout in ("standard", "legacy"):
        with tempfile.TemporaryDirectory() as temporary:
            cwd = Path(temporary)
            write_candidate(cwd, layout=layout)
            ok, note = run_regenerate.materialize_candidate(cwd)
            assert ok, note
            manifest = json.loads((cwd / "candidate_task/materialization_manifest.json").read_text())
            expected = "standard" if layout == "standard" else "legacy_compat"
            assert manifest["source_layout"] == expected, manifest
            assert (cwd / "candidate_task/kernel" / ("op_kernel" if layout == "standard" else "kernel") / "example.asc").is_file()
            assert (cwd / "candidate_task/operator_project").is_symlink()

    with tempfile.TemporaryDirectory() as temporary:
        cwd = Path(temporary)
        write_candidate(cwd, layout="standard", with_adapter=False)
        ok, note = run_regenerate.materialize_candidate(cwd)
        assert not ok
        assert note == "candidate_adapter missing"


def prompt_contract_test() -> None:
    record = {"operator": "level1/003_Add"}
    for resume in (False, True):
        prompt = run_regenerate.prompt_for(record, resume_session=resume)
        for required in (
            "op_extension/",
            "op_host/",
            "op_kernel/",
            "candidate_adapter/model_new_ascendc.py",
            "candidate_adapter/operator_dir.txt",
            "independently enforces this contract",
        ):
            assert required in prompt

    assert run_regenerate.classify_trial_terminal(
        {"is_error": False}, provider_aborted=False, candidate_ok=True
    ) == ("completed", False)
    assert run_regenerate.classify_trial_terminal(
        {"is_error": False}, provider_aborted=False, candidate_ok=False
    ) == ("deliverable_interrupted", True)
    assert run_regenerate.classify_trial_terminal(
        {"is_error": True}, provider_aborted=False, candidate_ok=False
    ) == ("failed", False)


def completed_legacy_reconciliation_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        harness = root / "experiment_harness"
        harness.mkdir()
        cwd = root / "runs/example/regenerate/trial1"
        write_candidate(cwd, layout="legacy")
        final = cwd / "final_stdout.json"
        final.write_text('{"is_error": false}\n')
        record_path = cwd / "run_record.json"
        record_path.write_text(
            json.dumps(
                {
                    "run_id": "regen:level1/003_Add:t1",
                    "status": "completed",
                    "candidate_materialized": False,
                    "candidate_note": "generated kernel/extension source missing",
                    "attempt_index": 1,
                }
            )
        )
        manifest = harness / "run_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "runs": [
                        {
                            "run_id": "regen:level1/003_Add:t1",
                            "operator": "level1/003_Add",
                            "arm": "regenerate",
                            "trial": 1,
                            "cwd": str(cwd),
                            "run_record_path": str(record_path),
                            "final_result_path": str(final),
                            "journal_main_path": str(cwd / "unused.jsonl"),
                        }
                    ]
                }
            )
        )
        rc = run_regenerate.main(
            [
                "--manifest",
                str(manifest),
                "--experiment-root",
                str(root),
                "--repair-invalid-completed",
            ]
        )
        assert rc == 0
        reconciled = json.loads(record_path.read_text())
        assert reconciled["candidate_materialized"] is True
        assert "source_layout=legacy_compat" in reconciled["candidate_note"]


def completed_missing_adapter_resume_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        harness = root / "experiment_harness"
        harness.mkdir()
        cwd = root / "runs/example/regenerate/trial1"
        write_candidate(cwd, layout="standard", with_adapter=False)
        final = cwd / "final_stdout.json"
        final.write_text('{"is_error": false}\n')
        journal = cwd / ".home/.claude/projects/project/session.jsonl"
        journal.parent.mkdir(parents=True)
        journal.write_text("{}\n")
        record_path = cwd / "run_record.json"
        record_path.write_text(
            json.dumps(
                {
                    "run_id": "regen:level1/003_Add:t1",
                    "status": "completed",
                    "candidate_materialized": False,
                    "candidate_note": "candidate_adapter missing",
                    "attempt_index": 1,
                }
            )
        )
        row = {
            "run_id": "regen:level1/003_Add:t1",
            "operator": "level1/003_Add",
            "arm": "regenerate",
            "trial": 1,
            "cwd": str(cwd),
            "run_record_path": str(record_path),
            "final_result_path": str(final),
            "journal_main_path": str(journal),
        }
        manifest = harness / "run_manifest.json"
        manifest.write_text(json.dumps({"runs": [row]}))
        slot = run_regenerate.ProviderSlot("qwen", root / "qwen.env")
        observed: dict[str, object] = {}

        def fake_run_one(record, worker, provider, npu, args, **kwargs):
            del worker, provider, npu, args
            observed["record"] = record
            observed.update(kwargs)
            return {
                "run_id": record["run_id"],
                "status": "completed",
                "provider_aborted": False,
                "deliverable_repair_required": False,
            }

        with (
            mock.patch.object(run_regenerate, "make_provider_envs", return_value=[slot]),
            mock.patch.object(run_regenerate, "run_one", side_effect=fake_run_one),
        ):
            rc = run_regenerate.main(
                [
                    "--manifest",
                    str(manifest),
                    "--experiment-root",
                    str(root),
                    "--key-config",
                    str(root / "unused.json"),
                    "--npus",
                    "0",
                    "--concurrency",
                    "1",
                    "--repair-invalid-completed",
                ]
            )
        assert rc == 0
        assert observed["resume_session"] is True
        assert observed["attempt_index"] == 2
        assert "do not regenerate working code from scratch" in observed["record"]["resume_directive"]


if __name__ == "__main__":
    materializer_layout_tests()
    prompt_contract_test()
    completed_legacy_reconciliation_test()
    completed_missing_adapter_resume_test()
    print("PASS: prompt, compatibility materializer, terminal reconciliation, and same-session repair")
