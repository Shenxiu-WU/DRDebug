#!/usr/bin/env python3
"""Regression tests for frozen full-case filename staging."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from validate_manifest import prepare_eval_dir


SELF_NAMED_MODEL = """
import os

def get_input_groups():
    json_path = os.path.join(
        os.path.dirname(__file__),
        os.path.splitext(os.path.basename(__file__))[0] + '.json',
    )
    return json_path
"""


def case(index: int) -> str:
    return json.dumps({"inputs": [{"name": "x", "shape": [index + 1]}]})


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        candidate = root / "candidate"
        evaluation = root / "evaluation"
        source.mkdir()
        candidate.mkdir()
        (source / "model.py").write_text(SELF_NAMED_MODEL)
        full_text = case(0) + "\n" + case(1) + "\n"
        (source / "20_LongOperatorName.json.bak").write_text(full_text)

        full_cases, count, library_paths, build_alias = prepare_eval_dir(
            source, candidate, evaluation
        )

        assert full_cases == source / "20_LongOperatorName.json.bak"
        assert count == 2
        assert library_paths == []
        assert build_alias is None
        assert (evaluation / "20_LongOperatorName.json").read_text() == full_text
        assert (evaluation / "model.json").read_text() == full_text
        manifest = json.loads((evaluation / "_external_eval_manifest.json").read_text())
        assert manifest["reference_case_aliases"] == ["model.json"]
        assert manifest["staged_case_names"] == ["20_LongOperatorName.json", "model.json"]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        candidate = root / "candidate"
        evaluation = root / "evaluation"
        source.mkdir()
        candidate.mkdir()
        (source / "model.py").write_text("def get_input_groups():\n    return []\n")
        (source / "model.json").write_text(case(9) + "\n")
        full_text = case(0) + "\n" + case(1) + "\n" + case(2) + "\n"
        (source / "full.json.bak").write_text(full_text)

        full_cases, count, library_paths, build_alias = prepare_eval_dir(
            source, candidate, evaluation
        )

        assert full_cases == source / "full.json.bak"
        assert count == 3
        assert library_paths == []
        assert build_alias is None
        assert (evaluation / "model.json").read_text() == full_text
        assert (evaluation / "full.json").read_text() == full_text

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        candidate = root / "candidate"
        evaluation = root / "evaluation"
        source.mkdir()
        candidate.mkdir()
        (source / "model.py").write_text("def get_input_groups():\n    return []\n")
        (source / "model.json").write_text(case(0) + "\n")
        (candidate / "model_new_ascendc.py").write_text("CANDIDATE = True\n")
        (candidate / "kernel").mkdir()
        (candidate / "kernel" / "kernel.cpp").write_text("// source\n")
        legacy_build = candidate / "operator_project" / "kernel" / "build"
        legacy_build.mkdir(parents=True)
        (legacy_build / "_candidate_ext.so").write_bytes(b"test")

        _, _, library_paths, build_alias = prepare_eval_dir(
            source, candidate, evaluation
        )

        assert not (evaluation / "model_new_ascendc.py").is_symlink()
        assert (evaluation / "model_new_ascendc.py").read_text() == "CANDIDATE = True\n"
        assert (evaluation / "kernel" / "kernel.cpp").read_text() == "// source\n"
        assert (evaluation / "kernel" / "build").is_symlink()
        assert build_alias == str(evaluation / "kernel" / "build")
        assert library_paths == [str(evaluation / "operator_project" / "kernel" / "build")]
        assert (evaluation / "kernel" / "build" / "_candidate_ext.so").read_bytes() == b"test"

    print(
        "PASS: full-case aliases and legacy candidate build staging are compatible"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
