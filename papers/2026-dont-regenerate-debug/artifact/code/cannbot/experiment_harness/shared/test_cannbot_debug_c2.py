from __future__ import annotations

import json
from pathlib import Path

import prepare_cannbot_debug_c2 as prepare
import run_regenerate


def test_cannbot_debug_prompt_is_debug_only() -> None:
    prompt = run_regenerate.prompt_for(
        {
            "operator": "level1/003_Add",
            "arm": "cannbot-debug",
        }
    )
    assert "/ascendc-precision-debug" in prompt
    assert "./task/" in prompt
    assert "Do not delete the failed implementation" in prompt
    assert "operator-generation workflow" in prompt
    assert "ops-direct-invoke CANNBot workflow" not in prompt


def test_cannbot_debug_resume_preserves_session_and_workspace() -> None:
    prompt = run_regenerate.prompt_for(
        {
            "operator": "level1/003_Add",
            "arm": "cannbot-debug",
            "resume_directive": "rebuild only",
        },
        resume_session=True,
    )
    assert "same Claude session" in prompt
    assert "./task/" in prompt
    assert "rebuild only" in prompt
    assert "do not switch to a generation workflow" in prompt


def test_copy_frozen_near_miss_strips_historical_outcomes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("class Model:\n    pass\n")
    (source / "model_new_ascendc.py").write_text("class ModelNew:\n    pass\n")
    (source / "3_Add.json").write_text(
        json.dumps({"inputs": [{"shape": [1], "dtype": "float32"}]}) + "\n"
    )
    (source / "_claude_result.json").write_text('{"result":"historical"}\n')
    (source / "debug_status.json").write_text('{"success":true}\n')
    (source / "model_new_tilelang.py").write_text("historical alternative\n")
    kernel = source / "kernel"
    kernel.mkdir()
    (kernel / "add.cpp").write_text("// failed initial kernel\n")

    destination = tmp_path / "frozen"
    prepare.copy_frozen_near_miss(source, destination)

    assert (destination / "model.py").is_file()
    assert (destination / "model_new_ascendc.py").is_file()
    assert (destination / "kernel" / "add.cpp").is_file()
    assert not (destination / "_claude_result.json").exists()
    assert not (destination / "debug_status.json").exists()
    assert not (destination / "model_new_tilelang.py").exists()


def test_cannbot_debug_contract_keeps_existing_task() -> None:
    contract = run_regenerate.cannbot_debug_deliverable_contract()
    assert "existing failed implementation" in contract
    assert "operator_dir.txt` equal to `task`" in contract
    assert "standard direct-invoke source layout" not in contract


def test_arm_argument_accepts_cannbot_debug() -> None:
    args = run_regenerate.parse_args(
        [
            "--arm",
            "cannbot-debug",
            "--max-turns",
            "240",
            "--soft-task-turns",
            "480",
            "--max-task-turns",
            "600",
            "--max-internal-attempts",
            "5",
            "--timeout",
            "43200",
        ]
    )
    assert args.arm == "cannbot-debug"
    assert args.max_turns == 240
    assert args.soft_task_turns == 480
    assert args.max_task_turns == 600
    assert args.max_internal_attempts == 5
    assert args.timeout == 43200


def test_prompt_records_task_level_budget() -> None:
    prompt = run_regenerate.prompt_for(
        {
            "operator": "level1/003_Add",
            "arm": "cannbot-debug",
            "soft_task_turns": 480,
            "max_task_turns": 600,
            "max_internal_attempts": 5,
        }
    )
    assert "soft boundary at 480 turns" in prompt
    assert "hard boundary at 600 turns" in prompt
    assert "at most 5 valid same-session" in prompt


def test_fallback_provider_argument_is_explicit() -> None:
    args = run_regenerate.parse_args(
        [
            "--arm",
            "cannbot-debug",
            "--preferred-provider",
            "provider-slot-a",
            "--fallback-provider",
            "provider-slot-a",
        ]
    )
    assert args.preferred_provider == "provider-slot-a"
    assert args.fallback_provider == ["provider-slot-a"]
