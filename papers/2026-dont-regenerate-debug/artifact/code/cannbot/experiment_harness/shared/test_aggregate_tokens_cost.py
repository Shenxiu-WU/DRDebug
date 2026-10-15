#!/usr/bin/env python3
"""White-box unit checks for the token accounting core (stdlib only)."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from aggregate_tokens_cost import Pricing, classify_token_reconciliation, models_match_expected


HERE = Path(__file__).resolve().parent


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def message(message_id: str, usage: dict[str, int], *, sidechain: bool, model: str = "k3") -> dict[str, object]:
    return {
        "sessionId": "11111111-1111-4111-8111-111111111111",
        "isSidechain": sidechain,
        "message": {"id": message_id, "model": model, "usage": usage, "content": []},
    }


def main() -> None:
    pricing = Pricing(json.loads((HERE / "pricing_table.json").read_text()))
    assert models_match_expected(
        {"k3[1m]": {"input_tokens": 1}},
        "k3",
        pricing,
    )
    assert models_match_expected(
        {"qwen3.8-max-preview": {"input_tokens": 1}},
        "qwen3.8-max-preview",
        pricing,
    )
    assert not models_match_expected(
        {"qwen3.8-max-preview": {"input_tokens": 1}},
        "k3",
        pricing,
    )

    status, ratio = classify_token_reconciliation(
        {
            "input_tokens": -100,
            "output_tokens": 110,
            "cache_read_tokens": 5,
            "cache_creation_tokens": 0,
        },
        10_000,
    )
    assert status == "reconciled_with_category_drift"
    assert abs(ratio - 0.0015) < 1e-12

    with tempfile.TemporaryDirectory(prefix="cannbot-harness-test-") as raw:
        root = Path(raw)
        main_journal = root / "session.jsonl"
        sub_dir = root / "session" / "subagents"
        sub_journal = sub_dir / "agent-a.jsonl"
        sub_dir.mkdir(parents=True)
        # m1 evolves from 1 token to 100 tokens. Correct aggregation keeps 100,
        # while a naive row sum would produce 101.
        main_records = [
            message("m1", {"input_tokens": 1, "output_tokens": 0}, sidechain=False),
            message(
                "m1",
                {
                    "input_tokens": 100,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 3,
                    "cache_creation_input_tokens": 4,
                },
                sidechain=False,
            ),
        ]
        main_journal.write_text("\n".join(json.dumps(item) for item in main_records) + "\n")
        sub_journal.write_text(
            json.dumps(
                message(
                    "s1",
                    {
                        "input_tokens": 20,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 6,
                        "cache_creation_input_tokens": 7,
                    },
                    sidechain=True,
                )
            )
            + "\n"
        )
        final_result = root / "final.json"
        write_json(
            final_result,
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "session_id": "11111111-1111-4111-8111-111111111111",
                "num_turns": 9,
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 3,
                    "cache_creation_input_tokens": 4,
                },
                "modelUsage": {
                    "k3[1m]": {
                        "inputTokens": 121,
                        "outputTokens": 7,
                        "cacheReadInputTokens": 9,
                        "cacheCreationInputTokens": 11,
                    }
                },
            },
        )
        full_eval = root / "full_eval.json"
        anticheat = root / "anticheat.json"
        write_json(full_eval, {"full_eval_result": "pass"})
        write_json(anticheat, {"verdict": "CLEAN"})
        manifest = root / "manifest.json"
        write_json(
            manifest,
            {
                "runs": [
                    {
                        "run_id": "synthetic-regen",
                        "operator": "level1/003_Add",
                        "arm": "regenerate",
                        "trial": 1,
                        "cwd": str(root),
                        "expected_session_id": "11111111-1111-4111-8111-111111111111",
                        "model_expected": "k3[1m]",
                        "final_result_path": str(final_result),
                        "journal_main_path": str(main_journal),
                        "subagent_dir": str(sub_dir),
                        "full_eval_path": str(full_eval),
                        "anticheat_path": str(anticheat),
                    }
                ]
            },
        )
        output_json = root / "summary.json"
        subprocess.run(
            [
                "python3",
                str(HERE / "aggregate_tokens_cost.py"),
                "--manifest",
                str(manifest),
                "--pricing",
                str(HERE / "pricing_table.json"),
                "--output-json",
                str(output_json),
                "--output-csv",
                str(root / "summary.csv"),
                "--paired-output",
                str(root / "paired.csv"),
                "--strict",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        row = json.loads(output_json.read_text())["rows"][0]
        assert row["input_tokens"] == 121
        assert row["output_tokens"] == 7
        assert row["cache_read_tokens"] == 9
        assert row["cache_creation_tokens"] == 11
        assert row["total_tokens_all"] == 148
        assert row["subagent_tokens"] == 38
        assert row["main_journal_input_tokens"] == 100
        assert row["subagent_input_tokens"] == 20
        assert row["unattributed_input_tokens"] == 1
        assert row["token_reconciliation_status"] == "reconciled_with_overhead"
        assert row["external_clean"] is True
        assert row["same_model_eligible"] is True
        expected_cny = (121 * 20 + 7 * 100 + 9 * 2 + 11 * 20) / 1_000_000
        assert abs(row["cost_native"] - expected_cny) < 1e-9
        assert abs(row["cost_usd"] - expected_cny / 6.8047) < 1e-9
        print(
            "PASS: dedupe, subagent attribution, overhead reconciliation, "
            "expected-model matching, validation, and K3 cost"
        )


if __name__ == "__main__":
    main()
