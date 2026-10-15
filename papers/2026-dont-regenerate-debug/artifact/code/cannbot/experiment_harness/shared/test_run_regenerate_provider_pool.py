#!/usr/bin/env python3
"""Test provider diversity, replicas, and transient abort cooldown."""

from __future__ import annotations

import json
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import run_regenerate


def static_anthropic_provider_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config = root / "qwen.json"
        config.write_text(
            json.dumps(
                {
                    "base_url": "https://example.invalid/apps/anthropic",
                    "quota_probe": "disabled",
                    "auth_mode": "auth_token",
                    "keys": [{"name": "qwen", "api_key": "fake-secret"}],
                }
            )
        )
        with (
            mock.patch.object(run_regenerate.os, "chown"),
            mock.patch.object(
                run_regenerate,
                "query_kimi_capacities_bounded",
                side_effect=AssertionError("Kimi quota endpoint must not be queried"),
            ),
        ):
            slots = run_regenerate.make_provider_envs(config, root / "env", "qwen-test", 1, 1, 20)
        assert len(slots) == 1
        env = slots[0].path.read_text()
        assert "ANTHROPIC_AUTH_TOKEN=fake-secret" in env
        assert "unset ANTHROPIC_API_KEY" in env
        assert "ANTHROPIC_BASE_URL=https://example.invalid/apps/anthropic" in env
        selection = json.loads((root / "env/provider_selection.json").read_text())
        assert selection["quota_probe"] == "disabled"


def write_manifest(root: Path, count: int) -> Path:
    harness = root / "experiment_harness"
    harness.mkdir()
    records = []
    for index in range(count):
        cwd = root / "runs" / f"r{index}"
        cwd.mkdir(parents=True)
        records.append(
            {
                "run_id": f"r{index}",
                "operator": f"level1/{index:03d}_Test",
                "arm": "regenerate",
                "cwd": str(cwd),
                "final_result_path": str(cwd / "final.json"),
                "run_record_path": str(cwd / "run_record.json"),
                "journal_main_path": str(cwd / "journal.jsonl"),
            }
        )
    path = harness / "run_manifest.json"
    path.write_text(json.dumps({"runs": records}))
    return path


def diversity_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = write_manifest(root, 5)
        slots = [
            run_regenerate.ProviderSlot(name, root / f"{name}-{replica}.env")
            for name in ("workerc", "providerb", "excludedprovider", "providera", "workerd")
            for replica in (1, 2)
        ]
        barrier = threading.Barrier(5)
        used = []

        def fake_run(
            record,
            worker,
            provider,
            npu,
            args,
            *,
            clean_start=False,
            resume_session=False,
            attempt_index=1,
        ):
            del worker, npu, args, clean_start, resume_session, attempt_index
            used.append(provider.stem.split("-")[0])
            barrier.wait(timeout=3)
            return {"run_id": record["run_id"], "status": "completed", "provider_aborted": False}

        with (
            mock.patch.object(run_regenerate, "make_provider_envs", return_value=slots),
            mock.patch.object(run_regenerate, "run_one", side_effect=fake_run),
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
                    "1,2,3,4,5",
                    "--concurrency",
                    "5",
                    "--slots-per-key",
                    "2",
                    "--preferred-provider",
                    "providera",
                    "--exclude-provider",
                    "excludedprovider",
                ]
            )
        assert rc == 0
        assert sorted(used) == ["providera", "providera", "providerb", "workerc", "workerd"], used
        # Four healthy providers exist in this fixture; all must be used
        # before a second replica fills the fifth card.
        assert {"providera", "providerb", "workerc", "workerd"}.issubset(used)
        assert "excludedprovider" not in used


def cooldown_test() -> None:
    assert run_regenerate.provider_retry_allowed(9, 0)
    assert run_regenerate.provider_retry_allowed(8, 8)
    assert not run_regenerate.provider_retry_allowed(9, 8)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = write_manifest(root, 1)
        slots = [
            run_regenerate.ProviderSlot("bad", root / "bad.env"),
            run_regenerate.ProviderSlot("good", root / "good.env"),
        ]
        used = []

        def fake_run(
            record,
            worker,
            provider,
            npu,
            args,
            *,
            clean_start=False,
            resume_session=False,
            attempt_index=1,
        ):
            del worker, npu, args, clean_start, resume_session, attempt_index
            provider_name = provider.stem
            used.append(provider_name)
            return {
                "run_id": record["run_id"],
                "status": "failed" if provider_name == "bad" else "completed",
                "provider_aborted": provider_name == "bad",
            }

        with (
            mock.patch.object(run_regenerate, "make_provider_envs", return_value=slots),
            mock.patch.object(run_regenerate, "run_one", side_effect=fake_run),
            mock.patch.object(run_regenerate, "clear_run_artifacts"),
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
                    "1",
                    "--concurrency",
                    "1",
                    "--preferred-provider",
                    "bad",
                ]
            )
        assert rc == 0
        assert used == ["bad", "good"], used
        summary = json.loads((root / "experiment_harness/regenerate_run_summary.json").read_text())
        assert summary["excluded_providers"] == []
        assert summary["permanent_abort_quarantine"] is False
        assert summary["provider_abort_counts"] == {"bad": 1}
        runtime_state = json.loads((root / "experiment_harness/provider_runtime_state.json").read_text())
        assert runtime_state["providers"]["bad"]["reason"] == "transient_provider_abort"

        explicit_quota = {
            "is_error": True,
            "api_error_status": 403,
            "result": "You've reached your usage limit for this billing cycle. Purchase extra usage.",
        }
        assert run_regenerate.classify_provider_abort(explicit_quota) == "explicit_quota_exhausted"

        wall_now = datetime(2026, 7, 20, tzinfo=timezone.utc).timestamp()
        five_hour_only = {
            "windows": {
                "weekly": {"remaining_percent": 58, "reset_time": "2026-07-25T00:00:00Z"},
                "five_hour": {"remaining_percent": 0, "reset_time": "2026-07-20T01:00:00Z"},
            }
        }
        expires, basis, windows = run_regenerate.explicit_quota_unavailable_until(
            five_hour_only, wall_now, 21600
        )
        assert expires == datetime(2026, 7, 20, 1, tzinfo=timezone.utc).timestamp()
        assert basis == "exhausted_usage_window_reset"
        assert windows == ["five_hour"]

        no_exhausted_window = {
            "windows": {
                "weekly": {"remaining_percent": 20},
                "five_hour": {"remaining_percent": 100},
            }
        }
        expires, basis, windows = run_regenerate.explicit_quota_unavailable_until(
            no_exhausted_window, wall_now, 21600
        )
        assert expires == wall_now + 21600
        assert basis == "actual_request_quota_error_without_exhausted_usage_window"
        assert windows == []

        explicit_request_reset = (
            "429 Your token-plan 5-hour quota has been exhausted. "
            "The quota will reset at 07-24 04:09:00 UTC."
        )
        request_wall_now = datetime(2026, 7, 24, 4, 7, tzinfo=timezone.utc).timestamp()
        expires, basis, windows = run_regenerate.explicit_quota_unavailable_until(
            no_exhausted_window,
            request_wall_now,
            21600,
            explicit_request_reset,
        )
        assert expires == datetime(2026, 7, 24, 4, 9, tzinfo=timezone.utc).timestamp()
        assert basis == "actual_request_explicit_reset"
        assert windows == ["5-hour"]


if __name__ == "__main__":
    static_anthropic_provider_test()
    diversity_test()
    cooldown_test()
    print("PASS: provider diversity, replica fill, and transient abort cooldown")
