#!/usr/bin/env python3
"""Deterministic smoke test for the Regenerate rolling worker pool."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest import mock

import run_regenerate


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        harness = root / "experiment_harness"
        harness.mkdir()
        records = []
        for index in range(4):
            cwd = root / "runs" / f"r{index + 1}"
            cwd.mkdir(parents=True)
            records.append(
                {
                    "run_id": f"r{index + 1}",
                    "operator": f"level1/{index + 1:03d}_Test",
                    "arm": "regenerate",
                    "cwd": str(cwd),
                    "final_result_path": str(cwd / "final.json"),
                    "run_record_path": str(cwd / "run_record.json"),
                }
            )
        manifest = harness / "run_manifest.json"
        manifest.write_text(json.dumps({"runs": records}))

        slots = [
            run_regenerate.ProviderSlot("other-key", root / "other.env"),
            run_regenerate.ProviderSlot("provider-slot-a", root / "workerc.env"),
        ]
        starts: dict[str, float] = {}
        ends: dict[str, float] = {}
        providers: dict[str, str] = {}

        def fake_run_one(
            record,
            worker_index,
            provider_env,
            npu,
            args,
            *,
            clean_start=False,
            resume_session=False,
            attempt_index=1,
        ):
            del worker_index, npu, args, clean_start, resume_session, attempt_index
            run_id = record["run_id"]
            starts[run_id] = time.monotonic()
            providers[run_id] = provider_env.stem
            time.sleep(0.05 if run_id != "r2" else 0.30)
            ends[run_id] = time.monotonic()
            return {"run_id": run_id, "status": "completed", "provider_aborted": False}

        with (
            mock.patch.object(run_regenerate, "make_provider_envs", return_value=slots),
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
                    "4,5",
                    "--concurrency",
                    "2",
                    "--provider-refresh-seconds",
                    "10",
                ]
            )

        assert rc == 0
        assert providers["r1"] == "workerc", providers
        assert starts["r3"] < ends["r2"], "r3 did not refill the slot while r2 was still running"
        summary = json.loads((harness / "regenerate_run_summary.json").read_text())
        assert summary["scheduling_policy"] == "rolling_worker_pool"
        assert summary["completed"] == 4
    print("PASS: preferred workerc slot and immediate rolling refill")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
