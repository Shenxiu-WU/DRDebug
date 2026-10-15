#!/usr/bin/env python3
"""Verify all trial1 runs precede trial2 and trial3 in the rolling queue."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import run_regenerate


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        harness = root / "experiment_harness"
        harness.mkdir()
        records = []
        for operator in ("A", "B"):
            for trial in (1, 2, 3):
                cwd = root / "runs" / operator / f"trial{trial}"
                cwd.mkdir(parents=True)
                records.append(
                    {
                        "run_id": f"{operator}:t{trial}",
                        "operator": operator,
                        "trial": trial,
                        "arm": "regenerate",
                        "cwd": str(cwd),
                        "final_result_path": str(cwd / "final.json"),
                        "run_record_path": str(cwd / "run_record.json"),
                    }
                )
        manifest = harness / "run_manifest.json"
        manifest.write_text(json.dumps({"runs": records}))
        starts: list[str] = []
        slots = [run_regenerate.ProviderSlot("healthy", root / "healthy.env")]

        def fake_run(record, worker, provider, npu, args, **kwargs):
            del worker, provider, npu, args, kwargs
            starts.append(record["run_id"])
            return {"run_id": record["run_id"], "status": "completed", "provider_aborted": False}

        with (
            mock.patch.object(run_regenerate, "make_provider_envs", return_value=slots),
            mock.patch.object(run_regenerate, "run_one", side_effect=fake_run),
        ):
            rc = run_regenerate.main(
                [
                    "--manifest", str(manifest),
                    "--experiment-root", str(root),
                    "--key-config", str(root / "unused.json"),
                    "--npus", "1",
                    "--concurrency", "1",
                    "--queue-order", "trial_then_operator",
                ]
            )

        assert rc == 0
        assert starts == ["A:t1", "B:t1", "A:t2", "B:t2", "A:t3", "B:t3"], starts
        summary = json.loads((harness / "regenerate_run_summary.json").read_text())
        assert summary["queue_order"] == "trial_then_operator"

        starts.clear()
        with (
            mock.patch.object(run_regenerate, "make_provider_envs", return_value=slots),
            mock.patch.object(run_regenerate, "run_one", side_effect=fake_run),
        ):
            rc = run_regenerate.main(
                [
                    "--manifest", str(manifest),
                    "--experiment-root", str(root),
                    "--key-config", str(root / "unused.json"),
                    "--npus", "1",
                    "--concurrency", "1",
                    "--queue-order", "reverse_trial_then_operator",
                ]
            )
        assert rc == 0
        assert starts == ["B:t3", "A:t3", "B:t2", "A:t2", "B:t1", "A:t1"], starts
        summary = json.loads((harness / "regenerate_run_summary.json").read_text())
        assert summary["queue_order"] == "reverse_trial_then_operator"
    print("PASS: forward and strict-reverse trial-major queue order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
