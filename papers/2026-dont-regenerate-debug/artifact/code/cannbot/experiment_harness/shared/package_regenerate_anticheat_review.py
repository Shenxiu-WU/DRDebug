#!/usr/bin/env python3
"""Package full-eval-PASS/anti-cheat-CHEAT Regenerate candidates for review.

This is intentionally read-only with respect to experiment directories.  It
selects the Qwen-only 27x3 cohort (main Qwen rows plus the supplementary Qwen
reruns that replace original K3 rows), then copies the small set of evidence
needed for a human purity audit into one tarball.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable


SOURCE_SUFFIXES = {
    ".asc",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".py",
    ".txt",
    ".json",
}

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    "build",
    "build_out",
    "logs",
    "node_modules",
    "provider_abort_snapshots",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_run_records(root: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    """Yield canonical rows enriched from the audited token summary.

    The experiment stores ``run_record.json`` inside every trial directory; its
    operator/trial/model identity lives in the manifest/token summary rather
    than in the small mutable runtime record itself.
    """

    summary = load_json(root / "experiment_harness" / "token_cost_summary.json")
    for row in summary["rows"]:
        cwd = Path(str(row["cwd"]))
        path = cwd / "run_record.json"
        runtime = load_json(path) if path.is_file() else {}
        record = dict(runtime)
        record.update(
            {
                "run_id": row["run_id"],
                "operator": row["operator"],
                "trial": row["trial"],
                "model": row.get("model"),
                "candidate_dir": str(cwd),
                "cwd": str(cwd),
            }
        )
        yield path, record


def run_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["operator"]), int(record["trial"])


def is_qwen(record: dict[str, Any]) -> bool:
    return "qwen3.8-max-preview" in str(record.get("model") or "")


def verdict(record: dict[str, Any], filename: str) -> dict[str, Any]:
    candidate_dir = Path(str(record["candidate_dir"]))
    path = candidate_dir / "validation" / filename
    return load_json(path)


def copy_file(path: Path, destination: Path, root: Path | None = None) -> None:
    if not path.is_file():
        return
    relative = path.name if root is None else path.relative_to(root)
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)


def copy_tree_sources(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if source.is_file():
        copy_file(source, destination)
        return
    for dirpath, dirnames, filenames in os.walk(source):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        directory = Path(dirpath)
        for filename in filenames:
            path = directory / filename
            if path.suffix.lower() in SOURCE_SUFFIXES or filename in {
                "CMakeLists.txt",
                "Makefile",
            }:
                copy_file(path, destination, source)


def resolve_operator_dir(candidate_dir: Path) -> Path | None:
    marker = candidate_dir / "candidate_adapter" / "operator_dir.txt"
    if marker.is_file():
        raw = marker.read_text(encoding="utf-8").strip()
        if raw:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = candidate_dir / raw
            if candidate.exists():
                return candidate.resolve()
    task_dir = candidate_dir / "candidate_task"
    if task_dir.exists():
        return task_dir.resolve()
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection",
        choices=("pass-cheat", "all-pass"),
        default="pass-cheat",
        help="Package only PASS/CHEAT rows or every full-eval PASS row.",
    )
    args = parser.parse_args()

    # Supplement rows override the matching original K3 rows.
    combined: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    for path, record in iter_run_records(args.main_root):
        if is_qwen(record):
            combined[run_key(record)] = (path, record)
    for path, record in iter_run_records(args.supplement_root):
        if is_qwen(record):
            combined[run_key(record)] = (path, record)

    selected: list[tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for _, (record_path, record) in sorted(combined.items()):
        full_eval = verdict(record, "full_eval.json")
        anticheat = verdict(record, "anticheat.json")
        full_eval_pass = (
            str(full_eval.get("full_eval_result", full_eval.get("verdict", ""))).upper()
            == "PASS"
        )
        anticheat_cheat = str(anticheat.get("verdict", "")).upper() == "CHEAT"
        if full_eval_pass and (args.selection == "all-pass" or anticheat_cheat):
            selected.append((record_path, record, full_eval, anticheat))

    if len(combined) != 81:
        raise SystemExit(f"expected 81 Qwen cohort rows, got {len(combined)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="regen-anticheat-review-") as tmp:
        stage = Path(tmp) / "regenerate_anticheat_review"
        stage.mkdir()
        manifest: list[dict[str, Any]] = []

        for record_path, record, full_eval, anticheat in selected:
            run_id = str(record.get("run_id") or record_path.stem)
            safe_run_id = run_id.replace("/", "__").replace(":", "_")
            out = stage / safe_run_id
            out.mkdir()
            shutil.copy2(record_path, out / "run_record.json")

            candidate_dir = Path(str(record["candidate_dir"])).resolve()
            copy_tree_sources(candidate_dir / "candidate_adapter", out / "candidate_adapter")
            copy_tree_sources(candidate_dir / "validation", out / "validation")
            copy_tree_sources(candidate_dir / "candidate_task", out / "candidate_task")

            operator_dir = resolve_operator_dir(candidate_dir)
            if operator_dir is not None:
                copy_tree_sources(operator_dir, out / "resolved_operator_dir")

            manifest.append(
                {
                    "run_id": run_id,
                    "operator": record["operator"],
                    "trial": record["trial"],
                    "model": record.get("model"),
                    "candidate_dir": str(candidate_dir),
                    "resolved_operator_dir": str(operator_dir) if operator_dir else None,
                    "full_eval_verdict": full_eval.get(
                        "full_eval_result", full_eval.get("verdict")
                    ),
                    "full_eval_failure_class": full_eval.get(
                        "final_failure_type", full_eval.get("failure_class")
                    ),
                    "anticheat_verdict": anticheat.get("verdict"),
                    "anticheat_official_verdict": anticheat.get("official_verdict"),
                    "anticheat_reasons": anticheat.get("reasons"),
                    "compatibility_override": anticheat.get("compatibility_override"),
                }
            )

        (stage / "manifest.json").write_text(
            json.dumps(
                {
                    "qwen_cohort_rows": len(combined),
                    "selection": args.selection,
                    "selected_rows": len(selected),
                    "runs": manifest,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        with tarfile.open(args.output, "w:gz") as archive:
            archive.add(stage, arcname=stage.name)

    print(
        json.dumps(
            {
                "qwen_cohort_rows": len(combined),
                "selection": args.selection,
                "selected_rows": len(selected),
                "output": str(args.output),
                "output_bytes": args.output.stat().st_size,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
