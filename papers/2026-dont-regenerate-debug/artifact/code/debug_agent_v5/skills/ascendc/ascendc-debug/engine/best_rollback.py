"""Crash-consistent current-best checkpoints and kernel source rollback.

The workflow edits ``kernel/`` in place.  This module keeps the best validated
source tree recoverable across process termination:

* checkpoints are built in a temporary directory, hash-verified, then switched
  with a recoverable current/previous rename protocol;
* rollback writes a durable journal containing both desired and previous source
  trees, then replaces each target file atomically;
* an interrupted rollback is completed (or, if the desired tree is corrupt,
  restored) when the engine replays the post-validation action.

Only real source files are protected. ``kernel/build`` remains disposable and
is rebuilt by objective validation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

_HISTORY_DIR = "precision_tuning/history"
_BEST_DIR = f"{_HISTORY_DIR}/current_best"
_BEST_PREVIOUS = f"{_HISTORY_DIR}/.current_best.previous"
_BEST_TMP_PREFIX = ".current_best.tmp."
_BEST_SRC = "src"
_BEST_METRIC = "metric.json"
_BEST_MANIFEST = "manifest.json"
_ROLLBACK_TXN = f"{_HISTORY_DIR}/rollback_transaction"
_SRC_SUFFIXES = (".cpp", ".h", ".hpp", ".py")
_BUILD_DIRNAME = "build"


def _source_files(kernel_dir: Path) -> list[Path]:
    """Return editable source files under kernel/, excluding generated build/."""
    out: list[Path] = []
    if not kernel_dir.is_dir():
        return out
    for path in kernel_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(kernel_dir)
        if _BUILD_DIRNAME in rel.parts or path.name.startswith("."):
            continue
        if path.suffix in _SRC_SUFFIXES:
            out.append(path)
    return sorted(out, key=lambda p: p.as_posix())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _copy_source_tree(source_root: Path, destination: Path) -> list[dict]:
    destination.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for source in _source_files(source_root):
        rel = source.relative_to(source_root)
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest.append({
            "path": rel.as_posix(),
            "size": target.stat().st_size,
            "sha256": _sha256(target),
        })
    return manifest


def _tree_manifest(source_root: Path) -> list[dict]:
    return [
        {
            "path": path.relative_to(source_root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _source_files(source_root)
    ]


def _manifest_matches(source_root: Path, entries: object) -> bool:
    if not isinstance(entries, list) or not entries:
        return False
    expected: dict[str, tuple[int, str]] = {}
    try:
        for entry in entries:
            rel = str(entry["path"])
            expected[rel] = (int(entry["size"]), str(entry["sha256"]))
    except (KeyError, TypeError, ValueError):
        return False
    try:
        actual = {entry["path"]: (entry["size"], entry["sha256"])
                  for entry in _tree_manifest(source_root)}
    except OSError:
        return False
    return actual == expected


def _load_json(path: Path) -> Optional[dict]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _checkpoint_valid(path: Path, *, allow_legacy: bool = True) -> bool:
    src = path / _BEST_SRC
    metric = _load_json(path / _BEST_METRIC)
    if not src.is_dir() or metric is None or not _source_files(src):
        return False
    manifest = _load_json(path / _BEST_MANIFEST)
    if manifest is None:
        return allow_legacy
    return (
        manifest.get("complete") is True
        and manifest.get("attempt") == metric.get("attempt")
        and _manifest_matches(src, manifest.get("files"))
    )


def _recover_checkpoint(task_dir: Path) -> Optional[Path]:
    """Recover current_best after a process dies during directory switching."""
    task_dir = Path(task_dir)
    current = task_dir / _BEST_DIR
    previous = task_dir / _BEST_PREVIOUS
    history = task_dir / _HISTORY_DIR
    if _checkpoint_valid(current):
        if previous.exists():
            shutil.rmtree(previous, ignore_errors=True)
        if history.is_dir():
            for stale in history.iterdir():
                if stale.name.startswith(_BEST_TMP_PREFIX):
                    shutil.rmtree(stale, ignore_errors=True)
        return current
    if _checkpoint_valid(previous):
        if current.exists():
            shutil.rmtree(current, ignore_errors=True)
        os.replace(previous, current)
        return current
    if history.is_dir():
        candidates = sorted(
            (p for p in history.iterdir()
             if p.name.startswith(_BEST_TMP_PREFIX) and _checkpoint_valid(p, allow_legacy=False)),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
        if candidates:
            if current.exists():
                shutil.rmtree(current, ignore_errors=True)
            os.replace(candidates[0], current)
            return current
    return None


def _metric_key(metric: dict) -> tuple[float, float]:
    def finite_or_worst(value) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return -1.0
        return parsed if math.isfinite(parsed) else -1.0

    case = finite_or_worst(metric.get("case_pass_rate"))
    match = finite_or_worst(metric.get("match_rate"))
    return (case, match)


def read_attempt_metric(task_dir: Path, attempt: int) -> Optional[dict]:
    path = Path(task_dir) / "precision_tuning" / f"validation_result_attempt_{attempt}.json"
    data = _load_json(path)
    if data is None:
        return None
    return {
        "attempt": attempt,
        "case_pass_rate": data.get("case_pass_rate"),
        "match_rate": data.get("match_rate"),
        "correctness_passed": data.get("correctness_passed"),
    }


def read_best_metric(task_dir: Path) -> Optional[dict]:
    best = _recover_checkpoint(Path(task_dir))
    return _load_json(best / _BEST_METRIC) if best is not None else None


def checkpoint_is_valid(task_dir: Path) -> bool:
    return _recover_checkpoint(Path(task_dir)) is not None


def _rollback_record(event: dict) -> Optional[dict]:
    if event.get("type") == "rollback" and event.get("from_attempt") is not None:
        return event
    if event.get("type") != "action_completed":
        return None
    action = event.get("action") or {}
    result = event.get("result") or {}
    if action.get("step") != "checkpoint_and_rollback" or not result.get("rolled_back"):
        return None
    return {
        "type": "rollback",
        "from_attempt": result.get("from_attempt"),
        "best_attempt": result.get("best_attempt"),
        "best_case_pass_rate": result.get("best_case_pass_rate"),
        "best_match_rate": result.get("best_match_rate"),
        "source": "checkpoint_and_rollback_action",
    }


def latest_rollback_record(task_dir: Path) -> Optional[dict]:
    try:
        from engine.events import read_events
        records = [record for event in read_events(Path(task_dir))
                   if (record := _rollback_record(event)) is not None]
    except (OSError, TypeError, ValueError):
        return None
    return records[-1] if records else None


def _latest_rollback_attempt(task_dir: Path) -> Optional[int]:
    record = latest_rollback_record(task_dir)
    try:
        return int(record["from_attempt"]) if record is not None else None
    except (KeyError, TypeError, ValueError):
        return None


def checkpoint_allowed(gate_result) -> bool:
    if gate_result is None:
        return False
    checks = getattr(gate_result, "checks", None) or {}
    if checks.get("anticheat_pass") is False:
        return False
    if checks.get("cpp_regression_pass") is False:
        return False
    if checks.get("ast_degrade_pass") is False and not checks.get("ast_validator_errored"):
        return False
    return True


def ensure_current_best(
    task_dir: Path,
    attempt: int,
    metric: Optional[dict] = None,
    *,
    force: bool = False,
) -> dict:
    """Build and atomically activate a verified source checkpoint."""
    task_dir = Path(task_dir)
    metric = metric or read_attempt_metric(task_dir, attempt)
    if metric is None:
        return {"success": False, "updated": False, "error": "missing_attempt_metric"}
    best = read_best_metric(task_dir)
    if not force and best is not None and _metric_key(metric) <= _metric_key(best):
        return {"success": True, "updated": False, "best_metric": best}
    kernel = task_dir / "kernel"
    if not _source_files(kernel):
        return {"success": False, "updated": False, "error": "missing_kernel_sources"}

    history = task_dir / _HISTORY_DIR
    history.mkdir(parents=True, exist_ok=True)
    temp = history / f"{_BEST_TMP_PREFIX}{uuid.uuid4().hex}"
    current = task_dir / _BEST_DIR
    previous = task_dir / _BEST_PREVIOUS
    try:
        files = _copy_source_tree(kernel, temp / _BEST_SRC)
        _atomic_json(temp / _BEST_METRIC, metric)
        _atomic_json(temp / _BEST_MANIFEST, {
            "schema_version": 1,
            "complete": True,
            "attempt": metric.get("attempt", attempt),
            "files": files,
        })
        if not _checkpoint_valid(temp, allow_legacy=False):
            raise OSError("checkpoint manifest verification failed")
        if previous.exists():
            shutil.rmtree(previous)
        if current.exists():
            os.replace(current, previous)
        try:
            os.replace(temp, current)
        except BaseException:
            if not current.exists() and _checkpoint_valid(previous):
                os.replace(previous, current)
            raise
        if not _checkpoint_valid(current, allow_legacy=False):
            raise OSError("activated checkpoint verification failed")
        if previous.exists():
            shutil.rmtree(previous)
        return {"success": True, "updated": True, "best_metric": metric}
    except (OSError, shutil.Error) as exc:
        try:
            _recover_checkpoint(task_dir)
        except OSError:
            pass
        return {"success": False, "updated": False, "error": str(exc)}
    finally:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)


def save_current_best(task_dir: Path, attempt: int, metric: Optional[dict] = None) -> bool:
    return bool(ensure_current_best(task_dir, attempt, metric).get("updated"))


def should_rollback(task_dir: Path, current_attempt: int) -> bool:
    if current_attempt < 1:
        return False
    best = read_best_metric(task_dir)
    if best is None:
        return False
    try:
        best_attempt = int(best.get("attempt"))
    except (TypeError, ValueError):
        return False
    if current_attempt - best_attempt < 2:
        return False
    current = read_attempt_metric(task_dir, current_attempt)
    previous = read_attempt_metric(task_dir, current_attempt - 1)
    if current is None or previous is None:
        return False
    latest = _latest_rollback_attempt(task_dir)
    if latest is not None and current_attempt - 1 <= latest:
        return False
    best_key = _metric_key(best)
    return _metric_key(current) <= best_key and _metric_key(previous) <= best_key


def _write_file_atomically(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.rollback.{uuid.uuid4().hex}")
    try:
        shutil.copy2(source, temp)
        os.replace(temp, target)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _prepare_rollback(task_dir: Path, best_dir: Path, metric: dict) -> Path:
    txn = task_dir / _ROLLBACK_TXN
    if txn.exists():
        return txn
    temp = txn.with_name(f".{txn.name}.tmp.{uuid.uuid4().hex}")
    try:
        desired_files = _copy_source_tree(best_dir / _BEST_SRC, temp / "desired")
        before_files = _copy_source_tree(task_dir / "kernel", temp / "before")
        _atomic_json(temp / "journal.json", {
            "schema_version": 1,
            "state": "prepared",
            "best_metric": metric,
            "desired_files": desired_files,
            "before_files": before_files,
        })
        if not _manifest_matches(temp / "desired", desired_files):
            raise OSError("rollback desired tree verification failed")
        os.replace(temp, txn)
        return txn
    finally:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)


def _apply_tree(kernel: Path, source: Path, entries: list[dict]) -> None:
    desired = {str(entry["path"]) for entry in entries}
    for entry in entries:
        rel = Path(str(entry["path"]))
        _write_file_atomically(source / rel, kernel / rel)
    for current in _source_files(kernel):
        if current.relative_to(kernel).as_posix() not in desired:
            current.unlink()
    if not _manifest_matches(kernel, entries):
        raise OSError("kernel source verification failed after rollback")


def recover_pending_rollback(task_dir: Path) -> Optional[dict]:
    """Finish an interrupted rollback, restoring the previous tree if journal is corrupt."""
    task_dir = Path(task_dir)
    txn = task_dir / _ROLLBACK_TXN
    if not txn.is_dir():
        return None
    journal = _load_json(txn / "journal.json")
    if journal is None:
        return None
    desired = journal.get("desired_files")
    before = journal.get("before_files")
    try:
        if not _manifest_matches(txn / "desired", desired):
            if _manifest_matches(txn / "before", before):
                _apply_tree(task_dir / "kernel", txn / "before", before)
                # The staged desired copy is corrupt, but current_best remains
                # independently hash-verified.  Remove this transaction so
                # do_rollback() can immediately prepare a fresh one from best.
                shutil.rmtree(txn)
            return None
        journal["state"] = "applying"
        _atomic_json(txn / "journal.json", journal)
        _apply_tree(task_dir / "kernel", txn / "desired", desired)
        journal["state"] = "committed"
        _atomic_json(txn / "journal.json", journal)
        metric = journal.get("best_metric")
        shutil.rmtree(txn)
        return metric if isinstance(metric, dict) else None
    except (OSError, shutil.Error):
        return None


def do_rollback(task_dir: Path) -> Optional[dict]:
    task_dir = Path(task_dir)
    recovered = recover_pending_rollback(task_dir)
    if recovered is not None:
        return recovered
    best_dir = _recover_checkpoint(task_dir)
    metric = read_best_metric(task_dir)
    if best_dir is None or metric is None:
        return None
    try:
        _prepare_rollback(task_dir, best_dir, metric)
    except (OSError, shutil.Error):
        return None
    return recover_pending_rollback(task_dir)


def process_validation_checkpoint(task_dir: Path, attempt: int, gate_result) -> dict:
    """Idempotent post-validate action used as the engine recovery boundary."""
    task_dir = Path(task_dir)
    checkpoint = {"success": True, "updated": False}
    if checkpoint_allowed(gate_result):
        checkpoint = ensure_current_best(task_dir, attempt)
        if not checkpoint.get("success"):
            return {"success": False, "checkpoint": checkpoint,
                    "error": checkpoint.get("error")}
    rolled_back = False
    best = read_best_metric(task_dir)
    if should_rollback(task_dir, attempt):
        best = do_rollback(task_dir)
        if best is None:
            return {"success": False, "checkpoint": checkpoint,
                    "error": "rollback_transaction_incomplete"}
        rolled_back = True
    return {
        "success": True,
        "checkpoint": checkpoint,
        "rolled_back": rolled_back,
        "from_attempt": attempt if rolled_back else None,
        "best_attempt": best.get("attempt") if best else None,
        "best_case_pass_rate": best.get("case_pass_rate") if best else None,
        "best_match_rate": best.get("match_rate") if best else None,
    }
