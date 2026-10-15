#!/usr/bin/env python3
"""classify_verify_result.py — Verification 结果机器可读分类器。

从 utils/verification_ascendc.py 的 (exit_code, stdout_path, stderr_path) 推断
failure_type，写 {task_dir}/.verify_status/phase{N}_attempt{M}.json + latest.json。

schema_version=1；字段与老的 .eval_status/*.json 产物完全一致，下游
(trace-recorder / Phase 8 subagent / Gate-V) 无需改读取逻辑。

典型用法（主 agent Phase 4.3 / 6）:
    python3 utils/verification_ascendc.py {task_dir} \
        >$TASK_DIR/.verify_logs/phase4_attempt{N}.stdout \
        2>$TASK_DIR/.verify_logs/phase4_attempt{N}.stderr
    rc=$?
    python3 utils/classify_verify_result.py \
        --exit-code $rc \
        --stdout-path $TASK_DIR/.verify_logs/phase4_attempt{N}.stdout \
        --stderr-path $TASK_DIR/.verify_logs/phase4_attempt{N}.stderr \
        --task-dir $TASK_DIR --phase 4 --attempt {N} --write-status
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import signal
import sys
from pathlib import Path

SCHEMA_VERSION = 1


def _utcnow_iso() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _tail(text: str, n: int = 50) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-n:])


# -------- failure_type 分类 regex ---------------------------------------------

_COMPILE_ERR_PATTERNS = [
    r"error: ", r"fatal error:", r"undefined reference",
    r"compile .*failed", r"ascendc.*build.*fail",
]
_IMPORT_ENV_PATTERNS = [
    r"lib(ascend_hal|runtime|torch|torch_npu|torch_cpu|opapi)[^\s]*\.so",
    r"ASCEND_TOOLKIT_HOME", r"LD_LIBRARY_PATH",
    r"cannot open shared object file",
]
_IMPORT_KERNEL_PATTERNS = [
    r"pybind11", r"_ext\.so", r"undefined symbol.*(kernel|Kernel|_do)",
    r"TORCH_EXTENSION_NAME", r"ModuleNotFoundError: No module named .*_ext",
]
_PRECISION_PATTERNS = [r"mismatch_ratio=", r"max_abs_diff="]
_CRASH_SIGNALS = {
    -signal.SIGSEGV: "SIGSEGV",
    -signal.SIGABRT: "SIGABRT",
    -signal.SIGBUS: "SIGBUS",
    -signal.SIGFPE: "SIGFPE",
}
_SSH_DISCONNECT = [
    r"ssh: connect to host", r"Connection (refused|reset|timed out)",
    r"ssh_exchange_identification", r"port 22: Connection",
]
_DOCKER_UNREACHABLE = [
    r"container .* not running", r"Cannot connect to the Docker daemon",
    r"Error response from daemon", r"docker: Error",
]
# NPU kernel 设备端运行异常（aicore / MTE / ACL sync）。Python 进程通常以 rc=1 退出（RuntimeError
# 被 Python 正常抛出而非 POSIX signal），所以 (e) 的 `rc in _CRASH_SIGNALS` 分支抓不到；语义上这
# 属于 kernel runtime 错误，必须落到 runtime_error 以便 Phase 8 debug subagent 接手。
_NPU_RUNTIME_PATTERNS = [
    r"aicore exception",
    r"The instruction configuration of MTE is illegal",
    r"MTE (error|illegal)",
    r"Stream Synchronize failed,\s*stream_id=\d+,\s*retCode=0x\w+",
    r"Fail to synchronize forbbiden stream_id=\d+,\s*retCode=0x\w+",
    r"ACL stream synchronize failed",
    r"Kernel task happen error",
    r"module_name=E[EZ]9999",
    r"E[EZ]9999",
    r"The aicore execution is abnormal",
]


def _match_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text or "") for p in patterns)


def classify_failure(status: dict, proc=None) -> dict:
    """只改 failure_type / failed_step / import_subtype / abort_subtype / exit_signal /
    各阶段子状态。返回 dict 用于 status.update()。

    `proc` 保留为兼容 signature；分类仅依赖 status 内字段。
    """
    stdout = status.get("stdout_tail", "")
    stderr = status.get("stderr_tail", "")
    combined = f"{stdout}\n{stderr}"
    out: dict = {}

    # (a) wrapper 自触发 timeout（最高优先级）
    if status.get("timeout_marker_present"):
        out["failure_type"] = "timeout"
        out["failed_step"] = "execute"
        out["execute"] = {"status": "timeout", "crash_signal": None}
        return out

    rc = status.get("exit_code")

    # (b) success
    if rc == 0 and _match_any([r"PASS", r"all cases passed"], combined):
        out["failure_type"] = "success"
        out["verify"] = {
            "status": "passed", "total_cases": None,
            "passed_cases": None, "failed_cases": [],
        }
        return out
    if rc == 0:
        # verification_ascendc.py 退出 0 即视为 success
        out["failure_type"] = "success"
        return out

    # (c) build_failed
    if _match_any(_COMPILE_ERR_PATTERNS, combined):
        out["failure_type"] = "build_failed"
        out["failed_step"] = "compile"
        out["compile"] = {"status": "failed", "error_summary": _tail(stderr, 10)}
        return out

    # (d) import_failed + 子类
    if _match_any(
        [r"ImportError", r"ModuleNotFoundError", r"OSError: cannot open shared object"],
        combined,
    ):
        out["failure_type"] = "import_failed"
        out["failed_step"] = "import"
        if _match_any(_IMPORT_ENV_PATTERNS, combined):
            out["import_subtype"] = "import_env_side"
        else:
            out["import_subtype"] = "import_kernel_side"
        out["import"] = {"status": "failed", "traceback_path": status.get("log_path")}
        return out

    # (e) runtime_error：明确 crash signal
    if rc in _CRASH_SIGNALS:
        out["failure_type"] = "runtime_error"
        out["failed_step"] = "execute"
        out["exit_signal"] = _CRASH_SIGNALS[rc]
        out["execute"] = {"status": "crashed", "crash_signal": _CRASH_SIGNALS[rc]}
        return out

    # (e.2) runtime_error：NPU 设备端 aicore / MTE / ACL sync 异常
    #       Python 进程抛 RuntimeError 以 rc=1（非负）退出，(e) 的 signal-only 分支抓不到；
    #       但语义上是 kernel runtime 错误，必须落到 runtime_error 以便 Phase 8 debug subagent 接手。
    if rc != 0 and _match_any(_NPU_RUNTIME_PATTERNS, combined):
        out["failure_type"] = "runtime_error"
        out["failed_step"] = "execute"
        out["exit_signal"] = "NPU_AICORE_EXCEPTION"
        out["execute"] = {"status": "crashed", "crash_signal": "NPU_AICORE_EXCEPTION"}
        return out

    # (f) precision_failed：verify 阶段正常退出但数值对比失败
    if _match_any(_PRECISION_PATTERNS, combined) and rc != 0:
        out["failure_type"] = "precision_failed"
        out["failed_step"] = "verify"
        out["verify"] = {
            "status": "failed", "total_cases": None,
            "passed_cases": None, "failed_cases": [],
        }
        return out

    # (g) execution_aborted 兜底
    out["failure_type"] = "execution_aborted"
    if rc == 255 or _match_any(_SSH_DISCONNECT, combined):
        out["abort_subtype"] = "ssh_disconnected"
    elif _match_any(_DOCKER_UNREACHABLE, combined):
        out["abort_subtype"] = "docker_unreachable"
    elif rc is not None and rc < 0:
        out["abort_subtype"] = "killed_by_outer_harness"
        out["exit_signal"] = f"SIGNAL_{-rc}"
    else:
        out["abort_subtype"] = "unknown"
    return out


def build_status(
    *,
    phase: int,
    attempt: int,
    exit_code: int,
    stdout_text: str,
    stderr_text: str,
    stdout_path: Path,
    timeout_marker_present: bool = False,
) -> dict:
    """根据一次 verification 调用的原始结果构造完整 status dict。"""
    now = _utcnow_iso()
    status: dict = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "attempt": attempt,
        "started_at": now,
        "ended_at": now,
        "duration_sec": None,
        "exit_code": exit_code,
        "exit_signal": None,
        "failure_type": None,
        "failed_step": None,
        "log_path": str(stdout_path),
        "stdout_tail": _tail(stdout_text),
        "stderr_tail": _tail(stderr_text),
        "timeout_marker_present": timeout_marker_present,
        "import_subtype": None,
        "abort_subtype": None,
        "compile": {"status": "skipped", "error_summary": None},
        "import": {"status": "skipped", "traceback_path": None},
        "execute": {"status": "skipped", "crash_signal": None},
        "verify": {
            "status": "skipped", "total_cases": None,
            "passed_cases": None, "failed_cases": [],
        },
        "per_case_results": _extract_per_case_results(stdout_text),
    }
    status.update(classify_failure(status))
    return status


def _extract_per_case_results(stdout_text: str) -> list:
    """从 verification_ascendc.py 的 stdout 提取 per-case 结果。

    格式示例：
      case[0]: output[0]: matched
      case[1]: output[1]: dtype(ref=torch.float16, cand=torch.float16), unequal_elements=42, mismatch_ratio=0.123456%, max_abs_diff=0.0625, mean_abs_diff=0.0312

    返回：
      [
        {"case_index": 0, "passed": true, "detail": "output[0]: matched"},
        {"case_index": 1, "passed": false, "detail": "output[1]: dtype(...), unequal_elements=42, ..."},
        ...
      ]
    """
    results = []
    for line in stdout_text.splitlines():
        m = re.match(r"case\[(\d+)\]:\s*(.+)", line.strip())
        if not m:
            continue
        case_idx = int(m.group(1))
        detail = m.group(2).strip()
        passed = "matched" in detail
        results.append({"case_index": case_idx, "passed": passed, "detail": detail})
    return results


def write_status(task_dir: Path, status: dict, phase: int, attempt: int) -> tuple[Path, Path]:
    status_dir = task_dir / ".verify_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    per_path = status_dir / f"phase{phase}_attempt{attempt}.json"
    latest_path = status_dir / "latest.json"
    payload = json.dumps(status, indent=2, ensure_ascii=False)
    per_path.write_text(payload)
    latest_path.write_text(payload)
    return per_path, latest_path


def _read_text_safe(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exit-code", type=int, required=True, help="verification_ascendc.py 的退出码")
    ap.add_argument("--stdout-path", type=Path, required=True, help="verification stdout 日志文件")
    ap.add_argument("--stderr-path", type=Path, required=True, help="verification stderr 日志文件")
    ap.add_argument("--task-dir", type=Path, required=True, help="算子任务目录（含 model.py 等）")
    ap.add_argument("--phase", type=int, required=True, help="4 / 6 / 8")
    ap.add_argument("--attempt", type=int, default=0)
    ap.add_argument(
        "--timeout-marker",
        action="store_true",
        help="外层 timeout 命中时传入（对应 shell `timeout N python3 ...` 的 rc=124）",
    )
    ap.add_argument(
        "--write-status",
        action="store_true",
        help="落盘 {task_dir}/.verify_status/phase{N}_attempt{M}.json + latest.json",
    )
    args = ap.parse_args()

    stdout_text = _read_text_safe(args.stdout_path)
    stderr_text = _read_text_safe(args.stderr_path)

    status = build_status(
        phase=args.phase,
        attempt=args.attempt,
        exit_code=args.exit_code,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        stdout_path=args.stdout_path,
        timeout_marker_present=args.timeout_marker,
    )

    if args.write_status:
        write_status(args.task_dir.resolve(), status, args.phase, args.attempt)

    print(json.dumps(status, indent=2, ensure_ascii=False))

    ft = status.get("failure_type")
    if ft == "success":
        return 0
    if ft == "execution_aborted" and status.get("abort_subtype") == "unknown":
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
