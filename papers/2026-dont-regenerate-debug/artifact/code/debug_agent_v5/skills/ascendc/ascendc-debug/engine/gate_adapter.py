"""gate_adapter.py — 把现有 gate 输出包装为引擎可消费的纯判定 GateResult。

设计 (REWRITE_PLAN §2.4):
  - **不重算 loop_signal**: 直接消费 branch_*.py 已算好的 loop_signal/loop_reason/
    stop_reason_code。引擎只读，loop 逻辑保留在现有 gate 内 (修正历史误判)。
  - 双入口、解析与调用解耦:
      parse_gate_output(dict) -> GateResult   纯函数 (无副作用，UT 主测面)
      run_gate(...) -> GateResult             subprocess 跑稳定 CLI (生产路径)

为何 subprocess 而非 import branch 类:
  precision_gate.py 是稳定 CLI 契约 (--step/--op-name/--task-name/--task-dir/--attempt)，
  内含 2 层路由 (common 前置校验 + dispatch 按 failure_type 自动切分支)。subprocess
  复用这整条路由、不让引擎重实现，且与 agent 现行用法一致、隔离 sys.path 副作用。
  退出码语义: 0=passed / 1=failed / 2=prerequisite_error。

两种输出形状差异 (已读源码核实):
  - precision Gate-V: stop_reason_code / prerequisite_error / attempt / max_attempts
    被 _legacy_to_outcome 塞进 checks dict (顶层只有 gate/passed/checks/loop_signal/loop_reason)。
  - build/import/runtime/timeout: checks 里无上述键，loop_signal 来自 GateOutcome 直出。
  parse_gate_output 吸收此差异，把散落在 checks 的键提升到 GateResult 顶层字段。
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_ENGINE_DIR = Path(__file__).resolve().parent
# engine/ 的上一级是 ascendc-debug/，gate 脚本在其 scripts/ 下。
_GATE_SCRIPT = _ENGINE_DIR.parent / "scripts" / "precision_gate.py"

# precision gate 把这些诊断键塞进 checks，解析时提升到顶层。
_PROMOTED_FROM_CHECKS = ("stop_reason_code", "prerequisite_error", "attempt", "max_attempts")


@dataclass
class GateResult:
    """五分支 gate 输出的统一只读投影。loop_signal 等均直接取自 gate，不重算。"""

    gate: str
    passed: bool
    loop_signal: Optional[str] = None        # PASS / CONTINUE / STOP / None (forensics/audit step)
    loop_reason: Optional[str] = None
    stop_reason_code: Optional[str] = None    # precision 分支专有 (见模块 docstring)
    prerequisite_error: Optional[str] = None  # 前置校验失败原因 (退出码 2)
    attempt: Optional[int] = None
    max_attempts: Optional[int] = None
    # Objective validation may move the task to another debug branch.  This is
    # attached by runner after Gate-V and must survive event replay so the next
    # attempt does not inherit a stale branch from the previous attempt.
    failure_type: Optional[str] = None
    checks: dict = field(default_factory=dict)


def parse_gate_output(raw: dict) -> GateResult:
    """纯函数: gate 的 JSON dict → GateResult。无副作用，不重算任何信号。

    顶层字段优先；缺失时回退到 checks 里的同名键 (precision 分支把诊断塞 checks)。
    """
    checks = dict(raw.get("checks") or {})

    def pick(key: str):
        if key in raw and raw[key] is not None:
            return raw[key]
        return checks.get(key)

    return GateResult(
        gate=raw.get("gate", ""),
        passed=bool(raw.get("passed", False)),
        loop_signal=raw.get("loop_signal"),
        loop_reason=raw.get("loop_reason"),
        stop_reason_code=pick("stop_reason_code"),
        prerequisite_error=pick("prerequisite_error"),
        attempt=pick("attempt"),
        max_attempts=pick("max_attempts"),
        failure_type=pick("failure_type"),
        checks=checks,
    )


def _extract_first_json(stdout: str) -> dict:
    """从 CLI stdout 提取首个 JSON object。

    precision_gate.py main() 先 print(json.dumps(result, indent=2)) 再打印人类可读行
    (✅ PASSED 等)，故不能直接 json.loads 整段。用 raw_decode 从首个 '{' 解析一个
    完整 object，忽略其后内容。
    """
    s = stdout.lstrip()
    start = s.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in gate stdout: {stdout[:200]!r}")
    obj, _ = json.JSONDecoder().raw_decode(s[start:])
    if not isinstance(obj, dict):
        raise ValueError(f"gate stdout first JSON is not an object: {type(obj).__name__}")
    return obj


def run_gate(
    task_dir: Path,
    *,
    step: str,
    op_name: str,
    attempt: int,
    task_name: Optional[str] = None,
    timeout: Optional[float] = None,
) -> GateResult:
    """subprocess 跑 precision_gate.py，解析 stdout 为 GateResult。

    step ∈ {forensics, audit, fix, validate}。task_name 缺省时用 task_dir 末段
    (CLI 的 --task-name 是必填项，仅用于默认 task_dir 推断；这里既传 --task-dir
    又补 --task-name 以满足 argparse required)。

    退出码 2 (prerequisite_error) 不视为异常——gate 已在 JSON 里给出 prerequisite_error，
    解析即可。仅当无法解析 stdout 时才抛。
    """
    task_dir = Path(task_dir)
    cmd = [
        sys.executable, str(_GATE_SCRIPT),
        "--step", step,
        "--op-name", op_name,
        "--task-name", task_name or task_dir.name,
        "--task-dir", str(task_dir),
        "--attempt", str(attempt),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    raw = _extract_first_json(proc.stdout)
    result = parse_gate_output(raw)
    # 不变量校验: precision_gate.py 以退出码 2 退出时必定在 JSON 给出 prerequisite_error
    # (precision_gate.py main() L161-163)。若契约被破坏 (码 2 但无该字段)，fail-loud，
    # 不让引擎拿着不自洽的 gate 输出继续派发。
    if proc.returncode == 2 and not result.prerequisite_error:
        raise ValueError(
            f"gate exited 2 but no prerequisite_error in output (contract broken): {raw}"
        )
    return result
