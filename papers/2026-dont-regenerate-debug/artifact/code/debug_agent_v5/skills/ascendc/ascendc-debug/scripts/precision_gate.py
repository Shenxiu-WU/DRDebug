#!/usr/bin/env python3
"""
precision_gate.py — AscendC debug Gate 2 层路由入口

架构（findings.md §3.3 ①）:
  - 通用层 (gates.common): 反作弊 / AST / 结构 / verify_status / audit 文件
  - 分支层 (gates.branch_*): 按 .verify_status/latest.json 的 failure_type 派发
      * precision_failed → PrecisionBranch
      * build_failed     → BuildBranch
      * import_failed    → ImportBranch (kernel_side only)
      * runtime_error    → RuntimeBranch
      * timeout          → TimeoutBranch

路由顺序:
  1. 跑 gates.common.run_common(step, task_dir, op_name, attempt)
     - 失败直接返回（不再进分支层）
  2. 读 verify_status.latest.json 取 failure_type
     - 不存在或缺失 → 默认 precision_failed (向后兼容)
  3. 分支派发到 run_gate_f/a/v(task_dir, attempt)
     - fix step 不单独 Gate (findings §3.3)，实际复用 audit branch method

CLI 保持向后兼容（--step forensics|audit|fix|validate --op-name --task-name --attempt --task-dir）。
"""

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent.parent.parent.parent   # AscendOpGenAgent/

# 让 `from gates import ...` / `from verify_status import ...` 能工作，不管调用方 PYTHONPATH
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from gates import common as _gate_common  # noqa: E402
from gates.branch_precision import PrecisionBranch  # noqa: E402
from gates.branch_build import BuildBranch  # noqa: E402
from gates.branch_import import ImportBranch  # noqa: E402
from gates.branch_runtime import RuntimeBranch  # noqa: E402
from gates.branch_timeout import TimeoutBranch  # noqa: E402

try:
    from verify_status import load_latest_status  # type: ignore  # noqa: E402
except ImportError:  # pragma: no cover — fallback when verify_status 尚未落盘
    def load_latest_status(task_dir):  # type: ignore
        raise FileNotFoundError("verify_status not importable")


# 路由表: failure_type -> Branch class. 默认 PrecisionBranch 以向后兼容。
_BRANCHES = {
    "precision_failed": PrecisionBranch,
    "build_failed":     BuildBranch,
    "import_failed":    ImportBranch,
    "runtime_error":    RuntimeBranch,
    "timeout":          TimeoutBranch,
}


def _select_branch(failure_type, op_name: str):
    """根据 failure_type 选择分支实例。只有 PrecisionBranch 需要 op_name。"""
    cls = _BRANCHES.get(failure_type or "precision_failed", PrecisionBranch)
    if cls is PrecisionBranch:
        return cls(op_name)
    return cls()


def _load_failure_type(task_dir: Path):
    """从 .verify_status/latest.json 取 failure_type；失败返回 None (默认精度分支)。"""
    try:
        status = load_latest_status(task_dir)
        return status.get("failure_type")
    except Exception:
        return None


def _session_branch_path(task_dir: Path) -> Path:
    return task_dir / ".verify_status" / "session_branch.json"


def _record_session_branch(task_dir: Path, failure_type: str) -> None:
    """首次 Gate 调用时将 session 起始 failure_type 写入 session_branch.json（追踪用）。"""
    path = _session_branch_path(task_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"session_failure_type": failure_type}, ensure_ascii=False))


def _load_session_branch(task_dir: Path):
    """读取 session_branch.json；不存在返回 None。"""
    path = _session_branch_path(task_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("session_failure_type")
    except Exception:
        return None


def _merge_common_checks(branch_output: dict, common_outcome) -> dict:
    """Preserve common-layer evidence when branch gate decides the loop signal.

    Common failures are returned before branch dispatch.  On common pass, branch
    output remains authoritative for gate/loop_signal, while checks keep both
    common invariants (AST/hash/C++/structure) and branch-specific facts.
    """
    merged = dict(branch_output)
    checks = {}
    if isinstance(common_outcome.checks, dict):
        checks.update(common_outcome.checks)
    branch_checks = branch_output.get("checks")
    if isinstance(branch_checks, dict):
        checks.update(branch_checks)
    merged["checks"] = checks
    return merged


def _dispatch(step: str, task_dir: Path, op_name: str, attempt: int) -> dict:
    """主调度：通用层 → 分支层。返回单个 dict (gate 输出)。"""
    current_failure_type = _load_failure_type(task_dir)
    if step in {"audit", "fix"}:
        _gate_common.ensure_engine_audit(
            task_dir,
            attempt,
            current_failure_type or "precision_failed",
        )

    # 1. 通用层
    common_outcome = _gate_common.run_common(step, task_dir, op_name, attempt)
    if not common_outcome.ok:
        return common_outcome.to_gate_output()

    # 2. 分支派发
    # 首次 Gate 调用时记录 session 起始 failure_type（追踪用，不用于 STOP）
    if _load_session_branch(task_dir) is None:
        _record_session_branch(task_dir, current_failure_type or "unknown")

    # 派发 failure_type 选择：
    # - success 时回退到 session 起始分支（各分支 run_gate_v 均有 failure_type==success→PASS 检查；
    #   精度分支需走 PrecisionBranch 以更新 round_summary/tuning_directions）
    # - 其他情况始终按当前 failure_type 派发（failure_type 变化时自动切换分支继续）
    if current_failure_type == "success":
        dispatch_type = _load_session_branch(task_dir) or "precision_failed"
    else:
        dispatch_type = current_failure_type

    branch = _select_branch(dispatch_type, op_name)

    # step → branch method 映射
    # fix 不单独 gate（findings §3.3）；为向后兼容保留 choice，实际复用 audit 分支逻辑
    method_map = {
        "forensics": getattr(branch, "run_gate_f", None),
        "audit":     getattr(branch, "run_gate_a", None),
        "fix":       getattr(branch, "run_gate_a", None),
        "validate":  getattr(branch, "run_gate_v", None),
    }
    method = method_map.get(step)
    if method is None:
        # 未知 step 时返回通用层结果
        return common_outcome.to_gate_output()

    branch_outcome = method(task_dir=task_dir, attempt=attempt)
    return _merge_common_checks(branch_outcome.to_gate_output(), common_outcome)


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="AscendC debug Gate 验证 (2 层路由)")
    parser.add_argument("--step", required=True,
                        choices=["forensics", "audit", "fix", "validate"])
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--task-name", required=True, help="task 目录名")
    parser.add_argument("--task-dir", default=None, help="task 绝对路径，默认 {REPO_ROOT}/{task_name}")
    parser.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()

    task_dir = Path(args.task_dir) if args.task_dir else (REPO_ROOT / args.task_name)
    result = _dispatch(args.step, task_dir, args.op_name, args.attempt)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 前置依赖失败用返回码 2 (致命), 区别于产物不完整的返回码 1
    if result.get("prerequisite_error"):
        print(f"\n[{result['gate']}] ⛔ PREREQUISITE FAILED — {result['prerequisite_error']}")
        sys.exit(2)

    if result.get("passed"):
        print(f"\n[{result['gate']}] ✅ PASSED")
        if args.step == "validate":
            print(f"  loop_signal: {result.get('loop_signal')}")
            print(f"  reason: {result.get('loop_reason')}")
        sys.exit(0)
    else:
        checks = result.get("checks", {})
        failed = [k for k, v in checks.items() if isinstance(v, bool) and not v]
        print(f"\n[{result.get('gate')}] ❌ FAILED — missing: {failed}")
        if args.step == "validate":
            print(f"  loop_signal: {result.get('loop_signal')}")
            print(f"  reason: {result.get('loop_reason')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
