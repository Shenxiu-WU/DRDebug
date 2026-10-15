"""types.py — debug 引擎的闭集 dataclass 与 Literal。

对齐 lingxi runtime/engine/types.py 的「闭集决策」理念，但裁剪到 debug session 所需:
不含 spawn_workflow / local_action / SubworkflowOutcome / orchestrator 概念。

闭集来源 (均已对照现有代码核实，非臆造):
    FailureType    ← utils/classify_verify_result.py::classify_failure 的 out["failure_type"]
    ImportSubtype  ← 同文件 out["import_subtype"]
    AbortSubtype   ← 同文件 out["abort_subtype"]
    SessionOutcome ← agents/ascendc-debug-agent-{constructive,discovery}.md Step 7 退出产物
    LoopSignal     ← scripts/gates/branch_*.py 的 loop_signal 字段
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# failure_type 闭集 — 单 session 内可自动切换 (漂移=续跑)。
# 五个「可自动修复」分支 + success + execution_aborted (不可修复，归 crashed/abort)。
# 流水线偏序 (用 classify_failure 判定顺序核实):
#   build_failed → import_failed → runtime_error → precision_failed
#   (timeout 与 runtime 同级，单独成支; success/execution_aborted 是终态信号)
# ---------------------------------------------------------------------------
FailureType = Literal[
    "build_failed",
    "import_failed",
    "runtime_error",
    "timeout",
    "precision_failed",
    "success",
    "execution_aborted",
]

# 五个可被引擎驱动自动修复的分支 (success/execution_aborted 不在内)。
DEBUGGABLE_FAILURE_TYPES: tuple[str, ...] = (
    "build_failed",
    "import_failed",
    "runtime_error",
    "timeout",
    "precision_failed",
)

# 流水线偏序中各分支的「靠前程度」(数值越小越靠前)。
# 跨分支计数重置时用它判定「比新状态更靠前」的分支 (Step 2 state.py 消费)。
# timeout 与 runtime 同处第 2 阶段 (都属执行期故障)。precision 永远最末 → 计数永不被重置。
PIPELINE_ORDER: dict[str, int] = {
    "build_failed": 0,
    "import_failed": 1,
    "runtime_error": 2,
    "timeout": 2,
    "precision_failed": 3,
}

ImportSubtype = Literal["import_env_side", "import_kernel_side"]

AbortSubtype = Literal[
    "ssh_disconnected",
    "docker_unreachable",
    "killed_by_outer_harness",
    "unknown",
]

# Gate 已算好的 loop 信号 (引擎只读不重算，见 REWRITE_PLAN §2.4)。
LoopSignal = Literal["PASS", "CONTINUE", "STOP"]

# ---------------------------------------------------------------------------
# session 终态闭集 — 统一 schema。
# 取 constructive (8 值) 为基准；discovery 多出的 progressed_to_new_failure_type
# 是旧「漂移=session 结束」语义的产物，在新「漂移=续跑」模型下不作为单 session 终态
# (REWRITE_PLAN §2.2)，故不纳入闭集。Step 8 同步修订 discovery.md + 情景总览.md。
# stopped_by_budget (6.11 修复 4b-B): 跨 attempt 累计 turns 超任务级硬上限而停
# (用 num_turns——turns 模型无关，与 --model 解耦)。
# degenerate_no_progress (6.11 N6): 连续 N 轮既无 match_rate 改善、又命中作弊/audit
# 缺产物兜底 (12a + 修复4 的 CONTINUE 叠加)，退化空转早停，防烧满 branch_cap 预算。
# provider_api_error: diagnose 步 provider API 错误而 Abort (runner._run_main_loop)，
# 经 session_aborted.details.session_outcome 落入 debug_status，须在闭集内 (有退出码 8)。
# ---------------------------------------------------------------------------
SessionOutcome = Literal[
    "success",                    # 验证全过 (含 .json.bak 全量门)
    "failed",                     # 修复未收敛 (一般兜底)
    "stopped_by_gate",            # Gate 前置/不变量校验失败而停
    "stopped_by_loop_limit",      # 旧产物兼容；新代码使用下列两个精确 outcome
    "stopped_by_attempt_limit",   # 撞全局 MAX_ATTEMPTS 而停
    "stopped_by_branch_limit",    # 撞单 failure_type 分支硬上限而停
    "stopped_by_budget",          # 跨 attempt 累计 turns 撞任务级硬上限而停 (4b-B)
    "degenerate_no_progress",    # 连续 N 轮退化空转 (作弊/缺产物兜底无改善) 而停 (N6)
    "timeout",                    # wall-clock 超时主动终止
    "skipped_env_issue",          # import_env_side 等环境问题，非 kernel 可修
    "skipped_unsupported_type",  # failure_type 不在白名单
    "crashed",                    # 不可恢复错误 / 必填前置缺失
    "provider_api_error",         # diagnose 步 provider API 错误而 Abort (退出码 8)
    "ablation_violation",         # Agent 使用了当前 arm 明确禁用的能力
]


# ---------------------------------------------------------------------------
# 决策闭集 — next_action(state) 的返回类型。
# Action: 引擎要执行的一步 (spawn_agent 做诊断+修复 / py_action 跑确定性脚本)。
# Continue: attempt 边界 (落 attempt_started 事件，推进轮次 + 切换 failure_type)。
# Done/Abort/Escalate: 终态，触发退出产物生成。
# ---------------------------------------------------------------------------
@dataclass
class Action:
    """引擎要执行的一步。

    kind="spawn_agent": runner 拉起 constructive/discovery agent worker 做一轮
        诊断+修复 (取证解读 / Phase A-C 审计 / 改 kernel)。worker 内部 self-repair
        (就地编译 ≤3 次) 不回 runner。
    kind="py_action": runner 直接跑确定性脚本 (forensics / verify+classify / gate)。

    step: 对应 SKILL.md 的步骤标识 (如 "forensics"/"audit"/"fix"/"validate")，
        用于路由到 precision_gate.py --step，以及事件可读性。
    """

    kind: Literal["spawn_agent", "py_action"]
    name: str
    step: Optional[str] = None
    skill_args: Optional[dict] = None
    expected_artifacts: list[str] = field(default_factory=list)


@dataclass
class Continue:
    """attempt 边界。引擎据此落 attempt_started 事件并推进到下一轮。

    next_attempt: 即将开始的轮次号 (从 0 起)；= 已有 attempt_started 事件数。
    next_failure_type: 下一轮要处理的 failure_type。与上一轮不同即为「漂移」，
        触发跨分支计数重置 (state.py 派生时处理)。
    """

    next_attempt: int
    next_failure_type: str
    reason: Optional[str] = None


@dataclass
class Done:
    """成功或非异常收尾的终态。"""

    session_outcome: SessionOutcome
    reason: str


@dataclass
class Abort:
    """不可恢复的硬失败终态 (前置缺失 / 进程被杀 / 状态机卡死)。"""

    category: str
    reason: Optional[str] = None
    remediation: Optional[str] = None
    details: Optional[dict] = None


@dataclass
class Escalate:
    """需上抛人工介入的终态 (保留以对齐 lingxi 闭集；debug 默认走 Done/Abort)。"""

    reason: str
    details: Optional[dict] = None


# next_action(state) 的完整返回闭集。
Decision = "Action | Continue | Done | Abort | Escalate"
