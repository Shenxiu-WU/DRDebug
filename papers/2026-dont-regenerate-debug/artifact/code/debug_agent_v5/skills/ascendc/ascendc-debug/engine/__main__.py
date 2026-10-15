"""__main__.py — 引擎 CLI 入口 (容器内运行)。

用法 (方案 C，runner 在容器内):
    python -m engine {task_dir} --op-name <op> --agent constructive|discovery \
        [--npu N] [--deadline-sec S] [--max-attempts N] [--model M] [--claude-bin B]

职责: 解析参数 → 造 diagnose agent_callback (engine.agent_backend) → 跑
run_debug_session 到终态 → 打印 session_outcome → 退出码按 outcome 映射
(success=0，其余非 0，供 utils/run_ascendc_debug_batch_cc.sh 读)。

entry_failure_type 不在此显式传入: runner 首拍 DebugState.load 从已有事件 / 上游
verify_status 决定 (见 runner resume 语义)；首次运行时由 --entry-failure-type 兜底。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from engine.agent_backend import (
    _DEFAULT_AGENT,
    _DEFAULT_ALLOWED_TOOLS,
    _DEFAULT_CLAUDE_BIN,
    make_agent_callback,
)
from engine.runner import run_debug_session

# session_outcome → 进程退出码 (脚本据此分类；success=0，其余非 0 但区分语义)。
_OUTCOME_EXIT_CODE = {
    "success": 0,
    "failed": 1,
    "stopped_by_gate": 2,
    "stopped_by_loop_limit": 3,
    "stopped_by_attempt_limit": 12,
    "stopped_by_branch_limit": 13,
    "timeout": 4,
    "skipped_env_issue": 5,
    "skipped_unsupported_type": 6,
    "crashed": 7,
    "provider_api_error": 8,
    "stopped_by_budget": 9,
    "degenerate_no_progress": 10,
    "ablation_violation": 11,
}

# PLACEHOLDER_MAIN


def _resolve_agent_name(agent: str) -> str:
    """短名 (constructive/discovery) → 完整 agent spec 名 (--agent 加载用)。
    已是完整名时原样返回。"""
    short = {
        "constructive": "ascendc-debug-agent-constructive",
        "discovery": "ascendc-debug-agent-discovery",
    }
    return short.get(agent, agent)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m engine",
                                 description="AscendC debug 引擎 (方案 C: 引擎主导主循环)")
    ap.add_argument("task_dir", help="算子任务目录 (含 kernel/ model_new_ascendc.py 等)")
    ap.add_argument("--op-name", required=True, help="算子名 (传给 gate / agent)")
    ap.add_argument("--agent", default="constructive",
                    help="constructive (默认/AAAI 主力) | discovery | 完整 agent spec 名")
    ap.add_argument("--npu", default=None, help="NPU 设备 ID (注入 agent prompt)")
    ap.add_argument("--deadline-sec", type=float, default=None,
                    help="wall-clock 总时长上限 (秒)；超时引擎主动 emit timeout 终态")
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="覆盖 ASCENDC_DEBUG_MAX_ATTEMPTS (引擎全局闸)")
    ap.add_argument("--model", default=None, help="claude --model (缺省用容器 env)")
    ap.add_argument("--claude-bin", default=_DEFAULT_CLAUDE_BIN, help="claude 可执行名/路径")
    ap.add_argument("--allowed-tools", default=_DEFAULT_ALLOWED_TOOLS,
                    help="claude --allowedTools")
    ap.add_argument("--workdir", default=None,
                    help="claude --add-dir 的根 (缺省=task_dir)；批处理应传仓库根，"
                         "使 agent 能访问 skills/ archive_tasks/ 等参考资料")
    ap.add_argument("--agent-timeout-sec", type=float, default=None,
                    help="单次 diagnose agent 调用超时 (秒)")
    # --max-turns: 单 attempt agentic turn 数硬闸，模型无关。
    # 默认 240；任务级 remaining budget 会在每次 spawn 前进一步动态截断它。
    ap.add_argument("--max-turns", default="240",
                    help="单 session agentic turn 数硬上限；默认 240")
    # 跨 attempt 累计 turns: 480 软预算；只有 engine-owned objective progress
    # 证据才扩展到 600 硬上限。单 session 仍受 --max-turns 约束。
    ap.add_argument("--soft-task-turns", type=int, default=480,
                    help="无客观改善证据时的任务级软预算；默认 480")
    ap.add_argument("--max-task-turns", type=int, default=600,
                    help="有客观改善证据后的任务级硬上限；默认 600")
    ap.add_argument("--entry-failure-type", default=None,
                    help="首次运行时注入入口 failure_type；例如 precision_failed")
    # KB 编排: precision_failed 每轮 forensics 后先做确定性 search，把命中摘要
    # 注入 diagnose prompt；success 且无作弊时再把 candidate_kb_entry.json 入库。
    # 默认 None 不启用 (向后兼容)。
    ap.add_argument("--kb-path", default=None,
                    help="精度知识库 JSON 路径；配置后每轮检索并在 success 终态自动入库，默认不启用")
    args = ap.parse_args(argv)

    if args.max_attempts is not None:
        os.environ["ASCENDC_DEBUG_MAX_ATTEMPTS"] = str(args.max_attempts)
    if args.max_task_turns is not None:
        os.environ["ASCENDC_DEBUG_MAX_TASK_TURNS"] = str(args.max_task_turns)
    if args.soft_task_turns is not None:
        os.environ["ASCENDC_DEBUG_SOFT_TASK_TURNS"] = str(args.soft_task_turns)

    agent_full = _resolve_agent_name(args.agent)
    workdir = Path(args.workdir) if args.workdir else Path(args.task_dir)
    agent_callback = make_agent_callback(
        agent_name=agent_full, claude_bin=args.claude_bin, model=args.model,
        allowed_tools=args.allowed_tools, workdir=workdir,
        npu=args.npu, timeout_sec=args.agent_timeout_sec,
        max_turns=args.max_turns)

    status = run_debug_session(
        Path(args.task_dir), op_name=args.op_name, agent=args.agent,
        entry_failure_type=args.entry_failure_type,
        agent_callback=agent_callback, deadline_sec=args.deadline_sec,
        kb_path=args.kb_path)

    outcome = status.get("session_outcome", "crashed")
    print(f"[ENGINE_RESULT] session_outcome={outcome} "
          f"attempts_used={status.get('attempts_used')} "
          f"branch={status.get('session_branch')}")
    return _OUTCOME_EXIT_CODE.get(outcome, 7)


if __name__ == "__main__":
    sys.exit(main())
