"""transition.py — 写侧事件记录 helper (读侧派生在 state.py)。

镜像 lingxi runtime/engine/transition.py 的 record_* 形态，裁剪到 debug session 所需的
事件类型。所有写入经 EventWriter 落 events.jsonl，「先 started 后 completed」的配对由
action_id 维系，供 state.dangling_started 检测崩溃/resume 断点。

事件类型 (debug 域):
    session_started   — 首事件，记录 op/agent/arch/entry_failure_type (对应 lingxi run_started)
    attempt_started   — attempt 边界 (Continue 决策落地)，驱动双层预算计数
    action_started    — 一步开始 (spawn_agent / py_action)，带 action_id
    action_completed  — 一步完成，带同一 action_id + result
    session_done / session_aborted / session_escalated — 终态 (对应 lingxi workflow_*)
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional, Union

from engine.events import EventWriter
from engine.types import Abort, Action, Continue, Done, Escalate


def new_action_id() -> str:
    return f"act_{uuid.uuid4().hex[:8]}"


def record_session_started(
    task_dir: Path,
    *,
    op_name: str,
    agent: str,
    target_arch: Optional[str] = None,
    entry_failure_type: Optional[str] = None,
) -> None:
    EventWriter(task_dir).append({
        "type": "session_started",
        "op_name": op_name,
        "agent": agent,
        "target_arch": target_arch,
        "entry_failure_type": entry_failure_type,
    })


def record_attempt_started(task_dir: Path, cont: Continue) -> None:
    """落 attempt 边界事件。total_attempts = 此类事件计数；failure_type 驱动
    per_branch 计数与跨分支重置 (state.derive_counters)。"""
    EventWriter(task_dir).append({
        "type": "attempt_started",
        "attempt": cont.next_attempt,
        "failure_type": cont.next_failure_type,
        "reason": cont.reason,
    })


def record_rollback(task_dir: Path, *, from_attempt: int, best_metric: dict) -> None:
    """落 kernel 回滚事件 (问题 7): 连续无改善后用 current_best 覆盖工作区。
    供审计 + agent_backend 注入"已回滚、失败方向勿重复"到下一轮 prompt。"""
    EventWriter(task_dir).append({
        "type": "rollback",
        "from_attempt": from_attempt,
        "best_attempt": best_metric.get("attempt"),
        "best_case_pass_rate": best_metric.get("case_pass_rate"),
        "best_match_rate": best_metric.get("match_rate"),
    })


def _action_payload(action: Action) -> dict:
    return {
        "kind": action.kind,
        "name": action.name,
        "step": action.step,
        "skill_args": action.skill_args,
    }


def record_action_started(task_dir: Path, action: Action, action_id: str) -> None:
    payload = _action_payload(action)
    payload["expected_artifacts"] = action.expected_artifacts
    EventWriter(task_dir).append({
        "type": "action_started",
        "action_id": action_id,
        "action": payload,
    })


def record_action_completed(
    task_dir: Path, action: Action, action_id: str, result: dict
) -> None:
    EventWriter(task_dir).append({
        "type": "action_completed",
        "action_id": action_id,
        "action": _action_payload(action),
        "result": result,
    })


def record_decision(task_dir: Path, decision: Union[Done, Abort, Escalate]) -> None:
    """把终态决策落为终态事件。runner 在退出前调用一次。"""
    w = EventWriter(task_dir)
    if isinstance(decision, Done):
        w.append({
            "type": "session_done",
            "session_outcome": decision.session_outcome,
            "reason": decision.reason,
        })
    elif isinstance(decision, Abort):
        w.append({
            "type": "session_aborted",
            "category": decision.category,
            "reason": decision.reason,
            "remediation": decision.remediation,
            "details": decision.details,
        })
    elif isinstance(decision, Escalate):
        w.append({
            "type": "session_escalated",
            "reason": decision.reason,
            "details": decision.details,
        })
    else:
        raise TypeError(f"non-terminal decision: {type(decision).__name__}")
