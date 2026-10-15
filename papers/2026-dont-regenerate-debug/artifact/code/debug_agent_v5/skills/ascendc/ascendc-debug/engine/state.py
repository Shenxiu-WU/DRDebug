"""state.py — 从 events.jsonl 重放派生 DebugState (无独立 state 文件)。

镜像 lingxi runtime/engine/state.py 的 State.load = read_events + derive_counters 形态。
debug 域的派生核心是「双层预算 + 跨分支计数重置」(REWRITE_PLAN §2.3 / §2.3b):

  全局: total_attempts (= attempt_started 事件数)、entry/current_failure_type。
  分支: per_branch_attempt[ft] (每种 failure_type 各自轮次)，漂移时重置「更靠前」分支。

跨分支重置是 fold 内的有状态计算 (不是简单 groupby): 按行序逐个处理 attempt_started，
维护 prev_failure_type，每当 ft 变化就清零所有 PIPELINE_ORDER 更靠前的分支计数。
不变量: precision 在偏序最末 → 永不被 (order < new_ft) 命中 → 计数单调递增 →
振荡 (precision↔build…) 必然撞 precision cap 而停。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from engine.events import read_events
from engine.types import PIPELINE_ORDER


@dataclass
class Counters:
    # 全局层 (兑现全局 MAX_ATTEMPTS)。
    total_attempts: int = 0
    entry_failure_type: Optional[str] = None
    current_failure_type: Optional[str] = None
    # 分支层 (兑现分支硬上限 BRANCH_CAP)。仅含「跨分支重置后」的存活计数。
    per_branch_attempt: dict[str, int] = field(default_factory=dict)
    # 各分支 action 完成数 / 最近一次 loop_signal (供 next_action 派发与诊断)。
    last_loop_signal: Optional[str] = None
    last_stop_reason_code: Optional[str] = None


def _reset_upstream(per_branch: dict[str, int], new_ft: str) -> None:
    """清零流水线中比 new_ft 更靠前的所有分支计数 (就地修改)。

    new_ft 自身及更靠后的分支不动。new_ft 不在偏序表 (success/execution_aborted)
    时不重置任何分支 (这类不是可调试分支，理论上不会作为 attempt 的 failure_type)。
    """
    new_order = PIPELINE_ORDER.get(new_ft)
    if new_order is None:
        return
    for ft in list(per_branch):
        order = PIPELINE_ORDER.get(ft)
        if order is not None and order < new_order:
            per_branch[ft] = 0


def derive_counters(events: list[dict]) -> Counters:
    """纯 fold: events → Counters。无副作用，按行序处理。"""
    cf = Counters()
    prev_ft: Optional[str] = None

    for e in events:
        etype = e.get("type")

        if etype == "session_started":
            cf.entry_failure_type = e.get("entry_failure_type")
            if cf.current_failure_type is None:
                cf.current_failure_type = e.get("entry_failure_type")

        elif etype == "attempt_started":
            ft = e.get("failure_type")
            cf.total_attempts += 1
            if ft is not None:
                # 漂移检测: failure_type 相比上一轮变化 → 重置更靠前分支。
                if prev_ft is not None and ft != prev_ft:
                    _reset_upstream(cf.per_branch_attempt, ft)
                cf.per_branch_attempt[ft] = cf.per_branch_attempt.get(ft, 0) + 1
                cf.current_failure_type = ft
                prev_ft = ft

        elif etype == "action_completed":
            result = e.get("result") or {}
            sig = result.get("loop_signal")
            if sig is not None:
                cf.last_loop_signal = sig
            src = result.get("stop_reason_code")
            if src is not None:
                cf.last_stop_reason_code = src

    return cf


_TERMINAL_EVENT_TYPES = frozenset({
    "session_done",
    "session_aborted",
    "session_escalated",
})


@dataclass
class DebugState:
    """一次 debug session 的派生状态 (events.jsonl 的只读投影)。"""

    task_dir: Path
    cf: Counters
    events: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, task_dir: Path) -> "DebugState":
        events = read_events(task_dir)
        return cls(task_dir=task_dir, cf=derive_counters(events), events=events)

    # --- 预算查询 (next_action 的三道闸消费) -------------------------------

    @property
    def total_attempts(self) -> int:
        return self.cf.total_attempts

    @property
    def current_failure_type(self) -> Optional[str]:
        return self.cf.current_failure_type

    @property
    def entry_failure_type(self) -> Optional[str]:
        return self.cf.entry_failure_type

    def branch_attempt(self, failure_type: str) -> int:
        return self.cf.per_branch_attempt.get(failure_type, 0)

    # --- 信号查询 ----------------------------------------------------------

    def last_loop_signal(self) -> Optional[str]:
        return self.cf.last_loop_signal

    def last_stop_reason_code(self) -> Optional[str]:
        return self.cf.last_stop_reason_code

    def last_action_completed(self, name: Optional[str] = None) -> Optional[dict]:
        """最近一个 action_completed 事件 (可按 action.name 过滤)。"""
        for e in reversed(self.events):
            if e.get("type") != "action_completed":
                continue
            if name is not None and e.get("action", {}).get("name") != name:
                continue
            return e
        return None

    # --- 崩溃 / resume 支持 ------------------------------------------------

    def dangling_started(self) -> list[dict]:
        """有 action_started 无配对 action_completed 的事件 (按 action_id 配对)。

        resume / 崩溃后据此知道「上次卡在哪一步」。对应 lingxi state.dangling_started。

        去重: 同一 action_id 只返回最后一次 started (理论上 new_action_id 用 uuid 保证
        唯一，不会重复；此去重是防御性的，确保 resume 不会对同一动作重复处理)。
        """
        completed_ids = {
            e.get("action_id")
            for e in self.events
            if e.get("type") == "action_completed"
        }
        dangling: dict = {}  # action_id -> 最后一次 started 事件 (dict 保序，后者覆盖前者)
        for e in self.events:
            if e.get("type") != "action_started":
                continue
            aid = e.get("action_id")
            if aid not in completed_ids:
                dangling[aid] = e
        return list(dangling.values())

    def last_event(self) -> Optional[dict]:
        return self.events[-1] if self.events else None

    def is_terminal(self) -> bool:
        ev = self.last_event()
        return bool(ev) and ev.get("type") in _TERMINAL_EVENT_TYPES
