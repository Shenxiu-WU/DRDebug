"""AscendC debug session 的 event-sourcing 引擎。

把 debug session 的「编排 / 状态 / 落盘」从 SKILL.md 散文 + LLM 自驱，
提升到「单一事实源 events.jsonl + 纯函数 next_action + 通用 runner」的工程形态。
对齐 lingxi-ascendc-cli 的 runtime/engine，但范围限定为「驱动单个 debug session
到终态」，不含多 workflow orchestrator / batch / remote backend。

模块:
    events.py        — 单一事实源 events.jsonl 的原子读写
    types.py         — 闭集 dataclass (Action/Continue/Done/Abort/Escalate)
    state.py         — 从 events 重放派生 DebugState (Step 2)
    gate_adapter.py  — 包装现有 gate 输出为纯判定 (Step 3)
    next_action.py   — 核心纯函数决策器 (Step 4)
    runner.py        — 通用主循环 (Step 5)
    exit_artifacts.py— 退出产物从事件重建 (Step 5)
"""
