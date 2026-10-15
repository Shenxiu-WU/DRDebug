"""branch_timeout.py — 超时分支 Gate.

契约（findings.md §3.3 ⑦）:
  - Gate-F: verify_status.duration_sec ≥ 配置阈值；timeout_marker_present == true
  - Gate-A: audit 含 [SYNC_POINT_ANALYSIS] [ROOT_CAUSE] [FIX_PLAN]；
            fix_plan 指向同步/tiling/barrier/pipe
  - Gate-V: 新一轮在时限内完成（failure_type != timeout 且无 timeout_marker）
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import GateOutcome, MAX_ATTEMPTS


REQUIRED_SECTIONS = ("[SYNC_POINT_ANALYSIS]", "[ROOT_CAUSE]", "[FIX_PLAN]")


def _extract_section(content: str, section_name: str) -> str | None:
    """提取 [SECTION_NAME] 到下一个 [ 标记之间的文本。"""
    marker = f"[{section_name}]"
    start = content.find(marker)
    if start == -1:
        return None
    start += len(marker)
    end_marker = content.find("\n[", start)
    end_audit = content.find("=== END AUDIT ===", start)
    candidates = [pos for pos in [end_marker, end_audit] if pos != -1]
    end = min(candidates) if candidates else len(content)
    text = content[start:end].strip()
    return text if text else None


# 官方文档依据：
#   SyncAll Constraints §4 — blockDim > 实际核数导致框架插入异常同步，kernel 挂死
#   SyncAll Constraints §1 — 软同步 gmWorkspace 未初始化为 0 导致行为未定义
#   AscendCSync.md — CrossCoreSetFlag/WaitFlag 是 AIC↔AIV 跨核点对点同步，配对错误是死锁高频根因
TIMEOUT_FIX_KEYWORDS = {
    "sync", "SyncAll", "tiling", "barrier", "pipe",
    "CrossCoreSetFlag", "CrossCoreWaitFlag",
    "gmWorkspace", "blockDim",
}


class TimeoutBranch:

    def run_gate_f(self, task_dir, attempt: int) -> GateOutcome:
        task_dir = Path(task_dir)
        latest = task_dir / ".verify_status" / "latest.json"
        checks = {"latest_present": latest.exists()}
        if latest.exists():
            try:
                s = json.loads(latest.read_text())
            except (OSError, json.JSONDecodeError):
                s = {}
            checks["failure_type_is_timeout"] = s.get("failure_type") == "timeout"
            checks["timeout_marker_present"] = bool(s.get("timeout_marker_present"))
        ok = all(v for v in checks.values() if isinstance(v, bool))
        return GateOutcome("GATE-TIMEOUT-F", ok, checks)

    def run_gate_a(self, task_dir, attempt: int) -> GateOutcome:
        task_dir = Path(task_dir)
        audit = task_dir / "precision_tuning" / f"precision_audit_{attempt}.md"
        checks = {"audit_exists": audit.exists()}
        if audit.exists():
            try:
                content = audit.read_text()
            except OSError:
                content = ""
            for sec in REQUIRED_SECTIONS:
                checks[f"has_{sec.strip('[]').lower()}"] = sec in content
            # 只在 [FIX_PLAN] section 内搜索关键词，避免 [SYNC_POINT_ANALYSIS] 中
            # 必然出现的 SyncAll 导致检查形同虚设
            fix_plan_text = _extract_section(content, "FIX_PLAN")
            checks["fix_plan_mentions_sync_or_tiling"] = any(
                k in (fix_plan_text or "") for k in TIMEOUT_FIX_KEYWORDS
            )
        ok = all(v for v in checks.values() if isinstance(v, bool))
        return GateOutcome("GATE-TIMEOUT-A", ok, checks)

    def run_gate_v(self, task_dir, attempt: int) -> GateOutcome:
        task_dir = Path(task_dir)
        curr = task_dir / ".verify_status" / f"phase8_attempt{attempt}.json"
        checks = {"curr_present": curr.exists()}
        loop_signal = "CONTINUE"
        if curr.exists():
            try:
                c = json.loads(curr.read_text())
            except (OSError, json.JSONDecodeError):
                c = {}
            no_timeout = (
                c.get("failure_type") != "timeout"
                and not c.get("timeout_marker_present")
            )
            checks["no_longer_timeout"] = no_timeout
            if c.get("failure_type") == "success":
                loop_signal = "PASS"
            elif no_timeout:
                loop_signal = "CONTINUE"
            else:
                loop_signal = "STOP" if attempt >= MAX_ATTEMPTS - 1 else "CONTINUE"
        return GateOutcome(
            "GATE-TIMEOUT-V",
            loop_signal != "STOP",
            checks,
            loop_signal=loop_signal,
            reason="timeout presence progression",
        )
