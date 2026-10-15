"""events.py — debug session 的单一事实源 (event-sourcing 地基)。

所有 session 状态都由 `{task_dir}/.debug_events/events.jsonl` 重放派生，不存独立
state 文件。本模块只负责「原子追加 + 顺序读取」，不解释事件语义 (语义在 state.py)。

设计契约:
    - 文件行顺序 = 事件的权威先后序。读取按行序返回，调用方不应依赖 seq 排序。
    - `seq = time.time_ns()` 仅作时间戳元数据；Windows 上 time_ns 分辨率可能较粗，
      同一拍写入的多条事件 seq 可能相等，故**不以 seq 作为排序键**。
    - debug session 是单进程串行推进，无并发写。这里的文件锁是 best-effort 防御
      (防 resume 时的并发读 / 外部工具同时写)，拿不到锁也不阻断写入。

与 lingxi runtime/engine/events.py 的唯一差异:
    lingxi 用 `fcntl.flock` (POSIX-only)。本项目需在 Windows 运行 (CANN 开发机有
    Windows 宿主)，故抽象为跨平台 `_advisory_lock`: POSIX 走 fcntl，Windows 走
    msvcrt.locking，两者都不可用时退化为无锁 O_APPEND (单进程串行下原子且安全)。
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

EVENTS_DIRNAME = ".debug_events"
EVENTS_FILENAME = "events.jsonl"

# 跨平台锁后端探测一次，缓存结果。
try:  # POSIX
    import fcntl  # type: ignore

    _LOCK_BACKEND = "fcntl"
except ImportError:  # Windows
    try:
        import msvcrt  # type: ignore

        _LOCK_BACKEND = "msvcrt"
    except ImportError:
        _LOCK_BACKEND = None


@contextmanager
def _advisory_lock(f: IO) -> Iterator[None]:
    """best-effort 独占文件锁。拿不到 / 平台不支持时静默退化为无锁。

    单进程串行的 debug session 下，O_APPEND 已保证小行的原子追加；此锁仅为防御
    外部并发，不能因锁失败而阻断核心写路径。
    """
    if _LOCK_BACKEND == "fcntl":
        # 与 msvcrt 路径一致的 best-effort 策略: 加锁失败 (如 NFS 不支持 flock、
        # 句柄状态异常) 退化为无锁，绝不阻断核心写路径。生产路径走这里 (openEuler)。
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except OSError:
            yield
            return
        try:
            yield
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    elif _LOCK_BACKEND == "msvcrt":
        # msvcrt.locking 锁的是「当前文件指针起 N 字节」。append 模式下指针在末尾，
        # 锁 1 字节即可达成互斥 (Windows 上是强制锁)。锁定区可能超出文件实际长度，
        # 这是允许的。任何异常 (区间冲突 / 句柄状态) 都退化为无锁，绝不阻断写入。
        try:
            f.flush()
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        except OSError:
            yield
            return
        try:
            yield
        finally:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
    else:
        yield


def events_path(task_dir: Path) -> Path:
    return Path(task_dir) / EVENTS_DIRNAME / EVENTS_FILENAME


class EventWriter:
    """向 `{task_dir}/.debug_events/events.jsonl` 原子追加事件。"""

    def __init__(self, task_dir: Path):
        self.path = events_path(task_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict) -> dict:
        """追加一条事件。自动补 `seq=time.time_ns()` (若调用方未提供)。

        返回实际写入的 (已补全的) 事件 dict，便于调用方拿到 seq。
        """
        stamped = {**event}
        stamped.setdefault("seq", time.time_ns())
        line = json.dumps(stamped, ensure_ascii=False, default=str) + "\n"
        # mode "a" → O_APPEND，每次 write 由 OS 保证定位到当前末尾。
        with self.path.open("a", encoding="utf-8") as f:
            with _advisory_lock(f):
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        return stamped


def read_events(task_dir: Path) -> list[dict]:
    """按文件行序读取全部事件 (= 权威先后序)。文件不存在时返回空 list。

    容错策略 (单进程串行 + O_APPEND + fsync 的现实):
      - 空行: 跳过。
      - **最后一行**坏 (无法 json 解析): 视为「上次进程在 write 中途崩溃留下的半行」，
        丢弃并继续 (这是 resume 的核心场景——半行不该让整个事实源不可读)。
      - **中间任一行**坏: fail-loud 抛 ValueError。中间坏行意味着真正的事实源损坏
        (非崩溃半行)，静默吞掉会让状态派生用错误数据续跑，对实验复现是灾难。
    """
    p = events_path(task_dir)
    if not p.exists():
        return []
    raw_lines = p.read_text(encoding="utf-8").splitlines()
    events: list[dict] = []
    last_idx = len(raw_lines) - 1
    for i, line in enumerate(raw_lines):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as e:
            if i == last_idx:
                # 末行半行 → 崩溃残留，丢弃。
                break
            raise ValueError(f"corrupt event at {p}:{i + 1}: {e}") from e
    return events
