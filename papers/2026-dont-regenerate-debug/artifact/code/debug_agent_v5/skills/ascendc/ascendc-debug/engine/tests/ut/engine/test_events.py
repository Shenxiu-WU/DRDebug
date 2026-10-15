"""test_events.py — events.jsonl 读写 round-trip 与边界。

验收点 (REWRITE_PLAN §6.1): 事件读写 round-trip。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine.events import EventWriter, events_path, read_events


class TestEventsRoundTrip(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_empty_when_no_file(self) -> None:
        # 文件不存在 → 空 list，不抛。
        self.assertEqual(read_events(self.task_dir), [])

    def test_path_layout(self) -> None:
        self.assertEqual(
            events_path(self.task_dir),
            self.task_dir / ".debug_events" / "events.jsonl",
        )

    def test_append_then_read_preserves_order(self) -> None:
        w = EventWriter(self.task_dir)
        w.append({"type": "attempt_started", "attempt": 0})
        w.append({"type": "action_started", "name": "forensics"})
        w.append({"type": "action_completed", "name": "forensics"})

        events = read_events(self.task_dir)
        self.assertEqual([e["type"] for e in events],
                         ["attempt_started", "action_started", "action_completed"])
        self.assertEqual(events[0]["attempt"], 0)

    def test_seq_auto_stamped_and_not_used_for_order(self) -> None:
        # append 自动补 seq；行序才是权威序，即便 seq 相等。
        w = EventWriter(self.task_dir)
        e1 = w.append({"type": "a"})
        e2 = w.append({"type": "b"})
        self.assertIn("seq", e1)
        self.assertIn("seq", e2)
        events = read_events(self.task_dir)
        self.assertEqual([e["type"] for e in events], ["a", "b"])

    def test_caller_seq_preserved(self) -> None:
        # 调用方显式给 seq 时不被覆盖。
        w = EventWriter(self.task_dir)
        w.append({"type": "a", "seq": 42})
        self.assertEqual(read_events(self.task_dir)[0]["seq"], 42)

    def test_unicode_payload_round_trip(self) -> None:
        # 中文 reason 等必须无损 (ensure_ascii=False)。
        w = EventWriter(self.task_dir)
        w.append({"type": "decision", "reason": "精度未收敛，进入第二轮"})
        ev = read_events(self.task_dir)[0]
        self.assertEqual(ev["reason"], "精度未收敛，进入第二轮")

    def test_blank_lines_skipped(self) -> None:
        w = EventWriter(self.task_dir)
        w.append({"type": "a"})
        # 手动插入空行模拟意外写入。
        with events_path(self.task_dir).open("a", encoding="utf-8") as f:
            f.write("\n   \n")
        w.append({"type": "b"})
        self.assertEqual([e["type"] for e in read_events(self.task_dir)], ["a", "b"])

    def test_corrupt_middle_line_raises(self) -> None:
        # 中间坏行 = 真正的事实源损坏，fail-loud。
        w = EventWriter(self.task_dir)
        w.append({"type": "a"})
        with events_path(self.task_dir).open("a", encoding="utf-8") as f:
            f.write("{not valid json}\n")
        w.append({"type": "b"})  # 坏行后还有正常行 → 坏行在中间
        with self.assertRaises(ValueError):
            read_events(self.task_dir)

    def test_corrupt_last_line_tolerated(self) -> None:
        # 末行半行 = 崩溃残留 (write 中途断电)，丢弃并返回前面完好的事件。
        w = EventWriter(self.task_dir)
        w.append({"type": "a"})
        w.append({"type": "b"})
        with events_path(self.task_dir).open("a", encoding="utf-8") as f:
            f.write('{"type": "c", "partia')  # 半行，无换行，模拟崩溃
        events = read_events(self.task_dir)
        self.assertEqual([e["type"] for e in events], ["a", "b"])

    def test_last_line_half_written_with_newline(self) -> None:
        # 末行坏但带换行 (极少见) 也应被容错丢弃，不影响前面。
        w = EventWriter(self.task_dir)
        w.append({"type": "a"})
        with events_path(self.task_dir).open("a", encoding="utf-8") as f:
            f.write('{"broken\n')
        self.assertEqual([e["type"] for e in read_events(self.task_dir)], ["a"])

    def test_non_serializable_falls_back_to_str(self) -> None:
        # default=str 兜底非 JSON 原生类型 (如 Path)，不崩。
        w = EventWriter(self.task_dir)
        w.append({"type": "a", "path": Path("/tmp/x")})
        ev = read_events(self.task_dir)[0]
        self.assertIsInstance(ev["path"], str)

    def test_jsonl_one_object_per_line(self) -> None:
        w = EventWriter(self.task_dir)
        w.append({"type": "a"})
        w.append({"type": "b"})
        raw = events_path(self.task_dir).read_text(encoding="utf-8")
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        self.assertEqual(len(lines), 2)
        for ln in lines:
            json.loads(ln)  # 每行独立可解析


if __name__ == "__main__":
    unittest.main()
