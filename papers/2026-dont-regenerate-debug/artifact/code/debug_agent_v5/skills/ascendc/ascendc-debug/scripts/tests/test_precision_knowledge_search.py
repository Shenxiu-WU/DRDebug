"""test_precision_knowledge_search.py — 问题7 KB query 增强单元测试。

本机无 pytest, 用 unittest:
  python -m unittest scripts.tests.test_precision_knowledge_search
覆盖: op_name 分词 / IDF 加权 / op_type 模糊回退 / all_wrong 降权 /
      op_name 关键词召回 / match_reason / task_dir 回退自取 / 双金标准召回。
"""
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import precision_knowledge as pk  # noqa: E402


def _silent_search(kb_path, **kw):
    """吞掉脚本 stdout 噪声, 只取返回 dict。"""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        return pk.search_knowledge_base(kb_path, **kw)
    finally:
        sys.stdout = old


def _write_kb(entries):
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)
    return path


def _entry(title, *, op_types=None, patterns=None, type_="FIX_PRECISION_LOGIC",
           feature="占位特征", reason="占位原因", fix="占位修复"):
    return {"title": title, "op_types": op_types or [], "patterns": patterns or [],
            "type": type_, "feature": feature, "reason": reason, "fix": fix}


class TestOpNameKeywords(unittest.TestCase):
    def test_camel_and_position_prefix_split(self):
        kw = pk._op_name_keywords("023_HyenaFftSizePaddingRfft")
        # 驼峰拆出专有词
        self.assertIn("rfft", kw)
        self.assertIn("fft", kw)
        self.assertIn("padding", kw)
        self.assertIn("hyena", kw)
        # 纯数字位号前缀剔除
        self.assertNotIn("023", kw)

    def test_stopwords_dropped(self):
        kw = pk._op_name_keywords("MyKernelWithNpuOp")
        for sw in ("kernel", "npu", "op", "with"):
            self.assertNotIn(sw, kw)

    def test_none_returns_empty(self):
        self.assertEqual(pk._op_name_keywords(None), set())


class TestKnowledgeAblation(unittest.TestCase):
    def test_search_does_not_read_entries_when_kb_is_ablated(self):
        path = _write_kb([_entry("sentinel", op_types=["matmul"])])
        self.addCleanup(lambda: os.unlink(path))
        with mock.patch.dict(os.environ, {"ABLATE_KB": "1"}):
            result = _silent_search(path, op_type="matmul")
        self.assertEqual(result["matched_entries"], [])
        self.assertEqual(result["total_kb_size"], 0)
        self.assertTrue(result["disabled_by_ablation"])


class TestIdfWeighting(unittest.TestCase):
    def test_rare_word_weighted_higher(self):
        # rare 只在 1 条出现, common 在 3 条 → idf(rare) > idf(common)
        kb = [_entry("rare rfft entry", op_types=["fft"]),
              _entry("common padding a", op_types=["x"]),
              _entry("common padding b", op_types=["y"]),
              _entry("common padding c", op_types=["z"])]
        idf = pk._keyword_idf(kb)
        self.assertGreater(idf["rfft"], idf["padding"])


class TestScoreEntry(unittest.TestCase):
    def test_op_type_fuzzy_fallback(self):
        # op_type=matmul 精确不在 op_types(matmul_v2), 但作子串命中 → 模糊回退
        e = _entry("some op", op_types=["matmul_v2"])
        score, reasons = pk._score_entry(e, None, "matmul", None)
        self.assertGreater(score, 0)
        self.assertTrue(any("op_type~" in r for r in reasons))

    def test_op_type_fuzzy_via_title(self):
        # op_type 作子串命中 title 也算模糊回退
        e = _entry("Conv3d 反向精度", op_types=["other"])
        score, reasons = pk._score_entry(e, None, "conv3d", None)
        self.assertGreater(score, 0)
        self.assertTrue(any("op_type~" in r for r in reasons))

    def test_unknown_op_type_no_match(self):
        e = _entry("anything", op_types=["matmul"])
        score, _ = pk._score_entry(e, None, "unknown", None)
        self.assertEqual(score, 0)

    def test_all_wrong_is_weak(self):
        # all_wrong 命中只给弱权 (W_PATTERN_WEAK), 远小于普通 pattern
        e = _entry("x", patterns=["all_wrong"])
        s_weak, reasons = pk._score_entry(e, "all_wrong", None, None)
        e2 = _entry("y", patterns=["tail_spike"])
        s_strong, _ = pk._score_entry(e2, "tail_spike", None, None)
        self.assertLess(s_weak, s_strong)
        self.assertTrue(any("弱" in r for r in reasons))

    def test_op_name_keyword_channel_with_idf(self):
        e = _entry("RFFT padding 对齐", op_types=["fft"])
        idf = {"rfft": 4.0, "fft": 3.0, "padding": 2.0}
        kw = {"rfft", "fft", "padding"}
        score, reasons = pk._score_entry(e, None, None, None, kw, idf)
        self.assertGreater(score, 0)
        self.assertTrue(any(r.startswith("op_name_kw:") for r in reasons))
        # reason 中命中词按 IDF 降序 (rfft 在 fft 前)
        kw_reason = [r for r in reasons if r.startswith("op_name_kw:")][0]
        self.assertLess(kw_reason.index("rfft"), kw_reason.index("padding"))

    def test_checklist_skips_op_name_channel(self):
        e = _entry("[CHECKLIST] something rfft")
        score, _ = pk._score_entry(e, None, None, None, {"rfft"}, {"rfft": 4.0})
        self.assertEqual(score, 0)


class TestSearchRecall(unittest.TestCase):
    """双金标准: op_name 关键词使专项条目召回 (回应'不只调单案例')。"""

    def setUp(self):
        # 仿真实库的稀疏结构: 1 条 RFFT 专项 + 1 条 matmul 专项 + 噪声泛条目
        self.kb = _write_kb([
            _entry("RFFT 动态长度尾部 Padding 与 Vector 对齐",
                   op_types=["fft", "frequency_domain", "vector"],
                   patterns=["tail_spike"]),
            _entry("Matmul Fixpipe CO1 Buffer 生命周期同步",
                   op_types=["matmul", "fixpipe"], patterns=["all_wrong"]),
            _entry("归约轴切分破坏 (Reduction noise)",
                   op_types=["reduction"], patterns=["all_wrong"]),
            _entry("Host InferShape 偏移不一致 (noise)",
                   op_types=["other"], patterns=["all_wrong"]),
        ])
        self.addCleanup(lambda: os.unlink(self.kb))

    def test_hyena_unknown_recalls_rfft(self):
        r = _silent_search(self.kb, op_type="unknown", pattern="all_wrong",
                           op_name="023_HyenaFftSizePaddingRfft", top_k=3)
        titles = [e["title"] for e in r["matched_entries"][:3]]
        self.assertTrue(any("RFFT" in t for t in titles),
                        f"RFFT 未进 top-3: {titles}")

    def test_matmul_recalls_fixpipe(self):
        r = _silent_search(self.kb, op_type="matmul", pattern="all_wrong",
                           op_name="006_QuantMatmul", top_k=3)
        titles = [e["title"] for e in r["matched_entries"][:3]]
        self.assertTrue(any("Fixpipe" in t for t in titles),
                        f"Matmul 专项未进 top-3: {titles}")

    def test_match_reason_in_result(self):
        r = _silent_search(self.kb, op_type="unknown", pattern="all_wrong",
                           op_name="023_HyenaFftSizePaddingRfft", top_k=3)
        self.assertIn("match_reason", r["matched_entries"][0])
        self.assertIn("op_name_keywords", r["query"])

    def test_search_returns_stable_knowledge_id(self):
        first = _silent_search(self.kb, op_type="matmul", pattern="all_wrong",
                               op_name="006_QuantMatmul", top_k=3)
        second = _silent_search(self.kb, op_type="matmul", pattern="all_wrong",
                                op_name="006_QuantMatmul", top_k=3)
        first_ids = [e["knowledge_id"] for e in first["matched_entries"]]
        second_ids = [e["knowledge_id"] for e in second["matched_entries"]]
        self.assertEqual(first_ids, second_ids)
        self.assertTrue(all(i.startswith("kb-") and len(i) == 15 for i in first_ids))


class TestKnowledgeId(unittest.TestCase):
    def test_explicit_id_preserved(self):
        entry = _entry("anything")
        entry["knowledge_id"] = "kb-0123456789ab"
        self.assertEqual(pk._knowledge_id(entry), "kb-0123456789ab")

    def test_legacy_id_is_title_deterministic(self):
        a = _entry("  Float16   Reduction ")
        b = _entry("float16 reduction")
        self.assertEqual(pk._knowledge_id(a), pk._knowledge_id(b))


class TestTaskDirFallback(unittest.TestCase):
    def test_op_name_from_events(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        ev_dir = os.path.join(tmp, ".debug_events")
        os.makedirs(ev_dir)
        with open(os.path.join(ev_dir, "events.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "session_started",
                                "op_name": "042_FooBar"}) + "\n")
        self.assertEqual(pk._op_name_from_task_dir(tmp), "042_FooBar")

    def test_op_name_from_dirname_when_no_events(self):
        tmp = tempfile.mkdtemp(prefix="099_Baz_")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        got = pk._op_name_from_task_dir(tmp)
        self.assertTrue(got and got.startswith("099_Baz_"))

    def test_none_for_missing_dir(self):
        self.assertIsNone(pk._op_name_from_task_dir("/no/such/dir/xyz"))


if __name__ == "__main__":
    unittest.main()
