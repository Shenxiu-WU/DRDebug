"""test_precision_knowledge_dump.py — 问题7 KB 入库(写侧)闭环增强单元测试。

本机无 pytest, 用 unittest:
  python -m unittest scripts.tests.test_precision_knowledge_dump
覆盖: derived_keywords 回填(写读对称)/ 可检索性自检告警 / 近重复 merge 提示 /
      回填后能被同类算子 op_name 召回(端到端闭环)。
"""
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import precision_knowledge as pk  # noqa: E402


def _candidate(title, *, op_types=None, patterns=None, feature="占位特征",
               reason="占位原因", fix="占位修复", type_="FIX_PRECISION_SYNC"):
    return {"title": title, "op_types": op_types or [], "patterns": patterns or [],
            "feature": feature, "reason": reason, "fix": fix, "type": type_}


def _dump(tmp, op_name, candidate, *, kb_init=None, action="new",
          merge_target_title=None):
    """在临时任务目录写 candidate, 调 dump, 返回 (entry, stderr_text, kb_list)。"""
    td = os.path.join(tmp, op_name)
    os.makedirs(os.path.join(td, "precision_tuning"), exist_ok=True)
    with open(os.path.join(td, "precision_tuning", "candidate_kb_entry.json"),
              "w", encoding="utf-8") as f:
        json.dump(candidate, f, ensure_ascii=False)
    kb_path = os.path.join(tmp, "kb.json")
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb_init or [], f)
    out, err = io.StringIO(), io.StringIO()
    o_out, o_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        entry = pk.dump_success_knowledge(kb_path, td, op_name, action=action,
                                          merge_target_title=merge_target_title)
    finally:
        sys.stdout, sys.stderr = o_out, o_err
    with open(kb_path, encoding="utf-8") as f:
        kb = json.load(f)
    return entry, err.getvalue(), kb


class TestDerivedKeywordBackfill(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def test_empty_op_types_backfilled_from_op_name(self):
        # agent 留空 op_types → 从 op_name 派生 derived_keywords
        e, _, _ = _dump(self.tmp, "006_QuantMatmul",
                        _candidate("Matmul Fixpipe 同步", op_types=[],
                                   patterns=["scattered"]))
        self.assertEqual(e["op_types"], [])          # 不篡改 agent 原意
        self.assertIn("matmul", e["derived_keywords"])
        self.assertIn("quant", e["derived_keywords"])

    def test_derived_excludes_existing_op_types(self):
        # 已在 op_types 的词不重复进 derived_keywords
        e, _, _ = _dump(self.tmp, "006_QuantMatmul",
                        _candidate("x", op_types=["matmul"], patterns=["all_wrong"]))
        self.assertNotIn("matmul", e.get("derived_keywords", []))

    def test_no_derived_field_when_nothing_to_add(self):
        # op_name 全是停用词/纯位号 → 不加空 derived_keywords 字段
        e, _, _ = _dump(self.tmp, "001_Op",
                        _candidate("x", patterns=["all_wrong"]))
        self.assertNotIn("derived_keywords", e)

    def test_read_only_mode_rejects_dump_without_mutation(self):
        candidate = _candidate(
            "readonly", op_types=["matmul"], patterns=["all_wrong"])
        with mock.patch.dict(
                os.environ, {"ASCENDC_DEBUG_KB_READ_ONLY": "1"}):
            entry, err, kb = _dump(
                self.tmp, "006_QuantMatmul", candidate, kb_init=[])
        self.assertIsNone(entry)
        self.assertIn("READ_ONLY", err)
        self.assertEqual(kb, [])

    def test_no_kb_mode_rejects_dump_without_mutation(self):
        candidate = _candidate(
            "ablated", op_types=["matmul"], patterns=["all_wrong"])
        with mock.patch.dict(os.environ, {"ABLATE_KB": "1"}):
            entry, err, kb = _dump(
                self.tmp, "006_QuantMatmul", candidate, kb_init=[])
        self.assertIsNone(entry)
        self.assertIn("ABLATE_KB", err)
        self.assertEqual(kb, [])


class TestRetrievabilitySelfCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def test_warns_when_no_anchor(self):
        # 无 op_types + 纯中文 title (无英文锚点) + op_name 派生不出词 → 告警
        _, err, _ = _dump(self.tmp, "Kernel",  # 纯停用词 op_name → 无派生
                          _candidate("纯中文标题没有英文", op_types=[],
                                     feature="纯中文特征", patterns=["all_wrong"]))
        self.assertIn("可检索性弱", err)

    def test_no_warn_when_title_has_english_anchor(self):
        # title 含英文专有词 → 有锚点 → 不告警
        _, err, _ = _dump(self.tmp, "Kernel",
                          _candidate("RFFT Padding 对齐问题", op_types=[],
                                     feature="x", patterns=["all_wrong"]))
        self.assertNotIn("可检索性弱", err)

    def test_no_warn_when_derived_keywords_present(self):
        # op_name 可派生词 → derived_keywords 非空 → 有锚点 → 不告警
        _, err, _ = _dump(self.tmp, "006_QuantMatmul",
                          _candidate("纯中文标题", op_types=[],
                                     feature="纯中文", patterns=["all_wrong"]))
        self.assertNotIn("可检索性弱", err)


class TestDuplicateMergeHint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def test_high_similarity_suggests_merge(self):
        existing = _candidate("Matmul Fixpipe CO1 Buffer 生命周期同步问题",
                              op_types=["matmul"], patterns=["scattered"],
                              feature="矩阵乘 CO1 buffer 提前复用导致分散错误")
        existing["_meta"] = {}
        near_dup = _candidate("Matmul Fixpipe CO1 Buffer 生命周期同步问题 (变体)",
                              op_types=["matmul"], patterns=["scattered"],
                              feature="矩阵乘 CO1 buffer 提前复用导致分散错误")
        _, err, kb = _dump(self.tmp, "006_QuantMatmul", near_dup,
                           kb_init=[existing])
        self.assertIn("高相似", err)
        self.assertIn("merge", err.lower())
        self.assertEqual(len(kb), 2)  # 仅提示, 不阻断写入

    def test_distinct_entry_no_merge_hint(self):
        existing = _candidate("归约轴切分错误", op_types=["reduction"],
                              patterns=["all_wrong"])
        existing["_meta"] = {}
        distinct = _candidate("RFFT 尾部 Padding 对齐", op_types=["fft"],
                              patterns=["tail_spike"],
                              feature="频域变换尾块 padding 未对齐")
        _, err, _ = _dump(self.tmp, "023_Hyena", distinct, kb_init=[existing])
        self.assertNotIn("高相似", err)


class TestWriteReadClosure(unittest.TestCase):
    """端到端: 留空 op_types 的经验入库后能被同类算子 op_name 召回。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def test_backfilled_entry_recalled_by_sibling_op(self):
        e, _, _ = _dump(self.tmp, "006_QuantMatmul",
                        _candidate("Matmul Fixpipe 同步", op_types=[],
                                   patterns=["all_wrong"],
                                   feature="矩阵乘结果分散"))
        kb_path = os.path.join(self.tmp, "kb.json")
        buf = io.StringIO()
        o = sys.stdout
        sys.stdout = buf
        try:
            res = pk.search_knowledge_base(
                kb_path, op_type="unknown", pattern="all_wrong",
                op_name="099_QuantMatmulVariant", top_k=3)
        finally:
            sys.stdout = o
        titles = [m["title"] for m in res["matched_entries"]]
        self.assertTrue(any("Matmul" in t for t in titles),
                        f"回填条目未被同类算子召回: {titles}")


if __name__ == "__main__":
    unittest.main()
