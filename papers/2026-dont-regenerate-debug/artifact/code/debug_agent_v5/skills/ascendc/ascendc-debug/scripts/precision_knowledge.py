#!/usr/bin/env python3
from __future__ import annotations

"""
precision_knowledge.py — 精度问题知识库管理

职责:
  1. load   — 全量加载知识库 (fallback 用)
  2. search — 结构化 RAG 检索: 按 op_types + patterns + position 评分排序, 返回 top-K + CHECKLIST
  3. check  — dump 前检查候选条目与知识库的相似度，辅助 Agent 决策 new/merge/abandon
  4. dump   — 精度通过后，将 Agent 生成的候选条目写入知识库（支持 new / merge / abandon 三种操作）

知识库格式: 七字段 JSON，RAG-ready。
每条记录:
  {
    "title":    "标准化中文标题 (含英文关键词)",
    "patterns": ["tail_spike", "boundary_concentration"],   # 误差模式数组，可为空 []
    "op_types": ["reduction", "pooling"],                   # 算子类型数组，可为空 []
    "feature":  "自然语言描述错误特征 (泛化, 中文)，不嵌入 pattern=/op_type= 标签",
    "reason":   "深层原因 (中文)",
    "fix":      "通用修复指南 (代码级别, 中文)",
    "type":     "FIX_PRECISION_xxx"
  }

patterns 合法值（VALID_PATTERNS）:
  tail_spike, uniform_offset, scattered, magnitude_correlated,
  nan_inf_contamination, dimension_concentration, boundary_concentration, all_wrong

op_types: 自由字符串数组，如 "reduction"、"pooling"、"matmul"、"convolution" 等，无枚举限制。

type 枚举 (精度专项):
  FIX_PRECISION_PADDING     — Padding 值导致精度问题
  FIX_PRECISION_TAIL        — 尾块处理精度问题
  FIX_PRECISION_REDUCTION   — 归约操作精度损失
  FIX_PRECISION_TYPECAST    — 类型转换精度问题
  FIX_PRECISION_LAYOUT      — 数据布局导致精度错误
  FIX_PRECISION_SYNC        — 同步问题导致精度随机错误
  FIX_PRECISION_OVERFLOW    — 数值溢出 (NaN/Inf)
  FIX_PRECISION_LOGIC       — 算法逻辑导致精度偏差
  FIX_PRECISION_OTHER       — 其他精度问题

用法:
    # 全量加载知识库 (fallback, stdout 输出 JSON)
    python3 precision_knowledge.py load --kb-path <path>

    # 结构化 RAG 检索 (推荐, stdout 输出 JSON)
    python3 precision_knowledge.py search --kb-path <path> --op-type <type> --pattern <hint> [--position <pos>] [--top-k 3]

    # dump 前检查候选条目与知识库的相似度 (stdout 输出 JSON)
    python3 precision_knowledge.py check --kb-path <path> --candidate-path <path> [--top-k 3] [--threshold 0.10]

    # 写入知识库 (精度通过后)
    python3 precision_knowledge.py dump --kb-path <path> --task-name <name> --op-name <name> [--action new|merge|abandon] [--merge-target-title "<title>"]
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent.parent.parent.parent

_KNOWLEDGE_ID_RE = re.compile(r"^kb-[0-9a-f]{12}$")


def _knowledge_id(entry: dict) -> str:
    """Return a stable ID, preserving explicit IDs and deriving legacy ones.

    Existing knowledge bases do not need a bulk rewrite: legacy entries receive a
    deterministic title-based ID at read time.  New/merged entries persist the ID.
    """
    explicit = entry.get("knowledge_id") or entry.get("id")
    if isinstance(explicit, str) and _KNOWLEDGE_ID_RE.fullmatch(explicit.lower()):
        return explicit.lower()
    normalized = " ".join(str(entry.get("title") or "").strip().lower().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"kb-{digest}"


VALID_TYPES = [
    "FIX_PRECISION_PADDING",
    "FIX_PRECISION_TAIL",
    "FIX_PRECISION_REDUCTION",
    "FIX_PRECISION_TYPECAST",
    "FIX_PRECISION_LAYOUT",
    "FIX_PRECISION_SYNC",
    "FIX_PRECISION_OVERFLOW",
    "FIX_PRECISION_LOGIC",
    "FIX_PRECISION_OTHER",
]

VALID_PATTERNS = [
    "tail_spike",
    "uniform_offset",
    "scattered",
    "magnitude_correlated",
    "nan_inf_contamination",
    "dimension_concentration",
    "boundary_concentration",
    "all_wrong",
]

# 文本必填字段（非空字符串）
REQUIRED_FIELDS = ["title", "feature", "reason", "fix", "type"]


def _kb_disabled() -> bool:
    """Hard guard for the no_kb arm, including direct Agent CLI calls."""
    return os.environ.get("ABLATE_KB") == "1"


def _kb_read_only() -> bool:
    """Whether reads are allowed but mutation is forbidden."""
    return os.environ.get("ASCENDC_DEBUG_KB_READ_ONLY") == "1"


def _is_valid_entry(entry: dict) -> bool:
    """校验知识库条目格式：文本字段非空 + patterns/op_types 为 list。"""
    for k in REQUIRED_FIELDS:
        if not entry.get(k):
            return False
    if not isinstance(entry.get("patterns"), list):
        return False
    if not isinstance(entry.get("op_types"), list):
        return False
    return True


# ============================================================
# Tokenize / Similarity helpers (用于 check 子命令)
# ============================================================

def _tokenize(text: str) -> set:
    """
    提取混合中英文文本中的 token 集合。
    - 英文: 按非字母边界切分，snake_case / camelCase 进一步拆分，min 长度 2
    - 中文: 每个汉字单独作为 token
    """
    tokens = set()
    for ch in re.findall(r'[一-鿿]', text):
        tokens.add(ch)
    for word in re.findall(r'[a-zA-Z][a-zA-Z0-9_]*', text):
        w = word.lower()
        if len(w) >= 2:
            tokens.add(w)
        for part in w.split('_'):
            if len(part) >= 2:
                tokens.add(part)
        for part in re.sub(r'([A-Z][a-z]+)', r' \1', word).split():
            if len(part) >= 2:
                tokens.add(part.lower())
    return tokens


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard 相似度: |A∩B| / |A∪B|"""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ============================================================
# Load
# ============================================================

def load_knowledge_base(kb_path: str) -> list:
    """加载知识库, 返回条目列表"""
    if _kb_disabled():
        print("[KB] disabled by ABLATE_KB=1", file=sys.stderr)
        print("[]")
        return []
    if not os.path.exists(kb_path):
        print(f"[KB] ⚠️ 知识库文件不存在: {kb_path}, 使用空知识库", file=sys.stderr)
        return []

    with open(kb_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"[KB] ⚠️ 知识库格式错误 (期望 list), 使用空知识库", file=sys.stderr)
        return []

    valid = [e for e in data if _is_valid_entry(e)]

    print(f"[KB] ✅ 已加载 {len(valid)} 条精度知识")
    print(json.dumps(valid, indent=2, ensure_ascii=False))
    return valid


# ============================================================
# Search (结构化 RAG 检索)
# ============================================================

# pattern → 最可能关联的 type 映射 (用于 type 字段交叉加权)
PATTERN_TYPE_AFFINITY = {
    "tail_spike": ["FIX_PRECISION_TAIL", "FIX_PRECISION_PADDING", "FIX_PRECISION_REDUCTION"],
    "uniform_offset": ["FIX_PRECISION_PADDING", "FIX_PRECISION_LOGIC", "FIX_PRECISION_REDUCTION"],
    "scattered": ["FIX_PRECISION_SYNC", "FIX_PRECISION_REDUCTION"],
    "magnitude_correlated": ["FIX_PRECISION_TYPECAST", "FIX_PRECISION_REDUCTION"],
    "nan_inf_contamination": ["FIX_PRECISION_OVERFLOW"],
    "dimension_concentration": ["FIX_PRECISION_LAYOUT", "FIX_PRECISION_LOGIC"],
    "boundary_concentration": ["FIX_PRECISION_PADDING", "FIX_PRECISION_TAIL"],
    "all_wrong": ["FIX_PRECISION_LOGIC", "FIX_PRECISION_LAYOUT", "FIX_PRECISION_REDUCTION", "FIX_PRECISION_SYNC"],
}

# position 特征 → 关联的 pattern (用于第二次检索的辅助加分)
POSITION_PATTERN_AFFINITY = {
    "tail": ["tail_spike", "boundary_concentration"],
    "boundary": ["boundary_concentration", "tail_spike"],
    "head": ["uniform_offset", "all_wrong"],
    "scattered": ["scattered", "magnitude_correlated"],
}

# 评分权重
W_PATTERN = 3
W_PATTERN_WEAK = 1        # all_wrong 等泛化 pattern 降权 (无区分度, 占库近半)
W_OP_TYPE = 2
W_TYPE = 1
W_POSITION = 1
W_OP_TYPE_ALL_WRONG_BOOST = 2  # all_wrong 是泛化 hint, op_type 精确匹配时额外加权
# 问题7 增强权重:
W_OP_TYPE_FUZZY = 1       # op_type 精确失败时的模糊回退 (子串/标题命中)，弱于精确
# op_name 关键词通道用 IDF 式稀有度加权: 命中分 = W_OP_NAME_KEYWORD * idf(词)。
# 专有稀有词 (rfft/hyena df 小→idf 大) 远重于泛词 (padding/size df 大→idf 小)，
# 避免泛词命中压过算子专项条目 (真实数据: RFFT 专项条目 #45 曾被 padding 泛匹配挤出 top-3)。
W_OP_NAME_KEYWORD = 2.0
W_OP_NAME_KEYWORD_CAP = 15.0  # op_name 关键词通道单条封顶: 放宽以让多专有词 (rfft+fft+...)
                             # 命中的算子专项条目压过单泛词命中的条目, 同时防超长名无限刷分

# op_name 分词通道的停用词: 纯位号/通用词，命中无区分度，剔除避免噪声匹配。
_OP_NAME_STOPWORDS = {
    "op", "kernel", "ascendc", "npu", "fwd", "bwd", "forward", "backward",
    "v2", "v3", "with", "and", "the", "for", "from",
}


def _is_checklist(entry: dict) -> bool:
    """判断条目是否为 CHECKLIST 类型"""
    return entry.get("title", "").startswith("[CHECKLIST]")


def _op_name_keywords(op_name: str | None) -> set:
    """把 op_name 拆成有区分度的关键词集合 (剔除位号前缀 + 停用词)。

    例: "023_HyenaFftSizePaddingRfft" → {hyena, fft, size, padding, rfft}。
    复用 _tokenize 的 snake/camel 拆分; 去掉纯数字 token 与 _OP_NAME_STOPWORDS。
    """
    if not op_name:
        return set()
    toks = _tokenize(op_name)
    return {t for t in toks
            if not t.isdigit() and t not in _OP_NAME_STOPWORDS}


def _entry_keyword_pool(entry: dict) -> set:
    """条目可供 op_name 关键词匹配的 token 池 (title + feature + op_types + derived_keywords)。

    derived_keywords 为入库时从 op_name 自动派生的检索辅助词 (写读对称, 见 dump
    侧回填): agent 留空 op_types 的通用经验也借此可被同类算子的 op_name 召回。
    """
    text = (entry.get("title", "") + " " + entry.get("feature", "")
            + " " + " ".join(entry.get("op_types", []))
            + " " + " ".join(entry.get("derived_keywords", [])))
    return _tokenize(text)


def _keyword_idf(normal_entries: list) -> dict:
    """对普通条目的 keyword_pool 计算每词 IDF: log((N+1)/(df+1)) + 1，词越稀有越大。

    用于 op_name 关键词通道加权——专有词 (rfft df=1) 远重于泛词 (padding df=5)。
    """
    import math
    pools = [_entry_keyword_pool(e) for e in normal_entries]
    n = len(pools)
    df: dict = {}
    for pool in pools:
        for tok in pool:
            df[tok] = df.get(tok, 0) + 1
    return {tok: math.log((n + 1) / (d + 1)) + 1.0 for tok, d in df.items()}


# 近重复阈值: new 入库时与现有条目 Jaccard 超此值则提示应 merge。与 check 子命令
# 默认 threshold(0.10) 拉开——0.10 是"值得 agent 复核"的弱相似, 0.55 才是"疑似重复"。
_DUP_MERGE_THRESHOLD = 0.55


def _entry_sim_text(entry: dict) -> str:
    """条目相似度比对文本: title + feature + patterns + op_types (对齐 check_similarity)。"""
    return (entry.get("title", "") + " " + entry.get("feature", "")
            + " " + " ".join(entry.get("patterns", []))
            + " " + " ".join(entry.get("op_types", [])))


def _most_similar_entry(entry: dict, kb: list):
    """返回 kb 中与 entry 最相似的 (jaccard, title); kb 空返回 None。"""
    cand_tokens = _tokenize(_entry_sim_text(entry))
    best = None
    for e in kb:
        score = _jaccard(cand_tokens, _tokenize(_entry_sim_text(e)))
        if best is None or score > best[0]:
            best = (score, e.get("title", ""))
    return best


def _op_name_from_task_dir(task_dir: str | None) -> str | None:
    """--op-name 缺省时的回退: 优先读 events.jsonl 的 session_started.op_name,
    否则用 task 目录名。任何异常静默返回 None (best-effort)。"""
    if not task_dir or not os.path.isdir(task_dir):
        return None
    events = os.path.join(task_dir, ".debug_events", "events.jsonl")
    try:
        with open(events, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                if e.get("type") == "session_started" and e.get("op_name"):
                    return e["op_name"]
    except (OSError, ValueError):
        pass
    return os.path.basename(os.path.normpath(task_dir)) or None


def _score_entry(entry: dict, query_pattern: str | None, query_op_type: str | None,
                 query_position: str | None,
                 op_name_kw: set | None = None,
                 idf: dict | None = None) -> tuple[float, list]:
    """对单条知识库条目评分。返回 (score, reasons): reasons 记录命中的维度 (why-matched)。

    直接从 patterns / op_types 数组读取，无需解析 feature 文本。op_name_kw 为 op_name
    分词集合 (问题7 增强): 当 op_type/pattern 失效 (unknown/all_wrong) 时提供关键词召回。
    idf 为各词的稀有度权重 (KB 内文档频率倒数), 专有词命中远重于泛词。
    """
    score = 0.0
    reasons: list = []
    entry_patterns = entry.get("patterns", [])
    entry_op_types = entry.get("op_types", [])
    entry_type = entry.get("type", "")
    is_checklist = _is_checklist(entry)

    # 1. pattern 匹配。all_wrong 是泛化 hint (占库近半, 无区分度), 降权为弱信号,
    #    避免它给所有条目同一虚高基底、压过真正相关的 op_name 专有词命中 (问题7)。
    if query_pattern and query_pattern in entry_patterns:
        if query_pattern == "all_wrong":
            score += W_PATTERN_WEAK
            reasons.append("pattern=all_wrong(弱)")
        else:
            score += W_PATTERN
            reasons.append(f"pattern={query_pattern}")

    # 2. op_type 匹配 (权重 2 精确 / 1 模糊) — 仅对非 CHECKLIST 条目
    if query_op_type and query_op_type != "unknown" and not is_checklist:
        if query_op_type in entry_op_types:
            score += W_OP_TYPE
            reasons.append(f"op_type={query_op_type}")
        else:
            # 模糊回退: op_type 作子串命中条目 op_types 或 title (补救 19 类稀疏 + 命名不齐)
            qt = query_op_type.lower()
            hay = (" ".join(entry_op_types) + " " + entry.get("title", "")).lower()
            if qt in hay or any(qt in ot.lower() or ot.lower() in qt
                                for ot in entry_op_types):
                score += W_OP_TYPE_FUZZY
                reasons.append(f"op_type~{query_op_type}")

    # 3. type 字段交叉匹配 (权重 1)
    if query_pattern and query_pattern in PATTERN_TYPE_AFFINITY:
        if entry_type in PATTERN_TYPE_AFFINITY[query_pattern]:
            score += W_TYPE
            reasons.append(f"type≈{entry_type}")

    # 4. position 辅助加分 (仅第二次检索时使用, 权重 1)
    if query_position and query_position in POSITION_PATTERN_AFFINITY:
        affine_patterns = POSITION_PATTERN_AFFINITY[query_position]
        for p in affine_patterns:
            if p in entry_patterns:
                score += W_POSITION
                reasons.append(f"position={query_position}")
                break

    # 5. all_wrong 特例: op_type 精确匹配额外加权 (仅精确, 模糊不享此 boost)
    if query_pattern == "all_wrong" and query_op_type and query_op_type != "unknown" \
            and not is_checklist and query_op_type in entry_op_types:
        score += W_OP_TYPE_ALL_WRONG_BOOST
        reasons.append("all_wrong+op_type")

    # 6. op_name 关键词软匹配通道 (问题7 核心) — pattern/op_type 失效时的主召回。
    #    当 pattern=all_wrong (无区分度) 时这是唯一有效信号。命中分按 IDF 加权:
    #    专有稀有词 (rfft/hyena) 远重于泛词 (padding/size), 单条封顶防霸榜。
    if op_name_kw and not is_checklist:
        hits = op_name_kw & _entry_keyword_pool(entry)
        if hits:
            if idf:
                kw_score = sum(W_OP_NAME_KEYWORD * idf.get(h, 1.0) for h in hits)
            else:
                kw_score = len(hits) * W_OP_NAME_KEYWORD
            kw_score = min(kw_score, W_OP_NAME_KEYWORD_CAP)
            score += kw_score
            # reason 标注命中词 (按 IDF 降序, 让最有区分度的词排前)
            ordered = sorted(hits, key=lambda h: -(idf.get(h, 1.0) if idf else 1.0))
            reasons.append("op_name_kw:" + ",".join(ordered))

    return round(score, 2), reasons


def search_knowledge_base(kb_path: str, op_type: str | None = None,
                          pattern: str | None = None, position: str | None = None,
                          top_k: int = 3, op_name: str | None = None) -> dict:
    """
    结构化 RAG 检索: 根据 op_type + pattern + position + op_name 关键词筛选并评分排序。

    返回:
      {
        "query": {"op_type": ..., "pattern": ..., "position": ..., "top_k": ..., "op_name": ...},
        "matched_entries": [...],       # top-K 普通条目 (按 score 降序, 含 match_reason)
        "checklists": [...],            # op_type 匹配的 CHECKLIST (不占 K 配额)
        "total_kb_size": N,
        "fallback_to_full_load": bool
      }
    """
    if _kb_disabled():
        result = _empty_search_result(op_type, pattern, position, top_k, op_name)
        result["disabled_by_ablation"] = True
        print("[KB-SEARCH] disabled by ABLATE_KB=1", file=sys.stderr)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    if not os.path.exists(kb_path):
        print(f"[KB-SEARCH] ⚠️ 知识库文件不存在: {kb_path}", file=sys.stderr)
        return _empty_search_result(op_type, pattern, position, top_k, op_name)

    try:
        with open(kb_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[KB-SEARCH] ⚠️ 知识库读取失败: {e}", file=sys.stderr)
        return _empty_search_result(op_type, pattern, position, top_k, op_name)

    if not isinstance(data, list):
        print(f"[KB-SEARCH] ⚠️ 知识库格式错误 (期望 list)", file=sys.stderr)
        return _empty_search_result(op_type, pattern, position, top_k, op_name)

    valid = [e for e in data if _is_valid_entry(e)]

    # 分离 CHECKLIST 和普通条目
    checklists = [e for e in valid if _is_checklist(e)]
    normal_entries = [e for e in valid if not _is_checklist(e)]

    # op_name 分词 (问题7): op_type/pattern 失效时的关键词召回通道
    op_name_kw = _op_name_keywords(op_name)
    # 关键词 IDF: 仅在用到 op_name 通道时计算 (省开销)
    idf = _keyword_idf(normal_entries) if op_name_kw else None

    # CHECKLIST 按 op_types 数组精确匹配 (unknown 不参与)
    matched_checklists = []
    if op_type and op_type != "unknown":
        for cl in checklists:
            if op_type in cl.get("op_types", []):
                matched_checklists.append(cl)

    # 普通条目评分排序
    scored = []
    for entry in normal_entries:
        score, reasons = _score_entry(entry, pattern, op_type, position,
                                      op_name_kw, idf)
        if score > 0:
            scored.append({"score": score, "entry": entry, "reasons": reasons})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top_entries = scored[:top_k]

    # Fallback: 无任何命中 → 返回前 top_k 条
    fallback = len(top_entries) == 0 and len(matched_checklists) == 0
    if fallback:
        print(f"[KB-SEARCH] ⚠️ 无匹配条目, fallback 到前 {top_k} 条", file=sys.stderr)
        top_entries = [{"score": 0, "entry": e, "reasons": ["fallback"]}
                       for e in normal_entries[:top_k]]
        matched_checklists = checklists[:top_k]

    result = {
        "query": {
            "op_type": op_type,
            "pattern": pattern,
            "position": position,
            "top_k": top_k,
            "op_name": op_name,
            "op_name_keywords": sorted(op_name_kw),
        },
        "matched_entries": [
            {
                "index": valid.index(s["entry"]) if s["entry"] in valid else -1,
                "knowledge_id": _knowledge_id(s["entry"]),
                "score": s["score"],
                "match_reason": s.get("reasons", []),
                "title": s["entry"]["title"],
                "patterns": s["entry"].get("patterns", []),
                "op_types": s["entry"].get("op_types", []),
                "feature": s["entry"]["feature"],
                "reason": s["entry"]["reason"],
                "fix": s["entry"]["fix"],
                "type": s["entry"]["type"],
            }
            for s in top_entries
        ],
        "checklists": [
            {
                "knowledge_id": _knowledge_id(cl),
                "title": cl["title"],
                "patterns": cl.get("patterns", []),
                "op_types": cl.get("op_types", []),
                "feature": cl["feature"],
                "reason": cl["reason"],
                "fix": cl["fix"],
                "type": cl["type"],
            }
            for cl in matched_checklists
        ],
        "total_kb_size": len(valid),
        "fallback_to_full_load": fallback,
    }

    n_matched = len(result["matched_entries"])
    n_checklists = len(result["checklists"])
    print(f"[KB-SEARCH] ✅ 检索完成 (op_type={op_type}, pattern={pattern}, "
          f"position={position}, op_name={op_name})")
    print(f"  知识库总条目: {len(valid)}")
    print(f"  命中普通条目: {n_matched} / top-K={top_k}")
    print(f"  命中 CHECKLIST: {n_checklists}")
    if fallback:
        print(f"  ⚠️ FALLBACK: 无匹配, 已返回全量 {len(valid)} 条")
    for s in top_entries[:top_k]:
        why = ",".join(s.get("reasons", []))
        print(f"    [{s['score']:.1f}] {s['entry']['title']}  ⟵ {why}")
    for cl in matched_checklists:
        print(f"    [CL] {cl['title']}")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _empty_search_result(op_type, pattern, position, top_k, op_name=None) -> dict:
    return {
        "query": {"op_type": op_type, "pattern": pattern, "position": position,
                  "top_k": top_k, "op_name": op_name,
                  "op_name_keywords": sorted(_op_name_keywords(op_name))},
        "matched_entries": [],
        "checklists": [],
        "total_kb_size": 0,
        "fallback_to_full_load": True,
    }


# ============================================================
# Check (相似度检查，辅助 dump 前决策)
# ============================================================

def check_similarity(kb_path: str, candidate_path: str,
                     top_k: int = 3, threshold: float = 0.10) -> dict:
    """
    检查候选条目与知识库现有条目的相似度，辅助 Agent 判断 new/merge/abandon。

    计算方式: 对 (title + feature + patterns + op_types) 拼接文本提取 token 集合，
    与候选条目同字段 token 集合计算 Jaccard 相似度，返回 score >= threshold 的 top-K 条。

    返回 (stdout JSON):
      {
        "candidate_title": ...,
        "candidate_type": ...,
        "similar_entries": [{"index": N, "score": 0.xx, "title": ..., ...}],
        "suggestion": "new | review_needed"
      }
    """
    if _kb_disabled():
        result = {
            "similar_entries": [],
            "suggestion": "disabled",
            "disabled_by_ablation": True,
        }
        print("[KB-CHECK] disabled by ABLATE_KB=1", file=sys.stderr)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    if not os.path.exists(candidate_path):
        print(f"[KB-CHECK] ⚠️ 候选条目文件不存在: {candidate_path}", file=sys.stderr)
        return {}

    try:
        with open(candidate_path) as f:
            candidate = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[KB-CHECK] ⚠️ 候选条目读取失败: {e}", file=sys.stderr)
        return {}

    missing = [field for field in REQUIRED_FIELDS if not candidate.get(field)]
    if missing:
        print(f"[KB-CHECK] ⚠️ 候选条目缺少必填字段: {missing}", file=sys.stderr)
        return {}

    # patterns / op_types 缺失时警告但不中止（check 仅预览）
    if not isinstance(candidate.get("patterns"), list):
        print(f"[KB-CHECK] ⚠️ 候选条目 patterns 不是数组，相似度计算可能偏低", file=sys.stderr)
    if not isinstance(candidate.get("op_types"), list):
        print(f"[KB-CHECK] ⚠️ 候选条目 op_types 不是数组，相似度计算可能偏低", file=sys.stderr)

    if not os.path.exists(kb_path):
        print(f"[KB-CHECK] ⚠️ 知识库不存在: {kb_path}，建议 new", file=sys.stderr)
        result = {
            "candidate_title": candidate["title"],
            "candidate_type": candidate["type"],
            "similar_entries": [],
            "suggestion": "new",
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    try:
        with open(kb_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[KB-CHECK] ⚠️ 知识库读取失败: {e}", file=sys.stderr)
        return {}

    valid = [e for e in data if isinstance(e, dict) and _is_valid_entry(e)]

    # 候选条目: title + feature + patterns + op_types 全部参与 tokenize
    cand_patterns = candidate.get("patterns", []) if isinstance(candidate.get("patterns"), list) else []
    cand_op_types = candidate.get("op_types", []) if isinstance(candidate.get("op_types"), list) else []
    cand_text = (candidate["title"] + " " + candidate["feature"]
                 + " " + " ".join(cand_patterns) + " " + " ".join(cand_op_types))
    cand_tokens = _tokenize(cand_text)

    scored = []
    for i, entry in enumerate(valid):
        entry_patterns = entry.get("patterns", [])
        entry_op_types = entry.get("op_types", [])
        entry_text = (entry["title"] + " " + entry["feature"]
                      + " " + " ".join(entry_patterns) + " " + " ".join(entry_op_types))
        entry_tokens = _tokenize(entry_text)
        score = _jaccard(cand_tokens, entry_tokens)
        if score >= threshold:
            scored.append({
                "index": i,
                "score": round(score, 4),
                "title": entry["title"],
                "type": entry["type"],
                "patterns": entry_patterns,
                "op_types": entry_op_types,
                "feature": entry["feature"],
                "reason": entry["reason"],
                "fix": entry["fix"],
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    top_entries = scored[:top_k]

    suggestion = "review_needed" if top_entries else "new"

    result = {
        "candidate_title": candidate["title"],
        "candidate_type": candidate["type"],
        "similar_entries": top_entries,
        "suggestion": suggestion,
    }

    print(f"[KB-CHECK] 检查完成: {candidate['title']}", file=sys.stderr)
    print(f"  知识库条目总数: {len(valid)}", file=sys.stderr)
    print(f"  相似条目 (score>={threshold}): {len(top_entries)}", file=sys.stderr)
    for e in top_entries:
        print(f"    [{e['score']:.4f}] {e['title']}", file=sys.stderr)
    print(f"  建议: {suggestion}", file=sys.stderr)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


# ============================================================
# Dump (精度通过后写入知识库)
# ============================================================

def dump_success_knowledge(kb_path: str, task_dir: str, op_name: str,
                           action: str = "new",
                           merge_target_title: str | None = None) -> dict | None:
    """
    将 Agent 生成的候选知识库条目写入知识库。

    action:
      new    — 追加为新条目（默认）
      merge  — 用候选内容替换 merge_target_title 对应的已有条目
      abandon — 候选已被现有知识完全覆盖，跳过写入

    读取: {task_dir}/precision_tuning/candidate_kb_entry.json
    """
    if _kb_disabled():
        print("[KB] dump rejected: ABLATE_KB=1", file=sys.stderr)
        return None
    if _kb_read_only():
        print("[KB] dump rejected: ASCENDC_DEBUG_KB_READ_ONLY=1",
              file=sys.stderr)
        return None

    tuning_dir = os.path.join(task_dir, "precision_tuning")

    # 1. 读取候选条目
    candidate_path = os.path.join(tuning_dir, "candidate_kb_entry.json")
    if not os.path.exists(candidate_path):
        print(f"[KB] ⚠️ 候选条目文件不存在: {candidate_path}", file=sys.stderr)
        print(f"    请先执行 Step 5.2: 生成候选知识库条目", file=sys.stderr)
        return None

    try:
        with open(candidate_path) as f:
            candidate = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[KB] ⚠️ 候选条目 JSON 解析失败: {e}", file=sys.stderr)
        return None

    # 2. 验证文本字段完整性
    missing = [f for f in REQUIRED_FIELDS if not candidate.get(f)]
    if missing:
        print(f"[KB] ⚠️ 候选条目缺少必填字段: {missing}", file=sys.stderr)
        return None

    # 3. 验证 type 枚举值
    if candidate["type"] not in VALID_TYPES:
        print(f"[KB] ⚠️ type 值非法: {candidate['type']}，应为以下之一: {VALID_TYPES}", file=sys.stderr)
        return None

    # 4. 严格校验 patterns 字段（写入端强制）
    patterns_val = candidate.get("patterns")
    if not isinstance(patterns_val, list):
        print(f"[KB] ❌ patterns 字段必须为数组，当前类型: {type(patterns_val).__name__}", file=sys.stderr)
        print(f"    正确示例: \"patterns\": [\"all_wrong\", \"scattered\"]", file=sys.stderr)
        return None
    invalid_patterns = [p for p in patterns_val if p not in VALID_PATTERNS]
    if invalid_patterns:
        print(f"[KB] ❌ patterns 包含非法值: {invalid_patterns}", file=sys.stderr)
        print(f"    合法值: {VALID_PATTERNS}", file=sys.stderr)
        return None

    # 5. 严格校验 op_types 字段（写入端强制）
    op_types_val = candidate.get("op_types")
    if not isinstance(op_types_val, list):
        print(f"[KB] ❌ op_types 字段必须为数组，当前类型: {type(op_types_val).__name__}", file=sys.stderr)
        print(f"    正确示例: \"op_types\": [\"reduction\"] 或空数组 []", file=sys.stderr)
        return None
    invalid_op_types = [t for t in op_types_val if not isinstance(t, str) or not t.strip()]
    if invalid_op_types:
        print(f"[KB] ❌ op_types 包含非法值（必须为非空字符串）: {invalid_op_types}", file=sys.stderr)
        return None

    # 6. abandon: 记录日志后直接返回
    if action == "abandon":
        print(f"[KB] ℹ️ action=abandon — 候选已被现有知识覆盖，跳过写入")
        print(f"  title: {candidate['title']}")
        return None

    # 7. 补充 _meta
    import glob as _glob
    forensics_files = _glob.glob(os.path.join(tuning_dir, "forensics_report_*.json"))
    num_attempts = 1
    if forensics_files:
        try:
            attempt_nums = []
            for fp in forensics_files:
                stem = os.path.basename(fp)[len("forensics_report_"):-len(".json")]
                if stem.isdigit():
                    attempt_nums.append(int(stem))
            if attempt_nums:
                num_attempts = max(attempt_nums)
        except (ValueError, OSError):
            pass

    entry = dict(candidate)

    # 7.5 检索闭环回填 (问题7 写侧): 从 op_name 派生检索辅助关键词存入 derived_keywords。
    # agent 留空 op_types 的通用经验 (真实库 46% 如此) 借此仍可被同类算子 op_name 召回——
    # 与检索侧 _entry_keyword_pool/_op_name_keywords 严格同源, 保证"写进去的查得到"。
    derived = sorted(_op_name_keywords(op_name)
                     - {kw.lower() for kw in op_types_val})
    if derived:
        entry["derived_keywords"] = derived

    entry["_meta"] = {
        "op_name": op_name,
        "created_at": datetime.now().isoformat(),
        "attempts_needed": num_attempts,
        "action": action,
    }

    # 7.6 可检索性自检 (质量门, 非阻断): 一条经验要能被 op_name 关键词通道召回, 必须
    # 至少有一个检索锚点 —— op_types 非空 / derived_keywords 非空 / title 含英文专有词
    # (中文标题分词后是单字, 区分度低, 不算锚点)。三者全无 = 纯通用中文标题 + 空类型 +
    # op_name 也派生不出词, 这种条目入库后任何算子都难定位到。告警建议补 title 关键词。
    title_kw = {t for t in _tokenize(entry.get("title", ""))
                if t.isascii() and len(t) >= 2}
    has_anchor = bool(entry.get("op_types") or entry.get("derived_keywords")
                      or title_kw)
    if not has_anchor:
        print(f"[KB] ⚠️ 可检索性弱: 条目无 op_types、无派生关键词、title 无英文特征词, "
              f"入库后难被算子 op_name 检索召回; 建议在 title 补充英文算子/机制关键词。",
              file=sys.stderr)

    # 8. 加载知识库
    kb = []
    if os.path.exists(kb_path):
        with open(kb_path) as f:
            kb = json.load(f)

    # 9. 执行写入操作
    if action == "merge":
        if not merge_target_title:
            print(f"[KB] ⚠️ action=merge 需要 --merge-target-title 参数", file=sys.stderr)
            return None
        target_idx = next(
            (i for i, e in enumerate(kb) if e.get("title") == merge_target_title), None
        )
        if target_idx is None:
            print(f"[KB] ⚠️ 未找到目标条目 (title): {merge_target_title}", file=sys.stderr)
            return None
        # Merge must preserve the target's identity even when the title is edited.
        entry["knowledge_id"] = _knowledge_id(kb[target_idx])
        kb[target_idx] = entry
        print(f"[KB] ✅ 已合并更新条目:")
        print(f"  原 title: {merge_target_title}")
        print(f"  新 title: {entry['title']}")
        print(f"  type: {entry['type']}")
        print(f"  kb_size: {len(kb)} 条 (条目数不变)")
    else:  # action == "new"
        entry["knowledge_id"] = _knowledge_id(entry)
        existing_titles = {e.get("title") for e in kb}
        if entry["title"] in existing_titles:
            print(f"[KB] ⚠️ 知识条目已存在 (title 重复), 跳过: {entry['title']}")
            return None
        # 近重复抑制 (问题7 写侧): new 入库前自检与现有条目相似度, 高相似提示应 merge,
        # 避免近义条目膨胀稀释检索信噪比 (真实库已有大量 Reduction/Padding 近义条目)。
        sim = _most_similar_entry(entry, kb)
        if sim and sim[0] >= _DUP_MERGE_THRESHOLD:
            print(f"[KB] ⚠️ 与现有条目高相似 (Jaccard={sim[0]:.2f}): \"{sim[1]}\"",
                  file=sys.stderr)
            print(f"    建议改用 --action merge --merge-target-title \"{sim[1]}\" "
                  f"合并, 而非新增近重复条目; 如确为不同经验可忽略本提示。",
                  file=sys.stderr)
        kb.append(entry)
        print(f"[KB] ✅ 已写入新知识条目: {entry['title']}")
        print(f"  type: {entry['type']}")
        print(f"  patterns: {entry.get('patterns', [])}")
        print(f"  op_types: {entry.get('op_types', [])}")
        if entry.get("derived_keywords"):
            print(f"  derived_keywords: {entry['derived_keywords']}")
        print(f"  attempts_needed: {num_attempts}")
        print(f"  kb_size: {len(kb)} 条")

    with open(kb_path, "w") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

    return entry


# ============================================================
# Search log
# ============================================================

def _append_search_log(log_dir: str, call_index: int, op_type, pattern,
                       position, top_k: int, result: dict,
                       attempt: int | None = None, op_name=None) -> None:
    if log_dir.endswith(".json"):
        log_path = log_dir
    else:
        log_path = os.path.join(log_dir, "knowledge_search_log.json")

    existing_entries = []
    if os.path.exists(log_path):
        try:
            with open(log_path, encoding="utf-8") as f:
                existing_entries = json.load(f)
            if not isinstance(existing_entries, list):
                existing_entries = []
        except (json.JSONDecodeError, OSError):
            existing_entries = []

    matched = result.get("matched_entries", [])
    retrieved = [
        {
            "rank": rank,
            "knowledge_id": e.get("knowledge_id") or _knowledge_id(e),
            "title": e["title"],
            "score": e["score"],
            "reason": e.get("match_reason", []),
            "kind": "entry",
        }
        for rank, e in enumerate(matched[:top_k], start=1)
    ]
    entry = {
        "attempt": attempt,
        "call_index": call_index,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": {
            "op_type": op_type,
            "pattern": pattern,
            "position": position,
            "top_k": top_k,
            "op_name": op_name,
            "op_name_keywords": result.get("query", {}).get("op_name_keywords", []),
        },
        "matched_count": len(matched),
        "checklist_count": len(result.get("checklists", [])),
        "fallback_to_full_load": result.get("fallback_to_full_load", False),
        "top_titles": [e["title"] for e in matched[:top_k]],
        "top_ids": [e["knowledge_id"] for e in retrieved],
        "retrieved": retrieved,
        # why-matched: 每条命中的维度, 供 trace 解释 + 离线评估召回质量 (问题7)
        "match_reasons": [
            {"knowledge_id": e.get("knowledge_id") or _knowledge_id(e),
             "title": e["title"], "score": e["score"],
             "reason": e.get("match_reason", [])}
            for e in matched[:top_k]
        ],
    }
    existing_entries.append(entry)

    try:
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(existing_entries, f, indent=2, ensure_ascii=False)
    except OSError:
        pass


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="精度问题知识库管理")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # load
    p_load = subparsers.add_parser("load", help="全量加载知识库 (fallback)")
    p_load.add_argument("--kb-path", required=True, help="知识库 JSON 路径")

    # search
    p_search = subparsers.add_parser("search", help="结构化 RAG 检索")
    p_search.add_argument("--kb-path", required=True, help="知识库 JSON 路径")
    p_search.add_argument("--op-type", default=None,
                          help="算子类型 (来自 L8, 如 reduction/pooling/loss/matmul/normalization)")
    p_search.add_argument("--pattern", default=None,
                          help=f"误差模式 (来自取证 primary_hint, 合法值: {VALID_PATTERNS})")
    p_search.add_argument("--position", default=None,
                          help="误差位置特征 (第二次检索用, 如 tail/boundary/head/scattered)")
    p_search.add_argument("--op-name", default=None,
                          help="算子名 (如 023_HyenaFftSizePaddingRfft); 分词后做关键词软匹配, "
                               "op_type=unknown/pattern=all_wrong 时的主召回通道 (问题7)")
    p_search.add_argument("--task-dir", default=None,
                          help="任务目录; --op-name 缺省时从此目录的 events/目录名回退自取")
    p_search.add_argument("--top-k", type=int, default=3)
    p_search.add_argument("--log-path", default=None)
    p_search.add_argument("--call-index", type=int, default=0)
    p_search.add_argument("--attempt", type=int, default=None)

    # check
    p_check = subparsers.add_parser("check", help="检查候选条目与知识库的相似度")
    p_check.add_argument("--kb-path", required=True)
    p_check.add_argument("--candidate-path", required=True)
    p_check.add_argument("--top-k", type=int, default=3)
    p_check.add_argument("--threshold", type=float, default=0.10)

    # dump
    p_dump = subparsers.add_parser("dump", help="精度通过后写入知识库 (支持 new/merge/abandon)")
    p_dump.add_argument("--kb-path", required=True)
    p_dump.add_argument("--task-name", required=True)
    p_dump.add_argument("--task-dir", default=None)
    p_dump.add_argument("--op-name", required=True)
    p_dump.add_argument("--action", choices=["new", "merge", "abandon"], default="new")
    p_dump.add_argument("--merge-target-title", default=None)

    args = parser.parse_args()

    if args.command == "load":
        load_knowledge_base(args.kb_path)
    elif args.command == "search":
        # op_name: 显式 --op-name 优先; 缺省时从 --task-dir 回退自取 (events.session_started
        # 或目录名)，使关键词通道在 agent 未传时仍可用 (问题7)。
        op_name = args.op_name
        if not op_name and getattr(args, "task_dir", None):
            op_name = _op_name_from_task_dir(args.task_dir)
        result = search_knowledge_base(
            args.kb_path,
            op_type=args.op_type,
            pattern=args.pattern,
            position=args.position,
            top_k=args.top_k,
            op_name=op_name,
        )
        if getattr(args, "log_path", None):
            _append_search_log(
                log_dir=args.log_path,
                call_index=getattr(args, "call_index", 0),
                op_type=args.op_type,
                pattern=args.pattern,
                position=args.position,
                top_k=args.top_k,
                result=result,
                attempt=getattr(args, "attempt", None),
                op_name=op_name,
            )
    elif args.command == "check":
        check_similarity(
            args.kb_path,
            args.candidate_path,
            top_k=args.top_k,
            threshold=args.threshold,
        )
    elif args.command == "dump":
        task_dir = args.task_dir or str(REPO_ROOT / args.task_name)
        result = dump_success_knowledge(
            args.kb_path,
            task_dir,
            args.op_name,
            action=args.action,
            merge_target_title=args.merge_target_title,
        )
        if result is None:
            sys.exit(1)


if __name__ == "__main__":
    main()
