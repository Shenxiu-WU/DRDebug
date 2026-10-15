#!/usr/bin/env python3
"""backfill_derived_keywords.py — 一次性迁移: 给存量 KB 空 op_types 条目回填 derived_keywords。

背景 (问题7 检索闭环): 真实库 46% 条目 op_types=[]。其中**有 _meta.op_name** 的条目,
其算子归属信息只存在 _meta 里, 而检索通道 (_entry_keyword_pool) 不读 _meta → 这些经验
难被同名算子召回 (实测 Softmax 类被归约条目淹没)。本脚本把 _meta.op_name 派生的关键词
写入 derived_keywords 字段 (与 dump_success_knowledge 回填逻辑同源), 让存量条目也可检索。

只处理「op_types 为空 且 _meta.op_name 有值」的条目:
  - op_types 非空: agent 已给算子归属, 检索已生效, 不动。
  - 无 _meta.op_name: 多为通用机制经验 (尾块/归约/数据竞争等), 本就该 op_types=[],
    其 title 机制词已保证可检索, 不应硬塞 op_type → 不动。

幂等: 已有 derived_keywords 的条目跳过。派生词与已有 op_types 取差集 (此处 op_types 空,
即全部派生词)。用法: python utils/backfill_derived_keywords.py --kb-path <path> [--dry-run]
"""
import argparse
import json
import os
import sys

# 复用 dump/检索同源的分词逻辑, 保证写读对称
_SKILL_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "skills", "ascendc", "ascendc-debug", "scripts")
sys.path.insert(0, _SKILL_SCRIPTS)
import precision_knowledge as pk  # noqa: E402


def backfill(kb_path: str, dry_run: bool = False) -> dict:
    with open(kb_path, encoding="utf-8") as f:
        kb = json.load(f)

    changed = []
    for i, e in enumerate(kb):
        if e.get("op_types"):           # 已有算子归属, 检索已生效
            continue
        if e.get("derived_keywords"):   # 幂等: 已回填过
            continue
        op = (e.get("_meta") or {}).get("op_name")
        if not op:                      # 通用机制经验, 不硬塞
            continue
        derived = sorted(pk._op_name_keywords(op))
        if not derived:
            continue
        if not dry_run:
            e["derived_keywords"] = derived
        changed.append({"index": i, "title": e.get("title", "")[:50],
                        "op_name": op, "derived_keywords": derived})

    if changed and not dry_run:
        with open(kb_path, "w", encoding="utf-8") as f:
            json.dump(kb, f, indent=2, ensure_ascii=False)

    return {"total": len(kb), "changed": changed, "dry_run": dry_run}


def main():
    ap = argparse.ArgumentParser(description="存量 KB 空 op_types 条目回填 derived_keywords")
    ap.add_argument("--kb-path", required=True)
    ap.add_argument("--dry-run", action="store_true", help="只预览不写盘")
    args = ap.parse_args()

    if not os.path.exists(args.kb_path):
        print(f"❌ 知识库不存在: {args.kb_path}")
        sys.exit(1)

    r = backfill(args.kb_path, dry_run=args.dry_run)
    tag = "[DRY-RUN] " if r["dry_run"] else ""
    print(f"{tag}知识库总条目: {r['total']}, 回填 {len(r['changed'])} 条:")
    for c in r["changed"]:
        print(f"  #{c['index']} op_name={c['op_name']:18} "
              f"→ {c['derived_keywords']}  ({c['title']})")
    if not r["changed"]:
        print("  (无需回填)")


if __name__ == "__main__":
    main()
