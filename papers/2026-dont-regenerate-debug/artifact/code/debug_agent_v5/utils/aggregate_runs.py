"""aggregate_runs.py — 跨 run 产物聚合统计 (文档 N3: 防止同名任务跨 run 混算)。

背景: 同一算子 (如 023_HyenaFftSizePaddingRfft) 在不同 run 中状态可能相反——
一个 run 作弊 (torch native 绕过 kernel)、另一个 run 真实修复成功。若统计
success/KB候选/作弊数 时不按 run 隔离, 同名任务会被混算, 污染论文结论。

本脚本以 (run_id, level, op_name, attempt) 为主键, 逐任务抽取一条记录, 每条带
provenance (source_task_dir)。success 计数采用与 knowledge_finalize 同源的
reportable 口径: 仅 success && 无作弊记录才计入 (cheat/unknown 不计成功)。

与 generate_report_dynamic.py 不同: 后者是单 run 的 trace.md 表格字符串拼接器
(服务算子生成主流程的性能/延迟报表)。本脚本面向 debug 引擎的结构化产物
(.verify_status / events.jsonl / precision_tuning/*), 做跨 run 隔离的事实聚合。

全程 best-effort: 任一产物缺失/损坏只置该字段 None/unknown, 不崩、不影响其他记录。
"""
import os
import re
import json
import argparse
import sys

# run 根目录下识别"任务目录"的标志产物 (二者之一存在即认定为一个任务)
_TASK_MARKERS = (
    os.path.join(".verify_status", "latest.json"),
    os.path.join(".debug_events", "events.jsonl"),
)
_LEVEL_RE = re.compile(r"level(\d+)", re.IGNORECASE)


def _safe_load_json(path):
    """读 JSON, 任何异常 (缺失/损坏/编码) 返回 None。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _infer_level(task_dir, run_root):
    """从 task_dir 相对 run_root 的中间路径推断 level (levelN/)。

    序号前缀 (NNN_OpName) 不等于 level, 不从 op_name 猜; 推断不到返回 "unknown"。
    """
    try:
        rel = os.path.relpath(task_dir, run_root)
    except ValueError:
        return "unknown"
    m = _LEVEL_RE.search(rel)
    return f"level{m.group(1)}" if m else "unknown"


def _op_name_from_events(task_dir):
    """从 events.jsonl 的 session_started 取 op_name; 失败返回 None。"""
    path = os.path.join(task_dir, ".debug_events", "events.jsonl")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                if e.get("type") == "session_started":
                    return e.get("op_name")
    except (OSError, ValueError):
        return None
    return None


def _cheat_summary(task_dir):
    """读 cheat_history.json → (作弊总数, severity 列表)。缺失/损坏 → (0, [])。

    与 knowledge_finalize 同口径: violation 与 warning 均算非 clean。
    """
    data = _safe_load_json(
        os.path.join(task_dir, "precision_tuning", "cheat_history.json"))
    if not isinstance(data, dict):
        return 0, []
    attempts = data.get("cheating_attempts") or []
    sev = [a.get("severity") for a in attempts if isinstance(a, dict)]
    return len(attempts), sev


def _kb_candidate(task_dir):
    """candidate_kb_entry.json 是否存在 + 其 action; 缺失 → (False, None)。"""
    data = _safe_load_json(
        os.path.join(task_dir, "precision_tuning", "candidate_kb_entry.json"))
    if not isinstance(data, dict):
        return False, None
    return True, data.get("action")


def extract_record(task_dir, run_id, run_root):
    """从单个任务目录抽一条聚合记录。全 best-effort, 缺字段置 None/unknown。"""
    verify = _safe_load_json(
        os.path.join(task_dir, ".verify_status", "latest.json")) or {}
    op_name = _op_name_from_events(task_dir) or os.path.basename(task_dir)
    attempt = verify.get("attempt")
    failure_type = verify.get("failure_type")
    verify_status = (verify.get("verify") or {}).get("status")
    objective_success = failure_type == "success" and verify_status == "passed"

    # 精度: validation_result_attempt_{N}.json (attempt 已知才定位)
    correctness = match_rate = None
    if attempt is not None:
        val = _safe_load_json(os.path.join(
            task_dir, "precision_tuning",
            f"validation_result_attempt_{attempt}.json"))
        if isinstance(val, dict):
            correctness = val.get("correctness_passed")
            match_rate = val.get("match_rate")

    cheat_count, cheat_sev = _cheat_summary(task_dir)
    kb_candidate, kb_action = _kb_candidate(task_dir)

    # reportable 口径 (与 knowledge_finalize 同源): success 且零作弊才算"干净成功"
    reportable_success = objective_success and cheat_count == 0

    return {
        "run_id": run_id,
        "level": _infer_level(task_dir, run_root),
        "op_name": op_name,
        "attempt": attempt,
        "objective_success": objective_success,
        "reportable_success": reportable_success,
        "correctness_passed": correctness,
        "match_rate": match_rate,
        "cheat_count": cheat_count,
        "cheat_severities": cheat_sev,
        "kb_candidate": kb_candidate,
        "kb_action": kb_action,
        "source_task_dir": os.path.abspath(task_dir),  # provenance
    }


def _is_task_dir(path):
    """目录含任一标志产物即为任务目录。"""
    return any(os.path.exists(os.path.join(path, m)) for m in _TASK_MARKERS)


def scan_run(run_root):
    """扫描单个 run 根目录, 返回其下全部任务记录。run_id = 根目录名。"""
    run_id = os.path.basename(os.path.normpath(run_root))
    records = []
    # 任务目录可能在 run 根的任意一级 (deepseek 有 level{N}/ 中间层)。
    for cur, dirs, _files in os.walk(run_root):
        if _is_task_dir(cur):
            records.append(extract_record(cur, run_id, run_root))
            dirs[:] = []  # 任务目录内不再深入 (避免把 work/ 子树当任务)
    records.sort(key=lambda r: (r["level"], r["op_name"], r["attempt"] or -1))
    return records


def _fmt(v):
    """单元格渲染: None → '-', bool → ✓/✗, list → 逗号拼接。"""
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, list):
        return ",".join(str(x) for x in v) if v else "-"
    return str(v)


def render_markdown(records):
    """渲染主表 (主键列在前) + 每 run 小计。cheat/unknown 不计入 success 小计。"""
    lines = ["## 📊 跨 run 产物聚合报告 (N3: 按 run 隔离)\n"]
    header = ("| run_id | level | op_name | attempt | 干净成功 | 客观成功 | "
              "精度 | match_rate | 作弊数 | severity | KB候选 | KB动作 |")
    sep = "| " + " | ".join(["---"] * 12) + " |"
    lines += [header, sep]
    for r in records:
        lines.append("| " + " | ".join(_fmt(r[k]) for k in (
            "run_id", "level", "op_name", "attempt", "reportable_success",
            "objective_success", "correctness_passed", "match_rate",
            "cheat_count", "cheat_severities", "kb_candidate", "kb_action",
        )) + " |")

    # 每 run 小计 (success 口径 = reportable_success, 排除作弊/未知)
    lines.append("\n### 每 run 小计 (success = 干净成功, 作弊不计)\n")
    lines += ["| run_id | 任务数 | 干净成功 | 客观成功 | 作弊任务 | KB候选 |",
              "| --- | --- | --- | --- | --- | --- |"]
    runs = {}
    for r in records:
        runs.setdefault(r["run_id"], []).append(r)
    for run_id in sorted(runs):
        rs = runs[run_id]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            run_id, len(rs),
            sum(1 for r in rs if r["reportable_success"]),
            sum(1 for r in rs if r["objective_success"]),
            sum(1 for r in rs if r["cheat_count"] > 0),
            sum(1 for r in rs if r["kb_candidate"]),
        ))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="跨 run 产物聚合统计 (N3: 防同名任务跨 run 混算)")
    parser.add_argument("-i", "--input", nargs="+", required=True,
                        help="【必填】一个或多个 run 根目录 (各自识别为独立 run_id)")
    parser.add_argument("-o", "--output", default="aggregate_report.md",
                        help="【可选】报告保存路径 (默认 aggregate_report.md)")
    parser.add_argument("--json", action="store_true",
                        help="额外输出 .json (同名, 含全部 provenance 字段)")
    args = parser.parse_args()

    all_records = []
    for run_root in args.input:
        if not os.path.isdir(run_root):
            print(f"⚠️ 跳过 (非目录): {run_root}")
            continue
        recs = scan_run(run_root)
        print(f"📂 {os.path.basename(os.path.normpath(run_root))}: "
              f"{len(recs)} 个任务")
        all_records.extend(recs)

    if not all_records:
        print("⚠️ 未发现任何任务目录 (需含 .verify_status/latest.json 或 "
              ".debug_events/events.jsonl)")
        sys.exit(0)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(render_markdown(all_records))
    print(f"✅ 报告已保存: {os.path.abspath(args.output)}")

    if args.json:
        jpath = os.path.splitext(args.output)[0] + ".json"
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON 已保存: {os.path.abspath(jpath)}")


if __name__ == "__main__":
    main()
