"""knowledge_candidate.py — engine-owned KB candidate synthesis on clean success.

The legacy skill flow expected the agent to create
``precision_tuning/candidate_kb_entry.json`` after Gate-V PASS.  In the engine
flow the agent exits before validate, so the PASS fact only exists in runner.
This module fills that gap without changing the debug loop: on terminal clean
success, synthesize a conservative candidate from already persisted artifacts,
run ``precision_knowledge.py check``, and leave final dumping to
``knowledge_finalize.py``.

The synthesis is best-effort.  Failure only writes a traceable meta file and
never changes the session outcome.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from engine.exit_artifacts import build_diagnosis_summary

_ENGINE_DIR = Path(__file__).resolve().parent
_KB_SCRIPT = _ENGINE_DIR.parent / "scripts" / "precision_knowledge.py"
_CHECK_TIMEOUT_SEC = 120
_DUPLICATE_SCORE = 0.55

_VALID_PATTERNS = {
    "tail_spike",
    "uniform_offset",
    "scattered",
    "magnitude_correlated",
    "nan_inf_contamination",
    "dimension_concentration",
    "boundary_concentration",
    "all_wrong",
}


def _skip(reason: str, **extra) -> dict:
    return {"generated": False, "skipped": True, "reason": reason, **extra}


def _read_json(path: Path):
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _write_meta(task_dir: Path, result: dict) -> None:
    try:
        out = task_dir / "precision_tuning" / "candidate_kb_entry_meta.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    except OSError:
        pass


def _latest_attempt(status: dict) -> Optional[int]:
    attempts = status.get("attempts_used")
    if not isinstance(attempts, int) or attempts <= 0:
        return None
    return attempts - 1


def _load_diagnosis(task_dir: Path, attempt: int) -> dict:
    tuning = task_dir / "precision_tuning"
    data = _read_json(tuning / f"diagnosis_summary_attempt_{attempt}.json")
    if isinstance(data, dict):
        return data
    built = build_diagnosis_summary(task_dir, attempt)
    return built if isinstance(built, dict) else {}


def _load_claude_result(task_dir: Path, attempt: int) -> dict:
    data = _read_json(task_dir / f"_claude_result_attempt{attempt}.json")
    return data if isinstance(data, dict) else {}


def _latest_report(task_dir: Path, attempt: int, stem: str) -> dict:
    tuning = task_dir / "precision_tuning"
    for idx in range(attempt, -1, -1):
        data = _read_json(tuning / f"{stem}_{idx}.json")
        if isinstance(data, dict):
            return data
    return {}


def _clean_line(line: str) -> str:
    line = re.sub(r"^\s*[-*+]\s+", "", line)
    line = re.sub(r"^\s*\d+[.)]\s*", "", line)
    line = line.strip(" \t#`")
    line = line.replace("**", "").replace("__", "")
    return line.strip()


def _line_score(line: str) -> int:
    lower = line.lower()
    weights = (
        ("dtype", 8), ("typecast", 8), ("cast", 7), ("fp16", 6),
        ("bf16", 6), ("float32", 6), ("round", 6), ("类型", 6),
        ("精度", 5), ("reduce", 5), ("归约", 5), ("sum", 4),
        ("tiling", 4), ("layout", 4), ("布局", 4), ("mask", 4),
        ("sync", 4), ("barrier", 4), ("padding", 4), ("tail", 4),
        ("根因", 3), ("原因", 3), ("不一致", 3), ("未", 2),
    )
    return sum(w for token, w in weights if token in lower)


def _headline(text: str, op_name: str) -> str:
    lower_text = text.lower()
    if "p_dropout" in lower_text and "dtype" in lower_text and (
        "参考" in text or "reference" in lower_text
    ):
        return "p_dropout 分支输出 dtype 与参考不一致"
    if "dtype" in lower_text and ("参考" in text or "reference" in lower_text) and (
        "不一致" in text or "mismatch" in lower_text
    ):
        return "输出 dtype 与参考不一致"

    candidates: list[tuple[int, str]] = []
    in_code = False
    for raw in text.splitlines():
        if raw.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        line = _clean_line(raw)
        if not line or len(line) < 6 or len(line) > 120:
            continue
        if any(x in line for x in ("构建状态", "本地验证", "验证交给", "修改文件")):
            continue
        if re.search(r"\.(cpp|hpp|h|py|json|md)\b", line):
            continue
        score = _line_score(line)
        if score:
            candidates.append((score, line))
    if candidates:
        candidates.sort(key=lambda x: (-x[0], len(x[1])))
        return candidates[0][1][:80].rstrip("。；;")
    return f"{op_name} 精度修复经验"


def _focused_excerpt(text: str, keywords: tuple[str, ...], max_lines: int = 4) -> str:
    if not keywords:
        return ""
    lower_keys = tuple(k.lower() for k in keywords if k)
    lines = []
    in_code = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        line = _clean_line(raw)
        if not line:
            continue
        if any(x in line for x in ("修改文件", "构建状态", "本地验证")):
            continue
        lower = line.lower()
        if any(k in lower for k in lower_keys):
            lines.append(line)
        if len(lines) >= max_lines:
            break
    return "；".join(lines)[:420].strip("； ")


def _focus_keywords(headline: str) -> tuple[str, ...]:
    lower = headline.lower()
    keys = []
    for token in ("p_dropout", "dtype", "reference", "参考", "输出",
                  "fp16", "bf16", "round", "reduce", "mask", "tiling",
                  "layout", "padding", "tail", "barrier"):
        if token.lower() in lower:
            keys.append(token)
    return tuple(keys)


def _section(text: str, start_words: tuple[str, ...],
             stop_words: tuple[str, ...]) -> str:
    lines = text.splitlines()
    start = None
    for i, raw in enumerate(lines):
        line = _clean_line(raw)
        if any(w in line for w in start_words):
            start = i + 1
            break
    if start is None:
        return ""
    body = []
    in_code = False
    for raw in lines[start:]:
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        line = _clean_line(raw)
        if not line:
            continue
        if any(w in line for w in stop_words):
            break
        body.append(line)
        if len(" ".join(body)) > 420:
            break
    return "；".join(body)[:520].strip("； ")


def _summarize_text(text: str, kind: str) -> str:
    if kind == "reason":
        out = _section(
            text,
            ("根因分析", "根因", "原因"),
            ("修复内容", "修复方案", "构建状态", "验证"),
        )
    else:
        out = _section(
            text,
            ("修复内容", "修复方案", "主要改动"),
            ("构建状态", "本地验证", "验证结果", "最终"),
        )
    if out:
        return out
    lines = [_clean_line(x) for x in text.splitlines()]
    lines = [x for x in lines if x and not x.startswith("```")]
    return "；".join(lines[:6])[:520]


def _infer_type(text: str) -> str:
    lower = text.lower()
    if any(x in lower for x in ("dtype", "cast", "fp16", "bf16", "float16",
                                "float32", "round", "类型转换", "输出类型")):
        return "FIX_PRECISION_TYPECAST"
    if any(x in lower for x in ("nan", "inf", "overflow", "溢出")):
        return "FIX_PRECISION_OVERFLOW"
    if any(x in lower for x in ("reduce", "sum", "归约")):
        return "FIX_PRECISION_REDUCTION"
    if any(x in lower for x in ("barrier", "sync", "pipe", "同步")):
        return "FIX_PRECISION_SYNC"
    if any(x in lower for x in ("layout", "tiling", "stride", "transpose",
                                "布局", "转置")):
        return "FIX_PRECISION_LAYOUT"
    if any(x in lower for x in ("padding", "pad")):
        return "FIX_PRECISION_PADDING"
    if any(x in lower for x in ("tail", "尾块", "尾部")):
        return "FIX_PRECISION_TAIL"
    if any(x in lower for x in ("mask", "branch", "条件", "逻辑")):
        return "FIX_PRECISION_LOGIC"
    return "FIX_PRECISION_OTHER"


def _infer_pattern(forensics: dict, text: str) -> list[str]:
    hint = str(forensics.get("primary_hint") or "")
    if hint in _VALID_PATTERNS:
        return [hint]
    lower = text.lower()
    if "nan" in lower or "inf" in lower:
        return ["nan_inf_contamination"]
    if "all wrong" in lower or "全错" in lower:
        return ["all_wrong"]
    if "tail" in lower or "尾" in lower:
        return ["tail_spike"]
    if "scattered" in lower or "分散" in lower:
        return ["scattered"]
    if "boundary" in lower or "边界" in lower:
        return ["boundary_concentration"]
    return []


def _op_type_from_name(op_name: str) -> list[str]:
    lower = op_name.lower()
    pairs = (
        ("matmul", "matmul"), ("conv", "convolution"),
        ("softmax", "attention"), ("attention", "attention"),
        ("norm", "normalization"), ("sum", "reduction"),
        ("cumsum", "reduction"), ("fft", "fft"),
        ("rope", "attention"), ("adam", "optimizer"),
    )
    found = []
    for token, value in pairs:
        if token in lower and value not in found:
            found.append(value)
    return found


def _infer_op_types(forensics: dict, op_name: str) -> list[str]:
    op = forensics.get("L8_operator") or {}
    op_type = op.get("op_type")
    if isinstance(op_type, str) and op_type and op_type != "unknown":
        return [op_type]
    return _op_type_from_name(op_name)


def _feature(forensics: dict, validation: dict) -> str:
    parts = []
    hint = forensics.get("primary_hint")
    if hint:
        parts.append(f"取证显示主要误差形态为 {hint}")
    first = {}
    for item in forensics.get("outputs") or []:
        if isinstance(item, dict) and item.get("pass_fail") is False:
            first = item
            break
    if first:
        details = []
        for key in ("mismatch_ratio", "max_abs_diff", "max_diff", "mean_abs_diff"):
            if first.get(key) is not None:
                details.append(f"{key} 约为 {first[key]}")
        if details:
            parts.append("失败输出中 " + "、".join(details))
    if validation:
        passed = validation.get("passed_cases")
        total = validation.get("total_cases")
        if passed is not None and total is not None:
            parts.append(f"修复后验证用例 {passed}/{total} 通过")
        elif validation.get("correctness_passed") is True:
            parts.append("修复后 Gate-V 精度验证通过")
    return "；".join(parts) or "修复前存在精度不一致，修复后 Gate-V 精度验证通过"


def _candidate_from_artifacts(task_dir: Path, op_name: str, attempt: int) -> Optional[dict]:
    diagnosis = _load_diagnosis(task_dir, attempt)
    final_response = diagnosis.get("final_response")
    if not final_response:
        claude = _load_claude_result(task_dir, attempt)
        value = claude.get("result")
        final_response = value if isinstance(value, str) else None
    root_cause = diagnosis.get("root_cause") or ""
    fix_plan = diagnosis.get("fix_plan") or ""
    text = "\n".join(x for x in (final_response, root_cause, fix_plan) if x)
    if not text.strip():
        return None

    forensics = _latest_report(task_dir, attempt, "forensics_report")
    validation = _latest_report(task_dir, attempt, "validation_result_attempt")
    title_head = _headline(text, op_name)
    title = f"{op_name} {title_head} (Engine Synthesized Precision Fix)"
    if len(title) > 160:
        title = title[:157].rstrip() + "..."
    focus = _focus_keywords(title_head)
    reason = _focused_excerpt(text, focus) or _summarize_text(text, "reason")
    fix = _focused_excerpt(
        text,
        focus + ("修复", "改为", "保持", "分配", "round-trip", "cast"),
    ) or _summarize_text(text, "fix")
    if not reason:
        reason = title_head
    if not fix:
        fix = "按最终成功轮的最小代码改动修正 AscendC kernel/wrapper 中的精度语义，并用 Gate-V 验证。"
    return {
        "title": title,
        "feature": _feature(forensics, validation),
        "patterns": _infer_pattern(forensics, text),
        "op_types": _infer_op_types(forensics, op_name),
        "reason": reason,
        "fix": fix,
        "type": _infer_type(text),
    }


def _parse_json_stdout(stdout: str) -> dict:
    text = stdout or ""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _check_candidate(candidate_path: Path, kb_path: str, _run: Callable) -> dict:
    if not kb_path or not _KB_SCRIPT.exists():
        return {"success": False, "reason": "kb_path 或 check 脚本缺失"}
    cmd = [
        sys.executable,
        str(_KB_SCRIPT),
        "check",
        "--kb-path",
        str(kb_path),
        "--candidate-path",
        str(candidate_path),
        "--top-k",
        "3",
        "--threshold",
        "0.10",
    ]
    try:
        proc = _run(cmd, capture_output=True, text=True,
                    timeout=_CHECK_TIMEOUT_SEC, check=False)
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "reason": f"check 子进程异常: {exc}"}
    parsed = _parse_json_stdout(proc.stdout or "")
    return {
        "success": proc.returncode == 0 and bool(parsed),
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
        "result": parsed,
    }


def _decide_action(check: dict) -> tuple[str, Optional[str], Optional[float]]:
    if not check.get("success"):
        return "new", None, None
    entries = (check.get("result") or {}).get("similar_entries") or []
    if not entries:
        return "new", None, None
    top = entries[0]
    score = top.get("score")
    title = top.get("title")
    if isinstance(score, (int, float)) and score >= _DUPLICATE_SCORE:
        return "abandon", title if isinstance(title, str) else None, float(score)
    return "new", title if isinstance(title, str) else None, float(score) if isinstance(score, (int, float)) else None


def ensure_candidate_entry(
    task_dir,
    *,
    kb_path: Optional[str],
    status: dict,
    op_name: str,
    _run: Callable = subprocess.run,
) -> dict:
    """Create candidate_kb_entry.json before finalize when a clean success lacks it."""
    task_dir = Path(task_dir)
    candidate_path = task_dir / "precision_tuning" / "candidate_kb_entry.json"
    try:
        if not kb_path:
            result = _skip("kb_path 未配置，跳过候选生成")
        elif candidate_path.exists():
            result = {
                "generated": False,
                "skipped": False,
                "reason": "candidate_kb_entry.json 已存在，保持 agent 产物",
                "candidate_path": str(candidate_path),
            }
        elif status.get("reportable_success") is not True:
            result = _skip(
                "非 clean/reportable success，不生成候选",
                session_outcome=status.get("session_outcome"),
                objective_success=status.get("objective_success"),
                anti_cheat_pass=status.get("anti_cheat_pass"),
                ast_degrade_pass=status.get("ast_degrade_pass"),
            )
        else:
            attempt = _latest_attempt(status)
            if attempt is None:
                result = _skip("attempts_used 缺失，无法定位成功轮")
            else:
                candidate = _candidate_from_artifacts(task_dir, op_name, attempt)
                if candidate is None:
                    result = _skip("成功轮缺少可用诊断文本，无法合成候选", attempt=attempt)
                else:
                    candidate_path.parent.mkdir(parents=True, exist_ok=True)
                    candidate_path.write_text(
                        json.dumps(candidate, ensure_ascii=False, indent=2),
                        encoding="utf-8")
                    check = _check_candidate(candidate_path, str(kb_path), _run)
                    action, similar_title, similar_score = _decide_action(check)
                    if action != "new":
                        candidate["action"] = action
                        candidate_path.write_text(
                            json.dumps(candidate, ensure_ascii=False, indent=2),
                            encoding="utf-8")
                    result = {
                        "generated": True,
                        "skipped": False,
                        "reason": "已生成候选知识条目",
                        "candidate_path": str(candidate_path),
                        "attempt": attempt,
                        "action": action,
                        "similar_title": similar_title,
                        "similar_score": similar_score,
                        "check": check,
                    }
    except Exception as exc:  # noqa: BLE001 — 副产物失败绝不影响 session 终态。
        result = _skip(f"候选生成异常: {exc}")
    _write_meta(task_dir, result)
    return result
