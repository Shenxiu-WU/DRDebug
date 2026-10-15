#!/usr/bin/env python3
"""Anti-cheat detector for AscendC kernel generation + precision tuning.

两类使用场景共享同一 verify：

1) 精度调优（有 baseline）：调优期间禁止改 Python wrapper
   - snapshot: 调优前保存 reference/wrapper 副本 + sha256
   - verify:   hash 对比 + AST 退化检测 + C++ kernel 扫描
   - restore:  检测到作弊时从基线恢复

2) Kernel 生成（无 baseline）：验收生成结果是否偷偷调 ATen / Tensor 计算方法
   - 直接 verify，跳过 hash 对比（无 .bench_baseline/ 时自动跳过）
   - 重点靠 AST + C++ 层扫描捕获 "kernel 目录存在但内部直接调 at::xxx / x.sum() 之类"

检测规则：
  - HASH_DIFF / DELETED:       reference/wrapper 与 baseline sha256 不一致（仅精度调优场景）
  - AST_FAIL:type{1-4}:        validate_ascendc_impl.py 命中 4 类 PyTorch 退化
  - CPP_ATEN_CALL:<op>:        kernel/*.cpp|h 里调 at::<非白名单 op>
  - CPP_ATEN_HEADER:<op>:      #include <ATen/ops/<op>.h>（算子头文件基本等于作弊意图）
  - CPP_TORCH_CALL:<op>:       kernel 里调 torch::<非白名单 op>
  - CPP_TENSOR_METHOD:<m>:     kernel 里用 tensor 计算方法（如 x.cumsum()、x.histc()）
  - CPP_PYTHON_EXEC:<api>:     C++ 扩展导入时执行任意 Python 代码
  - CPP_GLOBAL_MONKEY_PATCH:   C++/嵌入 Python 修改 torch/torch_npu 全局符号
  - NO_KERNEL_LAUNCH:          kernel/ 里找不到任何 <<<...>>> 或 aclrtLaunchKernel 触发

退出码：verify 0=CLEAN, 1=CHEAT；snapshot/restore 始终 0（除非 IO 错误）。

典型用法：
  # 精度调优（有 baseline）
  python3 anticheat.py snapshot /path/to/task_dir
  python3 anticheat.py verify   /path/to/task_dir
  python3 anticheat.py restore  /path/to/task_dir

  # Kernel 生成验收（无 baseline, 只做 AST + C++ 扫描）
  python3 anticheat.py verify /path/to/task_dir --json
"""
import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


WRAPPER_FILES = ["model.py", "model_new_ascendc.py", "model_new_tilelang.py"]
DEFAULT_BASELINE_DIRNAME = ".bench_baseline"

# ── C++ kernel 扫描规则 ──
CPP_SUFFIXES = {".cpp", ".cc", ".cxx", ".h", ".hpp"}

# at:: / torch:: 调用中放行的 "allocator / metadata" 白名单
CPP_ALLOWED_ATEN_FUNCS = {
    # allocator
    "empty", "empty_like", "empty_strided",
    "zeros", "zeros_like",
    "ones", "ones_like",
    "full", "full_like",
    "from_blob",
    # TensorOptions metadata builders — at::device(...).dtype(...) 这种链式 option 构造
    "device", "dtype", "scalar_type", "layout",
    "requires_grad", "memory_format", "pinned_memory",
}

ATEN_CALL_RE = re.compile(r"\bat::([A-Za-z_]\w*)\s*\(")
TORCH_CALL_RE = re.compile(r"\btorch::([A-Za-z_]\w*)\s*\(")
METHOD_CALL_RE = re.compile(r"\.([A-Za-z_]\w*)\s*\(")
ATEN_OPS_INCLUDE_RE = re.compile(r"#\s*include\s*[<\"]ATen/ops/([A-Za-z_]\w*)\.h[>\"]")
TRIPLE_CHEVRON_RE = re.compile(r"<<<[^<>]+>>>\s*\(")
ACLRT_LAUNCH_RE = re.compile(r"\b(aclrtLaunchKernel|ACLRT_LAUNCH_KERNEL|Launch[A-Za-z]*Kernel)\b")
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
LINE_COMMENT_RE = re.compile(r"//[^\n]*")
STRING_LITERAL_RE = re.compile(r'"(?:\\.|[^"\\])*"')
PYTHON_EXEC_RES = (
    ("PyRun_SimpleString", re.compile(r"\bPyRun_SimpleString(?:Flags)?\s*\(")),
    ("PyRun_String", re.compile(r"\bPyRun_String(?:Flags)?\s*\(")),
    ("PyEval_EvalCode", re.compile(r"\bPyEval_EvalCode\s*\(")),
    ("pybind11::exec", re.compile(r"\b(?:pybind11|py)::exec\s*\(")),
)
PYBIND_ATTR_ASSIGN_RE = re.compile(
    r"\b([A-Za-z_]\w*)\.attr\s*\(\s*[\"']([A-Za-z_]\w*)[\"']\s*\)\s*=(?!=)"
)
PYBIND_FRAMEWORK_IMPORT_RE = re.compile(
    r"\b([A-Za-z_]\w*)\s*=\s*(?:(?:pybind11|py)::)?(?:module_?|module)::import\s*"
    r"\(\s*[\"'](torch(?:_npu)?(?:\.[A-Za-z_]\w*)*)[\"']\s*\)"
)
FRAMEWORK_ASSIGN_RE = re.compile(
    r"\b((?:torch|torch_npu)(?:\.[A-Za-z_]\w*)+)\s*=(?!=)"
)
FRAMEWORK_SETATTR_RE = re.compile(
    r"\bsetattr\s*\(\s*(torch(?:_npu)?(?:\.[A-Za-z_]\w*)*)\s*,"
)
EMBEDDED_FRAMEWORK_IMPORT_RE = re.compile(
    r"\bimport\s+(torch(?:_npu)?(?:\.[A-Za-z_]\w*)*)\s+as\s+([A-Za-z_]\w*)"
)


def repo_root() -> Path:
    # scripts/ -> ascendc-debug/ -> ascendc/ -> skills/ -> <repo_root>
    return Path(__file__).resolve().parents[4]


def default_validator() -> Path:
    return repo_root() / "skills/ascendc/ascendc-translator/scripts/validate_ascendc_impl.py"


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_forbidden_tensor_methods() -> set:
    """Load FORBIDDEN_TENSOR_METHODS from validate_ascendc_impl.py (shared source of truth).

    Fallback to a built-in conservative set if the validator is missing.
    """
    try:
        validator = default_validator()
        if not validator.exists():
            raise FileNotFoundError(validator)
        spec = importlib.util.spec_from_file_location("_validator_mod", str(validator))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        methods = set(module.FORBIDDEN_TENSOR_METHODS)
        methods |= {"histc", "bincount", "histogram"}
        return methods
    except Exception:
        return {
            "sum", "mean", "max", "min", "prod", "cumsum", "cumprod",
            "matmul", "mm", "bmm", "add", "sub", "mul", "div",
            "relu", "sigmoid", "tanh", "gelu", "silu", "softmax",
            "exp", "log", "sqrt", "pow",
            "histc", "bincount", "histogram",
        }


def _strip_comments_and_strings(src: str) -> str:
    src = BLOCK_COMMENT_RE.sub("", src)
    src = LINE_COMMENT_RE.sub("", src)
    src = STRING_LITERAL_RE.sub('""', src)
    return src


def _strip_comments(src: str) -> str:
    """Strip comments but preserve string literals for embedded-Python scanning."""
    return LINE_COMMENT_RE.sub("", BLOCK_COMMENT_RE.sub("", src))


def _line_of(src: str, pos: int) -> int:
    return src[:pos].count("\n") + 1


def _check_cpp_regression(task_dir: Path) -> dict:
    """Scan kernel/**/*.{cpp,h,...} for cheating patterns.

    Returns a dict with:
      status: "pass" | "fail" | "no_kernel_dir"
      violations: list of {file, line, type, detail}
      launch_found: bool
      files_scanned: list[str]
    """
    kernel_dir = task_dir / "kernel"
    if not kernel_dir.exists():
        return {"status": "no_kernel_dir", "violations": [], "launch_found": False, "files_scanned": []}

    forbidden_methods = _load_forbidden_tensor_methods()
    violations = []
    launch_found = False
    files_scanned = []

    for path in sorted(kernel_dir.rglob("*")):
        if path.is_dir():
            continue
        if "build" in path.parts:
            continue
        if path.suffix not in CPP_SUFFIXES:
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = str(path.relative_to(task_dir))
        files_scanned.append(rel)
        src_with_strings = _strip_comments(raw)
        src = _strip_comments_and_strings(raw)

        # Runtime Python execution from a pybind extension can redefine the
        # protected reference API without changing wrapper hashes.  Treat the
        # execution primitive itself as a violation; normal bindings use m.def
        # and do not need PyRun/py::exec.
        for api_name, pattern in PYTHON_EXEC_RES:
            for m in pattern.finditer(src_with_strings):
                violations.append({
                    "file": rel, "line": _line_of(src_with_strings, m.start()),
                    "type": "CPP_PYTHON_EXEC", "detail": api_name,
                })

        framework_aliases = {"torch", "torch_npu"}
        framework_aliases.update(
            m.group(1) for m in PYBIND_FRAMEWORK_IMPORT_RE.finditer(src_with_strings)
        )
        for m in PYBIND_ATTR_ASSIGN_RE.finditer(src_with_strings):
            receiver, attribute = m.group(1), m.group(2)
            if receiver not in framework_aliases and not re.search(
                    r"(?:torch|npu)", receiver, re.IGNORECASE):
                continue
            violations.append({
                "file": rel, "line": _line_of(src_with_strings, m.start()),
                "type": "CPP_GLOBAL_MONKEY_PATCH",
                "detail": f'{receiver}.attr("{attribute}") assignment',
            })
        for m in FRAMEWORK_ASSIGN_RE.finditer(src_with_strings):
            violations.append({
                "file": rel, "line": _line_of(src_with_strings, m.start()),
                "type": "CPP_GLOBAL_MONKEY_PATCH", "detail": f"{m.group(1)} assignment",
            })
        embedded_aliases = {
            m.group(2): m.group(1)
            for m in EMBEDDED_FRAMEWORK_IMPORT_RE.finditer(src_with_strings)
        }
        for alias, module_name in embedded_aliases.items():
            alias_assign_re = re.compile(
                rf"\b({re.escape(alias)}(?:\.[A-Za-z_]\w*)+)\s*=(?!=)"
            )
            for m in alias_assign_re.finditer(src_with_strings):
                canonical = module_name + m.group(1)[len(alias):]
                violations.append({
                    "file": rel, "line": _line_of(src_with_strings, m.start()),
                    "type": "CPP_GLOBAL_MONKEY_PATCH",
                    "detail": f"{canonical} assignment",
                })
        for m in FRAMEWORK_SETATTR_RE.finditer(src_with_strings):
            violations.append({
                "file": rel, "line": _line_of(src_with_strings, m.start()),
                "type": "CPP_GLOBAL_MONKEY_PATCH", "detail": f"setattr({m.group(1)}, ...)",
            })

        if TRIPLE_CHEVRON_RE.search(src) or ACLRT_LAUNCH_RE.search(src):
            launch_found = True

        for m in ATEN_CALL_RE.finditer(src):
            name = m.group(1)
            if name in CPP_ALLOWED_ATEN_FUNCS:
                continue
            violations.append({
                "file": rel, "line": _line_of(src, m.start()),
                "type": "CPP_ATEN_CALL", "detail": f"at::{name}",
            })

        for m in TORCH_CALL_RE.finditer(src):
            name = m.group(1)
            if name in CPP_ALLOWED_ATEN_FUNCS:
                continue
            violations.append({
                "file": rel, "line": _line_of(src, m.start()),
                "type": "CPP_TORCH_CALL", "detail": f"torch::{name}",
            })

        for m in ATEN_OPS_INCLUDE_RE.finditer(src):
            name = m.group(1)
            if name in CPP_ALLOWED_ATEN_FUNCS:
                continue
            violations.append({
                "file": rel, "line": _line_of(src, m.start()),
                "type": "CPP_ATEN_HEADER", "detail": f"#include <ATen/ops/{name}.h>",
            })

        for m in METHOD_CALL_RE.finditer(src):
            name = m.group(1)
            if name in forbidden_methods:
                violations.append({
                    "file": rel, "line": _line_of(src, m.start()),
                    "type": "CPP_TENSOR_METHOD", "detail": f".{name}()",
                })

    if files_scanned and not launch_found:
        violations.append({
            "file": "kernel/", "line": 0,
            "type": "NO_KERNEL_LAUNCH",
            "detail": "kernel 目录下未发现任何 <<<...>>> 或 aclrtLaunchKernel 调用",
        })

    return {
        "status": "pass" if not violations else "fail",
        "violations": violations,
        "launch_found": launch_found,
        "files_scanned": files_scanned,
    }


def cmd_snapshot(args) -> int:
    task_dir = Path(args.task_dir).resolve()
    baseline_dir = task_dir / args.baseline_name
    baseline_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    skipped = []
    for fname in WRAPPER_FILES:
        src = task_dir / fname
        if not src.exists():
            continue
        dst = baseline_dir / fname
        hash_path = baseline_dir / f"{fname}.sha256"
        if dst.exists() and hash_path.exists():
            skipped.append(fname)
            continue
        shutil.copy2(src, dst)
        hash_path.write_text(sha256sum(src) + "\n")
        saved.append(fname)

    result = {"baseline_dir": str(baseline_dir), "saved": saved, "skipped": skipped}
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"[SNAPSHOT] baseline → {baseline_dir}")
        for f in saved:
            print(f"  - {f}")
        for f in skipped:
            print(f"  - {f} (existing baseline kept)")
        if not saved:
            print("  (no wrapper files found — nothing to snapshot)")
    return 0


def _check_ast(wrapper: Path, validator: Path) -> dict:
    if not validator.exists():
        return {"status": "validator_missing", "path": str(validator)}
    proc = subprocess.run(
        [sys.executable, str(validator), str(wrapper), "--json"],
        capture_output=True,
        text=True,
    )
    try:
        data = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except json.JSONDecodeError:
        return {"status": "parse_error", "stdout": proc.stdout[:500], "stderr": proc.stderr[:500]}

    if data.get("valid"):
        return {"status": "pass"}
    return {
        "status": "fail",
        "regression_type": data.get("regression_type"),
        "suggestion": data.get("suggestion", ""),
    }


def cmd_verify(args) -> int:
    task_dir = Path(args.task_dir).resolve()
    baseline_dir = task_dir / args.baseline_name
    reasons = []
    details = {"hash": {}, "ast": {}}

    # 1. hash 对比
    for fname in WRAPPER_FILES:
        # 对基线副本实时重算，不信任存储的 .sha256 文本（agent 可篡改文本使其等于
        # 被改文件的 hash）。与 engine gate (gates/common.py:check_anticheat) 对齐：
        # 两条校验路径都独立重算基线副本，不读存储 hash。
        base_file = baseline_dir / fname
        if not base_file.exists():
            continue
        base_hash = sha256sum(base_file)
        cur = task_dir / fname
        if not cur.exists():
            reasons.append(f"DELETED:{fname}")
            details["hash"][fname] = "deleted"
            continue
        cur_hash = sha256sum(cur)
        if cur_hash != base_hash:
            reasons.append(f"HASH_DIFF:{fname}")
            details["hash"][fname] = {"baseline": base_hash, "current": cur_hash}
        else:
            details["hash"][fname] = "unchanged"

    # 2. AST 退化检测（model_new_ascendc.py）
    wrapper = task_dir / "model_new_ascendc.py"
    if wrapper.exists():
        validator = Path(args.validator) if args.validator else default_validator()
        ast = _check_ast(wrapper, validator)
        details["ast"] = ast
        if ast["status"] == "fail":
            reasons.append(f"AST_FAIL:type{ast.get('regression_type')}")
        elif ast["status"] == "validator_missing":
            reasons.append(f"VALIDATOR_MISSING:{ast.get('path')}")
        elif ast["status"] in ("parse_error",):
            reasons.append(f"AST_ERROR:{ast['status']}")
    else:
        details["ast"] = {"status": "wrapper_missing"}

    # 3. C++ kernel 源码扫描
    cpp = _check_cpp_regression(task_dir)
    details["cpp"] = cpp
    if cpp["status"] == "fail":
        for v in cpp["violations"]:
            reasons.append(f"{v['type']}:{v['detail']}@{v['file']}:{v['line']}")

    verdict = "CHEAT" if reasons else "CLEAN"
    result = {"verdict": verdict, "reasons": reasons, "details": details}

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        icon = "🚨" if verdict == "CHEAT" else "✅"
        print(f"[VERIFY] {icon} {verdict}")
        if reasons:
            for r in reasons:
                print(f"  - {r}")
            if details.get("ast", {}).get("status") == "fail":
                print(f"  AST suggestion: {details['ast'].get('suggestion', '')}")
        else:
            print("  wrapper hash 未变 + AST 退化检测通过")

    return 1 if verdict == "CHEAT" else 0


def cmd_restore(args) -> int:
    task_dir = Path(args.task_dir).resolve()
    baseline_dir = task_dir / args.baseline_name
    restored = []
    for fname in WRAPPER_FILES:
        src = baseline_dir / fname
        if not src.exists():
            continue
        shutil.copy2(src, task_dir / fname)
        restored.append(fname)

    result = {"restored": restored}
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"[RESTORE] ← {baseline_dir}")
        for f in restored:
            print(f"  - {f}")
        if not restored:
            print("  (no baseline files — nothing to restore)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("task_dir", help="absolute path to task directory")
    common.add_argument("--baseline-name", default=DEFAULT_BASELINE_DIRNAME,
                        help=f"baseline subdirectory under task_dir (default: {DEFAULT_BASELINE_DIRNAME})")
    common.add_argument("--json", action="store_true", help="machine-readable JSON output")

    p = argparse.ArgumentParser(
        description="Anti-cheat detector for precision tuning "
                    "(wrapper hash + AST regression check)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("snapshot", parents=[common],
                   help="save baseline copies + sha256 before tuning starts")
    sp_ver = sub.add_parser("verify", parents=[common],
                            help="hash compare + AST regression check after tuning")
    sp_ver.add_argument("--validator",
                        help="path to validate_ascendc_impl.py (auto-detected by default)")
    sub.add_parser("restore", parents=[common],
                   help="restore wrappers from baseline (on cheat detection)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    dispatch = {"snapshot": cmd_snapshot, "verify": cmd_verify, "restore": cmd_restore}
    sys.exit(dispatch[args.cmd](args))


if __name__ == "__main__":
    main()
