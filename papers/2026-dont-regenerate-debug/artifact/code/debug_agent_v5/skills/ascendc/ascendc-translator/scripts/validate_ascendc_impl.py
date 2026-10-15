#!/usr/bin/env python3
"""AscendC 实现退化检测脚本 — 通过 AST 静态分析检查生成代码是否退化为 PyTorch 原生实现。

检测四种退化类型：
  Type 1: 无 AscendC kernel 扩展导入（纯 PyTorch）
  Type 2: 有扩展导入但 forward() 未调用 kernel 函数
  Type 3: forward() 调用了 kernel 但仍有部分计算使用 torch 接口
  Type 4: forward() 中存在逐元素 Python for 循环（标量写法退化）

用法:
    python validate_ascendc_impl.py <file_path> [--json]

退出码: 0 = 通过, 1 = 检测到退化
"""
import ast
import argparse
import json
import re
import sys


# ---------------------------------------------------------------------------
# 白名单：forward() 中允许的 torch 调用和 tensor 方法
# ---------------------------------------------------------------------------

ALLOWED_TORCH_FUNCS = {
    # buffer 分配
    "empty", "empty_like", "empty_strided",
    "zeros", "zeros_like",
    "ones", "ones_like",
    "full", "full_like",
    # tensor 创建（有时需要用于标量常量 / 索引）
    "tensor", "arange", "linspace",
    # 类型 / 设备
    "as_tensor",
}

ALLOWED_TENSOR_METHODS = {
    # 形状 / 元信息
    "size", "shape", "stride", "numel", "dtype", "device", "dim",
    "is_contiguous", "data_ptr", "element_size", "storage_offset",
    # 布局操作（不执行计算）
    "contiguous", "to", "view", "view_as", "reshape",
    "permute", "transpose", "expand", "expand_as",
    "flatten", "unflatten", "unsqueeze", "squeeze",
    "narrow", "clone", "detach", "t",
    "type", "float", "half", "bfloat16", "int", "long", "bool", "double",
    "cpu", "npu", "cuda",
    "item", "tolist",
    # 原地标记
    "requires_grad_", "zero_",
    # 切片相关
    "index_select",
    # 设备检查
    "is_npu", "is_cuda",
}

ALLOWED_BUILTIN_FUNCS = {
    # Python 内建函数（非 tensor 方法）
    "min", "max", "abs", "len", "range", "int", "float", "bool",
    "list", "tuple", "str", "type", "isinstance", "print",
    "enumerate", "zip", "map", "filter", "sorted", "reversed",
    "hasattr", "getattr", "setattr",
}

FORBIDDEN_TENSOR_METHODS = {
    # 归约操作
    "sum", "mean", "max", "min", "prod", "cumsum", "cumprod",
    "argmax", "argmin", "var", "std",
    # 矩阵 / 线性代数
    "matmul", "mm", "bmm", "addmm",
    # 逐元素算术
    "add", "sub", "mul", "div", "fmod", "remainder",
    "add_", "sub_", "mul_", "div_",
    # 激活函数
    "relu", "sigmoid", "tanh", "gelu", "silu", "elu", "leaky_relu",
    "relu_", "sigmoid_", "tanh_",
    # 数学函数
    "exp", "log", "log2", "log10", "sqrt", "pow", "abs",
    "sin", "cos", "clamp", "clamp_", "ceil", "floor", "round",
    "reciprocal", "neg", "sign",
    # softmax
    "softmax", "log_softmax",
    # 范数 / 归一化
    "norm", "layer_norm", "batch_norm", "group_norm",
    # 卷积 / 线性
    "conv1d", "conv2d", "conv3d", "conv_transpose2d", "linear",
    # 其他
    "dropout", "softplus", "hardtanh", "hardswish",
    # 比较（用于计算，非条件判断时）
    "eq", "ne", "lt", "gt", "le", "ge", "where",
}

# 已知的占位符导入名称（表示扩展模块未正确配置）
PLACEHOLDER_IMPORT_NAMES = {
    "TORCH_EXTENSION_NAME",
}

# AscendC 扩展模块的命名模式
ASCENDC_EXT_PATTERNS = [
    re.compile(r"_\w+_ext$"),          # _xxx_ext
    re.compile(r"\w+_ext$"),            # xxx_ext
    re.compile(r"\w+_ascendc\w*$"),     # xxx_ascendc, xxx_ascendc_ext
    re.compile(r"_ext$"),               # _ext
]


# ---------------------------------------------------------------------------
# AST 辅助函数
# ---------------------------------------------------------------------------

def _resolve_call_name(node):
    """尝试从 ast.Call 节点提取被调用函数的名称字符串。

    返回 (qualifier, attr) 或 (None, name) 或 None。
    例如：torch.empty -> ('torch', 'empty')
          _ext.run_kernel -> ('_ext', 'run_kernel')
          my_func -> (None, 'my_func')
    """
    func = node.func if isinstance(node, ast.Call) else node
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return (func.value.id, func.attr)
        # 处理 torch.nn.functional.relu 形式
        if isinstance(func.value, ast.Attribute):
            inner = func.value
            if isinstance(inner.value, ast.Name):
                return (f"{inner.value.id}.{inner.attr}", func.attr)
    if isinstance(func, ast.Name):
        return (None, func.id)
    return None


def _is_ext_module_name(name):
    """检查名称是否匹配 AscendC 扩展模块的命名模式。"""
    if name in PLACEHOLDER_IMPORT_NAMES:
        return False
    for pattern in ASCENDC_EXT_PATTERNS:
        if pattern.match(name):
            return True
    return False


# ---------------------------------------------------------------------------
# 核心检查
# ---------------------------------------------------------------------------

def find_ascendc_extension_imports(tree):
    """查找所有 AscendC 扩展模块的导入信息。

    检测模式：
    1. import _xxx_ext [as alias]
    2. import xxx_ext [as alias]
    3. from path import _xxx_ext [as alias]
    4. importlib 动态加载

    返回 dict: {alias_or_name: {"name": str, "alias": str|None,
                                  "line": int, "is_placeholder": bool,
                                  "import_style": str}}
    """
    extensions = {}

    for node in ast.walk(tree):
        # --- import xxx_ext [as alias] ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                actual_name = alias.name
                used_name = alias.asname if alias.asname else alias.name
                is_placeholder = actual_name in PLACEHOLDER_IMPORT_NAMES
                if is_placeholder or _is_ext_module_name(actual_name):
                    extensions[used_name] = {
                        "name": actual_name,
                        "alias": alias.asname,
                        "line": node.lineno,
                        "is_placeholder": is_placeholder,
                        "import_style": "import",
                    }

        # --- from xxx import yyy [as alias] ---
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                actual_name = alias.name
                used_name = alias.asname if alias.asname else alias.name
                is_placeholder = actual_name in PLACEHOLDER_IMPORT_NAMES
                if is_placeholder or _is_ext_module_name(actual_name):
                    extensions[used_name] = {
                        "name": actual_name,
                        "alias": alias.asname,
                        "line": node.lineno,
                        "is_placeholder": is_placeholder,
                        "import_style": "from_import",
                    }

    # --- importlib 动态加载检测 ---
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                resolved = _resolve_call_name(node.value)
                if resolved:
                    qual, attr = resolved
                    # importlib.util.module_from_spec(...)
                    if attr == "module_from_spec":
                        extensions[target.id] = {
                            "name": target.id,
                            "alias": None,
                            "line": node.lineno,
                            "is_placeholder": False,
                            "import_style": "importlib",
                        }

    return extensions


def find_model_forward(tree):
    """找到 ModelNew 或 Model 类的 forward 方法节点。

    优先查找 ModelNew，若不存在则查找 Model。
    """
    model_new_forward = None
    model_forward = None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name == "ModelNew":
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == "forward":
                            model_new_forward = item
            elif node.name == "Model":
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == "forward":
                            model_forward = item

    return model_new_forward or model_forward, "ModelNew" if model_new_forward else "Model"


def find_wrapper_functions(tree, ext_names):
    """找到模块级别的辅助函数，这些函数内部调用了扩展模块。

    返回函数名集合。
    """
    wrappers = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    resolved = _resolve_call_name(child)
                    if resolved:
                        qual, attr = resolved
                        if qual in ext_names:
                            wrappers.add(node.name)
                            break
    return wrappers


def collect_class_methods(tree):
    """收集 ModelNew(优先)/Model 类的所有方法定义 {method_name: FunctionDef}。

    供 check_forbidden_torch_ops 递归展开 self.method() 调用 (区分自定义方法 vs
    nn.Module 子模块属性)。优先 ModelNew，与 find_model_forward 口径一致。
    """
    model_new_methods = {}
    model_methods = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in ("ModelNew", "Model"):
            target = model_new_methods if node.name == "ModelNew" else model_methods
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    target[item.name] = item
    return model_new_methods or model_methods


def collect_module_functions(tree):
    """收集模块级函数定义 {func_name: FunctionDef}。

    供 check_forbidden_torch_ops 递归展开模块级 helper 调用，防止把 torch 计算藏进
    模块级函数逃逸检测。
    """
    funcs = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node
    return funcs


def collect_method_wrappers(class_methods, ext_names):
    """收集 (递归) 调用 kernel 扩展的本类方法名集合。

    forward 可能只调 self._run_kernel()，真正的 ext 调用藏在该方法内部。把这类方法名
    并入 wrapper_names，让 check_kernel_calls_in_forward 认出「forward → self.method →
    ext」也算 kernel 被调用 (否则误判 type2)。递归防环。
    """
    def _reaches_ext(method, visited):
        if method is None or id(method) in visited:
            return False
        visited.add(id(method))
        for child in ast.walk(method):
            if not isinstance(child, ast.Call):
                continue
            resolved = _resolve_call_name(child)
            if resolved is None:
                continue
            qual, attr = resolved
            if qual in ext_names:
                return True
            if qual == "self" and attr in class_methods:
                if _reaches_ext(class_methods[attr], visited):
                    return True
        return False

    return {name for name, fn in class_methods.items() if _reaches_ext(fn, set())}


def check_kernel_calls_in_forward(forward_node, ext_names, wrapper_names):
    """检查 forward 中是否调用了 AscendC 扩展模块的函数。

    检测模式：
    1. ext_module.function_name(...)  — 直接调用扩展模块方法
    2. wrapper_func(...)              — 通过 wrapper 函数调用
    3. self.wrapper_name(...)         — 通过类方法调用

    返回被调用信息列表 [{"call": str, "line": int}, ...]
    """
    called = []
    if forward_node is None:
        return called
    for node in ast.walk(forward_node):
        if not isinstance(node, ast.Call):
            continue
        resolved = _resolve_call_name(node)
        if resolved is None:
            continue
        qual, attr = resolved
        # ext_module.function(...)
        if qual in ext_names:
            called.append({"call": f"{qual}.{attr}", "line": node.lineno})
        # wrapper_func(...)
        if qual is None and attr in wrapper_names:
            called.append({"call": attr, "line": node.lineno})
        # self.wrapper(...)
        if qual == "self" and attr in wrapper_names:
            called.append({"call": f"self.{attr}", "line": node.lineno})
    return called


def check_forbidden_torch_ops(forward_node, class_methods=None,
                              module_functions=None, ext_names=None,
                              wrapper_names=None):
    """检查 forward (及其递归调用的 helper) 中是否使用禁止的 torch 计算操作。

    递归语义 (anti-cheat 盲区修复): forward 调用的本类方法 / 模块级 helper 会被递归
    展开检查，防止把 F.pad / torch.matmul 等藏进 helper 逃逸检测。同时借递归区分:
      - self.method() 且 method 是本类已定义方法 → 递归进入 (不判违规)，消除原先把
        「forward 调自定义纯 wrapper 方法」误判 type3 的 false positive。
      - self.xxx() 且 xxx 非本类方法 → 判违规 (方案A: __init__ 赋的属性绝大多数是
        nn.Linear/nn.Conv 子模块，调用等于用 PyTorch 算子，保持原语义)。

    class_methods/module_functions: {name: FunctionDef}，None 时退化为非递归 (向后兼容)。
    ext_names/wrapper_names: kernel 扩展 / wrapper 名集合，递归时跳过 (合法调用)。

    返回违规列表 [{"line": N, "call": str, "reason": str}, ...] (行号指向真实出现处，
    含 helper 内部行)。
    """
    class_methods = class_methods or {}
    module_functions = module_functions or {}
    ext_names = ext_names or set()
    wrapper_names = wrapper_names or set()
    violations = []
    if forward_node is None:
        return violations

    visited = set()  # 防递归环 (id(FunctionDef))

    def _scan(node):
        if node is None or id(node) in visited:
            return
        visited.add(id(node))

        for child in ast.walk(node):
            # --- 检测 @ 运算符（矩阵乘法）---
            if isinstance(child, ast.BinOp) and isinstance(child.op, ast.MatMult):
                violations.append({
                    "line": child.lineno,
                    "call": "@",
                    "reason": "矩阵乘法 @ 运算符必须在 AscendC kernel 中实现",
                })
                continue

            if not isinstance(child, ast.Call):
                continue

            resolved = _resolve_call_name(child)
            if resolved is None:
                continue

            qual, attr = resolved

            # --- torch.xxx(...) ---
            if qual == "torch":
                if attr not in ALLOWED_TORCH_FUNCS:
                    violations.append({
                        "line": child.lineno,
                        "call": f"torch.{attr}",
                        "reason": f"torch.{attr} 是计算操作，必须在 AscendC kernel 中实现",
                    })
                continue

            # --- F.xxx(...) / functional.xxx(...) ---
            if qual in ("F", "functional", "torch.nn.functional", "nn.functional"):
                violations.append({
                    "line": child.lineno,
                    "call": f"{qual}.{attr}",
                    "reason": f"{qual}.{attr} 是 PyTorch 计算操作，必须在 AscendC kernel 中实现",
                })
                continue

            # --- kernel 扩展 / wrapper 调用 —— 合法，跳过 ---
            if qual in ext_names:
                continue
            if qual is None and attr in wrapper_names:
                continue

            # --- Python 内建函数 —— 允许 ---
            if qual is None and attr in ALLOWED_BUILTIN_FUNCS:
                continue

            # --- 模块级 helper 调用 → 递归展开 ---
            if qual is None and attr in module_functions:
                _scan(module_functions[attr])
                continue

            # --- tensor 方法计算操作 ---
            if attr in FORBIDDEN_TENSOR_METHODS:
                if qual not in ("torch", "F", "functional",
                                "torch.nn.functional", "nn.functional"):
                    violations.append({
                        "line": child.lineno,
                        "call": f"{qual}.{attr}()" if qual else f"{attr}()",
                        "reason": f"{attr} 是计算操作，必须在 AscendC kernel 中实现",
                    })
                continue

            # --- self.xxx(...) ---
            if qual == "self":
                if attr in ("forward",):
                    continue
                # 本类已定义方法 → 递归进入 (消除 false positive)
                if attr in class_methods:
                    _scan(class_methods[attr])
                    continue
                # 非本类方法 → 方案A: 判违规 (疑似 nn.Module 子模块属性)
                violations.append({
                    "line": child.lineno,
                    "call": f"self.{attr}(...)",
                    "reason": f"self.{attr}() 疑似 nn.Module 前向调用，核心计算必须在 AscendC kernel 中实现",
                })
                continue

    _scan(forward_node)
    return violations



def check_for_loops_over_tensors(forward_node):
    """检查 forward 中是否存在用于计算的逐元素 Python for 循环（标量写法退化信号）。

    典型退化模式：
      for n in range(N):
          for c in range(C):
              x_nc = tensor[n, c]
              result = x_nc * weight + bias  # 逐元素计算
              output[n, c] = result.sum()    # 计算归约

    以下不视为退化：
      - 数据准备循环（仅做简单赋值 / 索引映射，无计算操作）

    返回违规列表 [{"line": N, "loop_var": str, "reason": str}, ...]
    """
    violations = []
    if forward_node is None:
        return violations

    for node in ast.walk(forward_node):
        if not isinstance(node, ast.For):
            continue

        # 检查是否是 for var in range(...)
        if isinstance(node.iter, ast.Call):
            resolved = _resolve_call_name(node.iter)
            if resolved and resolved == (None, "range"):
                loop_var = ""
                if isinstance(node.target, ast.Name):
                    loop_var = node.target.id

                # 循环体必须同时满足两个条件才判定为退化：
                # 1. 存在 tensor 索引操作
                # 2. 循环体内含计算操作（禁止的 tensor 方法、torch 计算、
                #    BinOp 算术或 @ 矩阵乘法）
                has_tensor_indexing = _loop_has_tensor_indexing(node, loop_var)
                has_computation = _loop_has_computation(node)

                if has_tensor_indexing and has_computation:
                    violations.append({
                        "line": node.lineno,
                        "loop_var": loop_var,
                        "reason": (
                            f"for {loop_var} in range(...) 循环中存在 tensor 索引 + 计算操作，"
                            "这是逐元素标量写法，必须使用 AscendC kernel 的向量化操作替代"
                        ),
                    })

    return violations


def _loop_has_tensor_indexing(for_node, loop_var):
    """检查 for 循环体中是否存在使用循环变量的 tensor 索引。"""
    if not loop_var:
        return False
    for child in ast.walk(for_node):
        if isinstance(child, ast.Subscript):
            for sub_node in ast.walk(child.slice):
                if isinstance(sub_node, ast.Name) and sub_node.id == loop_var:
                    return True
    return False


def _loop_has_computation(for_node):
    """检查 for 循环体中是否包含实际的计算操作。

    计算操作包括：
    - 禁止的 tensor 方法（.sum(), .mul(), ...）
    - torch.xxx 计算调用
    - F.xxx 计算调用
    - BinOp 算术运算符（+, -, *, /, ** 等，作用于 tensor 时）
    - @ 矩阵乘法运算符
    """
    # 检查 BinOp（除了整数索引运算，tensor 之间的算术是计算信号）
    arithmetic_ops = (
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Pow, ast.Mod, ast.MatMult,
    )
    # 统计循环体内 BinOp 出现次数；少量 BinOp 可能是索引计算，大量则是退化信号
    binop_count = 0

    for child in ast.walk(for_node):
        # 检查 @ 运算符
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.MatMult):
            return True

        # 统计算术 BinOp
        if isinstance(child, ast.BinOp) and isinstance(child.op, arithmetic_ops):
            binop_count += 1

        # 检查函数调用
        if isinstance(child, ast.Call):
            resolved = _resolve_call_name(child)
            if resolved is None:
                continue
            qual, attr = resolved

            # Python 内建函数 —— 允许
            if qual is None and attr in ALLOWED_BUILTIN_FUNCS:
                continue

            # 禁止的 tensor 方法
            if attr in FORBIDDEN_TENSOR_METHODS:
                if qual not in ("torch", "F", "functional",
                                "torch.nn.functional", "nn.functional"):
                    return True

            # torch 计算调用
            if qual == "torch" and attr not in ALLOWED_TORCH_FUNCS:
                return True

            # F.xxx 计算调用
            if qual in ("F", "functional", "torch.nn.functional", "nn.functional"):
                return True

    # 阈值：5 个以上算术操作视为计算密集型循环
    # (少量 BinOp 通常是索引计算如 g+1, len(x)-2，不应触发)
    if binop_count >= 5:
        return True

    return False


# ---------------------------------------------------------------------------
# 主验证逻辑
# ---------------------------------------------------------------------------

def validate(code, filepath="<unknown>"):
    """对生成代码执行完整的退化检查。

    返回结构化结果 dict。
    """
    result = {
        "valid": False,
        "filepath": filepath,
        "checks": {
            "ascendc_ext_imported": {
                "passed": False, "extensions": [], "error": None,
            },
            "kernel_called_from_forward": {
                "passed": False, "called": [], "error": None,
            },
            "no_forbidden_torch_ops": {
                "passed": False, "violations": [], "error": None,
            },
            "no_scalar_for_loops": {
                "passed": False, "violations": [], "error": None,
            },
        },
        "regression_type": None,
        "suggestion": "",
    }

    # --- 解析 ---
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result["checks"]["ascendc_ext_imported"]["error"] = f"SyntaxError: {e}"
        result["regression_type"] = 1
        result["suggestion"] = "代码存在语法错误，无法解析。"
        return result

    # --- Check 1: AscendC 扩展导入存在性 ---
    extensions = find_ascendc_extension_imports(tree)
    ext_names = set(extensions.keys())

    result["checks"]["ascendc_ext_imported"]["extensions"] = [
        {
            "used_name": k,
            "actual_name": v["name"],
            "line": v["line"],
            "is_placeholder": v["is_placeholder"],
            "import_style": v["import_style"],
        }
        for k, v in extensions.items()
    ]

    if not ext_names:
        result["checks"]["ascendc_ext_imported"]["error"] = (
            "未找到任何 AscendC 扩展模块导入（如 import _xxx_ext）"
        )
        result["regression_type"] = 1
        result["suggestion"] = (
            "代码中没有导入 AscendC 扩展模块。model_new_ascendc.py 必须导入编译好的 "
            "AscendC kernel 扩展（如 import _xxx_ext），并在 forward() 中调用其函数完成计算。"
        )
        return result

    # 检查是否全部为占位符导入
    placeholder_exts = [k for k, v in extensions.items() if v["is_placeholder"]]
    if len(placeholder_exts) == len(extensions):
        result["checks"]["ascendc_ext_imported"]["error"] = (
            f"扩展导入使用了占位符名称 {placeholder_exts}（如 TORCH_EXTENSION_NAME），"
            "扩展模块未正确配置"
        )
        result["regression_type"] = 1
        result["suggestion"] = (
            "扩展模块导入使用了占位符名称（如 import TORCH_EXTENSION_NAME），"
            "这表示 AscendC kernel 未正确编译或配置。"
            "请确保使用 NpuExtension 编译 kernel 并使用正确的模块名导入。"
        )
        return result

    # 过滤掉占位符，只保留有效的扩展名
    valid_ext_names = {k for k, v in extensions.items() if not v["is_placeholder"]}

    result["checks"]["ascendc_ext_imported"]["passed"] = True

    # --- Check 2: forward 是否调用 kernel ---
    forward_node, class_name = find_model_forward(tree)
    if forward_node is None:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            "未找到 ModelNew.forward() 或 Model.forward() 方法"
        )
        result["regression_type"] = 2
        result["suggestion"] = "代码缺少 ModelNew（或 Model）类或 forward 方法。"
        return result

    class_methods = collect_class_methods(tree)
    wrapper_names = find_wrapper_functions(tree, valid_ext_names)
    # forward → self._helper → ext 形式: 把递归调到 ext 的本类方法并入 wrapper_names，
    # 否则 Check 2 会误判 type2 (forward 表面只调 self.method 未直接见 ext)。
    wrapper_names = wrapper_names | collect_method_wrappers(class_methods, valid_ext_names)
    called = check_kernel_calls_in_forward(
        forward_node, valid_ext_names, wrapper_names
    )
    result["checks"]["kernel_called_from_forward"]["called"] = called

    if not called:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            f"已导入扩展模块 {list(valid_ext_names)} 但 {class_name}.forward() "
            f"未调用任何扩展函数"
        )
        result["regression_type"] = 2
        result["suggestion"] = (
            f"已导入 AscendC 扩展模块 {list(valid_ext_names)} 但 "
            f"{class_name}.forward() 中未调用。"
            "forward() 必须通过 ext_module.function_name(...) 形式调用 kernel。"
            f"{'也存在 wrapper 函数 ' + str(list(wrapper_names)) + ' 但 forward 也未调用它们。' if wrapper_names else ''}"
        )
        return result

    result["checks"]["kernel_called_from_forward"]["passed"] = True

    # --- Check 3: 禁止的 torch 操作 (递归展开 forward 调用的 helper) ---
    module_functions = collect_module_functions(tree)
    violations = check_forbidden_torch_ops(
        forward_node,
        class_methods=class_methods,
        module_functions=module_functions,
        ext_names=valid_ext_names,
        wrapper_names=wrapper_names,
    )
    result["checks"]["no_forbidden_torch_ops"]["violations"] = violations

    if violations:
        result["checks"]["no_forbidden_torch_ops"]["error"] = (
            f"forward() 中发现 {len(violations)} 处禁止的 PyTorch 计算操作"
        )
        violation_details = "; ".join(
            f"第{v['line']}行 {v['call']}" for v in violations[:5]
        )
        result["regression_type"] = 3
        result["suggestion"] = (
            f"forward() 调用了 AscendC kernel 但仍使用 PyTorch 进行部分计算: "
            f"{violation_details}。"
            "所有核心计算必须在 AscendC kernel 中完成，"
            "forward() 中只允许 buffer 分配（torch.empty 等）和形状操作（.view/.reshape 等）。"
        )
        return result

    result["checks"]["no_forbidden_torch_ops"]["passed"] = True

    # --- Check 4: 标量 for 循环退化 ---
    loop_violations = check_for_loops_over_tensors(forward_node)
    result["checks"]["no_scalar_for_loops"]["violations"] = loop_violations

    if loop_violations:
        result["checks"]["no_scalar_for_loops"]["error"] = (
            f"forward() 中发现 {len(loop_violations)} 处逐元素 Python for 循环"
        )
        loop_details = "; ".join(
            f"第{v['line']}行 for {v['loop_var']} in range(...)"
            for v in loop_violations[:5]
        )
        result["regression_type"] = 4
        result["suggestion"] = (
            f"forward() 中存在逐元素 Python for 循环: {loop_details}。"
            "不能用标量逐元素写法，必须使用 AscendC kernel 的向量化 / 块级操作。"
        )
        return result

    result["checks"]["no_scalar_for_loops"]["passed"] = True

    # --- 全部通过 ---
    result["valid"] = True
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="检查 AscendC 生成代码是否退化为 PyTorch 原生实现（AST 静态分析）"
    )
    parser.add_argument("file", help="要检查的 Python 文件路径")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()
    except FileNotFoundError:
        if args.json:
            print(json.dumps({"valid": False, "error": f"文件不存在: {args.file}"}))
        else:
            print(f"[ERROR] 文件不存在: {args.file}")
        sys.exit(1)

    result = validate(code, filepath=args.file)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if result["valid"]:
            exts = result["checks"]["ascendc_ext_imported"]["extensions"]
            called = result["checks"]["kernel_called_from_forward"]["called"]
            print("[PASS] AscendC 实现验证通过")
            print(f"  - 导入 {len(exts)} 个扩展模块: "
                  f"{', '.join(e['used_name'] for e in exts)}")
            print(f"  - forward() 调用: "
                  f"{', '.join(c['call'] for c in called)}")
            print("  - forward() 中无禁止的 PyTorch 计算操作")
            print("  - forward() 中无逐元素 Python for 循环")
        else:
            rtype = result["regression_type"]
            type_desc = {
                1: "无 AscendC 扩展导入（纯 PyTorch / 占位符导入）",
                2: "有扩展导入但 forward() 未调用 kernel",
                3: "部分计算仍使用 PyTorch（需全部移入 AscendC kernel）",
                4: "存在逐元素 Python for 循环（需使用向量化操作）",
            }
            print(f"[FAIL] 检测到 PyTorch 退化 — Type {rtype}: "
                  f"{type_desc.get(rtype, '未知')}")

            for check_name, check_result in result["checks"].items():
                status = "PASS" if check_result["passed"] else "FAIL"
                print(f"  [{status}] {check_name}")
                if check_result["error"]:
                    print(f"         {check_result['error']}")

            # 显示 torch 操作违规详情
            torch_violations = result["checks"]["no_forbidden_torch_ops"]["violations"]
            if torch_violations:
                print("  torch 操作违规详情:")
                for v in torch_violations:
                    print(f"    第 {v['line']} 行: {v['call']} — {v['reason']}")

            # 显示 for 循环违规详情
            loop_violations = result["checks"]["no_scalar_for_loops"]["violations"]
            if loop_violations:
                print("  for 循环违规详情:")
                for v in loop_violations:
                    print(f"    第 {v['line']} 行: for {v['loop_var']} in range(...)"
                          f" — {v['reason']}")

            print(f"\n  修复建议: {result['suggestion']}")

    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
