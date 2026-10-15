# AscendC Verification

统一验证入口：

```bash
bash skills/ascendc-translator/references/evaluate_ascendc.sh <task>
```

该脚本会先调用统一构建器 `utils/build_ascendc.py` 编译 `<task>/kernel/`，再调用 `utils/verification_ascendc.py` 做 reference/candidate 对拍，不再依赖任务目录内的 `run.sh`。

### model_new_ascendc.py 编写约定

参考实现：`matmul_leakyrelu/`。

**1. sys.path 注入** — 在模块顶部添加 `kernel/build/` 以确保无论从哪个目录运行都能导入 `.so`：

```python
import sys
from pathlib import Path

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import <module_name> as _ext
```

若 `.so` 位于其他项目（如 `matmul_leakyrelu` 复用独立项目的产物），需相应调整 `_KERNEL_BUILD`。

**2. forward 签名对齐** — `ModelNew.forward` 必须与 `model.py` 的 `forward` 签名完全一致，包括哪些参数显式传入、哪些由 kernel 内部从张量形状推导：

```python
# model.py: forward(self, x, h)  ← k 由 kernel 内部从 h.shape[0] 推导
# 正确写法
def forward(self, x, h):
    return _ext.run_op(x, h)

# model.py: forward(self, x, h, k)  ← k 是显式输入
# 正确写法
def forward(self, x, h, k):
    return _ext.run_op(x, h, k)
```

**3. 模块名** — import 名称必须与 `pybind11.cpp` 中 `PYBIND11_MODULE(<name>, ...)` 一致；这个 `<name>` 就是最终生成的 Python 扩展模块名。

推荐约定：
- 任务目录名保持为 `<op_name>`
- `PYBIND11_MODULE` 模块名使用 `_<op_name>_ext`
- `model_new_ascendc.py` 中统一写成 `import _<op_name>_ext as _ext`

示例：

```cpp
// kernel/pybind11.cpp
PYBIND11_MODULE(_matmul_leakyrelu_ext, m)
```

```python
# model_new_ascendc.py
import _matmul_leakyrelu_ext as _ext
```

这样做的原因是避免扩展模块名与任务目录名同名。例如任务目录是 `matmul_leakyrelu/` 时，若扩展模块也叫 `matmul_leakyrelu`，Python 导入时可能先命中同名目录而不是 `.so` 扩展模块，导致导入冲突。使用独立扩展名后，`model_new_ascendc.py` 可以保持普通 `import`，不需要额外写 `importlib` 加载逻辑。
