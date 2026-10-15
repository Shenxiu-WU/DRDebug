# Step 1-I: Import Error Analysis（import_failed + import_kernel_side 分支）

> **读取时机**：`Step 0.3` 按当前 `failure_type == import_failed` 且 `import_subtype == import_kernel_side` 路由到本分支时，立即 Read 本文件。`import_env_side` 应已被 Step 0.3 过滤为 `skipped_env_issue`，不进入本分支。

**输入**:
- `{task_dir}/.verify_status/latest.json` — 结构化状态，确认 `import_subtype == import_kernel_side`（若是 `import_env_side` 应已被 Step 0.3 过滤）
- `{task_dir}/.verify_status/import_traceback.log` 或 `.verify_logs/phase{N}_attempt{M}.log` — 原始 import traceback
- `{task_dir}/kernel/pybind11.cpp` — pybind 注册入口（`PYBIND11_MODULE` 名、导出符号）
- `{task_dir}/kernel/*_kernel.h` / `*.cpp` — 被 pybind 引用的 kernel 符号
- `{task_dir}/model_new_ascendc.py` — import 的 ext module 名（只读！不可改）
- `{task_dir}/trace.md` — 上下文

**Agent 任务**:
1. 读 traceback，定位缺失的符号 / 模块名 / pybind 入口
2. 对照 `skills/ascendc/ascendc-translator/references/dsl2Ascendc_host.md`（pybind 章节）核对：
   - `PYBIND11_MODULE` 的第一个参数（模块名）是否与 `model_new_ascendc.py` 中 `import` 的名字一致
   - kernel ext 的 `.so` 文件命名与 import 名是否匹配
   - 导出的函数符号是否与 `pybind11.cpp` 中 `m.def(...)` 注册的名字一致
3. 写 `{task_dir}/precision_tuning/precision_audit_{attempt}.md` 含（Gate-IMPORT-A 必填）：
   - `[IMPORT_TRACEBACK_CITATION]` — 原文摘录 traceback（至少 `ImportError` / `ModuleNotFoundError` / `OSError: cannot open shared object` 的关键行）
   - `[ROOT_CAUSE]` — 根因（pybind 模块名不一致 / kernel ext 名称错 / 符号未导出）
   - `[FIX_PLAN]` — 修改点（限定 `pybind11.cpp` 的 `PYBIND11_MODULE` / `m.def` 注册行，或 kernel 侧 `extern "C"` / 导出符号名）
   - `[FIX_TYPE]` — 必须 ∈ `{pybind_symbol_fix, kernel_ext_name_fix, kernel_export_fix}`；**明确拒绝** `ld_path_fix` / `abi_fix` / `toolkit_env_fix` / `cmakelists_fix` / `setup_py_fix` / `build_ascendc_fix`（这些属于 env_side，不在本 subagent 的 scope）
4. 修改 `{task_dir}/kernel/pybind11.cpp` 或 kernel 符号导出处（**不动** `model_new_ascendc.py`、`utils/build_ascendc.py`、`CMakeLists.txt`、`setup.py`）
5. 通过 Gate-通用 + Gate-IMPORT-A 验证（命令同 Step 1-B）

**推荐参考资料**:
- `skills/ascendc/ascendc-translator/references/dsl2Ascendc_host.md`（pybind 绑定规范）
- `skills/ascendc/ascendc-debug/references/`（若有环境变量 / pybind 相关条目）
- `skills/ascendc/ascendc-translator/references/TileLang-AscendC-API-Mapping.md`（`extern "C"` 导出规范）

**Step 4（共用）**: 修复后调 `utils/verification_ascendc.py` + `utils/classify_verify_result.py` 重跑，然后走 `Gate-IMPORT-V`：
- `failure_type == success` = 本分支完成
- 仍卡在 `import` 且 traceback 未变 = 停滞
- `import` 通过但 `failure_type` 变为 `build_failed` / `runtime_error` / `precision_failed` = 进步（Gate-V 自动切换到对应分支继续 debug）
