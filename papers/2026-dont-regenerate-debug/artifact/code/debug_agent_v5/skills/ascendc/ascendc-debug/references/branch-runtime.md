# Step 1-R: Runtime Error Analysis（runtime_error 分支）

> **读取时机**：`Step 0.3` 按当前 `failure_type == runtime_error` 路由到本分支时，立即 Read 本文件。

**输入**:
- `{task_dir}/.verify_status/latest.json` — 结构化状态 + `execute.crash_signal`（SIGSEGV / SIGABRT / SIGBUS / SIGFPE）
- `{task_dir}/.verify_logs/phase{N}_attempt{M}.log` — stderr / stack trace / core dump 信息
- `{task_dir}/kernel/*.cpp` / `*.h` — kernel 源码
- `{task_dir}/trace.md` — 上下文

**Agent 任务**:
1. 读 stderr / stack trace，提取 crash 位置（函数名 / 行号 / 同步点）；若 log 中只有 signal 编号没有 stack trace，结合 `crash_signal` 类型定位可能原因：
   - `SIGSEGV` → 越界访存（UB / GM 访存越界、Tensor 未分配就读取、TQue 协议违反）
   - `SIGABRT` → assertion 失败 / 运行时检查失败（AscendC runtime 内部 check）
   - `SIGBUS` → 内存对齐错（未满足 32/64/128 字节对齐）
   - `SIGFPE` → 除零 / 浮点异常（tiling 参数为 0、分母未保护）
2. 对照 `skills/ascendc/ascendc-translator/references/dsl2Ascendc_cross_core_sync.md`、`AscendCVerification.md`、`dsl2Ascendc_compute_*.md` 找 API 约束 / 同步点 / 对齐要求
3. 写 `{task_dir}/precision_tuning/precision_audit_{attempt}.md` 含（Gate-RUNTIME-A 必填）：
   - `[RUNTIME_ERROR_CITATION]` — 原文摘录 stderr / stack trace（含 crash_signal、函数名、行号）
   - `[ROOT_CAUSE]` — 根因（越界 / 对齐 / 同步缺失 / TQue 协议违反 / 除零）
   - `[FIX_PLAN]` — 文件 / 函数 / 行号级修改列表
4. 修改 `{task_dir}/kernel/*.cpp` / `*.h`
5. 通过 Gate-通用 + Gate-RUNTIME-A 验证

**推荐参考资料**:
- `skills/ascendc/ascendc-translator/references/dsl2Ascendc_cross_core_sync.md`（跨核同步 / SyncAll）
- `skills/ascendc/ascendc-translator/references/AscendCVerification.md`（runtime 语义与验证）
- `skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_vector.md`（对齐与 DataCopyPad）

**Step 4（共用）**: 修复后调 `utils/verification_ascendc.py` + `utils/classify_verify_result.py` 重跑，然后走 `Gate-RUNTIME-V`：
- `verify_status.failure_type != runtime_error` 或 crash 位置 / signal 变化 = 进步（若仍是 runtime_error 但位置变则视为 `progressed`）
- `failure_type` 变为 `precision_failed` = 进步（Gate-V 自动切换到对应分支继续 debug）
