# Step 1-T: Timeout Analysis（timeout 分支）

> **读取时机**：`Step 0.3` 按当前 `failure_type == timeout` 路由到本分支时，立即 Read 本文件。进入条件：`timeout_marker_present == true`，否则视为 `execution_aborted`，不应进本分支。

**输入**:
- `{task_dir}/.verify_status/latest.json` — 结构化状态；必须满足 `failure_type == timeout` 且 `timeout_marker_present == true`（否则视为 `execution_aborted`，不应进本分支）
- `{task_dir}/.verify_logs/phase{N}_attempt{M}.log` — 超时前的 stdout/stderr 尾部（最后一条日志提示死锁 / 死循环位置）
- `{task_dir}/kernel/*.cpp` / `*.h` — kernel 源码（重点看 `SyncAll` / `WaitFlag` / `SetFlag` / `for` 循环边界）
- `{task_dir}/kernel/{op_name}_tiling.h` + `kernel/pybind11.cpp` — tiling 配置（block_dim / tile_size）
- `{task_dir}/trace.md` — 上下文

**Agent 任务**:
1. 读 log 尾部，定位超时时 kernel 执行到哪一步（若能判断）；结合 `duration_sec` 与预期耗时量级判断是死锁（duration ≈ timeout 阈值且无输出推进）还是性能降级
2. 对照 `skills/ascendc/ascendc-translator/references/dsl2Ascendc_cross_core_sync.md` 分析：
   - `SyncAll` 是否遗漏或多余（多余的 SyncAll 在部分核未到达时死锁）
   - `SetFlag` / `WaitFlag` 配对是否一致
   - `CrossCoreSetFlag` / `CrossCoreWaitFlag` 配对是否一致（AIC↔AIV 跨核点对点同步，配对错误是死锁高频根因）
   - `gmWorkspace` 是否在 host 侧初始化为 0（软同步依赖初始值，未清零导致行为未定义）
   - `block_dim` 是否超过实际参与计算的核数（超出时框架插入异常同步，kernel 挂死）
   - Tiling 参数（`tile_size`、归约轴切分）是否导致循环不收敛
3. 写 `{task_dir}/precision_tuning/precision_audit_{attempt}.md` 含（Gate-TIMEOUT-A 必填）：
   - `[SYNC_POINT_ANALYSIS]` — 枚举 kernel 中所有同步点（`SyncAll` / `SetFlag` / `WaitFlag`）及其配对关系，标出疑似死锁点
   - `[ROOT_CAUSE]` — 根因（同步缺失 / 同步多余 / 死循环 / tiling 死锁）
   - `[FIX_PLAN]` — 文件 / 函数 / 行号级修改列表
4. 修改 `{task_dir}/kernel/*.cpp` / `*.h`（**不动** tiling host 逻辑若超出 `kernel/pybind11.cpp` 的 TilingFunc）
5. 通过 Gate-通用 + Gate-TIMEOUT-A 验证

**推荐参考资料**:
- `skills/ascendc/ascendc-translator/references/dsl2Ascendc_cross_core_sync.md`（同步原语与死锁反模式）
- `skills/ascendc/ascendc-translator/references/dsl2Ascendc_host.md`（tiling / workspace 分配）
- `skills/ascendc/ascendc-translator/references/AscendCVerification.md`（runtime 约束）

**Step 4（共用）**: 修复后调 `utils/verification_ascendc.py` + `utils/classify_verify_result.py` 重跑，然后走 `Gate-TIMEOUT-V`：
- `verify_status.duration_sec < timeout_threshold` 且 `failure_type != timeout` = 本分支完成（无论对错 — 精度对错由后续判断，但 timeout 语义已解除）
- 仍超时且 duration 基本不变 = 停滞
- 不再超时但转 `runtime_error` / `precision_failed` = 进步（Gate-V 自动切换到对应分支继续 debug）
