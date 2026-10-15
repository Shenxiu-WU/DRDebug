# Step 1-B: Build Error Analysis（build_failed 分支）

> **读取时机**：`Step 0.3` 按当前 `failure_type == build_failed` 路由到本分支时，立即 Read 本文件。

**输入**:
- `{task_dir}/.verify_status/latest.json` — 结构化状态 + `log_path` + `compile.error_summary`
- `{task_dir}/.verify_logs/phase{N}_attempt{M}.log` — 原始 build log（compile 阶段 stderr 全文）
- `{task_dir}/kernel/*.cpp` / `*.h` — 当前 kernel 源码
- `{task_dir}/trace.md` — Phase 1-7 上下文（Phase 4 ac_iterations 里记录了历次 build 尝试）

**Agent 任务**:
1. 读 build log，提取 compile error / fatal error / undefined reference / template instantiation error 块（每块最多 10 行，按 stderr 顺序）
2. 对照 `skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_*.md`、`dsl2Ascendc_host.md`、`TileLang-AscendC-API-Mapping.md` 找 API 用法差异（签名、模板参数、include 依赖）
3. 写 `{task_dir}/precision_tuning/precision_audit_{attempt}.md` 含（Gate-BUILD-A 必填）：
   - `[COMPILE_ERROR_CITATION]` — 原文摘录 error 块 + 对应 `kernel/*.cpp` 行号（引用不少于 1 处 error）
   - `[ROOT_CAUSE]` — 根因（签名不匹配 / 模板参数错 / include 缺失 / pipe-queue 协议违反 等）
   - `[FIX_PLAN]` — 文件 / 函数 / 行号级修改列表
   - `[FIX_TYPE]` — 必须 ∈ `{api_usage_fix, template_arg_fix, include_fix, signature_align_fix, pipe_queue_fix, tilingdata_field_fix}`；不在白名单的类型 Gate-A 直接 reject
4. 修改 `{task_dir}/kernel/*.cpp` / `*.h`（**绝对不动** `utils/build_ascendc.py` / `CMakeLists.txt` / `setup.py` / `utils/` 下任何文件）
5. 通过 Gate-通用 + Gate-BUILD-A 验证：
   ```bash
   python3 skills/ascendc/ascendc-debug/scripts/precision_gate.py \
       --step audit --op-name {op_name} --task-name {task_name} --attempt {attempt}
   ```

**推荐参考资料**:
- `skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_vector.md`（向量 API）
- `skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_scalar.md`（标量 API）
- `skills/ascendc/ascendc-translator/references/dsl2Ascendc_host.md`（host 侧 tiling / workspace）
- `skills/ascendc/ascendc-translator/references/TileLang-AscendC-API-Mapping.md`（API 权威参考）
- `skills/ascendc/ascendc-translator/references/AscendC_knowledge/api_reference/`（API 详细文档）

**Step 4（共用）**: 修复后调 `utils/verification_ascendc.py` + `utils/classify_verify_result.py` 重跑，然后走 `Gate-BUILD-V`：
- `verify_status.failed_step` 从 `compile` 推进到 `import`/`execute`/`verify`/`null` = 进步（Gate-V 自动切换到新 failure_type 对应分支继续 debug）
- 仍卡在 `compile` 且 error 行未变 = 停滞
- `failure_type == success` = 成功，进入 Step 5
