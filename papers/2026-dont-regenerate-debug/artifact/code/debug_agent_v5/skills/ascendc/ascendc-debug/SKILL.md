---
name: ascendc-debug
description: >
  修复 AscendC 算子的 build / import / runtime / timeout / precision 五类失败。
  由 ascendc-debug-agent-discovery 独立调用。
  通过取证 + Agent 深度分析 + 代码修复 + 重新验证的循环实现修复。
---

## What I do

修复 AscendC 算子的 **build / import / runtime / timeout / precision** 五类失败。流程:
1. 读取 `{task_dir}/.verify_status/latest.json` 确定 `failure_type`（缺失时由 agent 的 Initialization Protocol 先产出），分流到对应 Step 1 分支（failure_type 变化时 Gate-V 自动切换分支，持续 debug）
2. Agent 结合上下文 + 代码 + 日志 / 数值证据 + 知识库做深度分析, 定位根因并制定修复计划
3. Agent 修复代码（只能改 `{task_dir}/kernel/` 下文件）
4. 重新编译 + 验证（通过 `utils/verification_ascendc.py` + `utils/classify_verify_result.py`）
5. 引擎根据 Gate 循环控制信号决定继续或停止（agent 不自决）

## When to use me

由 `ascendc-debug-agent-discovery` agent 独立调用，输入为已有算子的产物目录。
支持的 `failure_type` 白名单：`precision_failed` / `build_failed` / `import_failed(import_kernel_side)` / `runtime_error` / `timeout`。
不在白名单的任务（`success` / `degraded` / `no_kernel` / `tilelang_only_failed` / `execution_aborted` / `import_failed` 的 `import_env_side` 子类）由 Step 0.3 直接判定 `skipped_*` 后退出。

## Prerequisites（通用，所有分支共享）

算子生成 agent 产物:
- `{task_dir}/model.py` 参考实现
- `{task_dir}/model_new_ascendc.py` AscendC wrapper（未 AST 退化，反作弊前置条件）
- `{task_dir}/kernel/` 下至少一个 `.cpp` 文件（保证有 kernel 可修）
- `{task_dir}/trace.md` 人类可读记录（可选读，不强制含 `final_status` block）
- `{task_dir}/{op_name}.json`（及可选 `.json.bak`）

由 agent 的 Initialization Protocol 保证（若缺则先跑 `utils/verification_ascendc.py + classify_verify_result.py --phase 8 --attempt 0 --write-status`）:
- `{task_dir}/.verify_status/latest.json` 存在

其中 `task_dir = {repo_root}/{task_name}`，`repo_root` 为 AscendOpGenAgent 仓库根目录。

> **分支专属前提**（进入 Step 1-X 后由各分支自校验）：
> - **1-P（precision_failed）**：`model.py` 参考实现、`kernel/pybind11.cpp` 编译通过且能运行
> - **1-B（build_failed）**：`.verify_logs/phase{N}_attempt{M}.log` 含 compile error 块
> - **1-I（import_failed + import_kernel_side）**：import traceback 指向 pybind 符号 / ext module；`import_env_side` 不进入本 skill
> - **1-R（runtime_error）**：execute 阶段有明确 crash signal (SIGSEGV/SIGABRT/SIGBUS/SIGFPE)
> - **1-T（timeout）**：`.verify_status` 含 `timeout_marker_present == true`

## Workflow

**所有思考、分析、推理必须使用中文。**

**核心原则: Python 脚本做确定性操作 (取证、Gate、知识库), Agent 做需要推理的工作 (分析、修复)。**

> **职责边界（方案 C，引擎驱动）**：本 Workflow 描述的是**单个 attempt 内**「取证 → 分析 → 修复 → 验证」的分支方法论，agent 须遵循。但**循环编排（轮次推进、`MAX_ATTEMPTS`、根据 `loop_signal` 决定继续/停止、退出产物 `debug_status.json`/`debug_trace.md` 重建）归确定性引擎**，agent 不自行管理轮次计数、不读 `loop_signal` 自决、不写退出产物。下文凡涉及「继续/停止」「下一轮」的表述均为方法论说明，实际由引擎控制。

---

### Step 0: 初始化

轮次计数 `attempt` 由引擎注入（本轮值见调用上下文），agent 不自行递增。

**0.1 保存不可变基线快照（原始代码，仅首次执行）:**
```bash
if [ ! -d "{task_dir}/precision_tuning/history/baseline/code_snapshot" ]; then
    mkdir -p "{task_dir}/precision_tuning/history/baseline/code_snapshot"
    cp -r "{task_dir}/kernel/" \
       "{task_dir}/precision_tuning/history/baseline/code_snapshot/kernel/"
    cp "{task_dir}/model_new_ascendc.py" \
       "{task_dir}/precision_tuning/history/baseline/code_snapshot/model_new_ascendc.py"
    python3 skills/ascendc/ascendc-debug/scripts/anticheat.py snapshot {task_dir}
    echo "基线快照已保存，后续可从 baseline 恢复"
fi
```

> 基线快照保存在 `history/baseline/code_snapshot/`，整个调优过程中**不覆盖**。如需恢复到最初始状态，使用以下命令：
> ```bash
> cp -r "{task_dir}/precision_tuning/history/baseline/code_snapshot/kernel/" \
>    "{task_dir}/kernel/"
> cp "{task_dir}/precision_tuning/history/baseline/code_snapshot/model_new_ascendc.py" \
>    "{task_dir}/model_new_ascendc.py"
> ```

**0.2 保存本轮起始快照:**
```bash
mkdir -p "{task_dir}/precision_tuning/history/attempt_0/code_snapshot"
cp -r "{task_dir}/kernel/" \
   "{task_dir}/precision_tuning/history/attempt_0/code_snapshot/kernel/"
cp "{task_dir}/model_new_ascendc.py" \
   "{task_dir}/precision_tuning/history/attempt_0/code_snapshot/model_new_ascendc.py"
```

> 知识库将在 Sub-step 2.1 完成后通过 `search` 命令按需检索, 无需在此全量加载。

---

### Step 0.3: 读 verify_status，路由 Step 1 分支

```bash
# 读结构化 verify_status（最后一次 evaluate 的 failure_type / failed_step / log 路径）
python3 skills/ascendc/ascendc-debug/scripts/verify_status.py \
    --task-dir {task_dir} | jq '.failure_type, .import_subtype, .timeout_marker_present'
```

> 本 skill 的唯一 failure 事实源是 `{task_dir}/.verify_status/latest.json`。

**分支路由规则（每轮按当前 failure_type 动态路由）**：

- 每轮入口按 `verify_status.latest.json.failure_type` 查表（见下方路由表），进入对应 Step 1-X 分支
- 首轮调用任何 Gate 时，`session_branch.json` 自动记录起始 failure_type（仅供 `debug_status.json` 的 `entry_failure_type` / `session_branch` 字段使用，不限制后续路由）
- `import_failed` 还要读 `verify_status.latest.json.import_subtype`：
  - `import_kernel_side` → 进入 Step 1-I
  - `import_env_side` → 环境库 / LD_LIBRARY_PATH 问题，本 skill 不处理；引擎直接判 `session_outcome: skipped_env_issue` 并由 Step 7 重建退出产物（`debug_trace.md` + `debug_status.json`，agent 不手写）

**Step 1 分支路由表（每轮按当前 failure_type 查表，CONTINUE 后同样适用）**：

| session_branch | failure_type | 进入 |
|---|---|---|
| `1-P` | `precision_failed` | Step 1-P（现有精度取证路径） |
| `1-B` | `build_failed` | Step 1-B |
| `1-I` | `import_failed` + `import_kernel_side` | Step 1-I |
| `1-R` | `runtime_error` | Step 1-R |
| `1-T` | `timeout` | Step 1-T |
| — | 其他（`success` / `degraded` / `no_kernel` / `tilelang_only_failed` / `execution_aborted`） | 引擎判 `session_outcome: skipped_unsupported_type`，由 Step 7 重建退出产物（`debug_trace.md` + `debug_status.json`，agent 不手写），退出 |

---

### Step 1-P: 精度取证（precision_failed 分支）

#### 1.1 精度取证 (Python 脚本, 不可跳过)

```bash
python3 skills/ascendc/ascendc-debug/scripts/precision_forensics.py \
    {task_name} --attempt {attempt}
```

Gate 验证:
```bash
python3 skills/ascendc/ascendc-debug/scripts/precision_gate.py \
    --step forensics --op-name {op_name} --task-name {task_name} --attempt {attempt}
```

⛔ **Gate-F 未通过 → 停止, 检查错误输出。不要在没有取证数据的情况下分析代码。**
如果报错含 `FileNotFoundError`，先确认 `{task_dir}/kernel/pybind11.cpp` 存在，再检查 `utils/verification_ascendc.py` 路径。

> 1-P 分支继续走 Step 2（精度深度分析 6 Sub-step）→ Step 3（修复）→ Step 4（重编译+验证，走 Gate-V 的精度语义）。本轮到此结束，后续循环/退出由引擎控制。

---

### Step 1-B: Build Error Analysis（build_failed 分支）

> 详细分析步骤见 `skills/ascendc/ascendc-debug/references/branch-build.md`，
> 进入此分支时必须 **Read 该文件**后再开始分析。

---

### Step 1-I: Import Error Analysis（import_failed + import_kernel_side 分支）

> 详细分析步骤见 `skills/ascendc/ascendc-debug/references/branch-import.md`，
> 进入此分支时必须 **Read 该文件**后再开始分析。

---

### Step 1-R: Runtime Error Analysis（runtime_error 分支）

> 详细分析步骤见 `skills/ascendc/ascendc-debug/references/branch-runtime.md`，
> 进入此分支时必须 **Read 该文件**后再开始分析。

---

### Step 1-T: Timeout Analysis（timeout 分支）

> 详细分析步骤见 `skills/ascendc/ascendc-debug/references/branch-timeout.md`，
> 进入此分支时必须 **Read 该文件**后再开始分析。

---

### Step 2: 深度分析 + 修复计划（仅精度分支 1-P，Agent 推理, 核心步骤）

**本步骤分为 6 个 Sub-step, 每个 Sub-step 有明确的输入文件和产出 section, 不可跳过或合并。**

将全部分析结果写入 `{task_dir}/precision_tuning/precision_audit_{attempt}.md`。

**历史扫描（attempt > 0 时必须执行，首轮跳过）：**

**扫描 A：读方向学习表（一次读完，直接获得跨轮全貌）**
```bash
cat "{task_dir}/precision_tuning/tuning_directions.json"
```

从 `tuning_directions.json` 一次性获得：
- 每轮的 `fix_type`（哪些修复类型已被尝试）
- `outcome`（passed / improved / stagnant / regressed）— 快速判断方向是否有效
- `improvement_ratio`（数值趋势一览）
- `direction_verdict`（是否曾切换方向）
- `forensics_hint`（每轮取证信号）
- `final_status`（in_progress / success / failed）

> ⚠️ **禁止重复已证实无效的方向**：outcome 为 regressed 或连续 stagnant 的 fix_type，本轮不得再用。

**扫描 B：按需深入（仅在确实需要时通过 round_summary 的 index 定位）**
```bash
# 读某轮的 round_summary 获取文件路径索引
cat "{task_dir}/precision_tuning/round_summary_{N}.json"
# 再按 index.sections.* 路径读对应的 section 小文件
```

- 想了解某轮根因细节 → 读 `round_summary_N.index.sections.root_cause` 指向的文件
- 想了解某轮修复计划 → 读 `round_summary_N.index.sections.fix_plan` 指向的文件
- 想查看完整审计 → 读 `round_summary_N.index.audit_full` 指向的文件

**禁止**：不得跳过 `tuning_directions.json` 直接全量读 `history/attempt_*/precision_audit.md`。

---

#### Sub-step 2.1: 取证数据解读

**读取**: `{task_dir}/precision_tuning/forensics_report_{attempt}.json`

**可选前置读取（仅 attempt == 0 且文件存在）**: `{task_dir}/trace.md`

> trace.md 由 `ascend-kernel-developer` 在生成阶段产出，记录 Phase 4 AscendC 转译的迭代历史、走偏点、已知平台/API 限制、kernel 结构意图。读取它可以**避免重蹈生成阶段已走偏的方向**，并补全 kernel 设计背景。文件不存在时跳过，Gate-A 不强制。

**产出**: `[FORENSICS_SUMMARY]` section + `[PRIOR_TRACE_CONTEXT]`（可选，仅首轮 + trace.md 存在时）

逐字段摘录取证报告中的关键数值, 不允许跳过任何字段:

```
=== PRECISION AUDIT REPORT ===

[FORENSICS_SUMMARY]
  取证数据摘要 (L0-L4):
    - primary_hint: <来自取证 primary_hint>
    - primary_confidence: <来自取证 primary_confidence>
    - primary_evidence: <来自取证 primary_evidence>
    - mismatch_ratio: <来自取证 outputs[0].basic_stats.mismatch_ratio>
    - max_abs_diff: <来自取证 outputs[0].basic_stats.max_abs_diff>
    - mean_abs_diff: <来自取证 outputs[0].basic_stats.mean_abs_diff>
    - error_distribution: <来自取证 outputs[0].error_distribution, 特别关注 sign_analysis.bias_direction>
    - worst 元素位置: <来自取证 outputs[0].worst_elements, 列出 top 3>
    - 首错下标初步反推（若 tileLength 可从 pybind11.cpp 读取时填写）:
      - worst element 线性下标: <outputs[0].worst_elements[0].index 线性值>
      - 对应 tile 编号: <线性下标 ÷ tileLength（整除）>
      - 周期性初判: <错误间隔是否等于 tileLength（偏移问题）/ 向量宽度（计算问题）/ 核边界（多核问题）>
    - 尾块分析: <来自取证 outputs[0].tail_analysis, 标注各 tile_size 下的 tail/body mismatch rate>
    - 维度分析: <来自取证 outputs[0].dimension_analysis, 标注各维度的 mismatch_rate 范围>
  L6 内存布局:
    - 输入 tensor layout: <来自取证 L6_memory_layout.inputs, 标注 shape/stride/对齐>
    - 最后一维对齐情况: <是否对齐 8/16/32/64/128/256>
  L8 算子类型:
    - op_type: <来自取证 L8_operator.op_type>
    - source: <来自取证 L8_operator.source>
    - attributes: <来自取证 L8_operator.attributes, 特别关注 dim/reduction/kernel_size 等>
    - reduction_axis: <来自取证 L8_operator.reduction_axis, 如果有>
    - 该类型的 checklist: <将在下方 search 命令中自动返回>
  可用文件:
    - reference: <来自取证 available_files.reference>
    - custom: <来自取证 available_files.custom>
  dtype 精度级别判断:
    - dtype: <来自取证 outputs[0] 或 L8_operator, 如 float32/float16/bfloat16>
    - max_abs_diff (来自取证): <值>
    - 精度阈值参考 (与 utils/verification_ascendc.py 实际使用的阈值对齐):
      * float32: atol=1e-4, rtol=1e-4 → max_diff > 1e-4 为逻辑错误; ≤ 1e-4 为精度达标
      * float16: atol=1e-2, rtol=1e-3 → max_diff > 1e-2 为逻辑错误; 1e-3~1e-2 为 float16 精度损失(可能可接受); ≤ 1e-3 为精度达标
      * bfloat16: atol=5e-2, rtol=5e-3 → max_diff > 5e-2 为逻辑错误; 5e-3~5e-2 为 bfloat16 精度损失; ≤ 5e-3 为精度达标
      * int8: atol=1.5, rtol=0.0 → 整数类型使用绝对误差判断
    - 判断: <逻辑错误(实现缺陷, 必须修复) / float16精度损失(检查 float32 下是否通过) / 精度达标>
    - 对分析方向的影响: <逻辑错误→重点查实现缺陷; float16精度损失→检查归约是否需要 upcast>
  我对取证 hint 的初步判断:
    - 取证给出的 hint 是否合理? <结合数值证据判断, 不要在此步做代码分析>
    - 是否有数值异常未被 hint 覆盖? <如 sign_analysis 显示偏向但 hint 未提及>
  多 case 聚合（来自 outputs[0].case_aggregate + 顶层字段，必须填写）:
    - num_test_cases: <来自取证 num_test_cases，即 NPUKernelBench 该算子的 case 总数>
    - pass_case_count / fail_case_count: <来自 outputs[0].case_aggregate>
    - mismatch_ratio 跨 case 范围: min=<> / max=<> / mean=<>
    - all_cases_same_pattern: <true/false>
      → true: 所有 case 根因一致，Sub-step 2.5 用 representative_case_idx 单组实验即可
      → false: 不同 case 失败原因不同，Sub-step 2.5 需按 dtype 分组实验（见 2.5 多 case 前置判断）
    - shape_conditional: <true/false>
      → true: mismatch 与 last_dim 显著相关（相关系数>0.7），实验 D 优先，提示 tail/tiling 类问题
    - representative_case_idx: <来自 outputs[0].representative_case_idx，即 mismatch_ratio 最高的 case 编号>
    - representative case 的 input_dtype / shape: <从 outputs[0].per_case 中找 case_idx == representative_case_idx 的条目，读取其 input_dtype 字段；shape 从 L6_memory_layout.inputs[0].shape 读取>
```

**可选段（仅当 attempt == 0 且 `{task_dir}/trace.md` 存在时写入）**:

```
[PRIOR_TRACE_CONTEXT]
  来源: {task_dir}/trace.md (ascend-kernel-developer 生成阶段产出)
  最终结果: <如 "SKIP (tilelang) | FAIL (ascendc)" 或 "PASS">
  Phase 4 AscendC 迭代次数: <evaluate_ascendc.sh 执行次数>

  已尝试方向（本轮修复时避免重复）:
    - 第 N 轮: <一句话总结做了什么、结果如何>
    - ...

  走偏点记录（trace.md "走偏点" 章节原文提炼）:
    - <如 "把 device kernel 写成模板入口导致 host stub 找不到实际符号">
    - <如 "Muls 在 bfloat16 下不支持, 当前平台 API 限制">

  剩余未解决的平台/API 限制:
    - <trace.md 揭示的硬性约束, 如 "当前平台 Muls 不支持 __bf16">

  kernel 结构要点（若 trace.md 提及）:
    - <如 "分 fp32/fp16/bf16 三个独立入口">
```

> 如 trace.md 不存在, 省略此 section, 不影响 Gate-A。写入时只摘录关键点, **不要**粘贴 trace.md 全文或长代码块。

**知识库检索 (第一次 — 基于取证 hint + 算子类型):**

> 若 engine prompt 注入 `ABLATE_KB=1` 或
> `ABLATE_DIAGNOSTIC_EVIDENCE=1`，跳过本节及后续第二次检索，且不得手工读取 KB。

从 `[FORENSICS_SUMMARY]` 中提取 `primary_hint` 和 `op_type`, 检索相关知识条目:
```bash
python3 skills/ascendc/ascendc-debug/scripts/precision_knowledge.py search \
    --kb-path skills/ascendc/ascendc-debug/references/precision_knowledge_base.json \
    --op-type <L8_operator.op_type> \
    --pattern <primary_hint> \
    --op-name <算子名, 如 023_HyenaFftSizePaddingRfft> \
    --top-k 3 \
    --log-path "{task_dir}/precision_tuning" \
    --attempt {attempt} \
    --call-index 0
```

记住检索到的 `matched_entries` 和 `checklists`, 后续分析时参考。
如果输出 `fallback_to_full_load: true`, 说明无精确匹配, 已返回全量条目。
`--op-name` 提供关键词软匹配通道: 当 op_type=unknown 或 pattern=all_wrong (无区分度)
时, 算子名分词 (如 rfft/fft) 仍能召回算子专项条目, 是检索的关键兜底信号。

**取证 hint 快速跳转表**（按 `primary_hint` 确定 Sub-step 2.3 重点方向，以及 Sub-step 2.5 对应实验）：

| `primary_hint` | 疑似根因 | Sub-step 2.3 重点检查 | Sub-step 2.5 优先实验 |
|---|---|---|---|
| `tail_spike` | TailTile / TailPadding | `curTileLength` vs `tileLength` 在向量 API 的用法；尾块 Duplicate 初始化 | 实验 D-baseline（确认算法基础）→ D-boundary（定位触发 last_dim 边界） |
| `uniform_offset` | FP16Upcast 或 GMOffset | Cast 升精度路径；GM 偏移公式（元素 vs 字节） | 实验 C（全1/arange 输入） |
| `nan_inf_contamination` | 除零 / Exp 溢出 / Log 负数 | Div/Reciprocal 防零；Ln 正数约束；Exp 范围钳制 | 实验 C + Sub-step 2.6 插桩 |
| `scattered` | 未知（低置信度）| 全面检查 REFERENCE_IMPL_SPEC 5 个维度 | 实验 A→B→C→D 全流程 |
| `boundary_concentration` | MulticoreTiling | 核间 GM 区间覆盖公式；formerNum × formerLength 完整性 | 实验 A（单核隔离） |
| `magnitude_correlated` | FP16Upcast 或 ReduceOverwrite | Cast 路径；ReduceSum/Max 后是否复用 src（参见知识库 ReduceOverwrite 条目） | 实验 C（全1输入） |
| `dimension_concentration` | Layout / GMOffset | DataCopy stride 计算；多维 offset 公式 | 实验 A（多核场景）或实验 C |
| `all_wrong` | 多种可能（优先级：GM 偏移 > Cast > TQue 流程）| TBuf→GM 路径；CopyOut 是否真正执行 | 实验 C（全1输入） |

> ⚠️ 映射表仅用于**优先化分析方向**，不替代 Sub-step 2.3 的全面检查。置信度 LOW 时即使 hint 明确，也要执行 Sub-step 2.5 对应实验。

---

#### Sub-step 2.2: 计算链分解 + 参考规范（同步完成）

> 读取前必须 **Read** `skills/ascendc/ascendc-debug/references/phase-a-checklist.md`（获取 `[REFERENCE_IMPL_SPEC]` 和 `[KERNEL_STEP_TRACE]` 的完整格式模板，Phase C 使用）。

**【跨轮复用检查 — attempt > 0 时先执行；attempt == 0 直接跳到"全量读取"】**

> `[REFERENCE_IMPL_SPEC]` 和 `[COMPUTATION_DECOMPOSITION]` 均来自不随 kernel 修改而变化的静态文档（model.py / AscendC 规范 / archive case），每轮可安全复用。

**步骤 0：查询 round_summary_0 的 spec 和 decomposition 索引路径**
```bash
python3 -c "
import json, os, sys
p = '{task_dir}/precision_tuning/round_summary_0.json'
if not os.path.exists(p):
    print('NO_SUMMARY'); sys.exit(0)
d = json.load(open(p))
s = d.get('index', {}).get('sections', {})
spec = s.get('reference_impl_spec')
decomp = s.get('computation_decomposition')
print(spec if spec else 'NO_SPEC')
print(decomp if decomp else 'NO_DECOMP')
"
```

根据输出选择路径：

| 输出 | 含义 | 处理方式 |
|------|------|---------|
| 两条路径均有效（不以 `NO_` 开头） | 首轮两个 section 均已索引 | → 执行**复用步骤** |
| 任一为 `NO_SUMMARY` / `NO_SPEC` / `NO_DECOMP` | 首轮未完成或 section 缺失 | → **回退**，执行完整"全量读取" |

**复用步骤（两条路径均有效时）：**
```bash
[ -f "{task_dir}/<spec路径>" ] && [ -f "{task_dir}/<decomp路径>" ] && echo "EXISTS" || echo "FILE_MISSING"
```

| 输出 | 处理方式 |
|------|---------|
| `EXISTS` | ① Read 两个文件 → ② 将内容**原样**写入本轮 `precision_audit_{attempt}.md` → **直接跳到 Sub-step 2.3 Phase B** |
| `FILE_MISSING` | 文件已丢失 → **回退**，执行完整"全量读取" |

---

**全量读取**（attempt == 0，或上方复用检查回退时执行）:

1. **必须读取**: `{task_dir}/model.py` — 参考实现的 forward() 逻辑

2. 根据 `L8_operator.op_type` 从 `archive_tasks/` 路由，读取对应案例的 `kernel/` 目录（仅含有完整 kernel/ 的案例）：
   - pooling → `archive_tasks/avg_pool3_d/kernel/`
   - normalization / rmsnorm / layernorm → `archive_tasks/rms_norm/kernel/`（含 vector_tile.h）
   - matmul / gemm / linear → `archive_tasks/matmul_leakyrelu/kernel/` 或 `archive_tasks/quant_matmul/kernel/`
   - gather / scatter / index → `archive_tasks/gather_elements_v2/kernel/`
   - attention / softmax → `archive_tasks/flash_attention/`（有 TileLang 设计 model_new_tilelang.py，无 AscendC kernel/；读取设计理解计算结构，AscendC 约束依赖 dsl2Ascendc_compute_cv.md）
   - 纯 elementwise / padding / activation → `archive_tasks/circular_pad/kernel/`
   - 无精确匹配 → 选最近似案例，在 `[REFERENCE_IMPL_SPEC]` 中标注"参考案例非精确匹配"

3. **必须读取**: `skills/ascendc/ascendc-translator/references/dsl2Ascendc.md`
   （含禁用 API 模式和常见错误）

4. **必须读取**: `skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_vector.md`
   （含 DataCopyPad 触发条件和非对齐处理）

5. **必须读取**: `skills/ascendc/ascendc-translator/references/TileLang-AscendC-API-Mapping.md`
   （AscendC API 权威参考）
   API 详细文档：`skills/ascendc/ascendc-translator/references/AscendC_knowledge/api_reference/`

**产出**: `[COMPUTATION_DECOMPOSITION]` + `[REFERENCE_IMPL_SPEC]` sections（同步产出，按 `phase-a-checklist.md` 模板填写）

**要求**:

`[COMPUTATION_DECOMPOSITION]` — 参考 `decomposition_examples/` 中最匹配示例，每步必须包含：操作名、输入来源、输出 shape、数值范围预期、**精度风险点（结合 AscendC API 约束标注，如 ReduceMax 需 Duplicate 初始化、DataCopy 需 32 字节对齐等）**；标注算子计算模式：单行归约 / 跨核归约 / 分块累加 / 滑窗累加 / 前缀累加 / 逐元素。

`[REFERENCE_IMPL_SPEC]` — 必须覆盖：TQue/TBuf 规范、关键 API 规范（含 ReduceMax/ReduceSum 初始化）、非对齐处理规范、禁用模式；无精确匹配案例时标注"参考案例非精确匹配"。

```
[COMPUTATION_DECOMPOSITION]
  算子: {op_name}
  计算模式: <单行归约 / 跨核归约 / 分块累加 / 滑窗累加 / 前缀累加 / 逐元素>
  参考分解示例: <使用的示例文件名, 如 softmax.md, 或 "无匹配, 自行分解">
  归约维度: <dim={dim}, axis={axis}, 归约轴长度={length}> (如适用)
  数据类型: <dtype>

  计算链:
    Step 0: 输入
      - shape: <input_shape>
      - 数值范围: <来自取证 value_range>

    Step 1: <operation_name>
      - 来源: reference.py 中的 <具体函数/表达式>
      - 输入: <上一步输出 / 原始输入>
      - 输出 shape: <shape>
      - 数值范围预期: <基于输入范围推断>
      - 精度风险点: <该步可能引入误差的原因，结合 AscendC API 约束标注>
      - 知识库关联: <匹配的条目编号和标题, 或 "无">

    Step N: 最终输出
      - 与取证报告的 golden output 统计对照

  跨核通信: (仅跨核归约模式)
    - workspace buffer: <是否存在, 大小>
    - Phase 1 → Phase 2 的同步机制: <描述>
```

---

#### Sub-step 2.3: AscendC 实现逐步对照

> `[REFERENCE_IMPL_SPEC]` 和 `[COMPUTATION_DECOMPOSITION]` 已由 Sub-step 2.2 产出；`[KERNEL_STEP_TRACE]` 格式模板已在 Sub-step 2.2 开始时 Read（`phase-a-checklist.md`）。直接从 Phase B 开始。

**Phase C 执行中途按需查阅（无需重走 Sub-step 2.2 全量读取）：**

| 触发情形 | 应 Read 的文档 |
|----------|--------------|
| spec 未覆盖的 API（如 `Vmax`、`Sub`、负无穷常量写法） | `skills/ascendc/ascendc-translator/references/TileLang-AscendC-API-Mapping.md` |
| 对齐阈值不明确（32-byte 触发条件细节） | `skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_vector.md` |
| 出现 spec 未列举的禁用模式 | `skills/ascendc/ascendc-translator/references/dsl2Ascendc.md` |
| archive case 与当前 op_type 关联性存疑 | `archive_tasks/<最近似案例>/kernel/`（路由表见 Sub-step 2.2 全量读取第 2 条） |

---

**Phase B: 读取当前实现**

**Phase B 读取** (全部在 `{task_dir}/kernel/` 下):
1. `{op_name}_tiling.h` — TilingData 结构体定义
2. `*_kernel.h` — 所有 kernel 类定义（可能有多个）
3. `*.cpp`（排除 pybind11.cpp）— 所有 kernel entry 文件
4. `kernel_common.h`、`vector_tile.h`、`matmul_tile.h`（若存在）— helper 逻辑
5. `pybind11.cpp` — host tiling 计算、workspace 分配、launch 逻辑

注意：AscendOpGenAgent 中 host 逻辑（TilingFunc）在 pybind11.cpp 内，不是单独的 op_host.cpp。

**产出**: `[KERNEL_STEP_TRACE]` section（经 Phase B+ 实测数据补充后填写）

---

**Phase B+: 插桩探针（读代码后，比对前）**

在对全部 kernel 文件有了完整认知（Phase B 已读完）、但还未开始结构化比对之前，对 representative case 跑一次带中间值输出的探针，使 Phase C 的逐步比对能同时呈现"规范期望"与"代码实测"。

**跳过条件（4 条全满足才可跳过，缺任一必须执行）**：
1. **首轮**快路径：`attempt == 0` 默认**跳过**探针（例外：`primary_hint == nan_inf_contamination`
   时首轮即须执行）；`attempt > 0` 时本条不成立，须结合下列 2~4 条判定
2. `primary_hint` 与上一轮相同（问题性质未变）
3. 上一轮 `[INSTRUMENTATION_FINDINGS]` 已存在且覆盖了 Phase B 识别出的疑似阶段
4. Phase B 阅读中未发现上一轮未覆盖的新疑似路径

**跳过时**：必须写入 `[L5_PROBE]` section 并注明跳过理由（Gate-A 强制 section 存在，内容可以是理由说明）。

**执行步骤（自包含，共 6 步）**：

1. 若 `{task_dir}/debug_{op_name}_precision.py` 不存在，复制 debug_precision_template.py 并替换占位符，将 `CASE_INDEX` 设为 `representative_case_idx`（来自 `[FORENSICS_SUMMARY]` 多 case 聚合字段）
2. 在 Compute() 函数的 3 个阶段各插一个 printf 探针点：
   - P1：CopyIn 后紧跟 `inQueue.DeQue()` 之后（DeQue 后立即读取，违反 R2 导致 UB 脏数据）
   - P2：主计算逻辑中间点（如归约结束后、主乘法后）
   - P3：CopyOut 前紧跟计算写入完成、`outQueue.EnQue()` 之前
   - 必须遵守 5 条核心规则：R1（GetBlockIdx()==0 过滤）、R2（DeQue 后读取）、R3（half/bf16 转 float 再 printf）、R4（阶段标记字符串）、R5（DumpTensor 只 dump 首 16~32 个元素）
3. 重编译
4. 运行 `debug_{op_name}_precision.py` Section 1（仅运行 representative case 原始复现），从 stdout 提取 printf 中间值
5. 将每阶段实测值写入 `[L5_PROBE]` section（见下方格式）
6. 从 `precision_tuning/history/attempt_{attempt}/code_snapshot/kernel/` 恢复 kernel 文件（清除 printf），重编译（确保 binary 与源码一致；Sub-step 2.5 C/D 实验使用此 clean binary，无需再次重编译）

**`[L5_PROBE]` 产出格式**：
```
[L5_PROBE]
状态: 已执行 / 跳过（理由: <满足4条跳过条件的具体举证>）
探针阶段:
  P1 (CopyIn 后, DeQue 后读取): <实测值，如 x[0]=0.5000 x[1]=0.7031 len=128 / N/A>
  P2 (计算中点): <实测值 / N/A>
  P3 (CopyOut 前, EnQue 前读取): <实测值 / N/A>
异常首现: <P1前 / P1-P2间 / P2-P3间 / P3后，或"暂无明确异常">
```

---

**Phase C: 结构化对照**

**Phase C 要求 (结构化对照)**:
- 将 Kernel 的 Compute() 函数拆成与 Sub-step 2.2 对应的步骤
- 每步标注: AscendC API 名称、count 参数值、buffer 来源、代码行号、**L5_PROBE 实测中间值**
- 逐步与 2.2 的计算链对齐, 用 ✅/⚠️/❌ 标注匹配状态
- **对照 `[REFERENCE_IMPL_SPEC]` 逐项检查以下 5 个维度**:
  1. TQue/TBuf 数据流是否与规范一致 (特别: TBuf 是否绕过 outQueue 直接 DataCopy)
  2. ReduceMax/ReduceSum work_buf 是否按规范初始化 (Duplicate 到 -INF 或 0)
  3. DataCopy 对齐是否满足规范 (count × sizeof(dtype) 不是 32 倍数时是否换用 DataCopyPad)
  4. SyncAll 同步点是否与规范一致 (跨核场景是否遗漏)
  5. dsl2Ascendc.md 中列出的禁用模式是否在代码中出现
- 遇到不确定的 API 名称时，查阅 `TileLang-AscendC-API-Mapping.md` 确认（如 Max vs Vmax、Subs 是否存在、负无穷常量写法等）

> `[KERNEL_STEP_TRACE]` 的完整格式模板在 `references/phase-a-checklist.md` 中（Sub-step 2.2 开始时已 Read），直接使用其格式填写，每个 K-Step 补充 L5_PROBE 实测中间值字段。

---

#### Sub-step 2.4: 知识库匹配 + 根因判断 + 修复计划

**读取**: Sub-step 2.1 检索到的知识库条目 + Sub-step 2.1~2.3 的全部分析结果

**知识库检索 (第二次 — 精化, 增加位置特征):**

> ⚠️ **在开始匹配前，用完整取证数据做第二次精化检索**（避免长上下文遗忘, 并利用 2.1~2.3 分析中发现的位置特征）。

从取证报告中提取 `--position` 参数:
- 若 worst_elements 集中在尾部区域 或 tail_analysis 显示尾块 mismatch 率显著偏高 → `--position tail`
- 若 worst_elements 集中在边界/起始区域 → `--position boundary`
- 若 worst_elements 分散 → `--position scattered`
- 若无明显位置特征 → 不传 `--position`

```bash
python3 skills/ascendc/ascendc-debug/scripts/precision_knowledge.py search \
    --kb-path skills/ascendc/ascendc-debug/references/precision_knowledge_base.json \
    --op-type <L8_operator.op_type> \
    --pattern <primary_hint> \
    --position <tail/boundary/scattered 或不传> \
    --op-name <算子名, 如 023_HyenaFftSizePaddingRfft> \
    --top-k 3 \
    --log-path "{task_dir}/precision_tuning" \
    --attempt {attempt} \
    --call-index 1
```

记住检索到的条目, 用于下方的 `[KNOWLEDGE_MATCH]`。

**产出**: `[KNOWLEDGE_MATCH]` + `[ROOT_CAUSE]` + `[CAUSAL_CHAIN_ANALYSIS]` + `[FIX_PLAN]` + `[TARGET_FILES]` + `[DIRECTION_ASSESSMENT]` sections

**要求**: 根因判断必须基于 2.1~2.3 的具体发现, 不允许"凭直觉"给出根因。证据链中必须引用具体的 K-Step 编号和取证数据字段。

> ⚠️ **若 Sub-step 2.1 产出了 `[PRIOR_TRACE_CONTEXT]`**，`[FIX_PLAN]` 的修复方向**不得**与其"已尝试方向"里的失败路径重复，也**不得**违反"剩余未解决的平台/API 限制"（如 trace.md 明确记录"当前平台 Muls 不支持 __bf16"，则不得在修复中使用 `Muls` 处理 bf16 dtype）。在 `[ROOT_CAUSE].证据链` 里显式引用 trace 中对应的走偏点或平台限制条目。

> ⚠️ **写 [FIX_PLAN] 前必须查阅 `TileLang-AscendC-API-Mapping.md`，核实所有将要使用的 AscendC API 名称**：
> - 逐元素向量最大值：`Max`（不是 `Vmax`，该 API 不存在）
> - 逐元素减法：`Sub`（无 `Subs`），逐元素除法：`Div`（无 `Divs`）
> - float32 负无穷：`-3.402823466e+38f` 或 `(float)(-INFINITY)`（不是 `AscendC::INFINITY`，该常量不存在）
> - DataCopy 写 GM 必须从 VECOUT TQue DeQue 后的 tensor，不能直接用 TBuf.Get() 的结果

```
[KNOWLEDGE_MATCH]
  知识库匹配:
    - 匹配的知识条目: <title> / 无匹配
    - 匹配度: 完全匹配 / 部分匹配 / 不匹配
    - 如何借鉴: <参考知识条目的 fix 内容>
  算子类型 checklist 检查:
    - <checklist 项 1>: 通过 / 未通过 / 不适用 (证据: <引用 K-Step 或取证数据>)
    - <checklist 项 2>: ...

[ROOT_CAUSE]
  根因判断: <综合 2.1 取证数据 + 2.3 步骤对齐结论 + 知识库匹配>
  置信度: HIGH / MEDIUM / LOW
  证据链:
    1. 数值证据: <取证 L1-L4 中哪些现象支持此判断, 引用具体字段值>
    2. 布局证据: <L6 内存布局是否有异常>
    3. 代码证据: <引用 K-Step 编号, 哪行代码有什么问题>
    4. 分解对照: <2.2 的哪个 Step 与 2.3 的哪个 K-Step 不一致>
    5. 逻辑推导: <为什么此代码问题会产生取证中观察到的 diff 模式>

[CAUSAL_CHAIN_ANALYSIS]
  第1步 - 哪些输出量错了:
    - 主输出: <名称, mismatch_ratio, max_abs_diff（来自 FORENSICS_SUMMARY）>
    - 辅助输出（如 scales, workspace）: <名称, mismatch_ratio 或 "无"，来自 forensics per_output>
  第2步 - 哪些中间量也错了:
    - L5_PROBE P1 (CopyIn 后): <正常 / 异常，引用实测值>
    - L5_PROBE P2 (计算中点): <正常 / 异常，引用实测值>
    - L5_PROBE P3 (CopyOut 前): <正常 / 异常，引用实测值>
    - 错误首现区间: <P1前 / P1-P2间 / P2-P3间 / P3后>
  第3步 - 计算链排除法:
    - 已排除（产出正常的阶段）: <列出 K-Step 编号>
    - 最小疑似路径: <缩小后的 K-Step 范围>
  第4步 - 特异性 vs 通用性:
    - L6: 最后一维对齐情况是否影响此路径 <是/否，引用具体值>
    - L8: 是否依赖特定属性分支（如 activate_left=True）<是/否，引用属性>
    - 结论: case 特有问题 / 通用实现缺陷

[FIX_PLAN]
  修复方向: <具体描述, 引用变量名和行号>
  修复类型: <对应知识库 type, 如 FIX_PRECISION_TAIL>
  修改文件: <file1, file2>
  修改点:
    1. 文件: <文件名>, 位置: <行号或函数名>, 操作: <修改/新增/删除>
       当前代码: <现在是什么>
       修改为: <改成什么>
       对应 K-Step: <编号>
    2. ...
  预期效果: <修复后应该改善什么, 如 "尾块 mismatch 应消除">

[TARGET_FILES]
  <需要修改的文件列表, 逗号分隔>

[DIRECTION_ASSESSMENT]
  上一轮 (attempt={attempt-1}) 修复方向: <从 history/attempt_{{attempt-1}}/precision_audit.md 的 [FIX_PLAN] 中提取, 一句话描述>
  上一轮修复后 mismatch 变化: <从 forensics_report_{attempt}.json 的 history_trend 中读取, 如 "0.25→0.12 (改善)" 或 "0.25→0.28 (恶化)">
  本轮是否延续上一轮方向: <严格填写 "是" 或 "否"，不得填写其他任何文字>
  延续理由 / 换方向理由: <一句话>

=== END AUDIT ===
```

**重试轮次注意事项 (attempt > 0):**
- 取证报告中的 `history_trend` 显示了历史变化
- 你**必须**先读 `tuning_directions.json` 获取跨轮方向全貌，再按需通过 `round_summary_N.json` 的 index 路径深入具体 section 小文件
- **禁止重复 outcome 为 regressed 或连续 stagnant 的 fix_type**
- 如某轮 `index.sections.root_cause` 为 null，fallback 读 `index.audit_full`

---

### Sub-step 2.5: 实验隔离（以"运行"为默认，跳过需要明确举证）

**触发规则**（优先运行实验，跳过有严格门槛）：

以下 4 条**同时满足**时方可跳过实验：
1. `[ROOT_CAUSE]` 置信度 = HIGH
2. 证据链三元组完整：①数值证据（引用 forensics 具体字段值）②代码证据（引用具体行号+变量名）③逻辑推导（解释为何该代码问题产生观察到的 diff 模式）
3. `attempt > 0` 且 `tuning_directions.json` 中上一轮 `improvement_ratio > 0.2`
4. 当前 `primary_hint` 与上一轮一致（问题性质未变）

**条件 3 对 attempt == 0 永不成立，因此首轮必须执行实验，无论代码分析多"有把握"。**

缺任意一条 → 必须执行实验。

#### 多 case 前置判断（在生成调试脚本前执行，基于 [FORENSICS_SUMMARY] 中的多 case 聚合数据）

查阅 `[FORENSICS_SUMMARY]` 末尾的"多 case 聚合"字段，按下表决定实验组织策略：

| all_cases_same_pattern | shape_conditional | 实验策略 |
|---|---|---|
| true | any | **单组实验**：用 `representative_case_idx` 运行 C+D-baseline+D-boundary，结论适用所有 case |
| false | true | **按 dtype 分组，优先 D-boundary**：float32 / float16 / bfloat16 各取一个失败 case，D-baseline 必须先跑确认算法基础，再测 last_dim 对齐 vs 非对齐 |
| false | false | **按 dtype 分组，优先实验 C**：各取一个失败 case 运行全1对照，在 `[EXPERIMENT_RESULTS]` 中分组汇报结论 |

**分组边界约束**：最多选 3 组（每种 dtype 各一个代表 case），不对所有 case 逐一运行。**直接读取 `outputs[0].case_aggregate.dtype_representative_cases`**——该字段已按 input_dtype 预计算各分组的最高 mismatch_ratio 代表 case 编号，格式如 `{"torch.float16": 12, "torch.float32": 5, "torch.bfloat16": 20}`，无需手动遍历 per_case。

#### 实验分层（按执行成本）

| 实验 | 需要重编译 | 默认执行条件 |
|------|-----------|------------|
| **C：固定/规律输入** | ❌ 否 | **每次必做**（通过调试脚本，成本极低） |
| **D-baseline：1D 退化** | ❌ 否 | **每次必做**（消除多维复杂性，验证算法在最简 1D 场景的正确性） |
| **D-boundary：last_dim 边界** | ❌ 否 | **每次必做**（D-baseline 通过后定位 tail-tile 触发的 last_dim 边界） |
| **A：block_dim → 1** | ✅ 是 | `primary_hint` ∈ {`boundary_concentration`, `scattered`}，或 C/D 结论不明确 |
| **B：PipeBarrier\<PIPE_ALL\>** | ✅ 是 | `primary_hint` = `scattered`，或实验 A 单核通过后追加 |

#### 实验执行前：生成调试脚本

```bash
# 1. 复制模板到 task_dir（替换三个占位符）
cp skills/ascendc/ascendc-debug/scripts/debug_precision_template.py \
   {task_dir}/debug_{op_name}_precision.py
# 将文件内 {{OP_NAME}} → {op_name}，{{TASK_DIR}} → {task_name}，
# {{CASE_INDEX}} → {representative_case_idx}（来自 [FORENSICS_SUMMARY] 多 case 聚合字段）
# 分组实验时，从 outputs[0].case_aggregate.dtype_representative_cases 读取各 dtype 代表编号，
# 每组分别替换 CASE_INDEX 为对应代表 case 编号后重新运行

# 2. 运行（本地/SSH/Docker 自动检测）
bash skills/ascendc/ascendc-debug/references/run_precision_debug.sh {task_name} {op_name}
```

> 实验 C/D 通过修改调试脚本的输入直接运行，无需重编译。
> 实验 A/B 需先修改 `kernel/` 文件再重编译，用 `bash skills/ascendc/ascendc-debug/references/run_precision_debug.sh` 运行。

#### 实验 A：block_dim → 1（多核隔离）

在 `{task_dir}/kernel/pybind11.cpp` TilingFunc 中临时硬编码 `blockDim = 1`，重编译后运行调试脚本。

| 结果 | 结论 | 下一步 |
|------|------|--------|
| 单核过、多核挂 | 核间问题（GM 区间重叠 / tiling 映射 / 核间同步） | 查 pybind11.cpp tiling 公式；加载 `bug_examples/multicore-tiling-overlap.md` |
| 单核也挂 | 非多核问题 | 实验 B |

#### 实验 B：PipeBarrier\<PIPE_ALL\>（同步隔离）

将 kernel Process 中所有阶段之间临时插入 `AscendC::PipeBarrier<PIPE_ALL>()`（CopyIn/Compute/CopyOut 之间各加一个），重编译后运行。

| 结果 | 结论 | 下一步 |
|------|------|--------|
| 全屏障后过 | 核内同步不足 → 逐步恢复细粒度同步定位 | 查 EnQue/DeQue 配对；加载 `bug_examples/async-sync-missing.md` |
| 仍失败 | 非同步问题 | 实验 C |

> ⚠️ `PIPE_ALL` 仅用于实验，**绝不可作为最终修复方案**。

#### 实验 C：固定/规律输入（地址隔离）

分别用全 1、等差序列（arange）、随机输入测试（调试脚本自动完成）。

| 结果 | 结论 | 下一步 |
|------|------|--------|
| 全 1 过、等差/随机挂 | 地址/偏移/stride 错误（常数输入掩盖偏移问题） | 查 GM 偏移公式；加载 `bug_examples/gm-offset-error.md` |
| 全都挂 | 计算逻辑或全局 tiling 错误 | 查 Cast 路径或 TQue 流程；加载 `bug_examples/fp16-no-upcast.md` |
| 全都过 | 特定数值范围触发精度问题 | 查边界值/极值（Exp 溢出、Ln 负数等） |

#### 实验 D：shape 缩减隔离（含两个子实验，顺序执行）

**D-baseline：1D 退化（算法基础验证）**

将输入压缩为 1D 小张量（size=32 和 64），消除所有多维 layout 复杂性，验证算法在最简单 tile 场景下是否正确。

| 结果 | 结论 | 下一步 |
|------|------|--------|
| 1D 也失败 | 算法本身有误，与 shape/维度/tiling 无关 | 查计算逻辑根因（Cast 路径 / TQue 流程 / 公式错误） |
| 1D 通过 | 算法基础正确，问题由多维 layout 或 tail-tile 引入 | 进入 D-boundary |

**D-boundary：last_dim 边界（tail-tile 触发定位）**

保留所有维度不变，仅修改 `last_dim` 为 tile 整数倍（对齐）和 tile-1（非对齐）。调试脚本自动生成候选序列（基于常见 tile_size 32/64/128/256），无需手动构造。

示例：原始 shape `[128, 257]` → 脚本自动测试 `[128, 256]`（tile=32/64/128/256 的最大对齐值）和 `[128, 255]`（非对齐）。若 orig_last 较大（如 1025），则各 tile_size 可产生不同的对齐候选（1024, 1023, 896, 895 等）。

若 `shape_conditional=true`，**必须**读取 `outputs[0].tail_analysis`，找出 `tail_mismatch_rate` 最高的 `tile_size`（如 96），确认该 tile_size 在调试脚本自动生成的 candidates 中已覆盖（脚本只枚举 32/64/128/256）。若该 tile_size 不在枚举列表中，在调试脚本的 candidates 列表中手动追加 `aligned = ((orig_last - 1) // tile_size) * tile_size`（严格小于 orig_last 的最大对齐值）和 `aligned - 1`（非对齐值）。

| 结果 | 结论 | 下一步 |
|------|------|--------|
| 对齐 last_dim 通过、非对齐挂 | 尾 tile 处理错误 | 查 `curTileLength` vs `tileLength`；加载 `bug_examples/tail-tile-misalign.md` |
| 小 last_dim 通过、大挂 | 多核/tiling 边界 | 配合实验 A 确认 |
| 全部失败 | 与 last_dim 无关（但 D-baseline 已排除算法错误） | 查多维 layout（DataCopy stride / GMOffset 多维 offset） |

#### 首错下标 + tiling 反推

```
首错线性下标 first_idx（来自调试脚本输出或 forensics worst_elements）
  → 对应 tile 编号  = first_idx ÷ tileLength（整除）
  → 对应核编号     = tile编号 ÷ （每核 tile 数）
  → 该核 GM 起始偏移 = 核编号 × formerLength
  → 预期搬运字节数   = curTileLength × sizeof(T)

错误周期判断：
  周期 = tileLength       → 搬运/偏移问题（优先实验 C）
  周期 = 向量操作宽度(8/16/32) → 计算流程问题（参见 [L5_PROBE]，不足时启用 Sub-step 2.6）
  与核边界对齐            → 多核 tiling 问题（优先实验 A）
```

#### 典型案例按需加载

仅在误差特征匹配时加载对应案例，**不要一次性全部加载**：

| 实验结论 / 误差现象 | 案例文件 |
|---|---|
| FP16 挂 FP32 过，全部偏差 | [`references/bug_examples/fp16-no-upcast.md`](references/bug_examples/fp16-no-upcast.md) |
| 实验 C：全1过随机挂；周期 = tileLength | [`references/bug_examples/gm-offset-error.md`](references/bug_examples/gm-offset-error.md) |
| 实验 D-boundary：非整除 last_dim 挂、D-baseline 通过，仅尾部错 | [`references/bug_examples/tail-tile-misalign.md`](references/bug_examples/tail-tile-misalign.md) |
| 实验 A：单核过多核挂 | [`references/bug_examples/multicore-tiling-overlap.md`](references/bug_examples/multicore-tiling-overlap.md) |
| 实验 B：PIPE_ALL 后稳定；多次运行结果不同 | [`references/bug_examples/async-sync-missing.md`](references/bug_examples/async-sync-missing.md) |

#### [EXPERIMENT_RESULTS] 产出规范

将实验结论写入 `{task_dir}/precision_tuning/precision_audit_{attempt}.md` 的 `[EXPERIMENT_RESULTS]` section，并据此更新 `[ROOT_CAUSE]` 的置信度与证据链：

```
[EXPERIMENT_RESULTS]
  执行的实验: <C / C+D-baseline+D-boundary / A+C+D / 全流程 / 跳过（说明理由）>

  实验 C 结论: <全1过+随机挂=地址问题 / 全挂=计算问题 / 全过=数值范围触发>
  实验 D-baseline 结论: <1D-32/64 通过=算法基础正确，问题由多维/tiling引入 / 失败=算法本身有误（与shape无关）>
  实验 D-boundary 结论: <对齐过非对齐挂=尾tile / 小last_dim过大挂=多核边界 / 全失败=与last_dim无关（查多维layout）>
  实验 A 结论: <如执行: 单核过多核挂=核间 / 单核也挂=非多核>
  实验 B 结论: <如执行: 屏障后过=同步不足 / 仍失败=非同步>

  首错反推: first_idx=<N> → tile=<M> → 核=<K>，与实验结论<一致/矛盾>

  实验后置信度更新: <LOW→MEDIUM / MEDIUM→HIGH / 仍 LOW>
  更新后疑似根因: <与初始 [ROOT_CAUSE] 一致 / 修正为 ...>

  （若跳过实验，逐条核对跳过条件）
  跳过条件核查:
    ① 置信度 HIGH: <是/否>
    ② 证据三元组完整: <是/否，缺少哪项>
    ③ attempt>0 且上轮 improvement_ratio>0.2: <是/否>
    ④ primary_hint 与上轮一致: <是/否>
    → 全部满足: 跳过有效 / 存在不满足项: 不应跳过
```

⚠️ **[EXPERIMENT_RESULTS] section 的存在由 Gate-A 强制校验**——无论执行还是跳过，此 section 必须写入 precision_audit_{attempt}.md。

**反模式（NEVER）**：
- NEVER 将 `PipeBarrier<PIPE_ALL>` 保留在最终修复代码中
- NEVER 同时改变多个变量（如同时单核 + 固定输入）——无法定位原因
- NEVER 跳过调试脚本生成直接靠肉眼判断实验结果

Gate 验证:
```bash
python3 skills/ascendc/ascendc-debug/scripts/precision_gate.py \
    --step audit --op-name {op_name} --task-name {task_name} --attempt {attempt}
```

⛔ **Gate-A 未通过 → 补全缺失的 section，不计入轮次。Gate-A 现在检查 10 个必填 section:
FORENSICS_SUMMARY, COMPUTATION_DECOMPOSITION, REFERENCE_IMPL_SPEC, KERNEL_STEP_TRACE,
L5_PROBE, ROOT_CAUSE, CAUSAL_CHAIN_ANALYSIS, FIX_PLAN, TARGET_FILES, EXPERIMENT_RESULTS。
注意：EXPERIMENT_RESULTS 若跳过实验，需写入跳过理由（满足 4 条跳过条件的举证），同样视为通过。L5_PROBE 若跳过探针，需写入跳过理由（满足 4 条跳过条件的举证），同样视为通过。**

> Gate-A 通过后，脚本自动提取 sections 小文件并写入 `round_summary_{attempt}.json` 初始字段（diagnostics + index）。**Agent 无需手动写 round_summary。**

---

### Sub-step 2.6: 插桩定位（首轮默认跳过；attempt ≥ 1 默认执行，4 条全满足才可跳过）

#### 首轮快路径（attempt == 0）：默认**跳过**插桩

首轮走静态诊断快路径，**不做插桩定位**，仅凭 forensics/code 静态证据 + Phase A/B 分析给出
修复。用"首轮修复是否失败"作为"困难算子"的天然信号——简单算子首轮即中，无谓的插桩开销
被省掉；只有首轮没修好、进入 attempt ≥ 1 的算子才值得插桩深挖。
`[L5_PROBE]` section 仍须产出，写跳过版：「状态: 跳过（理由: 首轮快路径，未插桩）」，
不触发 Gate-A 缺失。

例外（首轮即须插桩，满足任一即执行）：
- `primary_hint == nan_inf_contamination`（溢出传播类，静态证据不足以定位首现点，必须插桩）。

#### attempt ≥ 1（首轮已失败）：默认执行，4 条**同时满足**才可跳过

1. Phase B+ 已产出 `[L5_PROBE]`（非"跳过理由"版本，是实测数值版本）
2. `[CAUSAL_CHAIN_ANALYSIS]` 的"错误首现区间"已精确定位到单个 K-Step
3. `[ROOT_CAUSE]` 置信度 = HIGH 且修复位置已精确到代码行号
4. `primary_hint ≠ nan_inf_contamination`（溢出传播类问题必须用插桩追踪，不可跳过）

**attempt ≥ 1 且 Phase B+ 未执行时：跳过条件 1 永不成立，必须执行。**

**职责边界**：Sub-step 2.6 是 Phase B+ 的补充精化，不是替代。Phase B+ 做三阶段粗粒度探针（P1/P2/P3）；Sub-step 2.6 在 Phase B+ 已定位的可疑阶段内做 API 级二分搜索，精确到单行。

---

#### 核心规则（5 条，违反任一则插桩结果不可信）

| # | 规则 | 说明 |
|---|---|---|
| R1 | **仅在 GetBlockIdx()==0 的核打印** | 多核并发 printf 输出乱序，只看 core-0 输出 |
| R2 | **在 DeQue 之后立即读取** | DeQue 前 UB 内容未定义，读取是 UB 脏数据 |
| R3 | **FP16/BF16 转 float 后再 printf** | 直接 printf half 结果未定义；先 `(float)val` |
| R4 | **添加阶段标记字符串** | `printf("[phase=CopyIn tile=%d] val=%.6f\n", i, v)` — 区分阶段和循环轮次 |
| R5 | **DumpTensor 用小 dumpSize** | 只 dump 首 16~32 个元素，避免输出淹没日志 |

---

#### 工具选择表

| 场景 | 推荐工具 | 理由 |
|---|---|---|
| 标量/单元素验证（循环下标、偏移量、长度变量） | `printf` | 轻量，不影响 TBuf 布局 |
| 向量中间结果（UB tensor 前 N 个元素） | `DumpTensor` | 直接输出 LocalTensor 内容，无需手写循环 |
| NaN/Inf 传播追踪 | `printf` + 每步 `IsNan`/`IsInf` 判断 | 确认哪一步产生 NaN |
| GM 偏移验证 | `printf` 打印 `blockIdx * tileLength` 和 `progress * tileLength` | 验证地址计算公式 |

---

#### 插桩策略（逐步缩小）

**原则**：二分法——先在计算中点插桩，根据中点结果决定向前还是向后缩进，直到定位到单个 API。

```cpp
// 步骤1: 在 Compute() 入口确认输入正确（DeQue 后立即）
__aicore__ inline void Compute(int32_t progress) {
    auto xLocal = inQueueX_.DeQue<T>();
    // [插桩-P1] 验证输入
    if (GetBlockIdx() == 0 && progress == 0) {
        printf("[P1-input tile=0] x[0]=%.6f x[1]=%.6f len=%d\n",
               (float)xLocal.GetValue(0), (float)xLocal.GetValue(1), tileLength_);
    }

    // ... 中间计算 ...

    // [插桩-P2] 在某中间结果后验证
    if (GetBlockIdx() == 0 && progress == 0) {
        // 例：确认 Mul 结果
        printf("[P2-after-Mul] out[0]=%.6f\n", (float)tmpBuf_.GetValue(0));
    }

    // [插桩-P3] 在 EnQue 前验证最终输出
    if (GetBlockIdx() == 0 && progress == 0) {
        printf("[P3-before-EnQue] result[0]=%.6f\n", (float)outLocal.GetValue(0));
    }
    outQueueY_.EnQue(outLocal);
}
```

**二分缩进流程**：
1. P1 值错 → 问题在 CopyIn（GM 偏移 / DMA 长度），退出 Compute 检查
2. P1 正确、P2 错 → 问题在 P1→P2 之间的 API（缩进该区间）
3. P2 正确、P3 错 → 问题在 P2→P3 之间的 API（缩进该区间）
4. P3 正确但输出错 → 问题在 CopyOut（GM 偏移 / 长度），检查 CopyOut

**NaN 追踪专用模式**（`primary_hint = nan_inf_contamination`）：
```cpp
// 在每个关键 API 后插入 NaN 检测
auto v0 = outLocal.GetValue(0);
if (GetBlockIdx() == 0 && (v0 != v0 || v0 > 1e30f || v0 < -1e30f)) {
    printf("[NaN-detected after <API_NAME> tile=%d] val=%.6e\n", progress, (float)v0);
}
```

---

#### [INSTRUMENTATION_FINDINGS] 产出规范

将插桩结论写入 `{task_dir}/precision_tuning/precision_audit_{attempt}.md` 的 `[INSTRUMENTATION_FINDINGS]` section：

```
[INSTRUMENTATION_FINDINGS]
插桩阶段: <P1/P2/P3/...>
首次异常出现位置: <API 调用名称 + Compute() 第几行>
异常值: <printf 输出的具体数值>
预期值: <对应的 reference 值或理论值>
结论: <缩小后的根因假设，更新 ROOT_CAUSE 置信度>
```

> Gate-A 不强制 [INSTRUMENTATION_FINDINGS]，但若已执行插桩，必须写入此 section。写入后 `_write_audit_index()` 会自动索引到 `history/attempt_N/sections/instrumentation_findings.txt`。

---

**反模式（NEVER）**：
- NEVER 在多核（blockDim>1）下不加 `GetBlockIdx()==0` 过滤就打印——输出乱序导致结论错误
- NEVER 在 DeQue 之前读取 LocalTensor——UB 内容未定义
- NEVER 同时插多于 3 个插桩点——信号过多反而难定位，用二分法逐步缩进
- NEVER 将插桩代码保留在提交的修复版本中——插桩完成后必须清除所有 printf/DumpTensor

---

### Step 3: 代码修复 (Agent 执行)

> **非精度分支（1-B/I/R/T）注意**：代码修复已在 `branch-*.md` 的步骤 4 中完成，Step 3 的 Gate-Fix 仍需运行（验证 audit 文件完整性），但不需要再次修改代码。

根据审计报告 [FIX_PLAN] 中的修改点, 逐一修复代码。

**修复原则:**
1. **严格遵循 FIX_PLAN**: 不要自行扩大修改范围
2. **完整文件**: 写入修改后的完整文件, 不要截断
3. **真实变量名**: 使用代码中实际存在的变量名
4. **禁止逃避**: 不得缩小 shape、添加 if 跳过、放大 tolerance、删除功能

修复完成后, Gate 验证代码文件完整性（`--step fix` 等价于 Gate-A，验证 audit 文件结构完整）:
```bash
python3 skills/ascendc/ascendc-debug/scripts/precision_gate.py \
    --step fix --op-name {op_name} --task-name {task_name} --attempt {attempt}
```

⛔ **Gate 未通过 → 检查 audit 文件 [FIX_PLAN] 等必填 section 是否完整写入。**

---

### Step 4: 重新编译 + 精度验证

```bash
STDOUT="{task_dir}/.verify_logs/phase8_attempt{attempt}.stdout"
STDERR="{task_dir}/.verify_logs/phase8_attempt{attempt}.stderr"
mkdir -p "{task_dir}/.verify_logs"
python3 utils/verification_ascendc.py {task_dir} >"$STDOUT" 2>"$STDERR"; rc=$?
python3 utils/classify_verify_result.py --exit-code $rc --stdout-path "$STDOUT" --stderr-path "$STDERR" --task-dir {task_dir} --phase 8 --attempt {attempt} --write-status
```

> 产出 `{task_dir}/.verify_status/phase8_attempt{attempt}.json`（结构化 failure 数据，所有分支共用）。

**失败分类处理**（根据 stdout 判断）：

| 失败类型 | 特征 | 处理 |
|---|---|---|
| Infra 失败 | SSH 超时、docker exec 失败 | 停止，报告环境问题，不进入修复循环 |
| Build 失败 | `build_ascendc.py` 报编译错误 | 修复 kernel .cpp/.h，最多 3 次 |
| Import 失败 | import 阶段报 ModuleNotFoundError 或 PYBIND11_MODULE 名不一致 | 检查 model_new_ascendc.py import 名 vs pybind11.cpp |
| Numerical 失败 | verification_ascendc.py 报 mismatch | 进入 precision_forensics → 审计 → 修复循环 |

**每次编译失败后，更新 `{task_dir}/precision_tuning/compilation_log_{attempt}.json`**（追加 entry）：
```json
{
  "attempt": <N>,
  "entries": [
    {
      "compile_retry": <0/1/2>,
      "error_category": "<undefined_api|type_mismatch|count_alignment|other_compile>",
      "error_snippet": "<编译器报错核心行，最多3行>",
      "fix_applied": "<本次修复简述>"
    }
  ]
}
```

**验证结果产物**（方案 C 下由引擎 `validate_runner._write_validation_result` 产出，agent 无需手写；`{task_dir}/precision_tuning/validation_result_attempt_{attempt}.json`）：
```json
{
  "attempt": <N>,
  "correctness_passed": true/false,
  "match_rate": "<如 87.50 或 100.00>",
  "max_diff": "<如 1.23e-04>",
  "first_error_lines": ["<stdout 首个错误起最多30行>"],
  "stdout_path": "<.verify_logs/phase8_attemptN.stdout 全文路径指针>",
  "stderr_path": "<.verify_logs/phase8_attemptN.stderr 全文路径指针>"
}
```
> 修复 4a：不再内嵌 `evaluate_stdout`/`evaluate_stderr` 全文（数千行），改紧凑摘要 + 路径指针；需全文时按指针读盘。`correctness_passed`/`match_rate` 是 Gate-V 消费字段，必留。

提取规则（`verification_ascendc.py` 输出格式）：
- `match_rate`: 用正则 `r"mismatch_ratio=([0-9.]+)%"` 取所有 case 平均，转换为 match_rate = 100 - avg_mismatch；若无 mismatch 行则写 `100.0`
- `max_diff`: 用正则 `r"max_abs_diff=([0-9.eE+\-g]+)"`；若无 mismatch 行则写 `0.0`

**Gate 验证:**
```bash
python3 skills/ascendc/ascendc-debug/scripts/precision_gate.py \
    --step validate --op-name {op_name} --task-name {task_name} --attempt {attempt}
```

> 注意：Gate-V 只校验”当前 `{op_name}.json`”对应的验证结果。若任务目录还存在 `{op_name}.json.bak`，则这通常意味着当前 `.json` 是精简用例，**还不能直接宣布最终成功**；全量用例验证由引擎在后续轮次驱动，agent 本轮无需自行触发。

---

## Note

- **每步 Gate 验证不可跳过** — Gate 是流程稳定性的保证
- **loop_signal 由 Gate 脚本决定, Agent 必须遵守** — 防止钻牛角尖
- **取证数据是分析的基础** — 不要在没有取证的情况下分析代码
- **知识库条目只在精度通过时写入** — 避免失败经验污染知识库
- **编译失败不计入精度调优轮次** — 编译问题就地修复 (最多 3 次)
- 修复后代码直接写回 AscendC 项目目录 (覆盖原文件)
- 参考 `references/precision_knowledge_base.json` 中的已知精度问题模式
