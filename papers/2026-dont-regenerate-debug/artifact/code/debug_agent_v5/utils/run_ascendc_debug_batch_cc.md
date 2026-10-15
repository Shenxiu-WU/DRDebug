# run_ascendc_debug_batch_cc.sh 批量 Debug 脚本

Claude Code 版 AscendC 算子 debug 批量调度脚本。跨多个 Docker 容器 + 多 NPU 并行执行 debug agent，自动修复主流程（ascend-kernel-developer）产出的失败算子。

## 运行环境

- **操作系统**：openEuler（Docker 容器内）
- **硬件**：Ascend 910B NPU
- **工具链**：Claude Code CLI（`claude` 命令）
- **脚本位置**：宿主机或容器内均可执行，通过 `docker exec` 将任务分发到容器

## 核心逻辑

```
┌──────────────────────────────────────────────────────────────┐
│  输入：task_dir 列表（方式 A 直接传入 / 方式 B 从文件读取）      │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  初始化：写入共享队列 + 创建报告文件                             │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  并行 Worker（每容器绑定一张 NPU）                               │
│                                                              │
│  while 队列不空:                                              │
│    1. flock 原子出队一个 task_dir                              │
│    2. claude --bare -p --agent ascendc-debug-agent-discovery  │
│    3. 若 pause_turn → --resume 恢复（最多 MAX_RESUMES 次）     │
│    4. 停滞检测（文件 mtime + 活跃进程监控）                      │
│    5. 读 debug_status.json 判定结果                            │
│    6. 若 progressed_to_new_failure_type → 新 session 重入      │
│    7. 反作弊检测（anticheat.py verify）                         │
│    8. 写报告行                                                 │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  输出：batch_report.md（逐算子状态 + 汇总统计）                  │
└──────────────────────────────────────────────────────────────┘
```

### 关键机制说明

**动态工作队列**：容器间并行（每容器绑定一张 NPU），容器内串行。谁先完成当前任务，就从共享队列拉下一个，不做预分配。通过 `flock` 文件锁保证原子出队。

**Session 管理**：每个 task_dir 分配一个 UUID session_id。Claude Code 的 `--session-id`（首次）和 `--resume`（续接）支持多轮会话。当 Claude 因 turn 限制暂停（`pause_turn`）时自动 resume 继续。

**跨分支重入**：debug agent 可能将 build_failed 修复后转变为 precision_failed（`progressed_to_new_failure_type`）。脚本检测到此状态后，自动创建新 session 继续修复新的 failure type，直到 attempt 配额耗尽。

**停滞检测**：失败后持续监控 task_dir 下关键文件的 mtime 和活跃子进程。若超过阈值（默认 1 小时）无进展且无活跃进程，判定为 stale 并终止该任务。

**致命错误熔断**：检测到 API 认证失败（401/402/403）或配额耗尽时，标记全局熔断，所有 worker 停止拉取新任务。

## 参数

### 必填参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--task-dirs` | 逗号分隔的 task_dir 列表（与 `--task-dirs-file` 二选一） | `/path/31_ELU,/path/32_GELU` |
| `--task-dirs-file` | 每行一个 task_dir 的文件路径（与 `--task-dirs` 二选一） | `debug_targets.txt` |
| `--containers` | 逗号分隔的 Docker 容器名 | `excludedprovider_cann1,excludedprovider_cann2` |
| `--npus` | 逗号分隔的 NPU ID（与 containers 一一对应） | `1,6` |
| `--output` | 输出目录路径 | `/path/to/debug_output` |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--agent` | `constructive` | agent name；可传 `constructive` / `discovery` 短名或完整 agent spec 名 |
| `--model` | 环境变量 `ANTHROPIC_MODEL` | Claude 模型 ID |
| `--max-attempts` | `5` | 单个算子最大 debug 轮数（`ASCENDC_DEBUG_MAX_ATTEMPTS`） |
| `--max-resumes` | `3` | pause_turn 最大恢复次数 |
| `--max-turns` | `240` | 单 attempt agentic turn 数硬上限 |
| `--soft-task-turns` | `480` | 无客观改善证据时的任务级软预算 |
| `--max-task-turns` | `600` | 有客观改善证据后的任务级硬上限 |
| `--kb-path` | 不启用 | 精度分支每轮检索并注入摘要；success 且无作弊时再入库的知识库 JSON 路径 |
| `--timeout` | `5400` | 单任务超时秒数（默认 1.5 小时） |
| `--stale-after-failure` | `3600` | 失败后停滞多久判定为 stale（秒） |
| `--stale-check-interval` | `60` | 停滞检测轮询间隔（秒） |
| `--workdir` | `/workspace/debug_agent` | 容器内项目根目录 |
| `--tilelang-env` | `/workspace/tilelang_ascend/set_env.sh` | tilelang 环境变量脚本 |
| `--claude-env` | 空 | API 凭证脚本路径（设置 `ANTHROPIC_API_KEY` 等） |
| `--claude-bin` | `claude` | Claude CLI 可执行文件路径 |
| `--allowed-tools` | `Bash,Read,Write,Edit,Glob,Grep,Skill` | 允许 Claude 使用的工具列表 |

## 使用方式

### 方式 A：直接指定 task_dir 列表

```bash
bash utils/run_ascendc_debug_batch_cc.sh \
    --task-dirs /workspace/debug_agent/outputs/run_20260422/31_ELU,/workspace/debug_agent/outputs/run_20260422/32_GELU \
    --containers excludedprovider_cann1,excludedprovider_cann2 \
    --npus 1,6 \
    --output /workspace/debug_agent/outputs/debug_$(date +%Y%m%d_%H%M)
```

### 方式 B：从文件读取 task_dir 列表

先生成目标文件（自动筛选验证失败的算子）：

```bash
find /workspace/debug_agent/outputs/run_20260422 \
    -name "latest.json" -path "*/.verify_status/*" \
    | xargs grep -l '"failure_type"' \
    | sed 's|/.verify_status/.*||' \
    | sort -u > debug_targets.txt
```

然后执行：

```bash
bash utils/run_ascendc_debug_batch_cc.sh \
    --task-dirs-file debug_targets.txt \
    --containers excludedprovider_cann1,excludedprovider_cann2,excludedprovider_cann3 \
    --npus 1,6,7 \
    --output /workspace/debug_agent/outputs/debug_run_01 \
    --max-attempts 7
```

### 切换为构建式 agent

```bash
bash utils/run_ascendc_debug_batch_cc.sh \
    --task-dirs /path/outputs/31_ELU \
    --containers excludedprovider_cann1 --npus 1 \
    --output /path/to/debug_output \
    --agent ascendc-debug-agent-constructive
```

### 指定 API 凭证

```bash
bash utils/run_ascendc_debug_batch_cc.sh \
    --task-dirs-file debug_targets.txt \
    --containers excludedprovider_cann1 --npus 1 \
    --output /path/to/debug_output \
    --claude-env /workspace/debug_agent
```

## 输入要求

每个 task_dir 目录必须包含主流程（ascend-kernel-developer-anti-cheat）的产物：

```
{task_dir}/
├── model.py                  # 参考实现（不可修改）
├── model_new_ascendc.py      # AscendC wrapper（不可修改）
├── kernel/                   # AscendC 源码（debug agent 可修改）
│   ├── *.cpp
│   ├── *.h
│   └── pybind11.cpp
├── trace.md                  # 主流程执行记录
└── {op_name}.json            # 算子配置（可选 .bak）
```

## 输出产物

### 目录结构

```
{output_dir}/
├── batch_report.md                      # 汇总报告
├── .queue                               # 任务队列（运行期间存在）
├── .lock                                # flock 锁文件
├── .fatal                               # 全局熔断标记（若触发）
├── worker_{container}_npu{N}.log        # 每个 worker 的详细日志
└── {op_name}/                           # 每个算子的输出
    ├── debug_trace.md                   # debug agent 的执行记录
    ├── debug_status.json                # 机器可读的 verdict
    ├── _claude_result.json              # Claude Code 最新输出
    ├── _claude_result_0.json            # 首次调用输出
    ├── _claude_result_1.json            # 第 1 次 resume 输出
    ├── _claude_result_reentry_1.json    # 第 1 次跨分支重入输出
    ├── _anticheat.json                  # 反作弊检测结果
    ├── .verify_status/                  # 验证状态
    ├── .verify_logs/                    # 验证日志
    └── kernel/                          # 修复后的 kernel 代码
```

### batch_report.md 格式

```markdown
# Claude Code AscendC Debug 批量执行报告

- containers: excludedprovider_cann1,excludedprovider_cann2
- npus: 1,6
- task_dirs: 5 个
- max_attempts: 5
- agent: ascendc-debug-agent-discovery
- start: 2026-05-06 14:30:00

| # | task_dir | session_outcome | 耗时(s) | 容器@NPU |
|---|----------|-----------------|---------|----------|
| 1 | 31_ELU   | ✅ success      | 1823    | excludedprovider_cann1@npu1 |
| 2 | 32_GELU  | ❌ failed       | 3601    | excludedprovider_cann2@npu6 |
| 3 | 33_RELU  | 🚨 CHEAT        | 902     | excludedprovider_cann1@npu1 |

## 汇总

- 总数: 5
- success: 2
- stopped_by_attempt_limit: 1
- stopped_by_branch_limit: 0
- stopped_by_budget: 0
- stopped_by_loop_limit (legacy): 0
- ablation_violation: 0
- skipped_*: 0
- engine timeout: 0
- terminated_by_sigterm: 0
- killed_by_sigkill: 0
- stale_after_failure: 0
- failed / crashed / ablation_violation / other engine errors: 1
- 作弊 (🚨 CHEAT, 与 outcome 正交): 1
- 结束: 2026-05-06 18:45:00
```

## 结果判定

| session_outcome | 含义 | 图标 |
|----------------|------|------|
| `success` | 验证通过 | ✅ |
| `failed` | 验证未通过，attempt 耗尽 | ❌ |
| `stopped_by_gate` | Gate 检测到有害回退，主动停止 | ❌ |
| `stopped_by_attempt_limit` | 达到全局 attempt 上限 | ⛔ |
| `stopped_by_branch_limit` | 达到单 failure-type 分支上限 | ⛔ |
| `stopped_by_budget` | 达到算子任务 turn 预算 | ⛔ |
| `stopped_by_loop_limit` | 旧产物兼容字段；新运行不再写入 | ⛔ |
| `ablation_violation` | Agent 使用了当前 arm 明确禁用的能力 | ❌ |
| `progressed_to_new_failure_type` | 修复了一种 failure，转变为另一种 | ↗（自动重入） |
| `skipped_env_issue` | 环境问题，非 kernel 实现错误 | ⊘ |
| `skipped_unsupported_type` | 不在支持的 failure_type 白名单 | ⊘ |
| `crashed` | agent 异常终止 | ❌ |
| `timeout` | agent 内部超时 | ⏱ |
| `engine_timeout` | 外层 `timeout` 达到 `--timeout` 后终止 engine，退出码 124 | ⏱ |
| `terminated_by_sigterm` | engine/docker exec 收到 SIGTERM，退出码 143；不等同于超时 | 🛑 |
| `killed_by_sigkill` | engine/docker exec 收到 SIGKILL，退出码 137；常见于强杀或 OOM | 💥 |

反作弊（🚨 CHEAT）与上述 outcome 正交：wrapper 被修改则无论 outcome 如何均判作弊。

## 与 Codex 版的差异

| 方面 | Codex 版 (`run_ascendc_debug_batch.sh`) | Claude Code 版 (本脚本) |
|------|----------------------------------------|------------------------|
| CLI 调用 | `codex exec --dangerously-bypass-approvals-and-sandbox` | `claude --bare -p --agent` |
| Agent 加载 | prompt injection（读文件作为 developer instructions） | `--agent` 原生加载 |
| Session 管理 | 无（单次执行） | `--session-id` + `--resume` + `MAX_RESUMES` |
| 停滞检测 | 无 | 文件 mtime + 活跃进程监控 |
| 致命错误熔断 | 无 | API 401/402/403/quota 检测 |
| 进程清理 | 无 | TERM→KILL 两阶段清理；engine 路径只按 task_dir/cwd 清理，避免宽 token 误杀同容器其它任务 |
| 跨分支重入 | 重新调用 codex exec | 新 session_id + `--agent` |
