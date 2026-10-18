# 安装 DRDebug，由当前宿主智能体修复算子

[English](skill-guide.md) · [简体中文](skill-guide.zh-CN.md)

DRDebug 将公开论文的确定性修复引擎封装成 Agent Skill。**当前宿主智能体用自己的模型、工具和对话提出补丁并编辑候选。** CLI 不再启动嵌套编码智能体，也不调用模型 API。当前流程为 `prepare → next → 宿主编辑 → submit`，循环到终态，再独立 `verify`。

## 1. 安装到你的宿主

```bash
git clone https://github.com/Shenxiu-WU/DRDebug.git
cd DRDebug
./install.sh --codex
```

选择现有宿主能够发现的目录；上例的 Codex 只是其中一种：

| 可选宿主／安装目标 | 命令 | Skill 安装位置 |
| --- | --- | --- |
| Codex，用户级 | `./install.sh --codex` | `~/.agents/skills/drdebug` |
| Claude Code，用户级 | `./install.sh --claude` | `~/.claude/skills/drdebug` |
| Codex，项目级 | `./install.sh --codex --project /path/to/project` | `/path/to/project/.agents/skills/drdebug` |
| Claude Code，项目级 | `./install.sh --claude --project /path/to/project` | `/path/to/project/.claude/skills/drdebug` |
| 其他兼容宿主，包括使用 Kimi 的 Agent | `./install.sh --dest /path/to/your-agent/skills` | `/path/to/your-agent/skills/drdebug` |

自定义安装时，将路径替换为实际 skills 父目录，再通过宿主机制加载或注册 `drdebug/SKILL.md`。模型名称本身不等于宿主：它还需要读取证据、编辑文件、执行或传递 CLI 命令的工具。Codex、Kimi 和通用宿主均不依赖内部 Claude Code 后端。认证、模型选择、权限与 SSH 由所选宿主配置，密钥不要写入任务、提示词、结果或日志。

`--project` 不带路径时使用当前目录；`--dry-run` 只查看安装位置。相同安装不重复写入；`--force` 更新不同时会先备份。安装器复制运行源码和来源记录，安装后不依赖原 checkout。它不安装宿主程序、NPU 驱动或 Python 运行包，也不修改模型账号配置。

采用上传方式时，先运行 `./install.sh --dest /path/to/staging-skills` 组装，再传输**生成的完整 `drdebug/` 目录**，包括 `runtime/`、`runtime/manifest.json` 和 Skill 脚本。在目标宿主中加载／注册其 `SKILL.md`，并在数值执行机器上运行 `doctor`。待修复任务单独提供。

## 环境要求

这些要求不能混成一个通用的 pip `requirements.txt`：

| 操作 | 所需环境 |
| --- | --- |
| 安装／查看公开 CLI | Python 3.10+ 与标准库；不要求 provider SDK 或编码智能体 CLI。 |
| 构建／数值验证算子 | Linux + 可访问的昇腾 NPU 与驱动、兼容 CANN/AscendC、CMake 3.16+、C++ 编译器／Python 头文件、匹配的 Torch/torch_npu、NumPy 与 pybind 头文件。 |
| 提出修复 | 一个宿主已有的模型与工具能力；Claude Code、Codex、使用 Kimi 的 Agent 或其他兼容宿主均可。 |

独立的[执行环境说明](skill-guide-requirements.md)区分系统／运行要求与 pip 包输入。

### 使用匹配的隔离 Python 环境

依照[昇腾兼容表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)选择 Python、包、CANN 与驱动组合。数值命令在 NPU 机器或其兼容容器中执行；远程宿主可使用自己已获授权的 SSH 工具。命令中的 `<skill-dir>` 和工作区路径指执行机器上的文件。

所有命令使用同一个已激活的隔离环境。冻结构建还会从 `PATH` 调用 `python3`，并固定扩展的 `_GLIBCXX_USE_CXX11_ABI=1`；仅包名相同不能证明 ABI 兼容。`doctor` 核对实际 ABI、解释器身份、NumPy/Torch 往返、头文件与工具链文件。优先使用 Torch 自带的 pybind 头文件，独立 pybind11 包作为后备。按供应商组合安装，不改全局启动钩子，也不把历史版本组合强行套到另一台机器。

## 2. 检查执行机器

以 Codex 用户级安装为例：

```bash
DRDEBUG_SKILL="$HOME/.agents/skills/drdebug"
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" doctor \
  --npu 0 --soc YOUR_SOC_VERSION --runtime-probe --strict
```

Claude Code、项目级和自定义安装使用各自实际路径。选择执行环境可见的空闲设备 ID 和已安装编译器接受的 SoC 标识，不根据排行榜硬件标签猜测。`--npu` 通过 `ASCEND_RT_VISIBLE_DEVICES` 选择该环境暴露的设备，适配器内部使用逻辑 `npu:0`。容器可能重映射宿主机编号；请在实际执行环境中核对枚举结果，并单独记录宿主机物理编号。

`doctor --json` 输出机器可读清单，`--strict` 在缺少执行依赖时失败。可选的 runtime probe 执行内置 NPU add 和固定输入 FP16 cumsum 重复测试，不运行用户内核、不认证宿主，也不调用模型。探针通过不等于任务接纳。预检不要求 Claude 可执行文件，也不再检查内部 worker 认证。

若包和工具链检查通过，但设备枚举失败，应单独核对设备占用。本次验证中，前一项实验已完成的容器仍在运行，后续容器报 `aclInit 507899 / EL0005 Resource_Busy`，而 NPU 进程表为空。停止这个已核实空闲的前驱容器后，最小张量测试与原版 `doctor` 均恢复。保留日志和文件，只释放自己持有且已完成的容器，再重试。DRDebug 不自动停止容器或重置硬件。

## 3. 提供一个已有任务

```text
my-task/
├── model.py                   # 受保护的参考 torch.nn.Module
├── model_new_ascendc.py        # 受保护的原生扩展 wrapper
├── MyOperator.json            # 紧凑输入测例 JSONL
├── MyOperator.json.bak         # 明确提供的完整测例集
└── kernel/
    ├── pybind11.cpp
    └── ... .cpp / .h / .hpp   # 已有候选实现
```

至少提供一个与活动 JSON 同名的完整 `.json.bak` / `.json.full` 集合。紧凑／完整集可相同，也不要求嵌套；冲突的完整集合会被拒绝。保留原始参考、测例与容差。`--op-name` 使用 JSON 文件名主体，例如 `003_Add/` 内的 `3_Add.json` 对应 `3_Add`。

仓库分别提供[参考与测例补充包](../papers/2026-dont-regenerate-debug/dataset/)和 [27 个初始候选任务](../papers/2026-dont-regenerate-debug/artifact/inputs/)，它们不复制到安装后的 Skill；安装包含 10 个用于方法参考的历史案例。只有 dataset 条目还没有候选内核。

## 4. 准备新工作区

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" prepare /path/to/my-task \
  --output /path/to/my-repair --op-name MyOperator --determinism on
```

新工作区使用 schema 4 和原生宿主交接。准备过程复制任务与 runtime 到新目录，保留原件并记录哈希。确定性默认开启且在工作区内固定；需要关闭时须在 prepare 前明确选择。运行副本还会先检查 NaN／正无穷／负无穷输出位置一致，再使用原有限值比较器和不变容差，详见[运行规范](../skills/drdebug/references/runtime.md#nonfinite-output-contract)。

若有独立参考案例，可加 `--reference-root /path/to/reference-cases`，单独复制为只读指导。保留 Phase A → B → C，遵循[公开资源表](../skills/drdebug/references/runtime.md#public-resource-map)，不构造缺失的历史 API 或 canonical KB 文件。

旧工作区不会热修改成新协议。保留其事件、检查点与 runtime，使用 `status` 或受支持的独立 `verify`；当前修复须另建 schema-4 工作区。旧 `run` 命令已经退役并提供迁移说明。

## 5. 由宿主完成每次修复请求

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" next /path/to/my-repair \
  --npu 0 --soc YOUR_SOC_VERSION --json
```

`next` 运行确定性步骤，直到返回待处理的宿主请求或终态。待处理响应为 `state: "awaiting_host"`、`terminal: false`。完整读取 `instructions` 路径指向的文件，并查看 `request_id`、`candidate_dir`、证据和允许路径；引擎拥有的 `request` 文件保持只读。宿主用**自己的模型和工具**遵循提示，编辑隔离候选并记录真实诊断；不能直接改活动任务、参考、wrapper、测例、runtime、请求或历史。

审计文件相对于 `candidate_dir` 的完整路径为 `precision_tuning/precision_audit_<attempt>.md` 和 `precision_tuning/candidate_kb_entry.json`。按返回指令中的完整路径写入或移动文件，保留 `precision_tuning/` 目录。

覆盖已有审计文件前先读取它。部分宿主编辑工具（包括本次测试的 Claude Code 版本）要求先读后写，即使完整路径正确也如此。

构建和验证可能超过宿主工具的前台等待上限。需要时使用宿主支持的后台／会话机制，保留 stdout/stderr 与最终退出状态，轮询同一个进程直到完成。不要在同一工作区启动重叠命令，也不要把进度文件当成最终结果。若命令被中断，先检查 `status` 和日志；结果不明的确定性操作需要排查，不能直接盲目重试。

方法保留规范优先的 Phase A、实现全量阅读与探针策略的 Phase B，以及结构化对照的 Phase C。引擎注入 KB 检索摘要；宿主只列实际采用的 ID，不重复检索或修改种子库。必要审计字段与完整数值／完整性检查仍有效。

请求中的 `diagnostics.task` 是独立的可写诊断副本，可执行修复所需的编译、局部测试和 L5 探针。先从当前候选刷新内核，再使用 `diagnostics.execution` 中固定的 Python、环境与命令参数，并遵循原探针策略。诊断输出仅作为分析证据，不会被提交导入，也不替代正式验收。

向返回的 `result_file` 路径写 JSON，放在候选源码之外。格式为：

```json
{
  "outcome": "completed",
  "final_response": "实际诊断和补丁总结，末尾附完整提示要求的 ENGINE_ATTEMPT_METADATA"
}
```

宿主工作失败时使用 `"outcome": "failed"`，且必须附非空 `"error"`。上面的说明文字须替换为真实回复和元数据。`completed` 只表示宿主已完成这一轮工作，不表示内核通过。不要伪造 success、精度或 turns。随后提交同一请求：

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" submit /path/to/my-repair \
  --request-id REQUEST_ID --result /path/to/request/host-result.json \
  --npu 0 --soc YOUR_SOC_VERSION --json
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" status /path/to/my-repair
```

引擎检查请求和源码身份，验证补丁，再推进任务。查看返回结果，仅在非终态时继续 `next → 宿主编辑 → submit`，直到终态接纳或其他终止结果。过期 ID、意外修改或完整性失败不构成手工更改引擎文件以续跑的理由。

新任务上限为 **5 次尝试、86,400 秒总墙钟时间**，包括等待宿主处理的时间与确定性命令执行。已保存预算跨命令保留。若授权使用不同上限，在首个原生命令设置 `--max-attempts` / `--deadline-sec`，后续命令保留原值。没有嵌套 worker、适配器选择的模型、单 worker turns 或伪造的宿主 turn 记账。宿主自己的模型调用可能收费，由该宿主管理。

引擎在推进工作区时检查截止时间，不控制或终止宿主的模型会话。请在宿主自身设置轮次、费用或时间上限；达到宿主上限后，保留会话、待处理请求与工作区证据，再按原状态续跑。

恢复前读取 `status`，保留请求 ID、事件与已消耗预算，不删台账／锁、不重置尝试次数或重开终态。授权继承一个新试验时，另建工作区并记录父任务／检查点与此前预算，详见[首次失败与恢复](../skills/drdebug/references/runtime.md#first-failure-or-resume)。

可以直接对宿主说：

> 使用 drdebug 和你当前的模型与工具，检查我的昇腾环境，创建独立工作区，按 next/submit 请求修复这个失败算子，再独立验证实际被接纳的源码。

## 6. 独立验证

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" verify /path/to/my-repair \
  --npu 0 --soc YOUR_SOC_VERSION --casewise --json
```

此操作将实际活动内核复制到新的验证目录，清洁重建并运行原紧凑／完整集比较，以及非有限值、原生绑定和完整性检查。`--casewise` 为每个原完整集 case 增加独立新进程。它不调用模型、不提升检查点，也不修改引擎历史。不要与待处理宿主编辑或其他工作区操作并发运行。

以最终 `verification/<id>/summary.json`、命令退出状态和 guard 覆盖共同判定。`progress.json` 只是过程快照。验证默认每阶段／总计 1,800／10,800 秒，可用 `--stage-timeout-sec` / `--overall-timeout-sec` 设置，与宿主循环预算独立。

## 什么才算修复成功

需要 `state: "terminal"`、`terminal: true`、`engine_status.session_outcome: "success"`、`returncode: 0` 和 `accepted_checkpoint.ready: true`，并有完整数值／完整性闸作为依据。宿主 `completed`、部分通过的最佳检查点或编译成功均不够。已有基线全部正确时可无需宿主补丁直接接纳，不要伪造尝试。独立验证须单独报告，并包含紧凑／完整计数、源码身份、模式、完整性与剩余失败。

[原生宿主 Cumsum 记录](../examples/cumsum/native-host-validation.zh-CN.md)包含全新安装、真实 `next`/`submit` 修复与单独的独立验收，并注明使用的公开参考和精确宿主配置。此前的[重放／引导修复记录](../examples/cumsum/validation.zh-CN.md)单独保留。安装、环境预检、CPU 适配器测试和真实 NPU 修复是不同结果。环境／权限、宿主／模型、超时、运行与数值失败应分别记录。
