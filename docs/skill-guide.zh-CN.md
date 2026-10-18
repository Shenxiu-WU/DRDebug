# 安装 DRDebug，完成第一次算子修复

[English](skill-guide.md) · [简体中文](skill-guide.zh-CN.md)

DRDebug Skill 将公开论文中的确定性引擎与操作指引一起安装。安装包携带实际运行源码，可以接收一个接近正确的 AscendC 算子任务，完成基于失败证据的修复。

## 1. 安装

在使用智能体的机器上执行：

```bash
git clone https://github.com/Shenxiu-WU/DRDebug.git
cd DRDebug
./install.sh --codex
```

按使用的智能体选择一个安装目标：

| 目标 | 命令 | 安装后的 Skill |
| --- | --- | --- |
| Codex 用户级 | `./install.sh --codex` | `~/.agents/skills/drdebug` |
| Claude Code 用户级 | `./install.sh --claude` | `~/.claude/skills/drdebug` |
| Codex 项目级 | `./install.sh --codex --project /path/to/project` | `/path/to/project/.agents/skills/drdebug` |
| Claude Code 项目级 | `./install.sh --claude --project /path/to/project` | `/path/to/project/.claude/skills/drdebug` |
| 自定义父目录 | `./install.sh --dest /path/to/skills` | `/path/to/skills/drdebug` |

`--project` 不带路径时使用当前目录。`--dry-run` 只预览位置，不写文件。相同内容重复安装不会改动已有安装；内容不同则默认保留，使用 `--force` 更新时会先备份。安装仅需 Python 3.10+ 与标准库，不下载依赖、不修改驱动或配置模型账号。

安装目录包含 `SKILL.md`、启动器、参考说明，以及 `runtime/source/debug_agent_v5/` 和源码哈希、来源记录。之后移动或删除克隆的仓库，不影响已安装的 Skill；需要更新时重新执行安装器即可。

Codex 中可用 `$drdebug`，Claude Code 中可用 `/drdebug`。如果列表中尚未出现，开启一个新的智能体会话。安装路径依据 [Codex 官方说明](https://learn.chatgpt.com/docs/build-skills)与 [Claude Code 官方说明](https://code.claude.com/docs/en/skills)。

## 2. 检查实际运行机器

实际执行需要可访问 NPU 的 Linux 昇腾机器或兼容容器、已安装的 CANN/AscendC、CMake 3.16+、C++ 编译器及 Python 开发头文件，还需要 NumPy、匹配的 Torch/torch_npu 和 pybind 头文件。实时修复另需已认证的 Claude Code 命令与模型服务；重放已修复候选不调用模型。

以 Codex 用户级安装为例：

```bash
DRDEBUG_SKILL="$HOME/.agents/skills/drdebug"
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" doctor
```

Claude Code 改为 `DRDEBUG_SKILL="$HOME/.claude/skills/drdebug"`；项目级或自定义安装使用实际绝对路径。`doctor --json` 输出结构化报告，`--strict` 在必要检查失败时返回非零状态。默认检查会在子进程中导入运行库，但不导入用户任务、不执行其内核、不调用付费模型。

### 使用匹配且隔离的 Python 环境

按照[昇腾官方兼容表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)选择 Python、Torch、torch_npu、CANN 和驱动组合，不能仅凭 Torch/torch_npu 主次版本相同就认定兼容。冻结扩展构建固定 `_GLIBCXX_USE_CXX11_ABI=1`，`doctor` 现在检查实际 Torch ABI、解释器/PATH 身份、NumPy 与 Torch 的数组往返、头文件及编译器/工具链文件，不再仅凭导入和版本串判断。优先采用 Torch 自带 pybind 头文件，独立 `pybind11` 包为备选。

Cumsum 的已验组合为 **Ascend910B4 + CANN 9.1.0 + Torch 2.9.0+cpu + torch_npu 2.9.0.post6 + 确定性模式开启**。该版本组合在厂商兼容表中有对应关系，只作为实例，不代表通用最低要求，也不外推 A3/910C。候选重放与引导修复的实际结果、各自运行版本见[验证记录](../examples/cumsum/validation.zh-CN.md)。

下列独立 Python 3.11 配方创建了已测依赖环境；后续引导修复和新 guard 检查另装 Skill、另建工作区，复用同一依赖。保留 venv 自带 pip、不升级 pip，并固定构建辅助包版本以便复验；这些是已测配方，不是对所有支持系统的通用要求。`set_env.sh` 使用本机已装工具链的实际路径；这些命令不更改驱动、CANN 安装或全局 Python 包。其他架构和版本应选择厂商对应 wheel，不要强行套用此组版本。

```bash
source /path/to/installed/cann/set_env.sh
python3.11 -m venv /path/to/drdebug-venv
source /path/to/drdebug-venv/bin/activate
python3 -m pip install 'torch==2.9.0' --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install 'torch-npu==2.9.0.post6' 'numpy==1.26.4'
python3 -m pip install 'setuptools==80.9.0' 'wheel==0.45.1' 'PyYAML==6.0.3'
```

下载缓慢或停滞时，可以使用可信镜像，或预先下载与运行机器 Python ABI、架构匹配的 wheel。镜像文件安装到同一个新 venv 前，应将其 SHA-256 与该文件在 PyPI 的官方元数据逐一核对，例如 [torch-npu 发布文件](https://pypi.org/project/torch-npu/2.9.0.post6/#files)或 [PyYAML 发布文件](https://pypi.org/project/PyYAML/6.0.3/#files)，不能只看包名和版本。保留下载、中断及哈希检查收据。这属于下载传输恢复，不是借用旧环境中已安装的包，也不代表数值链路已验收通过。安装方法见 [pip 的本地 wheel 指南](https://pip.pypa.io/en/stable/user_guide/#installing-from-wheels)。

后续每条命令均激活同一 venv，并确保其 `python3` 位于 `PATH` 最前，因为冻结 CMake 构建也会调用 `python3`。Torch 版本串中的 `+cpu` 可与 `torch_npu` 提供的 NPU 后端共存，应以真实设备检查为准。避免继承无关 `sitecustomize` 启动钩子；适配器会报告冲突，不静默覆盖全局启动配置。

### 修复前检查所选设备

选择执行环境可见的空闲物理设备 ID，以及本机工具链接受的真实 SoC 标识。`--npu ID` 通过 `ASCEND_RT_VISIBLE_DEVICES` 选择该物理 ID，适配层内部随后统一使用逻辑 `npu:0`。本次验证配置为 `--npu 1`、内部逻辑 `0`，SoC 为 `Ascend910B4`。以下命令中的物理 `0` 仅为示例，应替换为你选定的空闲 ID；不要仅凭排行榜硬件名称猜测 SoC 标识。

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" doctor \
  --npu 0 --soc YOUR_SOC_VERSION --runtime-probe --strict
```

`--runtime-probe` 执行内置的小规模 NPU 加法与固定 FP16 输入的重复 cumsum，不构建或执行用户内核，也不代表完整任务验收。`run` 和 `verify` 同样要求此探针。只重放已修复候选、不安装 Claude Code 时，可附加 `--no-worker`，它只跳过 worker/CLI 检查，仍检查构建与 NPU 依赖。

实时修复需要先按 Claude Code [官方安装指南](https://code.claude.com/docs/en/setup)安装，并完成[快速入门](https://code.claude.com/docs/en/quickstart)，确认命令位于 `PATH`：

```bash
claude --version
```

`claude_cli_contract` 使用无 prompt 参数探针检查部分 CLI 帮助中省略的选项，不调用模型。已有 provider 还需完成最小授权文本/工具往返：真实成功的 `tool_use`/`tool_result`，以及回传后模型的正常回复。版本检查或纯文本回答不能单独证明这条链路可用。

工作进程使用 `--setting-sources project`。仅存放在 `~/.claude/settings.json` 中的自定义 provider 环境变量不会自动继承，应通过启动进程的环境或已有密钥管理方式提供授权配置，不把 key 复制到任务、prompt、仓库或日志。OAuth 是独立认证路径。Provider/API 故障须与内核失败区分，可能令历史引擎进入终态；保留记录，不修改事件来重新打开旧 session。

## 3. 提供已有算子任务

```text
my-task/
├── model.py                   # torch.nn.Module 参考实现
├── model_new_ascendc.py        # 调用已有扩展的 Python wrapper
├── MyOperator.json            # 输入 JSONL，每个对象含 inputs
├── MyOperator.json.bak         # 显式提供的完整测试集
└── kernel/
    ├── pybind11.cpp
    └── ... .cpp / .h / .hpp   # 待修复的 AscendC 源码
```

至少显式提供一份 `.json.bak` / `.json.full` 完整集合，保留参考语义与原有容差。完整集合可以与精简集合相同，启动器不会代为生成覆盖；两组测例不要求包含关系。等价的完整备份可以共存，互相冲突的完整集合会被拒绝。`--op-name` 使用 JSON 文件 stem，例如目录名为 `003_Add` 时，`3_Add.json` 对应 `3_Add`。

[论文测例补充包](../papers/2026-dont-regenerate-debug/dataset/)提供参考实现与输入规格；独立的 [27 个初始候选](../papers/2026-dont-regenerate-debug/artifact/inputs/)补充逐字节核验的原始 wrapper 与内核。[历史参考案例包](../papers/2026-dont-regenerate-debug/artifact/code/debug_agent_v5/archive_tasks/)恢复工作智能体使用的 archive 参考目录。

测例补充包与 27 个初始任务属于仓库材料，不随 Skill 安装复制；请从仓库选择任务，或提供自己的任务。10 个历史参考案例则随安装提供。

## 4. 创建独立修复目录

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" prepare /path/to/my-task \
  --output /path/to/my-repair --op-name MyOperator --determinism on
```

准备过程复制任务和运行代码，原任务保留在原处。新建 schema-3 工作区默认记录 `--determinism on`，上例显式传参便于读者看到数值合同。若有意关闭，可在准备时选择 `--determinism off`。启动配置会在托管 Python 的任务入口执行前应用 `torch.use_deterministic_algorithms(..., warn_only=False)` 并写入每进程收据；配置失败会停止该解释器，不带未知设置继续执行。

新建工作区还会在运行副本中加入非有限输出检查：先核对候选与参考输出中 NaN、正无穷、负无穷的类别和位置，参考中对应的非有限值允许保留；有限值继续使用原比较器及原容差。这是公开适配层新增的规则，论文冻结源码保持不变；`drdebug-workspace.json` 的 `nonfinite_guard` 记录比较器与辅助文件的哈希来源。没有该字段的旧工作区（包括此前的 schema-3）保留原运行副本；应保留其证据，另建工作区采用新检查。详见[输出合同](../skills/drdebug/references/runtime.md#nonfinite-output-contract)。

模式对该工作区保持固定。旧 schema-1/2 工作区继续保留 legacy off 行为，不静默迁移；保留旧目录，另建新工作区使用新配置。不要用改参考或放宽阈值抵消非确定性。确定性模式遇到不支持的操作可能报错，也不保证跨版本或硬件一致，详见 [PyTorch 确定性 API](https://docs.pytorch.org/docs/2.9/generated/torch.use_deterministic_algorithms.html)及[复现说明](https://docs.pytorch.org/docs/2.9/notes/randomness.html)。输出选择新目录；已有目录应通过续跑继续使用。若有独立 AscendC 参考案例，可添加 `--reference-root /path/to/reference-cases`，材料独立保存在 `WORKSPACE/runtime/archive_tasks/user_references/`。未提供该选项时，此目录可以不存在，应直接跳过。

首次使用时，Claude Code 可能提示生成的 runtime 尚未受信任，并忽略项目的 `permissions.allow` 或 `additionalDirectories`。遇到此提示，应由用户审查生成目录及其配置，再从 `/path/to/my-repair/runtime` 交互启动 Claude Code，明确确认信任该目录后续跑同一 workspace。安装器和 Skill 不会自动修改全局信任；项目配置文件存在，不等于其权限已经生效。不要用权限 bypass 消除警告。

工作副本注册论文 worker 和方法资源；有相关的用户参考案例时先使用它们，再选用历史案例或受保护的任务参考与公开文档。[公开资源表](../skills/drdebug/references/runtime.md#public-resource-map)列出只读 KB 种子与未打包的历史 API/KB 路径，按表使用，不构造替代文件。

也可从项目根目录直接选择归档中的原始候选，例如 Add：

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" prepare \
  papers/2026-dont-regenerate-debug/artifact/inputs/tasks/level1/003_Add \
  --output /path/to/add-repair --op-name 3_Add
```

## 5. 在明确预算内运行

使用当前 CANN 工具链接受的实际 SoC 标识，替换下例的 `YOUR_SOC_VERSION`；不要仅根据排行榜的硬件名称猜测。全新任务的默认 profile 为最多 5 轮尝试、每次完整 worker 调用最多 240 turns、单次最多 6 小时、总计最多 1 天：

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" run /path/to/my-repair \
  --npu 0 --soc YOUR_SOC_VERSION \
  --max-attempts 5 --max-turns 240 \
  --worker-timeout-sec 21600 --deadline-sec 86400
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" status /path/to/my-repair
```

`--worker-timeout-sec` 限制单次 worker 调用；总 `--deadline-sec` 独立限制整个任务，启动器会将单次超时压到剩余总预算以内。这些宽裕默认值是上限，不是最低消耗：验收通过即可提前结束，尝试次数与 turn 闸仍有效。6 小时提供宽裕兜底，并非实验证明的最优值；可按用户任务需要配置，但不改变测例与验收条件。

首次执行保存预算，续跑保留原值。旧版本限制、预算记录缺失、中断及终态处理详见[预算与恢复规则](../skills/drdebug/references/runtime.md#first-failure-or-resume)。

可用 `--model MODEL` 指定已配置的工作模型。`run --dry-run --json` 只展示计划命令，不调用模型或执行 NPU 验证；命令可审阅不表示运行环境已经验收。检查指定任务可用 `doctor --workspace /path/to/my-repair --npu 0 --soc YOUR_SOC_VERSION --strict`。

默认工作进程修改内核源码与指定诊断报告，构建与评测由确定性引擎执行。若某个诊断探针需要额外 shell 执行能力，请先核对用途，仅在可信、隔离的运行目录中显式选择 `--allow-worker-shell`。它不绕过智能体权限系统，完整数值与完整性验收依然由引擎裁决。

续跑前查看 `status`，保留事件、尝试次数和已保存预算，不重置状态或重新打开终态 session。若授权另开试验，从已审阅源码准备新工作区，保留原参考与测例，记录父任务/checkpoint 和此前预算消耗。环境、provider、权限问题须与内核失败区分，详细情形见[恢复规则](../skills/drdebug/references/runtime.md#first-failure-or-resume)。

也可以直接对智能体说：

> 使用 drdebug 检查我的昇腾环境，为这个失败算子创建独立工作目录，再根据失败证据修复它。使用 0 号设备，沿用全新任务默认上限，引擎验收通过即停止。

## 6. 独立验收实际候选

修复后，或不使用模型、仅测试已有候选时，执行：

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" verify /path/to/my-repair \
  --npu 0 --soc YOUR_SOC_VERSION --determinism on --casewise
```

`verify` 要求新建的 schema-3 工作区。它将**当前实际内核源码**复制到新的 `WORKSPACE/verification/<id>/`，清洁重建原生扩展，以新解释器分别运行原紧凑集、完整集和新增非有限输出检查，再检查原生 wrapper 绑定、原始 C++ 完整性、wrapper AST、保护材料、源码和历史完整性。`--casewise` 另为每个原完整集 case 启动独立进程。此命令不调用模型、不提升检查点、不改写引擎历史，每次验证独立保留汇总及阶段日志。

以最终 `verification/<id>/summary.json` 和 `verify` 命令退出状态共同判定结果，并核对非有限输出检查覆盖。`progress.json` 仅记录过程快照，完成后仍可能保留初始的 `passed: false` 或返回码，不能据此判定最终失败。

默认每阶段 1,800 秒，总计 10,800 秒；较大任务可显式设置 `--stage-timeout-sec` 和 `--overall-timeout-sec`，二者与实时 worker 预算分开。确定性模式须与 prepare 一致。如果别的 worker 持有工作区锁，应停止，不删除锁来并发验收。

### 从 Cumsum 示例开始

[Cumsum 操作示例](../examples/cumsum/README.zh-CN.md)提供两条路径：无需模型，重放八份固定哈希的已修复源码；或者用 Kimi 从论文原始输入开始有参考的修复，添加 `--reference-root "$PWD/examples/cumsum/repaired"`，将公开修复实现单独作为只读案例。后者用于验证安装、模型、补丁与验收整条链路，不声称无提示独立再发现；如研究不提供该参考时的能力，可删去参数，另开可能更耗时的研究运行。两者均保留原 10 例紧凑集和 51 例完整集。带新增非有限值检查的候选重放通过 10 例紧凑集、51 例完整集及 51 个逐例新进程；引导修复另在 1 次尝试后通过相同计数。实时修复使用较早运行版本，相同产出随后完成新 guard 检查，详见[版本化验证记录](../examples/cumsum/validation.zh-CN.md)。实时 LLM 修复具有随机性，不能保证每次轨迹和结果完全相同。

运行副本还会在 baseline 验证前检查并创建常规 `precision_tuning/` 目录，使尚未调用 worker 的首次完整结果也能保存。论文冻结源码保持不变。

## 修复成功的依据

智能体自述成功或进程成功启动都不足以证明修复完成。适配层现在将客观结果绑定到精确的内核和保护输入哈希，并在最佳检查点排序中优先比较完整执行的全量结果，因此当紧凑集同为 10/10 时，完整集 50/51 → 51/51 的改善不会再丢失。完整执行但部分通过的候选也可能是当前 best；“best”本身不等于接纳，接纳仍需原始数值与完整性闸、新增非有限输出检查全部通过，并核对最终源码与检查点一致。查看引擎生成的任务状态、轨迹、汇总，以及完整数值测试和完整性结果。修复后的内核须在参考实现、wrapper、测例和容差保持原样的条件下通过全部必要测试。失败和停止的运行同样保留，便于后续定位。

若已有 baseline 全部正确，引擎精度接纳依据为 `run` 退出 0、`accepted_checkpoint.ready: true` 和已解锁的成功终态，无需一次修复尝试。旧 `engine_status.objective_success` / `reportable_success` 描述修复末轮验证，零尝试时可能仍为 false；独立 `verify` 仍提供单独的数值结论。

Skill 封装公开论文版精度修复流程。CPU 适配测试验证安装、任务准备、命令转发和失败处理，不代表完成了真实 NPU 修复，也不代表复跑了论文的 27 个历史任务。论文归档及其[已知限制](../papers/2026-dont-regenerate-debug/artifact/KNOWN_LIMITATIONS.md)与新增的移植层分别维护。
