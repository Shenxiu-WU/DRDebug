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

真实数值验证需要兼容的 Linux 昇腾机器或其已有容器：可访问的 NPU、CANN/AscendC、CMake/C++ 工具、匹配的 PyTorch/torch_npu、NumPy 与 pybind 头文件。论文引擎还需要已认证的 Claude Code 命令和模型服务；安装到 Codex 不会替换这一工作进程后端。

以 Codex 用户级安装为例：

```bash
DRDEBUG_SKILL="$HOME/.agents/skills/drdebug"
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" doctor
```

Claude Code 用户级安装则改为 `DRDEBUG_SKILL="$HOME/.claude/skills/drdebug"`；项目级或自定义安装使用实际绝对路径。`doctor --json` 输出结构化报告，`--strict` 在必要运行检查失败时返回非零状态。检查不调用付费模型，也不更改机器配置。

运行环境按硬件厂商的版本兼容关系配置，模型认证沿用用户已有设置。不要将凭据写入 Skill 或源码。没有 NPU 的电脑仍可安装、阅读源码、查看依赖报告；数值修复需要在实际运行机器上执行。

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

## 4. 创建独立修复目录

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" prepare /path/to/my-task \
  --output /path/to/my-repair --op-name MyOperator
```

准备过程复制任务和运行代码，原任务保留在原处。输出选择新目录；已有目录应通过续跑继续使用。若有独立 AscendC 参考案例，可添加 `--reference-root /path/to/reference-cases`。

工作副本保留论文引擎的目录关系，并注册 Claude 工作进程与技能。安装器同时打包 10 个历史参考案例，其中 7 个带 AscendC 内核。适配层优先读取匹配的 archive 案例；未覆盖该算子时，以只读 `model.py` 和已公开的 AscendC 文档建立参考规范，再审计当前内核。知识检索显式使用已公开的 102 条初始知识 fixture，保持只读。历史参考树已按 Git 对象恢复并记录来源；初始知识 fixture 不冒充原实验的 canonical KB。详见[运行与任务规范](../skills/drdebug/references/runtime.md)。

也可从项目根目录直接选择归档中的原始候选，例如 Add：

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" prepare \
  papers/2026-dont-regenerate-debug/artifact/inputs/tasks/level1/003_Add \
  --output /path/to/add-repair --op-name 3_Add
```

## 5. 在明确预算内运行

使用当前 CANN 工具链接受的实际 SoC 标识，替换下例的 `YOUR_SOC_VERSION`；不要仅根据排行榜的硬件名称猜测。

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" run /path/to/my-repair \
  --npu 0 --soc YOUR_SOC_VERSION \
  --max-attempts 5 --max-turns 80 --deadline-sec 1800
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" status /path/to/my-repair
```

可用 `--model MODEL` 指定已配置的工作模型。`run --dry-run --json` 只展示计划命令，不调用模型或执行 NPU 验证；命令可审阅不表示运行环境已经验收。检查指定任务可用 `doctor --workspace /path/to/my-repair --npu 0 --soc YOUR_SOC_VERSION --strict`。

默认工作进程修改内核源码与指定诊断报告，构建与评测由确定性引擎执行。若某个诊断探针需要额外 shell 执行能力，请先核对用途，仅在可信、隔离的运行目录中显式选择 `--allow-worker-shell`。它不绕过智能体权限系统，完整数值与完整性验收依然由引擎裁决。

续跑前先查看 `status`，保留事件日志与尝试次数。不要清空证据或重置预算来把未完成的任务描述为成功。环境缺失、模型认证/配额错误和权限拒绝应与内核失败分别说明。

也可以直接对智能体说：

> 使用 drdebug 检查我的昇腾环境，为这个失败算子创建独立工作目录，再根据失败证据修复它。使用 0 号设备，最多尝试 5 轮或运行 30 分钟。

## 修复成功的依据

智能体自述成功或进程成功启动都不足以证明修复完成。查看引擎生成的任务状态、轨迹、汇总，以及完整数值测试和完整性结果。修复后的内核须在参考实现、wrapper、测例和容差保持原样的条件下通过全部必要测试。失败和停止的运行同样保留，便于后续定位。

Skill 封装公开论文版精度修复流程。CPU 适配测试验证安装、任务准备、命令转发和失败处理，不代表完成了真实 NPU 修复，也不代表复跑了论文的 27 个历史任务。论文归档及其[已知限制](../papers/2026-dont-regenerate-debug/artifact/KNOWN_LIMITATIONS.md)与新增的移植层分别维护。
