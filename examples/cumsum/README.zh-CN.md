# Cumsum：重放候选，或由当前宿主修复

[English](README.md) · [简体中文](README.zh-CN.md)

可选择**不调用模型，独立验证保存的候选**，或**从原始失败任务出发，由当前宿主进行有参考修复**。两条路径都保留原始参考、wrapper、10 例紧凑集、51 例完整集与按 dtype 区分的容差；论文归档与历史验证回执保持不变。

| 路径 | 起始内核 | 模型／宿主 | 核验内容 |
| --- | --- | --- | --- |
| 已修复候选重放 | 固定哈希的八份修复后原生源码 | 不需要模型 | 独立清洁构建及原数值／完整性检查 |
| 原生宿主引导修复 | 论文原始输入，修复源码单独作为只读参考 | 现有宿主的模型与工具 | 真实请求 → 宿主补丁 → 引擎接纳 → 独立验证 |

## 运行准备

先阅读[当前安装指南](../../docs/skill-guide.zh-CN.md)和[执行环境说明](../../docs/skill-guide-requirements.md)。数值命令在 Linux 昇腾机器／容器中运行，需要兼容的驱动／CANN、ABI 匹配的 Torch/torch_npu、NumPy 与构建工具。启动器和 `PATH` 上的 `python3` 使用同一隔离环境；安装器自身仅需 Python 3.10+ 与标准库。

Claude Code、Codex、使用 Kimi 的 Agent 及其他兼容宿主任选其一。宿主使用自己的模型与工具，管理密钥、权限和远程 SSH；DRDebug 不启动嵌套编码智能体，也不选择 provider。候选重放不调用模型。

以下命令从仓库根执行，输入与工作区都选新目录，并使用执行环境可见的空闲设备 ID 和本机编译器接受的实际 SoC 标识。`--npu ID` 通过 `ASCEND_RT_VISIBLE_DEVICES` 选择该环境暴露的设备，适配器内部使用逻辑 `npu:0`。容器可能重映射宿主机的物理 ID；请在实际执行环境中核对枚举结果，单独记录宿主机物理 ID，并替换示例中的 `0` 和 `YOUR_SOC_VERSION`。

```bash
./install.sh --dest "$HOME/.local/share/drdebug-example-skills"
DRDEBUG_SKILL="$HOME/.local/share/drdebug-example-skills/drdebug"
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" doctor --runtime-probe \
  --npu 0 --soc YOUR_SOC_VERSION --strict
python3 examples/cumsum/prepare.py --check
```

如果已安装 Codex／Claude／自定义 Skill，改用实际安装路径。示例源码来自 checkout，安装器不会复制完整示例或全部论文任务。

## 1. 独立重放保存的候选

此入口不要求宿主修复，也不继承历史成功状态：

```bash
python3 examples/cumsum/prepare.py --mode replay --output /path/to/cumsum-replay-input
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" prepare /path/to/cumsum-replay-input \
  --output /path/to/cumsum-replay --op-name 5_Cumsum --determinism on
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" verify /path/to/cumsum-replay \
  --npu 0 --soc YOUR_SOC_VERSION --casewise --json
```

以实际最终 `verification/<id>/summary.json` 和 CLI 退出状态判定。完整通过要求**紧凑集 10/10、完整集 51/51、全部完整性／原生绑定／AST 检查与非有限值 guard 完整覆盖**；使用 `--casewise` 时，还需 51 个逐例新进程完成且匹配。这些是验收目标，不表示上面的命令已经在你的环境执行。`progress.json` 是过程快照，不是最终结论。

每个逐例进程按原 seed 0 重建完整输入集，再选择原始下标。整套完整集与逐例新进程是两种测量，均不更改参考或容差。

## 2. 使用原生宿主协议进行引导修复

修复目标保持原始失败任务。`--reference-root` 将 `examples/cumsum/repaired` 单独作为只读 AscendC 参考；需明确披露：这是**有参考的引导修复**，不是无提示独立再发现。若另行研究不提供参考的能力，可在另一个新工作区省略该参数，预算可能更高。

```bash
python3 examples/cumsum/prepare.py --mode live --output /path/to/cumsum-live-input
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" prepare /path/to/cumsum-live-input \
  --output /path/to/cumsum-live --op-name 5_Cumsum --determinism on \
  --reference-root "$PWD/examples/cumsum/repaired"
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" next /path/to/cumsum-live \
  --npu 0 --soc YOUR_SOC_VERSION --max-attempts 5 --deadline-sec 86400 --json
```

返回 `state: "awaiting_host"` 时，完整读取 `instructions` 指向的文件、证据、`request_id` 和 `candidate_dir`。当前宿主用自己的工具遵循完整诊断方法，只修改允许的隔离候选／审计文件，将真实回复写入候选源码之外的 `result_file`。不能改活动任务、受保护输入、请求元数据、引擎报告或历史。

结果 JSON 必须包含 `outcome`（`completed` 或 `failed`）和真实 `final_response`，并以完整提示要求的真实 `[ENGINE_ATTEMPT_METADATA]` 结尾；`failed` 必须附非空 `error`，`completed` 时可选。不需要自行给出 success／精度，也不伪造 turns。`completed` 仅表示宿主完成工作，不表示测试通过。

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" submit /path/to/cumsum-live \
  --request-id REQUEST_ID --result /path/from/result_file/host-result.json \
  --npu 0 --soc YOUR_SOC_VERSION --json
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" status /path/to/cumsum-live
```

两个占位符须替换为本请求的实际值。查看引擎验证结果，仅在非终态时重复 `next → 宿主编辑 → submit`。已保存上限为 5 次尝试、86,400 秒总墙钟，**包含等待宿主与确定性命令时间**，续跑保留原值。不删除历史、不改预算、不重开终态，也不编造模型 turn 记账，详见[恢复规则](../../skills/drdebug/references/runtime.md#first-failure-or-resume)。

接纳要求 `state: "terminal"`、`terminal: true`、`engine_status.session_outcome: "success"`、`returncode: 0` 与 `accepted_checkpoint.ready: true`。随后独立验证实际被接纳的源码：

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" verify /path/to/cumsum-live \
  --npu 0 --soc YOUR_SOC_VERSION --casewise --json
```

宿主回复或部分通过的最佳检查点不是成功。基线已通过全部闸时可无需宿主补丁接纳，不要伪造尝试。旧工作区只保留原证据，供 `status` 与受支持的独立 `verify`；当前修复使用新的 schema-4 工作区。原 `run` 命令已经退役。

## 原生宿主实测

[全新安装的原生宿主记录](native-host-validation.zh-CN.md)从原始紧凑集 6/10 出发，实际提交修复，取得紧凑集 10/10 + 完整集 51/51 接纳，再在另一份清洁工作区完成 51 个逐例新进程验证。记录列出实际测试宿主、K3／1M 配置、源码身份与提供的公开参考；另见[机器记录](native-host-validation.json)和[接纳源码快照](validation-sources/README.md)。

## 历史结果单独保留

[版本化验证记录](validation.zh-CN.md)与[机器可读回执](validation.json)早于原生 `next`/`submit`，保留了 **Ascend910B4 / CANN 9.1.0 / Torch 2.9.0 / torch_npu 2.9.0.post6**、确定性开启环境下的候选重放与 Kimi 引导修复。历史引导运行从 **6/10** 达到**紧凑集 10/10 + 完整集 51/51**，使用 1 次尝试／38 个真实 turns，随后 **51/51** 个逐例新进程通过。它采用较早适配器，相同产出随后通过新增非有限值 guard 的独立检查。

这些回执和模型专属耗时是历史证据，**不是新宿主协议的首次通过声明**。安装、待处理请求或适配器 CPU 测试都不代表新的 NPU 验收通过；应自行建立并保留绑定源码的数值结果。

## 已修复候选改变了什么

观察到的运行异常涉及短 FP32 UB 到 UB 拷贝：7 个 float 共 28 字节，不满足该拷贝指令的对齐条件。修复后用零值 `Adds` 完成该拷贝。之后的数值诊断发现了累加顺序和舍入顺序差异：保存的候选按 dtype/layout 分支，在 `outerSize == 1` 的非末轴 FP16/FP32 路径采用 40 个均衡连续分段，并显式执行 FP16 回舍；存在多个 outer slice 时保持顺序路径。这是精度修复，不代表性能收益，不是按 case ID 查表，也不能据此证明闭源参考后端普遍采用相同算法。

原始精度阈值保持为：

| 输出 dtype | `atol` | `rtol` |
| --- | ---: | ---: |
| FP32 | `1e-4` | `1e-4` |
| FP16 | `1e-2` | `1e-3` |
| BF16 | `5e-2` | `5e-3` |

原完整集 51 例包含 16 例 FP16、26 例 FP32 和 9 例 BF16。紧凑集和完整集是原有的两个测例集合，不能假定其中一个包含另一个；应分别报告通过情况。

## 来源与适用范围

[example.json](example.json) 固定了全部 15 份原始任务文件和八份修复后原生源码的 SHA-256 及大小。`prepare.py --check` 只核对字节，不导入任务代码、不编译、不使用 NPU、不调用模型。`--mode live` 复制原始任务；`--mode replay` 只替换其八份 `kernel/` 文件。两者都会在新输入任务内生成 `example-origin.json`，记录模式和每份 staged 源码哈希，请将该输入目录和修复工作区一起保留。

继承的 `SOURCE_NOTICE.md` 逐字节保留，描述的是论文原始输入；重放任务的内核来源以 `example-origin.json` 和本例[来源说明](SOURCE_NOTICE.md)为准，参考与测例的归属声明继续适用。本例不导入旧 engine 状态、检查点成功标记、编译产物、模型对话或凭据。适配层如何将完整结果绑定到候选源码，见[检查点合同](../../skills/drdebug/references/runtime.md#best-checkpoint-and-acceptance)。

[验证记录](validation.zh-CN.md)及[机器可读身份](validation.json)将实际结果绑定到各自源码和运行版本。已测范围为所列环境中的原 51 例完整集及单独紧凑集；任意更大 shape、非零 `storage_offset` view，以及通用空输入、溢出和 UB 容量处理不在本例实测范围内。请运行 `verify` 建立你自己的结果。
