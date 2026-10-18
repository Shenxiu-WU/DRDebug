# Cumsum：重放已修复内核，或亲自完成一次修复

[English](README.md) · [简体中文](README.zh-CN.md)

本例提供两个明确区分的入口：**不调用模型，验收保存的已修复内核**；或者**用 Kimi 从论文原始失败内核发起有参考的实时修复**，另外提供公开修复源码作为只读案例。两者均使用原始参考实现、Python wrapper、10 例紧凑集、51 例完整集及按输出 dtype 区分的精度阈值，`papers/` 中的论文归档保持不变。

| 入口 | 起始内核 | 是否需要模型 | 可以验证什么 |
| --- | --- | --- | --- |
| 已修复候选重放 | 保存的八份修复后原生源码 | 不需要 | 该候选在你的环境中能否通过原始测例 |
| 有参考的实时修复 | 论文原始近成功内核；单独提供已公开修复源码 | 需要配置好 Kimi 的 Claude Code | 在公开参考辅助下验证全新安装、诊断/补丁与完整验收链路 |

## 运行准备

实际执行需要 Linux 昇腾环境。请先完成[安装与依赖指南](../../docs/skill-guide.zh-CN.md)，并确保启动器和 `PATH` 中的 `python3` 使用同一 Python 环境。运行需要可用 NPU、兼容的驱动与 CANN、CMake、C++ 编译器以及 ABI 兼容的 Torch/torch_npu；单纯复制安装 Skill 只需要 Python 3.10+。

保存的候选已在 **Ascend910B4、CANN 9.1.0、Torch 2.9.0、torch_npu 2.9.0.post6** 上独立验收，并显式启用确定性算法、设置 `warn_only=False`。[安装指南](../../docs/skill-guide.zh-CN.md#使用匹配且隔离的-python-环境)说明匹配环境及 Torch 的 `+cpu` 版本串，其他硬件和版本仍未完成资格验证。[验证记录](validation.zh-CN.md)给出带新增 guard 的候选重放及另行全新引导修复结果，并区分运行版本和依赖复用。确定性模式是已测合同的一部分：关闭时，原始参考本身可能产生影响精度比较的运行间差异。

以下命令均在仓库根目录执行，示例路径必须选择新目录，两类 prepare 都不会覆盖已有任务或工作区。将 `YOUR_SOC_VERSION` 替换为本机工具链接受的 SoC 标识，并选用执行环境可见的空闲物理设备 ID。`--npu ID` 通过 `ASCEND_RT_VISIBLE_DEVICES` 选择该 ID，适配层内部统一使用逻辑 `npu:0`。本次验证配置为 `--npu 1`、内部逻辑 `0`，SoC 为 `Ascend910B4`。请将下方示例中的 `--npu 0` 替换为你选定的空闲物理 ID。

如尚未安装，可在源码仓根目录安装一份独立 Skill：

```bash
./install.sh --dest "$HOME/.local/share/drdebug-example-skills"
DRDEBUG_SKILL="$HOME/.local/share/drdebug-example-skills/drdebug"
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" doctor --no-worker --runtime-probe \
  --npu 0 --soc YOUR_SOC_VERSION --strict
python3 examples/cumsum/prepare.py --check
```

如果已经采用 Codex 或 Claude 的用户安装方式，将 `DRDEBUG_SKILL` 改为对应的 `drdebug` 安装目录即可。示例源码来自仓库 checkout；安装 Skill 不会将全部论文任务和示例一起复制进去。

## 1. 重放已修复候选

此入口独立清洁构建并进行精度验证，不让 Kimi 再次修复，也不导入历史成功状态。

```bash
python3 examples/cumsum/prepare.py --mode replay --output /path/to/cumsum-replay-input
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" prepare /path/to/cumsum-replay-input \
  --output /path/to/cumsum-replay --op-name 5_Cumsum --determinism on
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" verify /path/to/cumsum-replay \
  --npu 0 --soc YOUR_SOC_VERSION --determinism on --casewise
```

应同时满足 **紧凑集 10/10、完整集 51/51、全部完整性及 AST 检查通过**；加上 `--casewise` 时，还应有 51 个独立进程完成且全部匹配。准备好源码、成功编译或进程退出，均不能单独视为精度通过；以最终 `verification/<id>/summary.json` 和 CLI 退出状态为准；`progress.json` 是过程快照，结束后仍可能保留初始状态。失败时再读取阶段日志。

逐例验证会在每个新进程中按原 seed 0 重建整套输入，再选择对应原始 case 下标，不修改参考或精度阈值。一次运行完整 51 例和逐例新进程是两种不同测量，请保留两份结果。

## 2. 用 Kimi 完成有参考的实时修复

请先配置已认证的 Claude Code CLI 和有权使用的 Kimi 服务；官方 Claude Code 安装入口与 provider/工具检查见[安装指南](../../docs/skill-guide.zh-CN.md#修复前检查所选设备)。本例不附带凭据或服务设置。所选模型须由你的服务支持，此前使用的标识为 `k3[1m]`。

首次使用推荐此路径：任务保持论文原始失败内核，通过 `--reference-root` 将 `examples/cumsum/repaired` 另行提供给 worker，作为诊断和修改时可读取的 AscendC 实现案例。这属于**有公开参考的引导修复（guided repair）**，不声称无提示独立再发现修复方法。它验证全新安装、真实模型与工具调用、补丁以及原始验收整条链路，不继承旧 events 或 checkpoint。实时 Kimi 调用可能产生服务费用，最终候选仍须通过未修改的原测例。

```bash
python3 examples/cumsum/prepare.py --mode live --output /path/to/cumsum-live-input
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" prepare /path/to/cumsum-live-input \
  --output /path/to/cumsum-live --op-name 5_Cumsum --determinism on \
  --reference-root "$PWD/examples/cumsum/repaired"
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" doctor --workspace /path/to/cumsum-live \
  --npu 0 --soc YOUR_SOC_VERSION --strict
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" run /path/to/cumsum-live \
  --npu 0 --soc YOUR_SOC_VERSION --model 'k3[1m]' \
  --max-attempts 5 --max-turns 240 --worker-timeout-sec 21600 --deadline-sec 86400 \
  --allow-worker-shell
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" status /path/to/cumsum-live
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" verify /path/to/cumsum-live \
  --npu 0 --soc YOUR_SOC_VERSION --determinism on --casewise
```

`--allow-worker-shell` 允许在可信、隔离的执行工作区内运行诊断命令，不绕过 Claude Code 的权限或目录信任机制。若 CLI 要求信任目录，请按安装指南审阅生成的 runtime，并交互式确认；将其他任务和重要数据放在工作区之外。最终验收仍独立于模型的文字结论。

这些预算是上限，不代表预期消耗或成功保证。保留中断或失败的工作区，按[续跑与新试验规则](../../skills/drdebug/references/runtime.md#first-failure-or-resume)处理。

实测全新引导修复从 **6/10** 基线出发，使用 **1 次尝试 / 38 turns**，达到**紧凑集 10/10、完整集 51/51**，随后 **51/51** 个逐例新进程通过。worker 耗时 **6 分 25 秒**，含基线和引擎验收共 **15 分 5 秒**。它使用公开参考和较早适配器，随后对相同产出独立完成新增非有限值 guard 检查。LLM 不保证每次轨迹相同或成功。若研究不提供该参考时的修复能力，可在另一个新工作区中删去 `--reference-root`；这属于独立、可能消耗更多预算的研究运行。要复核公开的修复结果，请使用上面的已修复候选重放入口。

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
