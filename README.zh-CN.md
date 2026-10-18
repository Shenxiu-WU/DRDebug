<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/readme/hero-dark.svg">
    <img src="assets/readme/hero-light.svg" alt="DRDebug: Don’t Regenerate, Debug. Repair near-miss accelerator kernels from failure evidence." width="100%">
  </picture>
</p>

<p align="center"><strong>从失败证据出发，定向修复接近正确的加速器算子。</strong></p>

<p align="center">
  <a href="https://shenxiu-wu.github.io/DRDebug/"><img src="https://img.shields.io/badge/Website-DRDebug-6a50a8?style=flat-square" alt="Project website"></a>
  <a href="#安装-skill"><img src="https://img.shields.io/badge/Skill-Native_Host-6a50a8?style=flat-square" alt="安装 DRDebug Skill"></a>
  <a href="https://arxiv.org/abs/2608.02712"><img src="https://img.shields.io/badge/arXiv-2608.02712-9b5769?style=flat-square" alt="Paper on arXiv"></a>
  <a href="papers/2026-dont-regenerate-debug/artifact/"><img src="https://img.shields.io/badge/Artifact-Frozen-586c88?style=flat-square" alt="Frozen paper artifact"></a>
  <a href="papers/2026-dont-regenerate-debug/dataset/"><img src="https://img.shields.io/badge/Dataset-27_tasks-5c7e68?style=flat-square" alt="27 paper tasks and input cases"></a>
</p>

<p align="center">
<strong>Yansong Sun</strong><sup>1,*</sup> · <strong>Shenxiu Wu</strong><sup>1,*</sup> · <strong>Siyuan Chen</strong><sup>1</sup> · <strong>Runlin Hou</strong><sup>1</sup> · <strong>Junhao Qiu</strong><sup>1</sup><br>
<strong>Junming Cao</strong><sup>2</sup> · <strong>Shudi Shao</strong><sup>2</sup> · <strong>Zhichao Lu</strong><sup>1,†</sup> · <strong>Qingfu Zhang</strong><sup>1,†</sup>
</p>

<p align="center">
<sup>1</sup> 香港城市大学计算机科学系<br>
<sup>2</sup> 华为技术有限公司应用软件工程实验室
</p>

<p align="center"><sup>*</sup> 共同第一作者 · <sup>†</sup> 通讯作者</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong> &nbsp; | &nbsp;
  <a href="#安装-skill">安装</a> · <a href="#工作方式">方法</a> · <a href="#研究工作">研究</a> · <a href="#cannbench-成果">CANNBench</a> · <a href="#复现与核验">复现</a>
</p>

DRDebug 从一个接近正确的算子出发，依据失败证据修复已有实现，再由确定性验证决定修改能否被接纳。本仓库集中展示 DRDebug 系列研究的论文、公开代码、实验记录与评测成果。

| 从哪里开始 | 可以看到什么 |
| --- | --- |
| [项目主页](https://shenxiu-wu.github.io/DRDebug/) | 修复示例、系统原理与交互式成果展示 |
| [安装 Skill](#安装-skill) | 由已有宿主智能体驱动的确定性修复闭环 |
| [论文与方法](#研究工作) | 精度修复研究、冻结源码与实验记录 |
| [Cumsum 操作示例](examples/cumsum/README.zh-CN.md) | 重放候选，或按原生宿主协议进行有参考修复 |
| [复现与核验](#复现与核验) | 在 CPU 上复现论文表格，核对论文任务与测例 |

## 工作方式

**大语言模型提出定向补丁，确定性引擎负责失败复现、诊断取证与结果验证。** 检查点与有界的修复尝试在循环中保留已有进展。

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/readme/repair-loop-dark.svg">
    <img src="assets/readme/repair-loop.svg" alt="Near-miss kernel → reproduce → diagnose → patch → validate → accepted repair. Failed validation returns to diagnosis. The engine owns evidence and validation; the large language model proposes patches." width="100%">
  </picture>
</p>

修复接纳要求完整数值测试和完整性核验。可以继续阅读[公开方法源码](papers/2026-dont-regenerate-debug/artifact/code/debug_agent_v5/README_RELEASE.md)或[论文详细架构图](website/assets/research/architecture-overview.svg)。

## 安装 Skill

**同一个确定性引擎，使用当前宿主的模型和工具。** [`drdebug`](skills/drdebug/SKILL.md) 将取证、完整验证、检查点／回滚与公开精度知识种子库带入已有 Agent 工作流，并附带 10 个历史参考案例，其中 7 个包含 AscendC 内核。

```bash
git clone https://github.com/Shenxiu-WU/DRDebug.git
cd DRDebug
./install.sh --codex    # 可选宿主；也可使用 ./install.sh --claude
```

其他兼容宿主，包括使用 Kimi 的 Agent，可选择其实际 skills 父目录：

```bash
./install.sh --dest /path/to/your-agent/skills
```

安装器追加 `drdebug/` 子目录，由宿主加载其中的 `SKILL.md`，用自己的工具读取请求、编辑文件。Codex 与 Claude Code 快捷入口分别选择 `~/.agents/skills/drdebug` 和 `~/.claude/skills/drdebug`。`--project /path/to/project` 选择项目级安装，`--dry-run` 预览位置，`--force` 更新时先备份。安装后的 runtime 不依赖原 checkout。

**安装仅需 Python 3.10+ 与标准库。** 数值执行还需要已有任务、Linux + 昇腾 NPU、兼容的 CANN/AscendC、Torch/torch_npu、NumPy 与构建工具。Claude Code、Codex、使用 Kimi 的 Agent 或其他兼容宿主任选其一；编码智能体 CLI 不是内部依赖，Skill 不启动嵌套模型 worker。模型密钥、权限与 SSH 由宿主管理。

当前流程：**`doctor → prepare → next → 宿主编辑 → submit → … → verify`**。`next` 返回确定性证据、`request_id`、隔离的 `candidate_dir` 和需要完整读取的 `instructions` 文件。宿主只编辑允许的候选文件，向其外部的 `result_file` 写结果 JSON，包含 `outcome`（`completed` 或 `failed`）、真实 `final_response` 和要求的元数据尾部。`submit --request-id ID --result RESULT_JSON` 验证并推进任务。接纳由引擎决定，宿主完成不等于数值通过。

新 schema-4 工作区上限为 **5 次尝试、86,400 秒总墙钟时间**，包括等待宿主与确定性命令执行；不伪造模型 turns。旧工作区保留历史证据，原 `run` 命令已退役。完整命令和迁移边界见[安装与首次修复指南](docs/skill-guide.zh-CN.md)。

可以对宿主说：

> 使用 drdebug 和你当前的模型与工具，检查我的昇腾环境，创建独立工作区，按 next/submit 请求修复这个算子，再独立验证实际被接纳的源码。

[Skill 操作说明](skills/drdebug/SKILL.md) · [运行与任务规范](skills/drdebug/references/runtime.md)

### Cumsum：原生宿主验证

[全新安装验证记录](examples/cumsum/native-host-validation.zh-CN.md)覆盖 **Claude Code、Codex 与 Kimi Code**。三者均从原始 **6/10** 基线出发，实际完成宿主请求与提交、引擎接纳 **10/10 紧凑集 + 51/51 完整集**，并通过 **51 个完整集用例的逐例新进程独立验证**。记录包含实际 CLI 版本、K3／1M 配置、源码哈希和分开的验收收据；运行在 Ascend910B4，确定性开启。

这些运行使用单独提供的公开修复实现作为只读参考，属于**参考引导验证**，用于核对安装与原生协议，不衡量无参考的独立发现能力。精确接纳源码与[机器记录](examples/cumsum/native-host-validation.json)一并保留。

此前的[固定候选重放与引导修复记录](examples/cumsum/validation.zh-CN.md)保持不变，分别呈现。

## 研究工作

### Don’t Regenerate, Debug

*A Domain-Specific Agent for Repairing Near-Miss Accelerator Kernels* · 2026

[论文摘要](https://arxiv.org/abs/2608.02712) · [全文 PDF](https://arxiv.org/pdf/2608.02712) · [代码与结果](papers/2026-dont-regenerate-debug/artifact/) · [任务与测例](papers/2026-dont-regenerate-debug/dataset/) · [初始内核与 wrapper](papers/2026-dont-regenerate-debug/artifact/inputs/)

研究采用来自 [NPUKernelBench](https://github.com/weich97/NPUKernelBench/tree/master) 的 **27 个固定近成功算子**，使用 `Qwen3.8-max-preview` 进行受控比较：

| 指标 | 重新生成 · 三次独立尝试 | 定向调试 · 单个连续任务 |
| --- | ---: | ---: |
| 成功算子 | 11 / 27 · 40.7% | **18 / 27 · 66.7%** |
| 每成功产出一个算子的总 token | 397.50M | **28.44M** |

**每成功产出一个算子的 token 开销减少 92.8%。** 统计先汇总所有运行消耗的 token，再除以成功算子数，包含失败运行、缓存创建与缓存读取。M 表示百万 token；一个调试任务可包含多轮修复。

数据来源：[调试结果](papers/2026-dont-regenerate-debug/artifact/results/rq1_rq2_debug_agent_qwen_27.csv) · [重新生成结果](papers/2026-dont-regenerate-debug/artifact/results/rq1_rq2_regenerate_qwen_81.csv) · [Token 汇总](papers/2026-dont-regenerate-debug/artifact/results/token_accounting_summary.json)。

## CANNBench 成果

我们将这套修复思路延伸到更广泛的社区评测，用标准集与隐藏集共同检验正确性。以下成果与论文中的 NPUKernelBench 实验分别报告。

**2026 年 10 月 15 日快照** · Official-Tasks 1.1（任务包 1.1.2）· A3 / Ascend 910C · 标准集 + 隐藏集。

| 范围 | 测试用例通过数 | 通过率 | 按通过用例数排名 |
| --- | ---: | ---: | ---: |
| 全量 53 个算子 | **5,278 / 5,300** | **99.58%** | **第 1** |
| Level 4 · 难度最高的 8 个任务 | **799 / 800** | **99.88%** | **第 1** |

**8 个 Level 4 算子中，7 个通过全部测试。** 另一个算子在其 100 个标准与隐藏测例中仅剩 1 例未通过。

官方聚合名为 **Debug-Agent Claude Code Kimi K3**。该快照包含 170 个公开聚合方案；通过率比较仅纳入完整覆盖测试集的方案（全量 15 个、Level 4 为 19 个），我们的聚合方案在这两项通过率上也均排名第一。

[官方排行榜](https://cannbench.com/leaderboard/official-tasks) · [日期快照与统计口径](website/data/leaderboard-2026-10-15.json) · [交互式结果明细](https://shenxiu-wu.github.io/DRDebug/#results)

## 复现与核验

以下检查仅需 **Python 3.10+ 与标准库**。在仓库根目录运行：

```bash
git clone https://github.com/Shenxiu-WU/DRDebug.git
cd DRDebug
python3 -B tools/reproduce_paper.py
python3 -B papers/2026-dont-regenerate-debug/dataset/verify_dataset.py
python3 -B tools/verify_paper_inputs.py
```

- **论文表格**：核对冻结包，在临时副本中重算统计，输出到 `reproduced/tables.md`、`summary.json` 及相关分析文件。再次运行时，使用 `--output PATH` 指定新目录。
- **初始候选源码**：按历史哈希核对 27 个任务的 289 个原始文件，包括 181 个内核文件；只检查语法与测例，不执行任务代码。
- **任务数据**：按照冻结清单核对 27 个参考实现、347 条精简和 1,134 条完整输入测例规格；此检查读取文件，不执行算子。

<details>
<summary>公开材料范围与执行条件</summary>

公开材料支持 CPU 统计复现、方法源码阅读与任务数据核验，并已补齐 27 个逐字节核验的初始候选任务与历史算子参考案例。真实数值重跑仍需兼容运行环境，提出新补丁使用所选宿主的模型与工具，详见论文包的[已知限制](papers/2026-dont-regenerate-debug/artifact/KNOWN_LIMITATIONS.md)。

独立的[测例补充包](papers/2026-dont-regenerate-debug/dataset/README.md)提供参考实现与输入规格；81 份参考/测例文件均已恢复为与历史冻结哈希一致的原始字节，来源与依赖记录一并提供。运行参考实现可能需要 PyTorch、`torch_npu` 与兼容的昇腾硬件。

</details>

<details>
<summary>运行精度示例或本地预览网站</summary>

在 CPU 上复现横幅中的 FP16 舍入示例：

```bash
python3 -B website/examples/fp16-reduction-demo.py
```

6 组构造检查从 3/6 通过变为 6/6。示例用于说明数值原因和修复思路；真实算子的接纳仍需完整设备验证。

本地预览支持中英双语和深浅主题的项目网站：

```bash
python3 -m http.server 8000 --directory website
```

打开 <http://localhost:8000>。网站直接在 [`website/`](website/README.md) 中维护。

</details>

## 仓库导航

| 路径 | 内容 |
| --- | --- |
| [`papers/2026-dont-regenerate-debug/artifact/`](papers/2026-dont-regenerate-debug/artifact/) | 冻结的方法源码、实验记录与分析脚本 |
| [`papers/2026-dont-regenerate-debug/dataset/`](papers/2026-dont-regenerate-debug/dataset/) | 参考实现、输入测例、来源记录与校验工具 |
| [`skills/drdebug/`](skills/drdebug/) | 可安装 Skill、便携启动器与任务规范 |
| [`examples/cumsum/`](examples/cumsum/) | 已修复候选精度重放与独立的实时修复教程 |
| [`tools/`](tools/) | CPU 统计复现入口 |
| [`website/`](website/) | 项目网站与带日期的 CANNBench 快照 |

每篇论文独立存放在 `papers/` 下。论文已发布版本保留各自的标签，补充源码在归档的公开变更记录中说明；网站与 Skill 在仓库中同步维护。

## 引用

```bibtex
@misc{sun2026dontregeneratedebug,
  title={Don't Regenerate, Debug: A Domain-Specific Agent for Repairing Near-Miss Accelerator Kernels},
  author={Yansong Sun and Shenxiu Wu and Siyuan Chen and Runlin Hou and Junhao Qiu and Junming Cao and Shudi Shao and Zhichao Lu and Qingfu Zhang},
  year={2026},
  eprint={2608.02712},
  archivePrefix={arXiv},
  primaryClass={cs.SE},
  url={https://arxiv.org/abs/2608.02712}
}
```

## 许可证

项目自有代码和网页资产使用 [MIT 许可证](LICENSE)。第三方代码、评测材料与论文图片保留各自的署名和许可条款，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 及各论文包中的许可说明。
