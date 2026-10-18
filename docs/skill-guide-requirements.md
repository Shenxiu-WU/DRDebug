# DRDebug execution requirements / 执行环境要求

[English guide](skill-guide.md) · [中文指南](skill-guide.zh-CN.md)

This file documents requirements; **it is not a pip requirements file**. Do not pass it to `pip install -r` or install every dependency from the archived experiment harness.

本文件是环境说明，**不是 pip requirements 文件**；不要用于 `pip install -r`，也不要将历史实验 harness 的全部依赖当作当前 Skill 的安装清单。

| Layer / 层次 | Required / 要求 | Not included / 不由安装器提供 |
| --- | --- | --- |
| Install and CLI / 安装与 CLI | Python 3.10+ standard library / 标准库 | No provider SDK or nested agent CLI / 无 provider SDK 或嵌套 Agent CLI |
| Numerical execution / 数值执行 | Linux, Ascend NPU + driver, compatible CANN/AscendC, CMake 3.16+, C++ compiler and Python headers, matching Torch/torch_npu, NumPy, pybind headers | Driver/toolkit, binary wheels, device reservation / 驱动、工具链、二进制包、设备预约 |
| Host repair / 宿主修复 | One existing host's model, file-editing and command tools / 一个已有宿主的模型、文件编辑与命令工具 | Model credentials, permissions, SSH / 模型密钥、权限、SSH |

Use the [Ascend compatibility table](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md) for the actual Python/Torch/torch_npu/CANN/driver combination. There is no universal package pin across architectures and toolkits. Keep an isolated environment active, with launcher Python and `python3` on `PATH` identical. The archived extension build fixes `_GLIBCXX_USE_CXX11_ABI=1`; actual ABI compatibility must pass `doctor`. Prefer Torch's bundled pybind headers.

具体版本组合依照[昇腾兼容表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)，不同架构和工具链不共享一份通用 pin。使用隔离环境，使启动器与 `PATH` 上的 `python3` 一致；历史扩展固定 `_GLIBCXX_USE_CXX11_ABI=1`，实际 ABI 兼容性须通过 `doctor`。优先采用 Torch 自带的 pybind 头文件。

Claude Code, Codex, a Kimi-backed agent and other compatible hosts are alternatives. The host handles its own model configuration and any remote transport. `doctor` checks deterministic execution prerequisites, not a coding-agent CLI, provider account or host-model tool chain. An installation or probe pass is not full numerical task acceptance.

Claude Code、Codex、使用 Kimi 的 Agent 与其他兼容宿主任选其一；宿主处理自己的模型配置和远程传输。`doctor` 检查确定性执行依赖，不检查编码智能体 CLI、provider 账号或宿主模型工具链。安装成功或探针通过不等于任务完整数值验收通过。
