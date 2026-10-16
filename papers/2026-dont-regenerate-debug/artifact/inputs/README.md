# Original 27-task near-miss inputs / 原始待修复算子输入

This package restores the original near-miss candidates used by *Don't Regenerate, Debug* ([paper](https://arxiv.org/abs/2608.02712)). It contains all 27 tasks in [the frozen cohort](../manifests/frozen_27_operators.csv), with their initial wrapper/kernel sources, reference programs, and compact/full input-case specifications. These are the original inputs to repair, not the repaired outputs.

本包恢复论文 27 个任务的原始待修复内核、wrapper、参考实现及紧凑/完整测例。所有原始任务文件逐字节匹配历史冻结输入清单；当前数值结果需在用户环境中重新验证。

| Material | Original files | Identity |
| --- | ---: | --- |
| Near-miss `model_new_ascendc.py` wrappers | 27 | 27/27 frozen SHA-256 hashes match |
| Near-miss `kernel/` sources | 181 | 181/181 frozen SHA-256 hashes match |
| Reference `model.py` | 27 | 27/27 frozen SHA-256 hashes match |
| Compact cases | 27 / 347 rows | Original byte and normalized case-set hashes match |
| Full cases | 27 / 1134 rows | Original byte and normalized case-set hashes match |

Each `tasks/<level>/<operator>/` is an input directory for the installed [DRDebug skill](../../../../skills/drdebug/SKILL.md). The wrapper, local `kernel/`, `model.py`, and both JSONL case files remain together. Pass that directory to the skill's `prepare` command, use the JSON filename stem as `--op-name` (for example `3_Add` in `003_Add/`), and choose a separate output workspace. Preserve both case collections. The original engine supplies a `model.json` compatibility alias when the task requires one. Keep this source package unchanged.

每个任务可以独立交给已安装 Skill 的 prepare 入口创建工作副本；请以 JSON 文件 stem 指定算子名，并选择独立输出目录，保留两组测例。紧凑集和完整集是不同的评测口径，不将紧凑集通过描述为完整验证。

## Read-only verification

From the repository root, using Python 3.10+ and its standard library:

```bash
python3 -B tools/verify_paper_inputs.py
```

The verifier checks this package's complete `MANIFEST.sha256`, the selected historical expectations in `FROZEN_INPUT_SHA256.json`, all 27 task identities, candidate/reference syntax, case counts and normalized case-set hashes. It reads source without importing task code, compiling a kernel, invoking an agent, or running an NPU.

## Source identity and limits

The 208 wrapper/kernel files were recovered from the historical May 2026 clean precision-failure input archive. Their same-operator relative paths and bytes match the later July paper input-file manifest. The original generation provider/attempt and historical `precision_failed`/`CLEAN` labels are retained per task in [PROVENANCE.json](PROVENANCE.json). These input identities are separate from the later repaired-output candidate IDs in the paper's result tables.

The 81 reference/case files match the [dataset companion](../../dataset/README.md), including the subsequently recovered original full FusedRope file. The full historical input manifest also described logs and other files; only the necessary task-source/case hash expectations and the original manifest digest are bundled here.

A total of 289 original task files are included. Each task additionally carries `LICENSE`, `LICENSE-APACHE-2.0.txt` and `SOURCE_NOTICE.md` so attribution survives preparation; those 81 redistribution metadata files are additional to the historical input identity. Root README/license/provenance/checksum files are also supplemental.

PyTorch, compatible Ascend hardware, CANN/AscendC, `torch_npu`, a supported execution environment and a configured agent host are required for actual repair and evaluation. Installation and static verification do not establish current numerical success, build compatibility or historical experiment reproduction. This recovery includes no model transcripts, provider settings, machine configuration, repaired outputs or post-paper CANNBench optimization assets.

The project MIT license and NPUKernelBench Apache-2.0 declaration are retained separately; existing file-level notices remain authoritative. See [NOTICE](NOTICE) and the dataset's license provenance. No current upstream revision is claimed as the byte source of these historical task assets.
