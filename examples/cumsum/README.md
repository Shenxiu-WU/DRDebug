# Cumsum: reproduce a repair, or repair it yourself

[English](README.md) · [简体中文](README.zh-CN.md)

This example keeps two tasks distinct: **verify the saved repaired kernel** without calling a model, or **start a guided live repair** with Kimi, the original paper's failing kernel and the published repaired source as a read-only example. Both use the original reference, Python wrapper, 10 compact cases, 51 full cases and dtype-specific comparison thresholds. The paper archive under `papers/` stays unchanged.

| Path | Starting kernel | Model needed? | What it establishes |
| --- | --- | --- | --- |
| Saved-candidate replay | Eight saved repaired native sources | No | Whether this candidate passes the original tests in your environment |
| Guided live repair | Original paper near-miss input; published repaired source provided separately | Yes, configured Claude Code with Kimi | A fresh installation → diagnosis/patch → full acceptance workflow with a public reference |

## Before running

Use a Linux Ascend execution environment. Complete the [installation and dependency guide](../../docs/skill-guide.md) first, and activate the same Python environment for the launcher and `python3` on `PATH`. Actual execution requires an NPU, compatible driver/CANN, CMake, a C++ compiler and ABI-compatible Torch/torch_npu. The copy installer itself needs only Python 3.10+.

The saved candidate was independently verified on **Ascend910B4, CANN 9.1.0, Torch 2.9.0 and torch_npu 2.9.0.post6**, with deterministic algorithms enabled and `warn_only=False`. The [installation guide](../../docs/skill-guide.md#use-a-matching-isolated-python-environment) covers the matching environment and Torch's `+cpu` version string. Other hardware and versions remain unqualified. The [validation record](validation.md) reports saved replay with the added guard and a separate fresh guided Kimi repair; it distinguishes their runtime versions and reused dependency environment. Deterministic mode is part of the tested contract: without it, the original reference may vary enough to affect its own numerical comparison.

Run the following commands from the repository root. Use new paths below; neither preparation command overwrites an existing task or workspace. Replace `YOUR_SOC_VERSION` with the identifier accepted by your installed toolkit and choose a free physical device ID exposed by your execution environment. `--npu ID` selects that ID through `ASCEND_RT_VISIBLE_DEVICES`; the adapter uses logical `npu:0` internally. The validation setup is `--npu 1`, internal logical `0`, with `Ascend910B4`. Replace the example `--npu 0` below with your selected free physical ID.

From a source checkout, install a separate copy of the skill if you have not already done so:

```bash
./install.sh --dest "$HOME/.local/share/drdebug-example-skills"
DRDEBUG_SKILL="$HOME/.local/share/drdebug-example-skills/drdebug"
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" doctor --no-worker --runtime-probe \
  --npu 0 --soc YOUR_SOC_VERSION --strict
python3 examples/cumsum/prepare.py --check
```

If you used a Codex or Claude user installation instead, set `DRDEBUG_SKILL` to that installed `drdebug` directory. This example's source assets come from the checkout; installation does not copy all paper tasks or examples into the skill.

## 1. Replay the saved repaired candidate

This route performs an independent clean build and numerical validation. It does not ask Kimi to repair anything and does not import a prior success status.

```bash
python3 examples/cumsum/prepare.py --mode replay --output /path/to/cumsum-replay-input
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" prepare /path/to/cumsum-replay-input \
  --output /path/to/cumsum-replay --op-name 5_Cumsum --determinism on
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" verify /path/to/cumsum-replay \
  --npu 0 --soc YOUR_SOC_VERSION --determinism on --casewise
```

Require **10/10 compact, 51/51 full, all integrity/AST checks passing**, and, with `--casewise`, 51 completed and matched fresh-process cases. Do not count a staged source tree, clean compile or process exit alone as a numerical pass. Use final `verification/<id>/summary.json` together with the CLI exit status; `progress.json` is a process snapshot and may retain its initial status after completion. Inspect stage logs if verification fails.

The fresh-process check regenerates the original seed-0 input collection in each child and selects the corresponding original case index; it does not change the reference or tolerances. The full-suite replay and fresh-process results are separate measurements. Keep both reports.

## 2. Run a guided live repair with Kimi

Use an authenticated Claude Code CLI and your Kimi provider configuration. Follow the [installation guide](../../docs/skill-guide.md#exercise-the-selected-device-before-repair) for official Claude Code setup and a provider/tool check. The example includes no credentials or provider settings. Your provider must support the selected model; the demonstrated identifier is `k3[1m]`.

This first-run route keeps the original failing kernel as the task and supplies `examples/cumsum/repaired` separately through `--reference-root`. The worker can read the published implementation as an AscendC example while diagnosing and patching the original. This is **guided repair with a public reference**, not an unaided rediscovery of the repair. It exercises fresh installation, live model/tool execution, patching and the original acceptance chain, without inheriting old events or checkpoints. Live Kimi calls can incur provider charges, and the model must still produce a candidate that passes the unmodified tests.

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

`--allow-worker-shell` allows diagnostic commands in a trusted, isolated execution workspace; it does not bypass Claude Code's permissions or directory trust. Review the prepared runtime and trust it interactively if your CLI requires this, following the installation guide. Keep other jobs and important data outside the workspace. The final verification remains independent of model text.

These budgets are ceilings, not expected consumption or a success guarantee. Preserve interrupted or failed workspaces and follow the [resume/new-trial policy](../../skills/drdebug/references/runtime.md#first-failure-or-resume).

The measured fresh guided run began at **6/10**, used **one attempt / 38 turns**, and reached **10/10 compact + 51/51 full**, followed by **51/51** fresh-case checks. Worker time was **6 min 25 sec**; the launcher including baseline and engine checks took **15 min 5 sec**. It used the public reference and an earlier adapter; the same output was then independently checked with the added nonfinite guard. A stochastic LLM run is not guaranteed to follow the same sequence or succeed. To study repair without that reference, omit `--reference-root` in a separate new workspace; that is an independent, potentially more costly research run. The saved-candidate replay is the reproducible acceptance route for the published repaired source.

## What changed in the saved candidate

The observed runtime fault involved an unaligned short FP32 UB-to-UB copy: seven floats occupied 28 bytes. The repaired kernel uses a zero-valued `Adds` operation for that copy. Numerical diagnosis then identified different accumulation/rounding orders. The saved kernel uses dtype/layout branches, including 40 balanced continuous partitions for the `outerSize == 1`, non-last-axis FP16/FP32 path and explicit FP16 roundback. It keeps sequential behavior for multiple outer slices. This is a precision repair, not a performance claim, a case-ID lookup, or proof of the closed reference backend's algorithm.

The original thresholds remain:

| Output dtype | `atol` | `rtol` |
| --- | ---: | ---: |
| FP32 | `1e-4` | `1e-4` |
| FP16 | `1e-2` | `1e-3` |
| BF16 | `5e-2` | `5e-3` |

The original 51 full cases contain 16 FP16, 26 FP32 and 9 BF16 cases. Compact and full are two original collections; do not assume one contains the other. Report both results.

## Source identity and scope

[example.json](example.json) pins all 15 original task files and the eight repaired native sources with SHA-256 and sizes. `prepare.py --check` validates these bytes without importing task code, compiling, using an NPU or contacting a model. `--mode live` copies the original task; `--mode replay` replaces only its eight `kernel/` files. Both write `example-origin.json` in the new input task, recording the selected mode and every staged source hash. Preserve that input directory together with the repair workspace.

The inherited `SOURCE_NOTICE.md` is retained byte-for-byte and describes the original paper input. For a replay task, its kernel-origin statement is superseded by `example-origin.json` and this example's [source notice](SOURCE_NOTICE.md); the reference/input attribution still applies. No old engine state, checkpoint status, compiled extension, model transcript or credentials are imported. See the [checkpoint contract](../../skills/drdebug/references/runtime.md#best-checkpoint-and-acceptance) for how the portable layer binds full results to candidate source.

The [validation record](validation.md) and [machine-readable identities](validation.json) bind the completed checks to their exact source/runtime versions. The original 51 full cases and separate compact set are qualified in the stated environment. Arbitrarily larger shapes, nonzero-`storage_offset` views and general empty/overflow/UB-capacity handling are outside this example's measured scope. Run `verify` to establish your own result.
