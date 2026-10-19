<a id="cumsum-replay-a-candidate-or-repair-with-your-host"></a>

# Cumsum: replay a candidate or repair with your agent

[English](README.md) · [简体中文](README.zh-CN.md)

Use either **model-free independent verification of the saved candidate**, or **reference-guided repair by your coding agent from the original failing task**. Both retain the original reference, wrapper, 10 compact cases, 51 full cases and dtype-specific tolerances. The paper archive and historical validation receipts remain unchanged.

| Route | Starting kernel | Model / agent | What to check |
| --- | --- | --- | --- |
| Saved-candidate replay | Eight hash-pinned repaired native sources | No model | Independent clean build and original numerical/integrity checks |
| Agent-driven repair | Original paper input, with repaired source separately supplied as a read-only example | Your current model and tools | Real request → agent patch → engine acceptance → independent verification |

## Before running

Follow the [current installation guide](../../docs/skill-guide.md) and [execution requirements](../../docs/skill-guide-requirements.md). Numerical commands run on a Linux Ascend machine/container with the compatible driver/CANN, ABI-compatible Torch/torch_npu, NumPy and build tools. Use one isolated Python environment for the launcher and `python3` on `PATH`. The installer itself needs only Python 3.10+ and its standard library.

Choose Codex, Claude Code, Kimi Code or another compatible coding agent. It uses its existing model and tools and manages credentials, permissions and remote SSH. DRDebug neither launches a nested coding agent nor chooses a provider. Saved replay calls no model.

Run commands from the repository root, choosing new input/workspace paths, a free device ID exposed to the execution environment and the installed compiler's real SoC identifier. `--npu ID` selects an exposed device through `ASCEND_RT_VISIBLE_DEVICES`; the adapter uses logical `npu:0`. Containers may remap host physical IDs, so check enumeration inside the execution environment and record the host ID separately. Replace the illustrative `0` and `YOUR_SOC_VERSION` below.

```bash
./install.sh --dest "$HOME/.local/share/drdebug-example-skills"
DRDEBUG_SKILL="$HOME/.local/share/drdebug-example-skills/drdebug"
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" doctor --runtime-probe \
  --npu 0 --soc YOUR_SOC_VERSION --strict
python3 examples/cumsum/prepare.py --check
```

For an existing Codex/Claude Code/Kimi Code/custom installation, use its actual installed Skill path instead. Source assets come from this checkout; the installer does not copy the entire example or all paper tasks.

<a id="1-replay-the-saved-repaired-candidate"></a>

## 1. Independently replay the saved candidate

This does not ask your agent to repair anything or inherit an old success status:

```bash
python3 examples/cumsum/prepare.py --mode replay --output /path/to/cumsum-replay-input
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" prepare /path/to/cumsum-replay-input \
  --output /path/to/cumsum-replay --op-name 5_Cumsum --determinism on
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" verify /path/to/cumsum-replay \
  --npu 0 --soc YOUR_SOC_VERSION --casewise --json
```

Judge the actual final `verification/<id>/summary.json` and CLI exit status. A full pass requires **10/10 compact, 51/51 full, all integrity/native-binding/AST checks and complete nonfinite-guard coverage**, plus 51 completed matching fresh-case processes with `--casewise`. These are acceptance targets, not a claim that the commands above were run in your environment. `progress.json` is a process snapshot, not the final verdict.

Each fresh-case process regenerates the original seed-0 full input collection and selects its original index. Full-suite and fresh-process outcomes are separate measurements; neither changes reference or tolerance.

<a id="2-guided-repair-with-the-native-host-protocol"></a>

## 2. Reference-guided repair with your agent

The original failing task remains the repair target. `--reference-root` supplies `examples/cumsum/repaired` separately as a read-only AscendC example. Disclose that reference: this is **guided repair**, not unaided rediscovery. A separate study without the reference may omit that option in another new workspace and consume more budget.

```bash
python3 examples/cumsum/prepare.py --mode live --output /path/to/cumsum-live-input
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" prepare /path/to/cumsum-live-input \
  --output /path/to/cumsum-live --op-name 5_Cumsum --determinism on \
  --reference-root "$PWD/examples/cumsum/repaired"
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" next /path/to/cumsum-live \
  --npu 0 --soc YOUR_SOC_VERSION --max-attempts 5 --deadline-sec 86400 --json
```

For `state: "awaiting_host"`, read the complete file at `instructions`, its evidence, `request_id` and `candidate_dir`. Your coding agent follows the full diagnostic method with its own tools, edits only allowlisted isolated candidate/audit files and writes the actual response to `result_file`, outside candidate sources. Do not edit the active task, protected inputs, request metadata, engine reports or history.

The result JSON must contain `outcome` (`completed` or `failed`) and the actual `final_response` ending in the full prompt's factual `[ENGINE_ATTEMPT_METADATA]` footer; `error` must be nonempty for `failed` and is optional for `completed`. No success/precision values or fabricated turns are needed. `completed` means the agent finished its work, not that tests passed.

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" submit /path/to/cumsum-live \
  --request-id REQUEST_ID --result /path/from/result_file/host-result.json \
  --npu 0 --soc YOUR_SOC_VERSION --json
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" status /path/to/cumsum-live
```

Replace both placeholders with that packet's actual values. Read engine validation and repeat `next → agent edit → submit` while nonterminal. The saved ceilings are five attempts and 86,400 seconds total wall time, **including time spent waiting for the agent and running deterministic commands**; resume retains them. Do not delete history, change budgets, reopen terminal state or invent model-turn counts. See [recovery rules](../../skills/drdebug/references/runtime.md#first-failure-or-resume).

Acceptance requires `state: "terminal"`, `terminal: true`, `engine_status.session_outcome: "success"`, `returncode: 0` and `accepted_checkpoint.ready: true`. Independently verify the actual accepted source:

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" verify /path/to/cumsum-live \
  --npu 0 --soc YOUR_SOC_VERSION --casewise --json
```

An agent response or best-but-partial checkpoint is not a pass. A baseline already passing all gates may be accepted without any agent patch; do not manufacture an attempt. Legacy workspaces preserve their evidence for `status` and supported independent `verify`; current repair uses a new schema-4 workspace. The old `run` command is retired.

<a id="native-host-measurements"></a>

## Agent-driven repair measurements

The [fresh-install agent-driven repair record](native-host-validation.md) follows the original 6/10 compact baseline through a real repair submission, 10/10 compact + 51/51 full acceptance, and a separate clean-workspace verification with 51 fresh-case processes. It lists coding agents actually tested, K3 / 1M configuration, source identities and the supplied public reference. See the [machine-readable record](native-host-validation.json) and [captured source fixtures](validation-sources/README.md).

## Historical results, kept separately

The [versioned validation record](validation.md) and [machine-readable receipts](validation.json) predate native `next`/`submit`. They retain the measured saved replay and reference-guided Kimi repair on **Ascend910B4 / CANN 9.1.0 / Torch 2.9.0 / torch_npu 2.9.0.post6**, determinism on. The historical guided run went from **6/10** to **10/10 compact + 51/51 full** in one attempt / 38 actual turns, followed by **51/51** fresh-case checks. It used an earlier adapter; the identical output was later verified with the added nonfinite guard.

Those receipts and model-specific timings remain historical evidence, **not a fresh pass claim for the current agent-driven workflow**. No new NPU acceptance is implied by installation, a pending request or adapter CPU tests. Establish and retain your own source-bound numerical results.

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
