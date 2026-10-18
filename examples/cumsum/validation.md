# Cumsum validation record

[English](validation.md) · [简体中文](validation.zh-CN.md) · [Source identities and receipts](validation.json)

Measured on **18 October 2026** with the original paper reference, wrapper, **10 compact cases and 51 full cases**, seed 0 and unchanged dtype tolerances.

| Check | Compact | Full | Fresh full-case processes | Model |
| --- | ---: | ---: | ---: | --- |
| Saved repaired candidate with the added nonfinite guard | **10/10** | **51/51** | **51/51** | None |
| Guided repair from the original **6/10** baseline | **10/10** | **51/51** | **51/51** | Kimi `k3[1m]` |
| Latest adapter: default baseline and best-checkpoint acceptance | **10/10** | **51/51** | Covered by the separate guarded replay | None |

The latest default-budget `run` accepted the already-correct baseline with **zero worker calls**, source-bound best attempt **−1**, complete compact/full guard receipts and a successful, unlocked post-status audit. It did not repeat the 51 fresh-case processes. For a zero-attempt baseline, legacy `engine_status.objective_success` / `reportable_success` may remain false because they describe repair-final-validation; the acceptance evidence is `run` exit 0, `accepted_checkpoint.ready: true` and the unlocked terminal success status. Independent `verify` remains a separate numerical verdict.

The guided run used a fresh Skill installation and original-input workspace, with the published repaired implementation supplied separately through `--reference-root`. **One attempt / 38 turns** produced the accepted candidate. Worker execution took **6 min 25 sec**; the launcher including baseline and engine checks took **15 min 5 sec**. Independent clean-build verification then passed the complete collection and 51 fresh single-case processes. This is guided repair with a public reference, rather than unaided rediscovery.

That model run preceded the new nonfinite guard. Its eight output source files are byte-identical to the saved public implementation. A later model-free clean build verified those exact bytes through both real verifier CLI and import entrypoints, including all 51 fresh cases with the added guard. The 55 numerical processes have bound startup, actual native-extension loading, wrapper binding and guard receipts. Original C++ source-integrity and Python wrapper AST checks also passed. No later paid model repair is claimed.

## Environment and installation

**Linux aarch64 · Python 3.11.14 · Ascend910B4 · CANN 9.1.0 · Torch 2.9.0+cpu · torch_npu 2.9.0.post6 · NumPy 1.26.4.** Deterministic algorithms were on with `warn_only=False`, ABI=1 and a working NumPy/Torch bridge. The dependency recipe is in the [installation guide](../../docs/skill-guide.md#use-a-matching-isolated-python-environment).

The isolated dependency environment was freshly installed for the initial saved replay. The guided run and later checks reused that environment with separate Skill installations and workspaces; they were not additional package installations. No earlier task history, model session or native binary was imported into the new tasks.

## What the adapter corrects

The public adapter binds completed full results to exact source/protected-input hashes when selecting the best checkpoint, checks dependencies and deterministic startup, and creates a checked regular result directory before first baseline validation. The extra output guard requires matching NaN, positive-infinity and negative-infinity positions before the frozen finite comparator; finite thresholds are unchanged. These are explicit runtime-copy adaptations. The paper archive stays frozen.

Use final `verification/<id>/summary.json` together with the command exit status. `progress.json` is a process snapshot and can retain an initial failure/default status after completion. The [JSON record](validation.json) pins the original 15 files, repaired eight files, tested runtime versions and evidence identities.

## Scope

This example qualifies the original compact/full cases in the measured environment. Arbitrarily larger shapes, nonzero-`storage_offset` views and general empty/overflow/UB-capacity handling are outside its measured scope. The live model may produce a different trajectory or fail; the model-free [saved-candidate replay](README.md#1-replay-the-saved-repaired-candidate) lets you independently check the published result on your own machine.
