---
name: drdebug
description: Repair existing AscendC accelerator kernels from build, runtime or numerical failure evidence with the public DRDebug engine. Use for near-miss kernel repair and its environment setup, with full validation deciding acceptance.
---

# DRDebug

Repair an existing AscendC kernel in a separate workspace using the public paper engine and its portable adapter. The model proposes patches; the engine owns reproduction, forensics, validation and rollback. The frozen paper source stays unchanged. Only newly prepared runtime copies receive the documented checkpoint correction, explicit startup configuration and nonfinite-output guard.

## Check and prepare

Resolve `<skill-dir>` from this file's directory, not from the current working directory. Read [the runtime and task contract](references/runtime.md) for dependencies, resource paths and recovery rules.

```bash
python3 <skill-dir>/scripts/drdebug.py doctor \
  --npu 0 --soc YOUR_SOC_VERSION --runtime-probe --strict
python3 <skill-dir>/scripts/drdebug.py prepare /path/to/task \
  --output /path/to/new-repair-workspace --op-name MyOperator --determinism on
```

Choose the actual device and installed CANN SoC identifier; the example values do not reserve a device. Numerical work requires Linux, an Ascend NPU, matching CANN/Torch/torch_npu, NumPy, ABI-compatible build tools and one active Python environment. The preflight checks the real ABI/NumPy bridge and can execute a tiny built-in NPU probe; it is not full task validation. Append `--no-worker` for saved-candidate checks without Claude Code. Live repair still requires an authenticated Claude Code worker even when Codex orchestrates this skill.

The task must contain `model.py`, `model_new_ascendc.py`, native `kernel/` sources including pybind, compact `<op>.json` and an explicitly identified full `.json.bak` / `.json.full` collection. Use the JSON stem as `--op-name`. Preparation never overwrites an existing output. New schema-3 workspaces default to determinism on and fix the mode before managed Python entrypoints, with `warn_only=False` and process receipts. Legacy schema-1/2 stays off; use a new workspace to change mode or adopt this adapter, keeping old evidence intact.

New preparation records `nonfinite_guard` provenance in the workspace manifest. This public-adapter rule first requires reference/candidate NaN, positive-infinity and negative-infinity masks to match, then applies the original finite-value comparator and unchanged tolerances. Matching nonfinite reference values are allowed. An existing workspace without this manifest field, including earlier schema 3, is not patched in place; prepare a new workspace and retain the old evidence. See [the output contract](references/runtime.md#nonfinite-output-contract).

Use supplied independent examples with `--reference-root` when relevant. Otherwise follow the bundled historical examples, protected task reference and [public resource map](references/runtime.md#public-resource-map). Retain constructive Phase A → B → C. Skip the explicitly unbundled legacy API/canonical-KB paths; use the engine's injected public KB summary rather than inventing missing resources.

## Run and inspect

```bash
python3 <skill-dir>/scripts/drdebug.py run /path/to/new-repair-workspace \
  --npu 0 --soc YOUR_SOC_VERSION --max-attempts 5 --max-turns 240 \
  --worker-timeout-sec 21600 --deadline-sec 86400
python3 <skill-dir>/scripts/drdebug.py status /path/to/new-repair-workspace
```

These are generous fresh-task ceilings, not minimum consumption or a guarantee. Resume preserves the saved budget and deterministic mode; older budgets without a worker-timeout field retain their implicit 600-second cap. Read [first failure or resume](references/runtime.md#first-failure-or-resume) before touching a stopped task. Do not delete ledgers/events, reset history or reopen terminal state. A new trial must retain its parent/checkpoint and prior budget usage.

Use `--model MODEL` only for the user's configured worker service. Provider variables stay in the launching environment or existing secret manager, outside task files/prompts/logs. Project-only Claude settings do not inherit custom-provider variables stored only in user settings. Confirm provider/tool round trips separately when needed. Shell diagnostics require explicit `--allow-worker-shell` in a trusted isolated workspace; this does not bypass permissions or directory trust.

Only copied `kernel/` sources are repair targets. Reference implementations, wrappers, cases, tolerances and verifier/history files stay protected. Preserve runtime-error evidence before interpreting numerical mismatches. Do not skip failing cases, specialize to case identities, substitute reference computation or relax acceptance.

The portable checkpoint selector binds objective results to source/protected-input bytes and prioritizes completed full results before compact metrics. A best checkpoint may still fail some full cases; it is not a success certificate. Report full improvements even when compact counts tie, and require a source-consistent final checkpoint before claiming the engine accepted a repair.

## Independently verify and report

```bash
python3 <skill-dir>/scripts/drdebug.py verify /path/to/new-repair-workspace \
  --npu 0 --soc YOUR_SOC_VERSION --determinism on --casewise
```

This schema-3 path copies the active kernel to a new verification directory, performs a clean build and the original compact/full comparisons with the added nonfinite guard, checks native binding, original C++ integrity, wrapper AST and source/protected/history integrity, and optionally starts a fresh process per full-case index. It calls no model and changes neither checkpoints nor engine history. Do not run it concurrently with a worker. See [verification details](references/runtime.md#independent-verification) for time limits and reports.

Judge independent verification from its final `verification/<id>/summary.json` and command exit status, including guard coverage. `progress.json` is only a process snapshot and may retain initial failure/default values after completion.

Read actual status, source-bound validation and independent summaries. Report the environment/deterministic mode, kernel identity, root cause/change, both compact and full counts, integrity and remaining failures. Keep timeout, provider/authentication failure, permission denial, environment failure and numerical failure distinct. A stopped, partial or unverified result is not a repaired kernel. The [Cumsum example](https://github.com/Shenxiu-WU/DRDebug/tree/main/examples/cumsum) offers model-free saved-candidate replay and guided live Kimi repair from the original failing kernel, with the published repair supplied separately as a reference. Disclose that reference; do not claim unaided rediscovery or confuse replay with a new model repair.
