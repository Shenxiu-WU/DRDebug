---
name: drdebug
description: Repair existing AscendC accelerator kernels from build, runtime or numerical failure evidence with the public DRDebug engine. Use for near-miss kernel repair and its environment setup, with full validation deciding acceptance.
---

# DRDebug

Use the released paper engine to repair a user-supplied AscendC task in a separate workspace. The engine owns reproduction, diagnostic forensics, knowledge retrieval, bounded attempts, full numerical validation, integrity checks and checkpoint/rollback. The model proposes kernel patches; its own success claim is not an acceptance signal.

## Locate and check

Resolve paths from this skill's directory, called `<skill-dir>` below. Never assume the current directory or a particular home-directory installation path.

```bash
python3 <skill-dir>/scripts/drdebug.py doctor
```

Read [the runtime and task contract](references/runtime.md) when preparing a task or resolving missing dependencies. Installation bundles the public paper source and recovered reference examples; a source checkout also works. A real repair requires Linux, an Ascend NPU, compatible CANN/AscendC, PyTorch/torch_npu, and an authenticated Claude Code worker. Codex can orchestrate this skill; the frozen engine's worker backend is Claude Code. Do not present a CPU check as NPU validation.

## Prepare a repair

Use the user's existing task: `model.py`, `model_new_ascendc.py`, `kernel/` C++ sources and pybind binding, and the public input-case specification. Provide at least one explicitly identified complete `.json.bak` / `.json.full` case collection. It may equal the compact set; the launcher does not invent additional coverage. Use the case JSON stem as `--op-name` (for example `3_Add`, even when the directory is named `003_Add`).

```bash
python3 <skill-dir>/scripts/drdebug.py prepare /path/to/task \
  --output /path/to/new-repair-workspace --op-name MyOperator
```

Preparation copies the task and a runtime into a new workspace. Preserve the user's originals and existing experiment records. Do not overwrite an existing workspace to start again; use `status` and resume it, or choose a new output path. If the user provides independent AscendC reference examples, pass `--reference-root /path/to/reference-cases` at preparation time.

The public adapter retains constructive Phase A → B → C: establish a reference specification before auditing the candidate. When historical `archive_tasks` examples are unavailable, use the task's read-only reference implementation and the bundled AscendC documents. This is a documented portability adaptation, not a reconstruction of the original study's assets. The ten historical reference examples are included with installation; seven contain AscendC kernels. The released 102-entry KB fixture is an explicitly identified, read-only seed.

## Run and inspect

Choose the device, the actual CANN SoC identifier and the budget from the user's request and available resources. Replace `YOUR_SOC_VERSION` with the installed toolkit's identifier. This bounded example uses device 0 and the user's already configured worker model:

```bash
python3 <skill-dir>/scripts/drdebug.py run /path/to/new-repair-workspace \
  --npu 0 --soc YOUR_SOC_VERSION --max-attempts 5 --max-turns 80 --deadline-sec 1800
python3 <skill-dir>/scripts/drdebug.py status /path/to/new-repair-workspace
```

An optional `--model MODEL` chooses an already configured Claude-compatible model. Keep credentials in the user's existing authentication or environment; never write them into the skill, task, prompt, logs or source control. Do not select credentials, install system packages, change driver versions or submit to an external benchmark without task authorization. If an execution permission is denied, report it as blocked execution; do not bypass the permission system.

Only the copied task's `kernel/` source is a repair target. References, Python wrappers, case inputs, tolerances, validation scripts and verification records remain protected. Do not skip cases, tune to case identities, replace kernel computation with a Python fallback or relax acceptance criteria. Keep diagnostic evidence distinct from final full-suite validation.

## Report the outcome

Read the engine-generated status, summary, trace and validation records, rather than inferring success from a process exit or model text. Report the workspace path, failure/root-cause evidence, kernel change, complete validation result and any remaining failures. Keep timeout, provider/authentication failure, missing environment, integrity failure and numerical failure distinct. A stopped or unverified run is not a repaired kernel. Preserve its records for inspection and resume only within the user's remaining scope and budget.
