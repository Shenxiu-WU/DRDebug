---
name: drdebug
description: Repair existing AscendC accelerator kernels from build, runtime or numerical failure evidence. Use the host agent to propose patches through the public DRDebug engine's native next/submit protocol, with deterministic full validation deciding acceptance.
---

# DRDebug

Use this host agent's own model, tools and conversation to repair an existing kernel. The public CLI runs deterministic preparation, evidence collection and validation; it never launches another coding agent or configures a model provider. Resolve `<skill-dir>` from this file's directory.

## Prepare

Read [runtime and task requirements](references/runtime.md). Use the actual execution machine's device and installed CANN SoC identifier, with an authorized free NPU:

```bash
python3 <skill-dir>/scripts/drdebug.py doctor \
  --npu 0 --soc YOUR_SOC_VERSION --runtime-probe --strict
python3 <skill-dir>/scripts/drdebug.py prepare /path/to/task \
  --output /path/to/new-workspace --op-name MyOperator --determinism on
```

Installation needs Python 3.10+ and the standard library. Numerical work additionally needs Linux, Ascend NPU access, matching CANN/Torch/torch_npu, NumPy and ABI-compatible build tools. `doctor` checks execution prerequisites, not host authentication or model availability. Model credentials, host permissions and any SSH connection belong to the host's existing configuration.

The input must supply `model.py`, `model_new_ascendc.py`, native `kernel/` sources including pybind, compact `<op>.json` and an explicit full `.json.bak` / `.json.full` collection. Use the JSON stem as `--op-name`. Preparation never overwrites an output. New schema-4 workspaces bind deterministic mode, the nonfinite guard and native-host protocol; legacy workspaces retain historical evidence and are available for `status` / supported independent `verify`, not live migration. Prepare a new workspace rather than hot-patching one.

## Native host repair loop

```bash
python3 <skill-dir>/scripts/drdebug.py next /path/to/new-workspace \
  --npu 0 --soc YOUR_SOC_VERSION --json
```

Numerical commands can outlast a host tool's foreground limit. Use that host's background/session mechanism, retain stdout/stderr and the process's final exit status, and poll the same operation until it ends. Do not start another workspace command while it is running or treat a progress file as completion. If an operation was interrupted, inspect `status` and preserved logs before proceeding; an indeterminate action requires attention rather than a blind retry.

1. Read the full file at the returned `instructions` path, plus evidence, `request_id` and `candidate_dir`. A pending response has `state: "awaiting_host"` and `terminal: false`. If it reports a terminal outcome, inspect that outcome and stop requesting patches.
2. Work only on the isolated candidate and the explicitly allowed audit files. Follow the [host handoff contract](references/portable-worker.md) and the request's full method prompt; use real diagnostics and the Phase A → B → C method below.
   Necessary diagnostic builds and probes run in the separate `diagnostics.task` using `diagnostics.execution`; they are already within the authorized repair scope. Refresh that copy from the candidate, follow the probe policy and retain real outputs. These artifacts are not imported or counted as acceptance.

   The two audit paths, relative to `candidate_dir`, are `precision_tuning/precision_audit_<attempt>.md` and `precision_tuning/candidate_kb_entry.json`. Use the full paths in the returned instructions, including the `precision_tuning/` directory.

3. Write the host-result JSON to the returned `result_file` path, outside candidate source files. Set `outcome` to `completed` or `failed`, `final_response` to the actual diagnosis plus the required `[ENGINE_ATTEMPT_METADATA]` footer, and a nonempty `error` for `failed`. `completed` means the host finished its turn, not that the kernel passed.
4. Submit the same pending request exactly as issued:

```bash
python3 <skill-dir>/scripts/drdebug.py submit /path/to/new-workspace \
  --request-id REQUEST_ID --result /path/to/request/host-result.json \
  --npu 0 --soc YOUR_SOC_VERSION --json
```

5. Inspect engine validation, then repeat `next → host edit → submit` until terminal acceptance or another terminal outcome. Never edit the active workspace task, request evidence, engine reports or history to make a submission pass.

Fresh tasks allow at most **5 attempts** and **86,400 seconds total wall time**, including time while the request is pending with the host and deterministic command execution. These are ceilings; acceptance stops early. Retain saved budget and history on resume. There is no adapter-supplied model, nested worker, per-worker turn budget or fabricated host-turn count. The former `run` command is retired; use this protocol on a newly prepared workspace.

The engine checks the saved deadline when advancing the workspace; it does not control or terminate the host's model session. Set the host's own turn/cost/time limits through that host, and preserve its session plus the pending request when a limit is reached.

## Required diagnostic method

- **Phase A — establish the specification before reading the candidate kernel.** Read the protected `model.py`, relevant supplied/bundled reference examples, the public translator guidance and `phase-a-checklist.md`. Produce `[COMPUTATION_DECOMPOSITION]` and `[REFERENCE_IMPL_SPEC]` with API, dataflow, buffers, alignment and forbidden-pattern requirements. Read-only user references come first when supplied; otherwise use bundled history or explicitly record the public reference fallback. Existing valid static specifications may be reused only as the full method prompt permits.
- **Phase B — inspect the whole implementation.** Read tiling, all kernel and helper files and pybind launch logic. Produce `[KERNEL_STEP_TRACE]`. Follow the request's probe policy: retain genuine measurements and required `[L5_PROBE]` / `[INSTRUMENTATION_FINDINGS]` fields, or accurately record an allowed skip or unavailable capability. Never invent probe results.
- **Phase C — compare steps to the specification.** Check API/count arguments, buffer provenance, source lines and measured intermediate values; explicitly compare queues/buffers, reduction initialization, copy alignment, synchronization and forbidden patterns. Explain the root cause before making a focused patch. Keep the original audit fields and acceptance gates.

The engine injects public KB retrieval summaries. Report only IDs actually used; do not rerun KB search or modify the seed. [The public resource map](references/runtime.md#public-resource-map) overrides unavailable legacy API/canonical-KB locations. Read relevant method sections; global scheduling, gates, checkpoints and termination belong to the engine.

Protect references, wrappers, cases, tolerances, verifier/runtime code, ledgers and checkpoints. Do not skip failing cases, specialize to case identities, substitute reference computation or relax validation. Preserve runtime-error evidence before interpreting numerical mismatches.

## Independently verify and report

```bash
python3 <skill-dir>/scripts/drdebug.py verify /path/to/new-workspace \
  --npu 0 --soc YOUR_SOC_VERSION --casewise --json
```

Verification uses a fresh copy of the actual active kernel, clean build, original compact/full collections, nonfinite guard, native binding and integrity checks; `--casewise` adds fresh processes per original full-case index. It invokes no model and changes no engine history. Do not verify concurrently with pending host edits or another workspace operation.

Read the final `verification/<id>/summary.json` and command exit status; `progress.json` is not the final verdict. A best checkpoint may still fail full cases. Acceptance requires `state: "terminal"`, `terminal: true`, `engine_status.session_outcome: "success"`, `returncode: 0` and `accepted_checkpoint.ready: true`; report the separate independent verification as well. Preserve timeout, environment, permission, host/model and numerical failures as distinct outcomes. Report source identity, root cause/patch, compact/full counts, deterministic mode, integrity and remaining failures. A host response, completed submission or partial result is not a repaired kernel.
