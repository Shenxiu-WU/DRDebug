# Runtime and task contract

## What is installed

`install.sh` copies the Skill, public adapters, references and frozen `debug_agent_v5` method source into a self-contained directory:

```text
drdebug/
├── SKILL.md
├── agents/openai.yaml
├── references/
├── scripts/
└── runtime/
    ├── source/debug_agent_v5/
    ├── manifest.json
    └── provenance/
```

Source hashes and provenance identify the installed snapshot. The launcher can also resolve the paper source in a repository checkout. Installation copies files; preparation creates a new task/runtime copy. Neither modifies the paper archive. New preparation records schema 4 and the native-host protocol, with separate original/adapted runtime hashes. Existing workspaces are not upgraded by reinstalling the Skill.

## Public resource map

These paths are relative to prepared `WORKSPACE/runtime`. The resource base is `skills/ascendc/`; historical `.claude/skills/` registrations do not make that directory a resource base or a required host.

| Status | Resource | Use |
| --- | --- | --- |
| Bundled | `archive_tasks/` | Ten historical examples; seven contain AscendC kernels. Read a relevant existing example and record its source. |
| Bundled | `skills/ascendc/ascendc-debug/references/decomposition_examples/` | Read `README.md`, then an existing relevant example. |
| Bundled | `skills/ascendc/ascendc-debug/references/phase-a-checklist.md` | Required Phase A/C field templates. |
| Bundled | `skills/ascendc/ascendc-translator/references/` | Published `dsl2Ascendc.md`, `dsl2Ascendc_compute_vector.md` and `TileLang-AscendC-API-Mapping.md`; not a complete SDK API tree. |
| Bundled, read-only | `experiments/v5_ablation/fixtures/initial_kb_102_cleaned_20260715.json` | Public seed: 18 titled entries plus one metadata object. `102` is historical filename text. Use engine-injected summaries; do not rerun retrieval or write to the seed. |
| Optional | `archive_tasks/user_references/` | Created with `--reference-root`; skip when not supplied. It does not replace bundled examples. |
| Not bundled | `skills/ascendc/ascendc-translator/references/AscendC_knowledge/api_reference/` | Legacy detailed API tree and `INDEX.md`; skip these locations. |
| Not bundled | `skills/ascendc/ascendc-debug/references/precision_knowledge_base.json` | Historical canonical KB; neither the public seed nor `old_precision_knowledge_base.json` reconstructs it. |

This resource map overrides unavailable location instructions in historical method/DSL files. Do not repeatedly probe missing paths, create empty replacements or alias a seed to the canonical name. Use the protected reference, published guidance and engine evidence; report unresolved API-information gaps. Phase A → B → C and required audit fields remain in force.

The repository's `papers/2026-dont-regenerate-debug/dataset/` contains references/cases, and `papers/2026-dont-regenerate-debug/artifact/inputs/tasks/` contains 27 recovered initial task projects. These collections are not copied into the installed Skill. Select a repository task or supply your own.

## Task input

A task must supply:

- `model.py`: the reference `torch.nn.Module`.
- `model_new_ascendc.py`: the existing wrapper calling the native extension.
- `kernel/`: real `.cpp` / `.h` / `.hpp` sources, including pybind binding.
- `<op>.json`: compact JSONL input specifications, each row containing `inputs`.
- At least one explicit full `<op>.json.bak` / `<op>.json.full` collection. Equivalent backups may coexist; conflicting ones are rejected. Compact/full may coincide or be nonnested; neither is rewritten.

Use the JSON stem for `--op-name`: `3_Add` in directory `003_Add/`. The reference-and-case dataset alone has no candidate kernel. Never synthesize a missing reference, wrapper, full collection or passing record. Preparation requires a new output directory and retains the input task.

## Requirements: installation, execution and host

These are separate requirements, not a universal pip requirements file:

| Layer | Requirements |
| --- | --- |
| Installer and public CLI | Python 3.10+ and the standard library; no provider SDK or coding-agent CLI is required. |
| Numerical execution | Linux, available Ascend NPU/driver, compatible CANN/AscendC environment, CMake 3.16+, C++ compiler/Python headers, matching Torch/torch_npu, NumPy and pybind headers. |
| Host repair | A host able to read the Skill/request/evidence, edit the isolated candidate and run or transport the public CLI commands. Its own model, credentials, tools and permissions remain host-managed. |

Claude Code, Codex, Kimi-backed agents and other compatible hosts are alternatives. None is an internal dependency or automatically launched by the Skill. Installation flags select a Skill directory, not an engine model. Remote access/SSH is configured by the host/user; the Skill does not store credentials, open an SSH service or configure a provider.

Use the [Ascend compatibility table](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md) for package/toolkit/driver matching. The launcher interpreter and `python3` on `PATH` must use the same environment because the frozen build also invokes `python3`. Its extension fixes `_GLIBCXX_USE_CXX11_ABI=1`; the selected Torch libraries must be compatible. Torch-bundled pybind headers are preferred, with a separately installed pybind11 package as fallback.

`doctor` checks actual ABI, headers, compiler/toolkit files, interpreter identity and the NumPy → Torch → NumPy bridge in addition to imports/version inventory. It never probes Claude Code, host authentication or a model API. `doctor --runtime-probe --npu ID --soc SOC --strict` adds built-in NPU add and repeated fixed-input FP16 cumsum. The probe does not run the task kernel or establish task acceptance. Native execution and independent verification retain execution preflight.

Choose a free physical device ID exposed to the process and the installed toolkit's actual SoC identifier. `--npu` selects through `ASCEND_RT_VISIBLE_DEVICES`; execution uses logical `npu:0`. Do not infer a compiler SoC identifier from a leaderboard hardware label. Keep packages isolated; use the [guide](https://github.com/Shenxiu-WU/DRDebug/blob/main/docs/skill-guide.md#requirements) instead of modifying global startup hooks/packages.

## Native-host protocol

```bash
python3 <skill-dir>/scripts/drdebug.py next WORKSPACE --npu ID --soc SOC --json
# Host reads instructions and edits the returned isolated candidate_dir.
# Host writes actual final_response + metadata to a result JSON outside candidate files.
python3 <skill-dir>/scripts/drdebug.py submit WORKSPACE \
  --request-id REQUEST_ID --result /path/to/request/host-result.json \
  --npu ID --soc SOC --json
```

`next` advances deterministic engine work until a host request or terminal outcome. A pending packet has `state: "awaiting_host"`, `terminal: false`, `request_id`, `candidate_dir`, an `instructions` file path, the engine-owned `request` path and a separate writable `result_file` path. Read the instructions file in full; do not edit the request file. The host follows the original failure-specific method, including constructive precision Phase A/B/C and factual probes. The request's candidate and allowlist are the edit boundary; never edit the active `WORKSPACE/task`, reference, wrapper, cases, runtime, evidence or ledgers directly.

The request's `diagnostics.task` is a separate writable copy for necessary diagnostic builds, local tests and L5 probes. Use the pinned Python/environment and command vectors in `diagnostics.execution`, refresh its kernel from the current candidate, and follow the original probe policy. Diagnostic outputs are evidence only; they are neither imported on submit nor accepted as formal validation.

Write the actual host result to the packet's `result_file` path, outside candidate source files:

```json
{
  "outcome": "completed",
  "final_response": "Actual diagnosis and changes, followed by the required ENGINE_ATTEMPT_METADATA footer"
}
```

Use `outcome: "failed"` plus a nonempty `error` for failed host work. Replace the explanatory string with the actual response. Completion is not an engine pass claim. Do not supply fabricated `success`, precision, measurements or host turns. The full prompt specifies the footer fields; [the handoff reference](portable-worker.md) explains them.

`submit` checks the pending request and source/protected-file integrity, then runs engine-owned validation before advancing. Inspect its output, and repeat `next → host edit → submit` only while nonterminal. A wrong/stale request ID or unexpected file change is an error, not permission to repair engine-owned state manually. `status WORKSPACE` is the read-only inspection entry point.

The retired `run` command gives migration guidance. New repair work uses freshly prepared schema-4 workspaces. Legacy schema-1/2/3 workspaces retain their runtime and evidence; inspect them with `status` and use supported independent verification. Do not hot-patch/migrate them into this host loop or reopen their terminal sessions.

## Deterministic execution contract

New preparation defaults to `--determinism on`; deliberately choose `off` before preparation if required. The workspace records its mode and holds it fixed for native commands and verification. Preserve an existing workspace and prepare a new one to change the mode.

The workspace-local startup hook applies `torch.use_deterministic_algorithms(..., warn_only=False)` before managed Python task entrypoints. Hash-bound process receipts record interpreter/package/configuration identity and flags. Missing/changed configuration or startup failure stops execution instead of continuing with unknown settings. No global `sitecustomize` is installed. Deterministic mode does not alter references or tolerances, guarantee cross-version/hardware equality or make an unsupported deterministic operation valid.

## Nonfinite output contract

New runtime copies first require matching reference/candidate NaN, positive-infinity and negative-infinity masks at corresponding output positions. Matching nonfinite reference outputs are allowed. Finite values then use the frozen comparator and unchanged tolerances. This additional public-adapter acceptance rule is recorded separately from the frozen paper source.

`drdebug-workspace.json` records `nonfinite_guard`, policy `exact_nan_posinf_neginf_masks_then_frozen_comparator`, helper/verifier hashes and CLI/import coverage. Compact/full acceptance and independent verification require complete matching guard receipts; a finite-only pass is insufficient. Old workspaces without this field remain `legacy_unchecked`; reinstalling never backfills or relabels their evidence.

## Best checkpoint and acceptance

The public adapter binds objective results to kernel/protected-input hashes before and after validation and prioritizes complete full-case results before compact metrics. A completed 50/51 candidate may be the best available checkpoint without being accepted. Missing, malformed, incomplete or source-mismatched evidence does not certify success.

The runtime copy also ensures a regular `precision_tuning/` directory before baseline validation, so an already-correct baseline can retain evidence and reach terminal acceptance without a host patch or repair attempt. Do not manufacture an attempt to change historical status fields.

Success requires `state: "terminal"`, `terminal: true`, `engine_status.session_outcome: "success"`, `returncode: 0` and `accepted_checkpoint.ready: true`, backed by full numerical/integrity gates and nonfinite coverage. Check actual responses/checkpoint reports, not a filename, best score or host self-report. Record independent `verify` separately. Source-bound checkpoint corrections apply only to copied runtimes; old history and the frozen archive stay intact.

## First failure or resume

Fresh native tasks allow **five attempts** and **86,400 seconds total wall time**. Time spent with a pending host request counts, as does deterministic command execution. Fresh `next` / `submit` accept `--max-attempts` / `--deadline-sec`; later commands must retain the saved values. There is no adapter model/worker timeout or synthetic turn counter. Stop early on terminal acceptance or any other terminal condition; the ceilings are not minimum consumption.

This is an engine deadline checked on advancement, not a watchdog for an externally running host. DRDebug does not own or terminate that host's model session. Configure host turn/cost/time limits in the host itself; a host limit does not authorize resetting the workspace or discarding its pending request.

Saved budgets, pending request identity and consumed time survive subsequent commands. Read `status` before resuming; do not reset attempts, delete ledgers/events/locks or reopen terminal state. An existing ledger with unusable budget information is not granted fresh defaults. Another operation holding the lock prevents concurrent advancement/verification.

`next`, `submit` and `verify` may spend minutes building or evaluating. If the host has a shorter foreground tool limit, use its supported background/session mechanism and retain the command's logs and final exit status. Poll that same process; do not launch a duplicate operation or infer success from a progress file. An interrupted deterministic action without a durable result is preserved as `needs_attention`, not silently replayed. Inspect its status and logs before deciding how to recover.

If a new trial is authorized, prepare a new complete workspace from reviewed sources, preserve reference/wrapper/cases, and record parent workspace/checkpoint, prior evidence and consumed budget. Never present an inherited candidate as the untouched initial input. Keep environment, permissions, host/model failures, timeout, build/runtime and numerical outcomes distinct.

## Independent verification

```bash
python3 <skill-dir>/scripts/drdebug.py verify WORKSPACE \
  --npu ID --soc SOC --casewise --json
```

Verification copies the actual active kernel into a new `verification/<id>/task/`, clean-builds the native extension and executes the original compact/full collections in fresh interpreters with the nonfinite guard. It checks native binding, C++ integrity, wrapper AST, source/protected identity and unchanged engine history. It calls no model and neither promotes a checkpoint nor makes a partial session successful. Supported legacy verification retains that workspace's historical contract; it does not upgrade it to schema 4.

`--casewise` adds a fresh process for each original full-case index, regenerating the original seed-0 full collection before selecting the index. Report whole-suite and per-case measurements separately. Defaults remain 1,800 seconds per stage and 10,800 overall; `--stage-timeout-sec` / `--overall-timeout-sec` are independent verification limits.

Judge final `verification/<id>/summary.json` together with command exit status and nonfinite coverage. `progress.json` is only a process snapshot and may retain initial defaults after completion. Do not verify concurrently with pending host edits or another operation.

The [Cumsum validation record](https://github.com/Shenxiu-WU/DRDebug/blob/main/examples/cumsum/validation.md) describes versioned historical replay/guided repair and source identities. Those measurements predate native `next`/`submit`; they are not evidence that this new host protocol has passed a fresh remote repair. Installation, preflight, adapter tests, historical replay and new host-driven repair are separate results.

## Container device IDs

Use the device ID exposed inside the execution container for `--npu`. A container restricted to host physical NPU 1 may expose it as device 0; verify actual enumeration rather than copying the host index. Record the host physical ID and container/runtime ID separately. Model-provider variables, including the current Kimi Code `KIMI_MODEL_*` family, are removed from deterministic tool subprocesses.
