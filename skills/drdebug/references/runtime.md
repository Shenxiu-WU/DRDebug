# Runtime and task contract

## What is installed

`install.sh` copies this skill and the public `debug_agent_v5` paper source into a self-contained directory:

```text
drdebug/
├── SKILL.md
├── agents/openai.yaml
├── references/
├── scripts/                 # launcher, checkpoint, runtime and replay adapters
└── runtime/
    ├── source/debug_agent_v5/
    ├── manifest.json
    └── provenance/
```

`runtime/manifest.json` records source paths and hashes. In a repository checkout, the launcher can instead resolve the paper source under `papers/2026-dont-regenerate-debug/artifact/code/debug_agent_v5/`. Preparation and repair never edit the source artifact. New schema-3 preparation records the original source hashes and separately records adapter changes in the copied runtime; its checkpoint bridge adjusts only that workspace copy. The documented public supplement adds inputs and references to the artifact; the prior published version remains available at its existing Git tag.

## Public resource map

Paths in this table are relative to the prepared `WORKSPACE/runtime`, not the installed skill directory. The original method resources live under `skills/ascendc/`; `.claude/skills/` contains registration instructions and is not their reference-file base.

| Status | Resource | Use |
| --- | --- | --- |
| Bundled | `archive_tasks/` | Ten historical reference examples; seven have AscendC kernels. Select a relevant case and record its source. |
| Bundled | `skills/ascendc/ascendc-debug/references/decomposition_examples/` | Read its `README.md`, then an existing matching example; do not infer an unlisted filename. |
| Bundled | `skills/ascendc/ascendc-debug/references/phase-a-checklist.md` | Required Phase A/C field templates. |
| Bundled | `skills/ascendc/ascendc-translator/references/` | Contains `dsl2Ascendc.md`, `dsl2Ascendc_compute_vector.md` and `TileLang-AscendC-API-Mapping.md`; these are published guidance, not the full SDK API tree. |
| Bundled, read-only | `experiments/v5_ablation/fixtures/initial_kb_102_cleaned_20260715.json` | Public initial KB snapshot: 18 titled knowledge entries + 1 metadata object. `102` is part of the retained filename, not the entry count. The engine injects retrieval summaries; the worker does not repeat KB search or write to the seed. |
| Optional | `archive_tasks/user_references/` | Created only with `--reference-root`; if none was supplied, skip it. It never replaces bundled history. |
| Not bundled | `skills/ascendc/ascendc-translator/references/AscendC_knowledge/api_reference/` | The legacy detailed API tree, including its `INDEX.md`, is unavailable. Skip these legacy locations. |
| Not bundled | `skills/ascendc/ascendc-debug/references/precision_knowledge_base.json` | The historical canonical KB is unavailable. Neither the public seed nor retained `old_precision_knowledge_base.json` reconstructs it. |

These known absences are not installation failures. This resource map overrides legacy resource-location instructions, including links inside the bundled DSL documents. Do not repeatedly probe those missing locations, create empty API trees, or rename/symlink a seed or old KB to the canonical filename. Use the published guidance, protected task reference and engine evidence; report an unresolved API-information gap rather than inventing a signature. Phase A/B/C and all acceptance checks still apply.

The reference-and-case dataset and 27 recovered initial task projects are available separately in the public repository, under `papers/2026-dont-regenerate-debug/dataset/` and `papers/2026-dont-regenerate-debug/artifact/inputs/tasks/`. Neither collection is copied by the skill installer. Select an existing repository task or a user-supplied task for `prepare`; do not search for these collections inside an installed skill.

## Task input

Provide a task directory containing:

- `model.py`: the reference `torch.nn.Module` implementation.
- `model_new_ascendc.py`: the existing wrapper calling the compiled AscendC extension.
- `kernel/`: actual `.cpp` / `.h` / `.hpp` sources, including a `pybind11.cpp` binding.
- An operator input JSONL file such as `MyOperator.json`: each nonempty line is an input-case object with an `inputs` field, in the format accepted by the released verifier.
- At least one explicit complete `.json.bak` / `.json.full` case collection, with the same base name as the active JSON. It may equal the compact set. Equivalent backups may coexist; conflicting full collections are rejected. Compact cases need not be a subset of full cases; the launcher records their relation without rewriting either file.

Do not synthesize a missing reference, wrapper or passing record. Use the case JSON stem as `--op-name`, not the directory prefix: `3_Add` for `3_Add.json` in `003_Add/`. The repository now supplies the recovered [27 initial candidates](https://github.com/Shenxiu-WU/DRDebug/tree/main/papers/2026-dont-regenerate-debug/artifact/inputs) separately from the reference-and-case dataset. A dataset entry alone has no candidate kernel.

## Runtime environment

Installation and the launcher use Python 3.10+ and its standard library. `doctor` imports the execution libraries in child interpreters to check actual compatibility. Numerical execution additionally needs:

- Linux with access to an Ascend NPU and working device drivers.
- Compatible CANN/AscendC toolkit, CMake and C++ build tools, with the toolkit environment sourced.
- Matching PyTorch and `torch_npu`; NumPy. The frozen build prefers Torch-bundled pybind headers, with a separately installed `pybind11` package as fallback.
- For live repair, a configured and authenticated `claude` executable. Authentication and model access stay with the user; a Codex installation alone does not supply this backend. Independent `verify` needs no worker/model, and `doctor --no-worker` omits only worker checks.

The launcher interpreter and `python3` on `PATH` must use the same Python environment: the frozen build invokes `python3` from CMake as well. It fixes `_GLIBCXX_USE_CXX11_ABI=1` for the extension, so the selected Torch/torch_npu libraries must be ABI-compatible. Satisfy these execution prerequisites in an isolated environment; do not change the frozen engine to conceal a mismatch. Preflight verifies the actual ABI flag, `python3`/launcher environment identity, Python/pybind headers and a NumPy → Torch → NumPy round trip, in addition to imports, compiler/CANN files and version inventory. An installed NumPy package alone does not prove its Torch bridge works.

Use the [vendor's compatibility table](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md) for full Torch/torch_npu/CANN/Python/driver matching. The demonstrated saved-Cumsum profile is Torch 2.9.0, torch_npu 2.9.0.post6, CANN 9.1.0, Ascend910B4, deterministic mode on; this is one tested profile, not a universal version requirement. Saved replay with the new guard and guided live repair have separate completed [validation records](https://github.com/Shenxiu-WU/DRDebug/blob/main/examples/cumsum/validation.md); the latest default-baseline check is recorded there separately. The [installation guide](https://github.com/Shenxiu-WU/DRDebug/blob/main/docs/skill-guide.md#use-a-matching-isolated-python-environment) gives an isolated venv recipe; do not inherit the previous experiment's startup hook or modify global Python packages.

`doctor --runtime-probe --npu ID --soc SOC --strict` adds a small built-in NPU add and repeated fixed-input FP16 cumsum. It allocates NPU tensors but does not build or run user kernels or call a model. `run` and `verify` require this probe. A probe pass establishes only these bounded prerequisites, not arbitrary-operation reproducibility, correct task compilation or full numerical acceptance.

The worker launches Claude Code with `--setting-sources project`. A custom provider configured only through the `env` object in `~/.claude/settings.json` is therefore not automatically inherited. Provide existing approved provider configuration to the launching process through the user's environment or secret-management setup; never copy credentials into task files, the repository or logs. OAuth sign-in is a separate authentication mechanism; this settings boundary alone does not invalidate an existing OAuth session.

`doctor` does not verify model availability or agent tool execution; if needed, confirm the existing service with minimal, explicitly authorized text and side-effect-free tool requests (such as reading a test file you own), requiring actual successful `tool_use`/`tool_result` and a normal follow-up model reply after returning the tool result—the complete round trip. A text response, model self-report or single tool execution does not validate the tool chain. API failures may terminate the historical engine: retain terminal evidence without editing events or state to restart it, and record any separate acceptance run as a new trial with prior budget usage and the remaining allowance carried forward.

Use the actual `doctor` report for CLI compatibility. Its `claude_cli_contract` check matches missing-required-argument errors to determine whether the CLI recognizes options omitted from `--help`. These probes provide no prompt and do not call a model. A passing preflight establishes neither provider authentication nor successful device evaluation.

Use the host vendors' installation guidance for the machine's actual driver, toolkit and Python versions. The installer copies files; it does not install drivers, Python packages, a model account or a benchmark client. Run on the NPU machine or inside its existing compatible container. Remote access is user-provided, not embedded in the skill.

## Deterministic execution contract

New `prepare` creates a schema-3 workspace with `--determinism on` by default; `off` must be deliberately chosen at preparation. The mode is recorded in `drdebug-workspace.json` and held fixed for run/resume/verify. Legacy schema-1/2 is interpreted as off and retains its existing runtime; it is not silently upgraded. Preserve an old workspace and prepare a new one to adopt this contract or change modes.

The adapter creates a workspace-local startup hook and configuration, validates their hashes, and applies `torch.use_deterministic_algorithms(mode == "on", warn_only=False)` before managed Python user entrypoints. Receipts under `WORKSPACE/adapter-receipts/` bind the process, interpreter, package versions, configuration hashes and observed flags. A missing/changed hook or configuration fails closed; startup failure exits 78 instead of continuing with unknown settings. Independent replay checks the early receipt and current flags, rather than setting them after the task has already run. Nothing is installed into the global interpreter or an existing venv's `sitecustomize`.

This condition changes the selected execution algorithms, not the reference source, case collection or dtype tolerances. Unsupported deterministic operations may raise, and identical seeds/settings do not establish cross-version or cross-hardware equality; see [PyTorch deterministic algorithms](https://docs.pytorch.org/docs/2.9/generated/torch.use_deterministic_algorithms.html) and [reproducibility](https://docs.pytorch.org/docs/2.9/notes/randomness.html). The saved Cumsum pass was in on mode; it does not prove the earlier default/off environment passed.

## Nonfinite output contract

New `prepare` adds an output guard only to the copied runtime, before the verifier's CLI main entrypoint and for imported verification. Before delegating finite-value comparisons to the frozen comparator with its original tolerances, it requires the candidate and reference to have identical NaN, positive-infinity and negative-infinity masks at corresponding tensor-output positions. Matching nonfinite reference values are allowed; newly introduced, misplaced or differently classified nonfinite values fail. This is an additional public-adapter acceptance rule, not a rule retroactively attributed to the frozen paper comparator. The paper source, task reference, cases and thresholds remain unchanged.

`drdebug-workspace.json` records `nonfinite_guard` with policy `exact_nan_posinf_neginf_masks_then_frozen_comparator`, original/adapted verifier hashes, helper path/hash, coverage of CLI/import entrypoints and `frozen_archive_modified: false`. CLI verification emits `DRDEBUG_NONFINITE_GUARD=<json>`; imported results expose `public_nonfinite_guard`. Each receipt binds its policy/helper and entrypoint, expected/checked tensor leaves, nonfinite mismatch/error counts, complete coverage and pass status. New-workspace checkpoint acceptance and independent verification require valid matching receipts; a finite comparison pass alone is insufficient.

The independent summary records these receipts under `nonfinite_guard.suites`; an accepted checkpoint retains compact/full guard receipts. Existing workspaces without manifest `nonfinite_guard`, including earlier schema 3, remain byte-preserved and report `status.adapter_support.nonfinite_guard = "legacy_unchecked"` and `status.nonfinite_guard.requires_new_workspace = true`. Reinstalling the skill does not migrate that workspace. Preserve its evidence and use a new `prepare` output to adopt the guard; earlier successful results do not establish this added contract.

## Portability boundary

Preparation retains the repository-shaped runtime so the paper engine's relative `skills/`, `utils/` and worker paths continue to work. It registers the worker/skill in that workspace and records the public adapter. The installer also includes the recovered historical `archive_tasks/` tree. The adapter prefers relevant archived examples and uses the task's read-only reference plus bundled API documents when no matching case is available; additional user references can be supplied through `--reference-root`.

For a newly prepared workspace, Claude Code may treat `WORKSPACE/runtime` as untrusted and ignore project `permissions.allow` or `additionalDirectories`. If warned, the user should review that runtime and its settings, start Claude Code interactively there, and explicitly trust the directory before resuming. The installer and skill do not automatically modify global trust; writing project settings does not establish that those permissions are active. Do not use permission bypass as a substitute for the user's trust decision.

The public initial KB snapshot is a read-only seed, explicitly selected for this new-task workflow. It is not asserted to be the missing canonical KB used by the historical batch harness. The repository separately supplies all 27 initial candidates; the installed historical reference tree is the ten-example `archive_tasks/` collection. Historical batch infrastructure, complete later candidate trajectories and hidden benchmark inputs remain outside the installed skill.

The original full-evaluation escalation, finite-value numerical comparator, integrity/acceptance gates and crash-consistent rollback machinery remain in use. Newly prepared runtime copies add source-bound completed-full checkpoint ordering and the nonfinite-output contract above; these changes do not weaken an acceptance condition or rewrite the frozen paper source.

## Best checkpoint and acceptance

The frozen best selector used compact metrics, so two compact 10/10 candidates could tie even when full coverage improved from 50/51 to 51/51. The portable adapter records source/protected-input identity before and after objective validation, preserves the actual result/log hashes, and prioritizes complete full-case measurements before compact metrics. Incomplete, missing, malformed or mismatched reports do not become a full-score certificate. The workspace manifest records both original and adapted runtime hashes. Before original objective validation, the adapter also ensures a regular, non-symlink `precision_tuning/` result directory exists, so a successful first baseline can retain full-result evidence before any worker runs. This correction applies only to copied runtimes.

“Best” means the strongest validated candidate available for rollback; it may still have full failures. A completely executed 50/51 candidate can be best without being accepted. Success still requires the original full numerical and integrity gates, the added nonfinite guard and a final source-consistent checkpoint. Use actual reports and source maps, not `current_best`'s name or the model's text, when deciding which kernel to export. Legacy history remains intact; the correction does not backfill or relabel old runs.

An already-correct baseline can be accepted without a worker or repair attempt: require `run` exit 0, `accepted_checkpoint.ready: true` and an unlocked terminal success status. Legacy `engine_status.objective_success` / `reportable_success` describe repair-final-validation and can remain false for zero attempts; do not manufacture a repair attempt to change them. Independent `verify` remains the separate numerical verdict.

## First failure or resume

`prepare` uses a new output directory and retains the original task. `run` uses the workspace's task and runtime, not the current shell directory. Engine events and validation artifacts are the basis of resume and outcome reporting. Use `status` before rerunning an existing workspace. Do not delete its evidence or reset attempt counts to obtain additional attempts.

For a fresh budget record, the default profile is five attempts, 240 turns per complete worker invocation, `--worker-timeout-sec 21600` and `--deadline-sec 86400`. The worker timeout caps one invocation; the total deadline independently covers deterministic work and worker calls. The launcher clamps the worker timeout to the remaining total deadline. Attempt/turn guards and numerical/integrity acceptance remain unchanged.

The default fresh-task profile can be made explicit:

```bash
python3 <skill-dir>/scripts/drdebug.py run /path/to/long-repair \
  --npu 0 --soc YOUR_SOC_VERSION --max-attempts 5 --max-turns 240 \
  --deadline-sec 86400 --worker-timeout-sec 21600
```

These are generous upper bounds, not minimum consumption; a task stops early when acceptance succeeds or another engine guard stops it. Six hours is operational headroom, not an experimentally established optimum, and users may choose different initial caps. Budgets are saved on first execution and remain fixed on resume: prior values are retained, including an implicit 600-second worker limit for older records missing that field. A terminal session is not restarted. If a new or larger budget is explicitly authorized, retain the old workspace and prepare a new complete task using the reviewed checkpoint kernel with references, wrappers and cases unchanged. Record the parent workspace, checkpoint identity, prior validation, prior budget consumption and the new allowance as an inherited new trial; never rewrite old events/status or present it as an untouched initial candidate.

An existing run ledger without a usable saved budget is rejected, rather than receiving fresh defaults. The launcher rechecks history and session state under its exclusive lock; a change during preflight aborts that launch without overwriting newer records. Inspect `status` before retrying. An outer active-time timeout and user interruption are recorded separately from a natural engine exit, while preserving the engine's own events and status. On POSIX, stopping the outer invocation also sends a final kill to its dedicated process group even if the engine parent has already exited.

An environment preflight is evidence that dependencies were detected, not evidence of a successful kernel repair. CPU tests and mock-worker tests verify the adapter only. Complete acceptance requires the engine's actual device evaluation for the supplied task.


## Independent verification

```bash
python3 <skill-dir>/scripts/drdebug.py verify /path/to/workspace \
  --npu 0 --soc YOUR_SOC_VERSION --determinism on --casewise
```

This operation requires schema 3 and retains its prepared mode. It uses an exclusive workspace lock, copies the actual active task sources into a fresh `WORKSPACE/verification/<id>/task/`, rebuilds with the original clean build tool, and runs original compact/full collections in new Python interpreters with the original finite-value comparator and added nonfinite guard. It also verifies actual native extension loading and wrapper binding, the original C++ integrity check and wrapper AST guard, source/protected-file identity and unchanged engine history. It does not require provider credentials, invoke a model, promote a checkpoint or turn a partial engine session into success.

`--casewise` adds one fresh process per original full-case index. Each child regenerates the original seed-0 full input collection before selecting its index; the reference and thresholds remain unchanged. Report whole-suite and per-case measurements separately. A full-suite pass does not itself prove fresh-process behavior, nor do individual passes replace a complete original full-suite run.

The summary, per-stage stdout/stderr, raw compact/full results and optional case reports remain under the new verification directory. Default stage/overall limits are 1,800/10,800 seconds; configure `--stage-timeout-sec` and `--overall-timeout-sec` for larger tasks. These are independent of worker turn/time limits. A preflight, timeout, native-binding or integrity failure remains a verification failure; preserve its output and do not infer acceptance from compilation or a process exit alone.

Use final `verification/<id>/summary.json` together with the `verify` command's exit status as the outcome, checking numerical/integrity results and `nonfinite_guard` coverage. `progress.json` is a process snapshot, not a final verdict: it may retain initial `passed: false` or return-code defaults even after the final summary reports success. A progress snapshot cannot replace the final summary or establish failure by itself.

The [Cumsum example](https://github.com/Shenxiu-WU/DRDebug/tree/main/examples/cumsum) includes hash-pinned saved sources for model-free replay and a separate guided live Kimi route from the original paper input. The latter supplies the published repaired implementation through `--reference-root` as a read-only example: it exercises the complete installation/model/patch/acceptance workflow without claiming unaided rediscovery. An independent no-reference study can omit that flag in another workspace and may consume more budget. A measured fresh guided run improved the original 6/10 baseline to compact10/full51 in one attempt and 38 turns, with 51 fresh-case checks. It used the pre-guard runtime; the identical candidate was then verified with the added guard. See the [versioned results](https://github.com/Shenxiu-WU/DRDebug/blob/main/examples/cumsum/validation.md). This does not guarantee every live repair.
