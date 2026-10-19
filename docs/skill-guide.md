<a id="install-drdebug-and-repair-through-your-host-agent"></a>

# Install DRDebug and repair with your coding agent

[English](skill-guide.md) · [简体中文](skill-guide.zh-CN.md)

DRDebug supplies the public paper's deterministic repair engine as an Agent Skill. **Your coding agent proposes and edits patches using its existing model and tools.** The CLI does not launch a nested coding agent or call a model API. The current workflow is `prepare → next → agent edit → submit`, repeated until terminal, then independent `verify`.

<a id="1-install-for-your-host"></a>

## 1. Install for your coding agent

```bash
git clone https://github.com/Shenxiu-WU/DRDebug.git
cd DRDebug
./install.sh --codex
```

Choose a skills directory recognized by your coding agent; the example above uses Codex:

| Coding agent / installation target | Command | Installed Skill |
| --- | --- | --- |
| Codex, user | `./install.sh --codex` | `~/.agents/skills/drdebug` |
| Claude Code, user | `./install.sh --claude` | `~/.claude/skills/drdebug` |
| Kimi Code | `./install.sh --dest "$HOME/.config/agents/skills"` | `~/.config/agents/skills/drdebug` |
| Codex, project | `./install.sh --codex --project /path/to/project` | `/path/to/project/.agents/skills/drdebug` |
| Claude Code, project | `./install.sh --claude --project /path/to/project` | `/path/to/project/.claude/skills/drdebug` |
| Other compatible agent | `./install.sh --dest /path/to/your-agent/skills` | `/path/to/your-agent/skills/drdebug` |

Start Kimi Code with an explicit skills directory after installing:

```bash
kimi --skills-dir "$HOME/.config/agents/skills"
```

[`--skills-dir`](https://github.com/moonshotai/kimi-cli/blob/main/docs/en/customization/skills.md) takes the **skills parent**, not `drdebug/` or `SKILL.md`. It overrides user/project skill discovery while retaining built-in skills; repeat the flag for any additional custom skill roots. This explicit loading method avoids relying on version-specific automatic discovery paths.

For another compatible agent, replace the custom target with its actual skills parent and load/register `drdebug/SKILL.md` through that agent's mechanism. Kimi Code is a coding agent; Kimi K3 is a model, which can also be configured in other compatible agents. Your agent needs tools to read evidence, edit files and run or transport CLI commands. Authentication, model selection, permissions and SSH remain with your agent; do not put keys in task data, prompts, results or logs.

`--project` without a path uses the current directory. `--dry-run` previews the destination. Identical installation is a no-op; `--force` updates a different installation with a backup. The installer copies runtime source and provenance so the Skill is independent of the original checkout. It installs neither coding agents nor NPU drivers/packages and changes no model account configuration.

For upload-based setup, first assemble the bundle with `./install.sh --dest /path/to/staging-skills`, then transfer the **entire generated `drdebug/` directory**, including `runtime/`, `runtime/manifest.json` and the Skill scripts. Load/register its `SKILL.md` with your agent on the destination machine and run `doctor` on the numerical execution machine. Input tasks are supplied separately.

## Requirements

Do not confuse execution requirements with a pip `requirements.txt`:

| Operation | Required environment |
| --- | --- |
| Install / inspect the public CLI | Python 3.10+ and standard library; no provider SDK or coding-agent CLI. |
| Build / numerically evaluate a kernel | Linux + accessible Ascend NPU/driver, compatible CANN/AscendC, CMake 3.16+, C++ compiler/Python headers, matching Torch/torch_npu, NumPy and pybind headers. |
| Propose a repair | Your coding agent's existing model and tools. Codex, Claude Code, Kimi Code or another compatible agent can fill this role. |

The separate [execution requirements note](skill-guide-requirements.md) distinguishes system/runtime requirements from pip package inputs.

### Use a matching isolated Python environment

Choose the package/Python/CANN/driver combination from the [Ascend compatibility table](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md). Run numerical commands on that NPU machine or its compatible container; your agent may use its approved SSH tools for remote execution. `<skill-dir>` and workspace paths refer to files on the execution machine.

Keep one isolated Python environment active for every CLI command. The frozen build also runs `python3` from `PATH` and fixes `_GLIBCXX_USE_CXX11_ABI=1`; matching package names alone do not establish ABI compatibility. `doctor` checks the actual ABI, interpreter identity, NumPy/Torch round trip, headers and toolkit/compiler files. Torch-bundled pybind headers are preferred, with a separate pybind11 package as fallback. Follow the vendor's compatible packages instead of changing global startup hooks or forcing one historical version tuple onto a different machine.

## 2. Check the execution machine

For a Codex user install, for example:

```bash
DRDEBUG_SKILL="$HOME/.agents/skills/drdebug"
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" doctor \
  --npu 0 --soc YOUR_SOC_VERSION --runtime-probe --strict
```

Use the actual installed path for Claude Code, Kimi Code, project or custom installs. Select a free device ID exposed to the execution environment and the SoC identifier accepted by the installed compiler, not merely a benchmark label. `--npu` selects an exposed device through `ASCEND_RT_VISIBLE_DEVICES`; the adapter uses logical `npu:0` internally. Containers can remap host IDs: verify enumeration inside the actual execution environment and record the host physical ID separately.

`doctor --json` provides machine-readable inventory; `--strict` fails on missing execution prerequisites. The optional runtime probe performs a built-in NPU add and repeated fixed-input FP16 cumsum. It does not run the user's kernel, authenticate a coding agent or call a model. A probe pass is not task acceptance. There is no required Claude executable and no worker-authentication preflight.

If packages and the toolchain pass but device enumeration fails, check device ownership separately. During validation, a later container reported `aclInit 507899 / EL0005 Resource_Busy` while an earlier completed container remained alive, despite an empty NPU process table. Stopping that verified-idle predecessor restored the tiny probe and the unchanged `doctor`. Preserve its logs and files, and release only a completed container you own before retrying. DRDebug does not stop containers or reset hardware.

## 3. Provide an existing task

```text
my-task/
├── model.py                   # protected reference torch.nn.Module
├── model_new_ascendc.py        # protected native-extension wrapper
├── MyOperator.json            # compact input-case JSONL
├── MyOperator.json.bak         # explicit full collection
└── kernel/
    ├── pybind11.cpp
    └── ... .cpp / .h / .hpp   # existing candidate implementation
```

Supply at least one explicit `.json.bak` / `.json.full` full collection with the active JSON stem. Compact/full may coincide or be nonnested; conflicting full collections are rejected. Preserve original reference, cases and tolerances. Use the JSON stem for `--op-name`: `3_Add` for `3_Add.json` in `003_Add/`.

The repository separately provides the [reference/case companion](../papers/2026-dont-regenerate-debug/dataset/) and [27 initial candidate projects](../papers/2026-dont-regenerate-debug/artifact/inputs/). They are not copied into installed Skills. Ten historical reference examples are installed for method guidance. A dataset entry alone has no candidate kernel.

## 4. Prepare a new workspace

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" prepare /path/to/my-task \
  --output /path/to/my-repair --op-name MyOperator --determinism on
```

New workspaces use schema 4 and hand repair requests directly to your agent. Preparation copies task/runtime into a fresh directory, retains originals and records hashes. Determinism defaults to on and is fixed for the workspace; choose off deliberately before preparation if needed. The copied verifier also requires matching NaN/+Inf/−Inf output masks before the original finite comparator and unchanged tolerances. See [the runtime contract](../skills/drdebug/references/runtime.md#nonfinite-output-contract).

Add `--reference-root /path/to/reference-cases` for supplied independent examples. They are copied separately as read-only guidance. Follow Phase A → B → C and the [public resource map](../skills/drdebug/references/runtime.md#public-resource-map); do not invent missing legacy APIs or canonical KB files.

Existing workspaces are never hot-patched into the new protocol. Preserve their events/checkpoints/runtime and use `status` or supported independent `verify`; prepare a new schema-4 workspace for current repair work. The former `run` command is retired with migration guidance.

<a id="5-let-the-host-perform-each-requested-repair"></a>

## 5. Let your agent perform each requested repair

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" next /path/to/my-repair \
  --npu 0 --soc YOUR_SOC_VERSION --json
```

`next` runs deterministic engine steps until it returns a pending repair request or terminal outcome. A pending response has `state: "awaiting_host"` and `terminal: false`. Read the full file at `instructions`, together with `request_id`, `candidate_dir`, evidence and allowed paths; leave the engine-owned `request` file unchanged. Your agent follows that prompt with its **own model and tools**, edits the isolated candidate and records real diagnostics. It must not edit the active task, references, wrappers, cases, runtime, requests or history.

The request's `diagnostics.task` is a separate writable copy for necessary diagnostic builds, local tests and L5 probes. Use the pinned Python/environment and command vectors in `diagnostics.execution`, refresh its kernel from the current candidate, and follow the original probe policy. Diagnostic outputs are evidence only; they are neither imported on submit nor accepted as formal validation.

Use the complete audit paths from the instructions: `precision_tuning/precision_audit_<attempt>.md` and `precision_tuning/candidate_kb_entry.json`, both relative to `candidate_dir`. Keep the `precision_tuning/` directory when writing or moving an audit file.

Read an existing audit file before overwriting it. Some agent editing tools, including the Claude Code version tested here, require a prior read even when the full path is correct.

Build and verification commands can take longer than the foreground wait limit of your agent's command tool. Use your agent's background/session mechanism when needed, retain stdout/stderr and the final exit status, and poll the same process to completion. Do not start overlapping commands on the workspace or use a progress file as a final result. If interrupted, inspect `status` and logs; an indeterminate deterministic action needs attention rather than a blind retry.

The method retains specification-first Phase A, complete implementation reading/probe policy in Phase B, and structured comparison in Phase C. The engine injects KB retrieval; your agent lists only IDs actually used and does not repeat retrieval or modify the seed. Required audit fields and full numerical/integrity checks remain in force.

Write a result JSON to the returned `result_file` path, outside candidate sources. Its contract is:

```json
{
  "outcome": "completed",
  "final_response": "The actual diagnosis and patch summary, ending with the ENGINE_ATTEMPT_METADATA footer from the full prompt"
}
```

Use `"outcome": "failed"` and a nonempty `"error"` when the agent cannot complete the requested work. Replace the explanatory text above with the real response and factual metadata. `completed` means the agent finished its turn, not that a kernel passed. Do not invent success/precision/turn values. Then submit the same request:

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" submit /path/to/my-repair \
  --request-id REQUEST_ID --result /path/to/request/host-result.json \
  --npu 0 --soc YOUR_SOC_VERSION --json
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" status /path/to/my-repair
```

The engine checks the request and source identities, validates the patch and advances the task. Inspect its response, then repeat `next → agent edit → submit` until terminal acceptance or another terminal outcome. A stale ID, unknown edit or failed integrity check is not permission to modify the engine's files to continue.

Fresh tasks have ceilings of **five attempts** and **86,400 seconds total wall time**, including time spent waiting for your agent and running deterministic commands. Saved budgets persist across commands. Set `--max-attempts` / `--deadline-sec` on the first native command only if different authorized ceilings are needed; subsequent commands retain the saved values. There is no nested worker, adapter-selected model, per-worker turns or fabricated agent-turn accounting. Your agent manages any model charges it incurs.

The engine checks its deadline when advancing the workspace; it does not control or terminate your agent's model session. Set your agent's turn/cost/time limits separately. If one is reached, retain the agent session, request and workspace evidence before resuming.

Before resuming, inspect `status`; keep request IDs, events and consumed budget. Never delete ledgers/locks, reset attempts or reopen terminal sessions. An authorized inherited trial uses a new workspace and records parent/checkpoint and prior budget usage. Details: [first failure or resume](../skills/drdebug/references/runtime.md#first-failure-or-resume).

An initial natural-language request can be:

> Use drdebug with your current model and tools. Check my Ascend environment, prepare a separate workspace, follow next/submit requests to repair this failing kernel, and independently verify the actual accepted source.

## 6. Independently verify

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" verify /path/to/my-repair \
  --npu 0 --soc YOUR_SOC_VERSION --casewise --json
```

This copies the actual active kernel to a fresh verification directory, clean-builds it and runs original compact/full comparisons plus nonfinite, native-binding and integrity checks. `--casewise` adds a fresh process per original full-case index. It calls no model, promotes no checkpoint and changes no engine history. Do not run concurrently with pending candidate edits by your agent or another workspace operation.

Use final `verification/<id>/summary.json`, command exit status and guard coverage. `progress.json` is only a process snapshot. Stage/overall verification limits default to 1,800/10,800 seconds and can be set with `--stage-timeout-sec` / `--overall-timeout-sec`; these are independent of the repair-loop budget.

## What counts as a repair

Require `state: "terminal"`, `terminal: true`, `engine_status.session_outcome: "success"`, `returncode: 0` and `accepted_checkpoint.ready: true`, backed by full numerical/integrity gates. An agent's `completed` result, a best-but-partial checkpoint or compilation success is insufficient. An already-correct baseline may be accepted without an agent patch; do not fabricate an attempt. Report independent verification separately, with compact/full counts, source identity, mode, integrity and remaining failures.

The [agent-driven Cumsum validation record](../examples/cumsum/native-host-validation.md) reports fresh installations, real `next`/`submit` repairs and separate independent acceptance, including the supplied public reference and exact agent configuration. The earlier [replay/guided-repair measurements](../examples/cumsum/validation.md) remain separate. Installation, environment preflight, CPU adapter tests and actual NPU repair are separate outcomes. Keep environment/permission, agent/model, timeout, runtime and numerical failures distinct.
