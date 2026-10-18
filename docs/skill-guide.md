# Install DRDebug and repair your first task

[English](skill-guide.md) · [简体中文](skill-guide.zh-CN.md)

DRDebug is an installable Agent Skill backed by the public paper’s deterministic engine. Installation includes the runtime source, rather than leaving a pointer to your checkout. Use it with an existing AscendC task that almost works.

## 1. Install

On the machine where your agent will use the skill:

```bash
git clone https://github.com/Shenxiu-WU/DRDebug.git
cd DRDebug
./install.sh --codex
```

Choose one target:

| Target | Command | Installed skill |
| --- | --- | --- |
| Codex, user | `./install.sh --codex` | `~/.agents/skills/drdebug` |
| Claude Code, user | `./install.sh --claude` | `~/.claude/skills/drdebug` |
| Codex, project | `./install.sh --codex --project /path/to/project` | `/path/to/project/.agents/skills/drdebug` |
| Claude Code, project | `./install.sh --claude --project /path/to/project` | `/path/to/project/.claude/skills/drdebug` |
| Custom parent | `./install.sh --dest /path/to/skills` | `/path/to/skills/drdebug` |

`--project` without a path uses the current directory. `--dry-run` previews the destination without writing files. An identical installation is a no-op; a different existing installation is preserved unless you use `--force`, which keeps a backup. Installation uses Python 3.10+ and the standard library, without package downloads, driver changes or account configuration.

The installed directory contains `SKILL.md`, a launcher and references, plus `runtime/source/debug_agent_v5/`, source hashes and provenance. Moving or deleting the cloned repository does not remove the installed skill. Copy installation is intentional: updates take effect when you rerun the installer, rather than changing a live skill behind your back.

Codex discovers `$drdebug`; Claude Code discovers `/drdebug`. If the skill is not listed, start a fresh agent session. See [Codex skill discovery](https://learn.chatgpt.com/docs/build-skills) and [Claude Code skills](https://code.claude.com/docs/en/skills).

## 2. Check the execution machine

Use a Linux Ascend machine or a compatible container with NPU device access, the installed CANN/AscendC toolkit, CMake 3.16+, a C++ compiler and Python development headers. The execution environment also needs NumPy, matching Torch/torch_npu and pybind headers. Only live repair requires an authenticated Claude Code executable and model service; saved-candidate verification does not call a model.

For a Codex user install, set:

```bash
DRDEBUG_SKILL="$HOME/.agents/skills/drdebug"
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" doctor
```

For Claude Code use `DRDEBUG_SKILL="$HOME/.claude/skills/drdebug"`; project/custom installs use their actual absolute path. `doctor --json` is machine-readable and `--strict` returns nonzero if required checks fail. The default inventory imports runtime libraries in child processes, but does not import a user task, run its kernel or call a paid model.

### Use a matching, isolated Python environment

Choose the package, Python, CANN and driver combination from the [Ascend compatibility table](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md). Matching the Torch/torch_npu major/minor family alone is insufficient. The frozen extension build fixes `_GLIBCXX_USE_CXX11_ABI=1`; `doctor` now checks the actual Torch ABI, interpreter/PATH identity, NumPy-to-Torch round trip, headers and compiler/toolkit files, rather than accepting import/version strings alone. Torch-bundled pybind headers are preferred; a separate `pybind11` package is a fallback.

The verified Cumsum profile is **Ascend910B4 + CANN 9.1.0 + Torch 2.9.0+cpu + torch_npu 2.9.0.post6 + determinism on**. This tuple is listed by the vendor and is an example, not a universal minimum or a claim about A3/910C. Saved-candidate and guided-live results, with their separate runtime versions, are recorded in the [validation record](../examples/cumsum/validation.md).

This isolated Python 3.11 recipe created the tested dependency environment. Later guided and guarded checks used new Skill installations/workspaces in that same environment. It retains the venv-provided pip and pins build helpers for repeatability; these pins describe the tested recipe, not every supported system. Use the existing toolkit's actual `set_env.sh`; no driver, CANN installation or global Python package is changed here. On a different architecture/version, use the vendor's matching wheels instead of forcing these pins.

```bash
source /path/to/installed/cann/set_env.sh
python3.11 -m venv /path/to/drdebug-venv
source /path/to/drdebug-venv/bin/activate
python3 -m pip install 'torch==2.9.0' --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install 'torch-npu==2.9.0.post6' 'numpy==1.26.4'
python3 -m pip install 'setuptools==80.9.0' 'wheel==0.45.1' 'PyYAML==6.0.3'
```

If a download stalls, use a trusted mirror or predownload wheels matching the execution machine's Python ABI and architecture. Before installing a mirrored wheel into this same new venv, compare its SHA-256 with the exact file's official PyPI metadata, for example the [torch-npu release files](https://pypi.org/project/torch-npu/2.9.0.post6/#files) or [PyYAML release files](https://pypi.org/project/PyYAML/6.0.3/#files); package/version names alone are insufficient. Keep download, interruption and hash-check receipts. This is download-transport recovery, not reuse of an old environment's installed packages or proof that the numerical workflow passed. See [pip's wheel installation guide](https://pip.pypa.io/en/stable/user_guide/#installing-from-wheels).

Activate this venv for every subsequent command and keep its `python3` first on `PATH`: the frozen CMake build also invokes `python3`. A Torch version ending in `+cpu` can coexist with the NPU backend provided by `torch_npu`; actual NPU checks decide readiness. Avoid inheriting unrelated `sitecustomize` startup hooks; the adapter reports them instead of silently replacing global startup configuration.

### Exercise the selected device before repair

Select a free physical device ID exposed by your execution environment and the real SoC identifier accepted by your toolkit. `--npu ID` selects that physical ID through `ASCEND_RT_VISIBLE_DEVICES`; the adapter then uses logical `npu:0` internally. The validation setup uses `--npu 1` with internal logical `0` and `Ascend910B4`. The commands below use physical `0` as an example; replace it with your selected free ID, and do not infer the SoC identifier from a benchmark label.

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" doctor \
  --npu 0 --soc YOUR_SOC_VERSION --runtime-probe --strict
```

`--runtime-probe` performs a small built-in NPU add and repeated fixed-input FP16 cumsum. It does not compile or run the user's kernel and is not full task acceptance. `run` and `verify` also require this probe. For saved-candidate replay without Claude Code installed, append `--no-worker`; this skips worker/CLI checks, not build/NPU checks.

For live repair, install Claude Code using its [official setup guide](https://code.claude.com/docs/en/setup), complete the [quickstart](https://code.claude.com/docs/en/quickstart), and confirm the executable is on `PATH`:

```bash
claude --version
```

`claude_cli_contract` uses no-prompt argument probes for options omitted from some CLI help output; these do not contact a model. Separately confirm your configured provider with a minimal authorized text/tool round trip: an actual successful `tool_use`/`tool_result` followed by a normal model response. A version check or text reply alone does not establish this.

The worker uses `--setting-sources project`. Custom provider variables stored only in `~/.claude/settings.json` are not automatically inherited; provide the approved configuration through the launching process's environment or existing secret manager. Never copy keys into a task, prompt, repository or logs. OAuth is a separate authentication path. Provider/API failures remain distinct from kernel failures and may end a historical engine session; preserve its terminal evidence rather than editing events to reopen it.

## 3. Provide an existing task

```text
my-task/
├── model.py                   # reference torch.nn.Module
├── model_new_ascendc.py        # wrapper calling the existing extension
├── MyOperator.json            # input-case JSONL, each object has inputs
├── MyOperator.json.bak         # explicitly identified full case collection
└── kernel/
    ├── pybind11.cpp
    └── ... .cpp / .h / .hpp   # candidate AscendC implementation
```

Provide at least one explicit `.json.bak` / `.json.full` complete collection and keep the original reference/tolerances. The complete collection may equal the compact collection; it is not automatically synthesized. Compact and full sets need not be nested. Equivalent backups can coexist; conflicting complete collections are rejected. Use the JSON stem for `--op-name`: `3_Add` for `3_Add.json`, even when the task directory is `003_Add`.

The [paper task companion](../papers/2026-dont-regenerate-debug/dataset/) contains references and input specifications. The separate [27 initial candidates](../papers/2026-dont-regenerate-debug/artifact/inputs/) now add the byte-verified original wrappers and kernels. The [historical reference examples](../papers/2026-dont-regenerate-debug/artifact/code/debug_agent_v5/archive_tasks/) restore the public archive used for method guidance.

The dataset and 27 initial tasks are repository materials and are not copied into the installed skill; choose a task from the checkout or supply your own. The ten historical reference examples are included with installation.

## 4. Prepare a separate workspace

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" prepare /path/to/my-task \
  --output /path/to/my-repair --op-name MyOperator --determinism on
```

Preparation copies the task and runtime. Originals stay in place. New schema-3 workspaces record `--determinism on` by default; the explicit flag above makes the numerical contract visible. `--determinism off` is available when deliberately selected before preparation. Startup configuration applies `torch.use_deterministic_algorithms(..., warn_only=False)` before managed Python task entrypoints and writes per-process receipts; failure stops that interpreter instead of continuing with an unknown setting.

New preparation also adds a nonfinite-output guard to the copied runtime. Candidate and reference outputs must first have matching NaN, positive-infinity and negative-infinity positions; matching nonfinite reference values are allowed. Finite values then use the original comparator and unchanged tolerances. This additional public-adapter rule leaves the frozen paper source untouched and records verifier/helper hashes in `drdebug-workspace.json` under `nonfinite_guard`. Existing workspaces without that field, including earlier schema-3 workspaces, retain their old runtime: preserve their evidence and prepare a new workspace to adopt the guard. See the [output contract](../skills/drdebug/references/runtime.md#nonfinite-output-contract).

The mode is fixed for that workspace. Existing schema-1/2 workspaces retain legacy off behavior and are not silently migrated; preserve them and prepare a new workspace to use the new configuration. Never edit references or thresholds to compensate for nondeterminism. Deterministic mode can raise for unsupported operations and does not guarantee identical results across versions or hardware, as described in [PyTorch's deterministic-mode API](https://docs.pytorch.org/docs/2.9/generated/torch.use_deterministic_algorithms.html) and [reproducibility notes](https://docs.pytorch.org/docs/2.9/notes/randomness.html). Use a new output directory; an existing workspace is resumed rather than overwritten. If you have independent AscendC examples, add `--reference-root /path/to/reference-cases`; they are kept separately in `WORKSPACE/runtime/archive_tasks/user_references/`. Without that option, the directory may be absent and should be skipped.

On first use, Claude Code may warn that the generated runtime is untrusted and ignore project `permissions.allow` or `additionalDirectories`. If this occurs, review the generated runtime and its settings, open Claude Code interactively from `/path/to/my-repair/runtime`, and explicitly trust that directory before resuming the same workspace. The installer and skill do not automatically change global trust. A project settings file alone does not prove its permissions are active; do not use permission bypass to suppress the warning.

The runtime registers the paper worker and method resources. Use supplied reference examples when relevant, then bundled historical examples or the protected task reference and public guidance. The [public resource map](../skills/drdebug/references/runtime.md#public-resource-map) identifies the read-only KB seed and unavailable legacy API/KB paths; follow it instead of creating substitutes.

From the repository root, you can also choose a recovered paper input, for example Add:

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" prepare \
  papers/2026-dont-regenerate-debug/artifact/inputs/tasks/level1/003_Add \
  --output /path/to/add-repair --op-name 3_Add
```

## 5. Run a bounded repair

Use the SoC identifier accepted by the installed CANN toolkit. Replace `YOUR_SOC_VERSION` below; do not infer it from a benchmark platform label. For a fresh task, the default profile is five attempts, 240 turns per complete worker invocation, six hours per worker and one day overall:

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" run /path/to/my-repair \
  --npu 0 --soc YOUR_SOC_VERSION \
  --max-attempts 5 --max-turns 240 \
  --worker-timeout-sec 21600 --deadline-sec 86400
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" status /path/to/my-repair
```

`--worker-timeout-sec` caps one worker invocation; `--deadline-sec` independently caps the whole task, and the launcher clamps the worker timeout to the remaining total allowance. These generous defaults are upper bounds, not minimum consumption: acceptance can stop the task early, and attempt/turn guards still apply. Six hours is operational headroom rather than an experimentally established optimum; configure caps to fit the user's task without changing its cases or acceptance criteria.

The first execution saves budgets; resume retains those values. See [budget and recovery rules](../skills/drdebug/references/runtime.md#first-failure-or-resume) for legacy limits, missing ledgers, interruption and terminal sessions.

An optional `--model MODEL` selects your configured worker model. `run --dry-run --json` shows the planned invocation without calling the model or running NPU evaluation. This is a command review, not an execution test. For a workspace-specific preflight, use `doctor --workspace /path/to/my-repair --npu 0 --soc YOUR_SOC_VERSION --strict`.

The default worker edits kernel sources and designated diagnostic reports; the deterministic engine performs build/evaluation. If a diagnostic probe needs additional worker shell execution, inspect the request and explicitly choose `--allow-worker-shell` only for a trusted, isolated execution workspace. This option does not bypass the agent permission system. Integrity and full numerical acceptance still belong to the engine.

Inspect `status` before resuming and preserve the workspace's events, attempts and saved budget. Do not reset state or reopen a terminal session. If a separate trial is authorized, prepare a new workspace from a reviewed source, keep original references/cases, and record the parent/checkpoint plus prior budget use. Keep environment, provider and permission failures distinct from kernel failures; the [recovery rules](../skills/drdebug/references/runtime.md#first-failure-or-resume) describe the detailed cases.

Or ask your agent:

> Use drdebug to check my Ascend environment, prepare a separate workspace for this failing kernel, and repair it from the failure evidence. Use device 0 with the fresh-task default caps and stop once the engine accepts the repair.

## 6. Independently verify the actual candidate

After repair, or when testing an existing candidate without a model, run:

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" verify /path/to/my-repair \
  --npu 0 --soc YOUR_SOC_VERSION --determinism on --casewise
```

`verify` requires a newly prepared schema-3 workspace. It copies the **actual active kernel source** into a new `WORKSPACE/verification/<id>/` directory, cleans and rebuilds the native extension, runs the original compact and full case collections with the added nonfinite guard in fresh interpreters, and checks native wrapper binding, original C++ integrity, wrapper AST and protected/source/history integrity. `--casewise` additionally starts a fresh process for every original full-case index. It never calls the model, promotes a checkpoint or rewrites engine history. Each verification retains its own summary and stage logs.

Use the final `verification/<id>/summary.json` together with the `verify` command's exit status to determine the outcome, including guard coverage. `progress.json` is a process snapshot and may retain an initial `passed: false` or return-code value after completion; it is not the final verdict.

The defaults are 1,800 seconds per verification stage and 10,800 seconds overall. Set `--stage-timeout-sec` and `--overall-timeout-sec` explicitly for a larger task; they are separate from live worker budgets. The mode must match preparation. Stop if another worker holds the workspace lock; do not delete its lock to run concurrently.

### Start with the Cumsum example

The [Cumsum walkthrough](../examples/cumsum/README.md) provides two routes: replay eight hash-pinned repaired sources without a model, or launch guided Kimi repair from the original paper input with `--reference-root "$PWD/examples/cumsum/repaired"`. The published repaired source is supplied separately as a read-only example; this tests the installation/model/patch/acceptance chain and is not an unaided rediscovery claim. Omit that flag only for a separate, potentially more expensive independent research run. It preserves the original 10 compact and 51 full cases. Saved replay with the added nonfinite guard passed 10 compact, 51 full and 51 fresh-case processes. Guided live separately passed the same counts after one repair attempt; it used the earlier runtime, and its identical output was later checked with the guard. See the [versioned validation record](../examples/cumsum/validation.md). A stochastic live repair is not guaranteed to finish identically each time.

The copied runtime also creates a checked, regular `precision_tuning/` directory before baseline validation, so a first-pass full result can be retained even before any worker runs. The frozen paper source is unchanged.

## What counts as a repair

The model’s response or a successful process launch is not enough. The portable layer now binds objective results to exact kernel/protected-input hashes and ranks completed full-case results before compact metrics, so a full improvement such as 50/51 → 51/51 cannot be lost when both compact results are 10/10. A complete but partially passing result may be the best available checkpoint; “best” alone is not acceptance. Acceptance still requires all original numerical and integrity checks, the added nonfinite guard and a source-consistent final checkpoint. Read the engine’s task status, trace, summary and complete numerical/integrity results. A repaired kernel must pass the complete required test collection without changing references, wrappers, test inputs or tolerances. Preserve failed and stopped runs for diagnosis.

For an already-correct baseline, engine precision acceptance is `run` exit 0, `accepted_checkpoint.ready: true` and an unlocked terminal success status, without a repair attempt. Legacy `engine_status.objective_success` / `reportable_success` describe repair-final-validation and may remain false when there are zero attempts. Independent `verify` remains a separate numerical verdict.

The skill packages the publicly released precision-repair workflow. CPU adapter tests validate installation, task preparation, command routing and failure handling. They do not establish a successful NPU repair or reproduce the historical 27-task experiment. The versioned paper artifact and its [known limitations](../papers/2026-dont-regenerate-debug/artifact/KNOWN_LIMITATIONS.md) remain separate from this portability layer.
