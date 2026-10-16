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

Use a compatible Linux Ascend machine or its existing container for actual numerical execution. You need accessible NPU devices, CANN/AscendC, CMake/C++ tools, matching PyTorch/torch_npu, NumPy and pybind headers. The engine also needs an authenticated Claude Code executable and model access; installation into Codex does not replace this worker backend.

For a Codex user install:

```bash
DRDEBUG_SKILL="$HOME/.agents/skills/drdebug"
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" doctor
```

For a Claude Code user install, set `DRDEBUG_SKILL="$HOME/.claude/skills/drdebug"` instead. For a project or custom install, use that installation’s absolute path. `doctor --json` provides a machine-readable report; `--strict` exits nonzero if required runtime checks fail. It does not call a paid model or change the machine.

Use your hardware vendor’s matching runtime setup and your existing model authentication. Do not paste credentials into the skill or source files. On a laptop without an NPU, installation, source inspection and the dependency report still work; numerical repair requires the execution machine.

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

## 4. Prepare a separate workspace

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" prepare /path/to/my-task \
  --output /path/to/my-repair --op-name MyOperator
```

Preparation copies the task and runtime. Originals stay in place. Use a new output directory; an existing workspace is resumed rather than overwritten. If you have independent AscendC examples, add `--reference-root /path/to/reference-cases`.

The prepared runtime keeps the paper engine’s directory structure and registers its Claude worker and skill. The installer includes ten historical reference examples (seven with AscendC kernels). The public adapter first uses relevant archive examples, then the task’s read-only `model.py` and bundled API documents when the archive does not cover the operator. It uses the explicitly named 102-entry public KB fixture as a read-only seed. The recovered archive is recorded by its historical Git tree; the public KB seed is not asserted to reproduce the canonical study KB. See [the runtime contract](../skills/drdebug/references/runtime.md).

From the repository root, you can also choose a recovered paper input, for example Add:

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" prepare \
  papers/2026-dont-regenerate-debug/artifact/inputs/tasks/level1/003_Add \
  --output /path/to/add-repair --op-name 3_Add
```

## 5. Run a bounded repair

Use the SoC identifier accepted by the installed CANN toolkit. Replace `YOUR_SOC_VERSION` below; do not infer it from a benchmark platform label.

```bash
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" run /path/to/my-repair \
  --npu 0 --soc YOUR_SOC_VERSION \
  --max-attempts 5 --max-turns 80 --deadline-sec 1800
python3 "$DRDEBUG_SKILL/scripts/drdebug.py" status /path/to/my-repair
```

An optional `--model MODEL` selects your configured worker model. `run --dry-run --json` shows the planned invocation without calling the model or running NPU evaluation. This is a command review, not an execution test. For a workspace-specific preflight, use `doctor --workspace /path/to/my-repair --npu 0 --soc YOUR_SOC_VERSION --strict`.

The default worker edits kernel sources and designated diagnostic reports; the deterministic engine performs build/evaluation. If a diagnostic probe needs additional worker shell execution, inspect the request and explicitly choose `--allow-worker-shell` only for a trusted, isolated execution workspace. This option does not bypass the agent permission system. Integrity and full numerical acceptance still belong to the engine.

Inspect `status` before rerunning. Resume retains the event log and attempt history. Do not delete evidence or reset budgets to make an unfinished run appear successful. Report missing environment, authentication/provider errors and permission denial separately from kernel failures.

Or ask your agent:

> Use drdebug to check my Ascend environment, prepare a separate workspace for this failing kernel, and repair it from the failure evidence. Use device 0 and stop after five attempts or 30 minutes.

## What counts as a repair

The model’s response or a successful process launch is not enough. Read the engine’s task status, trace, summary and complete numerical/integrity results. A repaired kernel must pass the complete required test collection without changing references, wrappers, test inputs or tolerances. Preserve failed and stopped runs for diagnosis.

The skill packages the publicly released precision-repair workflow. CPU adapter tests validate installation, task preparation, command routing and failure handling. They do not establish a successful NPU repair or reproduce the historical 27-task experiment. The versioned paper artifact and its [known limitations](../papers/2026-dont-regenerate-debug/artifact/KNOWN_LIMITATIONS.md) remain separate from this portability layer.
