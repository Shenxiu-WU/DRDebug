# Runtime and task contract

## What is installed

`install.sh` copies this skill and the public `debug_agent_v5` paper source into a self-contained directory:

```text
drdebug/
├── SKILL.md
├── agents/openai.yaml
├── references/
├── scripts/drdebug.py
└── runtime/
    ├── source/debug_agent_v5/
    ├── manifest.json
    └── provenance/
```

`runtime/manifest.json` records source paths and hashes. In a repository checkout, the launcher can instead resolve the paper source under `papers/2026-dont-regenerate-debug/artifact/code/debug_agent_v5/`. Preparation and repair never edit the source artifact. The documented public supplement adds inputs and references to the artifact; the prior published version remains available at its existing Git tag.

## Task input

Provide a task directory containing:

- `model.py`: the reference `torch.nn.Module` implementation.
- `model_new_ascendc.py`: the existing wrapper calling the compiled AscendC extension.
- `kernel/`: actual `.cpp` / `.h` / `.hpp` sources, including a `pybind11.cpp` binding.
- An operator input JSONL file such as `MyOperator.json`: each nonempty line is an input-case object with an `inputs` field, in the format accepted by the released verifier.
- At least one explicit complete `.json.bak` / `.json.full` case collection, with the same base name as the active JSON. It may equal the compact set. Equivalent backups may coexist; conflicting full collections are rejected. Compact cases need not be a subset of full cases; the launcher records their relation without rewriting either file.

Do not synthesize a missing reference, wrapper or passing record. Use the case JSON stem as `--op-name`, not the directory prefix: `3_Add` for `3_Add.json` in `003_Add/`. The repository now supplies the recovered [27 initial candidates](https://github.com/Shenxiu-WU/DRDebug/tree/main/papers/2026-dont-regenerate-debug/artifact/inputs) separately from the reference-and-case dataset. A dataset entry alone has no candidate kernel.

## Runtime environment

Installation and `doctor` use Python 3.10+ and its standard library. Numerical execution additionally needs:

- Linux with access to an Ascend NPU and working device drivers.
- Compatible CANN/AscendC toolkit, CMake and C++ build tools, with the toolkit environment sourced.
- Matching PyTorch and `torch_npu`; NumPy; the build's pybind headers (Torch-bundled or separately installed).
- A configured and authenticated `claude` executable. Authentication and model access stay with the user. A Codex installation alone does not supply this worker backend.

Use the host vendors' installation guidance for the machine's actual driver, toolkit and Python versions. The installer copies files; it does not install drivers, Python packages, a model account or a benchmark client. Run on the NPU machine or inside its existing compatible container. Remote access is user-provided, not embedded in the skill.

## Portability boundary

Preparation retains the repository-shaped runtime so the paper engine's relative `skills/`, `utils/` and worker paths continue to work. It registers the worker/skill in that workspace and records the public adapter. The installer also includes the recovered historical `archive_tasks/` tree. The adapter prefers relevant archived examples and uses the task's read-only reference plus bundled API documents when no matching case is available; additional user references can be supplied through `--reference-root`.

The 102-entry public fixture is a read-only seed, explicitly selected for this new-task workflow. It is not asserted to be the missing canonical KB used by the historical batch harness. The archive supplies all 27 initial candidates, while historical batch infrastructure, complete later candidate trajectories and hidden benchmark inputs remain outside the package.

The deterministic engine, full-evaluation escalation, numerical checks, integrity guards, attempt budgets and checkpoint/rollback decisions are preserved. Do not alter these to make a portability check pass.

## First failure or resume

`prepare` uses a new output directory and retains the original task. `run` uses the workspace's task and runtime, not the current shell directory. Engine events and validation artifacts are the basis of resume and outcome reporting. Use `status` before rerunning an existing workspace. Do not delete its evidence or reset attempt counts to obtain additional attempts.

An environment preflight is evidence that dependencies were detected, not evidence of a successful kernel repair. CPU tests and mock-worker tests verify the adapter only. Complete acceptance requires the engine's actual device evaluation for the supplied task.
