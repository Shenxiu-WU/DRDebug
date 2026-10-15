# V5 Ablation Control

This directory preserves the launch-control source for the historical
27-task V5 ablation experiment. It supports method inspection; it is not a
self-contained public rerun environment. The original task projects, case
files, and deployment control assets are absent. The redacted address and
paths below describe the historical layout rather than an available service.

## Frozen treatment

- Arms: `full`, `no_kb`, `no_diagnostic_evidence`, `no_loopguard`,
  `no_anticheat`, `no_fulleval`, `baseline`
- Model: `qwen3.8-max-preview`
- Provider mode: one fixed provider, five concurrent workers, no quota query
- NPUs: `3,4,5,6,7`
- Task budgets: `max_attempts=5`, `max_turns=240`,
  `soft_task_turns=480`, `max_task_turns=600`
- KB: the included initial snapshot, read-only when enabled

The API credential is intentionally absent. On the server it must exist only at
`/workspace/debug_agent/.secrets/provider_config.json` with mode
`0600`.

## Historical preparation procedure (inspection only)

The following procedure documents the source's deployment assumptions. It
cannot be completed from this package alone. In particular, the CANNBot
asset-copy harness also expects a canonical
`references/precision_knowledge_base.json` that is not included. The included
initial fixture and `old_precision_knowledge_base.json` are retained as
separate inspection assets, not fabricated replacements for that dependency.

Deploy the repository to `/workspace/debug_agent`. Keep frozen experiment
assets in the sibling directory `/workspace/debug_agent_assets`, with the
dataset at `cannbot_debug_inputs_n27_20260723` and this directory's control
files and frozen KB at `v5_ablation_control_20260724`. Write the deployed Git
commit to:

`/workspace/debug_agent_assets/v5_ablation_control_20260724/V5_GIT_COMMIT.txt`

Then run `prepare_v5_runtime.sh`. It builds/starts the container, regenerates
fingerprints, verifies the actual model, proves all ablation profiles, runs the
27-task clean-build and initial-validation preflight across NPU 3-7, and makes
the code, control, source and KB mounts read-only while leaving only `outputs`
writable in the container. Preparation also verifies that every frozen arm
manifest matches the executable profile and launch constants.
An explicit `507015`/`NPU_AICORE_EXCEPTION` preflight result is clean-built
again at most twice; every transient attempt is archived, and a stable
non-infrastructure result becomes the formal initial classification.

During a formal run, code/control-plane/source/KB fingerprints are checked
before and after every arm. Provider environment files live in an ephemeral
directory below `.secrets` and are removed by the supervisor. Each post-hoc
task is rebuilt with `build_ascendc.py --clean` before verification; recognized
NPU infrastructure faults receive two isolated rechecks and block closure if
they remain unresolved. Terminal, observability, model, post-hoc and worker
closure must pass before the next arm starts. After all seven arms,
`utils/analyze_v5_ablation.py` writes full-vs-arm paired success and cost,
full-arm-threshold long-failure, failed-cycle and arm-compliance evidence.

Preparation does not start a formal arm. `launch_v5_ablation.sh` remains locked
until
`/workspace/debug_agent_assets/v5_ablation_control_20260724/ARMED_BY_USER.txt`
contains exactly `START_V5_ABLATION`.

The internal master-design document is not distributed. See the existing
[public ablation contract](../../../ablation_protocol/README.md) for the
released configuration map, and [Known limitations](../../../../KNOWN_LIMITATIONS.md)
for the limits of statistical recomputation and method reruns.
