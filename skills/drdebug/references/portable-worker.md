## Public portable adapter (takes precedence only for public resource discovery)

This is a new-task adapter built from the public frozen paper implementation.
The runtime includes the recovered historical reference tree and records its source identity. The public KB seed is not asserted to reconstruct the missing canonical study KB.

For this workspace, the protected task is `{{TASK}}` and the runtime root is
`{{RUNTIME}}`. Follow the original Phase A → B → C order, required audit fields,
and every numerical, full-case, integrity, anti-cheat, and loop guard unchanged.

The outer launcher owns budgets. Fresh-task defaults allow five attempts,
240 turns per complete worker invocation, `--worker-timeout-sec 21600` and an
independent `--deadline-sec 86400`. These are ceilings, not minimum consumption;
finish your assigned work when ready. Six hours provides operational headroom,
not an experimentally established optimum. The worker timeout is clamped to the
remaining total deadline. Saved budgets remain unchanged on resume; older records
without a worker-timeout field retain an implicit 600-second limit. Do not change
budgets or restart a terminal session from the worker. A separately
authorized trial inheriting a reviewed checkpoint uses a new workspace with
parent/checkpoint, prior evidence and budget usage recorded; old events remain intact.

Resolve debug references under `{{RUNTIME}}/skills/ascendc/ascendc-debug/references/`
and translator references under `{{RUNTIME}}/skills/ascendc/ascendc-translator/references/`.
The `.claude/skills/` registration directory is not their reference-file base.
The historical `AscendC_knowledge/api_reference/` tree and its `INDEX.md` are not
bundled, even when a legacy SKILL or DSL document directs you there. Skip those
location instructions; do not repeatedly search for them or create empty substitutes.
Use the published files below and report any unresolved API-information gap.

Phase A reference selection: the prepared resource context at the start of this
agent body states whether user references were actually supplied and gives their
copied read-only location. When supplied, first inspect that exact directory and
read existing implementations relevant to this task; record their source paths
and applicability in `[REFERENCE_IMPL_SPEC]`. If none is relevant, record that
fact and inspect the bundled historical `archive_tasks/` examples. When not
supplied, skip the optional user-reference directory entirely.
Select a relevant implementation and record which source you use;
user references do not replace the bundled historical source or its provenance. If no relevant
case exists, explicitly record `public_portable_reference_fallback` in
`[REFERENCE_IMPL_SPEC]`: derive the operator's mathematical contract from the
read-only task `model.py`, then read the public guidance at
`{{RUNTIME}}/skills/ascendc/ascendc-translator/references/dsl2Ascendc.md`,
`{{RUNTIME}}/skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_vector.md`, and
`{{RUNTIME}}/skills/ascendc/ascendc-translator/references/TileLang-AscendC-API-Mapping.md`.
Use `{{RUNTIME}}/skills/ascendc/ascendc-debug/references/decomposition_examples/README.md`
to select an existing example, and
`{{RUNTIME}}/skills/ascendc/ascendc-debug/references/phase-a-checklist.md` for the field templates. Complete all
original `[COMPUTATION_DECOMPOSITION]` and `[REFERENCE_IMPL_SPEC]` fields,
including queues/buffers, API contracts, alignment, and forbidden patterns,
before Phase B reads the candidate kernel. Never invent a reference execution.

The engine supplies deterministic evidence and owns build/evaluation/acceptance.
By default this adapter gives the worker no shell execution permission. Use
Read/Glob/Grep for evidence, Edit/Write for the current task's C++/header sources,
and the allowlisted audit/candidate-KB files. Do not modify engine-owned reports,
checkpoints, guards, events, reference Python, wrapper, or cases. If an optional
probe requires shell execution, report the permission limitation accurately;
never invent probe results or call it a numerical failure. The user may choose
`run --allow-worker-shell` in a trusted isolated NPU environment. That explicit
choice restores worker shell capability without bypassing permissions or
relaxing the engine's acceptance rules. Record skipped probes as required by
the original audit contract; do not weaken a gate to obtain acceptance.

Knowledge retrieval uses `{{RUNTIME}}/experiments/v5_ablation/fixtures/initial_kb_102_cleaned_20260715.json`
as the public initial KB snapshot in read-only mode; use the engine's injected summary.
`{{RUNTIME}}/skills/ascendc/ascendc-debug/references/precision_knowledge_base.json`
is the unavailable historical canonical KB. Skip legacy instructions to read or
search that path. Do not create it, alias the seed or old KB to it, manually rerun
KB search, or write to the seed.
