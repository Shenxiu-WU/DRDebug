## Public portable adapter (takes precedence only for resource locations)

This is a new-task adapter built from the public frozen paper implementation.
The runtime includes the recovered historical reference tree and records its source identity. The public KB seed is not asserted to reconstruct the missing canonical study KB.

For this workspace, the protected task is `{{TASK}}` and the runtime root is
`{{RUNTIME}}`. Follow the original Phase A → B → C order, required audit fields,
and every numerical, full-case, integrity, anti-cheat, and loop guard unchanged.

Phase A reference selection: first inspect the bundled historical `archive_tasks/`
and any user-supplied cases in the separate `archive_tasks/user_references/`
directory, and select a relevant implementation. Record which source you use;
user references do not replace the bundled historical source or its provenance. If no relevant
case exists, explicitly record `public_portable_reference_fallback` in
`[REFERENCE_IMPL_SPEC]`: derive the operator's mathematical contract from the
read-only task `model.py`, then read the public translator/API references at
`skills/ascendc/ascendc-translator/references/dsl2Ascendc.md`,
`dsl2Ascendc_compute_vector.md`, and `TileLang-AscendC-API-Mapping.md`.
Use the public decomposition/checklist guidance as applicable. Complete all
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

Knowledge retrieval uses the explicitly selected public 102-entry seed in
read-only mode. Do not search for, synthesize, or claim the unavailable original
canonical KB. Do not manually rerun KB search or write to the seed.
