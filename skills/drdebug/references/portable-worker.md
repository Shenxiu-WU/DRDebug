# Native-host handoff contract

This public adapter governs host handoff and public resource discovery for a **new task**. The host uses its own model and tools; the CLI never calls Claude Code, Codex, Kimi or a provider API internally. Historical method files may mention a worker CLI, settings, model or scheduling commands; those are not instructions to launch a nested agent.

The protected prepared task is `{{TASK}}`; the runtime resource root is `{{RUNTIME}}`. The actual editable task is the pending request's **isolated `candidate_dir`**, not the active prepared task. Read the full file at the request's `instructions` path and its method prompt before acting. Preserve its `request_id`; operate on one pending request at a time.

## Complete one host turn

1. Inspect the deterministic failure, source identity, probe policy and injected KB summary.
2. Follow the original failure-specific diagnostic method. For precision repair, retain Phase A → B → C, required audit sections and every numerical/full-case/integrity guard.
3. Edit only request-allowlisted candidate kernel files and audit outputs. Do not change the protected reference, wrapper, input cases, tolerances, runtime, reports, history, budget or request metadata. Use only host tools and permissions already available for this task. If a required diagnostic cannot be run, report the limitation; do not manufacture a measurement or call it a numerical failure.

Audit files live under the candidate's `precision_tuning/` directory: `precision_tuning/precision_audit_<attempt>.md` and `precision_tuning/candidate_kb_entry.json`, relative to `candidate_dir`. Use the complete paths returned in the request's instructions.
4. Finish the actual diagnosis and patch, then write the host result to the returned `result_file` path **outside candidate source files**. It has `outcome: "completed"` or `"failed"`, the actual `final_response`, and a nonempty `error` when `outcome` is `failed`. Completion describes host work only. Do not add claimed engine success, precision scores or invented turns.
5. Include the exact metadata footer required by the full prompt, with factual values:

```text
[ENGINE_ATTEMPT_METADATA]
fix_type: <stable name of the actual repair direction>
direction_verdict: <initial|continue|switch|uncertain>
direction_reason: <actual evidence for that choice>
probe_status: <skipped|executed|not_applicable>
probe_reason: <actual measurements or policy-consistent reason>
kb_used_ids: <IDs actually used, comma-separated; otherwise none>
```

Submit using `submit WORKSPACE --request-id ID --result RESULT_JSON --npu ID --soc SOC --json`. The engine checks request/source integrity and performs deterministic validation before advancing. Read that response; call `next` for the next packet only while the task remains nonterminal. Do not self-certify a patch or perform engine-owned promotion, rollback or termination.

The saved fresh-task ceilings are five attempts and 86,400 seconds of total wall time, **including pending host time and deterministic commands**. There is no synthetic turn accounting. Preserve budgets, events and terminal outcomes. A separately authorized new trial uses a new workspace and records parent/checkpoint and prior budget usage; old evidence stays intact.

## Phase A: specification and references

Before candidate-kernel inspection, read the protected reference and the template at `{{RUNTIME}}/skills/ascendc/ascendc-debug/references/phase-a-checklist.md`. Complete `[COMPUTATION_DECOMPOSITION]` and `[REFERENCE_IMPL_SPEC]`, including queues/buffers, API contracts, alignment and forbidden patterns. Reuse valid prior static sections only when the original method explicitly permits it.

The request states whether user references were supplied. If so, inspect that exact copied read-only directory, read relevant existing implementations and record applicability/source paths. If none applies, record this and use the bundled `{{RUNTIME}}/archive_tasks/` examples. If no user references were supplied, skip the optional directory.

If no relevant archived example exists, record `public_portable_reference_fallback`, derive the mathematical contract from protected `model.py`, and use the published translator guidance:

- `{{RUNTIME}}/skills/ascendc/ascendc-translator/references/dsl2Ascendc.md`
- `{{RUNTIME}}/skills/ascendc/ascendc-translator/references/dsl2Ascendc_compute_vector.md`
- `{{RUNTIME}}/skills/ascendc/ascendc-translator/references/TileLang-AscendC-API-Mapping.md`
- `{{RUNTIME}}/skills/ascendc/ascendc-debug/references/decomposition_examples/README.md`, then an existing relevant example.

References guide analysis; they never replace the task contract or establish execution success. Do not invent filenames, API signatures or reference measurements.

## Phase B and B+: implementation and probes

Read all candidate tiling structures, kernel classes/entrypoints, helper headers and `pybind11.cpp` launch/workspace logic. Fill `[KERNEL_STEP_TRACE]`. Follow the full method prompt and engine probe policy for representative-case instrumentation; capture measured intermediate values in `[INSTRUMENTATION_FINDINGS]` and `[L5_PROBE]`, or the accurately permitted skip reason. Host access limits do not waive a required gate.

The request also provides `diagnostics.task` and `diagnostics.execution`, separate from the source-only candidate. Necessary compilation, local verification and L5 probes within the authorized repair run there. Read `execution.json` for the pinned Python, environment and exact command argument vectors; refresh its kernel from the current candidate before a new diagnostic build. Retain genuine outputs and follow the engine probe policy. Diagnostic artifacts never count as formal acceptance and are never imported on submit; remove temporary instrumentation from the submitted candidate.

## Phase C: structured comparison

Align the implementation with each specification step. Record API names/counts, buffer origins, source lines and real intermediate values. Explicitly inspect TQue/TBuf flow, reduction workspace initialization, copy alignment/padding, cross-core synchronization and forbidden API patterns. Form a source/evidence-based root cause and minimal repair plan before editing. Retain the original required audit fields and any failure-specific protocol.

## Public resource discovery

Resource paths are under the actual runtime's `skills/ascendc/`, not its historical `.claude/skills/` registration directory. The detailed legacy `AscendC_knowledge/api_reference/` tree and historical `precision_knowledge_base.json` are not bundled. Skip instructions to find them: do not create empty substitutes, alias an old KB or invent missing signatures.

The read-only public seed is `{{RUNTIME}}/experiments/v5_ablation/fixtures/initial_kb_102_cleaned_20260715.json`. Use the engine's injected retrieval summary, do not repeat KB search or write to the seed, and list only genuinely used IDs in the metadata footer. Consult [runtime.md](runtime.md#public-resource-map) for the resource inventory.
