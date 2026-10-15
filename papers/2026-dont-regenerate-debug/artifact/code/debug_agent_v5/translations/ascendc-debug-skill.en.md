---
name: ascendc-debug
description: English functional translation of the frozen Debug-Agent V5 Skill protocol.
translation_status: semantic translation organized for review; the original SKILL.md is authoritative
---

# Ascend C Debugging Workflow

## 1. Purpose and Ownership

This Skill defines one repair attempt for an existing Ascend C operator. It covers build, import, runtime, timeout, and precision failures. The deterministic Python engine owns attempt numbering, state transitions, budgets, checkpoint/rollback, validation, final artifacts, and continue/stop decisions. The Agent owns semantic diagnosis, repair planning, and kernel edits.

The discovery Agent invokes this Skill directly. The constructive Agent uses the same state machine but must first establish a reference implementation specification.

## 2. Inputs and Failure Source of Truth

The task directory must contain the reference model, generated wrapper, native kernel project, case files, and `.verify_status/latest.json`. If the status file is absent, run the official evaluator and classifier before the Skill starts.

The status record is the only failure-type source of truth. Route exactly as follows:

- `precision_failed` -> precision forensics and numerical repair;
- `build_failed` -> compiler diagnosis;
- `import_failed` with kernel-side classification -> extension/export repair;
- `runtime_error` -> runtime and crash diagnosis;
- `timeout` -> synchronization, tiling, and non-progress diagnosis;
- environment-side import failures -> `skipped_env_issue`;
- unsupported or already successful states -> `skipped_unsupported_type`.

Failure types may change after validation. The next attempt must use the newly observed type rather than stale state.

## 3. Attempt Sequence

Each attempt follows the closed sequence:

1. load state and failure evidence;
2. snapshot protected files for integrity checking;
3. retrieve knowledge when enabled and log every retrieved/injected/declared-used identifier;
4. collect deterministic forensics when enabled;
5. ask the Agent to diagnose and edit the kernel;
6. run Gate-A on the audit and integrity evidence;
7. build and run compact objective evaluation;
8. run Gate-V and update the current-best checkpoint;
9. when compact evaluation passes and a larger case set differs, restore the larger case set and run full evaluation;
10. run anti-cheat/integrity checks;
11. transition to success, another attempt, rollback, or a terminal outcome.

An action may not be skipped because the Agent believes the candidate is good enough. The engine records each action and transition in the event log.

## 4. Evidence Collection

### Precision forensics

The deterministic report contains:

- case-level match rate and maximum/mean error;
- worst coordinates and localized slices;
- sign, magnitude, periodicity, tail, and distribution patterns;
- shape, stride, memory-layout, and alignment facts;
- operator type, attributes, and reduction axes;
- links to compile/runtime evidence when direct probing is unavailable.

For deterministic build or runtime failures, an existing compiler/runtime log is valid degraded forensics. Preserve and pass it to the Agent rather than repeating the same failing prebuild three times. Transient I/O or process-launch failures may be retried separately.

### Knowledge retrieval

When enabled, perform the initial and targeted searches through the deterministic script. Both calls require a log path and call index. Record retrieved, injected, and Agent-declared-used IDs separately. A candidate may be proposed after a valid success, but publication occurs only through deterministic finalization.

### Direction history

Read `tuning_directions.json` before per-round details. Each direction records fix type, intended target, result, and improvement. The Agent's direction-continuity answer is binary. Repeating a disproven direction without new evidence is rejected.

## 5. Branch Contracts

### 5.1 Precision branch

The audit must map measured evidence to the kernel path and state a minimal `FIX_PLAN`. It must examine:

- logical versus padded shapes and strides;
- tail handling and 32-byte transfer alignment;
- TQue/TBuf allocation, enqueue/dequeue, and buffer aliasing;
- synchronization between DMA, vector, scalar, and cross-core stages;
- reduction identities and work-buffer initialization;
- accumulation data types, cast modes, constants, overflow, and cancellation;
- host/tiling/kernel agreement on dimensions, offsets, and core partitions.

Constructive audit additionally requires the reference specification and step trace before candidate inspection. Discovery audit may localize from failure evidence first but must still document the relevant contracts.

### 5.2 Build branch

Use the compiler log as evidence. Identify the first causal error rather than downstream diagnostics. Verify includes, namespaces, signatures, templates, supported types, and kernel/host interfaces. The repair must remain in the native kernel project.

### 5.3 Import branch

For kernel-side failures, reconcile extension/module naming, exported entry points, pybind declarations, registration, shared-library placement, and symbol visibility. Do not edit the environment to hide an implementation error. Environment-side failures terminate as infrastructure outcomes.

### 5.4 Runtime branch

Use the runtime log and crash signal. Audit pointers, shapes, offsets, aligned accesses, launch dimensions, assertions, and device error codes. When deterministic runtime evidence already exists, treat it as degraded forensics and proceed to diagnosis.

### 5.5 Timeout branch

Inspect loop bounds and progress, core partitioning, queue/flag pairing, `SyncAll`, and tiling. Separate a deadlock or infinite loop from an unusually slow but progressing kernel.

## 6. Gate-A: Audit and Integrity Gate

Gate-A validates the required sections for the active branch and verifies that the proposal is evidence-backed. For precision, it checks the reference specification where required, kernel-step mapping, direction assessment, root cause, fix plan, and declared knowledge use. It also verifies that protected files remain unchanged.

Gate-A extracts compact section artifacts and writes the round summary. The Agent must not fabricate or overwrite these machine-generated records.

## 7. Validation and Gate-V

Run the official build/evaluator with captured stdout/stderr. The validation result must preserve:

- original exit code;
- exact correctness boolean;
- parsed match rate and maximum difference;
- compact or full coverage identity;
- failure type observed after the run;
- case-set fingerprint.

Gate-V returns only:

- `PASS`: the current coverage endpoint passed;
- `CONTINUE`: another repair action is justified;
- `STOP`: no permitted continuation remains.

STOP includes exhausted attempt/task budget, harmful oscillation after a historical best, and the configured near-success condition. The engine, not the Agent, interprets this signal.

## 8. Full Evaluation

When the active JSON is a compact subset and a `.json.bak` or `.json.full` file has broader or non-equivalent cases, full evaluation is mandatory after compact success. Equality must be based on normalized case content or case-set fingerprints, not case count alone. If the case sets are identical, record `coverage_equivalent=true` and avoid duplicate execution.

A task is reportably successful only after the required full coverage and integrity endpoint pass.

## 9. Adaptive Scheduling and Budgets

The workflow enforces:

- a per-Agent-call turn cap;
- a soft task budget that can trigger evidence-based early stopping;
- a hard task budget;
- a maximum repair-attempt count;
- branch and wall-time limits.

Adaptive scheduling uses observed progress, failure-type transitions, direction history, and current-best state. The `minus_adaptive_scheduling` configuration disables semantic early-stop/soft-budget behavior but retains hard safety limits, validation, integrity checks, and the attempt cap.

## 10. Current Best and Rollback

After a valid improvement, checkpoint the candidate source and evaluation evidence. Track the current and historical best by objective quality under the same coverage. Before a terminal regression, restore the best valid candidate when the rollback contract allows it. Record source hashes, checkpoint identity, trigger, and restored evaluation.

Rollback cannot turn an unverified candidate into success; the restored candidate must pass the required endpoint.

## 11. Integrity Policy

The repair target is the native Ascend C implementation. The reference model and generated wrappers are protected.

The deterministic scanner records a baseline, verifies hashes, parses wrappers, and checks that:

- the compiled extension is imported;
- the wrapper actually calls the kernel;
- no framework computation replaces part or all of the kernel;
- no scalar Python loop substitutes for device execution;
- no forbidden file is altered.

The automatic verdict is preserved separately from the final source-evidence integrity classification used for reporting.

## 12. Isolation

The Agent may inspect only the current task and designated historical references under `archive_tasks/`. It may not inspect sibling output directories, previous results for the same benchmark split, or unrelated candidates. This prevents solution leakage.

## 13. Exit Artifacts

Every terminal path writes:

### `debug_status.json`

Machine-readable fields include outcome, failure type, attempts, objective/full-evaluation state, integrity state, final source hashes, budgets, tokens, cost proxy, checkpoint/rollback metadata, knowledge use, direction history, and crash or stop reason.

### `debug_trace.md`

The trace contains:

1. initial failure and evidence;
2. attempts, directions, edits, and measured results;
3. final validation/integrity decision;
4. unresolved cause or success summary.

Required outcomes include success, failed, stopped by gate, stopped by loop or budget, progressed to a new failure type, skipped environment issue, skipped unsupported type, and crashed.

## 14. Ablation Switch Semantics

- `minus_kb`: no knowledge retrieval, injection, declaration, or publication.
- `minus_diagnostic`: disables probe, forensics, and knowledge evidence as one bundled diagnostic-evidence mechanism; direct status/log evidence remains available.
- `minus_adaptive_scheduling`: disables semantic early stopping and the soft task budget, while preserving hard limits.
- `minus_anti_cheat`: disables the automatic integrity mechanism for the arm; evaluation outcomes remain separate.
- `minus_full_evaluation_shadow`: represents the first compact objective success along the Full trajectory and is derived post hoc rather than executed as an independent arm.

Each switch must be recorded at launch, propagated to the engine, and observable in per-task artifacts.

## 15. Communication and Platform

The executed prompts require Chinese reasoning/status text and English code/identifiers. The implementation targets CANN 8.1 RC1 or newer and Ascend 910B using Ascend C APIs. TBE/Tik syntax and host/framework fallbacks are prohibited.
