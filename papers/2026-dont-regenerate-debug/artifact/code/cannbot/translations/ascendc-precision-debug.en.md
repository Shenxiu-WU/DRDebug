---
name: ascendc-precision-debug
description: English functional translation of the frozen CANNBot Ascend C precision-debug Skill.
translation_status: semantic translation; the original SKILL.md is authoritative
---

# Ascend C Operator Precision Debugging

## Core Principle

Precision debugging combines understanding, analysis, localization, and repair. Understand data-type limits, identify numerical-instability patterns, and use controlled experiments rather than intuition-driven edits.

Use this Skill for failed `rtol`/`atol` checks, all-zero or random outputs, unexpectedly poor FP16 accuracy, incorrect casts, pipeline synchronization faults, and `DataCopy` alignment problems.

## Required Preparation

Before debugging:

1. Freeze a minimal reproducible case with a concrete shape, dtype, and values. Prefer a simple, 32-byte-aligned FP32 case first, then cover boundaries.
2. Cross-check at least two dtypes with the same shape and data when possible.
3. Search official examples and API documentation through `ascendc-docs-search`; do not edit by intuition.
4. Remove stale build/input/output artifacts and rebuild.

## Decision Procedure

1. If an edit does not change output, clear build and kernel caches and confirm that the binary hash changes.
2. Cross-check dtypes:
   - FP32 pass with FP16/BF16 failure suggests conversion, intermediate precision, synchronization, or cast-buffer issues.
   - BF16 pass with FP16/FP32 failure suggests different API fallback paths, overflow ranges, or dtype-specific thresholds.
3. Audit data movement first when output is zero or random:
   - pair `EnQue` and `DeQue` around asynchronous transfers;
   - use `DataCopyPad` for non-aligned transfers;
   - do not use `GlobalTensor.SetValue` as a substitute for local-buffer initialization and copy-out;
   - isolate movement with a CopyIn-to-CopyOut test.
4. Compare the current path with official examples.
5. Classify the remaining error as formula/API selection, boundary/overflow/division, or broad precision loss.

## Symptom Guide

| Symptom | Likely cause | Primary check |
|---|---|---|
| all-zero or random output | missing pipeline synchronization, unaligned copy, invalid global write | queue synchronization and aligned copy-out |
| output sum zero with input-scale error | output was not written | output queue type and copy-out |
| non-aligned cases fail | transfer alignment | `DataCopyPad` |
| FP16 fails while FP32 passes | insufficient intermediate precision | FP32 intermediate computation |
| incorrect values after cast | wrong rounding mode | half-to-float `CAST_NONE`, float-to-half `CAST_ROUND` |
| BF16 passes alone | fallback/API path difference or range/threshold difference | compare compile-time branches and dtype thresholds |
| source edit has no effect | stale binary or compiler cache | clean and verify binary hash |

## FP32 Pass, FP16/BF16 Fail

Treat this as evidence that the high-level algorithm may be correct but the half-precision path is not.

1. Verify that all sensitive intermediate operations are promoted to FP32.
2. Check cast directions and rounding modes.
3. Check queue synchronization after additional cast stages.
4. Confirm that tiling reserves sufficient FP32 cast buffers.

## BF16 Pass, FP16/FP32 Fail

Investigate two alternatives:

1. A BF16-specific fallback is simpler and correct while another path uses an API with incorrect mask, repeat, or stride parameters.
2. FP16 overflows because of its smaller exponent range, or verification thresholds are not dtype-aware.

Search compile-time BF16 branches, enumerate APIs unique to each path, temporarily align paths for diagnosis, and consult official support tables.

## Pipeline Synchronization

`DataCopy` and `DataCopyPad` are asynchronous. Allocation does not make copied data ready. The standard pattern is:

```cpp
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
inQueue.EnQue(x);
LocalTensor<T> xIn = inQueue.DeQue<T>();
Compute(xIn);
```

A temporary `PipeBarrier<PIPE_ALL>()` may confirm a synchronization hypothesis, but the final repair should use the correct queue/event mechanism because a global barrier is expensive.

## Common Traps

- unpaired async transfers;
- non-aligned copies;
- direct global-tensor scalar initialization;
- wrong cast mode;
- FP16 reduction/accumulation;
- unstable exponent/log formulas;
- catastrophic cancellation;
- uninitialized reduction buffers;
- missing epsilon protection;
- output queues declared as `VECIN` rather than `VECOUT`;
- double-buffer capacity omitted from thresholds.

## SIMT-specific Checks

SIMT paths add data-cache coherency, shared-memory, and cross-core synchronization concerns. Use `DataCacheCleanAndInvalid` where required, `AscendC::Simt::printf` inside vector functions, and explicit synchronization before workspace consumers read producer data.

## Debugging Strategy

Use fast methods first:

1. error-distribution analysis;
2. targeted `printf`;
3. the seven-step `DumpTensor` workflow across CopyIn, Compute, and CopyOut;
4. the common-trap checklist.

Do not continue blind trial-and-error beyond seven attempts. If fast methods stop producing evidence, switch immediately to binary-search debugging and reduce the computation to the first divergent stage.

## Localization Methods

- Compare line by line with a working official example.
- Find passing/failing boundary values and inspect branch selection.
- Calculate actual values rather than trusting approximate formulas.
- Check buffer class, size, double-buffer factors, and reuse.

## Precision-standard Priority

1. explicit precision requirements in the operator-development plan;
2. official Ascend precision standards;
3. Skill defaults only as fallback.

Default fallback thresholds are FP16 `rtol=1e-3`, `atol=1e-4`; FP32 `rtol=1e-5`, `atol=1e-6`; integer `atol=0`.

## Completion Checklist

- minimal reproducible case fixed;
- API use confirmed through documentation;
- caches cleared;
- pipeline synchronization audited;
- all-zero output causes checked;
- official implementation compared;
- attempt count tracked and binary search used when needed;
- final repair summarized with the confusing documentation, improvement opportunities, and diagnostic record.

The original Skill contains detailed links to the bundled workflow, binary-search, common-trap, data-comparison, `printf`, `DumpTensor`, tool-reference, case-study, and summary-template documents.
