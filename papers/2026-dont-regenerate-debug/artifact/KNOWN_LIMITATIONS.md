# Known Limitations of the Research Artifact

This frozen artifact supports CPU recomputation from the supplied CSVs and inspection of the supplied method source. It is not the complete Debug-Agent framework or a self-contained environment for rerunning the original NPU repair study. These limits do not change the frozen result rows or method algorithms.

## Numerical results and task assets

The 27-operator manifest records operator identities, compact/full case counts, and case-set hashes. The original task projects, reference `model.py` implementations, compact/full case files, before-repair kernel sources, and complete candidate corpus are not included. The package also lacks the complete per-case numerical execution logs needed to independently revalidate the frozen outcomes.

Ascend C, device/runtime binaries, and provider credentials are external requirements, but supplying them alone does not make this artifact an end-to-end rerun package. Historical `/workspace/...` and other redacted deployment paths can refer to assets that were not released.

## Integrity metadata

The integrity-building script joins supplied automatic verdicts and final-resolution metadata. It checks the structure of that join; it does not rescan every original candidate or rerun the numerical endpoint. Many rows retain a generic shared-criterion explanation, while some retain more specific resolution reasons. The final verdict table can be rebuilt from the packaged inputs, but the complete source-level review cannot be repeated from this artifact alone.

## Token accounting

Token data are task-level aggregates. Raw provider message journals and the detailed records needed to repeat session/agent/message deduplication are absent. In the Kimi result CSVs, the input, output, cache-creation, and cache-read breakdown fields are not populated; uncached and all-token totals are supplied. The scripts can recompute reported aggregates and ratios from these values, not independently recover the provider ledger. Pricing proxies are not auditable invoices.

## Pre-derived shadow and long-failure labels

`results/rq4_no_full_evaluation_shadow.csv` contains the already-derived first-compact-pass shadow. `scripts/derive_no_full_evaluation_shadow.py` summarizes that CSV; it does not replay Full trajectories or independently select the first compact-pass candidate. The original event trajectories needed to repeat that selection are absent.

Likewise, `scripts/reproduce_tables.py` counts the supplied `long_failure` labels rather than deriving them again from frozen threshold evidence. Recomputing these summaries demonstrates arithmetic consistency with the supplied labels, not independent reconstruction of their original derivation.

## Final implementation examples

`diffs/representative_successful_repairs/` contains three final kernel examples. In each, the before/after Python wrappers are identical and `wrapper.patch` is empty. Before-repair kernels are not included, so the directory cannot establish the actual kernel patch or reproduce the failure-to-fix transition. The examples also lack their original reference and test assets. Their `final_kernel/` layout is an inspection layout, not a ready-to-run task directory.

## Historical harness dependencies

The method source and launch scripts retain their historical external-asset assumptions. For example, `code/cannbot/experiment_harness/shared/setup_remote_assets.sh` reads `skills/ascendc/ascendc-debug/references/precision_knowledge_base.json`, which is not supplied. The separate initial fixture (`initial_kb_102_cleaned_20260715.json`) and `old_precision_knowledge_base.json` are retained, but are not substitutes for that missing canonical dependency. This public copy does not fabricate the missing KB or claim to have repaired the NPU rerun chain.

The internal master-design document is also omitted. The public configuration map is in [the ablation contract](code/ablation_protocol/README.md). The asset-path unit test checks that internal document only when it exists; the checks over all packaged control files remain active.

## Source identity and checksum scope

`MANIFEST.sha256` identifies the released file contents. It does not prove which private historical branch revision was executed. `manifests/source_provenance.json` retains source descriptions and digests; the corresponding upstream source records are not bundled or independently resolved by the CPU scripts.

Run checksum verification on an untouched copy. For analysis, use a disposable working copy: `build_final_integrity_results.py` rewrites the canonical final-integrity CSV and its manifest, and the analysis scripts add `reproduced/` outputs. The optional `--refresh-canonical` analysis flag additionally rewrites canonical statistics files. A changed working copy is distinct from the frozen release.
