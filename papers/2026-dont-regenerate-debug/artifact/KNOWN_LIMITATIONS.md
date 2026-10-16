# Known Limitations of the Research Artifact

This artifact supports CPU recomputation from the frozen CSVs, inspection of the paper method source, and preparation of the recovered 27 initial repair tasks. The 2026-10-17 public supplement restores task inputs and historical reference examples; it leaves the experimental result rows and engine algorithms unchanged. It does not bundle a self-contained NPU/model execution environment or the original provider journals.

## Numerical results and task assets

The 27-operator manifest records operator identities, compact/full case counts, and case-set hashes. [inputs/](inputs/README.md) now contains the 27 original initial task directories: reference `model.py`, compact/full input files, initial `model_new_ascendc.py` wrappers and 181 kernel files. Their source bytes are checked against the historical frozen input manifest. The separate task-data companion also now recovers all 81 reference/case files byte-for-byte, including the previously normalized-only full-case file.

The complete corpus of later repair/regeneration candidates and per-case execution logs is still not included. Source identity establishes which initial files were frozen; it does not independently reproduce every numerical outcome in the recorded result CSVs.

Ascend C, compatible device/runtime binaries, PyTorch/torch_npu and model access are external requirements. The installable DRDebug adapter prepares a fresh workspace and launches the released single-task engine using supplied tasks; it does not recreate the historical batch infrastructure or certify the original environment. Historical `/workspace/...` and redacted deployment paths are inspection records rather than working public defaults.

## Integrity metadata

The integrity-building script joins supplied automatic verdicts and final-resolution metadata. It checks the structure of that join; it does not rescan every original candidate or rerun the numerical endpoint. Many rows retain a generic shared-criterion explanation, while some retain more specific resolution reasons. The final verdict table can be rebuilt from the packaged inputs, but the complete source-level review cannot be repeated from this artifact alone.

## Token accounting

Token data are task-level aggregates. Raw provider message journals and the detailed records needed to repeat session/agent/message deduplication are absent. In the Kimi result CSVs, the input, output, cache-creation, and cache-read breakdown fields are not populated; uncached and all-token totals are supplied. The scripts can recompute reported aggregates and ratios from these values, not independently recover the provider ledger. Pricing proxies are not auditable invoices.

## Pre-derived shadow and long-failure labels

`results/rq4_no_full_evaluation_shadow.csv` contains the already-derived first-compact-pass shadow. `scripts/derive_no_full_evaluation_shadow.py` summarizes that CSV; it does not replay Full trajectories or independently select the first compact-pass candidate. The original event trajectories needed to repeat that selection are absent.

Likewise, `scripts/reproduce_tables.py` counts the supplied `long_failure` labels rather than deriving them again from frozen threshold evidence. Recomputing these summaries demonstrates arithmetic consistency with the supplied labels, not independent reconstruction of their original derivation.

## Final implementation examples

`diffs/representative_successful_repairs/` contains three final kernel examples. In each, the before/after Python wrappers are identical and `wrapper.patch` is empty. This final-example directory is still an inspection layout, not a ready-to-run task. The corresponding original initial candidates and test assets are now in `inputs/tasks/`. The supplied empty wrapper patches do not by themselves document the complete kernel repair trajectory or independently certify a new failure-to-fix run.

## Historical harness dependencies

The historical batch launch scripts retain external-asset assumptions. The worker’s `archive_tasks/` reference tree is now restored at [code/debug_agent_v5/archive_tasks/](code/debug_agent_v5/archive_tasks/README.md), containing ten historical examples, seven with AscendC kernels. Its Git-tree identity and per-file hashes are supplied; no fresh NPU qualification is asserted.

 For example, `code/cannbot/experiment_harness/shared/setup_remote_assets.sh` reads `skills/ascendc/ascendc-debug/references/precision_knowledge_base.json`, which is not supplied. The separate initial fixture (`initial_kb_102_cleaned_20260715.json`) and `old_precision_knowledge_base.json` are retained, but are not substitutes for that missing canonical dependency. The installable adapter explicitly uses the public initial fixture as a read-only seed; it does not assert that the missing historical canonical dependency has been reconstructed. The batch harness remains a historical source snapshot.

The internal master-design document is also omitted. The public configuration map is in [the ablation contract](code/ablation_protocol/README.md). The asset-path unit test checks that internal document only when it exists; the checks over all packaged control files remain active.

## Source identity and checksum scope

`MANIFEST.sha256` identifies the current released file contents, including the documented public supplement. The previous public artifact remains available at Git tag `paper-2608.02712-v1`; its original manifest is not rewritten by changes in this working revision. It does not prove which private historical branch revision was executed. `manifests/source_provenance.json` retains source descriptions and digests; the corresponding upstream source records are not bundled or independently resolved by the CPU scripts.

Run checksum verification on an untouched copy. For analysis, use a disposable working copy: `build_final_integrity_results.py` rewrites the canonical final-integrity CSV and its manifest, and the analysis scripts add `reproduced/` outputs. The optional `--refresh-canonical` analysis flag additionally rewrites canonical statistics files. A changed working copy is distinct from the frozen release.
