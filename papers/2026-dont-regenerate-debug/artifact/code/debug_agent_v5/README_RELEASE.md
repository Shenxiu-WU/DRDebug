# Debug-Agent V5 Release Map

The experiment used the project-scoped Agent definitions in `.claude/agents/`; `ascendc-debug-agent-constructive` was the default workflow selected by the batch launcher. The repository-level `agents/` copies are retained for source provenance. The full debugging protocol and deterministic orchestration live in `skills/ascendc/ascendc-debug/`.

`experiments/v5_ablation/` contains the historical launch, container, manifest, smoke-test, and arm-control source. `utils/run_ascendc_debug_batch_cc.sh` is the batch workflow-engine entry point, while `utils/supervise_ascendc_debug_batch_cc_quota.py` manages task queues and retry cycles. The remaining `utils/` files include objective/full evaluation, classification, V5 preflight, post-hoc observation, analysis, and associated tests. These entry points retain dependencies on omitted experiment assets and are provided for inspection, not as a ready-to-run public experiment.

This public copy identifies its released contents through `MANIFEST.sha256`; it does not claim a verifiable mapping to a private historical branch revision. Infrastructure literals were redacted in the supplied snapshot. Method source, operational comments, and prompt language are retained for inspection. The internal master-design document is omitted; the public configuration map is [the ablation contract](../ablation_protocol/README.md). See [Known limitations](../../KNOWN_LIMITATIONS.md) for the missing data, evidence, and canonical-KB dependencies.


## Public execution supplement (2026-10-17)

`archive_tasks/` now restores ten historical reference examples (seven with AscendC kernels) from the recorded pre-submission Git tree. [../../inputs/](../../inputs/README.md) restores 27 original initial task projects, including byte-verified wrappers, kernels, references and compact/full input files. Their provenance is recorded separately from the unchanged method algorithms and original result records.

The repository-level `drdebug` skill adds an installer, dependency preflight and portable single-task launcher. It prepares a fresh runtime, explicitly uses the public 102-entry initial KB fixture in read-only mode, and retains the released engine’s numerical and integrity acceptance. Historical batch scripts are still source snapshots with their original deployment assumptions. The earlier public artifact is preserved by Git tag `paper-2608.02712-v1`.
