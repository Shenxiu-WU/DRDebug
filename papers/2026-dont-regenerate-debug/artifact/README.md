# Frozen Research Artifact

## Purpose

This is a public copy of the frozen research artifact for *Don't Regenerate, Debug: A Domain-Specific Agent for Repairing Near-Miss Accelerator Kernels*. It contains structured outcome CSVs, task-level token aggregates, integrity metadata, deterministic analysis scripts, method and protocol source snapshots for Debug-Agent, Regenerate, and CANNBot-Debug, and selected final kernel implementations.

The package supports two uses:

1. **Statistical recomputation on a standard CPU machine.** The bundled scripts recompute the RQ1--RQ4 counts, paired statistics, factorial effects, token summaries, common-success sensitivity results, and shadow summary from the supplied CSVs. Numerical outcomes, integrity verdicts, shadow labels, and long-failure labels are frozen inputs rather than newly executed measurements.
2. **Method and protocol inspection.** The released source exposes the supplied Agent/Skill prompts, workflow state machine, adaptive scheduling, checkpoint/rollback, objective and full evaluation, integrity scanning, V5 experiment launch controls, Regenerate isolation/generation workflow, and CANNBot-Debug precision-repair workflow.

This package is not a complete environment for rerunning the 27-operator repair study. The original task projects, reference implementations, compact/full case files, before-repair kernels, complete candidate corpus, and raw provider journals are not distributed here. The launch sources retain external asset and runtime assumptions, including a missing canonical precision-KB dependency. Installing an Ascend C toolchain and NPU runtime alone does not fill these gaps. See [Known limitations](KNOWN_LIMITATIONS.md) before using the source for a new experiment.

## Quick Start

Python 3.10 or newer is sufficient for table reproduction; no third-party Python package is required.

Verify the untouched release first, then run the analysis commands in a disposable working copy. The integrity-building step rewrites canonical files in that copy; it does not write only to `reproduced/`.

```bash
python3 scripts/verify_manifest.py
python3 scripts/build_final_integrity_results.py
python3 scripts/reproduce_tables.py
python3 scripts/derive_no_full_evaluation_shadow.py
```

- `verify_manifest.py` checks the released file set and its SHA-256 hashes. Run it before creating analysis outputs.
- `build_final_integrity_results.py` rebuilds `integrity/final_integrity_results.csv` and `integrity/integrity_manifest.json` from the two supplied integrity layers. It does not rescan candidate source or rerun numerical evaluation.
- `reproduce_tables.py` writes `reproduced/summary.json` and `reproduced/tables.md`, and checks the headline numerators. Its optional `--refresh-canonical` flag also rewrites two files in `results/`; it is not needed above.
- `derive_no_full_evaluation_shadow.py` writes a summary to `reproduced/no_full_evaluation_shadow.json` from the already-derived shadow CSV. It does not reconstruct a first-compact-pass trajectory.

Generated `reproduced/` files are outside the release manifest. A strict file-set verification after analysis will therefore require a fresh untouched copy.

## Directory Structure

```text
README.md
KNOWN_LIMITATIONS.md
MANIFEST.sha256
environment/
manifests/
results/
integrity/
scripts/
code/
  README.md
  debug_agent_v5/
    .claude/agents/
    agents/
    skills/ascendc/ascendc-debug/
    skills/ascendc/ascendc-translator/
    experiments/v5_ablation/
    utils/
    translations/
  cannbot/
    skills/
    plugins/ops-direct-invoke/
    experiment_harness/
    translations/
  ablation_protocol/
diffs/
  representative_successful_repairs/
```

## Data Definitions

- `pass_at_1` is true only when the candidate passes full numerical evaluation and receives a final `CLEAN` integrity verdict.
- `uncached_tokens` equals input plus output tokens.
- `all_tokens` additionally includes cache-creation and cache-read tokens.
- RQ1 compares one Debug-Agent end-to-end task per operator with three isolated fresh-regeneration trials per operator.
- RQ3 is a balanced, operator-blocked method-by-model design over the same 27 operators.
- RQ4 uses `full`, `minus_kb`, `minus_diagnostic`, `minus_adaptive_scheduling`, `minus_anti_cheat`, and `minus_full_evaluation_shadow`. The last is a matched shadow derived at the first compact objective pass of the Full trajectory, not an independent run.
- `recorded_cost_proxy` is retained only for within-provider ablation sensitivity and is not treated as an auditable monetary charge.
- Token columns are supplied aggregates. The raw message records needed to repeat provider-ledger deduplication are absent; the Kimi result CSVs do not supply input/output/cache subfield breakdowns.

## Integrity Evidence Layers

`automatic_anticheat_verdicts.csv` preserves the compatibility-aware automatic verdict. `integrity_resolution_reasons.csv` contains the final integrity classification and available resolution metadata. `build_final_integrity_results.py` joins these immutable layers into `final_integrity_results.csv`; it does not overwrite either input. The same final criterion is applied to every method.

## Source and Language Policy

The multilingual Agent and Skill files from the research package are preserved for method inspection. English functional translations accompany the primary prompts and repair protocols; the original-language files define the supplied implementation semantics. The public manifest identifies released file contents, not a verifiable mapping to a private historical revision.

## Scope and Privacy

Provider credentials and hardware/runtime binaries are not supplied. The source retains redacted infrastructure examples and historical deployment paths for inspection; those paths are not evidence that the referenced assets are present. The supported CPU analysis workflow requires no provider account, NPU access, or network connection. Third-party attribution accompanies the bundled source.
