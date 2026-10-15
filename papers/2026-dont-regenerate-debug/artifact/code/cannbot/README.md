# CANNBot Experiment Source Map

`skills/ascendc-precision-debug/` and `skills/ascendc-docs-search/` are the two Skill packages installed for CANNBot-Debug. `skills/` also contains the exact allowlisted generation skills made available to Regenerate; independent diagnostic skills were excluded from that arm by the harness. `plugins/ops-direct-invoke/` contains the generation agents and workflow used by Regenerate.

`experiment_harness/shared/` preserves the consolidated trial preparation, isolation, prompting, execution, validation, token accounting, recovery controls, and protocol tests used by Regenerate and CANNBot-Debug. Its `run_regenerate.py` contains the explicit method dispatch for both arms. `experiment_harness/cannbot_debug_qwen/` and `experiment_harness/cannbot_debug_kimi/` preserve the model-cell preparation/finalization/validation snapshots collected from the experiment host. Runtime manifests, provider files, generated candidates, journals, and credentials are excluded.

The multilingual operational prompts are retained apart from anonymized infrastructure literals. `translations/ascendc-precision-debug.en.md` and `translations/ops-direct-invoke-core-prompts.en.md` provide English functional translations of the two principal method entry points; the original Agent and Skill files remain authoritative.
