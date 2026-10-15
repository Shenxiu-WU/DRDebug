# Released Source Scope

This directory contains frozen method and experiment-protocol source snapshots supplied for inspection of the reported studies. It preserves the operational hierarchy rather than flattening the implementation into excerpts. These sources are not a complete, self-contained NPU experiment environment; see [Known limitations](../KNOWN_LIMITATIONS.md).

- `debug_agent_v5/` contains the V5 method snapshot: Agent definitions, the `ascendc-debug` Skill, deterministic workflow engine, forensics/gates/integrity scripts, tests, translator references, V5 launch/container controls, batch/supervisor entry points, and evaluation utilities.
- `cannbot/` contains the CANNBot method and experiment-harness snapshot: the precision-debug Skill, docs-search dependency, the Regenerate allowlisted generation skills, the `ops-direct-invoke` plugin/workflow, and the Qwen/Kimi experiment harnesses. Historical harness paths may refer to omitted task data or canonical assets.
- `ablation_protocol/` gives a compact machine-readable map from the six reported configurations to disabled mechanisms.

The original multilingual Agent and Skill prompts are retained because they are part of the executed method. English functional translations of the three primary Debug-Agent prompts/protocol and the CANNBot precision-debug Skill are provided under each method's `translations/` directory. The original source is authoritative for exact execution semantics.

The release manifest identifies the bytes of these snapshots. It does not establish an independently verifiable private branch revision or upstream source-to-experiment mapping. Full private repositories, original task/case assets, complete candidate outputs, credentials, provider journals, remote recovery logs, and hardware/runtime binaries are excluded. The files under `diffs/` provide final implementation examples, not reconstructible before/after kernel repairs.
