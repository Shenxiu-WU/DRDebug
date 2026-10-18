# Cumsum: native-host validation

[English](native-host-validation.md) · [简体中文](native-host-validation.zh-CN.md) · [Machine-readable record](native-host-validation.json)

This is a **reference-guided integration check** of the public Skill. Each host starts from the original failing paper task and receives the eight published `repaired/kernel/` files separately as a read-only reference. It exercises installation, diagnosis, isolated editing, submission, engine acceptance and independent numerical verification. It does not measure unaided rediscovery or a general repair success rate.

## Measured results

**19 October 2026 · reference-guided · Ascend910B4**

| Host | Original compact | Accepted compact | Accepted full | Independent fresh full cases |
| --- | ---: | ---: | ---: | ---: |
| Codex 0.163.0 | 6/10 | 10/10 | 51/51 | **51/51** |
| Kimi Code 2.2.0 | 6/10 | 10/10 | 51/51 | **51/51** |
| Claude Code 2.1.204 | 6/10 | 10/10 | 51/51 | **51/51** |

Python 3.11.14 · Torch 2.9.0+cpu · torch_npu 2.9.0.post6 · NumPy 1.26.4 · CANN 9.1.0 · determinism on.

The original full evaluation was not run after the failing compact baseline. Each recorded host exited with code 0. The controller's independent pass is recorded separately from host completion.

Codex used a dedicated nonprivileged single-NPU container as its outer execution boundary, with CLI `--sandbox danger-full-access` and an isolated HOME configured for `approval_policy=never`. The inner workspace sandbox could not create its namespace in this container. This records the tested host configuration; DRDebug does not select a host sandbox or permission policy.

Host timings include the actual verification work performed and are not a speed comparison between agents.

Codex and Kimi Code also completed host-side verification of all 51 cases in fresh processes, before the controller's additional clean-workspace run.

The hosts use their own Kimi K3 configuration and tools. A 1M context setting records the intended configuration; this trial does not fill the context window to capacity. DRDebug launches no nested coding-agent worker or provider client.

## What was checked

Skill installation, identical reinstallation and payload identity were checked in new isolated directories for each host. The numerical Python environment and compatible installed Ascend toolchain were reused. The execution files in the machine-readable record match the frozen version tested on the server; ordinary and optimized-Python CPU suites each passed 202 tests with no skips there.

The original task's 15 files and eight initial native sources were verified before repair. Each successful repair requires an actual failing baseline, a native host request, an immutable submitted candidate and source-bound engine acceptance. The compact and full collections are separate original inputs; their references, seed 0 and dtype-specific tolerances remain unchanged.

After the host returns, the controller prepares another workspace from the original protected inputs and the actual accepted native source, performs a clean build, and runs compact, full and all 51 individual full cases in fresh processes. The final verdict also checks nonfinite masks, original comparison, native-module binding, C++ integrity, wrapper AST, protected inputs and unchanged original history. A host reply, an installed Skill or a progress file is not the acceptance verdict.

Exact captured native sources are retained as immutable fixtures where their bytes differ from the maintained example. Source hashes and validation-summary hashes are in the machine-readable record. The paper archive and older [validation records](validation.md) remain unchanged.

## Findings included in this release

- Native `next` / host edit / `submit` replaces the internal Claude Code worker. Model access, transport and permissions belong to the selected host.
- Repair instructions explicitly identify the two allowed audit paths under `precision_tuning/`.
- Kimi Code 2.2.0 rejected the initial host invocation combining `--prompt` and `--auto` before numerical repair started. Noninteractive prompt mode already uses automatic tool execution. The retry omits the conflicting flag and uses a fresh installation and host HOME; the failed setup is retained in the machine-readable history.
- Claude Code 2.1.204 required a read before overwriting an existing audit file. The host recovered and submitted its candidate; the installation guide now explains this editing-tool constraint.
- Known provider credential environment prefixes are filtered from deterministic child processes. Dependency and runtime probes check the actual numerical environment.
- During setup, an earlier completed container left alive was associated with `aclInit 507899 / EL0005 Resource_Busy` in a later container, even with an empty NPU process table. Stopping only that verified-idle owned predecessor restored the unchanged probe. The public Skill performs no container or hardware cleanup; see the installation guide's troubleshooting note.

These measurements qualify the original Cumsum collections in the recorded environment. Use the [current walkthrough](README.md) to establish results on another machine. Larger arbitrary shapes, nonzero-storage-offset views and general empty/overflow/UB-capacity handling remain outside this example's measured scope.
