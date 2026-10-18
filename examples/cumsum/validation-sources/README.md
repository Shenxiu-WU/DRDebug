# Captured native sources / 接纳源码快照

These are immutable source fixtures for the [native-host validation record](../native-host-validation.md). Exact SHA-256 identities are recorded in [native-host-validation.json](../native-host-validation.json). The maintained saved example is [`../repaired/kernel/`](../repaired/kernel/).

`codex/kernel/` captures the eight accepted files from the reference-guided Codex run. Six files equal the public reference byte-for-byte; the other two only omit comments. The executable code matches that reference. Retain these bytes to reproduce the recorded source identity.

Runs whose accepted sources match the maintained example byte-for-byte point directly to `repaired/kernel` in the machine record's `source_fixture` field, without another copy here.

这些目录保存[原生宿主验证](../native-host-validation.zh-CN.md)的不可变源码快照，精确 SHA-256 见[机器记录](../native-host-validation.json)。维护版示例仍在 [`../repaired/kernel/`](../repaired/kernel/)。Codex 快照中六份与公开参考逐字节一致，另两份仅删除注释，执行代码一致。

接纳源码与维护版示例逐字节一致的运行，在机器记录的 `source_fixture` 字段直接引用 `repaired/kernel`，此处不再复制一份。

The [example source notice](../SOURCE_NOTICE.md) and existing file-level notices apply.
