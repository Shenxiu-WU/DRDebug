# Historical operator reference examples / 历史算子参考案例

This companion restores the `archive_tasks/` reference tree named by the paper’s worker and AscendC translator documents. It is included in the paper artifact as a documented public supplement; the engine algorithms and experimental result rows are unchanged.

补充论文工作智能体和 AscendC 文档引用的 `archive_tasks/`。按历史源码身份补入论文归档，公开更新记录与校验清单；引擎算法和原始实验结果行保持不变。

| Examples | Available material |
| --- | --- |
| `avg_pool3_d`, `circular_pad`, `gather_elements_v2`, `matmul_leakyrelu`, `quant_matmul`, `reshape_matmul_rowwise_quant_int8`, `rms_norm` | AscendC kernels, wrappers/references and design files |
| `flash_attention`, `flash_attention_gqa`, `sparse_attention` | Reference/TileLang/design examples; no AscendC `kernel/` in the recovered tree |

All 117 source files are copied byte-for-byte from historical Git commit `ec3c277672dec41d23eae33d04c187f3e8ce0fed` (2026-05-13). The archive tree hash is `f453be76a4b1685b0f434b729b3c560eb968c022`, also matching the independently retained local source tree. [Provenance and per-file hashes](PROVENANCE.json) record the recovery; this does not prove which references every historical experiment read or certify fresh device execution.

117 个源码文件逐字节恢复自论文提交前的历史 Git 对象。恢复包包含 10 个案例，其中 7 个带 AscendC 内核；另外 3 个是参考/TileLang/设计案例，不将其描述为可直接运行的 AscendC 示例。

The DRDebug installer includes this tree in its runtime’s `archive_tasks/`. New tasks may still require an operator-specific reference from the user when the archive does not cover that operator.

The source repository’s [Apache-2.0 license](LICENSE-APACHE-2.0.txt) is preserved. Individual copyright notices remain in the files. The root project’s MIT license does not replace this license. Verify all supplied files with `MANIFEST.sha256`; verification reads source files without executing them.
