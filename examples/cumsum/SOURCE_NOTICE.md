# Cumsum example source and licensing

The original failing input is the public frozen paper task at
`papers/2026-dont-regenerate-debug/artifact/inputs/tasks/level1/005_Cumsum`.
Its original files are referenced, not changed by this example. `example.json`
records their exact SHA-256 hashes and sizes.

`repaired/kernel/` contains eight native source files from the separately
frozen, precision-repaired final candidate of the 2026-10-18 Cumsum acceptance
run. It contains no compiled extension, runtime binaries, model transcripts,
API keys or private performance, surrogate, few-for-many or joint-optimization
implementation. These candidate sources and the example preparation script
are distributed under the repository's MIT license, subject to existing
file-level notices.

The original `model.py` and input specifications are NPUKernelBench assets;
their Apache-2.0 attribution and standard license text are retained in the
original task. The exact historical upstream revision is not asserted.
Upstream: https://github.com/weich97/NPUKernelBench .

Staging a replay task copies the original attribution files unchanged, so the
inherited task `SOURCE_NOTICE.md` still describes the original near-miss
kernel input. For the staged kernel, `example-origin.json` explicitly records
`saved precision-repaired candidate`; that statement and this notice describe
the substituted eight files. Other original-task attribution remains valid.

The source pin proves which bytes are distributed. Numerical validity must be
established by a new device verification in the stated deterministic runtime
environment; preparing or installing this example performs no such execution.
