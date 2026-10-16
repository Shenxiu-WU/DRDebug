# Frozen 27-operator reference and input-case companion

This separate companion supplies the reference programs and input-case
descriptors for the 27-operator cohort used in *Don't Regenerate, Debug: A
Domain-Specific Agent for Repairing Near-Miss Accelerator Kernels*
([paper](https://arxiv.org/abs/2608.02712)). It adds data beside the previously
released `artifact/`; it does not change that frozen artifact or its results.

Supplemental dataset release: 2026-10-16. Its verifier checks the released files
without running NPU experiments, generating candidates, or calling providers.

## Scope

The cohort is fixed by
[`../artifact/manifests/frozen_27_operators.csv`](../artifact/manifests/frozen_27_operators.csv).
It contains L1: 8, L2: 9, L3: 6, L5: 2, L6: 1, and L7: 1 operators.

| Material | Files / cases | Identity check |
| --- | ---: | --- |
| Reference `model.py` | 27 files | 27/27 original frozen byte hashes match |
| Compact input cases | 27 files / 347 rows | 27/27 original byte hashes and normalized case-set hashes match |
| Full input cases | 27 files / 1134 rows | 27/27 normalized case-set hashes match; 26/27 original byte hashes match |

Each `tasks/<level>/<operator>/` contains exactly its frozen `model.py`, the
historical active case filename such as `3_Add.json`, and the full-case filename
such as `3_Add.json.bak`. Six operators have equivalent compact/full case sets;
the separate files preserve their distinct protocol roles. The files contain
shape, dtype, and attribute descriptors, not serialized random tensors or RNG
states. The task code constructs actual inputs in its execution environment.

The single raw-byte exception is the full input file of
`level2/020_FusedRopeWithQkNormAndKvCacheUpdate`: all 58 cases match the frozen
normalized multiset hash, but the original file bytes have not been recovered.
Normalization ignores JSON formatting and row order; neither the original
formatting nor the original row order is certified by that match. Both original
expected and released byte hashes are recorded, without rewriting the expected
value or claiming byte identity.

## Verify on a CPU machine

Python 3.9+ and its standard library are sufficient:

```bash
python3 -B verify_dataset.py
```

The command only reads files. It verifies the complete `MANIFEST.sha256`, the
cohort CSV checksum, all 27 operator identities, all reference/case byte hashes,
all compact/full case counts, and the frozen normalized case-set hashes. It
does not import, evaluate, or run any `model.py`, and does not create outputs.
Finder `.DS_Store` files, `__pycache__` contents, and `.pyc` files are explicitly
ignored as local metadata and are not release files. Other unlisted files are
rejected. Symlinks are always rejected, including inside these metadata paths.

The original normalization canonicalizes each nonempty JSONL row using sorted
keys, UTF-8, `ensure_ascii=False`, and compact separators; it sorts the resulting
rows, retains duplicates, appends one newline per row, and hashes those bytes.
The historical full input-file manifest is not bundled wholesale; only the
selected reference/case expectations and its digest are retained in provenance.
The checksum checks establish released-content identity, not a re-execution of
the original numerical endpoints.

## Execution requirements and limits

Running references requires the appropriate PyTorch environment. Some references
also import `torch_npu` or use Ascend-specific operators and require compatible
Ascend hardware, CANN, and device/runtime packages. This standard-library
verifier does not validate those dependencies or their numerical behavior.

The historical references are unchanged. Six L2 references derive their input
filename from `__file__`; when named `model.py`, their historical loader expects
a `model.json` compatibility alias. An execution harness must select the desired
compact/full cases and supply that alias in a separate working copy. This data
companion does not modify frozen sources or provide a new execution adapter.

Only reference code and case descriptors are included. Near-miss candidates,
optimized or repaired kernels, generated outputs, execution logs, agents,
skills, provider configuration, and the complete experiment runtime are outside
this companion. It is not an end-to-end rerun package. The original artifact's
other limitations still apply; the original task-asset omission is addressed
only by this separately identified companion.

## Attribution and license provenance

The publicly identified upstream project is
[NPUKernelBench](https://github.com/weich97/NPUKernelBench); its
[README](https://github.com/weich97/NPUKernelBench/blob/master/README.md) declares Apache 2.0. On 2026-10-16, the linked `LICENSE` and
the alternative `LICENSE.txt` / `LICENSE.md` URLs returned HTTP 404.
[`LICENSE-APACHE-2.0.txt`](LICENSE-APACHE-2.0.txt) is the standard text retrieved
from [the Apache Software Foundation](https://www.apache.org/licenses/LICENSE-2.0.txt), not an upstream-repository
license file. [`NOTICE`](NOTICE) records this distinction and the unresolved
exact historical upstream revision. Existing source comments remain intact;
the private development project's MIT license is not assigned to these assets.

[`PROVENANCE.json`](PROVENANCE.json) records public source URLs, historical
relative source paths, expected hashes, released hashes, and the single raw-byte
exception. Historical workspace paths are relative provenance identifiers, not
public-repository links. Machine absolute paths, credentials, and server
identifiers are excluded. No claim is made that current upstream HEAD has the
same bytes as these frozen historical files.
