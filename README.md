# DRDebug: Don’t Regenerate, Debug

Research artifacts and a project website for the DRDebug research series. DRDebug expands to **Don’t Regenerate, Debug**.

[Project website](https://Shenxiu-WU.github.io/DRDebug/) · [Paper](https://arxiv.org/abs/2608.02712)

Debug Agent starts from an accelerator kernel that nearly works, diagnoses its failures, and applies targeted repairs under deterministic validation.

## Research

| Paper | Public materials |
| --- | --- |
| **Don't Regenerate, Debug: A Domain-Specific Agent for Repairing Near-Miss Accelerator Kernels** (2026) | [arXiv](https://arxiv.org/abs/2608.02712) · [PDF](https://arxiv.org/pdf/2608.02712) · [Code and results](papers/2026-dont-regenerate-debug/artifact/) · [Tasks and test cases](papers/2026-dont-regenerate-debug/dataset/) |

The paper studies 27 frozen near-miss NPUKernelBench operators. This repository contains the paper's precision-repair methods, experiment protocols, released result records and table-reproduction scripts. Each paper's artifact is stored separately so that subsequent publications can be added without replacing earlier releases.

The website also presents a dated public CANNBench correctness snapshot. Those later community-benchmark results are separate from the paper experiments; this frozen paper artifact does not reproduce the current leaderboard pipeline.

## Reproduce the paper tables

Python 3.10+ is sufficient. No third-party Python packages, API credentials or accelerator hardware are required for this step.

```bash
python3 tools/reproduce_paper.py
```

This checks the artifact manifest and runs the paper's analysis scripts in a temporary copy. Results are saved under `reproduced/`; the released records are unchanged. Choose a fresh location for a second run with `--output PATH`.

The release supports **recomputing statistics from frozen records and inspecting method code**. It is not a ready-to-run distribution of the original NPU experiment environment. The companion [dataset](papers/2026-dont-regenerate-debug/dataset/) provides the 27 frozen reference implementations and compact/full input-case specifications. The original candidate-kernel corpus, full session ledgers and machine-specific runtime assets remain outside this release. See the artifact's [known limitations](papers/2026-dont-regenerate-debug/artifact/KNOWN_LIMITATIONS.md).

## Paper task dataset

The companion dataset contains 27 reference implementations, 347 compact input-case specifications and 1,134 full input-case specifications. Reference bytes and normalized case-set hashes are checked against the frozen paper manifests, without importing or executing the reference code. See its [README](papers/2026-dont-regenerate-debug/dataset/README.md) for provenance, the one documented raw-byte difference, dependencies and license attribution. The original tagged code/data artifact is unchanged.

```bash
python3 -B papers/2026-dont-regenerate-debug/dataset/verify_dataset.py
```

## Website

The static website is maintained directly in this repository's `website/` folder, the sole source for website updates. To preview it locally:

```bash
python3 -m http.server 8000 --directory website
```

Open `http://localhost:8000`. The site supports Chinese and English, light and dark themes, and responsive layouts. GitHub Pages deploys only this folder through the included workflow.

## Release policy

This repository is a public research archive and showcase. Paper artifacts are frozen and tagged; publication fixes are recorded in their release-change manifest. Website updates are published separately from paper artifacts.

## Citation

```bibtex
@misc{sun2026dontregeneratedebug,
  title={Don't Regenerate, Debug: A Domain-Specific Agent for Repairing Near-Miss Accelerator Kernels},
  author={Yansong Sun and Shenxiu Wu and Siyuan Chen and Runlin Hou and Junhao Qiu and Junming Cao and Shudi Shao and Zhichao Lu and Qingfu Zhang},
  year={2026},
  eprint={2608.02712},
  archivePrefix={arXiv},
  primaryClass={cs.SE},
  url={https://arxiv.org/abs/2608.02712}
}
```

## Licenses

Project-authored software and website code use the [MIT license](LICENSE). Third-party material retains its own licenses and notices, including CANN Open Software License v2 and the licenses recorded for reference templates. Paper text and figures remain attributed to their authors. See [third-party notices](THIRD_PARTY_NOTICES.md); the root MIT license does not replace those terms.
