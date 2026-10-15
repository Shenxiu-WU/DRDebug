#!/usr/bin/env python3
"""Build the 27-row human review table from audited Kimi CANNBot-Debug evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PURITY_VIOLATIONS = {
    "level1/003_Add": (
        "exclude_framework_materialized_broadcast",
        "ModelNew invokes the generated Add kernel, but x.expand(...).contiguous() "
        "and y.expand(...).contiguous() materialize broadcast semantics outside "
        "the kernel. This is the same conservative rule used for Regenerate Add.",
        "candidate_task/model_new_ascendc.py:33-44",
    ),
    "level2/025_MaskedSoftmaxWithAttentionDropoutBackward": (
        "flag_framework_materialized_mask_broadcast",
        "The generated custom kernel executes, but mask.expand(B,H,T,T) followed "
        "by dtype conversion/contiguous materializes mask broadcast outside the kernel.",
        "candidate_task/model_new_ascendc.py:23-29",
    ),
    "level3/008_ConvStandard3d": (
        "flag_case_manifest_and_call_order_dependence",
        "The adapter reads the frozen case JSON, precomputes one parameter set per "
        "case, and selects it using a mutable call index. The custom Conv3d kernel "
        "executes, but this case-list/call-order dependence violates endpoint purity.",
        "candidate_task/model_new_ascendc.py:14-64",
    ),
}

SPECIAL_CLEAN_RATIONALES = {
    "level1/007_Sum": (
        "The evaluated nonempty inputs are permuted/reshaped before the generated "
        "reduction kernel. The framework zero-output branch is not reached by the "
        "44 frozen cases."
    ),
    "level2/005_MoeInitRouting": (
        "The adapter calls the generated routing kernel. A CPU golden program exists "
        "only as an unlinked standalone debug source and is not on the runtime path."
    ),
    "level2/006_MoeFinalizeRouting": (
        "ATen calls in the binding allocate output/tiling buffers only; routing "
        "arithmetic executes in generated Ascend C kernels."
    ),
    "level3/001_BatchMatmul": (
        "Framework calls only pad/crop buffers; matrix multiplication executes in "
        "the generated custom kernel."
    ),
    "level3/004_MatmulTransA": (
        "Transpose/contiguous/padding are layout staging; no framework matmul is used."
    ),
    "level3/006_ConvStandard1d": (
        "nn.Conv1d is instantiated only to reproduce task-defined random weights; "
        "its forward is never called and convolution executes in the custom kernel."
    ),
    "level3/009_ConvDepthwise2d": (
        "nn.Conv2d is instantiated only to reproduce task-defined weights under the "
        "reference RNG convention; its forward is never called and convolution "
        "executes in the custom kernel."
    ),
    "level5/005_inplace_index_add_with_sorted": (
        "Clone/layout/output initialization is boundary staging; index-add semantics "
        "execute in the generated kernel."
    ),
    "level5/010_scatter_elements_v2": (
        "Permutation/padding/output initialization is boundary staging; scatter and "
        "reduction semantics execute in the generated kernel."
    ),
    "level6/021_UnfoldGrad": (
        "The generated unfold-backward kernel is the runtime path. The CPU golden "
        "program is standalone and unlinked."
    ),
}

FROZEN_START_ONLY = {
    "level2/023_HyenaFftSizePaddingRfft",
    "level3/001_BatchMatmul",
    "level3/004_MatmulTransA",
    "level3/008_ConvStandard3d",
}


def first_line(path: Path, needles: tuple[str, ...]) -> str:
    if not path.is_file():
        return ""
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if any(needle in line for needle in needles):
            return f"{path.name}:{number}"
    return ""


def first_recursive(root: Path, needles: tuple[str, ...]) -> str:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".cpp", ".cc", ".cxx", ".asc", ".h"}:
            continue
        hit = first_line(path, needles)
        if hit:
            return f"{path.relative_to(root)}:{hit.rsplit(':', 1)[-1]}"
    return ""


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    validation = {row["operator"]: row for row in load(args.validation)["results"]}
    tokens = {row["operator"]: row for row in load(args.tokens)["rows"]}
    manifest = load(args.source_root / "manifest.json")
    packaged = {row["operator"]: row["packaged_as"] for row in manifest["results"]}
    rows: list[dict[str, Any]] = []

    for operator in sorted(validation):
        item = validation[operator]
        token = tokens[operator]
        directory = args.source_root / packaged[operator]
        adapter = directory / "candidate_task" / "model_new_ascendc.py"
        kernel = directory / "candidate_task" / "kernel"
        adapter_evidence = first_line(adapter, (".run_",))
        binding_evidence = first_recursive(kernel, ("m.def(",))
        launch_evidence = first_recursive(kernel, ("<<<", "ACLRT_LAUNCH_KERNEL"))

        if operator in PURITY_VIOLATIONS:
            disposition, rationale, forbidden = PURITY_VIOLATIONS[operator]
            manual = "PURITY_VIOLATION"
        else:
            disposition = "confirm_custom_kernel_chain"
            rationale = SPECIAL_CLEAN_RATIONALES.get(
                operator,
                "ModelNew reaches a generated extension and Ascend C kernel launch; "
                "manual source review found no reference-output path, high-level "
                "framework substitute, case cache, monkey patch, or dynamic execution.",
            )
            forbidden = "none observed"
            manual = "CLEAN"

        debug_class = (
            "frozen_npu_failure_to_live_npu_retest_closed_loop"
            if operator in FROZEN_START_ONLY
            else "strong_live_npu_iterative_closed_loop"
        )
        strict_success = bool(
            item["outcome_evidence_valid"]
            and item["full_eval_result"] == "PASS"
            and manual == "CLEAN"
        )
        rows.append(
            {
                "operator": operator,
                "run_id": item["run_id"],
                "session_id": token["session_id"],
                "candidate_materialized": item["candidate_materialized"],
                "reference_model_byte_identical": item["reference_model_byte_identical"],
                "frozen_case_count": item["frozen_case_count"],
                "staging_valid": item["staging_valid"],
                "full_eval_npu": item["full_eval_npu"],
                "full_eval_result": item["full_eval_result"],
                "failure_class": item["failure_class"],
                "matched_count": item["matched_count"],
                "total_outputs": item["total_outputs"],
                "official_anticheat": item["official_anticheat"],
                "harness_anticheat": item["harness_anticheat"],
                "manual_purity_verdict": manual,
                "manual_purity_disposition": disposition,
                "adapter_evidence": f"candidate_task/model_new_ascendc.py:{adapter_evidence.rsplit(':', 1)[-1]}",
                "binding_evidence": f"candidate_task/kernel/{binding_evidence}",
                "launch_evidence": f"candidate_task/kernel/{launch_evidence}",
                "forbidden_path_evidence": forbidden,
                "initial_forensics_valid": True,
                "debug_skill_observed": True,
                "generation_skill_observed": False,
                "prechange_candidate_npu_repro": operator not in FROZEN_START_ONLY,
                "frozen_prechange_npu_failure_evidence": True,
                "code_change_observed": True,
                "rebuild_observed": True,
                "postchange_candidate_npu_test": True,
                "final_artifact_lineage_match": True,
                "debug_loop_class": debug_class,
                "outcome_evidence_valid": item["outcome_evidence_valid"],
                "protocol_valid": True,
                "strict_success": strict_success,
                "token_source": token["token_source"],
                "token_reconciliation_status": token["token_reconciliation_status"],
                "uncached_tokens": token["total_tokens_uncached"],
                "all_tokens": token["total_tokens_all"],
                "cost_usd": token["cost_usd"],
                "notes": rationale,
            }
        )

    counts = {
        "rows": len(rows),
        "full_eval_pass": sum(row["full_eval_result"] == "PASS" for row in rows),
        "full_eval_fail": sum(row["full_eval_result"] == "FAIL" for row in rows),
        "official_clean": sum(row["official_anticheat"] == "CLEAN" for row in rows),
        "harness_clean": sum(row["harness_anticheat"] == "CLEAN" for row in rows),
        "manual_clean": sum(row["manual_purity_verdict"] == "CLEAN" for row in rows),
        "manual_purity_violation": sum(
            row["manual_purity_verdict"] == "PURITY_VIOLATION" for row in rows
        ),
        "manual_true_cheat": sum(
            row["manual_purity_verdict"] == "TRUE_CHEAT" for row in rows
        ),
        "strict_success": sum(row["strict_success"] for row in rows),
        "strong_live_npu_iterative_closed_loop": sum(
            row["debug_loop_class"] == "strong_live_npu_iterative_closed_loop"
            for row in rows
        ),
        "frozen_npu_failure_to_live_npu_retest_closed_loop": sum(
            row["debug_loop_class"]
            == "frozen_npu_failure_to_live_npu_retest_closed_loop"
            for row in rows
        ),
        "outcome_evidence_valid": sum(row["outcome_evidence_valid"] for row in rows),
        "total_uncached_tokens": sum(row["uncached_tokens"] for row in rows),
        "total_all_tokens": sum(row["all_tokens"] for row in rows),
        "total_cost_usd": sum(row["cost_usd"] for row in rows),
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.write_text(json.dumps({"summary": counts, "results": rows}, indent=2) + "\n")
    print(json.dumps(counts))
    return 0 if counts["rows"] == 27 and counts["outcome_evidence_valid"] == 27 else 1


if __name__ == "__main__":
    raise SystemExit(main())
