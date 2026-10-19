"""Reproduce nine isolated failure fixtures in one constructed masked Softmax.

The operator accepts contiguous [rows, cols] binary16 scores and a validity mask,
with at least one valid position per row, and returns binary16 probabilities.
Its repaired implementation uses binary32 internal arithmetic and subtracts the
row maximum before exponentiation. Each fixture reintroduces ONLY its named
defect; these fixtures are not a sequential repair trace with all bugs present.

Binary16 and binary32 rounding are modeled using standard-library struct.
The independent reference uses valid positions only, Python math.exp/math.fsum,
and a final binary16 conversion. The order-dependent sum defect is specifically
a sequential binary16 accumulator, not a claim about every reduction algorithm.

These are constructed inputs and language-independent pseudocode, not historical
repair receipts, an NPU execution, an AscendC implementation, or evidence of
performance or superiority to code regeneration. Every fixture reports a target
row of a batched operator. An all-masked row is outside this example's contract.

Python 3.6+; no dependencies beyond the standard library. Writes strict JSON.
"""

import json
import math
import struct


FIXED_CODE = [
    "x = fp32(load_rows(scores, stride=cols))",
    "s = where(valid, x, -INF)",
    "z = s - max(s, axis=-1)",
    "e = exp_fp32(z)",
    "d = sum(e, axis=-1, accumulator=FP32)",
    "p = fp16(e / d)",
]

BEFORE_LINES = {
    "row_stride": (0, "x = fp32(load_rows(scores, stride=rows))"),
    "padding_mask": (1, "s = where(valid, x, 0)"),
    "exp_overflow": (2, "z = s"),
    "sum_rounding": (4, "d = sum(e, axis=-1, accumulator=FP16)"),
    "exp_underflow": (2, "z = s"),
    "mask_broadcast": (1, "s = where(valid[0], x, -INF)"),
    "reduction_axis": (4, "d = sum(e, axis=0, accumulator=FP32)"),
    "tile_local_sum": (4, "d = sum_per_tile(e, tile=2)"),
    "masked_max": (2, "z = s - max(x, axis=-1)"),
}


def quantize(value, format_code):
    """Round once to IEEE binary16/32; overflow becomes signed infinity."""
    try:
        return struct.unpack(format_code, struct.pack(format_code, value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def fp16(value):
    return quantize(value, "e")


def fp32(value):
    return quantize(value, "f")


def valid_only_reference(flat_scores, rows, cols, row, valid_rows):
    """Independent row access, valid-only normalization, final output rounding.

    Deliberately does not call candidate(), its row-offset calculation, masking,
    or its binary32 numerical pipeline. Inputs are binary16, as contracted;
    output is also rounded to binary16.
    """
    matrix = [flat_scores[start:start + cols]
              for start in range(0, len(flat_scores), cols)]
    assert len(matrix) == rows
    values = matrix[row]
    indices = [index for index, flag in enumerate(valid_rows[row]) if flag]
    anchor = max(values[index] for index in indices)
    numerators = {index: math.exp(values[index] - anchor) for index in indices}
    denominator = math.fsum(numerators.values())
    output = [0.0] * cols
    for index in indices:
        output[index] = fp16(numerators[index] / denominator)
    return output


def ieee_div(numerator, denominator):
    """Model IEEE division even when Python would raise on a zero divisor."""
    if denominator == 0.0:
        if numerator == 0.0 or math.isnan(numerator):
            return math.nan
        return math.copysign(math.inf, numerator * math.copysign(1.0, denominator))
    return numerator / denominator


def accumulate(values, quantizer):
    denominator = 0.0
    first_lost_addition = None
    for term_index, term in enumerate(values):
        updated = quantizer(denominator + term)
        if (first_lost_addition is None and term != 0.0 and
                math.isfinite(denominator) and updated == denominator):
            first_lost_addition = term_index + 1
        denominator = updated
    return denominator, first_lost_addition


def candidate(flat_scores, rows, cols, row, valid_rows, defect=None):
    """Execute the fixed six-line pipeline with at most one isolated defect."""
    if defect is not None and defect not in BEFORE_LINES:
        raise ValueError("unknown isolated defect")
    source_offsets = [index * (rows if defect == "row_stride" else cols)
                      for index in range(rows)]
    loaded_rows = [[fp32(value) for value in flat_scores[base:base + cols]]
                   for base in source_offsets]
    if any(len(values) != cols for values in loaded_rows):
        raise ValueError("constructed fixture must keep even the wrong load in bounds")
    invalid_score = 0.0 if defect == "padding_mask" else -math.inf
    mask_rows = ([valid_rows[0]] * rows if defect == "mask_broadcast" else valid_rows)
    masked_rows = [[value if flag else invalid_score
                    for value, flag in zip(values, flags)]
                   for values, flags in zip(loaded_rows, mask_rows)]
    shifted_rows = []
    for index, masked in enumerate(masked_rows):
        maximum = max(loaded_rows[index] if defect == "masked_max" else masked)
        shifted_rows.append(list(masked) if defect in ("exp_overflow", "exp_underflow")
                            else [fp32(value - maximum) for value in masked])
    exponential_rows = [[fp32(math.exp(value)) for value in shifted]
                        for shifted in shifted_rows]
    accumulator = fp16 if defect == "sum_rounding" else fp32
    if defect == "tile_local_sum":
        divisors = []
        for start in range(0, cols, 2):
            tile = exponential_rows[row][start:start + 2]
            subtotal, unused = accumulate(tile, accumulator)
            divisors.extend([subtotal] * len(tile))
        denominator = divisors
        first_lost_addition = None
    elif defect == "reduction_axis":
        reduced = [accumulate([values[col] for values in exponential_rows], accumulator)
                   for col in range(cols)]
        divisors = [item[0] for item in reduced]
        denominator = divisors
        first_lost_addition = None
    else:
        reduced = [accumulate(values, accumulator) for values in exponential_rows]
        denominator, first_lost_addition = reduced[row]
        divisors = [denominator] * cols
    probabilities = [fp16(fp32(ieee_div(value, divisor)))
                     for value, divisor in zip(exponential_rows[row], divisors)]
    return {
        "source_offset": source_offsets[row],
        "source_offsets": source_offsets,
        "loaded_scores": loaded_rows[row],
        "masked_valid": mask_rows[row],
        "masked_scores": masked_rows[row],
        "shifted_scores": shifted_rows[row],
        "exponentials": exponential_rows[row],
        "denominator_axis": 0 if defect == "reduction_axis" else -1,
        "denominator_scope": ("tile" if defect == "tile_local_sum" else
                              "batch_column" if defect == "reduction_axis" else "row"),
        "denominator": denominator,
        "first_lost_addition_number": first_lost_addition,
        "probabilities": probabilities,
    }


def compact_vector(values):
    """Keep the 4,096-position fixture inspectable without dumping repetitions."""
    if len(values) <= 8:
        return values
    first = values[0]
    if all(value == first for value in values):
        return {"repeat": first, "count": len(values)}
    return {"count": len(values), "first_eight": values[:8],
            "last_eight": values[-8:]}


def checks(probabilities, valid, reference):
    finite = all(math.isfinite(value) for value in probabilities)
    # Exact equality is appropriate only for these deliberately chosen fixtures:
    # both candidate and independent reference return the same binary16 values.
    # This is not a proposed general-purpose NPU precision tolerance.
    return {
        "all_outputs_finite": finite,
        "matches_reference_after_fp16_output_rounding": probabilities == reference,
        "padded_outputs_are_zero": all(
            value == 0.0 for value, flag in zip(probabilities, valid) if not flag),
        "probability_mass": math.fsum(probabilities) if finite else None,
        "padded_probability_mass": math.fsum(
            value for value, flag in zip(probabilities, valid) if not flag)
        if finite else None,
        "maximum_absolute_reference_error": max(
            abs(value - expected) for value, expected in zip(probabilities, reference))
        if finite else None,
    }


def summarized_candidate(trace, valid, reference):
    summary = {
        key: compact_vector(value) if isinstance(value, list) else value
        for key, value in trace.items()
    }
    summary["checks"] = checks(trace["probabilities"], valid, reference)
    return summary


def run_case(case_id, title_zh, explanation_zh, scores, rows, cols, row, valid,
             display_metric):
    if rows <= 0 or cols <= 0 or len(scores) != rows * cols:
        raise ValueError("scores must match a nonempty contiguous [rows, cols] matrix")
    valid_rows = valid if valid and isinstance(valid[0], list) else [valid] * rows
    if (not 0 <= row < rows or len(valid_rows) != rows or
            any(len(flags) != cols or not any(flags) for flags in valid_rows)):
        raise ValueError("every row must have its own mask with at least one valid position")
    valid = valid_rows[row]
    flat_scores = [fp16(value) for value in scores]
    if not all(math.isfinite(value) for value in flat_scores):
        raise ValueError("input scores must be finite after binary16 rounding")
    reference = valid_only_reference(flat_scores, rows, cols, row, valid_rows)
    before = candidate(flat_scores, rows, cols, row, valid_rows, defect=case_id)
    after = candidate(flat_scores, rows, cols, row, valid_rows)
    assert not checks(before["probabilities"], valid, reference)[
        "matches_reference_after_fp16_output_rounding"]
    assert checks(after["probabilities"], valid, reference)[
        "matches_reference_after_fp16_output_rounding"]
    line_index, old_line = BEFORE_LINES[case_id]
    before_code = list(FIXED_CODE)
    before_code[line_index] = old_line
    assert sum(left != right for left, right in zip(before_code, FIXED_CODE)) == 1
    return {
        "id": case_id,
        "title_zh": title_zh,
        "explanation_zh": explanation_zh,
        "input": {
            "dtype": "FP16",
            "shape": [rows, cols], "target_row": row,
            "scores": compact_vector(flat_scores), "valid": compact_vector(valid),
            "valid_rows": [compact_vector(flags) for flags in valid_rows],
        },
        "isolated_changed_line": line_index + 1,
        "input_precision_note": "Original scores are FP16; repaired internal arithmetic is FP32; output is FP16.",
        "before_code": before_code,
        "after_code": FIXED_CODE,
        "display_metric": display_metric,
        "independent_reference": {
            "probabilities": compact_vector(reference),
            "checks": checks(reference, valid, reference),
        },
        "before": summarized_candidate(before, valid, reference),
        "after": summarized_candidate(after, valid, reference),
    }


def json_safe(value):
    """Use explicit labels for IEEE special values, keeping output strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "-Infinity" if value < 0.0 else "Infinity"
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main():
    fixtures = [
        run_case(
            "row_stride", "读错了另一行的数据",
            "输入有2行、每行3项。行偏移误乘行数，第二行从索引2而非3开始读取；"
            "内存访问仍在界内，程序能运行，但输出与目标行参考不符。",
            [0.0, 1.0, 2.0, 2.0, 1.0, 0.0], 2, 3, 1, [True] * 3,
            {"label_zh": "第二行首项权重", "before": "42.2%", "after": "66.5%"}),
        run_case(
            "padding_mask", "空位拿走了83%的权重",
            "将补齐空位填0并不能将其排除。指数函数将0变为1，使空位参与归一化；"
            "掩码改为负无穷后，其权重为0。",
            [-2.0, -3.0, -4.0, 0.0], 1, 4, 0, [True, True, True, False],
            {"label_zh": "空位权重", "before": "83.1%", "after": "0%"}),
        run_case(
            "exp_overflow", "大分数让指数溢出",
            "输入90、89、88可由FP16精确表示，但exp(90)超过FP32范围。"
            "先减行最大值，指数输入变为0、-1、-2，避免Inf/Inf产生NaN。",
            [90.0, 89.0, 88.0], 1, 3, 0, [True] * 3,
            {"label_zh": "输出", "before": "NaN", "after": "66.5% / 24.5% / 9.0%"}),
        run_case(
            "sum_rounding", "本该100%的权重，算出了200%",
            "4096项分数均为0，指数项均为1。顺序FP16累加到2048后，每次再加1"
            "都舍入回2048；改用FP32累加得到4096，避免整行概率翻倍。",
            [0.0] * 4096, 1, 4096, 0, [True] * 4096,
            {"label_zh": "整行权重之和", "before": "200%", "after": "100%"}),
        run_case(
            "exp_underflow", "小分数让指数全变成0",
            "输入-104、-105、-106可由FP16精确表示，但其指数均舍入为FP32的0。"
            "分母为0，0/0产生NaN；先减行最大值后，指数输入为0、-1、-2。",
            [-104.0, -105.0, -106.0], 1, 3, 0, [True] * 3,
            {"label_zh": "输出", "before": "NaN", "after": "66.5% / 24.5% / 9.0%"}),
        run_case(
            "mask_broadcast", "把第一行的掩码用到了每一行",
            "两行有不同的有效位置。将第一行掩码广播到所有行，会让第二行"
            "无效的B位置参与计算，同时错删有效的C位置；应逐行应用完整掩码。",
            [2.0, 1.0, 0.0, -1.0, 2.0, 1.0, 0.0, -1.0], 2, 4, 1,
            [[True, True, False, False], [True, False, True, False]],
            {"label_zh": "第二行无效位置B的权重", "before": "26.9%", "after": "0%"}),
        run_case(
            "reduction_axis", "把不同行的分母加在了一起",
            "每行应独立归一化。若沿batch方向而非行内归约，同一列的不同样本"
            "会共享分母；第二行权重之和约为150%，而非应有的100%。",
            [0.0, 1.0, 2.0, 2.0, 1.0, 0.0], 2, 3, 1, [True] * 3,
            {"label_zh": "第二行权重之和（约）", "before": "150%", "after": "100%"}),
        run_case(
            "tile_local_sum", "每块各算100%，整行却变成200%",
            "一行分两块，每块各含2项。若用各自局部分母归一化，每块都会获得"
            "100%的权重；分母必须汇总整行全部指数项，再归一化。",
            [2.0, 1.0, 0.0, -1.0], 1, 4, 0, [True] * 4,
            {"label_zh": "整行权重之和（约）", "before": "200%", "after": "100%"}),
        run_case(
            "masked_max", "已经屏蔽的空位，仍改变了行最大值",
            "空位的分数已被掩码改成负无穷，但最大值却从原始分数x中求得0。"
            "有效项-104、-105、-106的指数全部下溢为0，造成0/0；"
            "应从已屏蔽的s中求最大值-104，保护有效项的数值范围。",
            [-104.0, -105.0, -106.0, 0.0], 1, 4, 0,
            [True, True, True, False],
            {"label_zh": "输出", "before": "NaN", "after": "有限且空位为0%"}),
    ]
    indexing, masking, overflow, accumulation, underflow, broadcast, axis, tiles, masked_max = fixtures
    assert indexing["before"]["source_offset"] == 2
    assert indexing["after"]["source_offset"] == 3
    assert math.isclose(masking["before"]["checks"]["padded_probability_mass"],
                        0.8310546875, abs_tol=0.0)
    assert not overflow["before"]["checks"]["all_outputs_finite"]
    assert overflow["after"]["checks"]["all_outputs_finite"]
    assert accumulation["before"]["denominator"] == 2048.0
    assert accumulation["after"]["denominator"] == 4096.0
    assert accumulation["before"]["first_lost_addition_number"] == 2049
    assert accumulation["before"]["checks"]["probability_mass"] == 2.0
    assert accumulation["after"]["checks"]["probability_mass"] == 1.0
    assert underflow["before"]["exponentials"] == [0.0, 0.0, 0.0]
    assert underflow["before"]["denominator"] == 0.0
    assert not underflow["before"]["checks"]["all_outputs_finite"]
    assert underflow["after"]["checks"]["all_outputs_finite"]
    assert broadcast["before"]["masked_valid"] == [True, True, False, False]
    assert broadcast["after"]["masked_valid"] == [True, False, True, False]
    assert broadcast["before"]["probabilities"][1] == 0.26904296875
    assert broadcast["after"]["probabilities"][1] == 0.0
    assert axis["before"]["denominator_axis"] == 0
    assert axis["after"]["denominator_axis"] == -1
    assert math.isclose(axis["before"]["checks"]["probability_mass"], 1.5, abs_tol=0.001)
    assert tiles["before"]["denominator_scope"] == "tile"
    assert tiles["after"]["denominator_scope"] == "row"
    assert tiles["before"]["checks"]["probability_mass"] == 2.0
    assert math.isclose(tiles["after"]["checks"]["probability_mass"], 1.0, abs_tol=0.001)
    assert masked_max["before"]["exponentials"] == [0.0] * 4
    assert masked_max["before"]["denominator"] == 0.0
    assert not masked_max["before"]["checks"]["all_outputs_finite"]
    assert masked_max["after"]["checks"]["all_outputs_finite"]
    assert masked_max["after"]["probabilities"][3] == 0.0
    output = {
        "kind": "constructed_cpu_masked_softmax_isolated_repairs",
        "contract": {
            "input": "IEEE binary16, contiguous [rows, cols] scores",
            "output": "IEEE binary16",
            "repaired_internal_arithmetic": "IEEE binary32",
            "validity": "finite input scores; at least one valid position per row",
            "mask_semantics": "invalid positions have exactly zero output",
            "reduction_broadcast": "max/sum use keepdims=True; each output row normalizes independently",
        },
        "isolation": {
            "one_defect_per_before_variant": True,
            "other_five_pseudocode_lines_already_fixed": True,
            "not_a_sequential_repair_trace": True,
        },
        "reference_method": "binary16 input rounding, valid-only math.exp/math.fsum in Python float, then binary16 output",
        "comparison": "Exact equality of rounded outputs for these fixtures only; not a general tolerance policy",
        "output_mass_note": "Final binary16 rounding can make a correct row sum slightly different from one; display percentages may be rounded",
        "fixed_six_line_pseudocode": FIXED_CODE,
        "scope": {
            "constructed_cpu_example": True,
            "python_standard_library_only": True,
            "historical_agent_repair_trace": False,
            "npu_experiment": False,
            "actual_ascendc_execution": False,
            "all_masked_row_supported": False,
            "performance_or_regeneration_comparison": False,
            "general_kernel_correctness_claim": False,
        },
        "fixtures": fixtures,
        "summary": {"isolated_defect_fixtures": 9,
                    "before_reference_matches": 0, "after_reference_matches": 9},
    }
    print(json.dumps(json_safe(output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
