"""Reproduce the constructed precision example on the project website.

This is a CPU rounding model, not an Ascend kernel, agent execution trace,
CANNBench task, or performance measurement. Each addition is explicitly rounded
to binary16 or binary32; both variants keep binary16 inputs and outputs.
Passing these six fixtures is not a general guarantee for FP32 accumulation.
Real kernels still require full correctness, integrity, and device validation.

Run with Python 3.6 or later; no third-party dependencies are required.
"""

import json
import math
import struct


def quantize(value, format_code):
    return struct.unpack(format_code, struct.pack(format_code, value))[0]


def reduce_sum(inputs, accumulator_format):
    accumulator = 0.0
    trace = []
    for value in inputs:
        accumulator = quantize(accumulator + value, accumulator_format)
        trace.append(accumulator)
    return quantize(accumulator, "e"), trace


def main():
    fixtures = [
        ("positive cancellation", [2048, 1, -2048]),
        ("negative cancellation", [-2048, -1, 2048]),
        ("small positive increments", [2048, 1, 1]),
        ("zeros", [0, 0, 0]),
        ("ordinary integers", [1, 2, 3]),
        ("binary fractions", [0.5, 0.25, -0.5]),
    ]
    results = []
    for name, values in fixtures:
        inputs = [quantize(value, "e") for value in values]
        reference = quantize(math.fsum(inputs), "e")
        before, before_trace = reduce_sum(inputs, "e")
        after, after_trace = reduce_sum(inputs, "f")
        results.append({
            "name": name,
            "inputs_fp16": inputs,
            "reference_fp16": reference,
            "before": {"output": before, "trace": before_trace, "passed": before == reference},
            "after": {"output": after, "trace": after_trace, "passed": after == reference},
        })

    before_passed = sum(case["before"]["passed"] for case in results)
    after_passed = sum(case["after"]["passed"] for case in results)
    assert results[0]["before"]["trace"] == [2048.0, 2048.0, 0.0]
    assert results[0]["after"]["trace"] == [2048.0, 2049.0, 1.0]
    assert (before_passed, after_passed) == (3, 6)
    print(json.dumps({
        "kind": "constructed_cpu_rounding_example",
        "input_and_output_dtype": "binary16",
        "change": "accumulator: binary16 -> binary32",
        "cases": results,
        "regression": {"before_passed": before_passed, "after_passed": after_passed, "total": len(results)},
    }, indent=2))


if __name__ == "__main__":
    main()
