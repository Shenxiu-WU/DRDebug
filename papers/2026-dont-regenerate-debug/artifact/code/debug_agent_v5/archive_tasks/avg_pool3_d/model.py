import torch
import torch.nn as nn
import torch.nn.functional as F


SCENARIOS = [
    {
        "name": "avg_pool3_d_ncdhw_big_kernel",
        "shape": (2, 16, 32, 32, 32),
        "kernel_size": (8, 8, 8),
        "stride": (4, 4, 4),
        "padding": (2, 2, 2),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 0,
    },
    {
        "name": "avg_pool3_d_ncdhw_normal",
        "shape": (4, 16, 8, 16, 16),
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 0,
    },
    {
        "name": "avg_pool3_d_ncdhw_reduce_d",
        "shape": (2, 32, 16, 32, 32),
        "kernel_size": (3, 1, 1),
        "stride": (2, 1, 1),
        "padding": (1, 0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 0,
    },
    {
        "name": "avg_pool3_d_ndhwc_multi_w",
        "shape": (2, 64, 16, 16, 16),
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 0,
        "tilelang_mode": "multi_w",
        "multi_w_window_w_num": 2,
    },
    {
        "name": "avg_pool3_d_ndhwc_split_c",
        "shape": (2, 512, 8, 8, 8),
        "kernel_size": (3, 3, 3),
        "stride": (1, 1, 1),
        "padding": (1, 1, 1),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 0,
        "tilelang_mode": "split_c",
    },
    {
        "name": "avg_pool3_d_ndhwc_split_w",
        "shape": (2, 32, 8, 16, 64),
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 4),
        "padding": (0, 0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 0,
        "tilelang_mode": "split_w",
    },
]

SCENARIO_BY_SHAPE = {scenario["shape"]: scenario for scenario in SCENARIOS}


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _resolve_scenario(self, x: torch.Tensor):
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (N, C, D, H, W), got shape={tuple(x.shape)}")

        shape = tuple(int(dim) for dim in x.shape)
        scenario = SCENARIO_BY_SHAPE.get(shape)
        if scenario is None:
            supported = ", ".join(str(case["shape"]) for case in SCENARIOS)
            raise ValueError(f"Unsupported avg_pool3_d input shape {shape}. Supported shapes: {supported}")
        return scenario

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scenario = self._resolve_scenario(x)
        divisor_override = scenario["divisor_override"] or None
        return F.avg_pool3d(
            x,
            kernel_size=scenario["kernel_size"],
            stride=scenario["stride"],
            padding=scenario["padding"],
            ceil_mode=scenario["ceil_mode"],
            count_include_pad=scenario["count_include_pad"],
            divisor_override=divisor_override,
        )


def get_input_groups():
    input_groups = []
    for scenario in SCENARIOS:
        n, c, d, h, w = scenario["shape"]
        x = torch.rand(n, c, d, h, w, dtype=torch.float32)
        input_groups.append([x])
    return input_groups

def get_init_inputs():
    return []
