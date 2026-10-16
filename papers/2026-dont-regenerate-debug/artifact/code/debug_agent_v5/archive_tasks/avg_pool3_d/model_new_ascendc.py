import sys
from pathlib import Path

import torch
import torch.nn as nn

from current_task.model import SCENARIO_BY_SHAPE

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _current_task_ext as _ext  # noqa: E402

SPLIT_MODE_AUTO = 0
SPLIT_MODE_C = 1
SPLIT_MODE_W = 2
SPLIT_MODE_MULTI_W = 3

IMPL_GENERIC = 0
IMPL_REDUCE_D = 1
IMPL_SPLIT_C = 2
IMPL_SPLIT_W = 3
IMPL_MULTI_W = 4


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _resolve_scenario(self, x: torch.Tensor):
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (N, C, D, H, W), got shape={tuple(x.shape)}")

        shape = tuple(int(dim) for dim in x.shape)
        scenario = SCENARIO_BY_SHAPE.get(shape)
        if scenario is None:
            supported = ", ".join(str(case_shape) for case_shape in SCENARIO_BY_SHAPE.keys())
            raise ValueError(f"Unsupported avg_pool3_d input shape {shape}. Supported shapes: {supported}")
        return scenario

    def _compute_output_size(self, d, h, w, kernel_size, stride, padding, ceil_mode):
        k_d, k_h, k_w = kernel_size
        s_d, s_h, s_w = stride
        p_d, p_h, p_w = padding

        if not ceil_mode:
            o_d = (d + 2 * p_d - k_d) // s_d + 1
            o_h = (h + 2 * p_h - k_h) // s_h + 1
            o_w = (w + 2 * p_w - k_w) // s_w + 1
        else:
            o_d = (d + 2 * p_d - k_d + s_d - 1) // s_d + 1
            o_h = (h + 2 * p_h - k_h + s_h - 1) // s_h + 1
            o_w = (w + 2 * p_w - k_w + s_w - 1) // s_w + 1
            if (o_d - 1) * s_d >= d + p_d:
                o_d -= 1
            if (o_h - 1) * s_h >= h + p_h:
                o_h -= 1
            if (o_w - 1) * s_w >= w + p_w:
                o_w -= 1

        return o_d, o_h, o_w

    def _choose_block_c(self, c: int) -> int:
        for candidate in (256, 128, 64, 32):
            if candidate <= c and c % candidate == 0:
                return candidate
        return 0

    def _resolve_split_mode(self, scenario: dict) -> int:
        mode = scenario.get("tilelang_mode", "auto")
        if mode == "split_c":
            return SPLIT_MODE_C
        if mode == "split_w":
            return SPLIT_MODE_W
        if mode == "multi_w":
            return SPLIT_MODE_MULTI_W
        return SPLIT_MODE_AUTO

    def _resolve_impl_mode(self, scenario: dict, c: int) -> int:
        k_d, k_h, k_w = scenario["kernel_size"]
        s_d, s_h, s_w = scenario["stride"]
        p_d, p_h, p_w = scenario["padding"]

        # Keep implementation dispatch identical to TileLang branch choice.
        if k_h == 1 and k_w == 1 and s_h == 1 and s_w == 1 and p_h == 0 and p_w == 0:
            return IMPL_REDUCE_D

        split_mode = self._resolve_split_mode(scenario)
        if split_mode == SPLIT_MODE_C:
            block_c = self._choose_block_c(c)
            if block_c > 0:
                return IMPL_SPLIT_C
            raise ValueError(f"split_c scenario requires divisible channel tiles, got C={c}")

        if split_mode == SPLIT_MODE_W:
            return IMPL_SPLIT_W

        if split_mode == SPLIT_MODE_MULTI_W:
            multi_w_window_w_num = int(scenario.get("multi_w_window_w_num", 1))
            return IMPL_MULTI_W if multi_w_window_w_num > 1 else IMPL_GENERIC

        return IMPL_GENERIC

    def _resolve_split_w_tile_kw(self, scenario: dict, k_w: int) -> int:
        tile_kw = int(scenario.get("split_w_tile_kw", 0))
        if tile_kw <= 0:
            return 0
        return min(tile_kw, k_w)

    def _resolve_multi_w_window_w_num(self, scenario: dict, o_w: int) -> int:
        window = int(scenario.get("multi_w_window_w_num", 1))
        if window <= 1:
            return 1
        return min(window, o_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 5
        assert x.dtype == torch.float32

        n, c, d, h, w = x.shape
        scenario = self._resolve_scenario(x)

        kernel_size = scenario["kernel_size"]
        stride = scenario["stride"]
        padding = scenario["padding"]
        ceil_mode = scenario["ceil_mode"]
        count_include_pad = scenario["count_include_pad"]
        divisor_override = int(scenario["divisor_override"])

        o_d, o_h, o_w = self._compute_output_size(d, h, w, kernel_size, stride, padding, ceil_mode)
        k_d, k_h, k_w = kernel_size
        s_d, s_h, s_w = stride
        p_d, p_h, p_w = padding

        split_mode = self._resolve_split_mode(scenario)
        block_c = 0
        if split_mode == SPLIT_MODE_C:
            block_c = self._choose_block_c(c)
            if block_c == 0:
                raise ValueError(f"split_c scenario requires divisible channel tiles, got C={c}")

        split_w_tile_kw = self._resolve_split_w_tile_kw(scenario, k_w)
        multi_w_window_w_num = self._resolve_multi_w_window_w_num(scenario, o_w)
        impl_mode = self._resolve_impl_mode(scenario, c)

        x_nhwc = x.permute(0, 2, 3, 4, 1).contiguous()
        x_flat = x_nhwc.reshape(n * d * h * w, c)

        y_flat = _ext.run_avg_pool3_d(
            x_flat,
            int(n),
            int(c),
            int(d),
            int(h),
            int(w),
            int(o_d),
            int(o_h),
            int(o_w),
            int(k_d),
            int(k_h),
            int(k_w),
            int(s_d),
            int(s_h),
            int(s_w),
            int(p_d),
            int(p_h),
            int(p_w),
            int(1 if count_include_pad else 0),
            int(divisor_override),
            int(split_mode),
            int(block_c),
            int(split_w_tile_kw),
            int(multi_w_window_w_num),
            int(impl_mode),
        )

        y = y_flat.reshape(n, o_d, o_h, o_w, c).permute(0, 4, 1, 2, 3).contiguous()
        return y
