import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _dequant_swiglu_quant_ext as _ext  # noqa: E402


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        weight_scale: torch.Tensor = None,
        activation_scale: torch.Tensor = None,
        bias: torch.Tensor = None,
        quant_scale: torch.Tensor = None,
        quant_offset: torch.Tensor = None,
        group_index: torch.Tensor = None,
        activate_left: bool = False,
        quant_mode: int = 0,
        swiglu_mode: int = 0,
        clamp_limit: float = 7.0,
        glu_alpha: float = 1.702,
        glu_bias: float = 1.0,
    ):
        if weight_scale is None:
            weight_scale = torch.ones(1, x.shape[1], dtype=torch.float32, device=x.device)
        if activation_scale is None:
            activation_scale = torch.ones(x.shape[0], 1, dtype=torch.float32, device=x.device)
        if quant_scale is None:
            quant_scale = torch.ones(1, x.shape[1] // 2, dtype=torch.float32, device=x.device)

        return _ext.run_dequant_swiglu_quant(
            x, weight_scale, activation_scale, quant_scale,
            activate_left, swiglu_mode, clamp_limit, glu_alpha, glu_bias
        )
