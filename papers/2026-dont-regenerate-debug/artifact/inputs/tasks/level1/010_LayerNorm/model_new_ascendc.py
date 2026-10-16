import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _layer_norm_ext as _ext  # noqa: E402


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        normalized_shape: list,
        weight: torch.Tensor = None,
        bias: torch.Tensor = None,
    ) -> torch.Tensor:
        # Compute total normalized dim
        norm_dim = 1
        for s in normalized_shape:
            norm_dim *= s

        original_shape = x.shape
        x_2d = x.reshape(-1, norm_dim).contiguous()

        # Default weight = ones, bias = zeros if not provided
        if weight is not None:
            weight_1d = weight.reshape(-1).contiguous()
        else:
            weight_1d = torch.ones(norm_dim, dtype=x.dtype, device=x.device)

        if bias is not None:
            bias_1d = bias.reshape(-1).contiguous()
            has_bias = 1
        else:
            bias_1d = torch.zeros(norm_dim, dtype=x.dtype, device=x.device)
            has_bias = 0

        (y_2d,) = _ext.run_layer_norm(x_2d, weight_1d, bias_1d, has_bias, 1e-5)
        return y_2d.reshape(original_shape)
