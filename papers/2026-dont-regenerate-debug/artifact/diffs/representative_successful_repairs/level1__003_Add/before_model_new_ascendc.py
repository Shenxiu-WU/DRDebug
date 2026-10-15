import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _add_ext as _ext  # noqa: E402


def _broadcast_shape(shape_a, shape_b):
    """Compute the broadcast output shape for two tensor shapes."""
    ndim = max(len(shape_a), len(shape_b))
    a_padded = [1] * (ndim - len(shape_a)) + list(shape_a)
    b_padded = [1] * (ndim - len(shape_b)) + list(shape_b)
    result = []
    for a, b in zip(a_padded, b_padded):
        if a != b and a != 1 and b != 1:
            raise ValueError(f"Shapes {shape_a} and {shape_b} are not broadcastable")
        result.append(max(a, b))
    return tuple(result)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
        # Compute broadcast shape
        out_shape = _broadcast_shape(x.shape, y.shape)

        # Expand and flatten for kernel
        x_expanded = x.expand(out_shape).contiguous()
        y_expanded = y.expand(out_shape).contiguous()

        N = x_expanded.numel()
        x_flat = x_expanded.reshape(N)
        y_flat = y_expanded.reshape(N)

        # Call AscendC kernel
        out_flat = _ext.run_add_alpha(x_flat, y_flat, alpha)

        return out_flat.reshape(out_shape)
