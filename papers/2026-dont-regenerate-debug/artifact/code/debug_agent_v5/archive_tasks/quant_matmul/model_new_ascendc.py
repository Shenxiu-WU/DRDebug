import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _quant_matmul_ext as _ext  # noqa: E402


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        assert a.ndim == 2 and b.ndim == 2, "a and b must be 2D"
        assert scale.ndim == 1, "scale must be 1D"
        assert a.shape[1] == b.shape[0], "k dimension must match"
        assert b.shape[1] == scale.shape[0], "scale size must match N"
        assert a.dtype == b.dtype == torch.int8, "a and b must be int8"
        assert scale.dtype == torch.float32, "scale must be float32"
        return _ext.run_int8_matmul_scale(a, b, scale)
