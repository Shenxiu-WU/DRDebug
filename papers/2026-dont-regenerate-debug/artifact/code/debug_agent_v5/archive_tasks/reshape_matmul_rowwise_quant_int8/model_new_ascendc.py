import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _reshape_matmul_rowwise_quant_int8_ext as _ext  # noqa: E402


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, "x must be 2D"
        assert h.ndim == 2 and h.shape[0] == h.shape[1], "h must be square"
        assert x.shape[1] % h.shape[0] == 0, "x.shape[1] must be divisible by k"
        assert x.dtype == h.dtype == torch.bfloat16, "x and h must be bfloat16"
        return _ext.run_reshape_matmul_quant(x, h)
