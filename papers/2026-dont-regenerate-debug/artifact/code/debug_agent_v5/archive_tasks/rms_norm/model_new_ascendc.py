import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _rms_norm_ext as _ext  # noqa: E402


class ModelNew(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, gamma: torch.Tensor):
        assert x.ndim >= 2, "x must be at least 2D"
        assert gamma.ndim == 1, "gamma must be 1D"
        assert x.shape[-1] == gamma.shape[0], "gamma shape mismatch"
        assert x.dtype == gamma.dtype, "x and gamma dtype must match"
        assert x.dtype in (torch.float16, torch.float32, torch.bfloat16), "unsupported dtype"
        assert x.is_contiguous() and gamma.is_contiguous(), "inputs must be contiguous"

        original_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        gamma_1d = gamma.contiguous()
        y_2d, inv_rms_1d = _ext.run_rms_norm(x_2d, gamma_1d, self.eps)
        return (
            y_2d.reshape(original_shape),
            inv_rms_1d.reshape(*original_shape[:-1], 1),
        )
