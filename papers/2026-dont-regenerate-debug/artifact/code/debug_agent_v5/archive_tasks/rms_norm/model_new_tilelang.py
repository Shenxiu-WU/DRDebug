import sys
from pathlib import Path

import torch
import torch.nn as nn

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from design.tile_level.rms_norm import rms_norm as tl_rms_norm


class ModelNew(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def _build_kernel(self, x: torch.Tensor):
        m, n = x.shape
        return tl_rms_norm(
            m,
            n,
            eps=self.eps,
            dtype=str(x.dtype).split(".")[-1],
        )

    def forward(self, x: torch.Tensor, gamma: torch.Tensor):
        assert gamma.ndim == 1
        assert x.ndim >= 2
        assert x.shape[-1] == gamma.shape[0]
        assert x.dtype == gamma.dtype
        assert x.dtype in (torch.float16, torch.float32, torch.bfloat16)

        original_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        gamma_1d = gamma.contiguous()

        kernel = self._build_kernel(x_2d)
        y_2d, inv_rms_1d = kernel(x_2d, gamma_1d)
        return (
            y_2d.reshape(original_shape),
            inv_rms_1d.reshape(*original_shape[:-1], 1),
        )
