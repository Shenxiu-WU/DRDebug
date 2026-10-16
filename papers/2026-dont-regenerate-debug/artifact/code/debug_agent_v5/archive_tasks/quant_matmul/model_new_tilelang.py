import sys
from pathlib import Path

import torch
import torch.nn as nn

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from design.tile_level.int8_matmul_scale import (
    int8_matmul_scale as tl_int8_matmul_scale,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _build_kernel(self, a: torch.Tensor, b: torch.Tensor):
        m, k = a.shape
        _, n = b.shape
        return tl_int8_matmul_scale(
            m,
            n,
            k,
            dtype="int8",
            accum_dtype="int32",
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor):
        assert a.ndim == 2 and b.ndim == 2 and scale.ndim == 1
        assert a.shape[1] == b.shape[0]
        assert b.shape[1] == scale.shape[0]
        assert a.dtype == b.dtype == torch.int8
        assert scale.dtype == torch.float32

        kernel = self._build_kernel(a, b)
        return kernel(a, b, scale)
