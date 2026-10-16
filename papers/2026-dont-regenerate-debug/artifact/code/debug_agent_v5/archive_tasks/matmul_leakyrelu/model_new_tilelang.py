import sys
from pathlib import Path

import torch
import torch.nn as nn

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from design.tile_level.matmul_leakyrelu import (
    matmul_leakyrelu as tl_matmul_leakyrelu,
)


class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def _build_kernel(self, a: torch.Tensor, b: torch.Tensor):
        m, k = a.shape
        _, n = b.shape
        dtype = str(a.dtype).split(".")[-1]
        accum_dtype = "int32" if a.dtype == torch.int8 else "float"
        return tl_matmul_leakyrelu(
            m,
            n,
            k,
            dtype=dtype,
            accum_dtype=accum_dtype,
            negative_slope=self.negative_slope,
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        assert a.ndim == 2 and b.ndim == 2
        assert a.shape[1] == b.shape[0]
        assert a.dtype == b.dtype
        assert a.dtype in (torch.float16, torch.int8)
        kernel = self._build_kernel(a, b)
        return kernel(a, b)
