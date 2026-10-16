import sys
from pathlib import Path

import torch
import torch.nn as nn

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from design.tile_level.reshape_matmul_rowwise_quant_int8 import (
    reshape_matmul_rowwise_quant_int8 as tl_reshape_matmul_rowwise_quant_int8,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _build_kernel(self, x: torch.Tensor, h: torch.Tensor):
        m, n = x.shape
        k = h.shape[0]
        assert h.shape == (k, k)
        dtype = str(x.dtype).split(".")[-1]
        return tl_reshape_matmul_rowwise_quant_int8(
            m,
            n,
            k,
            dtype=dtype,
            accum_dtype="float",
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        assert x.ndim == 2 and h.ndim == 2
        assert x.dtype == h.dtype == torch.bfloat16
        k = h.shape[0]
        assert x.shape[1] % k == 0
        assert h.shape[0] == h.shape[1] == k
        assert x.shape[0] % 128 == 0
        assert x.shape[1] % 256 == 0
        assert k % 256 == 0

        kernel = self._build_kernel(x, h)
        return kernel(x, h)
