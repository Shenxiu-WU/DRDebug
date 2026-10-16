import sys
from pathlib import Path

import torch
import torch.nn as nn

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from design.tile_level.flash_attention import (
    flash_attention_fwd as tl_flash_attention_fwd,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _build_kernel(self, q: torch.Tensor):
        batch, heads, seq_len, dim = q.shape
        return tl_flash_attention_fwd(
            batch,
            seq_len,
            heads,
            dim,
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
        assert q.shape == k.shape == v.shape
        assert q.dtype == k.dtype == v.dtype == torch.float16

        kernel = self._build_kernel(q)
        return kernel(q, k, v)
