import sys
from pathlib import Path

import torch
import torch.nn as nn

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from design.tile_level.flash_attention_gqa import (
    flash_attention_gqa_fwd as tl_flash_attention_gqa_fwd,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _build_kernel(self, q: torch.Tensor, k: torch.Tensor):
        batch, heads, q_seq_len, dim = q.shape
        _, kv_heads, kv_seq_len, _ = k.shape
        return tl_flash_attention_gqa_fwd(
            batch,
            q_seq_len,
            kv_seq_len,
            heads,
            kv_heads,
            dim,
        )

    def _pack_q(self, q: torch.Tensor, kv_heads: int) -> torch.Tensor:
        batch, heads, q_seq_len, dim = q.shape
        group_size = heads // kv_heads
        return q.reshape(batch, kv_heads, group_size, q_seq_len, dim).reshape(
            batch * kv_heads,
            group_size * q_seq_len,
            dim,
        )

    def _flatten_kv(self, x: torch.Tensor) -> torch.Tensor:
        batch, kv_heads, seq_len, dim = x.shape
        return x.reshape(batch * kv_heads, seq_len, dim)

    def _unpack_o(self, o: torch.Tensor, heads: int, q_seq_len: int) -> torch.Tensor:
        bh2, _, dim = o.shape
        kv_heads = bh2 // self._cached_batch
        group_size = heads // kv_heads
        return o.reshape(self._cached_batch, kv_heads, group_size, q_seq_len, dim).reshape(
            self._cached_batch,
            heads,
            q_seq_len,
            dim,
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
        assert k.shape == v.shape
        assert q.shape[0] == k.shape[0]
        assert q.shape[-1] == k.shape[-1] == v.shape[-1]
        assert q.shape[1] % k.shape[1] == 0
        assert (q.shape[1] // k.shape[1]) * q.shape[2] % 64 == 0
        assert k.shape[2] % 64 == 0
        assert q.dtype == k.dtype == v.dtype == torch.float16

        self._cached_batch = q.shape[0]
        packed_q = self._pack_q(q, k.shape[1])
        flat_k = self._flatten_kv(k)
        flat_v = self._flatten_kv(v)
        kernel = self._build_kernel(q, k)
        packed_o = kernel(packed_q, flat_k, flat_v)
        return self._unpack_o(packed_o, q.shape[1], q.shape[2])
