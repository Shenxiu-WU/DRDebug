import sys
from pathlib import Path

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _fused_rope_qk_norm_kv_cache_ext as _ext
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position_ids: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        inv_freq: torch.Tensor,
        rms_norm_eps: float,
    ) -> tuple:
        return _ext.run_fused_rope_qk_norm_kv_cache(
            query, key, value,
            position_ids, cache_position,
            q_norm_weight, k_norm_weight, inv_freq,
            key_cache, value_cache,
            rms_norm_eps,
        )
