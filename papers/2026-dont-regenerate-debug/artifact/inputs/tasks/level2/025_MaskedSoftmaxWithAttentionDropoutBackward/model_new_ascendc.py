import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
if _KERNEL_BUILD.is_dir() and str(_KERNEL_BUILD) not in sys.path:
    sys.path.insert(0, str(_KERNEL_BUILD))

import _masked_softmax_attn_dropout_bwd_ext as _ext


class ModelNew(nn.Module):
    def forward(
        self,
        grad_output: torch.Tensor,
        p_attn: torch.Tensor,
        mask: torch.Tensor,
        dropout_mask: torch.Tensor,
        p_dropout: float,
    ) -> torch.Tensor:
        B, H, T, _ = grad_output.shape
        # Expand mask to [B, H, T, T] for uniform indexing
        mask_expanded = mask.expand(B, H, T, T)
        # Convert bool masks to computation dtype (0.0/1.0) and ensure contiguous
        mask_f = mask_expanded.to(grad_output.dtype).contiguous()
        dropout_f = dropout_mask.to(grad_output.dtype).contiguous()
        return _ext.run_masked_softmax_attn_dropout_bwd(
            grad_output, p_attn, mask_f, dropout_f, p_dropout
        )
